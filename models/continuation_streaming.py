"""Experimental stateful continuation for the Breeze fast streaming runtime."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from breeze_infer.templates import (
    prepare_continuation_cfg_text_inputs,
    prepare_continuation_text_inputs,
)
from models.cudagraph.backbone_prefill_graph import BackbonePrefillGraphCache
from models.cudagraph.sampling import sample_logits
from models.fast_streaming import (
    FastBreezeStreamingRuntime,
    FastCfgSelection,
    FastStreamingChunk,
    is_backbone_eos_token,
    select_fast_cfg,
    should_decode_codec_frame,
)


class ContinuationRuntimeError(RuntimeError):
    """Fatal continuation error after which retained state is unsafe."""


class ContinuationContextError(ContinuationRuntimeError):
    """The retained sequence cannot fit another text/audio segment."""


@dataclass
class ContinuationRngState:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor]

    @classmethod
    def capture(cls) -> ContinuationRngState:
        return cls(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch_cpu=torch.get_rng_state().clone(),
            torch_cuda=[state.clone() for state in torch.cuda.get_rng_state_all()],
        )

    def restore(self) -> None:
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.torch_cpu)
        if self.torch_cuda:
            torch.cuda.set_rng_state_all(self.torch_cuda)


@dataclass
class ContinuationRuntimeSession:
    continuation_id: str
    codec_request_id: str
    graph: Any
    cfg: FastCfgSelection
    branch_batch_size: int
    token_history: torch.Tensor
    cache_length: int = 0
    next_position_ids: torch.Tensor | None = None
    pad_lens: torch.Tensor | None = None
    generated_frames: int = 0
    codec_started: bool = False
    codec_reset_pending: bool = True
    rng_state: ContinuationRngState | None = None
    chunk_index: int = 0
    created_at: float = field(default_factory=time.monotonic)
    last_used_at: float = field(default_factory=time.monotonic)
    last_raw_logits: torch.Tensor | None = None
    closed: bool = False
    last_timing: dict[str, float | int | bool] = field(default_factory=dict)


class ContinuationStreamingRuntime:
    """Retain one Breeze causal/codec trajectory across logical text chunks."""

    def __init__(self, runtime: FastBreezeStreamingRuntime, *, audio_eos: bool = False):
        self.runtime = runtime
        self.audio_eos = bool(audio_eos)

    @property
    def sample_rate(self) -> int:
        return self.runtime.sample_rate

    @torch.inference_mode()
    def new_session(
        self, continuation_id: str, inputs: dict[str, Any]
    ) -> ContinuationRuntimeSession:
        cfg = select_fast_cfg(inputs)
        branch_batch_size = 2 if cfg.mode == "single_cfg" else 1
        self.runtime._ensure_graphs(branch_batch_size, cfg.guidance_scale)
        graph = self.runtime._backbone_graph
        if graph is None:
            raise ContinuationRuntimeError("backbone graph is unavailable")
        return ContinuationRuntimeSession(
            continuation_id=continuation_id,
            codec_request_id=f"continuation-{continuation_id}",
            graph=graph,
            cfg=cfg,
            branch_batch_size=branch_batch_size,
            token_history=torch.empty(
                self.runtime.config.max_seq_len,
                dtype=torch.long,
                device=self.runtime.device,
            ),
        )

    def _check_open(self, session: ContinuationRuntimeSession) -> None:
        if session.closed:
            raise ContinuationRuntimeError("continuation session is closed")

    @torch.inference_mode()
    def _open_codec(self, session: ContinuationRuntimeSession) -> None:
        if session.codec_started:
            return
        self.runtime._codec().open_request(
            session.codec_request_id, reset=True, is_first_decode=True
        )
        session.codec_started = True

    @torch.inference_mode()
    def close(self, session: ContinuationRuntimeSession) -> None:
        if session.closed:
            return
        try:
            if session.codec_started:
                self.runtime._codec().close_request(session.codec_request_id)
        finally:
            session.graph.reset()
            session.closed = True

    def _combine_cfg_logits(
        self,
        session: ContinuationRuntimeSession,
        logits: torch.Tensor,
        *,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        if session.branch_batch_size == 2:
            scale = session.cfg.guidance_scale if guidance_scale is None else guidance_scale
            return logits[1:] + scale * (logits[:1] - logits[1:])
        return logits[:1]

    @staticmethod
    def _continuation_cfg_scale(
        target_scale: float,
        *,
        low_scale: float | None,
        hold_frames: int,
        ramp_frames: int,
        frame_index: int,
    ) -> float:
        """Return a continuation-only CFG scale for one acoustic frame."""
        if low_scale is None or (hold_frames <= 0 and ramp_frames <= 0):
            return target_scale
        if frame_index < hold_frames:
            return low_scale
        ramp_index = frame_index - hold_frames
        if ramp_index < ramp_frames:
            return low_scale + (target_scale - low_scale) * (
                (ramp_index + 1) / ramp_frames
            )
        return target_scale

    def _guided_eos_rank(
        self, session: ContinuationRuntimeSession, raw_logits: torch.Tensor, scale: float
    ) -> int:
        """Return the guided backbone EOS rank (1 is most likely)."""
        if session.branch_batch_size != 2:
            return int(raw_logits[0].numel())
        cond_logits = raw_logits[:1]
        uncond_logits = raw_logits[1:]
        guided = uncond_logits + scale * (cond_logits - uncond_logits)
        eos_id = int(self.runtime.model.config.vocab_size)
        eos_logit = guided[0, eos_id]
        return int((guided[0] > eos_logit).sum().item()) + 1

    @torch.inference_mode()
    def _initial_prefill(
        self, session: ContinuationRuntimeSession, inputs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        started = time.perf_counter()
        branch = self.runtime._build_branch_batch(inputs)
        if branch.branch_batch_size != session.branch_batch_size:
            raise ContinuationRuntimeError("continuation CFG branch shape changed")
        attention_mask = branch.attention_mask
        graph = session.graph
        if self.runtime._fast_backbone_prefill:
            prefill_graph = self.runtime._backbone_prefill_graphs.get(
                session.branch_batch_size
            )
            if prefill_graph is None:
                prefill_graph = BackbonePrefillGraphCache(graph, token_granularity=32)
                self.runtime._backbone_prefill_graphs[session.branch_batch_size] = (
                    prefill_graph
                )
            output = prefill_graph(branch.inputs_embeds, attention_mask)
            hidden = output.hidden_states
            logits = output.logits
            prefill_len = output.prefill_len
            generation_mask = output.attention_mask
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            output = self.runtime.model.backbone_model(
                inputs_embeds=branch.inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                cache_position=None,
                use_cache=True,
            )
            hidden = output.last_hidden_state
            logits = self.runtime.model.lm_head(hidden[:, -1, :].float()).float()
            prefill_len = graph.prefill_kv(output.past_key_values)
            generation_mask = attention_mask

        graph.set_generation_state(generation_mask)
        session.cache_length = int(prefill_len)
        session.next_position_ids = generation_mask.sum(dim=1).long().clone()
        session.pad_lens = (
            int(generation_mask.shape[1]) - session.next_position_ids
        ).clone()
        return hidden, self._combine_cfg_logits(session, logits), (
            time.perf_counter() - started
        ) * 1000.0

    def _audio_eos_embedding(self, session: ContinuationRuntimeSession) -> torch.Tensor:
        frame = torch.full(
            (1, 1, self.runtime.model.config.num_codebooks),
            int(self.runtime.model.config.codebook_eos_token_id),
            dtype=torch.long,
            device=self.runtime.device,
        )
        embed = session.graph.embed_tokens(frame).to(self.runtime.dtype)
        return embed.repeat(session.branch_batch_size, 1, 1)

    def preflight_continue(
        self,
        session: ContinuationRuntimeSession,
        text: str,
        *,
        refresh_instruction: str | None = None,
        estimated_audio_frames: int = 0,
        context_safety_frames: int = 64,
    ) -> dict[str, torch.Tensor | None]:
        """Tokenize and validate an append before HTTP response headers commit."""
        self._check_open(session)
        if refresh_instruction is not None and session.branch_batch_size == 2:
            cond_inputs, uncond_inputs = prepare_continuation_cfg_text_inputs(
                self.runtime.tokenizer,
                self.runtime.audio_tokenizer,
                self.runtime.model,
                text,
                instruction=refresh_instruction,
            )
            text_inputs: dict[str, torch.Tensor | None] = {
                "conditional": cond_inputs,
                "unconditional": uncond_inputs,
            }  # type: ignore[dict-item]
            append_len = max(
                int(cond_inputs["input_ids"].shape[1]),
                int(uncond_inputs["input_ids"].shape[1]),
            ) + int(self.audio_eos)
        else:
            text_inputs = prepare_continuation_text_inputs(
                self.runtime.tokenizer,
                self.runtime.audio_tokenizer,
                self.runtime.model,
                text,
            )
            append_len = int(text_inputs["input_ids"].shape[1]) + int(self.audio_eos)
        required = (
            session.cache_length
            + append_len
            + int(estimated_audio_frames)
            + int(context_safety_frames)
        )
        if required >= self.runtime.config.max_seq_len:
            raise ContinuationContextError(
                "continuation context limit exceeded while appending text"
            )
        return text_inputs

    @torch.inference_mode()
    def append_text(
        self,
        session: ContinuationRuntimeSession,
        text: str,
        *,
        refresh_instruction: str | None = None,
        guidance_scale: float | None = None,
        estimated_audio_frames: int = 0,
        context_safety_frames: int = 64,
        prepared_text_inputs: dict[str, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
        self._check_open(session)
        if session.next_position_ids is None or session.pad_lens is None:
            raise ContinuationRuntimeError("continuation has not completed initial prefill")

        text_started = time.perf_counter()
        text_inputs = prepared_text_inputs or self.preflight_continue(
            session,
            text,
            refresh_instruction=refresh_instruction,
            estimated_audio_frames=estimated_audio_frames,
            context_safety_frames=context_safety_frames,
        )
        refreshed_cfg = refresh_instruction is not None and session.branch_batch_size == 2
        if refreshed_cfg:
            cond_inputs = text_inputs["conditional"]  # type: ignore[index]
            uncond_inputs = text_inputs["unconditional"]  # type: ignore[index]
            cond_embeds, _ = self.runtime._merge_branch(
                input_ids=cond_inputs["input_ids"],
                attention_mask=cond_inputs["attention_mask"],
                text_ids_mask=cond_inputs["text_ids_mask"],
                text_ids_len=cond_inputs["text_ids_len"],
                input_values=None,
            )
            uncond_embeds, _ = self.runtime._merge_branch(
                input_ids=uncond_inputs["input_ids"],
                attention_mask=uncond_inputs["attention_mask"],
                text_ids_mask=uncond_inputs["text_ids_mask"],
                text_ids_len=uncond_inputs["text_ids_len"],
                input_values=None,
            )
            # The branches intentionally have different sequence lengths; the
            # aligned cache tensor is constructed below.
            text_embeds = cond_embeds
        else:
            text_embeds, _ = self.runtime._merge_branch(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                text_ids_mask=text_inputs["text_ids_mask"],
                text_ids_len=text_inputs["text_ids_len"],
                input_values=None,
            )
        text_encoder_ms = (time.perf_counter() - text_started) * 1000.0
        if refreshed_cfg:
            branches = [cond_embeds, uncond_embeds]
            branch_lengths = torch.tensor(
                [item.shape[1] for item in branches],
                dtype=torch.long,
                device=self.runtime.device,
            )
            append_len = int(branch_lengths.max().item()) + int(self.audio_eos)
            embeds = torch.zeros(
                session.branch_batch_size,
                append_len,
                cond_embeds.shape[-1],
                dtype=self.runtime.dtype,
                device=self.runtime.device,
            )
            append_valid = torch.zeros(
                session.branch_batch_size,
                append_len,
                dtype=torch.bool,
                device=self.runtime.device,
            )
            for row, branch_embeds in enumerate(branches):
                start = append_len - int(branch_embeds.shape[1])
                if self.audio_eos:
                    start -= 1
                    embeds[row, start : start + 1] = self._audio_eos_embedding(session)[
                        0
                    ]
                text_start = start + int(self.audio_eos)
                embeds[row, text_start:] = branch_embeds[0]
                append_valid[row, start:] = True
            new_text_tokens = int(branch_lengths.max().item())
        else:
            embeds = text_embeds.repeat(session.branch_batch_size, 1, 1).contiguous()
            if self.audio_eos:
                embeds = torch.cat([self._audio_eos_embedding(session), embeds], dim=1)
            append_valid = torch.ones(
                session.branch_batch_size,
                embeds.shape[1],
                dtype=torch.bool,
                device=self.runtime.device,
            )
            new_text_tokens = int(text_embeds.shape[1])

        append_len = int(embeds.shape[1])
        required = (
            session.cache_length
            + append_len
            + int(estimated_audio_frames)
            + int(context_safety_frames)
        )
        if required >= self.runtime.config.max_seq_len:
            raise ContinuationContextError(
                "continuation context limit exceeded while appending text"
            )

        append_started = time.perf_counter()
        graph = session.graph
        cache_position = torch.arange(
            session.cache_length,
            session.cache_length + append_len,
            dtype=torch.long,
            device=self.runtime.device,
        )
        relative = append_valid.long().cumsum(dim=1) - 1
        position_ids = session.next_position_ids[:, None] + relative
        position_ids.masked_fill_(~append_valid, 1)
        current_indices = torch.arange(append_len, device=self.runtime.device)
        # Existing valid slots precede the append; current slots map one-to-one
        # to the physical cache positions beginning at ``cache_length``.
        valid_kv_mask = getattr(graph, "_valid_kv_mask", None)
        if valid_kv_mask is None:
            valid_kv_mask = (
                graph._kv_indices.unsqueeze(0) >= session.pad_lens.unsqueeze(1)
            ) & (graph._kv_indices.unsqueeze(0) < session.cache_length)
        valid_history = valid_kv_mask[:, None, :] & (
            graph._kv_indices[None, None, :] < session.cache_length
        )
        valid_current = torch.zeros(
            session.branch_batch_size,
            append_len,
            self.runtime.config.max_seq_len,
            dtype=torch.bool,
            device=self.runtime.device,
        )
        valid_current[:, :, session.cache_length : session.cache_length + append_len] = (
            append_valid[:, None, :]
            & (current_indices[None, None, :] <= current_indices[None, :, None])
        )
        valid = (valid_history | valid_current) & append_valid[:, :, None]
        attention_mask = torch.full(
            (
                session.branch_batch_size,
                1,
                append_len,
                self.runtime.config.max_seq_len,
            ),
            torch.finfo(self.runtime.dtype).min,
            dtype=self.runtime.dtype,
            device=self.runtime.device,
        )
        attention_mask[:, 0].masked_fill_(valid, 0.0)
        output = graph.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            past_key_values=graph.static_cache,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
        )
        hidden = output.last_hidden_state
        logits = graph.lm_head(hidden[:, -1, :].float()).float()
        session.last_raw_logits = logits
        session.cache_length += append_len
        session.next_position_ids.add_(append_valid.sum(dim=1))
        if hasattr(graph, "_valid_kv_mask"):
            graph._valid_kv_mask[:, cache_position] |= append_valid
        graph.finish_direct_prefill(session.cache_length)
        graph._base_position.copy_(session.next_position_ids)
        graph._pad_lens.copy_(session.pad_lens)
        append_prefill_ms = (time.perf_counter() - append_started) * 1000.0
        return hidden, self._combine_cfg_logits(
            session, logits, guidance_scale=guidance_scale
        ), {
            "new_text_tokens": new_text_tokens,
            "text_encoder_ms": text_encoder_ms,
            "append_prefill_ms": append_prefill_ms,
        }

    @torch.inference_mode()
    def _decode_frames(
        self,
        session: ContinuationRuntimeSession,
        frames: list[torch.Tensor],
        *,
        is_final: bool,
        timing: dict[str, float | int | bool],
    ) -> FastStreamingChunk:
        chunk = self.runtime._decode_codec_frames(
            frames=frames,
            request_id=session.codec_request_id,
            reset=session.codec_reset_pending,
            is_final=is_final,
            timing=timing,
        )
        session.codec_reset_pending = False
        return chunk

    def _capture_rng(self, session: ContinuationRuntimeSession) -> None:
        session.rng_state = ContinuationRngState.capture()

    @torch.inference_mode()
    def _generate(
        self,
        session: ContinuationRuntimeSession,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        *,
        cancel_event: threading.Event | None,
        base_timing: dict[str, float | int],
        cfg_ramp_low_scale: float | None = None,
        cfg_ramp_hold_frames: int = 0,
        cfg_ramp_frames: int = 0,
        cfg_eos_taper_rank: int | None = None,
        cfg_eos_taper_scale: float | None = None,
        initial_cfg_scale: float | None = None,
        initial_eos_taper_active: bool = False,
        eos_observer: Callable[[int, torch.Tensor, float], None] | None = None,
        frame_observer: Callable[[torch.Tensor], None] | None = None,
    ) -> Iterator[FastStreamingChunk]:
        backbone_params = self.runtime._sampling_params(
            self.runtime.model.generation_config
        )
        depth_params = self.runtime._sampling_params(
            self.runtime.model.depth_decoder.generation_config
        )
        # Retain all acoustic tokens for diagnostics/session state, but apply
        # repetition penalty only within this target-text segment. Carrying
        # the heuristic across sentences progressively penalizes most of the
        # acoustic vocabulary and can produce unstable trailing speech.
        repetition_history_start = session.generated_frames
        token = sample_logits(
            logits,
            token_history=session.token_history[
                repetition_history_start : session.generated_frames
            ],
            repetition_penalty=self.runtime.config.repetition_penalty,
            suppress_tokens=self.runtime._reserved_codec_token_ids,
            **backbone_params,
        ).view(1)
        frames: list[torch.Tensor] = []
        decoded_frames = 0
        generated_steps = 0
        output_chunk_index = 0
        chunk_started = time.perf_counter()
        generation_started = chunk_started
        terminated = False
        raw_logits = session.last_raw_logits
        terminal_taper_active = initial_eos_taper_active
        current_cfg_scale = (
            session.cfg.guidance_scale
            if initial_cfg_scale is None
            else initial_cfg_scale
        )
        for local_step in range(self.runtime.config.max_new_tokens):
            if cancel_event is not None and cancel_event.is_set():
                raise ContinuationRuntimeError("continuation generation cancelled")
            if is_backbone_eos_token(token, self.runtime.model.config):
                terminated = True
                break
            if session.cache_length + local_step >= self.runtime.config.max_seq_len - 1:
                raise ContinuationContextError("continuation context limit exceeded")

            token_batch = (
                token.repeat(2) if session.branch_batch_size == 2 else token
            )
            depth_hidden = (
                hidden[:, -1, :]
                if session.branch_batch_size == 2
                else hidden[:1, -1, :]
            )
            frame_scale = current_cfg_scale
            if eos_observer is not None and raw_logits is not None:
                eos_observer(local_step, raw_logits, frame_scale)
            depth_tokens = self.runtime._depth_decoder_graph.run(
                depth_hidden,
                token_batch,
                guidance_scale=frame_scale,
                **depth_params,
            )
            frame = torch.cat([token.view(1), depth_tokens[0]], dim=0)
            if should_decode_codec_frame(frame, self.runtime.model.config):
                if frame_observer is not None:
                    frame_observer(frame.detach())
                frames.append(frame.detach())

            frame_batch = frame.view(1, 1, -1)
            if session.branch_batch_size == 2:
                frame_batch = frame_batch.repeat(2, 1, 1)
            hidden, logits = session.graph.run(frame_batch, step_idx=local_step)
            if eos_observer is not None:
                raw_logits = getattr(session.graph, "logits_buf", None)
            history_index = session.generated_frames + local_step
            session.token_history[history_index] = token[0]
            generated_steps += 1
            next_scale = self._continuation_cfg_scale(
                session.cfg.guidance_scale,
                low_scale=cfg_ramp_low_scale,
                hold_frames=cfg_ramp_hold_frames,
                ramp_frames=cfg_ramp_frames,
                frame_index=local_step + 1,
            )
            if (
                not terminal_taper_active
                and cfg_eos_taper_rank is not None
                and raw_logits is not None
                and self._guided_eos_rank(
                    session, raw_logits, session.cfg.guidance_scale
                )
                <= cfg_eos_taper_rank
            ):
                terminal_taper_active = True
            if terminal_taper_active:
                assert cfg_eos_taper_scale is not None
                next_scale = cfg_eos_taper_scale
            if (
                (cfg_ramp_low_scale is not None or cfg_eos_taper_rank is not None)
                and session.branch_batch_size == 2
            ):
                logits = session.graph.combine_logits(next_scale)
            current_cfg_scale = next_scale
            token = sample_logits(
                logits.float(),
                token_history=session.token_history[
                    repetition_history_start : history_index + 1
                ],
                repetition_penalty=self.runtime.config.repetition_penalty,
                suppress_tokens=self.runtime._reserved_codec_token_ids,
                **backbone_params,
            ).view(1)

            if len(frames) >= self.runtime.codec_chunk_frames:
                ready = frames
                frames = []
                decoded_frames += len(ready)
                yield self._decode_frames(
                    session,
                    ready,
                    is_final=False,
                    timing={
                        **base_timing,
                        "chunk_index": output_chunk_index,
                        "codec_frames": len(ready),
                        "decode_launch_ms": (time.perf_counter() - chunk_started)
                        * 1000.0,
                        "total_frames": decoded_frames,
                        "is_final": False,
                    },
                )
                output_chunk_index += 1
                chunk_started = time.perf_counter()

        if not terminated and is_backbone_eos_token(token, self.runtime.model.config):
            terminated = True
        if not terminated:
            raise ContinuationRuntimeError(
                "continuation chunk reached max_new_tokens before acoustic EOS"
            )

        if frames:
            decoded_frames += len(frames)
            yield self._decode_frames(
                session,
                frames,
                is_final=True,
                timing={
                    **base_timing,
                    "chunk_index": output_chunk_index,
                    "codec_frames": len(frames),
                    "decode_launch_ms": (time.perf_counter() - chunk_started)
                    * 1000.0,
                    "total_frames": decoded_frames,
                    "is_final": True,
                },
            )

        session.cache_length += generated_steps
        session.generated_frames += generated_steps
        session.next_position_ids.add_(generated_steps)
        session.graph.finish_direct_prefill(session.cache_length)
        session.graph._base_position.copy_(session.next_position_ids)
        session.last_used_at = time.monotonic()
        session.last_timing = {
            **base_timing,
            "acoustic_decode_ms": (time.perf_counter() - generation_started) * 1000.0,
            "generated_frames": generated_steps,
            "cache_length_after": session.cache_length,
        }
        session.chunk_index += 1
        self._capture_rng(session)

    def iter_start(
        self,
        session: ContinuationRuntimeSession,
        inputs: dict[str, Any],
        *,
        seed: int,
        estimated_audio_frames: int = 0,
        context_safety_frames: int = 64,
        cancel_event: threading.Event | None = None,
        frame_observer: Callable[[torch.Tensor], None] | None = None,
    ) -> Iterator[FastStreamingChunk]:
        self._check_open(session)
        self._open_codec(session)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        hidden, logits, prefill_ms = self._initial_prefill(session, inputs)
        if (
            session.cache_length
            + int(estimated_audio_frames)
            + int(context_safety_frames)
            >= self.runtime.config.max_seq_len
        ):
            raise ContinuationContextError("continuation context limit exceeded")
        yield from self._generate(
            session,
            hidden,
            logits,
            cancel_event=cancel_event,
            frame_observer=frame_observer,
            base_timing={
                "cache_length_before": 0,
                "new_text_tokens": session.cache_length,
                "text_encoder_ms": 0.0,
                "append_prefill_ms": prefill_ms,
            },
        )

    def iter_continue(
        self,
        session: ContinuationRuntimeSession,
        text: str,
        *,
        refresh_instruction: str | None = None,
        cfg_ramp_low_scale: float | None = None,
        cfg_ramp_hold_frames: int = 0,
        cfg_ramp_frames: int = 0,
        cfg_eos_taper_rank: int | None = None,
        cfg_eos_taper_scale: float | None = None,
        eos_observer: Callable[[int, torch.Tensor, float], None] | None = None,
        estimated_audio_frames: int = 0,
        context_safety_frames: int = 64,
        cancel_event: threading.Event | None = None,
        prepared_text_inputs: dict[str, torch.Tensor | None] | None = None,
        frame_observer: Callable[[torch.Tensor], None] | None = None,
    ) -> Iterator[FastStreamingChunk]:
        self._check_open(session)
        if session.rng_state is None:
            raise ContinuationRuntimeError("continuation RNG state is unavailable")
        if cfg_ramp_low_scale is not None:
            if session.branch_batch_size != 2:
                raise ContinuationRuntimeError("continuation CFG ramp requires CFG")
            if not 0 < cfg_ramp_low_scale <= session.cfg.guidance_scale:
                raise ContinuationRuntimeError(
                    "continuation CFG ramp low scale must be positive and no greater than target"
                )
            if cfg_ramp_hold_frames < 0 or cfg_ramp_frames < 0:
                raise ContinuationRuntimeError("continuation CFG ramp frame counts cannot be negative")
        if cfg_eos_taper_rank is not None:
            if session.branch_batch_size != 2:
                raise ContinuationRuntimeError("continuation EOS taper requires CFG")
            if cfg_eos_taper_rank <= 0:
                raise ContinuationRuntimeError("continuation EOS taper rank must be positive")
            if (
                cfg_eos_taper_scale is None
                or not 0 < cfg_eos_taper_scale <= session.cfg.guidance_scale
            ):
                raise ContinuationRuntimeError(
                    "continuation EOS taper scale must be positive and no greater than target"
                )
        session.rng_state.restore()
        cache_before = session.cache_length
        first_scale = self._continuation_cfg_scale(
            session.cfg.guidance_scale,
            low_scale=cfg_ramp_low_scale,
            hold_frames=cfg_ramp_hold_frames,
            ramp_frames=cfg_ramp_frames,
            frame_index=0,
        )
        hidden, logits, timing = self.append_text(
            session,
            text,
            refresh_instruction=refresh_instruction,
            guidance_scale=first_scale,
            estimated_audio_frames=estimated_audio_frames,
            context_safety_frames=context_safety_frames,
            prepared_text_inputs=prepared_text_inputs,
        )
        terminal_taper_active = (
            cfg_eos_taper_rank is not None
            and session.last_raw_logits is not None
            and self._guided_eos_rank(
                session, session.last_raw_logits, session.cfg.guidance_scale
            )
            <= cfg_eos_taper_rank
        )
        if terminal_taper_active:
            assert cfg_eos_taper_scale is not None
            first_scale = cfg_eos_taper_scale
            logits = self._combine_cfg_logits(
                session, session.last_raw_logits, guidance_scale=first_scale
            )
        yield from self._generate(
            session,
            hidden,
            logits,
            cancel_event=cancel_event,
            frame_observer=frame_observer,
            base_timing={
                "cache_length_before": cache_before,
                "cfg_ramp_low_scale": cfg_ramp_low_scale is not None,
                **timing,
            },
            cfg_ramp_low_scale=cfg_ramp_low_scale,
            cfg_ramp_hold_frames=cfg_ramp_hold_frames,
            cfg_ramp_frames=cfg_ramp_frames,
            cfg_eos_taper_rank=cfg_eos_taper_rank,
            cfg_eos_taper_scale=cfg_eos_taper_scale,
            initial_cfg_scale=first_scale,
            initial_eos_taper_active=terminal_taper_active,
            eos_observer=eos_observer,
        )
