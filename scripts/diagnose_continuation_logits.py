"""Compare retained-cache continuation logits with full eager recomputation."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

import torch

from breeze_infer.audio import encode_prompt_audio
from breeze_infer.profiles import ProfileStore
from breeze_infer.runtime import (
    HYBRID_SCALE_MODES,
    load_runtime,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import (
    get_template,
    prepare_continuation_text_inputs,
    prepare_inputs,
)
from breeze_infer.text_chunks import estimate_speech_frames
from models.continuation_streaming import ContinuationStreamingRuntime
from models.cudagraph.backbone_graph import BackboneGraph
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
from models.warmup_profile import load_warmup_profile


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual = actual.detach()
    expected = expected.detach()
    difference = (actual - expected).abs()
    return {
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "argmax_matches": bool(torch.equal(actual.argmax(-1), expected.argmax(-1))),
        "top5_overlap": int(
            torch.isin(actual.topk(5).indices, expected.topk(5).indices).sum()
        ),
    }


def _full_logits(
    runtime: FastBreezeStreamingRuntime,
    parts: list[torch.Tensor],
    *,
    cfg_scale: float,
) -> torch.Tensor:
    """Recompute logits with no retained StaticCache using the same embeddings."""
    max_len = max(part.shape[0] for part in parts)
    hidden_size = parts[0].shape[-1]
    embeds = torch.zeros(
        len(parts), max_len, hidden_size, dtype=parts[0].dtype, device=runtime.device
    )
    mask = torch.zeros(len(parts), max_len, dtype=torch.long, device=runtime.device)
    for index, part in enumerate(parts):
        embeds[index, -part.shape[0] :] = part
        mask[index, -part.shape[0] :] = 1
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 1)
    output = runtime.model.backbone_model(
        inputs_embeds=embeds,
        attention_mask=mask,
        position_ids=positions,
        past_key_values=None,
        cache_position=None,
        use_cache=False,
    )
    logits = runtime.model.lm_head(output.last_hidden_state[:, -1, :].float()).float()
    return logits[1:] + cfg_scale * (logits[:1] - logits[1:])


@torch.inference_mode()
def _fresh_static_graph(
    runtime: FastBreezeStreamingRuntime,
    branch: Any,
    frames: list[torch.Tensor],
    *,
    guidance_scale: float,
) -> tuple[BackboneGraph, int, torch.Tensor, torch.Tensor]:
    """Build the existing prefix in a new eager StaticCache graph."""
    graph = BackboneGraph(
        runtime.model.backbone_model,
        runtime.model.lm_head,
        runtime.model.backbone_model.embed_tokens,
        runtime.model.config,
        device=runtime.device,
        dtype=runtime.dtype,
        max_seq_len=runtime.config.max_seq_len,
        guidance_scale=guidance_scale,
        batch_size=branch.branch_batch_size,
    ).prepare_eager()
    mask = branch.attention_mask
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 1)
    prefix_length = branch.inputs_embeds.shape[1]
    graph.model(
        inputs_embeds=branch.inputs_embeds,
        attention_mask=mask,
        past_key_values=graph.static_cache,
        position_ids=positions,
        cache_position=torch.arange(prefix_length, device=runtime.device),
        use_cache=True,
    )
    graph.finish_direct_prefill(prefix_length)
    graph.set_generation_state(mask)
    for index, frame in enumerate(frames):
        graph.run(
            frame.view(1, 1, -1)
            .to(runtime.device)
            .repeat(branch.branch_batch_size, 1, 1),
            step_idx=index,
        )
    cache_length = prefix_length + len(frames)
    graph.finish_direct_prefill(cache_length)
    next_positions = mask.sum(dim=1).long() + len(frames)
    graph._base_position.copy_(next_positions)
    graph._pad_lens.copy_(prefix_length - mask.sum(dim=1).long())
    return graph, cache_length, next_positions, graph._pad_lens.clone()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/continuation-logits.json"))
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--hybrid-scale-mode", choices=sorted(HYBRID_SCALE_MODES), default="bf16_compat"
    )
    parser.add_argument("--audio-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--eager-static-cache",
        action="store_true",
        help="Disable CUDA-graph replay while retaining the same StaticCache path.",
    )
    args = parser.parse_args()

    model_dir = Path("models/Breeze-TTS-2")
    tokenizer, model, audio_tokenizer = load_runtime(
        model_dir,
        device="cuda:0",
        attn_implementation="eager",
        weights_path=model_dir / "Breeze-TTS-2-int8-hybrid.safetensors",
        hybrid_scale_mode=args.hybrid_scale_mode,
    )
    update_generation_config_for_breeze(model)
    runtime = FastBreezeStreamingRuntime(
        model,
        audio_tokenizer,
        FastStreamingConfig(
            max_new_tokens=1500,
            max_seq_len=2048,
            fast_backbone_decode=True,
            fast_depth_decoder=True,
            fast_codec=True,
            repetition_penalty=1.1,
        ),
        tokenizer=tokenizer,
    )
    runtime.warmup_from_profile(load_warmup_profile(Path("configs/fast.json")))
    continuation = ContinuationStreamingRuntime(runtime, audio_eos=args.audio_eos)
    profiles = ProfileStore(Path.home() / ".local/share/breeze-tts/voices")
    profile_id = profiles.resolve("sky")
    profile = profiles.get(profile_id)
    codes = profiles.load_codes(profile_id)
    if codes is None:
        codes = encode_prompt_audio(audio_tokenizer, profiles.audio_path(profile_id))

    instruction = "Warm, conversational and engaged, with a relaxed natural rhythm."
    first_text = "I spent the morning thinking about our next step."
    next_text = "The simplest option may actually be the strongest one."

    def initial_inputs(text: str) -> dict[str, Any]:
        return prepare_inputs(
            tokenizer,
            audio_tokenizer,
            model,
            [{"id": "logit-diagnostic", "text": text, "speaker": "S0", "instruction": instruction,
              "ref_text": profile["ref_text"], "ref_audio_codes": codes}],
            get_template("ref_edit_tata"),
            guidance_scale=4.0,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )

    inputs = initial_inputs(first_text)
    branch = runtime._build_branch_batch(inputs)
    state = continuation.new_session(uuid.uuid4().hex, inputs)
    if args.eager_static_cache:
        state.graph.prepare_eager()
    frames: list[torch.Tensor] = []
    try:
        list(
            continuation.iter_start(
                state,
                inputs,
                seed=args.seed,
                estimated_audio_frames=estimate_speech_frames(tokenizer, first_text),
                frame_observer=lambda frame: frames.append(frame.detach().clone()),
            )
        )
        text_inputs = prepare_continuation_text_inputs(tokenizer, audio_tokenizer, model, next_text)
        next_text_embeds, _ = runtime._merge_branch(
            input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"],
            text_ids_mask=text_inputs["text_ids_mask"], text_ids_len=text_inputs["text_ids_len"], input_values=None,
        )
        next_text_embeds = next_text_embeds.repeat(state.branch_batch_size, 1, 1)
        frame_tensor = torch.stack(frames).unsqueeze(0).to(runtime.device)
        frame_embeds = state.graph.embed_tokens(frame_tensor).repeat(state.branch_batch_size, 1, 1)
        eos = continuation._audio_eos_embedding(state) if args.audio_eos else None
        prefix_parts = []
        for index in range(state.branch_batch_size):
            length = int(branch.attention_mask[index].sum())
            parts = [branch.inputs_embeds[index, -length:], frame_embeds[index]]
            if eos is not None:
                parts.append(eos[index])
            parts.append(next_text_embeds[index])
            prefix_parts.append(torch.cat(parts, dim=0))

        _, retained_logits, _ = continuation.append_text(
            state, next_text, estimated_audio_frames=estimate_speech_frames(tokenizer, next_text)
        )
        eager_logits = _full_logits(runtime, prefix_parts, cfg_scale=state.cfg.guidance_scale)

        rebuilt_graph, rebuilt_cache_length, rebuilt_positions, rebuilt_pad_lens = (
            _fresh_static_graph(
                runtime,
                branch,
                frames,
                guidance_scale=state.cfg.guidance_scale,
            )
        )
        rebuilt_state = type(state)(
            continuation_id="rebuilt-static-cache",
            codec_request_id="rebuilt-static-cache",
            graph=rebuilt_graph,
            cfg=state.cfg,
            branch_batch_size=state.branch_batch_size,
            token_history=torch.empty_like(state.token_history),
            cache_length=rebuilt_cache_length,
            next_position_ids=rebuilt_positions,
            pad_lens=rebuilt_pad_lens,
        )
        _, rebuilt_logits, _ = continuation.append_text(
            rebuilt_state,
            next_text,
            estimated_audio_frames=estimate_speech_frames(tokenizer, next_text),
        )

        probe_frame = frames[-1].view(1, 1, -1).to(runtime.device)
        graph_frame = probe_frame.repeat(state.branch_batch_size, 1, 1)
        _, retained_second = state.graph.run(graph_frame, step_idx=0)
        _, rebuilt_second = rebuilt_graph.run(graph_frame, step_idx=0)
        probe_embeds = state.graph.embed_tokens(probe_frame).repeat(state.branch_batch_size, 1, 1)
        eager_second = _full_logits(
            runtime, [torch.cat([part, probe_embeds[index]], dim=0) for index, part in enumerate(prefix_parts)],
            cfg_scale=state.cfg.guidance_scale,
        )
    finally:
        continuation.close(state)

    result = {
        "seed": args.seed,
        "hybrid_scale_mode": args.hybrid_scale_mode,
        "audio_eos": args.audio_eos,
        "eager_static_cache": args.eager_static_cache,
        "segment_1_frames": len(frames),
        "first_segment_2_logits": _metrics(retained_logits, eager_logits),
        "second_segment_2_logits": _metrics(retained_second, eager_second),
        "retained_vs_rebuilt_static_first": _metrics(retained_logits, rebuilt_logits),
        "retained_vs_rebuilt_static_second": _metrics(retained_second, rebuilt_second),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
