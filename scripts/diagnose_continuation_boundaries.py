"""Generate one continuation trajectory and replay its boundaries several ways."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from breeze_infer.profiles import ProfileStore
from breeze_infer.runtime import (
    HYBRID_SCALE_MODES,
    load_runtime,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs
from breeze_infer.text_chunks import estimate_speech_frames
from models.continuation_streaming import ContinuationStreamingRuntime
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
from models.warmup_profile import load_warmup_profile

SKY_INSTRUCTION = (
    "Warm, conversational and engaged, with a relaxed natural rhythm."
)
SKY_SEGMENTS = (
    "I spent the morning thinking about our next step.",
    "The simplest option may actually be the strongest one.",
    "We can test it carefully, then compare what we learn.",
    "If it holds up, we will move forward with confidence.",
)
QUIET_THRESHOLDS_DBFS = (-50, -40, -30)


def _to_numpy(audio: torch.Tensor) -> np.ndarray:
    return audio.detach().float().cpu().numpy().reshape(-1)


def _codes(frames: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    return (
        torch.stack(frames)
        .long()
        .transpose(0, 1)
        .unsqueeze(0)
        .to(device=device)
        .contiguous()
    )


def _quiet_edges(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    window_samples = round(sample_rate * 0.010)
    usable = audio[: (audio.size // window_samples) * window_samples]
    if not usable.size:
        return {"window_ms": 10, "thresholds": {}}
    windows = usable.reshape(-1, window_samples).astype(np.float64)
    rms = np.sqrt(np.mean(np.square(windows), axis=1))
    result: dict[str, Any] = {"window_ms": 10, "thresholds": {}}
    for threshold_db in QUIET_THRESHOLDS_DBFS:
        threshold = 10 ** (threshold_db / 20)
        loud = rms > threshold
        if loud.any():
            first = int(np.flatnonzero(loud)[0])
            last = int(np.flatnonzero(loud)[-1])
            leading = first * 10
            trailing = (rms.size - last - 1) * 10
        else:
            leading = trailing = int(rms.size * 10)
        result["thresholds"][str(threshold_db)] = {
            "leading_quiet_ms": leading,
            "trailing_quiet_ms": trailing,
        }
    result["first_500ms_rms_10ms"] = rms[:50].tolist()
    result["last_500ms_rms_10ms"] = rms[-50:].tolist()
    return result


def _eos_event(
    frame_index: int,
    raw_logits: torch.Tensor,
    guidance_scale: float,
    eos_token_id: int,
) -> dict[str, float | int]:
    """Capture EOS competitiveness before one acoustic frame is decoded."""
    rows = raw_logits.float()
    cond = rows[0]
    uncond = rows[1] if rows.shape[0] > 1 else cond
    guided = uncond + guidance_scale * (cond - uncond)

    def metrics(values: torch.Tensor, prefix: str) -> dict[str, float | int]:
        eos_logit = values[eos_token_id]
        probability = torch.softmax(values, dim=-1)[eos_token_id]
        return {
            f"{prefix}_eos_rank": int((values > eos_logit).sum().item()) + 1,
            f"{prefix}_eos_probability": float(probability.item()),
        }

    return {
        "frame_index": frame_index,
        "guidance_scale": guidance_scale,
        **metrics(cond, "conditional"),
        **metrics(uncond, "unconditional"),
        **metrics(guided, "guided"),
    }


def _assemble(
    segments: list[np.ndarray], gap_samples: int
) -> tuple[np.ndarray, list[dict[str, int]]]:
    parts: list[np.ndarray] = []
    boundaries: list[dict[str, int]] = []
    cursor = 0
    for index, segment in enumerate(segments):
        if index:
            gap_start = cursor
            if gap_samples:
                parts.append(np.zeros(gap_samples, dtype=np.float32))
                cursor += gap_samples
            boundaries.append(
                {
                    "after_segment": index,
                    "gap_start_sample": gap_start,
                    "next_segment_sample": cursor,
                }
            )
        parts.append(segment)
        cursor += segment.size
    return np.concatenate(parts), boundaries


def _write_boundary_clips(
    output_dir: Path,
    label: str,
    segments: list[np.ndarray],
    sample_rate: int,
    gap_samples: int,
    flank_samples: int,
) -> None:
    silence = np.zeros(gap_samples, dtype=np.float32)
    for index in range(len(segments) - 1):
        clip = np.concatenate(
            [
                segments[index][-flank_samples:],
                silence,
                segments[index + 1][:flank_samples],
            ]
        )
        sf.write(
            output_dir / f"boundary_{index + 1:02d}_{label}.wav",
            clip,
            sample_rate,
        )


def _round_hybrid_values(
    model: torch.nn.Module, *, scales: bool, heads: bool
) -> None:
    if scales:
        from breeze_infer.int8_convrot import ConvRotInt8Linear

        for module in model.modules():
            if isinstance(module, ConvRotInt8Linear):
                module.weight_scale.data = module.weight_scale.data.to(
                    torch.bfloat16
                ).float()
    if heads:
        model.lm_head.weight.data = model.lm_head.weight.data.to(torch.bfloat16).float()
        model.depth_decoder.codebooks_head.weight.data = (
            model.depth_decoder.codebooks_head.weight.data.to(torch.bfloat16).float()
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/continuation-boundaries")
    )
    parser.add_argument("--original-weights", action="store_true")
    parser.add_argument(
        "--hybrid-scale-mode",
        choices=sorted(HYBRID_SCALE_MODES),
        default="bf16_compat",
    )
    parser.add_argument("--round-hybrid-heads", action="store_true")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="CFG scale used for the reference-direction template.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Backbone acoustic-token repetition penalty for every segment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed set once at the first segment; RNG then progresses through continuations.",
    )
    parser.add_argument(
        "--live-only",
        action="store_true",
        help="Write only the live retained-continuation assembly and review clips.",
    )
    parser.add_argument(
        "--audio-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject the Breeze Audio-EOS embedding before each appended segment.",
    )
    parser.add_argument(
        "--refreshed-cfg",
        action="store_true",
        help=(
            "For segment 2+, re-append the instruction only on the conditional "
            "CFG row. This is an experimental retained-cache layout."
        ),
    )
    parser.add_argument(
        "--continuation-cfg-ramp-low-scale",
        type=float,
        default=None,
        help="Optional low CFG scale used for the first appended acoustic frames.",
    )
    parser.add_argument(
        "--continuation-cfg-ramp-hold-frames",
        type=int,
        default=0,
        help="Number of appended acoustic frames held at the low CFG scale.",
    )
    parser.add_argument(
        "--continuation-cfg-ramp-frames",
        type=int,
        default=0,
        help="Number of following acoustic frames linearly ramped to target CFG.",
    )
    parser.add_argument(
        "--log-eos",
        action="store_true",
        help="Record pre-frame conditional/unconditional/guided EOS metrics.",
    )
    parser.add_argument(
        "--continuation-eos-taper-rank",
        type=int,
        default=None,
        help="Activate a terminal CFG taper when guided EOS reaches this rank.",
    )
    parser.add_argument(
        "--continuation-eos-taper-scale",
        type=float,
        default=None,
        help="CFG scale held after the terminal EOS-rank trigger.",
    )
    parser.add_argument("--gap-ms", type=float, default=120.0)
    parser.add_argument(
        "--boundary-flank-ms",
        type=float,
        default=2000.0,
        help="Audio retained on either side of each boundary clip.",
    )
    args = parser.parse_args()
    if args.original_weights and args.round_hybrid_heads:
        parser.error("--round-hybrid-heads cannot be used with --original-weights")
    if (
        args.gap_ms < 0
        or args.boundary_flank_ms <= 0
        or args.repetition_penalty <= 0
        or args.guidance_scale <= 0
        or (
            args.continuation_cfg_ramp_low_scale is not None
            and (
                args.continuation_cfg_ramp_low_scale <= 0
                or args.continuation_cfg_ramp_low_scale > args.guidance_scale
            )
        )
        or args.continuation_cfg_ramp_hold_frames < 0
        or args.continuation_cfg_ramp_frames < 0
        or (
            args.continuation_eos_taper_rank is not None
            and (
                args.continuation_eos_taper_rank <= 0
                or args.continuation_eos_taper_scale is None
                or args.continuation_eos_taper_scale <= 0
                or args.continuation_eos_taper_scale > args.guidance_scale
            )
        )
    ):
        parser.error(
            "--gap-ms cannot be negative; --boundary-flank-ms and "
            "--repetition-penalty and --guidance-scale must be positive; CFG ramp "
            "frame counts cannot be negative; EOS taper needs a positive rank and scale"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    segment_dir = args.output_dir / "segments"
    boundary_dir = args.output_dir / "boundaries"
    segment_dir.mkdir(exist_ok=True)
    boundary_dir.mkdir(exist_ok=True)

    model_dir = Path("models/Breeze-TTS-2")
    weights = (
        None
        if args.original_weights
        else model_dir / "Breeze-TTS-2-int8-hybrid.safetensors"
    )
    tokenizer, model, audio_tokenizer = load_runtime(
        model_dir,
        device="cuda:0",
        attn_implementation="eager",
        weights_path=weights,
        hybrid_scale_mode=args.hybrid_scale_mode,
    )
    _round_hybrid_values(model, scales=False, heads=args.round_hybrid_heads)
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
            repetition_penalty=args.repetition_penalty,
            collect_timing=True,
        ),
        tokenizer=tokenizer,
    )
    runtime.warmup_from_profile(load_warmup_profile(Path("configs/fast.json")))
    continuation = ContinuationStreamingRuntime(runtime, audio_eos=args.audio_eos)
    samples_per_frame = runtime._codec().samples_per_code

    profiles = ProfileStore(Path.home() / ".local/share/breeze-tts/voices")
    profile_id = profiles.resolve("sky")
    profile = profiles.get(profile_id)
    reference_codes = profiles.load_codes(profile_id)
    if reference_codes is None:
        raise RuntimeError("Sky profile has no cached reference codes")

    def initial_inputs(text: str) -> dict[str, Any]:
        return prepare_inputs(
            tokenizer,
            audio_tokenizer,
            model,
            [
                {
                    "id": "boundary-diagnostic",
                    "text": text,
                    "speaker": "S0",
                    "instruction": SKY_INSTRUCTION,
                    "ref_text": profile["ref_text"],
                    "ref_audio_codes": reference_codes,
                }
            ],
            get_template("ref_edit_tata"),
            guidance_scale=args.guidance_scale,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )

    state = continuation.new_session(uuid.uuid4().hex, initial_inputs(SKY_SEGMENTS[0]))
    live_segments: list[np.ndarray] = []
    segment_frames: list[list[torch.Tensor]] = []
    segment_reports: list[dict[str, Any]] = []
    try:
        for index, text in enumerate(SKY_SEGMENTS):
            frames: list[torch.Tensor] = []
            eos_events: list[dict[str, float | int]] = []
            chunks = list(
                continuation.iter_start(
                    state,
                    initial_inputs(text),
                    seed=args.seed,
                    estimated_audio_frames=estimate_speech_frames(tokenizer, text),
                    frame_observer=lambda frame, target=frames: target.append(
                        frame.cpu().clone()
                    ),
                )
                if index == 0
                else continuation.iter_continue(
                    state,
                    text,
                    refresh_instruction=SKY_INSTRUCTION if args.refreshed_cfg else None,
                    cfg_ramp_low_scale=args.continuation_cfg_ramp_low_scale,
                    cfg_ramp_hold_frames=args.continuation_cfg_ramp_hold_frames,
                    cfg_ramp_frames=args.continuation_cfg_ramp_frames,
                    cfg_eos_taper_rank=args.continuation_eos_taper_rank,
                    cfg_eos_taper_scale=args.continuation_eos_taper_scale,
                    eos_observer=(
                        lambda frame_index, logits, scale, target=eos_events: target.append(
                            _eos_event(
                                frame_index,
                                logits,
                                scale,
                                int(model.config.vocab_size),
                            )
                        )
                        if args.log_eos
                        else None
                    ),
                    estimated_audio_frames=estimate_speech_frames(tokenizer, text),
                    frame_observer=lambda frame, target=frames: target.append(
                        frame.cpu().clone()
                    ),
                )
            )
            audio = np.concatenate([chunk.audio for chunk in chunks]).astype(np.float32)
            decoded_frames = sum(chunk.codec_frames for chunk in chunks)
            np.save(segment_dir / f"segment_{index + 1:02d}_frames.npy", torch.stack(frames).numpy())
            sf.write(segment_dir / f"segment_{index + 1:02d}_live.wav", audio, runtime.sample_rate)
            live_segments.append(audio)
            segment_frames.append(frames)
            segment_reports.append(
                {
                    "index": index + 1,
                    "text": text,
                    "generated_steps": state.last_timing["generated_frames"],
                    "captured_decodable_frames": len(frames),
                    "chunk_decoded_frames": decoded_frames,
                    "pcm_samples": int(audio.size),
                    "expected_pcm_samples": int(len(frames) * samples_per_frame),
                    "quiet_edges": _quiet_edges(audio, runtime.sample_rate),
                    "eos_events": eos_events,
                }
            )
    finally:
        continuation.close(state)

    variants: dict[str, list[np.ndarray]] = {"live": live_segments}
    if not args.live_only:
        device = next(audio_tokenizer.model.parameters()).device
        codec = runtime._codec()
        persistent_segments: list[np.ndarray] = []
        persistent_id = "boundary-replay-persistent"
        with torch.inference_mode():
            try:
                codec.open_request(persistent_id, reset=True, is_first_decode=True)
                for index, frames in enumerate(segment_frames):
                    persistent_segments.append(
                        _to_numpy(
                            codec.decode_request_chunk(
                                persistent_id, _codes(frames, device), reset=index == 0
                            )
                        )
                    )
            finally:
                codec.close_request(persistent_id)

        reset_segments: list[np.ndarray] = []
        with torch.inference_mode():
            for index, frames in enumerate(segment_frames):
                request_id = f"boundary-replay-reset-{index}"
                try:
                    codec.open_request(request_id, reset=True, is_first_decode=True)
                    reset_segments.append(
                        _to_numpy(
                            codec.decode_request_chunk(
                                request_id, _codes(frames, device), reset=True
                            )
                        )
                    )
                finally:
                    codec.close_request(request_id)

        full_segments: list[np.ndarray] = []
        with torch.inference_mode():
            for frames in segment_frames:
                full_segments.append(
                    _to_numpy(
                        audio_tokenizer.model.decode(
                            _codes(frames, device).transpose(1, 2), return_dict=True
                        ).audio_values[0]
                    )
                )
        variants.update(
            {
                "persistent_replay": persistent_segments,
                "reset_replay": reset_segments,
                "full_independent": full_segments,
            }
        )
    gap_samples = round(runtime.sample_rate * args.gap_ms / 1000)
    boundary_flank_samples = round(runtime.sample_rate * args.boundary_flank_ms / 1000)
    assembly_reports = {}
    for label, segments in variants.items():
        for index, audio in enumerate(segments, start=1):
            if label != "live":
                sf.write(
                    segment_dir / f"segment_{index:02d}_{label}.wav",
                    audio,
                    runtime.sample_rate,
                )
        contiguous, contiguous_boundaries = _assemble(segments, 0)
        gapped, gapped_boundaries = _assemble(segments, gap_samples)
        sf.write(args.output_dir / f"assembly_{label}_contiguous.wav", contiguous, runtime.sample_rate)
        sf.write(args.output_dir / f"assembly_{label}_gap_{args.gap_ms:g}ms.wav", gapped, runtime.sample_rate)
        _write_boundary_clips(
            boundary_dir,
            f"{label}_contiguous",
            segments,
            runtime.sample_rate,
            0,
            boundary_flank_samples,
        )
        _write_boundary_clips(
            boundary_dir,
            f"{label}_gap_{args.gap_ms:g}ms",
            segments,
            runtime.sample_rate,
            gap_samples,
            boundary_flank_samples,
        )
        assembly_reports[label] = {
            "segment_samples": [int(segment.size) for segment in segments],
            "contiguous_boundaries": contiguous_boundaries,
            "gapped_boundaries": gapped_boundaries,
        }

    report = {
        "weights": "original" if args.original_weights else "hybrid",
        "hybrid_scale_mode": args.hybrid_scale_mode,
        "round_hybrid_heads": args.round_hybrid_heads,
        "repetition_penalty": args.repetition_penalty,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "sample_rate": runtime.sample_rate,
        "gap_ms": args.gap_ms,
        "boundary_flank_ms": args.boundary_flank_ms,
        "audio_eos": args.audio_eos,
        "refreshed_cfg": args.refreshed_cfg,
        "continuation_cfg_ramp_low_scale": args.continuation_cfg_ramp_low_scale,
        "continuation_cfg_ramp_hold_frames": args.continuation_cfg_ramp_hold_frames,
        "continuation_cfg_ramp_frames": args.continuation_cfg_ramp_frames,
        "log_eos": args.log_eos,
        "continuation_eos_taper_rank": args.continuation_eos_taper_rank,
        "continuation_eos_taper_scale": args.continuation_eos_taper_scale,
        "segments": segment_reports,
        "assemblies": assembly_reports,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
