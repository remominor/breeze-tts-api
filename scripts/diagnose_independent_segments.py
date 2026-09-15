"""Generate Sky segments as independent normal prompts for boundary comparison."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

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
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
from models.warmup_profile import load_warmup_profile

SKY_INSTRUCTION = "Warm, conversational and engaged, with a relaxed natural rhythm."
SKY_SEGMENTS = (
    "I spent the morning thinking about our next step.",
    "The simplest option may actually be the strongest one.",
    "We can test it carefully, then compare what we learn.",
    "If it holds up, we will move forward with confidence.",
)


def _assemble(segments: list[np.ndarray], gap_samples: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    for index, audio in enumerate(segments):
        if index and gap_samples:
            parts.append(np.zeros(gap_samples, dtype=np.float32))
        parts.append(audio)
    return np.concatenate(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--gap-ms", type=float, default=120.0)
    parser.add_argument(
        "--hybrid-scale-mode", choices=sorted(HYBRID_SCALE_MODES), default="bf16_compat"
    )
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    args = parser.parse_args()
    if args.seed_count < 1 or args.gap_ms < 0 or args.guidance_scale <= 0:
        parser.error("--seed-count and --guidance-scale must be positive; --gap-ms cannot be negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    profiles = ProfileStore(Path.home() / ".local/share/breeze-tts/voices")
    profile_id = profiles.resolve("sky")
    profile = profiles.get(profile_id)
    reference_codes = profiles.load_codes(profile_id)
    if reference_codes is None:
        raise RuntimeError("Sky profile has no cached reference codes")

    reports = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        seed_dir = args.output_dir / f"seed-{seed}"
        boundary_dir = seed_dir / "boundaries"
        seed_dir.mkdir(exist_ok=True)
        boundary_dir.mkdir(exist_ok=True)
        segments: list[np.ndarray] = []
        frames_per_segment: list[int] = []
        for index, text in enumerate(SKY_SEGMENTS):
            inputs = prepare_inputs(
                tokenizer,
                audio_tokenizer,
                model,
                [{"id": f"independent-{seed}-{index}", "text": text, "speaker": "S0",
                  "instruction": SKY_INSTRUCTION, "ref_text": profile["ref_text"],
                  "ref_audio_codes": reference_codes}],
                get_template("ref_edit_tata"),
                guidance_scale=args.guidance_scale,
                guidance_scale_ref=None,
                guidance_scale_ins=None,
            )
            frames: list[torch.Tensor] = []
            chunks = list(
                runtime.iter_audio_chunks(
                    inputs,
                    request_id=uuid.uuid4().hex,
                    seed=seed if index == 0 else None,
                    frame_observer=lambda frame, target=frames: target.append(frame.cpu().clone()),
                )
            )
            audio = np.concatenate([chunk.audio for chunk in chunks]).astype(np.float32)
            sf.write(seed_dir / f"segment_{index + 1:02d}.wav", audio, runtime.sample_rate)
            np.save(seed_dir / f"segment_{index + 1:02d}_frames.npy", torch.stack(frames).numpy())
            segments.append(audio)
            frames_per_segment.append(len(frames))
        gap_samples = round(runtime.sample_rate * args.gap_ms / 1000)
        sf.write(seed_dir / "assembly_contiguous.wav", _assemble(segments, 0), runtime.sample_rate)
        sf.write(seed_dir / f"assembly_gap_{args.gap_ms:g}ms.wav", _assemble(segments, gap_samples), runtime.sample_rate)
        flank = round(runtime.sample_rate * 2.0)
        for index in range(len(segments) - 1):
            sf.write(
                boundary_dir / f"boundary_{index + 1:02d}_contiguous.wav",
                np.concatenate([segments[index][-flank:], segments[index + 1][:flank]]),
                runtime.sample_rate,
            )
        reports.append({"seed": seed, "frames_per_segment": frames_per_segment})
    report = {
        "strategy": "independent_normal_prompts",
        "hybrid_scale_mode": args.hybrid_scale_mode,
        "guidance_scale": args.guidance_scale,
        "gap_ms": args.gap_ms,
        "reports": reports,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output_dir / "report.json")


if __name__ == "__main__":
    main()
