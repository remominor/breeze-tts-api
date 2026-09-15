"""Capture seed-sweep onset data and compare full versus streaming codec decode."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from breeze_infer.api import _normalise_instruction_for_cfg
from breeze_infer.audio import encode_prompt_audio
from breeze_infer.runtime import (
    HYBRID_SCALE_MODES,
    load_runtime,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
from models.stream_runtime import MultiRequestStreamRuntime, QwenStreamRuntimeConfig

WINDOWS_MS = (5, 20, 80, 160, 200)
SIGNIFICANT_THRESHOLDS = (1e-3, 1e-2)


def _capture_frame(
    destination: list[torch.Tensor], limit: int, frame: torch.Tensor
) -> None:
    if len(destination) < limit:
        destination.append(frame.to(device="cpu", dtype=torch.long).clone())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_metadata(model_dir: Path, weights: Path | None) -> dict[str, Any]:
    if weights is not None:
        checkpoint = weights.resolve()
        return {
            "kind": "hybrid",
            "path": str(checkpoint),
            "size": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint),
        }

    index = (model_dir / "model.safetensors.index.json").resolve()
    weight_map = json.loads(index.read_text())["weight_map"]
    shards = sorted({(model_dir / name).resolve() for name in weight_map.values()})
    return {
        "kind": "original",
        "index": str(index),
        "files": [
            {
                "path": str(shard),
                "size": shard.stat().st_size,
                "sha256": _sha256(shard),
            }
            for shard in shards
        ],
    }


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def _wave_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    first_20 = values[: round(sample_rate * 0.020)]
    result: dict[str, Any] = {
        "samples": int(values.size),
        "finite": bool(np.isfinite(values).all()),
        "first_sample": float(values[0]) if values.size else None,
        "first_20ms_peak": float(np.max(np.abs(first_20))) if first_20.size else None,
    }
    for duration_ms in WINDOWS_MS:
        window = values[: round(sample_rate * duration_ms / 1000)]
        result[f"first_{duration_ms}ms_peak"] = (
            float(np.max(np.abs(window))) if window.size else None
        )
        result[f"first_{duration_ms}ms_rms"] = (
            float(np.sqrt(np.mean(np.square(window)))) if window.size else None
        )
    return result


def _comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    sample_rate: int,
    samples_per_frame: int,
) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    size = min(reference.size, candidate.size)
    difference = np.abs(reference[:size] - candidate[:size])
    squared = np.square(reference[:size] - candidate[:size])
    result: dict[str, Any] = {
        "reference_samples": int(reference.size),
        "candidate_samples": int(candidate.size),
        "max_abs_error": float(difference.max()) if size else None,
        "rms_error": float(np.sqrt(squared.mean())) if size else None,
    }
    for threshold in SIGNIFICANT_THRESHOLDS:
        indices = np.flatnonzero(difference > threshold)
        result[f"first_error_over_{threshold:g}"] = (
            int(indices[0]) if indices.size else None
        )
    result["windows"] = {}
    for duration_ms in WINDOWS_MS:
        count = min(size, round(sample_rate * duration_ms / 1000))
        result["windows"][str(duration_ms)] = {
            "max_abs_error": float(difference[:count].max()) if count else None,
            "rms_error": float(np.sqrt(squared[:count].mean())) if count else None,
        }
    result["frames"] = []
    for frame in range(min(12, size // samples_per_frame)):
        start = frame * samples_per_frame
        end = start + samples_per_frame
        result["frames"].append(
            {
                "index": frame,
                "max_abs_error": float(difference[start:end].max()),
                "rms_error": float(np.sqrt(squared[start:end].mean())),
            }
        )
    return result


def _decode_streaming(
    audio_tokenizer: Any,
    codes: torch.Tensor,
    *,
    fast: bool,
    request_id: str,
) -> tuple[np.ndarray, int]:
    device = codes.device
    dtype = next(audio_tokenizer.model.parameters()).dtype
    runtime = MultiRequestStreamRuntime(
        audio_tokenizer,
        QwenStreamRuntimeConfig(
            chunk_frames=1,
            num_lanes=1,
            max_active_reqs=1,
            fast=fast,
            device=device,
            dtype=dtype,
        ),
    )
    chunks = []
    try:
        for index in range(codes.shape[-1]):
            chunks.append(
                runtime.decode_request_chunk(
                    request_id,
                    codes[..., index : index + 1],
                    reset=index == 0,
                )
            )
        waveform = torch.cat(chunks, dim=-1).float().cpu().numpy().reshape(-1)
        return waveform, runtime.samples_per_code
    finally:
        runtime.close_request(request_id)


def _codec_equivalence(
    audio_tokenizer: Any,
    frames: list[torch.Tensor],
    output_dir: Path,
    sample_rate: int,
) -> dict[str, Any]:
    frame_tensor = torch.stack(frames).long()
    device = next(audio_tokenizer.model.parameters()).device
    codes = (
        frame_tensor.transpose(0, 1)
        .unsqueeze(0)
        .to(device=device)
        .contiguous()
    )
    decoder = audio_tokenizer.model.decoder

    default_backend = getattr(decoder.pre_transformer.config, "_attn_implementation", None)
    with torch.inference_mode():
        full_default = (
            audio_tokenizer.model.decode(
                codes.transpose(1, 2), return_dict=True
            ).audio_values[0]
            .float()
            .cpu()
            .numpy()
            .reshape(-1)
        )
    stream_eager, samples_per_frame = _decode_streaming(
        audio_tokenizer, codes, fast=False, request_id="diagnostic-eager"
    )

    decoder.pre_transformer.config._attn_implementation = "eager"
    with torch.inference_mode():
        full_eager = (
            audio_tokenizer.model.decode(
                codes.transpose(1, 2), return_dict=True
            ).audio_values[0]
            .float()
            .cpu()
            .numpy()
            .reshape(-1)
        )
    stream_graph, graph_samples_per_frame = _decode_streaming(
        audio_tokenizer, codes, fast=True, request_id="diagnostic-graph"
    )
    if graph_samples_per_frame != samples_per_frame:
        raise RuntimeError("Eager and graph codec paths disagree on samples per frame")

    waveforms = {
        "full_default": full_default,
        "stream_eager": stream_eager,
        "full_eager_attention": full_eager,
        "stream_cuda_graph": stream_graph,
    }
    for name, waveform in waveforms.items():
        if not np.isfinite(waveform).all():
            raise RuntimeError(f"Non-finite samples in {name}")
        sf.write(output_dir / f"codec-{name}.wav", waveform, sample_rate)

    return {
        "frames": int(codes.shape[-1]),
        "samples_per_frame": samples_per_frame,
        "default_attention_backend": default_backend,
        "waveforms": {
            name: _wave_metrics(waveform, sample_rate)
            for name, waveform in waveforms.items()
        },
        "comparisons": {
            "default_full_vs_eager_stream": _comparison(
                full_default,
                stream_eager,
                sample_rate=sample_rate,
                samples_per_frame=samples_per_frame,
            ),
            "eager_full_vs_cuda_graph_stream": _comparison(
                full_eager,
                stream_graph,
                sample_rate=sample_rate,
                samples_per_frame=samples_per_frame,
            ),
            "eager_stream_vs_cuda_graph_stream": _comparison(
                stream_eager,
                stream_graph,
                sample_rate=sample_rate,
                samples_per_frame=samples_per_frame,
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/Breeze-TTS-2"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/Breeze-TTS-2/Breeze-TTS-2-int8-hybrid.safetensors"),
    )
    parser.add_argument(
        "--original-weights",
        action="store_true",
        help="Load the original sharded BF16 checkpoint from --model-dir.",
    )
    parser.add_argument(
        "--hybrid-scale-mode",
        choices=sorted(HYBRID_SCALE_MODES),
        default="bf16_compat",
        help="ConvRot scale-value policy for hybrid weights.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/startup-audio"))
    parser.add_argument("--label", default="diagnostic")
    parser.add_argument("--text", default="The cat sat on the mat.")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--cfg-scale", type=float)
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--ref-text")
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--capture-frames", type=int, default=32)
    parser.add_argument("--fast-all", action="store_true")
    parser.add_argument("--fast-backbone-decode", action="store_true")
    parser.add_argument("--fast-depth-decoder", action="store_true")
    parser.add_argument("--fast-codec", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seed_count <= 50:
        parser.error("--seed-count must be between 1 and 50")
    if args.capture_frames < 1:
        parser.error("--capture-frames must be positive")
    if (args.ref_audio is None) != (args.ref_text is None):
        parser.error("--ref-audio and --ref-text must be supplied together")

    instruction, cfg_scale = _normalise_instruction_for_cfg(
        args.instruction, args.cfg_scale
    )
    output_dir = args.output_dir / args.label
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = None if args.original_weights else args.weights
    tokenizer, model, audio_tokenizer = load_runtime(
        args.model_dir,
        device="cuda:0",
        attn_implementation="eager",
        weights_path=weights,
        hybrid_scale_mode=args.hybrid_scale_mode,
    )
    update_generation_config_for_breeze(model)
    runtime = FastBreezeStreamingRuntime(
        model,
        audio_tokenizer,
        FastStreamingConfig(
            max_new_tokens=1500,
            max_seq_len=2048,
            fast_all=True if args.fast_all else None,
            fast_backbone_decode=args.fast_backbone_decode,
            fast_depth_decoder=args.fast_depth_decoder,
            fast_codec=args.fast_codec,
        ),
        tokenizer=tokenizer,
    )
    sample_rate = runtime.sample_rate
    reference_codes = (
        encode_prompt_audio(audio_tokenizer, args.ref_audio)
        if args.ref_audio is not None
        else None
    )
    template_name = (
        "ref_edit_tata"
        if reference_codes is not None and instruction
        else "ref_clone_tata"
        if reference_codes is not None
        else "tts_instruction"
        if instruction
        else "tts_plain"
    )

    runs = []
    equivalence_frames: list[torch.Tensor] = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        request = {
            "id": f"diagnostic-{seed}",
            "text": args.text,
            "instruction": instruction,
            "speaker": "S0",
        }
        if reference_codes is not None:
            request.update(ref_audio_codes=reference_codes, ref_text=args.ref_text)
        inputs = prepare_inputs(
            tokenizer,
            audio_tokenizer,
            model,
            [request],
            get_template(template_name),
            guidance_scale=cfg_scale,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        captured: list[torch.Tensor] = []
        observe = partial(_capture_frame, captured, args.capture_frames)

        chunks = list(
            runtime.iter_audio_chunks(
                inputs,
                request_id=f"diagnostic-{seed}",
                seed=seed,
                frame_observer=observe,
            )
        )
        audio = np.concatenate([chunk.audio for chunk in chunks]).astype(np.float32)
        if not np.isfinite(audio).all():
            raise RuntimeError(f"Non-finite generated samples for seed {seed}")
        sf.write(output_dir / f"seed-{seed}.wav", audio, sample_rate)
        np.save(
            output_dir / f"seed-{seed}-first-frames.npy",
            torch.stack(captured).numpy() if captured else np.empty((0, 0), dtype=np.int64),
        )
        first_200 = audio[: round(sample_rate * 0.2)]
        sf.write(output_dir / f"seed-{seed}-first-200ms.wav", first_200, sample_rate)
        runs.append(
            {
                "seed": seed,
                "captured_frames": len(captured),
                "waveform": _wave_metrics(audio, sample_rate),
            }
        )
        if not equivalence_frames:
            equivalence_frames = captured

    codec_equivalence = _codec_equivalence(
        audio_tokenizer,
        equivalence_frames,
        output_dir,
        sample_rate,
    )
    report = {
        "label": args.label,
        "git_revision": _git_revision(),
        "checkpoint": _checkpoint_metadata(args.model_dir, weights),
        "hybrid_scale_mode": args.hybrid_scale_mode,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "request": {
            "text": args.text,
            "instruction": instruction,
            "cfg_scale": cfg_scale,
            "template": template_name,
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
            "fast_all": args.fast_all,
            "fast_backbone_decode": args.fast_backbone_decode,
            "fast_depth_decoder": args.fast_depth_decoder,
            "fast_codec": args.fast_codec,
        },
        "runs": runs,
        "codec_equivalence": codec_equivalence,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
