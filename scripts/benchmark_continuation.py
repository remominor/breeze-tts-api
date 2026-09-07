"""Run the gated A/B/C Breeze continuation proof on a local CUDA model."""

from __future__ import annotations

import argparse
import json
import time
import uuid
import wave
from pathlib import Path
from typing import Any

import numpy as np

from breeze_infer.audio import encode_prompt_audio
from breeze_infer.profiles import ProfileStore
from breeze_infer.runtime import (
    load_runtime,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs
from models.continuation_streaming import ContinuationStreamingRuntime
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
from models.warmup_profile import load_warmup_profile

DEFAULT_CHUNKS = (
    "I think that's probably the best approach.",
    "There is one thing we should test first.",
    "If the results stay consistent, we can move ahead with confidence.",
    "After that, we should know whether it is worth pursuing.",
)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _collect(iterator) -> tuple[np.ndarray, list[dict[str, Any]]]:
    audio: list[np.ndarray] = []
    timings: list[dict[str, Any]] = []
    for chunk in iterator:
        audio.append(chunk.audio)
        timings.append(dict(chunk.timing))
    return (
        np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32),
        timings,
    )


def _summary(audio: np.ndarray, sample_rate: int, wall_ms: float) -> dict[str, Any]:
    audio_seconds = float(audio.size / sample_rate)
    return {
        "wall_ms": round(wall_ms, 2),
        "audio_seconds": round(audio_seconds, 4),
        "rtf": round(wall_ms / 1000.0 / audio_seconds, 4) if audio_seconds else None,
        "samples": int(audio.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/Breeze-TTS-2"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--voice-dir",
        type=Path,
        default=Path.home() / ".local/share/breeze-tts/voices",
    )
    parser.add_argument("--voice", default="sky")
    parser.add_argument("--instruction", default="Warm, conversational and engaged.")
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/continuation-proof"))
    args = parser.parse_args()
    weights = args.weights or args.model / "Breeze-TTS-2-int8-hybrid.safetensors"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, audio_tokenizer = load_runtime(
        args.model,
        device="cuda:0",
        attn_implementation="eager",
        weights_path=weights,
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
            collect_timing=True,
        ),
        tokenizer=tokenizer,
    )
    profile = load_warmup_profile(Path("configs/fast.json"))
    runtime.warmup_from_profile(profile)

    store = ProfileStore(args.voice_dir)
    profile_id = store.resolve(args.voice.removeprefix("clone:"))
    profile_data = store.get(profile_id)
    codes = store.load_codes(profile_id)
    if codes is None:
        codes = encode_prompt_audio(audio_tokenizer, store.audio_path(profile_id))
        store.save_codes(profile_id, codes)

    def prepared(text: str, request_id: str) -> dict[str, Any]:
        request = {
            "id": request_id,
            "text": text,
            "speaker": "S0",
            "instruction": args.instruction,
            "ref_text": profile_data["ref_text"],
            "ref_audio_codes": codes,
        }
        return prepare_inputs(
            tokenizer,
            audio_tokenizer,
            model,
            [request],
            get_template("ref_edit_tata"),
            guidance_scale=args.guidance_scale,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )

    results: dict[str, Any] = {
        "config": {
            "voice": profile_id,
            "instruction": args.instruction,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "chunks": list(DEFAULT_CHUNKS),
        }
    }
    full_text = " ".join(DEFAULT_CHUNKS)
    set_all_seeds(args.seed)
    started = time.perf_counter()
    full_audio, full_timings = _collect(
        runtime.iter_audio_chunks(
            prepared(full_text, "proof-full"), request_id="proof-full", seed=args.seed
        )
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    _write_wav(args.output_dir / "a-full.wav", full_audio, runtime.sample_rate)
    results["a_full"] = {
        **_summary(full_audio, runtime.sample_rate, wall_ms),
        "chunks": full_timings,
    }

    independent_audio: list[np.ndarray] = []
    independent_runs: list[dict[str, Any]] = []
    independent_started = time.perf_counter()
    for index, text in enumerate(DEFAULT_CHUNKS):
        set_all_seeds(args.seed)
        chunk_started = time.perf_counter()
        audio, timings = _collect(
            runtime.iter_audio_chunks(
                prepared(text, f"proof-independent-{index}"),
                request_id=f"proof-independent-{index}",
                seed=args.seed,
            )
        )
        independent_audio.append(audio)
        independent_runs.append(
            {
                "chunk_index": index,
                **_summary(
                    audio,
                    runtime.sample_rate,
                    (time.perf_counter() - chunk_started) * 1000.0,
                ),
                "chunks": timings,
            }
        )
    independent = np.concatenate(independent_audio)
    independent_wall_ms = (time.perf_counter() - independent_started) * 1000.0
    _write_wav(args.output_dir / "b-independent.wav", independent, runtime.sample_rate)
    results["b_independent"] = {
        **_summary(independent, runtime.sample_rate, independent_wall_ms),
        "logical_chunks": independent_runs,
    }

    for audio_eos in (False, True):
        mode = "with-eos" if audio_eos else "without-eos"
        continuation = ContinuationStreamingRuntime(runtime, audio_eos=audio_eos)
        continuation_id = uuid.uuid4().hex
        state = continuation.new_session(
            continuation_id, prepared(DEFAULT_CHUNKS[0], f"proof-{mode}-0")
        )
        continued_audio: list[np.ndarray] = []
        logical_runs: list[dict[str, Any]] = []
        continued_started = time.perf_counter()
        try:
            for index, text in enumerate(DEFAULT_CHUNKS):
                chunk_started = time.perf_counter()
                iterator = (
                    continuation.iter_start(
                        state,
                        prepared(text, f"proof-{mode}-{index}"),
                        seed=args.seed,
                    )
                    if index == 0
                    else continuation.iter_continue(state, text)
                )
                audio, timings = _collect(iterator)
                continued_audio.append(audio)
                logical_runs.append(
                    {
                        "chunk_index": index,
                        **_summary(
                            audio,
                            runtime.sample_rate,
                            (time.perf_counter() - chunk_started) * 1000.0,
                        ),
                        "runtime": dict(state.last_timing),
                        "chunks": timings,
                    }
                )
        finally:
            continuation.close(state)
        combined = np.concatenate(continued_audio)
        total_wall_ms = (time.perf_counter() - continued_started) * 1000.0
        _write_wav(
            args.output_dir / f"c-stateful-{mode}.wav", combined, runtime.sample_rate
        )
        results[f"c_stateful_{mode.replace('-', '_')}"] = {
            **_summary(combined, runtime.sample_rate, total_wall_ms),
            "logical_chunks": logical_runs,
        }

    report = args.output_dir / "results.json"
    report.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
