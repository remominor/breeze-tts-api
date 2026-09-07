"""Generate five multi-request continuation WAVs for listening review."""

from __future__ import annotations

import json
import time
import uuid
import wave
from pathlib import Path

import numpy as np

from breeze_infer.audio import encode_prompt_audio
from breeze_infer.profiles import ProfileStore
from breeze_infer.runtime import load_runtime, update_generation_config_for_breeze
from breeze_infer.templates import get_template, prepare_inputs
from breeze_infer.text_chunks import estimate_speech_frames
from models.continuation_streaming import ContinuationStreamingRuntime
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
from models.warmup_profile import load_warmup_profile

SAMPLES = (
    (
        "sky",
        "Warm, conversational and engaged, with a relaxed natural rhythm.",
        (
            "I spent the morning thinking about our next step.",
            "The simplest option may actually be the strongest one.",
            "We can test it carefully, then compare what we learn.",
            "If it holds up, we will move forward with confidence.",
        ),
    ),
    (
        "adam",
        "Bright, energetic and optimistic, like sharing genuinely exciting news.",
        (
            "I have some excellent news to share with you.",
            "The early results came back better than we expected.",
            "Even the difficult cases improved by a noticeable margin.",
            "Now we get to turn that momentum into something remarkable.",
        ),
    ),
    (
        "whisper",
        "Quiet, confidential and suspenseful, while remaining clearly intelligible.",
        (
            "Keep your voice down, because someone may still be nearby.",
            "I noticed a light moving behind the upstairs window.",
            "Then the hallway went silent, all at once.",
            "We should leave now, before whoever is there comes looking.",
        ),
    ),
    (
        "soft_whisper",
        "Gentle, reassuring and intimate, with patient pauses and a soft delivery.",
        (
            "You do not have to solve everything tonight.",
            "Take a slow breath, and let your shoulders relax.",
            "Tomorrow will give us more room to understand the problem.",
            "For now, it is enough to know that you are not alone.",
        ),
    ),
    (
        "goblin",
        "Playful, theatrical and mischievous, with animated changes in emphasis.",
        (
            "At last, the mysterious package has arrived at my door.",
            "It rattles when I shake it, which is usually a promising sign.",
            "The label says not to open it beneath the moonlight.",
            "Naturally, that means we shall wait until midnight.",
        ),
    ),
)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def main() -> None:
    model_dir = Path("models/Breeze-TTS-2")
    tokenizer, model, audio_tokenizer = load_runtime(
        model_dir,
        device="cuda:0",
        attn_implementation="eager",
        weights_path=model_dir / "Breeze-TTS-2-int8-hybrid.safetensors",
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
    runtime.warmup_from_profile(load_warmup_profile(Path("configs/fast.json")))
    continuation = ContinuationStreamingRuntime(runtime, audio_eos=True)
    profiles = ProfileStore(Path.home() / ".local/share/breeze-tts/voices")
    report = []

    for sample_index, (voice, instruction, segments) in enumerate(SAMPLES, start=1):
        profile_id = profiles.resolve(voice)
        profile = profiles.get(profile_id)
        codes = profiles.load_codes(profile_id)
        if codes is None:
            codes = encode_prompt_audio(audio_tokenizer, profiles.audio_path(profile_id))
            profiles.save_codes(profile_id, codes)

        def initial_inputs(
            text: str,
            *,
            current_index=sample_index,
            current_instruction=instruction,
            current_profile=profile,
            current_codes=codes,
        ):
            return prepare_inputs(
                tokenizer,
                audio_tokenizer,
                model,
                [
                    {
                        "id": f"sample-{current_index}",
                        "text": text,
                        "speaker": "S0",
                        "instruction": current_instruction,
                        "ref_text": current_profile["ref_text"],
                        "ref_audio_codes": current_codes,
                    }
                ],
                get_template("ref_edit_tata"),
                guidance_scale=4.0,
                guidance_scale_ref=None,
                guidance_scale_ins=None,
            )

        state = continuation.new_session(uuid.uuid4().hex, initial_inputs(segments[0]))
        parts: list[np.ndarray] = []
        started = time.perf_counter()
        try:
            for segment_index, text in enumerate(segments):
                estimated = estimate_speech_frames(tokenizer, text)
                iterator = (
                    continuation.iter_start(
                        state,
                        initial_inputs(text),
                        seed=42,
                        estimated_audio_frames=estimated,
                    )
                    if segment_index == 0
                    else continuation.iter_continue(
                        state, text, estimated_audio_frames=estimated
                    )
                )
                parts.extend(chunk.audio for chunk in iterator)
        finally:
            continuation.close(state)
        audio = np.concatenate(parts)
        output = Path(f"continuation_sample_{sample_index:02d}_{voice}.wav")
        _write_wav(output, audio, runtime.sample_rate)
        audio_seconds = audio.size / runtime.sample_rate
        wall_seconds = time.perf_counter() - started
        report.append(
            {
                "file": str(output),
                "voice": voice,
                "instruction": instruction,
                "segments": len(segments),
                "audio_seconds": round(audio_seconds, 2),
                "rtf": round(wall_seconds / audio_seconds, 4),
            }
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
