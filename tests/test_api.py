from __future__ import annotations

import asyncio
import inspect
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi import HTTPException

from breeze_infer.api import (
    DEFAULT_CFG_SCALE,
    DEFAULT_INSTRUCTION,
    _normalise_instruction_for_cfg,
    _parse_seed,
    _parse_stream,
    _pcm16,
    _profile_request,
    _prompt_token_count,
    _quiet_expected_torch_compile_warnings,
    _voice_item,
    app,
    speech,
)
from breeze_infer.api import (
    MAX_NEW_TOKENS as API_MAX_NEW_TOKENS,
)
from breeze_infer.api import (
    MAX_SEQ_LEN as API_MAX_SEQ_LEN,
)
from breeze_infer.profiles import ProfileStore
from infer import MAX_NEW_TOKENS as CLI_MAX_NEW_TOKENS
from infer import MAX_SEQ_LEN as CLI_MAX_SEQ_LEN


class _JsonRequest:
    def __init__(self, body: dict):
        self.body = body
        self.headers = {"content-type": "application/json"}

    async def json(self) -> dict:
        return self.body


class _Tokenizer:
    def __call__(self, text: str, **_kwargs):
        return {"input_ids": list(range(max(1, len(text.split()))))}

    def convert_ids_to_tokens(self, ids):
        return ["token"] * len(ids)


def _configure_fake_speech_state(monkeypatch, *, runtime_error: bool = False) -> None:
    import breeze_infer.api as api_module

    class _Runtime:
        fast_enabled = True
        sample_rate = 24_000

        def iter_audio_chunks(self, _inputs, *, request_id, seed=None):
            if runtime_error:
                raise RuntimeError(f"failed {request_id}")
            yield SimpleNamespace(audio=np.array([0.25, -0.25], dtype=np.float32))

    app.state.metrics = {
        "requests_total": 0,
        "requests_success": 0,
        "requests_error": 0,
        "requests_busy": 0,
        "streaming_total": 0,
        "streaming_design": 0,
        "streaming_clone": 0,
        "ttfa_ms": [],
        "latency_ms": [],
    }
    app.state.profiles = _MissingProfileStore()
    app.state.runtime = _Runtime()
    app.state.eager_runtime = None
    app.state.tokenizer = _Tokenizer()
    app.state.audio_tokenizer = object()
    app.state.model = object()
    app.state.cfg = SimpleNamespace(max_ref_audio_bytes=1024)
    monkeypatch.setattr(api_module, "set_all_seeds", lambda _seed: None)
    monkeypatch.setattr(
        api_module,
        "prepare_inputs",
        lambda *_args, **_kwargs: {
            "input_ids": torch.zeros((1, 8), dtype=torch.long),
            "cfg_negative_prompt_ids": torch.zeros((1, 8), dtype=torch.long),
        },
    )


def test_api_exposes_only_health_and_streaming_speech() -> None:
    paths = {route.path for route in app.routes if route.path.startswith("/")}

    assert "/health" in paths
    assert "/v1/audio/speech" in paths
    assert "/api/ref-audio-codes" not in paths


def test_speech_accepts_a_request_object() -> None:
    assert list(inspect.signature(speech).parameters) == ["request"]


def test_api_cfg_defaults_to_one() -> None:
    assert DEFAULT_CFG_SCALE == 1.0


def test_explicit_cfg_without_direction_uses_neutral_instruction() -> None:
    instruction, scale = _normalise_instruction_for_cfg(
        "", "4", fast_enabled=True
    )

    assert instruction == DEFAULT_INSTRUCTION
    assert scale == 4.0


def test_fast_cfg_one_without_direction_is_promoted_to_fast_default() -> None:
    instruction, scale = _normalise_instruction_for_cfg(
        "", "1", fast_enabled=True
    )

    assert instruction == DEFAULT_INSTRUCTION
    assert scale == 4.0


def test_cfg_one_without_direction_remains_plain_without_fast_profile() -> None:
    instruction, scale = _normalise_instruction_for_cfg(
        "", "1", fast_enabled=False
    )

    assert instruction == ""
    assert scale == 1.0


def test_fast_auto_request_uses_neutral_cfg_four_instruction() -> None:
    instruction, scale = _normalise_instruction_for_cfg(
        "", "", fast_enabled=True
    )

    assert instruction == DEFAULT_INSTRUCTION
    assert scale == 4.0


@pytest.mark.parametrize("value", ["bad", object()])
def test_invalid_cfg_values_return_422(value: object) -> None:
    with pytest.raises(HTTPException) as exc_info:
        _normalise_instruction_for_cfg("", value, fast_enabled=False)

    assert exc_info.value.status_code == 422


def test_automatic_cfg_none_remains_supported() -> None:
    assert _normalise_instruction_for_cfg("", None, fast_enabled=False) == ("", 1.0)


def test_seed_and_stream_parsing_report_client_errors() -> None:
    assert _parse_seed("17") == 17
    assert _parse_stream("TRUE") is True
    assert _parse_stream(None) is False

    with pytest.raises(HTTPException, match="seed") as seed_error:
        _parse_seed("seventeen")
    with pytest.raises(HTTPException, match="stream") as stream_error:
        _parse_stream("sometimes")
    assert seed_error.value.status_code == 422
    assert stream_error.value.status_code == 422


def test_prompt_token_count_uses_longest_cfg_branch() -> None:
    inputs = {
        "input_ids": torch.zeros((1, 10)),
        "cfg_negative_prompt_ids": torch.zeros((1, 14)),
    }

    assert _prompt_token_count(inputs) == 14


def test_expected_torch_compile_warnings_are_quieted() -> None:
    _quiet_expected_torch_compile_warnings()

    assert logging.getLogger("torch.fx.experimental.symbolic_shapes").level == logging.ERROR
    assert logging.getLogger("torch._inductor.utils").level == logging.ERROR


class _MissingProfileStore:
    def resolve(self, _identifier: str) -> str:
        from breeze_infer.profiles import ProfileNotFoundError

        raise ProfileNotFoundError


def test_unknown_clone_voice_does_not_silently_use_voice_design() -> None:
    with pytest.raises(HTTPException) as exc_info:
        _profile_request(
            _MissingProfileStore(),
            "clone:missing",
            ref_text=None,
            instruction="",
        )

    assert exc_info.value.status_code == 404


def test_default_voice_can_fall_back_to_voice_design() -> None:
    request, template = _profile_request(
        _MissingProfileStore(), "default", ref_text=None, instruction=""
    )

    assert request == {"speaker": "S0"}
    assert template == "tts_plain"


def test_blank_ref_text_uses_stored_profile_transcript(tmp_path) -> None:
    store = ProfileStore(tmp_path)
    store.save("alice", b"reference", "stored transcript")

    request, template = _profile_request(
        store, "alice", ref_text="   ", instruction=""
    )

    assert request["ref_text"] == "stored transcript"
    assert template == "ref_clone_tata"


def test_voice_item_uses_stable_profile_id_not_editable_name() -> None:
    item = _voice_item(
        {"profile_id": "profile-123", "name": "Renamed Voice", "ref_text": "hi"}
    )

    assert item["id"] == "profile-123"
    assert item["voice_id"] == "profile-123"
    assert item["name"] == "Renamed Voice"


def test_cli_and_api_support_1500_generated_tokens() -> None:
    assert CLI_MAX_NEW_TOKENS == API_MAX_NEW_TOKENS == 1500
    assert CLI_MAX_SEQ_LEN == 2048
    assert API_MAX_SEQ_LEN == 2048


def test_pcm16_clips_and_encodes_little_endian() -> None:
    encoded = _pcm16(np.array([-2.0, 0.0, 2.0], dtype=np.float32))

    assert np.frombuffer(encoded, dtype="<i2").tolist() == [-32767, 0, 32767]


def test_abandoned_stream_background_releases_inference_lock(monkeypatch) -> None:
    import breeze_infer.api as api_module

    _configure_fake_speech_state(monkeypatch)
    response = asyncio.run(
        speech(_JsonRequest({"input": "hello", "stream": True}))
    )
    assert api_module._request_lock.locked()

    asyncio.run(response.background())

    assert not api_module._request_lock.locked()


def test_raw_stream_runtime_failure_is_not_reported_as_success(monkeypatch) -> None:
    import breeze_infer.api as api_module

    _configure_fake_speech_state(monkeypatch, runtime_error=True)
    response = asyncio.run(
        speech(_JsonRequest({"input": "hello", "stream": True}))
    )

    async def consume() -> None:
        async for _chunk in response.body_iterator:
            pass

    try:
        with pytest.raises(RuntimeError, match="failed api-"):
            asyncio.run(consume())
    finally:
        asyncio.run(response.background())

    assert app.state.metrics["requests_total"] == 1
    assert app.state.metrics["requests_success"] == 0
    assert app.state.metrics["requests_error"] == 1
    assert not api_module._request_lock.locked()
