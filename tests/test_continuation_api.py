from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi import HTTPException

import breeze_infer.api as api_module
from breeze_infer.api import app, continuation_speech, speech
from breeze_infer.continuation import ContinuationManager
from breeze_infer.observability import new_service_metrics


class _Request:
    def __init__(self, body: dict):
        self.body = body
        self.headers = {"content-type": "application/json"}

    async def json(self):
        return self.body

    async def is_disconnected(self) -> bool:
        return False


class _Profiles:
    def resolve(self, _identifier):
        from breeze_infer.profiles import ProfileNotFoundError

        raise ProfileNotFoundError

    def list(self):
        return []


class _Tokenizer:
    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(max(1, len(text.split()))))}

    def convert_ids_to_tokens(self, ids):
        return ["token"] * len(ids)


class _ContinuationRuntime:
    sample_rate = 24_000

    def __init__(self):
        self.closed = []

    def new_session(self, continuation_id, _inputs):
        return SimpleNamespace(
            continuation_id=continuation_id,
            chunk_index=0,
            last_timing={},
            last_used_at=time.monotonic(),
            closed=False,
        )

    def _chunks(self, state, operation):
        yield SimpleNamespace(audio=np.array([0.25, -0.25], dtype=np.float32))
        state.last_timing = {
            "text_encoder_ms": 1.0 if operation == "continue" else 0.0,
            "append_prefill_ms": 2.0,
            "cache_length_after": 20,
        }
        state.chunk_index += 1

    def iter_start(self, state, _inputs, **_kwargs):
        return self._chunks(state, "start")

    def iter_continue(self, state, _text, **_kwargs):
        return self._chunks(state, "continue")

    def close(self, state):
        state.closed = True
        self.closed.append(state.continuation_id)


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    continuation_runtime = _ContinuationRuntime()

    class _NormalRuntime:
        fast_enabled = True
        sample_rate = 24_000

        def iter_audio_chunks(self, _inputs, **_kwargs):
            yield SimpleNamespace(audio=np.array([0.1, -0.1], dtype=np.float32))

    app.state.metrics = new_service_metrics()
    app.state.profiles = _Profiles()
    app.state.runtime = _NormalRuntime()
    app.state.tokenizer = _Tokenizer()
    app.state.audio_tokenizer = object()
    app.state.model = object()
    app.state.continuations = ContinuationManager(continuation_runtime)
    app.state.cfg = SimpleNamespace(
        max_ref_audio_bytes=1024,
        weights=None,
        fast_all=None,
        fast_text_encoder=False,
        fast_backbone_prefill=False,
        fast_backbone_decode=True,
        fast_depth_decoder=False,
        fast_codec=False,
    )
    app.state.start_time = time.monotonic()
    monkeypatch.setattr(api_module, "set_all_seeds", lambda _seed: None)
    monkeypatch.setattr(
        api_module,
        "prepare_inputs",
        lambda *_args, **_kwargs: {
            "input_ids": torch.zeros((1, 8), dtype=torch.long),
            "cfg_negative_prompt_ids": torch.zeros((1, 8), dtype=torch.long),
        },
    )
    yield continuation_runtime
    if api_module._request_lock.locked():
        api_module._request_lock.release()


def _payload(continuation_id="one", **overrides):
    result = {
        "input": "hello there",
        "continuation_id": continuation_id,
        "guidance_scale": 1,
        "seed": 42,
        "response_format": "pcm",
    }
    result.update(overrides)
    return result


def test_id_implicitly_starts_then_continues(configured) -> None:
    first = asyncio.run(continuation_speech(_Request(_payload())))
    second = asyncio.run(continuation_speech(_Request(_payload(input="again"))))

    assert first.headers["x-continuation-chunk-index"] == "0"
    assert second.headers["x-continuation-chunk-index"] == "1"
    assert first.headers["x-continuation-restarted"] == "false"
    assert app.state.continuations.session.successful_chunks == 2
    assert configured.closed == []


def test_different_id_replaces_idle_session(configured) -> None:
    asyncio.run(continuation_speech(_Request(_payload("one"))))
    response = asyncio.run(continuation_speech(_Request(_payload("two"))))

    assert configured.closed == ["one"]
    assert response.headers["x-continuation-restarted"] == "true"
    assert app.state.continuations.session.continuation_id == "two"


def test_same_id_mismatch_is_rejected_without_destroying_state() -> None:
    asyncio.run(continuation_speech(_Request(_payload())))

    with pytest.raises(HTTPException, match="instruction") as exc_info:
        asyncio.run(
            continuation_speech(_Request(_payload(instructions="speak warmly")))
        )

    assert exc_info.value.status_code == 409
    assert app.state.continuations.session.continuation_id == "one"


def test_normal_speech_cleans_idle_continuation(configured) -> None:
    asyncio.run(continuation_speech(_Request(_payload())))
    response = asyncio.run(speech(_Request({"input": "normal", "response_format": "pcm"})))

    assert response.status_code == 200
    assert configured.closed == ["one"]
    assert app.state.continuations.session is None


def test_continuation_requires_client_id() -> None:
    with pytest.raises(HTTPException, match="continuation_id") as exc_info:
        asyncio.run(continuation_speech(_Request({"input": "hello"})))

    assert exc_info.value.status_code == 422


def test_omitted_and_explicit_default_cfg_match() -> None:
    asyncio.run(
        continuation_speech(
            _Request(
                {
                    "input": "hello",
                    "continuation_id": "one",
                    "response_format": "pcm",
                }
            )
        )
    )
    response = asyncio.run(continuation_speech(_Request(_payload(input="again"))))

    assert response.headers["x-continuation-chunk-index"] == "1"


def test_stream_releases_request_lock_but_retains_session() -> None:
    response = asyncio.run(
        continuation_speech(_Request(_payload(stream=True, response_format="pcm")))
    )
    assert api_module._request_lock.locked()

    async def consume() -> bytes:
        parts = []
        async for part in response.body_iterator:
            parts.append(part)
        return b"".join(parts)

    assert asyncio.run(consume())
    asyncio.run(response.background())

    assert not api_module._request_lock.locked()
    assert app.state.continuations.session is not None
    assert app.state.continuations.session.in_flight is False
