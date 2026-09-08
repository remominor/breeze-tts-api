from __future__ import annotations

from types import SimpleNamespace

import pytest

from breeze_infer.continuation import (
    ContinuationFingerprint,
    ContinuationManager,
    ContinuationMismatchError,
)


class _Runtime:
    def __init__(self) -> None:
        self.created = []
        self.closed = []

    def new_session(self, continuation_id, _inputs):
        state = SimpleNamespace(
            continuation_id=continuation_id, last_used_at=10.0, closed=False
        )
        self.created.append(state)
        return state

    def close(self, state) -> None:
        state.closed = True
        self.closed.append(state.continuation_id)


def _fingerprint(**overrides) -> ContinuationFingerprint:
    values = {
        "voice_identity": "sky",
        "effective_ref_text": "reference",
        "reference_code_identity": "codes",
        "template": "ref_edit_tata",
        "instruction": "warm",
        "guidance_scale": 4.0,
        "seed": 42,
        "cfg_mode": "single_cfg",
        "sample_rate": 24_000,
    }
    values.update(overrides)
    return ContinuationFingerprint(**values)


def test_first_id_starts_and_matching_id_continues() -> None:
    runtime = _Runtime()
    manager = ContinuationManager(runtime)
    fingerprint = _fingerprint()

    assert manager.classify("one", fingerprint) == ("start", False)
    state = manager.start("one", fingerprint, {})
    manager.finish_request(state)
    assert manager.classify("one", fingerprint) == ("continue", False)
    assert manager.continue_session("one") is state


def test_different_id_implicitly_replaces_idle_state() -> None:
    runtime = _Runtime()
    manager = ContinuationManager(runtime)
    first = manager.start("one", _fingerprint(), {})
    manager.finish_request(first)

    assert manager.classify("two", _fingerprint()) == ("start", True)
    manager.start("two", _fingerprint(), {})

    assert runtime.closed == ["one"]
    assert manager.session.continuation_id == "two"


def test_reject_policy_names_mismatched_field_and_preserves_state() -> None:
    runtime = _Runtime()
    manager = ContinuationManager(runtime, mismatch_policy="reject")
    first = manager.start("one", _fingerprint(), {})
    manager.finish_request(first)

    with pytest.raises(ContinuationMismatchError, match="instruction"):
        manager.classify("one", _fingerprint(instruction="cold"))

    assert manager.session is first
    assert runtime.closed == []


def test_fresh_start_policy_reuses_common_replacement_path() -> None:
    runtime = _Runtime()
    manager = ContinuationManager(runtime, mismatch_policy="fresh_start")
    first = manager.start("one", _fingerprint(), {})
    manager.finish_request(first)

    assert manager.classify("one", _fingerprint(seed=7)) == ("start", True)
    manager.start("one", _fingerprint(seed=7), {})

    assert runtime.closed == ["one"]


def test_ttl_expires_only_idle_state() -> None:
    runtime = _Runtime()
    manager = ContinuationManager(runtime, ttl_seconds=30)
    state = manager.start("one", _fingerprint(), {})

    assert manager.expire(now=100) is False
    manager.finish_request(state)
    state.runtime_state.last_used_at = 50
    assert manager.expire(now=79) is False
    assert manager.expire(now=80) is True
    assert manager.session is None


def test_unstarted_stream_reservation_expires_without_blocking_forever() -> None:
    runtime = _Runtime()
    manager = ContinuationManager(runtime, pending_stream_timeout_seconds=5)
    state = manager.start("one", _fingerprint(), {})
    state.in_flight_since = 10

    assert manager.expire(now=14) is False
    assert manager.expire(now=15) is True
    assert manager.session is None
    assert runtime.closed == ["one"]


def test_normal_cleanup_closes_idle_state() -> None:
    runtime = _Runtime()
    manager = ContinuationManager(runtime)
    state = manager.start("one", _fingerprint(), {})
    manager.finish_request(state)

    assert manager.cleanup_idle() is True
    assert runtime.closed == ["one"]
    assert manager.session is None
