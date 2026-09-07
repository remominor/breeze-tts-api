"""Ownership and fingerprinting for one retained Breeze continuation."""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from models.continuation_streaming import (
    ContinuationRuntimeSession,
    ContinuationStreamingRuntime,
)

MismatchPolicy = Literal["reject", "fresh_start"]


class ContinuationBusyError(RuntimeError):
    pass


class ContinuationMismatchError(RuntimeError):
    def __init__(self, field: str):
        self.field = field
        super().__init__(f"continuation {field} does not match session")


@dataclass(frozen=True)
class ContinuationFingerprint:
    voice_identity: str
    effective_ref_text: str | None
    reference_code_identity: str | None
    template: str
    instruction: str
    guidance_scale: float
    seed: int
    cfg_mode: str
    sample_rate: int

    def mismatch(self, other: ContinuationFingerprint) -> str | None:
        labels = {
            "voice_identity": "voice",
            "effective_ref_text": "reference_text",
            "reference_code_identity": "reference_audio",
            "instruction": "instruction",
            "guidance_scale": "guidance_scale",
            "seed": "seed",
            "cfg_mode": "CFG mode",
            "template": "template",
            "sample_rate": "sample rate",
        }
        for field, label in labels.items():
            if getattr(self, field) != getattr(other, field):
                return label
        return None


@dataclass
class ManagedContinuation:
    continuation_id: str
    fingerprint: ContinuationFingerprint
    runtime_state: ContinuationRuntimeSession
    in_flight: bool = False
    successful_chunks: int = 0
    total_audio_seconds: float = 0.0
    total_wall_seconds: float = 0.0


def reference_code_identity(codes: torch.Tensor | None) -> str | None:
    if codes is None:
        return None
    values = codes.detach().to(torch.int16).cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(np.asarray(values).tobytes())
    return digest.hexdigest()


class ContinuationManager:
    """Manage one retained session; request queuing deliberately lives elsewhere."""

    def __init__(
        self,
        runtime: ContinuationStreamingRuntime,
        *,
        ttl_seconds: float = 30.0,
        mismatch_policy: MismatchPolicy = "reject",
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("continuation TTL must be greater than zero")
        if mismatch_policy not in {"reject", "fresh_start"}:
            raise ValueError("continuation mismatch policy must be reject or fresh_start")
        self.runtime = runtime
        self.ttl_seconds = float(ttl_seconds)
        self.mismatch_policy = mismatch_policy
        self._session: ManagedContinuation | None = None
        self._lock = threading.RLock()

    @property
    def session(self) -> ManagedContinuation | None:
        with self._lock:
            return self._session

    def classify(
        self, continuation_id: str, fingerprint: ContinuationFingerprint
    ) -> tuple[Literal["start", "continue"], bool]:
        """Return operation and whether an existing idle session is replaced."""
        with self._lock:
            current = self._session
            if current is None:
                return "start", False
            if current.in_flight:
                raise ContinuationBusyError("continuation inference is already running")
            if current.continuation_id != continuation_id:
                return "start", True
            mismatch = current.fingerprint.mismatch(fingerprint)
            if mismatch is None:
                return "continue", False
            if self.mismatch_policy == "reject":
                raise ContinuationMismatchError(mismatch)
            return "start", True

    def start(
        self,
        continuation_id: str,
        fingerprint: ContinuationFingerprint,
        inputs: dict,
    ) -> ManagedContinuation:
        with self._lock:
            self._close_locked()
            managed = ManagedContinuation(
                continuation_id=continuation_id,
                fingerprint=fingerprint,
                runtime_state=self.runtime.new_session(continuation_id, inputs),
                in_flight=True,
            )
            self._session = managed
            return managed

    def continue_session(self, continuation_id: str) -> ManagedContinuation:
        with self._lock:
            current = self._session
            if current is None or current.continuation_id != continuation_id:
                raise RuntimeError("continuation state not available")
            if current.in_flight:
                raise ContinuationBusyError("continuation inference is already running")
            current.in_flight = True
            return current

    def finish_request(
        self,
        managed: ManagedContinuation,
        *,
        audio_seconds: float = 0.0,
        wall_seconds: float = 0.0,
    ) -> None:
        with self._lock:
            if self._session is not managed:
                return
            managed.in_flight = False
            managed.successful_chunks += 1
            managed.total_audio_seconds += float(audio_seconds)
            managed.total_wall_seconds += float(wall_seconds)
            managed.runtime_state.last_used_at = time.monotonic()

    def fail_request(self, managed: ManagedContinuation) -> None:
        with self._lock:
            if self._session is managed:
                self._close_locked()

    def cleanup_idle(self) -> bool:
        with self._lock:
            if self._session is None:
                return False
            if self._session.in_flight:
                raise ContinuationBusyError("continuation inference is already running")
            self._close_locked()
            return True

    def expire(self, now: float | None = None) -> bool:
        with self._lock:
            current = self._session
            if current is None or current.in_flight:
                return False
            now = time.monotonic() if now is None else now
            if now - current.runtime_state.last_used_at < self.ttl_seconds:
                return False
            self._close_locked()
            return True

    def _close_locked(self) -> None:
        current = self._session
        if current is None:
            return
        try:
            self.runtime.close(current.runtime_state)
        finally:
            self._session = None
