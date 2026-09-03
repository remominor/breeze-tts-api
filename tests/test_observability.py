from __future__ import annotations

import sys
from types import SimpleNamespace

from breeze_infer.observability import (
    RunningStats,
    _nvml_process_memory,
    cuda_snapshot,
)


def test_running_stats_use_constant_size_aggregates() -> None:
    stats = RunningStats()
    for value in (3.0, 1.0, 5.0):
        stats.observe(value)

    assert stats.snapshot() == {
        "count": 3,
        "last": 5.0,
        "mean": 3.0,
        "min": 1.0,
        "max": 5.0,
    }


def test_cuda_snapshot_does_not_initialize_cuda(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        "breeze_infer.observability._nvml_process_memory",
        lambda: {
            "nvml_available": True,
            "process_memory_mb": 0.0,
            "process_devices": [],
        },
    )

    snapshot = cuda_snapshot()

    assert snapshot["available"] is True
    assert snapshot["initialized"] is False
    assert snapshot["allocated_mb"] == 0.0


def test_nvml_failure_is_reported_without_raising(monkeypatch) -> None:
    fake = SimpleNamespace(
        nvmlInit=lambda: (_ for _ in ()).throw(RuntimeError("no driver"))
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    snapshot = _nvml_process_memory()

    assert snapshot["nvml_available"] is False
    assert snapshot["process_memory_mb"] is None
    assert "no driver" in snapshot["nvml_error"]


def test_nvml_is_shutdown_after_snapshot(monkeypatch) -> None:
    calls = []
    fake = SimpleNamespace(
        NVML_VALUE_NOT_AVAILABLE=-1,
        NVMLError=RuntimeError,
        nvmlInit=lambda: calls.append("init"),
        nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetCount=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    snapshot = _nvml_process_memory()

    assert snapshot["nvml_available"] is True
    assert calls == ["init", "shutdown"]
