"""Bounded service statistics and best-effort process/GPU telemetry."""

from __future__ import annotations

import os
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import psutil


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    last: float | None = None

    def observe(self, value: float) -> None:
        value = float(value)
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.last = value

    def snapshot(self, *, digits: int = 1) -> dict[str, float | int | None]:
        mean = self.total / self.count if self.count else None

        def rounded(value: float | None) -> float | None:
            return round(value, digits) if value is not None else None

        return {
            "count": self.count,
            "last": rounded(self.last),
            "mean": rounded(mean),
            "min": rounded(self.minimum),
            "max": rounded(self.maximum),
        }


def new_service_metrics() -> dict[str, Any]:
    return {
        "requests_total": 0,
        "requests_success": 0,
        "requests_error": 0,
        "requests_busy": 0,
        "streaming_total": 0,
        "streaming_design": 0,
        "streaming_clone": 0,
        "cfg_no_cfg_requests": 0,
        "cfg_guided_requests": 0,
        "model_load_attempts": 0,
        "model_load_successes": 0,
        "model_load_failures": 0,
        "model_unloads": 0,
        "generated_audio_seconds_total": 0.0,
        "latency_ms": RunningStats(),
        "ttfa_ms": RunningStats(),
        "rtf": RunningStats(),
        "model_load_ms": RunningStats(),
        "model_unload_ms": RunningStats(),
        "last_request": None,
    }


def observe(metrics: dict[str, Any], name: str, value: float) -> None:
    stats = metrics.get(name)
    if not isinstance(stats, RunningStats):
        stats = RunningStats()
        metrics[name] = stats
    stats.observe(value)


def stats_snapshot(metrics: dict[str, Any], name: str) -> dict[str, Any]:
    stats = metrics.get(name)
    return (
        stats.snapshot()
        if isinstance(stats, RunningStats)
        else RunningStats().snapshot()
    )


def process_snapshot(start_time: float) -> dict[str, Any]:
    process = psutil.Process()
    memory = process.memory_info()
    cpu = process.cpu_times()
    return {
        "pid": process.pid,
        "uptime_s": round(time.monotonic() - start_time, 1),
        "rss_mb": round(memory.rss / 2**20, 1),
        "vms_mb": round(memory.vms / 2**20, 1),
        "cpu_user_s": round(cpu.user, 2),
        "cpu_system_s": round(cpu.system, 2),
        "threads": process.num_threads(),
    }


def _nvml_process_memory() -> dict[str, Any]:
    """Return nvidia-smi-style memory without creating a CUDA context."""
    pynvml = None
    initialized = False
    try:
        import pynvml

        pynvml.nvmlInit()
        initialized = True
        used = 0
        matched_devices: list[int] = []
        unavailable = getattr(pynvml, "NVML_VALUE_NOT_AVAILABLE", None)
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            process_memory: dict[int, int] = {}
            for getter_name in (
                "nvmlDeviceGetComputeRunningProcesses",
                "nvmlDeviceGetGraphicsRunningProcesses",
            ):
                getter = getattr(pynvml, getter_name, None)
                if getter is None:
                    continue
                try:
                    for entry in getter(handle):
                        if entry.usedGpuMemory == unavailable:
                            continue
                        process_memory[entry.pid] = max(
                            process_memory.get(entry.pid, 0), int(entry.usedGpuMemory)
                        )
                except pynvml.NVMLError:
                    continue
            device_used = process_memory.get(os.getpid(), 0)
            if device_used:
                matched_devices.append(index)
                used += device_used
        return {
            "nvml_available": True,
            "process_memory_mb": round(used / 2**20, 1),
            "process_devices": matched_devices,
        }
    except Exception as exc:  # noqa: BLE001 - telemetry must remain best effort
        return {
            "nvml_available": False,
            "process_memory_mb": None,
            "process_devices": [],
            "nvml_error": str(exc),
        }
    finally:
        if initialized and pynvml is not None:
            with suppress(Exception):
                pynvml.nvmlShutdown()


def cuda_snapshot(lifetime_peak_allocated_mb: float = 0.0) -> dict[str, Any]:
    snapshot = _nvml_process_memory()
    try:
        import torch

        snapshot["available"] = torch.cuda.is_available()
        snapshot["initialized"] = torch.cuda.is_initialized()
        if not snapshot["initialized"]:
            snapshot.update(
                {
                    "device_index": None,
                    "device_name": None,
                    "allocated_mb": 0.0,
                    "reserved_mb": 0.0,
                    "peak_allocated_mb": round(lifetime_peak_allocated_mb, 1),
                    "peak_reserved_mb": 0.0,
                    "device_free_mb": None,
                    "device_total_mb": None,
                }
            )
            return snapshot
        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        current_peak = torch.cuda.max_memory_allocated(device) / 2**20
        snapshot.update(
            {
                "device_index": device,
                "device_name": torch.cuda.get_device_name(device),
                "allocated_mb": round(torch.cuda.memory_allocated(device) / 2**20, 1),
                "reserved_mb": round(torch.cuda.memory_reserved(device) / 2**20, 1),
                "peak_allocated_mb": round(
                    max(lifetime_peak_allocated_mb, current_peak), 1
                ),
                "peak_reserved_mb": round(
                    torch.cuda.max_memory_reserved(device) / 2**20, 1
                ),
                "device_free_mb": round(free / 2**20, 1),
                "device_total_mb": round(total / 2**20, 1),
            }
        )
    except Exception as exc:  # noqa: BLE001 - telemetry must remain best effort
        snapshot.update(
            {"available": False, "initialized": False, "torch_error": str(exc)}
        )
    return snapshot
