"""Benchmark a running Breeze API and emit machine-readable JSON results."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


def _json_request(url: str, payload: dict[str, Any] | None = None) -> Any:
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    with urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def _post_empty(url: str) -> Any:
    request = Request(url, data=b"", method="POST")
    with urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def _speech_request(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    request = Request(
        f"{url}/v1/audio/speech",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=600) as response:
        sample_rate = int(response.headers.get("X-Sample-Rate", "24000"))
        first = response.read(64 * 1024)
        first_byte_ms = (time.perf_counter() - started) * 1000
        size = len(first)
        while chunk := response.read(64 * 1024):
            size += len(chunk)
    wall_ms = (time.perf_counter() - started) * 1000
    audio_seconds = size / (2 * sample_rate)
    return {
        "wall_ms": round(wall_ms, 2),
        "client_first_read_ms": round(first_byte_ms, 2),
        "audio_seconds": round(audio_seconds, 4),
        "rtf": round(wall_ms / 1000 / audio_seconds, 4),
        "response_bytes": size,
        "server": _json_request(f"{url}/metrics").get("last_request"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--text", default="The cat sat on the mat.")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--reload-before-first", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup_runs < 0 or args.runs <= 0:
        parser.error("--warmup-runs must be >= 0 and --runs must be > 0")

    base_url = args.url.rstrip("/")
    before = _json_request(f"{base_url}/metrics")
    lifecycle = None
    if args.reload_before_first:
        _post_empty(f"{base_url}/v1/model/unload")
        load_started = time.perf_counter()
        load_result = _post_empty(f"{base_url}/v1/model/load")
        lifecycle = {
            "client_load_ms": round((time.perf_counter() - load_started) * 1000, 2),
            "result": load_result,
            "metrics": _json_request(f"{base_url}/metrics")["model_lifecycle"],
        }

    payload = {
        "input": args.text,
        "instructions": args.instruction,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "stream": args.stream,
        "response_format": "pcm",
    }
    for _ in range(args.warmup_runs):
        _speech_request(base_url, payload)
    runs = [_speech_request(base_url, payload) for _ in range(args.runs)]
    result = {
        "request": payload,
        "server_config": before.get("model"),
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.runs,
        "lifecycle": lifecycle,
        "summary": {
            "wall_ms_mean": round(statistics.mean(run["wall_ms"] for run in runs), 2),
            "rtf_mean": round(statistics.mean(run["rtf"] for run in runs), 4),
            "server_ttfa_ms_mean": round(
                statistics.mean(
                    run["server"]["ttfa_ms"]
                    for run in runs
                    if run.get("server") is not None
                    and run["server"].get("ttfa_ms") is not None
                ),
                2,
            ),
        },
        "runs": runs,
        "metrics_before": before,
        "metrics_after": _json_request(f"{base_url}/metrics"),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
