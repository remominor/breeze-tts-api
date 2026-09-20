"""Exercise repeated API load/unload cycles against host-visible GPU memory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _request_json(url: str, *, method: str = "GET") -> tuple[int, dict]:
    request = Request(url, method=method)
    try:
        with urlopen(request, timeout=180) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        with exc:
            return exc.code, json.load(exc)


def _wait_for_status(base_url: str, expected: int, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status, payload = _request_json(f"{base_url}/health")
            if status == expected:
                return payload
        except (ConnectionError, json.JSONDecodeError, URLError) as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(
        f"API did not reach HTTP {expected} within {timeout:.1f}s"
    ) from last_error


def _gpu_memory_mib(pid: int) -> int | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        process, memory = (part.strip() for part in line.split(",", maxsplit=1))
        if int(process) == pid:
            return int(memory)
    return None


def _wait_for_gpu_release(pid: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _gpu_memory_mib(pid) is None:
            return
        time.sleep(0.1)
    held = _gpu_memory_mib(pid)
    raise RuntimeError(f"server PID {pid} still holds {held} MiB after unload")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify repeated model unloads fully release host-visible VRAM."
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17861)
    parser.add_argument("--timeout", type=float, default=180.0)
    args, server_args = parser.parse_known_args()
    if server_args[:1] == ["--"]:
        server_args = server_args[1:]
    if args.cycles < 1:
        parser.error("--cycles must be at least 1")

    base_url = f"http://{args.host}:{args.port}"
    with tempfile.TemporaryDirectory(prefix="breeze-lifecycle-") as voice_dir:
        command = [
            sys.executable,
            "-m",
            "breeze_infer.api",
            str(args.model),
            "--voice-dir",
            voice_dir,
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
        if args.weights is not None:
            command.extend(("--weights", str(args.weights)))
        command.extend(server_args)
        server = subprocess.Popen(command)
        try:
            _wait_for_status(base_url, 200, args.timeout)
            for cycle in range(1, args.cycles + 1):
                loaded = _gpu_memory_mib(server.pid)
                if loaded is None:
                    raise RuntimeError(
                        f"server PID {server.pid} has no GPU allocation while loaded"
                    )

                status, payload = _request_json(
                    f"{base_url}/v1/model/unload", method="POST"
                )
                if status != 200 or not payload.get("was_loaded"):
                    raise RuntimeError(f"unexpected unload response: {status} {payload}")
                _wait_for_gpu_release(server.pid, args.timeout)
                idle = _wait_for_status(base_url, 503, args.timeout)
                if idle.get("status") != "unloaded":
                    raise RuntimeError(f"unexpected idle health response: {idle}")
                print(
                    f"cycle {cycle}: loaded={loaded} MiB, unloaded=0 MiB",
                    flush=True,
                )

                if cycle < args.cycles:
                    status, payload = _request_json(
                        f"{base_url}/v1/model/load", method="POST"
                    )
                    if status != 200 or payload.get("already_loaded"):
                        raise RuntimeError(
                            f"unexpected load response: {status} {payload}"
                        )
            print(f"passed {args.cycles} model lifecycle cycles", flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
