"""Fail-fast CUDA smoke test for the ConvRot comfy-kitchen kernel."""

from __future__ import annotations

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    try:
        import comfy_kitchen
    except ImportError as exc:
        raise SystemExit("comfy_kitchen is not installed; install the standalone INT8 dependencies") from exc
    from comfy_kitchen.tensor import TensorWiseINT8Layout

    backends = comfy_kitchen.list_backends()
    available = [name for name, info in backends.items() if info.get("available")]
    if not any("cuda" in name.lower() for name in available):
        raise SystemExit(f"No CUDA comfy_kitchen backend: {backends!r}")
    group = 256
    weight, params = TensorWiseINT8Layout.quantize(
        torch.randn(64, group * 2, device="cuda", dtype=torch.bfloat16),
        is_weight=True, per_channel=True, convrot=True,
        convrot_groupsize=group, stochastic_rounding=0,
    )
    output = comfy_kitchen.int8_linear(
        torch.randn(4, group * 2, device="cuda", dtype=torch.bfloat16),
        weight, params.scale, None, out_dtype=torch.bfloat16,
        convrot=True, convrot_groupsize=group,
    )
    if output.shape != (4, 64) or output.dtype != torch.bfloat16 or not bool(torch.isfinite(output).all()):
        raise SystemExit(f"Invalid kernel output: shape={output.shape}, dtype={output.dtype}")
    print(f"PASS: backends={available}, output={tuple(output.shape)} {output.dtype}")


if __name__ == "__main__":
    main()
