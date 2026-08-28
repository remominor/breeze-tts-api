"""Inspect a Breeze ConvRot safetensors checkpoint without loading a model."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    path = parser.parse_args().checkpoint
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        quant = [key for key in keys if key.endswith(".comfy_quant")]
        scales = [key for key in keys if key.endswith(".weight_scale")]
        components = Counter(key.split(".")[0] for key in keys)
        dtypes = Counter(str(handle.get_tensor(key).dtype) for key in keys if not key.endswith(".comfy_quant"))
        groups = Counter()
        for key in quant:
            import json
            metadata = json.loads(handle.get_tensor(key).numpy().tobytes())
            groups[metadata.get("convrot_groupsize")] += 1
        print(f"file: {path} ({path.stat().st_size / 1024**3:.2f} GiB)")
        print(f"total keys: {len(keys)}")
        print(f"comfy_quant entries: {len(quant)}")
        print(f"weight_scale tensors: {len(scales)}")
        print(f"group sizes: {dict(sorted(groups.items()))}")
        print(f"components: {dict(sorted(components.items()))}")
        print(f"non-metadata dtypes: {dict(dtypes)}")
        print("first quantized prefixes:")
        for key in quant[:20]:
            print(f"  {key.removesuffix('.comfy_quant')}")
        depth = [key for key in quant if key.startswith("depth_decoder.")]
        if depth:
            raise SystemExit(f"ERROR: depth decoder is quantized ({len(depth)} entries)")


if __name__ == "__main__":
    main()
