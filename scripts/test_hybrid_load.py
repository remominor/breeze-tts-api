"""Load the hybrid checkpoint and report residency/quantization invariants."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from breeze_infer.int8_convrot import ConvRotInt8Linear, model_quantization_stats
from breeze_infer.runtime import load_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("weights", type=Path)
    args = parser.parse_args()
    tokenizer, model, _audio_tokenizer = load_runtime(
        args.model_dir, device="cuda:0", attn_implementation="eager", weights_path=args.weights
    )
    del tokenizer
    stats = model_quantization_stats(model)
    components = sorted({name.split(".")[0] for name, module in model.named_modules() if isinstance(module, ConvRotInt8Linear)})
    print(f"weights path: {args.weights}")
    print(f"quantized modules: {stats['quantized_modules']}")
    print(f"quantized parameters: {stats['int8_parameters']}")
    print(f"quantized components: {components}")
    print(f"model stats: {stats}")
    print(f"CUDA allocated MiB: {torch.cuda.memory_allocated() / 2**20:.1f}")
    print(f"CUDA reserved MiB: {torch.cuda.memory_reserved() / 2**20:.1f}")
    first = next(module for module in model.modules() if isinstance(module, ConvRotInt8Linear))
    probe = first(torch.zeros(1, first.in_features, device="cuda", dtype=torch.bfloat16))
    assert probe.shape == (1, first.out_features) and bool(torch.isfinite(probe).all())
    print(f"real ConvRot call: PASS ({first.in_features}->{first.out_features})")
    assert stats["quantized_modules"] == 378
    assert not any(name.startswith("depth_decoder.") for name, module in model.named_modules() if isinstance(module, ConvRotInt8Linear))
    assert stats["meta_parameters"] == 0


if __name__ == "__main__":
    main()
