"""Load the hybrid checkpoint and report residency/quantization invariants."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open

from breeze_infer.int8_convrot import ConvRotInt8Linear, model_quantization_stats
from breeze_infer.runtime import load_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument(
        "--hybrid-scale-mode",
        choices=("bf16_compat", "exact_fp32"),
        default="bf16_compat",
    )
    args = parser.parse_args()
    tokenizer, model, _audio_tokenizer = load_runtime(
        args.model_dir,
        device="cuda:0",
        attn_implementation="eager",
        weights_path=args.weights,
        hybrid_scale_mode=args.hybrid_scale_mode,
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
    assert model.lm_head.weight.dtype == torch.float32
    assert model.depth_decoder.codebooks_head.weight.dtype == torch.float32

    parameters = dict(model.named_parameters(remove_duplicate=False))
    checked_scales = 0
    with safe_open(str(args.weights), framework="pt", device="cpu") as handle:
        for name in handle.keys():  # noqa: SIM118 - safe_open is not iterable
            if not name.endswith(".weight_scale"):
                continue
            actual = parameters[name].detach().cpu()
            expected = handle.get_tensor(name)
            if args.hybrid_scale_mode == "bf16_compat":
                expected = expected.to(torch.bfloat16).float()
            assert actual.dtype == torch.float32
            assert torch.equal(actual, expected), f"loaded ConvRot scale differs: {name}"
            checked_scales += 1
        for name in (
            "lm_head.weight",
            "depth_decoder.codebooks_head.weight",
        ):
            actual = parameters[name].detach().cpu()
            expected = handle.get_tensor(name).float()
            assert torch.equal(actual, expected), f"loaded FP32 head differs: {name}"
    assert checked_scales == stats["quantized_modules"]
    print(
        "checkpoint value audit: PASS "
        f"({checked_scales} scales + 2 heads; scale_mode={args.hybrid_scale_mode})"
    )


if __name__ == "__main__":
    main()
