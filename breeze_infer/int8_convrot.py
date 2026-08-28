"""Minimal standalone ConvRot INT8 support for Breeze TTS 2.

The checkpoint format and kernel call are adapted from
Saganaki22/ComfyUI-Breeze-TTS-2 ``int8.py``.  That implementation is
Apache-2.0 licensed; this module intentionally does not import ComfyUI.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)
QUANT_META_SUFFIX = "comfy_quant"
SUPPORTED_FORMAT = "int8_tensorwise"


@dataclass(frozen=True)
class QuantLayerInfo:
    prefix: str
    group_size: int
    in_features: int = 0
    out_features: int = 0
    has_bias: bool = False


def _validate_group_size(group_size: int, in_features: int) -> None:
    if group_size < 4 or group_size & (group_size - 1) or math.log(group_size, 4) % 1:
        raise ValueError(f"ConvRot group size must be a power of four, got {group_size}")
    if in_features % group_size:
        raise ValueError(f"in_features={in_features} is not divisible by group size {group_size}")


def scan_checkpoint_quantization(checkpoint: str | Path) -> dict[str, QuantLayerInfo]:
    """Read ConvRot metadata without materializing the model weights."""
    from safetensors import safe_open

    path = Path(checkpoint)
    if path.is_dir():
        index = path / "model.safetensors.index.json"
        if index.is_file():
            names = json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()
            shards = [path / name for name in sorted(set(names))]
        else:
            shards = [path / "model.safetensors"]
    else:
        shards = [path]

    result: dict[str, QuantLayerInfo] = {}
    for shard in shards:
        if not shard.is_file():
            raise FileNotFoundError(f"Checkpoint shard not found: {shard}")
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            for key in sorted(k for k in keys if k.endswith(f".{QUANT_META_SUFFIX}")):
                raw = handle.get_tensor(key).numpy().tobytes()
                metadata = json.loads(raw)
                if metadata.get("format") != SUPPORTED_FORMAT or metadata.get("convrot") is not True:
                    raise RuntimeError(f"Unsupported quantization metadata at {shard}:{key}: {metadata!r}")
                prefix = key[: -len(f".{QUANT_META_SUFFIX}")]
                result[prefix] = QuantLayerInfo(
                    prefix=prefix,
                    group_size=int(metadata["convrot_groupsize"]),
                    in_features=int(metadata.get("in_features", 0)),
                    out_features=int(metadata.get("out_features", 0)),
                    has_bias=bool(metadata.get("has_bias", f"{prefix}.bias" in keys)),
                )
    if any(prefix.startswith("depth_decoder.") for prefix in result):
        raise RuntimeError("Hybrid checkpoint unexpectedly quantizes depth_decoder; refusing to use it")
    return result


class ConvRotInt8Linear(nn.Module):
    """Drop-in linear layer backed by comfy-kitchen's ConvRot INT8 kernel."""

    def __init__(self, in_features: int, out_features: int, bias: bool, group_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.convrot_groupsize = group_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.int8), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.empty(out_features, 1, dtype=torch.float32), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False) if bias else None
        self.quant_format = SUPPORTED_FORMAT

    def _load_from_state_dict(self, state_dict: dict[str, Any], prefix: str, *args: Any, **kwargs: Any) -> None:
        state_dict.pop(f"{prefix}{QUANT_META_SUFFIX}", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import comfy_kitchen

        return comfy_kitchen.int8_linear(
            x.contiguous(), self.weight, self.weight_scale, self.bias,
            out_dtype=x.dtype, convrot=True, convrot_groupsize=self.convrot_groupsize,
        )


def replace_quantized_linears(model: nn.Module, quant_map: dict[str, QuantLayerInfo]) -> list[str]:
    modules = dict(model.named_modules())
    replaced: list[str] = []
    for prefix, info in quant_map.items():
        target = modules.get(prefix)
        if target is None or not isinstance(target, nn.Linear):
            raise RuntimeError(f"Quantized layer {prefix!r} is missing or is not nn.Linear")
        if info.in_features and info.in_features != target.in_features:
            raise RuntimeError(f"{prefix}: in_features checkpoint/model mismatch")
        if info.out_features and info.out_features != target.out_features:
            raise RuntimeError(f"{prefix}: out_features checkpoint/model mismatch")
        _validate_group_size(info.group_size, target.in_features)
        parent_name, _, child_name = prefix.rpartition(".")
        parent = modules[parent_name] if parent_name else model
        setattr(parent, child_name, ConvRotInt8Linear(target.in_features, target.out_features, target.bias is not None, info.group_size))
        replaced.append(prefix)
    return replaced


def validate_cuda_backend() -> None:
    """Reject a missing CUDA comfy-kitchen backend before model loading."""
    import comfy_kitchen

    backends = comfy_kitchen.list_backends()
    available = [name for name, details in backends.items() if details.get("available")]
    cuda = [name for name in available if "cuda" in name.lower()]
    if not cuda:
        raise RuntimeError(f"No CUDA-capable comfy_kitchen backend is available: {backends!r}")
    logger.info("comfy_kitchen backends: %s", ", ".join(available))


def model_quantization_stats(model: nn.Module) -> dict[str, int]:
    quantized = [m for m in model.modules() if isinstance(m, ConvRotInt8Linear)]
    int8_params = sum(m.weight.numel() for m in quantized)
    int8_bytes = sum(m.weight.numel() * m.weight.element_size() for m in quantized)
    bf16_fp32_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
        if p.dtype in (torch.bfloat16, torch.float32)
    )
    meta = sum(p.numel() for p in model.parameters() if p.device.type == "meta")
    return {"quantized_modules": len(quantized), "int8_parameters": int8_params,
            "int8_bytes": int8_bytes, "bf16_fp32_bytes": bf16_fp32_bytes, "meta_parameters": meta}
