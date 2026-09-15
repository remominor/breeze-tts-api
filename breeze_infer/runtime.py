from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from breeze_infer.model_download import ensure_hybrid_assets
from models.breeze import BreezeForConditionalGeneration
from models.breeze_config import BreezeConfig

logger = logging.getLogger(__name__)

HYBRID_FP32_TENSORS = frozenset(
    {
        "lm_head.weight",
        "depth_decoder.codebooks_head.weight",
    }
)
HYBRID_OMITTED_PREFIXES = ("codec_model.", "embed_text_tokens.")
HYBRID_SCALE_MODES = frozenset({"bf16_compat", "exact_fp32"})


def _validate_hybrid_scale_mode(scale_mode: str) -> str:
    if scale_mode not in HYBRID_SCALE_MODES:
        choices = ", ".join(sorted(HYBRID_SCALE_MODES))
        raise ValueError(f"Unknown hybrid scale mode {scale_mode!r}; expected one of {choices}")
    return scale_mode


def _hybrid_target_dtype(name: str, tensor: torch.Tensor) -> torch.dtype:
    """Return the on-device dtype for one hybrid-checkpoint tensor."""
    if name.endswith(".weight_scale"):
        if tensor.dtype != torch.float32:
            raise RuntimeError(
                f"ConvRot scale {name} must be FP32 in the checkpoint, got {tensor.dtype}"
            )
        return tensor.dtype
    if name in HYBRID_FP32_TENSORS:
        return torch.float32
    if tensor.is_floating_point():
        return torch.bfloat16
    return tensor.dtype


def _hybrid_tensor_value(
    name: str, tensor: torch.Tensor, *, scale_mode: str
) -> torch.Tensor:
    """Apply the checkpoint-specific ConvRot scale compatibility policy."""
    scale_mode = _validate_hybrid_scale_mode(scale_mode)
    if name.endswith(".weight_scale") and scale_mode == "bf16_compat":
        # Legacy hybrid loading first converted all floating checkpoint tensors
        # to BF16, then promoted ConvRot scales back to FP32 for the kernel.
        # Retain that numerically stable trajectory without changing the FP32
        # scale storage required by comfy_kitchen.int8_linear.
        return tensor.to(torch.bfloat16).float()
    return tensor


def _load_hybrid_checkpoint(
    model: torch.nn.Module,
    checkpoint: Path,
    *,
    device: str,
    scale_mode: str = "bf16_compat",
) -> dict[str, int | str]:
    """Stream a hybrid safetensors checkpoint into an already-built meta model."""
    from accelerate.utils.modeling import set_module_tensor_to_device
    from safetensors import safe_open

    scale_mode = _validate_hybrid_scale_mode(scale_mode)
    model_keys = set(model.state_dict().keys())
    loaded: set[str] = set()
    ignored = 0
    quant_metadata = 0

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for name in handle.keys():  # noqa: SIM118 - safe_open is not iterable
            if name.endswith(".comfy_quant"):
                quant_metadata += 1
                continue
            if name.startswith(HYBRID_OMITTED_PREFIXES):
                ignored += 1
                continue
            if name not in model_keys:
                raise RuntimeError(f"Unexpected tensor in hybrid checkpoint: {name}")

            tensor = handle.get_tensor(name)
            value = _hybrid_tensor_value(name, tensor, scale_mode=scale_mode)
            set_module_tensor_to_device(
                model,
                name,
                device=device,
                value=value.contiguous(),
                dtype=_hybrid_target_dtype(name, tensor),
            )
            loaded.add(name)

    tied_audio_embedding = "backbone_model.embed_tokens.embed_audio_tokens.weight"
    if (
        getattr(model.config, "tie_codebooks_embeddings", False)
        and tied_audio_embedding in model_keys
        and tied_audio_embedding not in loaded
    ):
        model.tie_weights()
        loaded.add(tied_audio_embedding)

    missing = sorted(model_keys - loaded)
    if missing:
        raise RuntimeError(
            f"Hybrid checkpoint is missing {len(missing)} model tensor(s): {missing[:8]}"
        )

    return {
        "loaded_tensors": len(loaded),
        "ignored_tensors": ignored,
        "quant_metadata": quant_metadata,
        "scale_mode": scale_mode,
    }


def _load_breeze_tokenizer(ckpt_dir: Path) -> AutoTokenizer:
    # Breeze uses GemmaTokenizerFast, not a Mistral tokenizer.  Transformers'
    # Mistral-only regex migration is therefore inapplicable and, with the
    # tokenizers version in our supported stack, can raise a TypeError.
    return AutoTokenizer.from_pretrained(ckpt_dir, fix_mistral_regex=False)


def get_dist_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def resolve_device(explicit_device: str | None = None) -> str:
    if explicit_device:
        return explicit_device

    _, _, local_rank = get_dist_info()
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return "cpu"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_generation_config_for_breeze(
    model: torch.nn.Module,
    generation_config: dict[str, Any] | None = None,
) -> None:
    generation_config = generation_config or {
        "depth_decoder_do_sample": True,
        "depth_decoder_temperature": 0.9,
        "depth_decoder_top_p": 1.0,
        "depth_decoder_top_k": 50,
        "do_sample": True,
        "top_p": 1.0,
        "top_k": 50,
        "max_new_tokens": 750,
        "temperature": 0.9,
    }

    prefix = "depth_decoder_"
    depth_decoder_attrs = {
        attr[len(prefix) :]: value
        for attr, value in generation_config.items()
        if attr.startswith(prefix)
    }
    vars(model.depth_decoder.generation_config).update(
        {"_from_model_config": False, **depth_decoder_attrs}
    )
    vars(model.generation_config).update(generation_config)


def load_runtime(
    ckpt_dir: Path,
    *,
    device: str,
    attn_implementation: str,
    weights_path: Path | None = None,
    hybrid_scale_mode: str = "bf16_compat",
) -> tuple[AutoTokenizer, BreezeForConditionalGeneration, Any]:

    if weights_path is not None:
        weights_path = ensure_hybrid_assets(ckpt_dir, weights_path)

    if device.startswith("cuda"):
        try:
            torch.cuda.set_device(device)
        except Exception as exc:
            rank, world_size, local_rank = get_dist_info()
            raise RuntimeError(
                "Failed to set CUDA device "
                f"device={device} rank={rank} world_size={world_size} local_rank={local_rank} "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                f"device_count={torch.cuda.device_count()}"
            ) from exc
    tokenizer = _load_breeze_tokenizer(ckpt_dir)
    if weights_path is None:
        config = BreezeConfig.from_pretrained(ckpt_dir)
        config._omit_service_unused_modules = True
        model = BreezeForConditionalGeneration.from_pretrained(
            ckpt_dir,
            config=config,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        model.to(device)
    else:
        from accelerate import init_empty_weights

        from breeze_infer.int8_convrot import (
            ConvRotInt8Linear,
            model_quantization_stats,
            replace_quantized_linears,
            scan_checkpoint_quantization,
            validate_cuda_backend,
        )

        if not weights_path.is_file():
            raise FileNotFoundError(f"INT8 weights not found: {weights_path}")
        if not device.startswith("cuda"):
            raise RuntimeError("ConvRot INT8 Breeze weights require a CUDA device")
        validate_cuda_backend()
        quant_map = scan_checkpoint_quantization(weights_path)
        if not quant_map:
            raise RuntimeError(f"Weights file has no ConvRot quantization metadata: {weights_path}")
        config = BreezeConfig.from_pretrained(ckpt_dir)
        config._omit_service_unused_modules = True
        with init_empty_weights():
            model = BreezeForConditionalGeneration(config)
        # The checkpoint omits the tied audio embedding duplicate.  Establish
        # the tie while parameters are still meta so dispatch sees no missing
        # storage to materialize.
        model.tie_weights()
        replaced = replace_quantized_linears(model, quant_map)
        if len(replaced) != len(quant_map):
            raise RuntimeError(f"Only {len(replaced)}/{len(quant_map)} quantized prefixes matched")
        load_stats = _load_hybrid_checkpoint(
            model,
            weights_path,
            device=device,
            scale_mode=hybrid_scale_mode,
        )
        # Streamed checkpoint placement does not dispatch non-persistent
        # constructor buffers (RoPE frequencies and audio token offsets). Move
        # those small buffers with the model before CUDA-graph warmup.
        model.to(device)
        if model.lm_head.weight.dtype != torch.float32:
            raise RuntimeError(
                f"lm_head.weight loaded as {model.lm_head.weight.dtype}, expected FP32"
            )
        if model.depth_decoder.codebooks_head.weight.dtype != torch.float32:
            raise RuntimeError(
                "depth_decoder.codebooks_head.weight loaded as "
                f"{model.depth_decoder.codebooks_head.weight.dtype}, expected FP32"
            )
        for name, module in model.named_modules():
            if not isinstance(module, ConvRotInt8Linear):
                continue
            if module.weight.dtype != torch.int8:
                raise RuntimeError(
                    f"{name}.weight loaded as {module.weight.dtype}, expected INT8"
                )
            if module.weight_scale.dtype != torch.float32:
                raise RuntimeError(
                    f"{name}.weight_scale loaded as {module.weight_scale.dtype}, expected FP32"
                )
        stats = model_quantization_stats(model)
        if stats["meta_parameters"]:
            raise RuntimeError(f"INT8 model has {stats['meta_parameters']} parameters left on meta")
        logger.info("loaded ConvRot INT8 model: load=%s model=%s", load_stats, stats)
    model.eval()

    from qwen_tts import Qwen3TTSTokenizer

    bundled_audio_tokenizer = ckpt_dir / "audio_tokenizer"
    if not bundled_audio_tokenizer.is_dir():
        raise FileNotFoundError(
            "Bundled audio tokenizer not found at "
            f"{bundled_audio_tokenizer}. The Breeze model package must include "
            "the audio_tokenizer directory."
        )
    audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
        str(bundled_audio_tokenizer), device_map=device
    )
    return tokenizer, model, audio_tokenizer
