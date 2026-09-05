from __future__ import annotations

import logging
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from breeze_infer.model_download import ensure_hybrid_assets
from models.breeze import BreezeForConditionalGeneration
from models.breeze_config import BreezeConfig

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_hybrid_quant_notice():
    """Hide Transformers' expected report for ConvRot metadata tensors only."""
    from transformers.utils import logging as transformers_logging

    # Accelerate emits this notice while dispatching the checkpoint; some
    # Transformers versions emit the equivalent message themselves.
    checkpoint_loggers = [
        logging.getLogger("accelerate.utils.modeling"),
        logging.getLogger("transformers.modeling_utils"),
    ]
    previous_verbosity = transformers_logging.get_verbosity()

    class _ExpectedQuantFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage()
            return not (
                "Some weights of the model checkpoint" in message
                and "were not used when" in message
                and ".comfy_quant" in message
            )

    quant_filter = _ExpectedQuantFilter()
    original_warnings = []
    for checkpoint_logger in checkpoint_loggers:
        original_warning = checkpoint_logger.warning

        def warning(message: object, *args: object, _original=original_warning, **kwargs: object) -> None:
            rendered = str(message)
            if args:
                try:
                    rendered = rendered % args
                except (TypeError, ValueError):
                    pass
            if (
                "Some weights of the model checkpoint" in rendered
                and "were not used when" in rendered
                and ".comfy_quant" in rendered
            ):
                return
            _original(message, *args, **kwargs)

        checkpoint_logger.addFilter(quant_filter)
        checkpoint_logger.warning = warning  # type: ignore[method-assign]
        original_warnings.append((checkpoint_logger, original_warning))
    # Transformers' logging setup can replace or bypass individual logger
    # filters.  The dispatch is a short, known-safe window, so also use its
    # official global verbosity control and restore it immediately afterwards.
    transformers_logging.set_verbosity_error()
    try:
        yield
    finally:
        transformers_logging.set_verbosity(previous_verbosity)
        for checkpoint_logger, original_warning in original_warnings:
            checkpoint_logger.warning = original_warning  # type: ignore[method-assign]
            checkpoint_logger.removeFilter(quant_filter)


def _load_breeze_tokenizer(ckpt_dir: Path) -> AutoTokenizer:
    try:
        return AutoTokenizer.from_pretrained(ckpt_dir, fix_mistral_regex=True)
    except TypeError as exc:
        # transformers 4.57.3 assumes tokenizers exposes a mutable pre-tokenizer
        # sequence, while tokenizers 0.22 exposes the single Split directly.
        if "does not support item assignment" not in str(exc):
            raise
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, fix_mistral_regex=False)
        import tokenizers

        tokenizer.backend_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Split(
            pattern=tokenizers.Regex(
                r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            ),
            behavior="isolated",
        )
        tokenizer.fix_mistral_regex = True
        return tokenizer


@contextmanager
def _suppress_accelerate_checkpoint_progress():
    """Keep the service's one-time checkpoint load out of its API logs."""
    import accelerate.utils.modeling as accelerate_modeling

    original = accelerate_modeling.is_tqdm_available
    accelerate_modeling.is_tqdm_available = lambda: False
    try:
        yield
    finally:
        accelerate_modeling.is_tqdm_available = original


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
        from accelerate import init_empty_weights, load_checkpoint_in_model

        from breeze_infer.int8_convrot import (
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
        # Accelerate otherwise maps every safetensors entry directly to CUDA
        # before noticing that the lean model has no matching parameter. Keep
        # those intentionally omitted tensors on CPU during staging so its
        # caching allocator does not retain their 1.2 GiB GPU allocation.
        # Do not include a catch-all CUDA mapping: Accelerate would then map
        # the omitted checkpoint entries to both CUDA and CPU while staging.
        # Unmapped entries fall back to CPU; every module retained by the lean
        # model is named here and is loaded directly to the target GPU.
        staging_device_map = {
            "backbone_model": device,
            "depth_decoder": device,
            "text_encoder": device,
            "text_encoder_proj": device,
            "lm_head": device,
            # Retain an explicit CPU entry so Accelerate takes its per-prefix
            # safetensors path instead of its one-device shortcut, which would
            # stage every checkpoint tensor on CUDA.
            "codec_model": "cpu",
        }
        with _suppress_hybrid_quant_notice(), _suppress_accelerate_checkpoint_progress():
            load_checkpoint_in_model(
                model, str(weights_path), device_map=staging_device_map, dtype=torch.bfloat16,
                strict=False, full_state_dict=True,
            )
        # ``load_checkpoint_in_model`` places checkpoint tensors but does not
        # dispatch non-persistent constructor buffers (RoPE frequencies and
        # audio token offsets). Move those small buffers with the model before
        # CUDA-graph warmup.
        model.to(device)
        # ConvRot scales are consumed as FP32 by comfy-kitchen.  The dispatch
        # dtype policy above intentionally materializes ordinary model weights
        # as BF16, so restore this quantization metadata dtype explicitly.
        for module in model.modules():
            if hasattr(module, "weight_scale"):
                module.weight_scale.data = module.weight_scale.data.float()
        stats = model_quantization_stats(model)
        if stats["meta_parameters"]:
            raise RuntimeError(f"INT8 model has {stats['meta_parameters']} parameters left on meta")
        logger.info("loaded ConvRot INT8 model: %s", stats)
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
