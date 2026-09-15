from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from accelerate import init_empty_weights
from safetensors.torch import save_file

from breeze_infer.runtime import (
    _hybrid_target_dtype,
    _hybrid_tensor_value,
    _load_breeze_tokenizer,
    _load_hybrid_checkpoint,
)
from models.fast_streaming import FastBreezeStreamingRuntime


def test_runtime_fast_properties_return_values_not_methods() -> None:
    runtime = FastBreezeStreamingRuntime.__new__(FastBreezeStreamingRuntime)
    runtime._fast_text_encoder = False
    runtime._fast_backbone_prefill = False
    runtime._fast_backbone_decode = False
    runtime._fast_depth_decoder = False
    runtime._fast_codec = False
    runtime._codec_chunk_frames = 2

    assert runtime.fast_enabled is False
    assert runtime.codec_chunk_frames == 2

    runtime._fast_codec = True
    runtime._codec_chunk_frames = 1

    assert runtime.fast_enabled is True
    assert runtime.codec_chunk_frames == 1


def test_breeze_tokenizer_disables_inapplicable_mistral_regex_fix(tmp_path) -> None:
    tokenizer = object()
    with patch(
        "breeze_infer.runtime.AutoTokenizer.from_pretrained", return_value=tokenizer
    ) as load_tokenizer:
        assert _load_breeze_tokenizer(tmp_path) is tokenizer

    load_tokenizer.assert_called_once_with(tmp_path, fix_mistral_regex=False)


class _HybridFixture(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_codebooks_embeddings=False)
        self.ordinary = torch.nn.Linear(2, 2, bias=False)
        self.lm_head = torch.nn.Linear(2, 2, bias=False)
        self.depth_decoder = torch.nn.Module()
        self.depth_decoder.codebooks_head = torch.nn.Linear(2, 2, bias=False)
        self.quant = torch.nn.Module()
        self.quant.weight = torch.nn.Parameter(
            torch.empty(2, 2, dtype=torch.int8), requires_grad=False
        )
        self.quant.weight_scale = torch.nn.Parameter(
            torch.empty(2, 1, dtype=torch.float32), requires_grad=False
        )


class _TiedHybridFixture(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_codebooks_embeddings=True)
        self.backbone_model = torch.nn.Module()
        self.backbone_model.embed_tokens = torch.nn.Module()
        self.backbone_model.embed_tokens.embed_audio_tokens = torch.nn.Embedding(2, 2)
        self.depth_decoder = torch.nn.Module()
        self.depth_decoder.model = torch.nn.Module()
        self.depth_decoder.model.embed_tokens = torch.nn.Embedding(2, 2)

    def tie_weights(self) -> None:
        self.backbone_model.embed_tokens.embed_audio_tokens.weight = (
            self.depth_decoder.model.embed_tokens.weight
        )


def _meta_hybrid_fixture() -> _HybridFixture:
    with init_empty_weights():
        return _HybridFixture()


def test_hybrid_checkpoint_streaming_uses_bf16_compatible_scale_values(tmp_path) -> None:
    checkpoint = tmp_path / "hybrid.safetensors"
    scale = torch.tensor([[0.001234567], [0.009876543]], dtype=torch.float32)
    tensors = {
        "ordinary.weight": torch.tensor(
            [[1.0001, 2.0001], [3.0001, 4.0001]], dtype=torch.float32
        ),
        "lm_head.weight": torch.arange(4, dtype=torch.bfloat16).view(2, 2),
        "depth_decoder.codebooks_head.weight": torch.arange(
            4, dtype=torch.bfloat16
        ).view(2, 2),
        "quant.weight": torch.arange(4, dtype=torch.int8).view(2, 2),
        "quant.weight_scale": scale,
        "codec_model.ignored": torch.ones(1),
        "embed_text_tokens.weight": torch.ones(1),
        "quant.comfy_quant": torch.tensor([123, 125], dtype=torch.uint8),
    }
    save_file(tensors, checkpoint)

    model = _meta_hybrid_fixture()
    stats = _load_hybrid_checkpoint(model, checkpoint, device="cpu")

    assert not torch.equal(scale, scale.to(torch.bfloat16).float())
    assert model.ordinary.weight.dtype == torch.bfloat16
    assert torch.equal(model.ordinary.weight, tensors["ordinary.weight"].to(torch.bfloat16))
    assert model.lm_head.weight.dtype == torch.float32
    assert torch.equal(model.lm_head.weight, tensors["lm_head.weight"].float())
    assert model.depth_decoder.codebooks_head.weight.dtype == torch.float32
    assert torch.equal(
        model.depth_decoder.codebooks_head.weight,
        tensors["depth_decoder.codebooks_head.weight"].float(),
    )
    assert model.quant.weight.dtype == torch.int8
    assert model.quant.weight_scale.dtype == torch.float32
    assert torch.equal(model.quant.weight_scale, scale.to(torch.bfloat16).float())
    assert stats == {
        "loaded_tensors": 5,
        "ignored_tensors": 2,
        "quant_metadata": 1,
        "scale_mode": "bf16_compat",
    }


def test_hybrid_checkpoint_exact_scale_mode_preserves_scale_values(tmp_path) -> None:
    checkpoint = tmp_path / "hybrid.safetensors"
    scale = torch.tensor([[0.001234567], [0.009876543]], dtype=torch.float32)
    save_file(
        {
            "ordinary.weight": torch.ones(2, 2),
            "lm_head.weight": torch.ones(2, 2),
            "depth_decoder.codebooks_head.weight": torch.ones(2, 2),
            "quant.weight": torch.ones(2, 2, dtype=torch.int8),
            "quant.weight_scale": scale,
        },
        checkpoint,
    )

    model = _meta_hybrid_fixture()
    _load_hybrid_checkpoint(
        model, checkpoint, device="cpu", scale_mode="exact_fp32"
    )

    assert torch.equal(model.quant.weight_scale, scale)


def test_hybrid_scale_value_policy_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown hybrid scale mode"):
        _hybrid_tensor_value(
            "quant.weight_scale", torch.ones(1), scale_mode="unsupported"
        )


def test_hybrid_dtype_policy_rejects_non_fp32_convrot_scale() -> None:
    with pytest.raises(RuntimeError, match="must be FP32"):
        _hybrid_target_dtype("quant.weight_scale", torch.ones(2, dtype=torch.bfloat16))


def test_hybrid_checkpoint_streaming_rejects_unknown_tensor(tmp_path) -> None:
    checkpoint = tmp_path / "unknown.safetensors"
    save_file({"unknown.weight": torch.ones(1)}, checkpoint)

    with pytest.raises(RuntimeError, match="Unexpected tensor"):
        _load_hybrid_checkpoint(_meta_hybrid_fixture(), checkpoint, device="cpu")


def test_hybrid_checkpoint_streaming_rejects_missing_tensor(tmp_path) -> None:
    checkpoint = tmp_path / "missing.safetensors"
    save_file({"ordinary.weight": torch.ones(2, 2)}, checkpoint)

    with pytest.raises(RuntimeError, match="is missing"):
        _load_hybrid_checkpoint(_meta_hybrid_fixture(), checkpoint, device="cpu")


def test_hybrid_checkpoint_streaming_reties_omitted_audio_embedding(tmp_path) -> None:
    checkpoint = tmp_path / "tied.safetensors"
    expected = torch.arange(4, dtype=torch.float32).view(2, 2)
    save_file({"depth_decoder.model.embed_tokens.weight": expected}, checkpoint)
    with init_empty_weights():
        model = _TiedHybridFixture()
    model.tie_weights()

    _load_hybrid_checkpoint(model, checkpoint, device="cpu")

    assert model.backbone_model.embed_tokens.embed_audio_tokens.weight is (
        model.depth_decoder.model.embed_tokens.weight
    )
    assert torch.equal(model.depth_decoder.model.embed_tokens.weight, expected.bfloat16())
