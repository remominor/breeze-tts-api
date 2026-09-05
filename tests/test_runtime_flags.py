from __future__ import annotations

from unittest.mock import patch

from breeze_infer.runtime import _load_breeze_tokenizer
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
