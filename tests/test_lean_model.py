from __future__ import annotations

from accelerate import init_empty_weights

from models.breeze import BreezeForConditionalGeneration
from models.breeze_config import BreezeConfig


def test_lean_model_omits_service_unused_checkpoint_modules() -> None:
    """The API loader must not materialize its replaced legacy components."""
    config = BreezeConfig.from_pretrained("models/Breeze-TTS-2")
    config._omit_service_unused_modules = True
    with init_empty_weights():
        model = BreezeForConditionalGeneration(config)

    state_keys = set(model.state_dict())
    assert model.text_encoder is not None
    assert model.codec_model is None
    assert model.embed_text_tokens is None
    assert not any(key.startswith("codec_model.") for key in state_keys)
    assert "embed_text_tokens.weight" not in state_keys
