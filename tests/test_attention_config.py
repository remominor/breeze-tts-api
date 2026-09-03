from types import SimpleNamespace

from models.breeze import _resolve_text_encoder_attn_implementation


def _config(*, caller=None, preferred=None):
    return SimpleNamespace(
        _attn_implementation=caller,
        text_encoder_config=SimpleNamespace(
            preferred_attn_implementation=preferred,
        ),
    )


def test_explicit_attention_choice_overrides_checkpoint_preference() -> None:
    assert _resolve_text_encoder_attn_implementation(
        _config(caller="eager", preferred="flash_attention_2")
    ) == "eager"


def test_checkpoint_attention_preference_and_legacy_default_are_preserved() -> None:
    assert _resolve_text_encoder_attn_implementation(
        _config(preferred="sdpa")
    ) == "sdpa"
    assert _resolve_text_encoder_attn_implementation(_config()) == "flash_attention_2"
