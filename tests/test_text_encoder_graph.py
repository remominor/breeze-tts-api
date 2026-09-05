from models.text_encoder_graph import TextEncoderGraphCache


def test_smallest_fitting_key_reuses_larger_token_bucket() -> None:
    keys = ((4, 64), (4, 96), (4, 128))

    assert TextEncoderGraphCache._smallest_fitting_key(
        keys, batch_size=4, token_length=32
    ) == (4, 64)


def test_smallest_fitting_key_does_not_cross_batch_sizes() -> None:
    keys = ((1, 256), (2, 256), (4, 128))

    assert (
        TextEncoderGraphCache._smallest_fitting_key(
            keys, batch_size=4, token_length=192
        )
        is None
    )
