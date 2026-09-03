from pathlib import Path

import pytest

from breeze_infer.model_download import (
    HYBRID_WEIGHTS_FILE,
    MIRROR_REPO_ID,
    MODEL_SIDECARS,
    ensure_hybrid_assets,
)


def test_complete_model_layout_does_not_download(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "models" / "Breeze-TTS-2"
    weights = model_dir.parent / HYBRID_WEIGHTS_FILE
    for filename in MODEL_SIDECARS:
        path = model_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ready")
    weights.write_bytes(b"ready")

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **_kwargs: pytest.fail("complete layout must not download"),
    )

    assert ensure_hybrid_assets(model_dir, weights) == weights.resolve()


def test_missing_default_assets_download_to_api_layout(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "models" / "Breeze-TTS-2"
    weights = model_dir.parent / HYBRID_WEIGHTS_FILE
    calls: list[dict[str, str]] = []

    def fake_download(*, repo_id: str, filename: str, local_dir: str, **_kwargs):
        calls.append({"repo_id": repo_id, "filename": filename, "local_dir": local_dir})
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"downloaded")
        return str(target)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    assert ensure_hybrid_assets(model_dir, weights) == weights.resolve()
    assert [call["filename"] for call in calls] == [*MODEL_SIDECARS, HYBRID_WEIGHTS_FILE]
    assert all(call["repo_id"] == MIRROR_REPO_ID for call in calls)
    assert all((model_dir / name).is_file() for name in MODEL_SIDECARS)
    assert weights.is_file()


def test_missing_custom_checkpoint_is_not_downloaded(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="custom checkpoint"):
        ensure_hybrid_assets(
            tmp_path / "Breeze-TTS-2", tmp_path / "my-custom-model.safetensors"
        )
