"""Fetch the hybrid Breeze serving assets when they are not mounted locally."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MIRROR_REPO_ID = "drbaph/Breeze-TTS-2-comfyui"
HYBRID_WEIGHTS_FILE = "Breeze-TTS-2-int8-hybrid.safetensors"
MODEL_SIDECARS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "audio_tokenizer/config.json",
    "audio_tokenizer/model.safetensors",
    "audio_tokenizer/preprocessor_config.json",
)


def ensure_hybrid_assets(model_dir: Path, weights_path: Path) -> Path:
    """Download missing hybrid serving files from the ComfyUI mirror.

    The API intentionally keeps the hybrid checkpoint beside the model folder,
    while the mirror stores it with the sidecar files. Individual downloads
    preserve the API's existing ``models/Breeze-TTS-2`` layout.
    """
    model_dir = model_dir.resolve()
    weights_path = weights_path.resolve()
    missing_sidecars = [
        filename for filename in MODEL_SIDECARS if not (model_dir / filename).is_file()
    ]
    missing_weights = not weights_path.is_file()
    if not missing_sidecars and not missing_weights:
        return weights_path

    if weights_path.name != HYBRID_WEIGHTS_FILE:
        missing = missing_sidecars + ([weights_path.name] if missing_weights else [])
        raise FileNotFoundError(
            "Missing model files for a custom checkpoint: " + ", ".join(missing)
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download missing Breeze model files"
        ) from exc

    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    for filename in missing_sidecars:
        logger.info("Downloading Breeze model sidecar %s from %s", filename, MIRROR_REPO_ID)
        hf_hub_download(
            repo_id=MIRROR_REPO_ID,
            filename=filename,
            local_dir=str(model_dir),
        )
    if missing_weights:
        logger.info("Downloading Breeze hybrid checkpoint from %s", MIRROR_REPO_ID)
        hf_hub_download(
            repo_id=MIRROR_REPO_ID,
            filename=HYBRID_WEIGHTS_FILE,
            local_dir=str(weights_path.parent),
        )

    still_missing = [
        filename for filename in MODEL_SIDECARS if not (model_dir / filename).is_file()
    ]
    if not weights_path.is_file():
        still_missing.append(str(weights_path))
    if still_missing:
        raise RuntimeError(
            "Breeze mirror download completed without required files: "
            + ", ".join(still_missing)
        )
    return weights_path
