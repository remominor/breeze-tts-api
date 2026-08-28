"""Persistent Breeze voice profiles and reusable reference-code cache."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

PROFILE_AUDIO = "ref_audio.wav"
PROFILE_META = "meta.json"
PROFILE_CODES = "ref_codes.safetensors"
_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ProfileNotFoundError(Exception):
    pass


class ProfileExistsError(Exception):
    pass


class ProfileStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, profile_id: str) -> Path:
        if not _ID.fullmatch(profile_id):
            raise ValueError(f"Invalid profile_id: '{profile_id}'")
        return self.root / profile_id

    def list(self) -> list[dict[str, Any]]:
        result = []
        for path in sorted(self.root.iterdir()):
            if path.is_dir():
                meta = self._read_meta(path)
                if meta and (path / PROFILE_AUDIO).exists():
                    result.append({"profile_id": path.name, **meta, "cached": (path / PROFILE_CODES).exists()})
        return result

    def _read_meta(self, path: Path) -> dict[str, Any] | None:
        try:
            return json.loads((path / PROFILE_META).read_text())
        except (OSError, ValueError):
            return None

    def get(self, profile_id: str) -> dict[str, Any]:
        path = self._path(profile_id)
        meta = self._read_meta(path)
        if not meta or not (path / PROFILE_AUDIO).exists():
            raise ProfileNotFoundError(profile_id)
        return {"profile_id": profile_id, **meta, "cached": (path / PROFILE_CODES).exists()}

    def audio_path(self, profile_id: str) -> Path:
        self.get(profile_id)
        return self._path(profile_id) / PROFILE_AUDIO

    def codes_path(self, profile_id: str) -> Path:
        return self._path(profile_id) / PROFILE_CODES

    def load_codes(self, profile_id: str) -> torch.Tensor | None:
        path = self.codes_path(profile_id)
        if not path.exists():
            return None
        try:
            return load_file(str(path), device="cpu")["audio_codes"].to(torch.int16).contiguous()
        except (KeyError, OSError, RuntimeError, SafetensorError, ValueError):
            path.unlink(missing_ok=True)
            return None

    def save_codes(self, profile_id: str, codes: torch.Tensor) -> None:
        path = self.codes_path(profile_id)
        temporary = path.with_suffix(".tmp")
        save_file({"audio_codes": codes.detach().to(torch.int16).cpu().contiguous()}, str(temporary))
        os.replace(temporary, path)

    def save(self, profile_id: str, audio: bytes, ref_text: str, *, name: str | None = None, overwrite: bool = False) -> dict[str, Any]:
        path = self._path(profile_id)
        if path.exists() and not overwrite:
            raise ProfileExistsError(profile_id)
        new = not path.exists()
        path.mkdir(parents=True, exist_ok=True)
        try:
            self._atomic(path / PROFILE_AUDIO, audio)
            meta = {"name": (name or profile_id).strip(), "ref_text": ref_text.strip(), "created_at": datetime.now(timezone.utc).isoformat()}
            self._atomic(path / PROFILE_META, json.dumps(meta, ensure_ascii=False, indent=2).encode())
            if overwrite:
                (path / PROFILE_CODES).unlink(missing_ok=True)
            return {"profile_id": profile_id, **meta, "cached": False}
        except Exception:
            if new:
                shutil.rmtree(path, ignore_errors=True)
            raise

    def update(self, profile_id: str, *, audio: bytes | None = None, ref_text: str | None = None, name: str | None = None) -> dict[str, Any]:
        current = self.get(profile_id)
        path = self._path(profile_id)
        if audio is not None:
            self._atomic(path / PROFILE_AUDIO, audio)
            (path / PROFILE_CODES).unlink(missing_ok=True)
        if ref_text is not None:
            current["ref_text"] = ref_text.strip()
        if name is not None:
            current["name"] = name.strip()
        meta = {key: current.get(key) for key in ("name", "ref_text", "created_at")}
        self._atomic(path / PROFILE_META, json.dumps(meta, ensure_ascii=False, indent=2).encode())
        return {"profile_id": profile_id, **meta, "cached": (path / PROFILE_CODES).exists()}

    def delete(self, profile_id: str) -> None:
        path = self._path(profile_id)
        self.get(profile_id)
        shutil.rmtree(path)

    def resolve(self, identifier: str) -> str:
        try:
            self.get(identifier)
            return identifier
        except (ProfileNotFoundError, ValueError):
            matches = [item["profile_id"] for item in self.list() if item.get("name") == identifier]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(f"Voice name '{identifier}' is ambiguous")
            raise ProfileNotFoundError(identifier)

    @staticmethod
    def _atomic(path: Path, data: bytes) -> None:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
