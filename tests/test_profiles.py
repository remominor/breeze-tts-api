from __future__ import annotations

import torch

from breeze_infer.profiles import ProfileStore


def test_profile_store_round_trip_and_code_cache(tmp_path):
    store = ProfileStore(tmp_path)
    created = store.save("alice", b"RIFF audio", "hello world", name="Alice")
    assert created["profile_id"] == "alice"
    assert store.resolve("Alice") == "alice"
    assert store.load_codes("alice") is None
    store.save_codes("alice", torch.ones((3, 16), dtype=torch.int16))
    assert store.load_codes("alice").shape == (3, 16)
    assert store.get("alice")["cached"] is True


def test_profile_audio_update_invalidates_codes(tmp_path):
    store = ProfileStore(tmp_path)
    store.save("alice", b"audio", "hello")
    store.save_codes("alice", torch.zeros((2, 16), dtype=torch.int16))
    store.update("alice", audio=b"new audio")
    assert store.load_codes("alice") is None
