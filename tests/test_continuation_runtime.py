from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import models.continuation_streaming as continuation_module
from models.continuation_streaming import (
    ContinuationContextError,
    ContinuationRuntimeSession,
    ContinuationStreamingRuntime,
)
from models.fast_streaming import FastCfgSelection


class _Graph:
    def __init__(self, batch_size=2):
        self.batch_size = batch_size
        self._kv_indices = torch.arange(16)
        self._base_position = torch.zeros(batch_size, dtype=torch.long)
        self._pad_lens = torch.zeros(batch_size, dtype=torch.long)
        self.static_cache = object()
        self.calls = []
        self.prefill_len = None
        self.reset_count = 0

    def embed_tokens(self, frame):
        return torch.ones(frame.shape[0], frame.shape[1], 4)

    def model(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"])

    def lm_head(self, hidden):
        return torch.zeros(hidden.shape[0], 8)

    def finish_direct_prefill(self, value):
        self.prefill_len = value

    def reset(self):
        self.reset_count += 1


class _Codec:
    def __init__(self):
        self.opened = []
        self.closed = []

    def open_request(self, request_id, **kwargs):
        self.opened.append((request_id, kwargs))

    def close_request(self, request_id):
        self.closed.append(request_id)


class _Runtime:
    def __init__(self):
        self.config = SimpleNamespace(max_seq_len=16)
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.tokenizer = object()
        self.audio_tokenizer = object()
        self.model = SimpleNamespace(
            config=SimpleNamespace(num_codebooks=2, codebook_eos_token_id=7),
            device=torch.device("cpu"),
        )
        self._codec_value = _Codec()

    def _codec(self):
        return self._codec_value

    def _merge_branch(self, **_kwargs):
        return torch.full((1, 2, 4), 2.0), torch.ones(1, 2, dtype=torch.long)


def _session(graph: _Graph) -> ContinuationRuntimeSession:
    return ContinuationRuntimeSession(
        continuation_id="one",
        codec_request_id="continuation-one",
        graph=graph,
        cfg=FastCfgSelection("single_cfg", 4.0, False),
        branch_batch_size=2,
        token_history=torch.empty(16, dtype=torch.long),
        cache_length=4,
        next_position_ids=torch.tensor([3, 3]),
        pad_lens=torch.tensor([1, 1]),
    )


def test_append_encodes_only_new_text_and_preserves_cache_positions(monkeypatch) -> None:
    runtime = _Runtime()
    graph = _Graph()
    continuation = ContinuationStreamingRuntime(runtime, audio_eos=True)
    state = _session(graph)
    monkeypatch.setattr(
        continuation_module,
        "prepare_continuation_text_inputs",
        lambda *_args, **_kwargs: {
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "text_ids_mask": torch.ones(1, 2, dtype=torch.bool),
            "text_ids_len": torch.tensor([2]),
        },
    )

    hidden, logits, timing = continuation.append_text(
        state, "new text", estimated_audio_frames=1, context_safety_frames=1
    )

    call = graph.calls[0]
    assert call["cache_position"].tolist() == [4, 5, 6]
    assert call["position_ids"].tolist() == [[3, 4, 5], [3, 4, 5]]
    assert call["inputs_embeds"][:, 0].eq(1).all()  # Existing audio-EOS path.
    assert call["inputs_embeds"][:, 1:].eq(2).all()  # New text only.
    assert state.cache_length == 7
    assert state.next_position_ids.tolist() == [6, 6]
    assert graph.prefill_len == 7
    assert hidden.shape == (2, 3, 4)
    assert logits.shape == (1, 8)
    assert timing["new_text_tokens"] == 2


def test_append_rejects_context_before_writing_kv(monkeypatch) -> None:
    runtime = _Runtime()
    graph = _Graph()
    continuation = ContinuationStreamingRuntime(runtime, audio_eos=False)
    state = _session(graph)
    state.cache_length = 12
    monkeypatch.setattr(
        continuation_module,
        "prepare_continuation_text_inputs",
        lambda *_args, **_kwargs: {
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "text_ids_mask": torch.ones(1, 2, dtype=torch.bool),
            "text_ids_len": torch.tensor([2]),
        },
    )

    with pytest.raises(ContinuationContextError, match="context limit"):
        continuation.append_text(
            state, "new text", estimated_audio_frames=2, context_safety_frames=1
        )

    assert graph.calls == []
    assert state.cache_length == 12


def test_preflight_rejects_context_without_running_text_encoder(monkeypatch) -> None:
    runtime = _Runtime()
    graph = _Graph()
    continuation = ContinuationStreamingRuntime(runtime, audio_eos=True)
    state = _session(graph)
    state.cache_length = 11
    monkeypatch.setattr(
        continuation_module,
        "prepare_continuation_text_inputs",
        lambda *_args, **_kwargs: {
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "text_ids_mask": torch.ones(1, 2, dtype=torch.bool),
            "text_ids_len": torch.tensor([2]),
        },
    )

    with pytest.raises(ContinuationContextError, match="context limit"):
        continuation.preflight_continue(
            state, "new text", estimated_audio_frames=1, context_safety_frames=1
        )

    assert graph.calls == []


def test_repetition_penalty_history_restarts_for_each_text_segment(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.config.repetition_penalty = 1.1
    runtime.config.max_new_tokens = 8
    runtime.model.config.vocab_size = 8
    runtime.model.config.codebook_pad_token_id = 99
    runtime.model.generation_config = object()
    runtime.model.depth_decoder = SimpleNamespace(generation_config=object())
    runtime._depth_decoder_graph = SimpleNamespace(
        run=lambda *_args, **_kwargs: torch.tensor([[2]])
    )
    runtime._reserved_codec_token_ids = ()
    runtime.codec_chunk_frames = 1
    runtime._sampling_params = lambda _config: {
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "do_sample": False,
    }
    graph = _Graph()
    graph.run = lambda *_args, **_kwargs: (
        torch.ones(2, 1, 4),
        torch.zeros(1, 9),
    )
    continuation = ContinuationStreamingRuntime(runtime)
    state = _session(graph)
    state.generated_frames = 3
    state.token_history[:3] = torch.tensor([4, 5, 6])
    histories = []
    sampled = iter((torch.tensor(1), torch.tensor(8)))

    def sample(_logits, *, token_history, **_kwargs):
        histories.append(token_history.clone())
        return next(sampled)

    monkeypatch.setattr(continuation_module, "sample_logits", sample)
    monkeypatch.setattr(
        continuation,
        "_decode_frames",
        lambda *_args, **_kwargs: SimpleNamespace(audio=torch.zeros(1)),
    )
    monkeypatch.setattr(continuation, "_capture_rng", lambda _state: None)

    list(
        continuation._generate(
            state,
            torch.ones(2, 1, 4),
            torch.zeros(1, 9),
            cancel_event=None,
            base_timing={},
        )
    )

    assert histories[0].numel() == 0
    assert histories[1].tolist() == [1]


def test_codec_lifetime_spans_logical_requests_and_closes_once() -> None:
    runtime = _Runtime()
    graph = _Graph()
    continuation = ContinuationStreamingRuntime(runtime)
    state = _session(graph)

    continuation._open_codec(state)
    continuation._open_codec(state)
    continuation.close(state)
    continuation.close(state)

    assert runtime._codec_value.opened == [
        ("continuation-one", {"reset": True, "is_first_decode": True})
    ]
    assert runtime._codec_value.closed == ["continuation-one"]
    assert graph.reset_count == 1
