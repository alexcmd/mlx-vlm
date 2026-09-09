from queue import Queue

import pytest

from mlx_vlm.server.generation import (
    CorruptedGenerationError,
    StreamingToken,
    _TokenIterator,
    get_max_zero_token_run,
)


def _tok(token, finish_reason=None, token_count=1):
    return StreamingToken(text="!", token=token, logprobs=0.0, finish_reason=finish_reason, token_count=token_count)


def _iterator(items, max_zero_run):
    q = Queue()
    for item in items:
        q.put(item)
    q.put(None)
    cancelled = []
    it = _TokenIterator(q, uid=7, cancel_fn=cancelled.append, queue_timeout=1, max_zero_run=max_zero_run)
    return it, cancelled


def test_run_of_zero_tokens_aborts_and_cancels():
    it, cancelled = _iterator([_tok(5)] + [_tok(0)] * 8 + [_tok(6)], max_zero_run=8)
    out = []
    with pytest.raises(CorruptedGenerationError):
        for token in it:
            out.append(token.token)
    assert out == [5] + [0] * 7  # the eighth zero raises before being yielded
    assert cancelled == [7]
    assert list(it) == []  # ended


def test_short_zero_runs_pass_and_counter_resets():
    it, cancelled = _iterator([_tok(0)] * 7 + [_tok(3)] + [_tok(0)] * 7 + [_tok(1, finish_reason="stop")], max_zero_run=8)
    tokens = [t.token for t in it]
    assert tokens == [0] * 7 + [3] + [0] * 7 + [1]
    assert cancelled == []


def test_multi_token_chunks_count_by_token_count():
    it, cancelled = _iterator([_tok(0, token_count=4), _tok(0, token_count=4)], max_zero_run=8)
    with pytest.raises(CorruptedGenerationError):
        list(it)
    assert cancelled == [7]


def test_guard_can_be_disabled():
    it, cancelled = _iterator([_tok(0)] * 20 + [_tok(1, finish_reason="stop")], max_zero_run=0)
    assert len(list(it)) == 21
    assert cancelled == []


def test_threshold_from_env(monkeypatch):
    monkeypatch.setenv("MLX_VLM_MAX_ZERO_TOKEN_RUN", "3")
    assert get_max_zero_token_run() == 3
    monkeypatch.setenv("MLX_VLM_MAX_ZERO_TOKEN_RUN", "junk")
    assert get_max_zero_token_run() == 8
    monkeypatch.delenv("MLX_VLM_MAX_ZERO_TOKEN_RUN")
    assert get_max_zero_token_run() == 8
