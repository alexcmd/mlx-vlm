"""Anthropic-endpoint request handling for coding-agent clients."""

import pytest

from mlx_vlm.server.anthropic import _anthropic_request_with_derived_fields
from mlx_vlm.server.schemas import AnthropicRequest


def _request(**kwargs) -> AnthropicRequest:
    base = {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    base.update(kwargs)
    return AnthropicRequest(**base)


def test_adaptive_thinking_defers_to_server_default(monkeypatch):
    monkeypatch.delenv("MLX_VLM_ADAPTIVE_THINKING", raising=False)
    req = _anthropic_request_with_derived_fields(_request(thinking={"type": "adaptive"}))
    assert req.enable_thinking is None


def test_adaptive_thinking_opt_in_enables(monkeypatch):
    monkeypatch.setenv("MLX_VLM_ADAPTIVE_THINKING", "1")
    req = _anthropic_request_with_derived_fields(_request(thinking={"type": "adaptive"}))
    assert req.enable_thinking is True


def test_enabled_and_disabled_thinking_unchanged(monkeypatch):
    monkeypatch.delenv("MLX_VLM_ADAPTIVE_THINKING", raising=False)
    assert _anthropic_request_with_derived_fields(_request(thinking={"type": "enabled"})).enable_thinking is True
    assert _anthropic_request_with_derived_fields(_request(thinking={"type": "disabled"})).enable_thinking is False


def test_explicit_enable_thinking_wins_over_adaptive(monkeypatch):
    monkeypatch.delenv("MLX_VLM_ADAPTIVE_THINKING", raising=False)
    req = _anthropic_request_with_derived_fields(_request(thinking={"type": "adaptive"}, enable_thinking=True))
    assert req.enable_thinking is True

def _slow_generator(server, delay):
    import time

    class FakeResponseGenerator:
        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            tokens = [
                server.StreamingToken(text="Hel", token=1, logprobs=0.0, finish_reason=None, cached_tokens=2),
                server.StreamingToken(text="lo", token=2, logprobs=0.0, finish_reason="stop", cached_tokens=2),
            ]

            def it():
                time.sleep(delay)  # the prefill
                yield from tokens

            return server.GenerationContext(uid=1, prompt_tokens=3), it()

    return FakeResponseGenerator()


def _stream_messages(monkeypatch, delay):
    from types import SimpleNamespace
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    import mlx_vlm.server as server

    monkeypatch.setattr(server.runtime, "response_generator", _slow_generator(server, delay))
    with (
        TestClient(server.app) as client,
        patch.object(server, "get_cached_model", return_value=(SimpleNamespace(), SimpleNamespace(), SimpleNamespace(model_type="qwen2_vl"))),
        patch.object(server, "apply_chat_template", return_value="prompt"),
    ):
        response = client.post(
            "/v1/messages",
            json={"model": "demo", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 4, "stream": True},
        )
    assert response.status_code == 200
    return [e for e in response.text.split("\n\n") if e.strip()]


def test_message_start_precedes_prefill_and_pings_fill_the_wait(monkeypatch):
    monkeypatch.delenv("MLX_VLM_EARLY_MESSAGE_START", raising=False)
    monkeypatch.setenv("MLX_VLM_SSE_PING_SECONDS", "0.05")
    events = _stream_messages(monkeypatch, delay=0.4)
    kinds = [e.split("\n", 1)[0] for e in events]
    assert kinds[0] == "event: message_start"
    assert kinds.count("event: message_start") == 1
    assert "event: ping" in kinds[1 : kinds.index("event: content_block_start")]
    delta = next(e for e in events if e.startswith("event: message_delta"))
    assert '"input_tokens": 1' in delta and '"cache_read_input_tokens": 2' in delta and '"output_tokens": 2' in delta
    assert kinds[-1] == "event: message_stop"


def test_early_message_start_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MLX_VLM_EARLY_MESSAGE_START", "0")
    monkeypatch.setenv("MLX_VLM_SSE_PING_SECONDS", "0.05")
    events = _stream_messages(monkeypatch, delay=0.3)
    kinds = [e.split("\n", 1)[0] for e in events]
    assert "event: ping" not in kinds
    assert kinds[0] == "event: message_start"
    start = next(e for e in events if e.startswith("event: message_start"))
    assert '"input_tokens": 1' in start
