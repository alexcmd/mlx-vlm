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
