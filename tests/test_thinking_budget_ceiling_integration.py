# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration tests for Layer 2 — thinking-budget ceiling."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from vllm_mlx import server as srv


def _chat_factory(captured_kwargs_list):
    """engine.chat coroutine stand-in — returns a real GenerationOutput."""
    from vllm_mlx.engine.base import GenerationOutput

    async def _chat(**kwargs):
        captured_kwargs_list.append(kwargs)
        return GenerationOutput(
            text="hi",
            tokens=[1, 2, 3],
            prompt_tokens=10,
            completion_tokens=2,
            finish_reason="stop",
            new_text="hi",
            finished=True,
            thinking_budget_applied=True,
            thinking_budget_noop_reason=None,
        )

    return _chat


@pytest.fixture
def qwen3_fake_server_ceiling(monkeypatch):
    """Layer 1 DISABLED so these tests isolate Layer 2 behavior."""
    fake = MagicMock()
    fake.preserve_native_tool_format = False
    fake.model_name = "mlx-community/Qwen3.6-35B-A3B-4bit"
    fake.tokenizer = None
    fake._is_mllm = False
    fake.is_mllm = False
    fake.__class__ = __import__(
        "vllm_mlx.engine.batched", fromlist=["BatchedEngine"]
    ).BatchedEngine
    fake._reasoning_parser = MagicMock()
    fake._reasoning_parser.start_token = "<think>"
    fake._reasoning_parser.end_tokens = ["</think>"]
    fake._reasoning_parser.channel_strip_prefix = None
    fake._reasoning_parser.extract_reasoning = MagicMock(
        side_effect=lambda text: ("", text)
    )
    fake._reasoning_parser.tokenizer = None

    captured: list[dict] = []
    fake.chat = _chat_factory(captured)

    monkeypatch.setattr(srv, "_engine", fake)
    monkeypatch.setattr(srv, "_model_name", fake.model_name)
    monkeypatch.setattr(srv, "_reasoning_parser", fake._reasoning_parser)
    monkeypatch.setattr(srv, "_reasoning_parser_name", "qwen3")
    monkeypatch.setattr(srv, "_disable_qwen3_first_turn_no_think", True)
    monkeypatch.setattr(srv, "_max_thinking_token_budget", None)

    return fake, captured


def test_anthropic_effort_high_clamps_to_ceiling(
    qwen3_fake_server_ceiling, monkeypatch
):
    fake, captured = qwen3_fake_server_ceiling
    monkeypatch.setattr(srv, "_max_thinking_token_budget", 2048)

    client = TestClient(srv.app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "high"},
            "max_tokens": 8192,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-thinking-budget-ceiling") == "2048"
    assert resp.headers.get("x-thinking-budget-clamped-to") == "2048"
    assert resp.headers.get("x-thinking-budget-resolved") == "2048"
    assert captured[0].get("thinking_token_budget") == 2048


def test_openai_reasoning_effort_high_also_clamps(
    qwen3_fake_server_ceiling, monkeypatch
):
    fake, captured = qwen3_fake_server_ceiling
    monkeypatch.setattr(srv, "_max_thinking_token_budget", 2048)

    client = TestClient(srv.app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
            "max_tokens": 8192,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-thinking-budget-ceiling") == "2048"
    assert resp.headers.get("x-thinking-budget-clamped-to") == "2048"


def test_chat_template_kwargs_budget_also_clamped(
    qwen3_fake_server_ceiling, monkeypatch
):
    """Documented extension field must be clamped — otherwise it's a bypass."""
    fake, captured = qwen3_fake_server_ceiling
    monkeypatch.setattr(srv, "_max_thinking_token_budget", 2048)

    client = TestClient(srv.app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"thinking_token_budget": 8192},
            "max_tokens": 16384,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert captured[0].get("thinking_token_budget") == 2048


def test_no_clamp_when_resolved_below_ceiling(qwen3_fake_server_ceiling, monkeypatch):
    fake, captured = qwen3_fake_server_ceiling
    monkeypatch.setattr(srv, "_max_thinking_token_budget", 8192)

    client = TestClient(srv.app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "medium"},
            "max_tokens": 4096,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-thinking-budget-ceiling") == "8192"
    assert "x-thinking-budget-clamped-to" not in resp.headers
    assert resp.headers.get("x-thinking-budget-resolved") == "2048"


def test_thinking_disabled_budget_zero_never_raised(
    qwen3_fake_server_ceiling, monkeypatch
):
    """Client sent thinking.type=disabled → budget=0 → ceiling MUST NOT raise."""
    fake, captured = qwen3_fake_server_ceiling
    monkeypatch.setattr(srv, "_max_thinking_token_budget", 2048)

    client = TestClient(srv.app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "disabled"},
            "max_tokens": 200,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-thinking-budget-resolved") == "0"
    assert "x-thinking-budget-clamped-to" not in resp.headers
    assert captured[0].get("thinking_token_budget") == 0


def test_mllm_engine_skips_clamp_with_truthful_header(
    qwen3_fake_server_ceiling, monkeypatch
):
    """MLLM engine can't enforce the logits processor → header tells truth."""
    fake, captured = qwen3_fake_server_ceiling
    # Flip to MLLM so _engine_supports_thinking_budget_processor returns False.
    fake._is_mllm = True
    monkeypatch.setattr(srv, "_max_thinking_token_budget", 2048)

    client = TestClient(srv.app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "high"},
            "max_tokens": 8192,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-thinking-budget-ceiling") == "2048"
    assert resp.headers.get("x-thinking-budget-clamp-skipped") == "engine-no-op"
    assert "x-thinking-budget-clamped-to" not in resp.headers


def test_metric_increments_on_clamp(qwen3_fake_server_ceiling, monkeypatch):
    fake, captured = qwen3_fake_server_ceiling
    monkeypatch.setattr(srv, "_max_thinking_token_budget", 2048)

    from vllm_mlx.metrics import thinking_budget_clamp_fired_total

    before = thinking_budget_clamp_fired_total.value
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "high"},
            "max_tokens": 8192,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert thinking_budget_clamp_fired_total.value == before + 1
