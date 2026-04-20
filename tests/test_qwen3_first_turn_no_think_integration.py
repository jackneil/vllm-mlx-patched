# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration tests for Layer 1 — Qwen3 first-turn auto-no-think."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from vllm_mlx import server as srv


class _FakeOutput:
    def __init__(self, new_text="hi"):
        self.new_text = new_text
        self.text = new_text
        self.prompt_tokens = 10
        self.completion_tokens = 2
        self.finished = True
        self.finish_reason = "stop"
        self.thinking_budget_applied = False
        self.thinking_budget_noop_reason = None


def _finite_stream_chat_factory(captured_kwargs_list):
    def _stream_chat(**kwargs):
        captured_kwargs_list.append(kwargs)

        async def _gen():
            yield _FakeOutput()

        return _gen()

    return _stream_chat


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
            thinking_budget_applied=None,
            thinking_budget_noop_reason=None,
        )

    return _chat


@pytest.fixture
def qwen3_fake_server(monkeypatch):
    """Install a fake Qwen3 engine so server+adapter flow through Layer 1.
    Returns (fake_engine, captured_kwargs_list)."""
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
    # Parser.extract_reasoning → (reasoning_text, text_for_tools). Return
    # ("", text) since fake output has no <think> block.
    fake._reasoning_parser.extract_reasoning = MagicMock(
        side_effect=lambda text: ("", text)
    )
    fake._reasoning_parser.tokenizer = None

    captured: list[dict] = []
    fake.stream_chat = _finite_stream_chat_factory(captured)
    fake.chat = _chat_factory(captured)

    monkeypatch.setattr(srv, "_engine", fake)
    monkeypatch.setattr(srv, "_model_name", fake.model_name)
    monkeypatch.setattr(srv, "_reasoning_parser", fake._reasoning_parser)
    monkeypatch.setattr(srv, "_reasoning_parser_name", "qwen3")
    monkeypatch.setattr(srv, "_disable_qwen3_first_turn_no_think", False)
    monkeypatch.setattr(srv, "_max_thinking_token_budget", None)

    return fake, captured


def _tool():
    return {
        "name": "Bash",
        "description": "Run shell",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }


def test_layer1_fires_on_qwen3_tools_no_history(qwen3_fake_server):
    fake, captured = qwen3_fake_server
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "messages": [{"role": "user", "content": "Reply OK"}],
            "tools": [_tool()],
            "max_tokens": 200,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-thinking-qwen3-auto-disabled") == "true"
    assert len(captured) == 1
    ctk = captured[0].get("chat_template_kwargs") or {}
    assert ctk.get("enable_thinking") is False


def test_layer1_does_not_fire_without_tools(qwen3_fake_server):
    fake, captured = qwen3_fake_server
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 200,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert "x-thinking-qwen3-auto-disabled" not in resp.headers
    ctk = captured[0].get("chat_template_kwargs") or {}
    assert "enable_thinking" not in ctk


def test_layer1_does_not_fire_with_prior_assistant(qwen3_fake_server):
    fake, captured = qwen3_fake_server
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "again"},
            ],
            "tools": [_tool()],
            "max_tokens": 200,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert "x-thinking-qwen3-auto-disabled" not in resp.headers


def test_layer1_respects_client_explicit_enable_thinking_true(qwen3_fake_server):
    fake, captured = qwen3_fake_server
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_tool()],
            "chat_template_kwargs": {"enable_thinking": True},
            "max_tokens": 200,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert "x-thinking-qwen3-auto-disabled" not in resp.headers
    ctk = captured[0].get("chat_template_kwargs") or {}
    assert ctk.get("enable_thinking") is True  # client's value preserved


def test_layer1_skips_when_operator_opted_out(qwen3_fake_server, monkeypatch):
    fake, captured = qwen3_fake_server
    monkeypatch.setattr(srv, "_disable_qwen3_first_turn_no_think", True)
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_tool()],
            "max_tokens": 200,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert "x-thinking-qwen3-auto-disabled" not in resp.headers


def test_metric_increments_on_layer1_fire(qwen3_fake_server):
    fake, captured = qwen3_fake_server
    from vllm_mlx.metrics import qwen3_first_turn_no_think_applied_total

    before = qwen3_first_turn_no_think_applied_total.value
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_tool()],
            "max_tokens": 200,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert qwen3_first_turn_no_think_applied_total.value == before + 1
