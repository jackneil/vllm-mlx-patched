# SPDX-License-Identifier: Apache-2.0
"""Layer 1 Anthropic-adapter wiring tests."""

from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx import server
from vllm_mlx.api.anthropic_adapter import anthropic_to_openai


@pytest.fixture(autouse=True)
def reset_server():
    orig = server._disable_qwen3_first_turn_no_think
    orig_parser = server._reasoning_parser_name
    yield
    server._disable_qwen3_first_turn_no_think = orig
    server._reasoning_parser_name = orig_parser


def _req_with_tools_no_history():
    req = MagicMock()
    req.model = "mlx-community/Qwen3.6-35B-A3B-4bit"
    req.messages = [MagicMock(role="user", content="hi")]
    req.max_tokens = 4000
    req.temperature = None
    req.top_p = None
    req.stream = True
    req.stop_sequences = None
    tool = MagicMock()
    tool.name = "Bash"
    tool.description = "Run a bash command"
    tool.input_schema = {"type": "object"}
    req.tools = [tool]
    req.tool_choice = None
    req.thinking_token_budget = None
    req.thinking_budget_message = None
    req.thinking = None
    req.output_config = MagicMock(effort="high")
    req.system = None
    req.chat_template_kwargs = None
    return req


def test_adapter_fires_layer1_on_matching_payload(monkeypatch):
    """Layer 1 reads reasoning_parser_name from server module."""
    monkeypatch.setattr(server, "_reasoning_parser_name", "qwen3")
    monkeypatch.setattr(server, "_disable_qwen3_first_turn_no_think", False)
    req = _req_with_tools_no_history()
    with patch(
        "vllm_mlx.api.anthropic_adapter._convert_message",
        side_effect=lambda m, **kw: [{"role": m.role, "content": m.content}],
    ):
        anthropic_to_openai(req, context_window=131072)
    assert req.chat_template_kwargs == {"enable_thinking": False}
    assert getattr(req, "_layer1_fired", False) is True


def test_adapter_skips_layer1_when_disabled(monkeypatch):
    monkeypatch.setattr(server, "_reasoning_parser_name", "qwen3")
    monkeypatch.setattr(server, "_disable_qwen3_first_turn_no_think", True)
    req = _req_with_tools_no_history()
    with patch(
        "vllm_mlx.api.anthropic_adapter._convert_message",
        side_effect=lambda m, **kw: [{"role": m.role, "content": m.content}],
    ):
        anthropic_to_openai(req, context_window=131072)
    assert req.chat_template_kwargs is None


def test_adapter_skips_layer1_on_non_qwen3(monkeypatch):
    monkeypatch.setattr(server, "_reasoning_parser_name", "gemma4")
    monkeypatch.setattr(server, "_disable_qwen3_first_turn_no_think", False)
    req = _req_with_tools_no_history()
    with patch(
        "vllm_mlx.api.anthropic_adapter._convert_message",
        side_effect=lambda m, **kw: [{"role": m.role, "content": m.content}],
    ):
        anthropic_to_openai(req, context_window=131072)
    assert req.chat_template_kwargs is None
