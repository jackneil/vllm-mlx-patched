# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Layer 1 surgical auto-disable."""

from unittest.mock import MagicMock

from vllm_mlx.api.thinking_policy import (
    maybe_disable_thinking_for_qwen3_agent_first_turn,
)


def _make_request(
    *,
    messages=None,
    tools=None,
    thinking=None,
    chat_template_kwargs=None,
):
    """Builds a MagicMock mimicking the common MessagesRequest /
    ChatCompletionRequest shape both helpers must handle."""
    req = MagicMock()
    req.messages = (
        messages
        if messages is not None
        else [MagicMock(role="user", content="Reply OK")]
    )
    req.tools = tools
    req.thinking = thinking
    req.chat_template_kwargs = chat_template_kwargs
    return req


def _tool(name="Bash"):
    return {"name": name, "description": "...", "input_schema": {}}


class TestFiringConditions:
    def test_fires_on_qwen3_with_tools_no_assistant(self):
        req = _make_request(tools=[_tool()])
        fired = maybe_disable_thinking_for_qwen3_agent_first_turn(
            req, reasoning_parser_name="qwen3", disabled=False
        )
        assert fired is True
        assert req.chat_template_kwargs == {"enable_thinking": False}

    def test_fires_when_chat_template_kwargs_was_none(self):
        req = _make_request(tools=[_tool()], chat_template_kwargs=None)
        fired = maybe_disable_thinking_for_qwen3_agent_first_turn(
            req, reasoning_parser_name="qwen3", disabled=False
        )
        assert fired is True
        assert req.chat_template_kwargs == {"enable_thinking": False}

    def test_fires_when_chat_template_kwargs_had_other_keys(self):
        req = _make_request(
            tools=[_tool()],
            chat_template_kwargs={"some_other_key": "value"},
        )
        fired = maybe_disable_thinking_for_qwen3_agent_first_turn(
            req, reasoning_parser_name="qwen3", disabled=False
        )
        assert fired is True
        assert req.chat_template_kwargs == {
            "some_other_key": "value",
            "enable_thinking": False,
        }

    def test_fires_when_thinking_is_adaptive(self):
        adaptive = MagicMock(type="adaptive")
        req = _make_request(tools=[_tool()], thinking=adaptive)
        fired = maybe_disable_thinking_for_qwen3_agent_first_turn(
            req, reasoning_parser_name="qwen3", disabled=False
        )
        assert fired is True


class TestNonFiringConditions:
    def test_skips_when_parser_is_not_qwen3(self):
        req = _make_request(tools=[_tool()])
        fired = maybe_disable_thinking_for_qwen3_agent_first_turn(
            req, reasoning_parser_name="gemma4", disabled=False
        )
        assert fired is False
        assert req.chat_template_kwargs is None

    def test_skips_when_parser_is_qwen3_coder(self):
        req = _make_request(tools=[_tool()])
        fired = maybe_disable_thinking_for_qwen3_agent_first_turn(
            req, reasoning_parser_name="qwen3_coder", disabled=False
        )
        assert fired is False

    def test_skips_when_no_tools(self):
        req = _make_request(tools=None)
        fired = maybe_disable_thinking_for_qwen3_agent_first_turn(
            req, reasoning_parser_name="qwen3", disabled=False
        )
        assert fired is False

    def test_skips_when_tools_empty_list(self):
        req = _make_request(tools=[])
        fired = maybe_disable_thinking_for_qwen3_agent_first_turn(
            req, reasoning_parser_name="qwen3", disabled=False
        )
        assert fired is False

    def test_skips_when_prior_assistant_message(self):
        req = _make_request(
            tools=[_tool()],
            messages=[
                MagicMock(role="user", content="first"),
                MagicMock(role="assistant", content="reply"),
                MagicMock(role="user", content="second"),
            ],
        )
        fired = maybe_disable_thinking_for_qwen3_agent_first_turn(
            req, reasoning_parser_name="qwen3", disabled=False
        )
        assert fired is False

    def test_skips_when_operator_opted_out(self):
        req = _make_request(tools=[_tool()])
        fired = maybe_disable_thinking_for_qwen3_agent_first_turn(
            req, reasoning_parser_name="qwen3", disabled=True
        )
        assert fired is False

    def test_skips_when_client_explicit_enable_thinking_true(self):
        req = _make_request(
            tools=[_tool()],
            chat_template_kwargs={"enable_thinking": True},
        )
        fired = maybe_disable_thinking_for_qwen3_agent_first_turn(
            req, reasoning_parser_name="qwen3", disabled=False
        )
        assert fired is False
        # client value preserved
        assert req.chat_template_kwargs == {"enable_thinking": True}

    def test_skips_when_client_explicit_enable_thinking_false(self):
        req = _make_request(
            tools=[_tool()],
            chat_template_kwargs={"enable_thinking": False},
        )
        fired = maybe_disable_thinking_for_qwen3_agent_first_turn(
            req, reasoning_parser_name="qwen3", disabled=False
        )
        assert fired is False

    def test_skips_when_thinking_type_enabled(self):
        enabled = MagicMock(type="enabled")
        req = _make_request(tools=[_tool()], thinking=enabled)
        fired = maybe_disable_thinking_for_qwen3_agent_first_turn(
            req, reasoning_parser_name="qwen3", disabled=False
        )
        assert fired is False

    def test_skips_when_thinking_type_disabled(self):
        disabled = MagicMock(type="disabled")
        req = _make_request(tools=[_tool()], thinking=disabled)
        fired = maybe_disable_thinking_for_qwen3_agent_first_turn(
            req, reasoning_parser_name="qwen3", disabled=False
        )
        assert fired is False

    def test_skips_when_messages_empty(self):
        req = _make_request(tools=[_tool()], messages=[])
        fired = maybe_disable_thinking_for_qwen3_agent_first_turn(
            req, reasoning_parser_name="qwen3", disabled=False
        )
        # No messages means no "first turn" to protect — return False cleanly.
        assert fired is False


def test_helper_sets_request_layer1_fired_marker_on_fire():
    """When the helper fires, it also sets request._layer1_fired = True
    so downstream handlers can emit the diagnostic header without a
    signature change to anthropic_to_openai."""
    req = _make_request(tools=[_tool()])
    fired = maybe_disable_thinking_for_qwen3_agent_first_turn(
        req, reasoning_parser_name="qwen3", disabled=False
    )
    assert fired is True
    assert getattr(req, "_layer1_fired", False) is True


def test_helper_does_not_set_marker_on_skip():
    """When the helper does NOT fire, the marker must remain unset
    — downstream handlers must not see a stale True. Uses a plain class
    instead of MagicMock because Mock auto-creates attributes on access."""

    class _PlainReq:
        messages: list = []
        tools = None
        thinking = None
        chat_template_kwargs = None

    req = _PlainReq()
    fired = maybe_disable_thinking_for_qwen3_agent_first_turn(
        req, reasoning_parser_name="qwen3", disabled=False
    )
    assert fired is False
    assert not hasattr(req, "_layer1_fired")
