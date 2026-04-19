# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Qwen3.x "interleaved thinking" trap.

When a multi-turn conversation has `type: "thinking"` content blocks in
prior assistant turns, dropping them from the OpenAI-format history
leaves Qwen3.x models in a state where they emit `<think>` but never
close it — they interpret the missing prior thinking as "we're mid-tool-
loop; don't close </think> until the final user-facing answer." The
result: indefinite reasoning until max_tokens or client disconnect.

Cross-referenced with:
- llama.cpp#21118: "Qwen3.5 models are in-context learning to not emit a
  </think> tag when the reasoning from previous turns in the same
  tool_call -> tool -> tool_call -> ... loop is not present."
- litellm#19849: reasoning missing in streaming responses when tools are
  provided.

Fix: when the active reasoning parser uses `<think>` as its start token,
convert Anthropic `type: "thinking"` blocks in assistant history back
into inline `<think>...</think>` text prefixes. Models using other
tokens (e.g. Gemma 4's channel protocol) keep the existing drop
behavior — injecting literal `<think>` text there would just look like
gibberish prose to the model.
"""

import json

from vllm_mlx.api.anthropic_adapter import anthropic_to_openai
from vllm_mlx.api.anthropic_models import AnthropicMessage, AnthropicRequest


def _mk_request(messages, **overrides) -> AnthropicRequest:
    base = {
        "model": "any-model",
        "messages": messages,
        "max_tokens": 512,
    }
    base.update(overrides)
    return AnthropicRequest(**base)


def test_qwen_parser_preserves_thinking_in_assistant_history():
    """When start_token is `<think>`, thinking blocks in assistant history
    must be preserved as inline `<think>…</think>` text in the OpenAI
    message so the model sees the closed-think pattern in history."""
    request = _mk_request(
        messages=[
            AnthropicMessage(role="user", content="What is 2+2?"),
            AnthropicMessage(
                role="assistant",
                content=[
                    {"type": "thinking", "thinking": "Compute 2+2=4."},
                    {"type": "text", "text": "4"},
                ],
            ),
            AnthropicMessage(role="user", content="Now what is 3+3?"),
        ]
    )

    openai_req, _ = anthropic_to_openai(request, reasoning_parser_start_token="<think>")

    # Find the converted assistant message.
    assistant_msgs = [m for m in openai_req.messages if m.role == "assistant"]
    assert len(assistant_msgs) == 1
    content = assistant_msgs[0].content or ""

    # The think tags MUST be present before the answer text.
    assert "<think>" in content, f"missing <think>; content={content!r}"
    assert "</think>" in content, f"missing </think>; content={content!r}"
    assert "Compute 2+2=4." in content
    assert content.index("<think>") < content.index("4"), (
        "thinking must precede answer text"
    )


def test_non_qwen_parser_drops_thinking_as_before():
    """When start_token is NOT `<think>` (e.g. Gemma's channel protocol),
    thinking blocks in history are dropped — injecting literal `<think>`
    text would look like prose to a non-Qwen model."""
    request = _mk_request(
        messages=[
            AnthropicMessage(role="user", content="hi"),
            AnthropicMessage(
                role="assistant",
                content=[
                    {"type": "thinking", "thinking": "Some reasoning."},
                    {"type": "text", "text": "Hello."},
                ],
            ),
            AnthropicMessage(role="user", content="again?"),
        ]
    )

    openai_req, _ = anthropic_to_openai(
        request, reasoning_parser_start_token="<|channel>"
    )

    assistant_msgs = [m for m in openai_req.messages if m.role == "assistant"]
    content = assistant_msgs[0].content or ""
    # Must NOT contain <think> tags — would confuse Gemma's tokenizer.
    assert "<think>" not in content
    assert "Some reasoning." not in content
    # Answer text preserved.
    assert "Hello." in content


def test_no_parser_drops_thinking_as_before():
    """When no parser is active, thinking blocks are dropped (preserves
    pre-fix behavior for non-reasoning models)."""
    request = _mk_request(
        messages=[
            AnthropicMessage(role="user", content="hi"),
            AnthropicMessage(
                role="assistant",
                content=[
                    {"type": "thinking", "thinking": "Reasoning text."},
                    {"type": "text", "text": "Response."},
                ],
            ),
            AnthropicMessage(role="user", content="again?"),
        ]
    )

    # No parser — kwarg omitted.
    openai_req, _ = anthropic_to_openai(request)

    assistant_msgs = [m for m in openai_req.messages if m.role == "assistant"]
    content = assistant_msgs[0].content or ""
    assert "<think>" not in content
    assert "Reasoning text." not in content
    assert "Response." in content


def test_thinking_plus_tool_use_in_same_assistant_turn():
    """Qwen tool-use turn: thinking block + tool_use block on same turn.
    Thinking goes in content field; tool_use becomes an OpenAI tool_calls
    entry. Model history shows <think>...</think> followed by tool invocation."""
    request = _mk_request(
        messages=[
            AnthropicMessage(role="user", content="read my file"),
            AnthropicMessage(
                role="assistant",
                content=[
                    {"type": "thinking", "thinking": "I'll use read_file."},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "read_file",
                        "input": {"path": "foo.txt"},
                    },
                ],
            ),
            AnthropicMessage(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "file contents",
                    }
                ],
            ),
            AnthropicMessage(role="user", content="what did it say?"),
        ]
    )

    openai_req, _ = anthropic_to_openai(request, reasoning_parser_start_token="<think>")

    assistant_msgs = [m for m in openai_req.messages if m.role == "assistant"]
    assert len(assistant_msgs) == 1
    assistant = assistant_msgs[0]

    # Content has closed think tags.
    assert "<think>" in (assistant.content or "")
    assert "</think>" in (assistant.content or "")
    assert "I'll use read_file." in assistant.content

    # Tool call is preserved as OpenAI tool_calls entry.
    assert assistant.tool_calls and len(assistant.tool_calls) == 1
    tc = assistant.tool_calls[0]
    tc_dict = tc if isinstance(tc, dict) else tc.model_dump()
    assert tc_dict.get("function", {}).get("name") == "read_file"


def test_empty_thinking_block_is_ignored():
    """Empty `thinking` field should not produce an empty <think></think>
    wrapper — nothing to preserve, drop cleanly."""
    request = _mk_request(
        messages=[
            AnthropicMessage(role="user", content="hi"),
            AnthropicMessage(
                role="assistant",
                content=[
                    {"type": "thinking", "thinking": ""},
                    {"type": "text", "text": "hello"},
                ],
            ),
            AnthropicMessage(role="user", content="again"),
        ]
    )
    openai_req, _ = anthropic_to_openai(request, reasoning_parser_start_token="<think>")
    content = (openai_req.messages[1].content or "") if openai_req.messages[1].role == "assistant" else ""
    assert "<think>" not in content
    assert "hello" in content


def test_multiple_assistant_turns_all_get_thinking_preserved():
    """Multi-turn with thinking in every assistant turn — each turn's
    thinking block must be preserved in its own assistant message so the
    model sees consistent `<think>...</think>` pattern across history."""
    request = _mk_request(
        messages=[
            AnthropicMessage(role="user", content="Q1"),
            AnthropicMessage(
                role="assistant",
                content=[
                    {"type": "thinking", "thinking": "reason 1"},
                    {"type": "text", "text": "A1"},
                ],
            ),
            AnthropicMessage(role="user", content="Q2"),
            AnthropicMessage(
                role="assistant",
                content=[
                    {"type": "thinking", "thinking": "reason 2"},
                    {"type": "text", "text": "A2"},
                ],
            ),
            AnthropicMessage(role="user", content="Q3"),
        ]
    )
    openai_req, _ = anthropic_to_openai(request, reasoning_parser_start_token="<think>")

    assistant_msgs = [m for m in openai_req.messages if m.role == "assistant"]
    assert len(assistant_msgs) == 2

    for i, msg in enumerate(assistant_msgs, start=1):
        c = msg.content or ""
        assert f"reason {i}" in c
        assert f"A{i}" in c
        assert c.index("<think>") < c.index(f"A{i}")
