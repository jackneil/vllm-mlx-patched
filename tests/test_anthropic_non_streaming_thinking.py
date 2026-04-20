# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for non-streaming /v1/messages thinking emission.

Pre-fix, the non-streaming Anthropic handler at server.py built the
intermediate OpenAI response with `content=final_content` only — never
calling `_reasoning_parser.extract_reasoning()`. Because `openai_to_anthropic`
reads `choice.message.reasoning` to decide whether to emit a
type:"thinking" content block, the thinking block was NEVER emitted on
this path. Any `<think>…</think>` content leaked into the text block.

These tests construct the FastAPI TestClient, monkey-patch the engine +
reasoning parser globals, and verify the type:"thinking" content block is
emitted with a signature, and `usage.thinking_tokens` carries a count.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from vllm_mlx import server as srv


class _FakeOutput:
    """Minimal shape matching what engine.chat returns."""

    def __init__(self, text: str):
        self.text = text
        self.prompt_tokens = 10
        self.completion_tokens = 50
        self.finish_reason = "stop"
        self.thinking_budget_applied = None
        self.thinking_budget_noop_reason = None


class _FakeTokenizer:
    """Tokenizer that returns char-count tokens (stable + predictable)."""

    def encode(self, text: str):
        return list(range(len(text)))


class _FakeReasoningParser:
    """Splits content on <think>…</think>; returns (reasoning, remainder)."""

    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def extract_reasoning(self, text: str):
        if "<think>" not in text or "</think>" not in text:
            return None, text
        start = text.index("<think>") + len("<think>")
        end = text.index("</think>")
        reasoning = text[start:end]
        remainder = text[: text.index("<think>")] + text[end + len("</think>") :]
        return reasoning, remainder or None


@pytest.fixture
def patched_server(monkeypatch):
    """Patch module globals so /v1/messages can run with a fake engine."""
    fake_engine = MagicMock()
    # Default happy-path output; individual tests override .chat return.
    fake_engine.chat = MagicMock()
    fake_engine.preserve_native_tool_format = False

    parser = _FakeReasoningParser()

    monkeypatch.setattr(srv, "_engine", fake_engine)
    monkeypatch.setattr(srv, "_reasoning_parser", parser)
    monkeypatch.setattr(srv, "_model_name", "fake-model")
    # Disable the validate_model check path (anthropic request model must match)
    # _validate_model_name sees request.model != _model_name and 404s otherwise.

    return fake_engine, parser


def _post_messages(engine, model: str, content: str, text: str):
    """Helper: set engine.chat to return a GenerationOutput with `text`,
    POST /v1/messages, return parsed response JSON."""

    async def _fake_chat(messages=None, **kwargs):
        return _FakeOutput(text)

    engine.chat = _fake_chat

    client = TestClient(srv.app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 100,
            "stream": False,
        },
    )
    return resp


def test_non_streaming_emits_thinking_block_with_signature(patched_server):
    """Bug repro: reasoning in `<think>…</think>` must surface as a
    type:'thinking' content block, NOT embedded in the text block."""
    engine, _parser = patched_server
    text = "<think>step 1, step 2</think>The answer is 42."

    resp = _post_messages(engine, "fake-model", "What is the answer?", text)
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["type"] == "message"
    blocks = body["content"]
    types = [b["type"] for b in blocks]

    # Thinking block MUST exist. Pre-fix it was absent — reasoning was
    # inside the text block.
    assert "thinking" in types, f"thinking block missing; got: {types}"

    thinking_block = next(b for b in blocks if b["type"] == "thinking")
    assert thinking_block["thinking"] == "step 1, step 2"

    # Signature (PR #14 contract): deterministic sha256-derived opaque tag.
    assert "signature" in thinking_block
    assert thinking_block["signature"].startswith("vllm-mlx:")
    assert len(thinking_block["signature"]) == len("vllm-mlx:") + 32


def test_non_streaming_emits_text_block_without_think_tags(patched_server):
    """The text block must contain ONLY the post-</think> remainder, not
    the full original output with tags embedded."""
    engine, _parser = patched_server
    text = "<think>pondering</think>Final answer: 42."

    resp = _post_messages(engine, "fake-model", "q", text)
    body = resp.json()

    text_block = next(b for b in body["content"] if b["type"] == "text")
    assert text_block["text"] == "Final answer: 42."
    assert "<think>" not in text_block["text"]
    assert "</think>" not in text_block["text"]


def test_non_streaming_populates_thinking_tokens_usage(patched_server):
    """AnthropicUsage.thinking_tokens must be populated when reasoning
    content was extracted. Fake tokenizer returns char-count tokens."""
    engine, _parser = patched_server
    reasoning = "step 1, step 2"
    text = f"<think>{reasoning}</think>Answer."

    resp = _post_messages(engine, "fake-model", "q", text)
    body = resp.json()

    usage = body["usage"]
    assert "thinking_tokens" in usage
    assert usage["thinking_tokens"] == len(reasoning)  # matches fake tokenizer


def test_non_streaming_no_reasoning_leaves_thinking_tokens_absent(
    patched_server,
):
    """When the model emits no <think> tags, no thinking block and no
    thinking_tokens field in the response body (field is None + excluded)."""
    engine, _parser = patched_server
    text = "Plain answer with no reasoning tags."

    resp = _post_messages(engine, "fake-model", "q", text)
    body = resp.json()

    types = [b["type"] for b in body["content"]]
    assert "thinking" not in types

    usage = body["usage"]
    # model_dump_json(exclude_none=True) drops the None thinking_tokens.
    assert "thinking_tokens" not in usage


def test_non_streaming_without_parser_leaves_think_tags_in_text(
    patched_server, monkeypatch
):
    """When no reasoning parser is configured, handler can't split
    reasoning — text block carries the raw tags. No crash, no thinking
    block."""
    engine, _parser = patched_server
    monkeypatch.setattr(srv, "_reasoning_parser", None)  # disable parser

    text = "<think>x</think>y"
    resp = _post_messages(engine, "fake-model", "q", text)
    body = resp.json()

    types = [b["type"] for b in body["content"]]
    assert "thinking" not in types
    text_block = next(b for b in body["content"] if b["type"] == "text")
    # With no parser, tags remain in the text — existing behavior preserved.
    assert "<think>" in text_block["text"]


def test_non_streaming_reasoning_only_emits_thinking_without_text(
    patched_server,
):
    """Edge: model output is entirely reasoning (no content after </think>).
    Thinking block should exist; text block either absent or empty-string."""
    engine, _parser = patched_server
    text = "<think>only thinking, no answer</think>"

    resp = _post_messages(engine, "fake-model", "q", text)
    body = resp.json()

    # Must have the thinking block
    thinking_blocks = [b for b in body["content"] if b["type"] == "thinking"]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0]["thinking"] == "only thinking, no answer"


def test_non_streaming_thinking_plus_tool_call(patched_server, monkeypatch):
    """Edge: model emits both reasoning AND a tool call. Reasoning
    extraction must run BEFORE tool parsing (issue #161) so the tool
    parser sees only the post-</think> remainder. Result: thinking
    block + tool_use block, no text block."""
    engine, _parser = patched_server

    # Stub the tool-call parser to return a tool call. This mirrors what
    # _parse_tool_calls_with_parser would do given a real tool-call in
    # the text_for_tools string.
    from vllm_mlx.api.models import FunctionCall, ToolCall

    def _fake_parse(text, request):
        if "call_my_tool" in (text or ""):
            return "", [
                ToolCall(
                    id="call_abc",
                    type="function",
                    function=FunctionCall(
                        name="my_tool",
                        arguments='{"x":1}',
                    ),
                )
            ]
        return text, []

    monkeypatch.setattr(srv, "_parse_tool_calls_with_parser", _fake_parse)

    text = "<think>plan: call the tool</think>call_my_tool"
    resp = _post_messages(engine, "fake-model", "q", text)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    types = [b["type"] for b in body["content"]]
    # MUST have thinking block AND tool_use, no text block (cleaned == "").
    assert "thinking" in types, f"thinking missing; got {types}"
    assert "tool_use" in types, f"tool_use missing; got {types}"

    thinking_block = next(b for b in body["content"] if b["type"] == "thinking")
    assert thinking_block["thinking"] == "plan: call the tool"


def test_non_streaming_tokenizer_raises_leaves_thinking_tokens_none(
    patched_server,
):
    """If the tokenizer raises on the reasoning text, thinking_tokens
    stays None (field excluded from response). The response itself must
    still succeed — tokenization is best-effort observability."""
    engine, parser = patched_server

    class _AngryTokenizer:
        def encode(self, text):
            raise ValueError("simulated tokenizer failure")

    parser.tokenizer = _AngryTokenizer()

    text = "<think>reasoning</think>answer"
    resp = _post_messages(engine, "fake-model", "q", text)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Thinking block still emitted (extract_reasoning is independent).
    types = [b["type"] for b in body["content"]]
    assert "thinking" in types

    # But thinking_tokens is absent — tokenization failed, field stays None.
    assert "thinking_tokens" not in body["usage"]
