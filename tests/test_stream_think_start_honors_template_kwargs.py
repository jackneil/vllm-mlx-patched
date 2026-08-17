# SPDX-License-Identifier: Apache-2.0
"""Streaming think-router start state must follow THIS request's template kwargs.

Regression for the "Claude Code sees no visible output" bug on Qwen3.x served
by vllm-mlx (hit 2026-08-16 on Qwen3.8-27B, stock and abliterated alike):

* Claude Code's opener carries ``tools`` and no prior assistant turn, so Layer 1
  (``thinking_policy.py``) injects ``chat_template_kwargs.enable_thinking=False``.
* With that kwarg the Qwen3 template renders a CLOSED tail
  (``<think>\\n\\n</think>\\n\\n``); the model answers directly, no tags.
* ``_detect_starts_thinking`` probed the template WITHOUT the request's kwargs,
  saw the default OPEN tail (``<think>\\n``) and started the router in thinking
  mode, so the entire tagless answer streamed out as a ``thinking`` block and
  no ``text`` block was ever emitted. Non-streaming was fine (Case 4: no tags =
  content), which is why curl smoke tests passed while Claude Code got nothing.

The OpenAI ``/v1/chat/completions`` stream had the same class of bug through
``BaseThinkingReasoningParser`` Case 3 (tagless output = reasoning), and worse:
tool-call markup landed in ``reasoning`` and never reached the tool parser.

Both paths now derive their initial state from the same probe rendered with the
request's own ``chat_template_kwargs``.
"""

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from vllm_mlx import server as srv
from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

MODEL = "hank-ai/Qwen3.8-27B-ABLITERATED-4bit"

# What the real Qwen3.5/3.6/3.8 template renders at the end of the generation
# prompt (verified 2026-08-16 against the served Qwen3.8 tokenizer):
OPEN_TAIL = "<|im_start|>assistant\n<think>\n"
CLOSED_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


class _FakeQwen3Tokenizer:
    """Stand-in for the served tokenizer: honors ``enable_thinking`` exactly
    like the real Jinja template does, and records every render call."""

    chat_template = (
        "{% for m in messages %}...{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n"
        "{% if enable_thinking is defined and enable_thinking is false %}"
        "<think>\n\n</think>\n\n{% else %}<think>\n{% endif %}{% endif %}"
    )

    def __init__(self):
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(dict(kwargs))
        body = "<|im_start|>user\nx<|im_end|>\n"
        if not kwargs.get("add_generation_prompt", False):
            return body
        if kwargs.get("enable_thinking") is False:
            return body + CLOSED_TAIL
        return body + OPEN_TAIL


class _FakeOutput:
    def __init__(self, new_text):
        self.new_text = new_text
        self.text = new_text
        self.prompt_tokens = 10
        self.completion_tokens = 2
        self.finished = True
        self.finish_reason = "stop"
        self.thinking_budget_applied = False
        self.thinking_budget_noop_reason = None


def _stream_chat_factory(captured, text):
    def _stream_chat(**kwargs):
        captured.append(kwargs)

        async def _gen():
            yield _FakeOutput(text)

        return _gen()

    return _stream_chat


@pytest.fixture
def qwen3_server(monkeypatch):
    """Fake Qwen3 BatchedEngine + a REAL Qwen3ReasoningParser, so the request
    flows through Layer 1, the adapter, and both streaming handlers unchanged.
    Yields (fake_engine, captured_engine_kwargs, set_model_output)."""
    fake = MagicMock()
    fake.preserve_native_tool_format = False
    fake.model_name = MODEL
    fake.tokenizer = _FakeQwen3Tokenizer()
    fake._is_mllm = False
    fake.is_mllm = False
    fake.__class__ = __import__(
        "vllm_mlx.engine.batched", fromlist=["BatchedEngine"]
    ).BatchedEngine

    parser = Qwen3ReasoningParser(tokenizer=None)
    fake._reasoning_parser = parser

    captured: list[dict] = []
    state = {"text": "PONG"}

    def _stream_chat(**kwargs):
        return _stream_chat_factory(captured, state["text"])(**kwargs)

    fake.stream_chat = _stream_chat

    monkeypatch.setattr(srv, "_engine", fake)
    monkeypatch.setattr(srv, "_model_name", MODEL)
    monkeypatch.setattr(srv, "_reasoning_parser", parser)
    monkeypatch.setattr(srv, "_reasoning_parser_name", "qwen3")
    monkeypatch.setattr(srv, "_disable_qwen3_first_turn_no_think", False)
    monkeypatch.setattr(srv, "_max_thinking_token_budget", None)

    def _set_output(text):
        state["text"] = text

    return fake, captured, _set_output


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


def _openai_tool():
    return {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run shell",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }


def _anthropic_events(body: str) -> list[dict]:
    """Parse an Anthropic SSE body into a list of event dicts."""
    events = []
    for chunk in body.strip().split("\n\n"):
        lines = [ln for ln in chunk.split("\n") if ln.strip()]
        if not lines:
            continue
        data = next((ln for ln in lines if ln.startswith("data: ")), None)
        if data is None:
            continue
        events.append(json.loads(data[len("data: ") :]))
    return events


def _anthropic_blocks(body: str) -> dict[str, str]:
    """Return {block_type: concatenated delta text} from an Anthropic SSE body."""
    started: dict[int, str] = {}
    text: dict[str, list[str]] = {}
    for ev in _anthropic_events(body):
        t = ev.get("type")
        if t == "content_block_start":
            started[ev["index"]] = ev["content_block"]["type"]
            text.setdefault(ev["content_block"]["type"], [])
        elif t == "content_block_delta":
            d = ev["delta"]
            if d.get("type") == "text_delta":
                text.setdefault("text", []).append(d["text"])
            elif d.get("type") == "thinking_delta":
                text.setdefault("thinking", []).append(d["thinking"])
    return {k: "".join(v) for k, v in text.items()}


def _openai_deltas(body: str) -> list[dict]:
    out = []
    for line in body.splitlines():
        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
            continue
        chunk = json.loads(line[len("data: ") :])
        for ch in chunk.get("choices", []):
            out.append(ch.get("delta") or {})
    return out


# --------------------------------------------------------------------------- #
# Anthropic /v1/messages streaming
# --------------------------------------------------------------------------- #


def test_anthropic_stream_layer1_closed_think_routes_tagless_answer_as_text(
    qwen3_server,
):
    """The Claude Code opener shape (tools + first turn): Layer 1 closes the
    think block, so a tagless answer MUST arrive as a text block."""
    fake, captured, _ = qwen3_server
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
            "tools": [_tool()],
            "max_tokens": 200,
            "stream": True,
        },
    )
    assert resp.status_code == 200
    # Layer 1 fired and reached the engine.
    assert resp.headers.get("x-thinking-qwen3-auto-disabled") == "true"
    assert (captured[0].get("chat_template_kwargs") or {}).get("enable_thinking") is False
    # The router's probe rendered with the SAME kwargs the engine used.
    probe_calls = [c for c in fake.tokenizer.calls if c.get("add_generation_prompt")]
    assert probe_calls, "router never probed the template"
    assert probe_calls[-1].get("enable_thinking") is False

    blocks = _anthropic_blocks(resp.text)
    assert blocks.get("text") == "PONG", blocks
    assert "thinking" not in blocks, blocks


def test_anthropic_stream_client_enable_thinking_false_routes_as_text(qwen3_server):
    """A client-set chat_template_kwargs.enable_thinking=False (no tools, so
    Layer 1 stays out) is the same closed tail and must route the same way."""
    fake, captured, _ = qwen3_server
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
            "max_tokens": 200,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-thinking-qwen3-auto-disabled") != "true"
    blocks = _anthropic_blocks(resp.text)
    assert blocks.get("text") == "PONG", blocks
    assert "thinking" not in blocks, blocks


def test_anthropic_stream_open_think_still_routes_tagless_prefix_as_thinking(
    qwen3_server,
):
    """Control: no tools -> Layer 1 does not fire -> the template's default OPEN
    tail (`<think>\\n`) means the model IS inside a think block, so a tagless
    prefix is reasoning (implicit mode). This is the legacy behavior and it
    must survive the fix. Model output: `reasoning</think>PONG`."""
    fake, captured, set_output = qwen3_server
    set_output("thinking hard</think>PONG")
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
            "max_tokens": 200,
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert (captured[0].get("chat_template_kwargs") or {}).get("enable_thinking") is None
    blocks = _anthropic_blocks(resp.text)
    assert blocks.get("thinking") == "thinking hard", blocks
    assert blocks.get("text") == "PONG", blocks


def test_anthropic_stream_layer1_closed_think_explicit_tags_still_split(
    qwen3_server,
):
    """Even with the think block closed by the template, a model that emits an
    explicit `<think>...</think>` pair still gets it routed as thinking."""
    fake, captured, set_output = qwen3_server
    set_output("<think>hmm</think>PONG")
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
            "tools": [_tool()],
            "max_tokens": 200,
            "stream": True,
        },
    )
    assert resp.status_code == 200
    blocks = _anthropic_blocks(resp.text)
    assert blocks.get("thinking") == "hmm", blocks
    assert blocks.get("text") == "PONG", blocks


# --------------------------------------------------------------------------- #
# OpenAI /v1/chat/completions streaming
# --------------------------------------------------------------------------- #


def test_openai_stream_layer1_closed_think_emits_tagless_answer_as_content(
    qwen3_server,
):
    """Same fingerprint on the OpenAI path: tagless answer is `content`, not
    `reasoning` (and therefore reaches the tool parser)."""
    fake, captured, _ = qwen3_server
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
            "tools": [_openai_tool()],
            "max_tokens": 200,
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert (captured[0].get("chat_template_kwargs") or {}).get("enable_thinking") is False
    deltas = _openai_deltas(resp.text)
    content = "".join(d.get("content") or "" for d in deltas)
    reasoning = "".join(d.get("reasoning") or "" for d in deltas)
    assert content == "PONG", deltas
    assert reasoning == "", deltas


def test_openai_stream_open_think_keeps_tagless_prefix_as_reasoning(qwen3_server):
    """Control for the OpenAI path: template open (no tools) -> implicit mode ->
    `reasoning</think>PONG` splits into reasoning + content, as before."""
    fake, captured, set_output = qwen3_server
    set_output("thinking hard</think>PONG")
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
            "max_tokens": 200,
            "stream": True,
        },
    )
    assert resp.status_code == 200
    deltas = _openai_deltas(resp.text)
    content = "".join(d.get("content") or "" for d in deltas)
    reasoning = "".join(d.get("reasoning") or "" for d in deltas)
    assert reasoning == "thinking hard", deltas
    assert content == "PONG", deltas
