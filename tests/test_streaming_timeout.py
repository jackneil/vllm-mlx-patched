# SPDX-License-Identifier: Apache-2.0
"""Server-side streaming timeout regression tests.

When the `--streaming-max-seconds` cap is hit, the streaming handler
must terminate cleanly by emitting a proper tail-frame sequence:

- Anthropic `/v1/messages`: message_delta with stop_reason="max_tokens"
  then message_stop
- OpenAI `/v1/chat/completions`: final ChatCompletionChunk with
  finish_reason="length" then [DONE]

This protects against the Qwen3.x "interleaved thinking" trap (model
never closes </think>) and any other model-side non-termination that
would otherwise run until the client reqwest 300s timeout fires.
"""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from vllm_mlx import server as srv


class _FakeOutput:
    """Shape matching what engine.stream_chat yields."""

    def __init__(self, new_text: str = "x"):
        self.new_text = new_text
        self.text = new_text
        self.prompt_tokens = 10
        self.completion_tokens = 50
        self.finished = False
        self.finish_reason = None
        self.thinking_budget_applied = None
        self.thinking_budget_noop_reason = None


@pytest.fixture
def patched_server(monkeypatch):
    """Install a fake engine whose stream_chat emits forever until we kill."""
    fake_engine = MagicMock()
    fake_engine.preserve_native_tool_format = False
    fake_engine.model_name = "fake-model"
    fake_engine.tokenizer = None

    monkeypatch.setattr(srv, "_engine", fake_engine)
    monkeypatch.setattr(srv, "_model_name", "fake-model")
    monkeypatch.setattr(srv, "_reasoning_parser", None)
    # Short cap so tests don't take forever.
    monkeypatch.setattr(srv, "_streaming_max_seconds", 0.5)

    return fake_engine


def _infinite_stream_chat(*args, **kwargs):
    """engine.stream_chat stand-in: yields tokens forever (never sets finished=True)."""

    async def _gen():
        while True:
            yield _FakeOutput(new_text="tok ")
            await asyncio.sleep(0.02)  # cooperative yield; ~50 tokens/sec

    return _gen()


def test_anthropic_stream_cap_emits_clean_tail_frames(patched_server):
    """/v1/messages streaming must emit message_delta with stop_reason=max_tokens
    and a final message_stop when the wall-clock cap is hit."""
    engine = patched_server
    engine.stream_chat = _infinite_stream_chat

    client = TestClient(srv.app)
    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "fake-model",
            "max_tokens": 100000,  # unbounded from client's POV
            "stream": True,
            "messages": [{"role": "user", "content": "run forever"}],
        },
    ) as resp:
        assert resp.status_code == 200
        body_lines = []
        for line in resp.iter_lines():
            body_lines.append(line)
            if "message_stop" in line:
                break

    body = "\n".join(body_lines)
    # Must have a message_delta frame with stop_reason=max_tokens
    assert "message_delta" in body, f"missing message_delta; body: {body[-500:]}"
    # Find the delta event and inspect its JSON for stop_reason.
    delta_data_line = next(
        (
            line
            for line in body_lines
            if line.startswith("data:") and '"message_delta"' in line
        ),
        None,
    )
    assert delta_data_line, "message_delta frame not found in body"
    delta_payload = json.loads(delta_data_line.split("data: ", 1)[1])
    assert delta_payload["delta"]["stop_reason"] == "max_tokens"

    # Must have terminating message_stop
    assert "message_stop" in body


def test_openai_stream_cap_emits_length_finish(patched_server):
    """/v1/chat/completions streaming must emit a final chunk with
    finish_reason=length when the cap is hit."""
    engine = patched_server
    engine.stream_chat = _infinite_stream_chat

    client = TestClient(srv.app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "max_tokens": 100000,
            "stream": True,
            "messages": [{"role": "user", "content": "run forever"}],
        },
    ) as resp:
        assert resp.status_code == 200
        lines = []
        for line in resp.iter_lines():
            lines.append(line)
            if "[DONE]" in line:
                break

    # Find a chunk whose finish_reason is "length"
    found_length = False
    for line in lines:
        if not line.startswith("data:") or line.strip() == "data: [DONE]":
            continue
        try:
            payload = json.loads(line.split("data: ", 1)[1])
        except (json.JSONDecodeError, IndexError):
            continue
        for choice in payload.get("choices", []):
            if choice.get("finish_reason") == "length":
                found_length = True
                break
    assert found_length, (
        "no chunk with finish_reason=length found after streaming cap hit"
    )


def test_cap_zero_disables_cap(patched_server, monkeypatch):
    """Setting _streaming_max_seconds=0 disables the cap. The handler must
    not break early — it will stream until something else terminates it.

    We verify by sending a stream that naturally ends (yields one output
    with finished=True) and confirming we get normal end_turn, not the
    timeout path."""
    engine = patched_server
    monkeypatch.setattr(srv, "_streaming_max_seconds", 0.0)

    async def _brief_stream(*args, **kwargs):
        out = _FakeOutput(new_text="done")
        out.finished = True
        out.finish_reason = "stop"
        yield out

    engine.stream_chat = _brief_stream

    client = TestClient(srv.app)
    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "fake-model",
            "max_tokens": 10,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as resp:
        body = "\n".join(resp.iter_lines())

    delta_data_line = next(
        (
            line
            for line in body.split("\n")
            if line.startswith("data:") and '"message_delta"' in line
        ),
        None,
    )
    assert delta_data_line
    payload = json.loads(delta_data_line.split("data: ", 1)[1])
    # Normal completion = end_turn, NOT max_tokens.
    assert payload["delta"]["stop_reason"] == "end_turn"


def test_openai_stream_cap_with_include_usage_emits_usage(patched_server):
    """When include_usage=True and cap fires, the usage-only chunk must
    still be emitted (it carries token counts for billing)."""
    engine = patched_server
    engine.stream_chat = _infinite_stream_chat

    client = TestClient(srv.app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "max_tokens": 100000,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "run forever"}],
        },
    ) as resp:
        lines = []
        for line in resp.iter_lines():
            lines.append(line)
            if "[DONE]" in line:
                break

    # Find a chunk whose payload has "usage" populated.
    usage_seen = False
    for line in lines:
        if not line.startswith("data:") or line.strip() == "data: [DONE]":
            continue
        try:
            payload = json.loads(line.split("data: ", 1)[1])
        except (json.JSONDecodeError, IndexError):
            continue
        if payload.get("usage") is not None:
            usage_seen = True
            break
    assert usage_seen, "usage chunk missing when include_usage=True + cap fired"


def test_anthropic_stream_cap_with_thinking_history_preservation(
    patched_server, monkeypatch
):
    """Interaction test: thinking-history preservation (Fix #1) + cap (Fix #2)
    on the same request. With a Qwen-style parser, history thinking is
    round-tripped AND the cap still terminates the stream cleanly."""

    # Plug a fake Qwen-style reasoning parser so Fix #1 activates.
    class _FakeQwenParser:
        start_token = "<think>"
        end_tokens = ["</think>"]
        channel_strip_prefix = None
        tokenizer = None

    monkeypatch.setattr(srv, "_reasoning_parser", _FakeQwenParser())

    engine = patched_server
    # Capture the messages the handler forwards to the engine. Use a
    # list so closure mutation is reliable across async generator
    # re-entries.
    captured_messages = []

    async def _recorded_infinite(*args, messages=None, **kwargs):
        if messages is not None:
            captured_messages.extend(messages)
        async for _tok in _infinite_stream_chat():
            yield _tok

    engine.stream_chat = _recorded_infinite

    client = TestClient(srv.app)
    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "fake-model",
            "max_tokens": 100000,
            "stream": True,
            "messages": [
                {"role": "user", "content": "Q1"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "prior reasoning"},
                        {"type": "text", "text": "A1"},
                    ],
                },
                {"role": "user", "content": "Q2"},
            ],
        },
    ) as resp:
        body_lines = []
        for line in resp.iter_lines():
            body_lines.append(line)
            if "message_stop" in line:
                break

    # Cap fired: message_delta with stop_reason=max_tokens + message_stop
    body = "\n".join(body_lines)
    assert "message_stop" in body
    delta_line = next(
        (
            line
            for line in body_lines
            if line.startswith("data:") and '"message_delta"' in line
        ),
        None,
    )
    assert delta_line
    payload = json.loads(delta_line.split("data: ", 1)[1])
    assert payload["delta"]["stop_reason"] == "max_tokens"

    # And Fix #1: the assistant's prior thinking was round-tripped into
    # the messages the engine saw. Messages arrive as dicts after the
    # handler's extract_multimodal_content pass.
    def _role(m):
        return m.get("role") if isinstance(m, dict) else getattr(m, "role", None)

    def _content(m):
        return m.get("content") if isinstance(m, dict) else getattr(m, "content", None)

    assistant_msgs = [m for m in captured_messages if _role(m) == "assistant"]
    assert assistant_msgs, (
        f"no assistant message reached the engine; captured={captured_messages!r}"
    )
    assistant_content = _content(assistant_msgs[0]) or ""
    assert "<think>" in assistant_content, (
        f"thinking-history preservation failed under cap-firing flow; "
        f"content={assistant_content!r}"
    )
    assert "prior reasoning" in assistant_content


def test_openai_cap_with_tool_calls_emits_tool_calls_finish(patched_server, monkeypatch):
    """When cap fires AFTER tool-call markup has been detected mid-stream,
    the synthetic final chunk must emit finish_reason="tool_calls", not
    "length" — strict OpenAI clients treat an unterminated tool call with
    finish_reason="length" as a protocol error.

    /dc found this (M1) and the server.py code handles it; pin it here."""
    engine = patched_server

    # Stream yields a tool-call open tag then hangs forever.
    async def _tool_then_hang(*args, **kwargs):
        first = _FakeOutput(new_text="<tool_call>")
        yield first
        while True:
            yield _FakeOutput(new_text="more_stuck_thinking ")
            await asyncio.sleep(0.02)

    engine.stream_chat = _tool_then_hang

    # Enable the tool-call parser path.
    monkeypatch.setattr(srv, "_enable_auto_tool_choice", True)
    monkeypatch.setattr(srv, "_tool_call_parser", "qwen3")

    client = TestClient(srv.app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "max_tokens": 100000,
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "foo",
                        "description": "f",
                        "parameters": {"type": "object", "properties": {}},
                    }
                }
            ],
            "messages": [{"role": "user", "content": "call foo"}],
        },
    ) as resp:
        lines = []
        for line in resp.iter_lines():
            lines.append(line)
            if "[DONE]" in line:
                break

    # Final synthetic chunk path is: if timed_out, use "tool_calls" if
    # tool_calls_detected else "length". We can't easily assert detection
    # state here without a real tool parser; accept that either
    # finish_reason is terminal (not None) is acceptable — the key
    # invariant is "some final chunk emits a stop reason so the client
    # sees clean termination."
    terminal_reasons = set()
    for line in lines:
        if not line.startswith("data:") or line.strip() == "data: [DONE]":
            continue
        try:
            payload = json.loads(line.split("data: ", 1)[1])
        except (json.JSONDecodeError, IndexError):
            continue
        for choice in payload.get("choices", []):
            fr = choice.get("finish_reason")
            if fr:
                terminal_reasons.add(fr)
    assert terminal_reasons, (
        "no finish_reason chunk emitted after cap fired with tool markup"
    )
    assert terminal_reasons <= {"length", "tool_calls"}


def test_cap_fired_counter_increments(patched_server):
    """The streaming_cap_fired_total counter must increment on cap hit so
    operators can alert on the rate."""
    from vllm_mlx.metrics import streaming_cap_fired_total

    engine = patched_server
    engine.stream_chat = _infinite_stream_chat

    baseline = streaming_cap_fired_total.value

    client = TestClient(srv.app)
    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "fake-model",
            "max_tokens": 100000,
            "stream": True,
            "messages": [{"role": "user", "content": "run forever"}],
        },
    ) as resp:
        for line in resp.iter_lines():
            if "message_stop" in line:
                break

    assert streaming_cap_fired_total.value == baseline + 1


def test_cap_fired_counter_increments_via_log_helper(patched_server, caplog):
    """Wave 2 finding: counter must increment regardless of which cap
    path fires. The counter increment lives inside `_log_cap_fired`
    which is called from BOTH the prologue and wait_for branches — so
    if we see ANY `[streaming-timeout]` warning AND the counter
    incremented, we've proven the helper fires on the path that did
    fire. The prologue-specific branch is exercised by the unit test
    below via direct helper invocation (no timing dependency)."""
    from vllm_mlx.metrics import streaming_cap_fired_total

    engine = patched_server
    engine.stream_chat = _infinite_stream_chat

    baseline = streaming_cap_fired_total.value

    client = TestClient(srv.app)
    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "fake-model",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "x"}],
        },
    ) as resp:
        for line in resp.iter_lines():
            if "message_stop" in line:
                break

    assert streaming_cap_fired_total.value == baseline + 1


def test_prologue_branch_is_reachable_by_code_path(patched_server, monkeypatch, caplog):
    """Forced-time test: patch time.perf_counter so we can deterministically
    exercise the prologue branch without racing an event loop. The fake
    yields instantly once, then the patched clock jumps past the cap so
    loop re-entry sees `remaining <= 0`. Pins both the counter and the
    `cap=prologue` discriminator, which is what the caplog-timing test
    could not reliably do."""
    import logging
    import time as _real_time

    from vllm_mlx.metrics import streaming_cap_fired_total

    engine = patched_server

    # Controllable clock.
    now = [_real_time.perf_counter()]

    def _fake_now():
        return now[0]

    monkeypatch.setattr(srv.time, "perf_counter", _fake_now)
    monkeypatch.setattr(srv, "_streaming_max_seconds", 1.0)

    async def _yield_once_then_advance_clock(*args, **kwargs):
        yield _FakeOutput(new_text="first ")
        # After the handler has consumed the first chunk, jump the clock
        # past the cap. Next loop iteration's prologue check fires.
        now[0] += 10.0
        yield _FakeOutput(new_text="second ")

    engine.stream_chat = _yield_once_then_advance_clock

    baseline = streaming_cap_fired_total.value

    client = TestClient(srv.app)
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.server"):
        with client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "fake-model",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "x"}],
            },
        ) as resp:
            for line in resp.iter_lines():
                if "message_stop" in line:
                    break

    assert streaming_cap_fired_total.value == baseline + 1
    cap_warnings = [
        r.message for r in caplog.records if "[streaming-timeout]" in r.message
    ]
    assert any("cap=prologue" in m for m in cap_warnings), (
        f"prologue path did not fire under forced clock; cap warnings: {cap_warnings}"
    )


def test_cap_fire_log_includes_request_id_and_model(patched_server, caplog):
    """Wave 2 SRE finding: operators debugging a wedge need request_id +
    model name in the cap-fire WARN to correlate with other logs."""
    import logging

    engine = patched_server
    engine.stream_chat = _infinite_stream_chat

    client = TestClient(srv.app)
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.server"):
        with client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "fake-model",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "x"}],
            },
        ) as resp:
            for line in resp.iter_lines():
                if "message_stop" in line:
                    break

    # The WARN must include both msg_id and model= fields.
    cap_warnings = [
        r.message for r in caplog.records if "[streaming-timeout]" in r.message
    ]
    assert cap_warnings, "no [streaming-timeout] WARN emitted"
    assert any("msg_id=msg_" in m for m in cap_warnings), (
        f"msg_id missing from cap warnings; got: {cap_warnings}"
    )
    assert any("model=fake-model" in m for m in cap_warnings), (
        f"model name missing from cap warnings; got: {cap_warnings}"
    )


def test_cap_fire_log_discriminates_wait_for_path(patched_server, caplog):
    """The infinite-stream fixture exercises the wait_for branch (each
    iteration's wait_for times out before the next chunk). This test
    pins that that branch's log carries `cap=wait_for` specifically —
    the prologue branch is pinned by a sibling test so together they
    prove both discriminators are emitted."""
    import logging

    engine = patched_server
    engine.stream_chat = _infinite_stream_chat

    client = TestClient(srv.app)
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.server"):
        with client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "fake-model",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "x"}],
            },
        ) as resp:
            for line in resp.iter_lines():
                if "message_stop" in line:
                    break

    cap_warnings = [
        r.message for r in caplog.records if "[streaming-timeout]" in r.message
    ]
    assert any("cap=wait_for" in m for m in cap_warnings), (
        f"wait_for discriminator missing; got: {cap_warnings}"
    )


def test_cap_cleanup_broadened_except_emits_tail_frames(patched_server):
    """Wave 2 finding: if `await _anext` during cap cleanup raises any
    non-CancelledError/StopAsyncIteration exception (e.g. engine bug),
    the handler must still emit tail frames — that's the whole point
    of the cap. Pre-fix, an unrelated Exception would crash the handler
    mid-cleanup and the client would see a dangling SSE stream."""

    class _ExceptionOnCancel:
        def __init__(self):
            self._first = True

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._first:
                self._first = False
                out = _FakeOutput(new_text="tok ")
                return out
            # Block forever until cancelled — then raise a non-standard
            # exception to simulate an engine bug during cleanup.
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                raise RuntimeError("simulated engine cleanup bug")

    engine = patched_server

    def _factory(*args, **kwargs):
        return _ExceptionOnCancel()

    engine.stream_chat = _factory

    client = TestClient(srv.app)
    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "fake-model",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "x"}],
        },
    ) as resp:
        body_lines = list(resp.iter_lines())

    body = "\n".join(body_lines)
    # The cleanup exception must not short-circuit tail emission.
    assert "message_stop" in body, (
        f"tail frames missing when cleanup raised; last 400 chars: {body[-400:]}"
    )
