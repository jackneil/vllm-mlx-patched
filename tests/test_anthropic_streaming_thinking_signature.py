"""Streaming Anthropic thinking-signature tests.

Covers the shared `compute_thinking_signature()` helper, the
`_emit_block_close()` dispatcher, and all wired-in emitter behaviors.
"""

import hashlib
import json
import re

import pytest

from vllm_mlx.api.anthropic_adapter import (
    compute_thinking_signature,
    openai_to_anthropic,
)
from vllm_mlx.api.anthropic_models import AnthropicRequest
from vllm_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
)
from vllm_mlx.server import _emit_block_close, _emit_content_pieces

SIG_RE = re.compile(r"^vllm-mlx:[0-9a-f]{32}$")


def test_helper_signature_format_and_determinism():
    """Helper returns a stable `vllm-mlx:<32hex>` for any input; bytes
    match PR #14's formula (sha256 then first 32 hex chars)."""
    sig = compute_thinking_signature("Hello world")
    assert SIG_RE.match(sig), f"bad format: {sig!r}"
    expected = (
        "vllm-mlx:"
        + hashlib.sha256(b"Hello world").hexdigest()[:32]
    )
    assert sig == expected
    assert compute_thinking_signature("Hello world") == sig
    assert compute_thinking_signature("Hello World") != sig

    # Empty-string signature is the SHA-256-of-empty constant — a known,
    # deterministic value. Callers that emit this must do so intentionally.
    assert (
        compute_thinking_signature("")
        == "vllm-mlx:" + hashlib.sha256(b"").hexdigest()[:32]
    )


def _parse_events(events: list[str]) -> list[dict]:
    """Parse SSE event strings into {event, data} dicts for assertion."""
    out = []
    for raw in events:
        lines = raw.strip().split("\n")
        assert lines[0].startswith("event: "), raw
        assert lines[1].startswith("data: "), raw
        out.append(
            {
                "event": lines[0][len("event: "):],
                "data": json.loads(lines[1][len("data: "):]),
            }
        )
    return out


def test_emit_block_close_thinking_emits_signature_then_stop():
    """Dispatcher's thinking branch: signature_delta then content_block_stop
    on the same index, buffer cleared in place."""
    thinking_buffer = ["Hello", " world"]
    events = _emit_block_close(
        block_type="thinking",
        block_index=7,
        thinking_buffer=thinking_buffer,
    )
    parsed = _parse_events(events)
    assert len(parsed) == 2
    assert parsed[0]["data"] == {
        "type": "content_block_delta",
        "index": 7,
        "delta": {
            "type": "signature_delta",
            "signature": compute_thinking_signature("Hello world"),
        },
    }
    assert SIG_RE.match(parsed[0]["data"]["delta"]["signature"])
    assert parsed[1]["data"] == {"type": "content_block_stop", "index": 7}
    assert thinking_buffer == []


def test_emit_block_close_text_emits_bare_stop_and_does_not_touch_buffer():
    """Dispatcher's text branch: bare content_block_stop; thinking_buffer
    is not touched (even if non-empty from a prior block's leftovers —
    normal callers clear it before reaching here)."""
    thinking_buffer = ["leftover — caller should have cleared"]
    events = _emit_block_close(
        block_type="text",
        block_index=3,
        thinking_buffer=thinking_buffer,
    )
    parsed = _parse_events(events)
    assert len(parsed) == 1
    assert parsed[0]["data"] == {"type": "content_block_stop", "index": 3}
    assert thinking_buffer == ["leftover — caller should have cleared"]


def test_emit_block_close_empty_thinking_buffer_emits_known_empty_sig():
    """Defensive: an empty thinking buffer still emits a signature (the
    SHA-256-of-empty constant). This matches Anthropic's spec that signature
    is REQUIRED on thinking blocks, not conditional on non-empty content.
    Logs a one-line diagnostic but is not an error."""
    thinking_buffer: list[str] = []
    events = _emit_block_close(
        block_type="thinking",
        block_index=0,
        thinking_buffer=thinking_buffer,
    )
    parsed = _parse_events(events)
    assert parsed[0]["data"]["delta"]["type"] == "signature_delta"
    assert parsed[0]["data"]["delta"]["signature"] == compute_thinking_signature("")


def test_emit_block_close_raises_on_unknown_block_type():
    """Regression guard: a new block type added to _emit_content_pieces
    that forgets to teach _emit_block_close about itself gets a loud
    ValueError instead of a silent content-drop."""
    with pytest.raises(ValueError, match="unhandled block_type"):
        _emit_block_close(
            block_type="redacted_thinking",
            block_index=0,
            thinking_buffer=[],
        )


def test_thinking_to_text_transition_emits_signature_before_stop():
    """Mid-stream close: thinking→text transition emits signature_delta
    then content_block_stop then the new content_block_start, in that
    exact order, with block_index preserved on the close and advanced
    on the open."""
    thinking_buffer: list[str] = []
    events, new_type, new_idx = _emit_content_pieces(
        [("thinking", "Hello"), ("thinking", " world"), ("text", "OK")],
        current_block_type=None,
        block_index=0,
        thinking_buffer=thinking_buffer,
    )
    parsed = _parse_events(events)
    types = [(p["event"], p["data"].get("delta", {}).get("type")) for p in parsed]
    assert types == [
        ("content_block_start", None),
        ("content_block_delta", "thinking_delta"),
        ("content_block_delta", "thinking_delta"),
        ("content_block_delta", "signature_delta"),
        ("content_block_stop", None),
        ("content_block_start", None),
        ("content_block_delta", "text_delta"),
    ], f"unexpected sequence: {types}"

    sig_event = parsed[3]["data"]
    assert sig_event["index"] == 0, "signature_delta must use pre-increment index"
    assert SIG_RE.match(sig_event["delta"]["signature"])
    assert sig_event["delta"]["signature"] == compute_thinking_signature("Hello world")

    assert new_type == "text"
    assert new_idx == 1
    assert thinking_buffer == []


def test_text_only_stream_emits_no_signature_delta():
    """Non-regression: pure text stream never emits signature_delta and
    never touches the thinking_buffer."""
    thinking_buffer: list[str] = []
    events, new_type, new_idx = _emit_content_pieces(
        [("text", "hello "), ("text", "world")],
        current_block_type=None,
        block_index=0,
        thinking_buffer=thinking_buffer,
    )
    joined = "".join(events)
    assert "signature_delta" not in joined
    assert "signature" not in joined
    assert thinking_buffer == []
    assert new_type == "text"
    assert new_idx == 0


def test_interleaved_thinking_each_block_gets_distinct_signature():
    """Interleaved thinking (thinking, text, thinking, text) emits a
    distinct signature per thinking block; buffer clears between them."""
    thinking_buffer: list[str] = []
    events, _, final_idx = _emit_content_pieces(
        [
            ("thinking", "reasoning A"),
            ("text", "answer 1"),
            ("thinking", "reasoning B"),
            ("text", "answer 2"),
        ],
        current_block_type=None,
        block_index=0,
        thinking_buffer=thinking_buffer,
    )
    parsed = _parse_events(events)
    sig_events = [
        p for p in parsed
        if p["data"].get("delta", {}).get("type") == "signature_delta"
    ]
    assert len(sig_events) == 2, f"expected 2 sig events, got {len(sig_events)}"
    assert sig_events[0]["data"]["delta"]["signature"] == compute_thinking_signature("reasoning A")
    assert sig_events[1]["data"]["delta"]["signature"] == compute_thinking_signature("reasoning B")
    assert sig_events[0]["data"]["delta"]["signature"] != sig_events[1]["data"]["delta"]["signature"]
    assert thinking_buffer == []
    assert final_idx == 3


def test_thinking_alone_keeps_block_open_then_final_close_signs_it():
    """Covers BOTH contracts in one Task-3 test so Task 3's commit ships
    both the implementation (Step 3.5b final-close wire-up) and the test
    that exercises it — preserving TDD discipline at the commit level.

    (a) pieces ending on an open thinking block are NOT closed by
        _emit_content_pieces — that's _stream_anthropic_messages'
        final-close responsibility.
    (b) when the caller then invokes the shared dispatcher on the open
        block (mirroring what the final-close in Step 3.5b does),
        signature_delta + content_block_stop are emitted and the buffer
        is cleared.
    """
    thinking_buffer: list[str] = []
    events, new_type, new_idx = _emit_content_pieces(
        [("thinking", "reasoning, no follow-up")],
        current_block_type=None,
        block_index=0,
        thinking_buffer=thinking_buffer,
    )
    joined = "".join(events)

    # (a) Not closed by _emit_content_pieces.
    assert "content_block_stop" not in joined, (
        "_emit_content_pieces must not close the last block; final-close "
        "path in _stream_anthropic_messages owns that."
    )
    assert thinking_buffer == ["reasoning, no follow-up"]
    assert new_type == "thinking"
    assert new_idx == 0

    # (b) The Step 3.5b final-close call pattern.
    close_events = _emit_block_close(new_type, new_idx, thinking_buffer)
    close_parsed = _parse_events(close_events)
    assert [(p["event"], p["data"].get("delta", {}).get("type")) for p in close_parsed] == [
        ("content_block_delta", "signature_delta"),
        ("content_block_stop", None),
    ]
    assert (
        close_parsed[0]["data"]["delta"]["signature"]
        == compute_thinking_signature("reasoning, no follow-up")
    )
    assert thinking_buffer == []


# =============================================================================
# Task 4: Parity + end-to-end coroutine + edge-case tests
# =============================================================================


def _non_streaming_signature_for(thinking_text: str) -> str:
    response = ChatCompletionResponse(
        id="test-id",
        object="chat.completion",
        created=0,
        model="any-model",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(
                    role="assistant",
                    content="final",
                    reasoning=thinking_text,
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    a = openai_to_anthropic(response, model="any-model")
    thinking_blocks = [b for b in a.content if b.type == "thinking"]
    assert len(thinking_blocks) == 1 and thinking_blocks[0].signature
    return thinking_blocks[0].signature


def test_streaming_signature_matches_non_streaming_byte_for_byte():
    """Parity: same thinking text → same signature on both paths, AND
    the buffered text the streaming path signs == the reasoning text the
    non-streaming path signs (byte-for-byte, no whitespace normalization)."""
    text = "Let me think step by step: 13 * 17 = 221."
    pieces = [
        ("thinking", "Let me think "),
        ("thinking", "step by step: "),
        ("thinking", "13 * 17 = 221."),
        ("text", "221"),
    ]

    thinking_buffer: list[str] = []
    events, _, _ = _emit_content_pieces(
        pieces,
        current_block_type=None,
        block_index=0,
        thinking_buffer=thinking_buffer,
    )
    parsed = _parse_events(events)
    sig_events = [
        p for p in parsed if p["data"].get("delta", {}).get("type") == "signature_delta"
    ]
    assert len(sig_events) == 1
    streaming_sig = sig_events[0]["data"]["delta"]["signature"]
    non_streaming_sig = _non_streaming_signature_for(text)

    # Signature bytes match on both paths.
    assert streaming_sig == non_streaming_sig

    # Independently verify the streaming side signed exactly `text` —
    # not some joined-with-separator variant.
    assert streaming_sig == compute_thinking_signature(text)
    # And the piece-concatenation equals `text` byte-for-byte.
    assert "".join(t for (bt, t) in pieces if bt == "thinking") == text


def test_final_close_on_open_thinking_via_dispatcher():
    """The final-close path calls _emit_block_close on an open thinking
    block — same contract as the mid-stream close."""
    thinking_buffer = ["only thinking, no text follows"]
    events = _emit_block_close(
        block_type="thinking",
        block_index=3,
        thinking_buffer=thinking_buffer,
    )
    parsed = _parse_events(events)
    assert [(p["event"], p["data"].get("delta", {}).get("type")) for p in parsed] == [
        ("content_block_delta", "signature_delta"),
        ("content_block_stop", None),
    ]
    assert (
        parsed[0]["data"]["delta"]["signature"]
        == compute_thinking_signature("only thinking, no text follows")
    )
    assert thinking_buffer == []


def test_thinking_to_tool_use_transition_closes_with_signature():
    """Thinking followed by tool_use (no intervening text) — the thinking
    block still gets its signature_delta before content_block_stop. The
    tool_use block itself is opened separately by _stream_anthropic_messages'
    tool emission loop, which is unchanged; this test pins only the
    thinking-close half."""
    thinking_buffer: list[str] = []
    events, new_type, new_idx = _emit_content_pieces(
        [("thinking", "I should call the tool")],
        current_block_type=None,
        block_index=0,
        thinking_buffer=thinking_buffer,
    )
    assert new_type == "thinking"
    assert "signature_delta" not in "".join(events)

    close_events = _emit_block_close(
        block_type="thinking",
        block_index=new_idx,
        thinking_buffer=thinking_buffer,
    )
    parsed = _parse_events(close_events)
    assert parsed[0]["data"]["delta"]["type"] == "signature_delta"
    assert (
        parsed[0]["data"]["delta"]["signature"]
        == compute_thinking_signature("I should call the tool")
    )


# -----------------------------------------------------------------------------
# End-to-end _stream_anthropic_messages coroutine tests
# -----------------------------------------------------------------------------


class _FakeOutput:
    """Minimal GenerationOutput shim — _stream_anthropic_messages reads
    only .new_text, .prompt_tokens, .completion_tokens (with hasattr
    guards on the latter two)."""
    def __init__(self, new_text: str):
        self.new_text = new_text
        self.prompt_tokens = 0
        self.completion_tokens = 0


class _FakeEngine:
    """Minimal engine shim matching the three attributes
    _stream_anthropic_messages actually reads. Accepts the chunk list
    at construction; stream_chat yields them in order."""
    preserve_native_tool_format = False
    tokenizer = None  # hasattr-guarded downstream

    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    async def stream_chat(self, *, messages, **kwargs):
        for c in self._chunks:
            yield _FakeOutput(c)


async def _drive_stream(engine) -> str:
    """Helper: run _stream_anthropic_messages against `engine` with a
    trivial prompt and return the concatenated SSE wire bytes."""
    from vllm_mlx.server import _stream_anthropic_messages

    req_openai = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "x"}],
        stream=True,
    )
    req_anthropic = AnthropicRequest(
        model="test-model",
        max_tokens=1000,
        messages=[{"role": "user", "content": "x"}],
        stream=True,
    )

    pieces = []
    async for ev in _stream_anthropic_messages(engine, req_openai, req_anthropic):
        pieces.append(ev)
    return "".join(pieces)


@pytest.mark.asyncio
async def test_e2e_mid_stream_thinking_to_text_emits_signature():
    """End-to-end (a): thinking→text transition."""
    engine = _FakeEngine(["<think>", "hello ", "world", "</think>", "answer"])
    wire = await _drive_stream(engine)
    assert wire.count('"signature_delta"') == 1, (
        f"expected 1 signature_delta on mid-stream close, wire:\n{wire}"
    )
    expected_sig = compute_thinking_signature("hello world")
    assert expected_sig in wire, f"expected signature {expected_sig!r} not on wire"


@pytest.mark.asyncio
async def test_e2e_final_close_on_open_thinking_emits_signature():
    """End-to-end (b): stream terminates with an open thinking block."""
    engine = _FakeEngine(["<think>", "hello ", "world"])  # no </think>, no text
    wire = await _drive_stream(engine)
    assert wire.count('"signature_delta"') == 1, (
        f"expected 1 signature_delta from final-close, wire:\n{wire}"
    )
    expected_sig = compute_thinking_signature("hello world")
    assert expected_sig in wire, f"expected signature {expected_sig!r} not on wire"


# =============================================================================
# Task 5: Anthropic-spec conformance + interleaved-thinking integration
# =============================================================================


def test_emitted_events_conform_to_anthropic_streaming_shape():
    """Anthropic streaming spec (extended thinking): each event is one of
    {message_start, content_block_start, content_block_delta,
    content_block_stop, message_delta, message_stop, ping}. Signature
    arrives via content_block_delta with delta.type == signature_delta
    and a string delta.signature. This test asserts the shape WITHOUT
    referencing our own string literals, so it catches drift between our
    emitter and the spec.

    Reference: Anthropic Messages streaming docs §'Extended thinking'.
    If Anthropic moves signature onto content_block_start or introduces
    a new event type, this test fails."""
    ALLOWED_EVENT_TYPES = {  # noqa: N806 — function-local constant, uppercase for semantic clarity
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
        "ping",
        "error",  # Anthropic emits `error` events on server-side failures.
                  # Our emitter doesn't produce them today but the
                  # allowlist should accept them so this test doesn't
                  # flag a future error-event addition.
    }
    ALLOWED_DELTA_TYPES = {  # noqa: N806 — function-local constant, uppercase for semantic clarity
        "text_delta",
        "thinking_delta",
        "signature_delta",
        "input_json_delta",
    }

    thinking_buffer: list[str] = []
    events, _, _ = _emit_content_pieces(
        [("thinking", "Hello"), ("text", "OK")],
        current_block_type=None,
        block_index=0,
        thinking_buffer=thinking_buffer,
    )
    parsed = _parse_events(events)

    for p in parsed:
        # Event type must be Anthropic-spec-known.
        assert p["event"] in ALLOWED_EVENT_TYPES, (
            f"unknown event type {p['event']!r}; not in Anthropic spec"
        )
        # content_block_delta carries a typed delta.
        if p["event"] == "content_block_delta":
            delta = p["data"].get("delta")
            assert isinstance(delta, dict), f"no delta: {p}"
            assert delta.get("type") in ALLOWED_DELTA_TYPES, (
                f"unknown delta type {delta.get('type')!r}"
            )
            # signature_delta carries a string `signature` field.
            if delta["type"] == "signature_delta":
                sig = delta.get("signature")
                assert isinstance(sig, str) and len(sig) > 0


def test_interleaved_thinking_via_emit_content_pieces():
    """Interleaved-thinking across multiple _emit_content_pieces calls
    (simulating deltas arriving in batches over many coroutine ticks).
    Each thinking block must get its own signature at transition time."""
    thinking_buffer: list[str] = []
    state = {"current_block_type": None, "block_index": 0}
    all_events: list[str] = []

    def drive(pieces):
        events, bt, bi = _emit_content_pieces(
            pieces,
            current_block_type=state["current_block_type"],
            block_index=state["block_index"],
            thinking_buffer=thinking_buffer,
        )
        state["current_block_type"] = bt
        state["block_index"] = bi
        all_events.extend(events)

    drive([("thinking", "first reasoning")])
    drive([("text", "first answer")])
    drive([("thinking", "second reasoning")])
    drive([("text", "second answer")])

    # Close whatever is still open via dispatcher.
    if state["current_block_type"] is not None:
        all_events.extend(
            _emit_block_close(
                state["current_block_type"],
                state["block_index"],
                thinking_buffer,
            )
        )

    parsed = _parse_events(all_events)
    sig_events = [
        p for p in parsed if p["data"].get("delta", {}).get("type") == "signature_delta"
    ]
    assert len(sig_events) == 2
    assert sig_events[0]["data"]["delta"]["signature"] == compute_thinking_signature("first reasoning")
    assert sig_events[1]["data"]["delta"]["signature"] == compute_thinking_signature("second reasoning")


def test_emit_block_close_increments_signature_counter():
    """/dc M4: the dispatcher must increment thinking_signature_emitted_total
    exactly once per thinking-block close. Guards against a refactor that
    drops or double-counts the inc() call — the counter is the only
    operational signal that the streaming contract is live in prod."""
    from vllm_mlx.metrics import thinking_signature_emitted_total

    initial = thinking_signature_emitted_total.value

    # Close a thinking block.
    _emit_block_close("thinking", 0, ["x"])
    assert thinking_signature_emitted_total.value == initial + 1

    # Close a text block — counter must NOT increment.
    _emit_block_close("text", 1, [])
    assert thinking_signature_emitted_total.value == initial + 1

    # Close another thinking block — counter increments again.
    _emit_block_close("thinking", 2, ["y"])
    assert thinking_signature_emitted_total.value == initial + 2


def test_empty_buffer_close_logs_info_with_msg_id(caplog):
    """/dc M3 + PM-2: empty thinking buffer at close fires an INFO log
    that includes msg_id so on-call can correlate unexpected SHA256-of-empty
    signatures (`vllm-mlx:e3b0c44298fc1c149afbf4c8996fb924`) back to the
    user-facing request."""
    import logging

    caplog.set_level(logging.INFO, logger="vllm_mlx.server")

    thinking_buffer: list[str] = []
    events = _emit_block_close(
        "thinking",
        7,
        thinking_buffer,
        msg_id="msg_deadbeef00000000000000",
    )

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    info_log = next(
        (m for m in messages if "[streaming-signature]" in m and "empty buffer" in m),
        None,
    )
    assert info_log is not None, f"empty-buffer INFO log not fired; got: {messages}"
    assert "msg_id=msg_deadbeef00000000000000" in info_log, info_log
    assert "idx=7" in info_log, info_log

    parsed = _parse_events(events)
    assert parsed[0]["data"]["delta"]["signature"] == compute_thinking_signature("")


def test_empty_buffer_close_with_none_msg_id_uses_unknown_placeholder(caplog):
    """/dc PM-2: when msg_id is not provided (unit tests, or a legacy
    caller), the log line uses `<unknown>` rather than crashing or
    emitting None."""
    import logging

    caplog.set_level(logging.INFO, logger="vllm_mlx.server")

    thinking_buffer: list[str] = []
    _emit_block_close("thinking", 0, thinking_buffer)  # no msg_id kwarg

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    info_log = next(
        (m for m in messages if "[streaming-signature]" in m and "empty buffer" in m),
        None,
    )
    assert info_log is not None
    assert "msg_id=<unknown>" in info_log, info_log


def test_compute_thinking_signature_bounded_latency_realistic_input():
    """/dc PM-1: guards against pathological latency in compute_thinking_signature
    on realistic-max inputs. In prod, `--max-thinking-token-budget` (Invariant 12)
    caps thinking-token budgets; a typical max (32k tokens × ~5 chars/tok) is
    ≈160 KB of text — sha256 on that should complete in single-digit ms on
    an M1/M2-class box.

    If this test fails, either:
    (a) max-thinking-token-budget ceiling is being bypassed in prod, OR
    (b) compute_thinking_signature has taken on unexpected work beyond the
        sha256 hash (e.g., a normalize/canonicalize step was added).
    Either way: investigate before shipping.

    Flakiness guard: 5 samples, assert MEDIAN — robust to single-shot GC
    pauses or scheduler hiccups on contended CI runners. Threshold is 500ms
    (100x expected) so a real order-of-magnitude regression still trips it
    while transient spikes don't.
    """
    import statistics
    import time

    text = "x" * 200_000  # ~200 KB — above the realistic 160 KB cap.

    samples_ms = []
    for _ in range(5):
        start = time.perf_counter()
        sig = compute_thinking_signature(text)
        samples_ms.append((time.perf_counter() - start) * 1000)
        assert SIG_RE.match(sig), f"bad signature format: {sig!r}"

    median_ms = statistics.median(samples_ms)
    assert median_ms < 500, (
        f"compute_thinking_signature on 200 KB median latency {median_ms:.1f}ms "
        f"(samples {[f'{s:.1f}' for s in samples_ms]}ms) — expected <500ms. "
        f"Either the ceiling was removed (PM-1 vulnerability) or the helper "
        f"picked up extra work beyond sha256. Investigate."
    )
