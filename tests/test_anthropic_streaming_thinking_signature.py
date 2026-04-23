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
from vllm_mlx.server import _emit_block_close, _emit_content_pieces


SIG_RE = re.compile(r"^vllm-mlx:[0-9a-f]{32}$")


def test_helper_signature_format_and_determinism():
    """Helper returns a stable `vllm-mlx:<32hex>` for any input; bytes
    match PR #14's formula (sha256 then first 32 hex chars)."""
    sig = compute_thinking_signature("Hello world")
    assert SIG_RE.match(sig), f"bad format: {sig!r}"
    expected = (
        "vllm-mlx:"
        + hashlib.sha256("Hello world".encode("utf-8")).hexdigest()[:32]
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
