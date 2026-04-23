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


from vllm_mlx.server import _emit_block_close


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
