# SPDX-License-Identifier: Apache-2.0
"""Tests for the `fold_thinking_as_text` emitter option.

Anthropic's thinking content block is a first-class SSE block type
(`content_block_start` with `type: "thinking"` + `thinking_delta` events
+ `signature_delta` + `content_block_stop`). Most clients handle it fine.

Claude Code's SDK (v2.1.118 observed) has a bug where a `thinking` block
followed by a `text` block causes the text block's AssistantMessage to be
dropped from the SDK output, leaving `.result = ''`. The wire bytes are
correct — the SDK's dual-block assembly drops the second message.

Workaround: fold `('thinking', text)` pieces into a single `text` block
using inline `<think>...</think>` tags. Claude Code sees one text block,
which it handles correctly. Clients that want the spec-correct thinking
block can leave the flag off.

This matches the pre-PR-#22 emission shape on Qwen3 (when thinking was
folded into text as `<think>...</think>`) — it's a documented, known-good
client-compatibility pattern that restores Claude Code support without
breaking spec-correct clients.

The flag is plumbed as a per-request option so proxies (e.g.
hank-secure-llm's Anthropic proxy) can toggle it based on the downstream
client family, rather than forcing a global server choice.
"""

import json

import pytest

from vllm_mlx.api.anthropic_models import AnthropicRequest
from vllm_mlx.server import _emit_content_pieces


def _parse_events(events: list[str]) -> list[dict]:
    """Parse SSE event strings into {event, data} dicts for assertion."""
    out = []
    for raw in events:
        lines = raw.strip().split("\n")
        assert lines[0].startswith("event: "), raw
        assert lines[1].startswith("data: "), raw
        out.append(
            {
                "event": lines[0][len("event: ") :],
                "data": json.loads(lines[1][len("data: ") :]),
            }
        )
    return out


def _collect_text_content(parsed: list[dict]) -> str:
    """Concatenate all text_delta text fields in order."""
    return "".join(
        e["data"]["delta"]["text"]
        for e in parsed
        if e["data"].get("type") == "content_block_delta"
        and e["data"].get("delta", {}).get("type") == "text_delta"
    )


# ---------------------------------------------------------------------------
# The failing test: fold_thinking=True should produce a single text block
# ---------------------------------------------------------------------------


def test_fold_thinking_produces_single_text_block():
    """With fold_thinking=True and mixed thinking+text pieces, the emitter
    produces exactly ONE content_block_start (type=text), no thinking
    block start, no signature_delta, no thinking_delta. All content goes
    through text_delta events with <think>...</think> wrapping the
    reasoning portion."""
    pieces = [("thinking", "Let me think..."), ("text", "OK")]
    thinking_buffer: list[str] = []
    events, final_block_type, final_block_index = _emit_content_pieces(
        pieces,
        current_block_type=None,
        block_index=0,
        thinking_buffer=thinking_buffer,
        fold_thinking=True,
    )
    parsed = _parse_events(events)

    # Exactly one content_block_start, of type=text
    starts = [e for e in parsed if e["data"]["type"] == "content_block_start"]
    assert len(starts) == 1, f"expected 1 start, got {len(starts)}: {parsed}"
    assert starts[0]["data"]["content_block"]["type"] == "text", parsed

    # No signature_delta, no thinking_delta
    for e in parsed:
        delta_type = e["data"].get("delta", {}).get("type")
        assert delta_type != "signature_delta", f"unexpected sig_delta: {parsed}"
        assert delta_type != "thinking_delta", f"unexpected thinking_delta: {parsed}"

    # Text content is think-wrapped reasoning + the plain text
    assert _collect_text_content(parsed) == "<think>Let me think...</think>OK"


def test_fold_thinking_multiple_thinking_chunks_single_wrapper():
    """Multiple consecutive thinking pieces should produce a single
    <think>...</think> wrapper, not one per piece."""
    pieces = [
        ("thinking", "Analyzing... "),
        ("thinking", "step one. "),
        ("thinking", "step two."),
        ("text", "Done."),
    ]
    events, _, _ = _emit_content_pieces(
        pieces,
        current_block_type=None,
        block_index=0,
        thinking_buffer=[],
        fold_thinking=True,
    )
    parsed = _parse_events(events)
    text = _collect_text_content(parsed)
    assert text == "<think>Analyzing... step one. step two.</think>Done."
    assert text.count("<think>") == 1
    assert text.count("</think>") == 1


def test_fold_thinking_only_thinking_leaves_block_open_for_final_close():
    """If pieces contain only thinking (no text yet), the text block stays
    open — caller's final-close is responsible for emitting </think> and
    content_block_stop. The emitter shouldn't close the block early."""
    pieces = [("thinking", "Half-finished reasoning...")]
    events, final_block_type, final_block_index = _emit_content_pieces(
        pieces,
        current_block_type=None,
        block_index=0,
        thinking_buffer=[],
        fold_thinking=True,
    )
    parsed = _parse_events(events)
    # Block is still text (fold_thinking maps thinking → text)
    assert final_block_type == "text"
    # No content_block_stop — caller handles that
    assert not any(
        e["data"]["type"] == "content_block_stop" for e in parsed
    ), parsed
    # The emitted text_delta opens <think> but doesn't close it
    text = _collect_text_content(parsed)
    assert text == "<think>Half-finished reasoning..."
    assert "<think>" in text
    assert "</think>" not in text


def test_fold_thinking_off_preserves_current_behavior():
    """With fold_thinking=False (or omitted), behavior is unchanged —
    thinking still gets its own content_block_start with type=thinking."""
    pieces = [("thinking", "foo"), ("text", "bar")]
    events, _, _ = _emit_content_pieces(
        pieces,
        current_block_type=None,
        block_index=0,
        thinking_buffer=[],
        # fold_thinking omitted — default False
    )
    parsed = _parse_events(events)
    types = [
        e["data"]["content_block"]["type"]
        for e in parsed
        if e["data"]["type"] == "content_block_start"
    ]
    assert types == ["thinking", "text"]


def test_fold_thinking_final_close_emits_closing_tag_if_still_open():
    """When `_emit_block_close` is called on a text block whose fold-mode
    wrapper is still open (thinking_buffer has the sentinel), it must
    emit a final `text_delta` carrying `</think>` BEFORE the
    content_block_stop. Otherwise the client sees an unbalanced
    `<think>` in its result field."""
    from vllm_mlx.server import _emit_block_close

    # Simulate: emitter is mid-<think>, no text piece ever arrived.
    pieces = [("thinking", "Mid-stream reasoning that was cut off")]
    thinking_buffer: list[str] = []
    events, block_type, block_index = _emit_content_pieces(
        pieces,
        current_block_type=None,
        block_index=0,
        thinking_buffer=thinking_buffer,
        fold_thinking=True,
    )
    # Now final-close: the wrapper is still open (buffer has sentinel).
    assert thinking_buffer, "fold mode must keep sentinel so close can detect"
    close_events = _emit_block_close(
        block_type,
        block_index,
        thinking_buffer,
        fold_thinking=True,
    )
    all_parsed = _parse_events(events + close_events)

    # Last two events should be: text_delta with "</think>", then stop.
    closing_deltas = [
        e for e in all_parsed
        if e["data"].get("type") == "content_block_delta"
        and e["data"].get("delta", {}).get("type") == "text_delta"
    ]
    assert closing_deltas[-1]["data"]["delta"]["text"] == "</think>", all_parsed

    stops = [e for e in all_parsed if e["data"].get("type") == "content_block_stop"]
    assert len(stops) == 1, all_parsed

    # No signature_delta in fold mode's close path.
    for e in all_parsed:
        assert e["data"].get("delta", {}).get("type") != "signature_delta"

    # thinking_buffer cleared after close.
    assert thinking_buffer == [], "close must clear the fold sentinel"


def test_fold_thinking_final_close_no_open_wrapper_is_bare_stop():
    """Final close on a fold-mode text block whose wrapper already closed
    (buffer empty) should be a plain content_block_stop — no stray
    </think>."""
    from vllm_mlx.server import _emit_block_close

    pieces = [("thinking", "reason"), ("text", "answer")]
    thinking_buffer: list[str] = []
    events, block_type, block_index = _emit_content_pieces(
        pieces,
        current_block_type=None,
        block_index=0,
        thinking_buffer=thinking_buffer,
        fold_thinking=True,
    )
    assert thinking_buffer == [], "wrapper should have closed after text piece"
    close_events = _emit_block_close(
        block_type,
        block_index,
        thinking_buffer,
        fold_thinking=True,
    )
    close_parsed = _parse_events(close_events)
    assert len(close_parsed) == 1
    assert close_parsed[0]["data"]["type"] == "content_block_stop"


def test_anthropic_request_accepts_fold_thinking_in_output_config():
    """The request schema plumbs `output_config.fold_thinking_as_text`
    through validation without rejecting it."""
    req = AnthropicRequest(
        model="mlx-community/whatever",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        output_config={"effort": "high", "fold_thinking_as_text": True},
    )
    assert req.output_config["fold_thinking_as_text"] is True
    # Omitted is fine
    req2 = AnthropicRequest(
        model="mlx-community/whatever",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        output_config={"effort": "high"},
    )
    assert "fold_thinking_as_text" not in req2.output_config


def test_anthropic_request_rejects_non_bool_fold_thinking():
    """Non-bool values for fold_thinking_as_text are rejected at ingress
    rather than silently coerced — prevents truthy/falsy surprises."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        AnthropicRequest(
            model="mlx-community/whatever",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
            output_config={"fold_thinking_as_text": "yes"},
        )


def test_fold_thinking_transition_from_text_to_thinking_and_back():
    """When fold_thinking=True and pieces go text → thinking → text,
    the wrappers should open/close in the middle of the text stream."""
    pieces = [
        ("text", "Before. "),
        ("thinking", "Aside: reasoning."),
        ("text", " After."),
    ]
    events, _, _ = _emit_content_pieces(
        pieces,
        current_block_type=None,
        block_index=0,
        thinking_buffer=[],
        fold_thinking=True,
    )
    parsed = _parse_events(events)
    # One content_block_start (text)
    starts = [e for e in parsed if e["data"]["type"] == "content_block_start"]
    assert len(starts) == 1 and starts[0]["data"]["content_block"]["type"] == "text"
    text = _collect_text_content(parsed)
    assert text == "Before. <think>Aside: reasoning.</think> After."
