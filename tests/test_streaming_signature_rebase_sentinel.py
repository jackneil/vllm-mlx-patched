"""Rebase sentinel for Invariant 13 (streaming signature contract).

If a rebase reverts the signature emission in _emit_content_pieces
(restoring the pre-fix bare content_block_stop path), this test fires
with a loud error pointing at UPSTREAM_PIN.md. The pattern mirrors
tests/test_reasoning_parser_properties.py::TestRebaseBreakageSentinel
and tests/test_mlx_lm_api_contract.py.
"""

import ast
import inspect

from vllm_mlx.server import (
    _emit_block_close,
    _emit_content_pieces,
    _stream_anthropic_messages,
)
from vllm_mlx.api.anthropic_adapter import compute_thinking_signature


def test_compute_thinking_signature_helper_exists():
    """Invariant 13: shared helper `compute_thinking_signature` is the
    single source of truth for the signature formula. Rebase must preserve
    both its existence and its name (server.py imports by name)."""
    assert callable(compute_thinking_signature)
    sig = compute_thinking_signature("")
    assert sig.startswith("vllm-mlx:") and len(sig) == len("vllm-mlx:") + 32


def test_emit_block_close_dispatcher_exists():
    """Invariant 13: `_emit_block_close` dispatches on block_type with a
    `thinking` branch that emits signature_delta. Rebase must preserve
    the dispatcher and its name."""
    assert callable(_emit_block_close)
    events = _emit_block_close("thinking", 0, ["x"])
    assert len(events) == 2
    assert "signature_delta" in events[0]


def test_emit_content_pieces_invokes_emit_block_close():
    """Invariant 13 rebase guard: _emit_content_pieces MUST call
    _emit_block_close when transitioning off a thinking block. AST walk
    of the function source asserts the call-by-name is present. If a
    rebase restores the old bare-stop pattern, this test fails with a
    pointer to UPSTREAM_PIN.md."""
    source = inspect.getsource(_emit_content_pieces)
    tree = ast.parse(source)
    call_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            call_names.add(node.func.id)
    assert "_emit_block_close" in call_names, (
        "Invariant 13 (streaming) regression: _emit_content_pieces no "
        "longer calls _emit_block_close. A rebase likely restored the "
        "pre-fix bare content_block_stop pattern — Anthropic streaming "
        "thinking blocks will emit no signature, and Claude Code will "
        "silently drop the text that follows. See UPSTREAM_PIN.md "
        "invariant 13 and docs/testing/"
        "2026-04-23-qwen3-streaming-thinking-missing-signature-silent-text-drop.md."
    )


def test_stream_anthropic_messages_final_close_invokes_emit_block_close():
    """Invariant 13 rebase guard: the final-close in _stream_anthropic_messages
    MUST also route through _emit_block_close so a stream that terminates
    with an open thinking block still emits a signature."""
    source = inspect.getsource(_stream_anthropic_messages)
    assert "_emit_block_close" in source, (
        "Invariant 13 (streaming) regression: _stream_anthropic_messages "
        "no longer references _emit_block_close in its final-close path. "
        "A stream ending on an open thinking block will emit no signature. "
        "See UPSTREAM_PIN.md invariant 13."
    )
