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
