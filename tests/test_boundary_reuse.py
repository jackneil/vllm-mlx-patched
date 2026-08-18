"""Regression tests for agentic multi-turn prefix-cache reuse.

The production failure: on hybrid-attention models (Qwen3.5/3.8), every turn
of a Claude Code conversation re-prefilled the full prompt. Chain:

1. ``_compute_prefix_boundary`` rendered with DEFAULT chat-template kwargs
   while requests rendered with ``enable_thinking=False``, shifting the
   boundary index out of the request's real token space.
2. The shifted boundary disabled the ``_boundary_segments`` split (or placed
   the boundary save past the true divergence point).
3. Later mid-prefill saves used ``evict_prefixes=True`` and evicted the
   boundary entry.
4. The next turn could then only LCP-match an entry containing turn-unique
   tokens; LCP requires trimming; hybrid recurrent state cannot be trimmed;
   the fetch MISSed and the whole prompt re-prefilled.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from vllm_mlx.memory_cache import MemoryAwarePrefixCache


class _KV:
    """Trimmable KV layer stand-in (has .offset and .keys)."""

    def __init__(self, n: int):
        self.offset = n
        self.keys = mx.zeros((1, 2, n, 8), dtype=mx.float16)
        self.values = mx.zeros((1, 2, n, 8), dtype=mx.float16)


class _Recurrent:
    """Hybrid linear-attention state stand-in (no .offset/.keys)."""

    def __init__(self):
        self.state = mx.zeros((1, 8, 8), dtype=mx.float16)


class _Model:
    pass


def _hybrid_cache(n: int):
    return [_KV(n), _Recurrent()]


BOUNDARY = 2_000
TURN1 = list(range(1, BOUNDARY + 1)) + [90_001, 90_002, 90_003]
TURN2 = list(range(1, BOUNDARY + 1)) + [91_001, 91_002, 91_003, 91_004]


def test_boundary_entry_prefix_matches_next_turn():
    """A boundary-aligned entry must PREFIX-match the next turn (no trim)."""
    cache = MemoryAwarePrefixCache(_Model())
    assert cache.store(TURN1[:BOUNDARY], _hybrid_cache(BOUNDARY))

    hit, remaining = cache.fetch(TURN2)
    assert hit is not None
    assert cache._last_match_type == "prefix"
    assert len(remaining) == len(TURN2) - BOUNDARY


def test_post_boundary_save_must_not_evict_boundary_entry():
    """Storing the full turn-1 prompt with evict_prefixes=False (the fixed
    post-boundary behavior) keeps the boundary entry fetchable by turn 2."""
    cache = MemoryAwarePrefixCache(_Model())
    assert cache.store(TURN1[:BOUNDARY], _hybrid_cache(BOUNDARY))
    assert cache.store(TURN1, _hybrid_cache(len(TURN1)), evict_prefixes=False)

    hit, remaining = cache.fetch(TURN2)
    assert hit is not None, (
        "boundary entry was evicted; turn 2 can only LCP-match a "
        "turn-unique entry, which hybrid caches cannot trim"
    )
    assert cache._last_match_type == "prefix"
    assert len(remaining) == len(TURN2) - BOUNDARY


def test_default_evict_prefixes_reproduces_the_regression():
    """Documents the failure mode: default evict_prefixes=True on the
    supersequence save evicts the boundary entry and turn 2 misses on a
    hybrid cache."""
    cache = MemoryAwarePrefixCache(_Model())
    assert cache.store(TURN1[:BOUNDARY], _hybrid_cache(BOUNDARY))
    assert cache.store(TURN1, _hybrid_cache(len(TURN1)))  # evicts the boundary

    hit, _ = cache.fetch(TURN2)
    assert hit is None
    assert cache._last_match_type == "miss"


def test_boundary_computation_uses_request_template_kwargs():
    """_compute_prefix_boundary must render real and dummy prompts with the
    request's chat_template_kwargs, not template defaults."""
    from vllm_mlx.engine.batched import BatchedEngine

    rendered_kwargs: list[dict | None] = []

    class _Tok:
        def apply_chat_template(self, msgs, **kw):
            rendered_kwargs.append(kw.get("enable_thinking"))
            parts = []
            for m in msgs:
                c = m["content"]
                if kw.get("enable_thinking") is False:
                    parts.append(f"[{m['role']}]{c}")
                else:
                    parts.append(f"[{m['role']}]<think skeleton>{c}")
            return "|".join(parts)

        def encode(self, text):
            return [ord(c) for c in text]

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._tokenizer = _Tok()
    engine._is_mllm = False
    engine._processor = None

    def _apply(messages, tools=None, num_images=0, chat_template_kwargs=None):
        kw = dict(chat_template_kwargs or {})
        return engine.tokenizer.apply_chat_template(messages, **kw)

    engine._apply_chat_template = _apply

    messages = [
        {"role": "user", "content": "context " * 50},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "question one"},
    ]

    rendered_kwargs.clear()
    boundary = engine._compute_prefix_boundary(
        messages, None, chat_template_kwargs={"enable_thinking": False}
    )
    assert boundary > 0
    assert rendered_kwargs == [False, False], (
        "both the real and dummy renders must carry the request's kwargs; "
        f"got {rendered_kwargs}"
    )

    # The boundary must be valid in the REQUEST's token space.
    real = engine._apply_chat_template(
        messages, None, chat_template_kwargs={"enable_thinking": False}
    )
    real_tokens = engine.tokenizer.encode(real)
    assert boundary <= len(real_tokens)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
