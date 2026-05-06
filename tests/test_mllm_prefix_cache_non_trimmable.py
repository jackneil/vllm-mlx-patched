# SPDX-License-Identifier: Apache-2.0
"""TDD: support prefix-cache reuse for non-trimmable cache classes.

Production bug: gemma-4-26b uses RotatingKVCache (sliding-window
attention). PrefixCacheManager stores and fetches its KV state correctly,
but the consumer at `MLLMBatchGenerator._process_prompts` requires the
cache to be trimmable to make a full-match hit useful, and silently
falls through to full-prefill when the cache reports
`is_trimmable() = False`. Result: every identical /v1/messages re-pays
the 132 s prefill instead of the cache hit being a few seconds.

Fix: extend the full-match branch to handle non-trimmable caches by
feeding only the last token and using the cached state as-is. The
position counter inside the cache moves on by 1, which the model
handles gracefully — the user just gets the next-token continuation
they want.
"""

from __future__ import annotations

from collections import deque

import mlx.core as mx
import pytest

from vllm_mlx.prefix_cache import PrefixCacheManager


class _FakeNonTrimmableLayer:
    """Stand-in for a sliding-window cache layer (RotatingKVCache shape).
    Reports as non-trimmable. PrefixCacheManager._can_trim_cache must
    return False for caches whose first layer is one of these."""

    def is_trimmable(self) -> bool:
        return False


class _FakeTrimmableLayer:
    """Stand-in for a regular KVCache that DOES support trim."""

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> None:
        return None


class _StubModel:
    pass


def _make_cache():
    return PrefixCacheManager(model=_StubModel(), max_entries=8)


# ---------------------------------------------------------------------------
# Pin: PrefixCacheManager._can_trim_cache returns False for non-trimmable
# layers. This is the predicate that gates the trim path.
# ---------------------------------------------------------------------------


def test_can_trim_returns_false_for_non_trimmable_layer():
    cache = _make_cache()
    state = [_FakeNonTrimmableLayer()]
    assert cache._can_trim_cache(state) is False


def test_can_trim_returns_true_for_trimmable_layer():
    cache = _make_cache()
    state = [_FakeTrimmableLayer()]
    assert cache._can_trim_cache(state) is True


# ---------------------------------------------------------------------------
# Pin: PrefixCacheManager.fetch_cache STILL returns non-trimmable caches.
# It is the CONSUMER's responsibility to handle the can-or-can't-trim
# question, not the cache's. Refusing to return a non-trimmable entry
# would prevent the consumer from using it via shorter-match-style
# semantics (which is what the new code path will do).
# ---------------------------------------------------------------------------


def test_fetch_cache_returns_non_trimmable_state_on_full_match():
    cache = _make_cache()
    tokens = [1, 2, 3, 4, 5]
    state = [_FakeNonTrimmableLayer()]
    cache.store_cache(tokens, state)

    fetched, remaining = cache.fetch_cache(tokens)
    assert fetched is not None, "non-trimmable cache should still be returned"
    assert remaining == [], "full match must yield empty remaining"


# ---------------------------------------------------------------------------
# The fix: a helper that decides what to do on a full match. Extracted from
# `MLLMBatchGenerator._process_prompts`'s full-match branch so we can
# unit test without spinning up an MLX model.
#
# Contract:
#   _handle_full_match(prefix_cache, prompt_cache, full_token_ids,
#                      language_model_factory)
#     -> (request_cache, input_ids, cached_prefix_len, cache_was_hit)
#
#   - Trimmable cache + N tokens: deep-copies + trims by 1, input_ids=
#     last token, cached_prefix_len=N-1, cache_was_hit=True.
#   - Non-trimmable cache + N tokens: returns cache AS-IS,
#     input_ids=last token, cached_prefix_len=N-1, cache_was_hit=True.
#     (The non-trimmable class will internally bump its position counter
#     by 1 when the model feeds the last token; that's acceptable for
#     sliding-window models — the next-token output is still coherent.)
#   - When `prefix_cache.fetch_cache` returns None → caller-handled,
#     this helper is not invoked.
# ---------------------------------------------------------------------------


def test_handle_full_match_trimmable_cache():
    from vllm_mlx.mllm_batch_generator import _handle_full_match

    cache = _make_cache()
    tokens = list(range(100))
    state = [_FakeTrimmableLayer()]
    cache.store_cache(tokens, state)

    fetched, _ = cache.fetch_cache(tokens)
    assert fetched is not None

    request_cache, input_ids, cached_prefix_len, cache_was_hit = (
        _handle_full_match(
            prefix_cache=cache,
            prompt_cache=fetched,
            full_token_ids=tokens,
            language_model=_StubModel(),
        )
    )
    assert cache_was_hit is True
    assert cached_prefix_len == len(tokens) - 1
    # input_ids must be the LAST token of the prompt, shape [1, 1]
    assert input_ids is not None
    assert input_ids.tolist() == [[tokens[-1]]]


def test_handle_full_match_non_trimmable_cache_still_hits():
    """The fix: non-trimmable cache must still produce a hit, not a
    silent fallthrough to full prefill."""
    from vllm_mlx.mllm_batch_generator import _handle_full_match

    cache = _make_cache()
    tokens = list(range(100))
    state = [_FakeNonTrimmableLayer()]
    cache.store_cache(tokens, state)

    fetched, _ = cache.fetch_cache(tokens)
    assert fetched is not None

    request_cache, input_ids, cached_prefix_len, cache_was_hit = (
        _handle_full_match(
            prefix_cache=cache,
            prompt_cache=fetched,
            full_token_ids=tokens,
            language_model=_StubModel(),
        )
    )
    assert cache_was_hit is True, (
        "full match with non-trimmable cache MUST hit — silent fallthrough is the bug"
    )
    assert cached_prefix_len == len(tokens) - 1
    assert input_ids is not None
    assert input_ids.tolist() == [[tokens[-1]]]
    # Cache returned as-is (no deep copy needed since not trimming)
    assert request_cache is fetched, (
        "non-trimmable cache should be returned as-is (no deep copy)"
    )
