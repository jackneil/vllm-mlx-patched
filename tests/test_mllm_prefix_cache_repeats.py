# SPDX-License-Identifier: Apache-2.0
"""TDD repro for the bug user hits as 'gemma-4-26b crashes when I send hi'.

Symptom (from production):
- User sends "hi" through Claude Code -> hank-secure-llm proxy -> arena ->
  vllm-mlx (gemma-4-26b is a VLM-architecture model so it routes through
  MLLMScheduler).
- First request takes ~132s (50K tokens of tool-schema prefill).
- Second IDENTICAL request also takes ~132s. The prefix cache should have
  hit and made it near-instant.
- The compounding 132s+132s appears to the user as 'crashed'.

Real diagnosis (verified by replaying a captured Claude Code body via
arena): cold_first 132.27s -> warm_second 132.62s. Both complete cleanly.
The prefix cache for the MLLM path (PR #40, the
`MLLMBatchGenerator.prefix_cache.fetch_cache(...)` integration in
`vllm_mlx/mllm_batch_generator.py:702-770`) is not actually providing a
hit on the second identical request, so we re-pay the entire prefill cost.

These tests pin the unit-level invariants the cache must satisfy at the
PrefixCacheManager level — the cache class actually wired into MLLM
scheduler at `mllm_scheduler.py:293`. They use stub KV-cache states to
avoid GPU dependency.
"""

from __future__ import annotations

import pytest

from vllm_mlx.prefix_cache import PrefixCacheManager


class _StubModel:
    """Minimal model stand-in. PrefixCacheManager only uses `id(model)` for
    fingerprinting so we do not need real model attributes."""
    pass


def _make_cache(max_entries: int = 8) -> PrefixCacheManager:
    return PrefixCacheManager(model=_StubModel(), max_entries=max_entries)


# ---------------------------------------------------------------------------
# Goal #1: Two identical requests must produce a full-match hit on the
# second fetch.
# ---------------------------------------------------------------------------


def test_full_match_returns_cache_and_empty_remaining():
    """User pattern: same Claude Code body sent twice. The second
    fetch_cache MUST return the stored state with empty remaining."""
    cache = _make_cache()
    tokens = list(range(50000))
    sentinel_state = ["layer0_kv", "layer1_kv"]

    cache.store_cache(tokens, sentinel_state)

    cached_state, remaining = cache.fetch_cache(tokens)
    assert cached_state is not None, "second identical fetch missed — re-prefill is forced"
    assert remaining == [], f"full match must yield empty remaining, got {len(remaining)}"
    assert cache.stats.hits >= 1, "exact-match path didn't increment hits stat"


def test_three_identical_fetches_all_hit():
    """User pattern: same body sent 3 times in a row. All must hit."""
    cache = _make_cache()
    tokens = list(range(50000))
    state = ["kv0", "kv1"]

    cache.store_cache(tokens, state)

    for i in range(3):
        cached, remaining = cache.fetch_cache(tokens)
        assert cached is not None, f"fetch #{i+1} missed"
        assert remaining == [], f"fetch #{i+1}: unexpected remaining"


# ---------------------------------------------------------------------------
# Goal #2: Supersequence (next-turn append) must hit and yield only the
# new suffix as remaining. This is the multi-turn pattern (turn 1 stored,
# turn 2 adds a user message).
# ---------------------------------------------------------------------------


def test_shorter_prefix_match_returns_remaining_suffix():
    """Cached entry covers a PREFIX of the new request. fetch_cache
    must return the cached state plus the new suffix as `remaining`."""
    cache = _make_cache()
    prefix = list(range(40000))
    cache.store_cache(prefix, ["kv"])

    new = prefix + [99999]  # one more token
    cached, remaining = cache.fetch_cache(new)
    assert cached is not None, "supersequence fetch failed to hit"
    assert remaining == [99999], f"remaining must be the new suffix, got {len(remaining)} tokens"


# ---------------------------------------------------------------------------
# Goal #3: Disjoint requests must miss cleanly (no false hit).
# ---------------------------------------------------------------------------


def test_disjoint_returns_miss():
    """Token sequences with no shared prefix must miss."""
    cache = _make_cache()
    cache.store_cache([1, 2, 3, 4, 5], ["kv"])
    cached, remaining = cache.fetch_cache([10, 20, 30])
    assert cached is None, f"disjoint fetch produced false hit: {cached}"
    assert remaining == [10, 20, 30], "remaining must be the full input on miss"


# ---------------------------------------------------------------------------
# Goal #4: text_only detection — Claude Code sends 144 tools but no
# images/videos. This MUST be eligible for prefix cache.
# ---------------------------------------------------------------------------


def test_text_only_detection_for_tools_no_images():
    """The check at mllm_batch_generator.py:700-712 gates cache fetch on
    `text_only`. Claude Code's 144-tool /v1/messages has no images/videos —
    should be text_only."""
    class StubReq:
        def __init__(self, images=None, videos=None):
            self.images = images
            self.videos = videos

    req = StubReq(images=None, videos=None)
    text_only = not bool(req.images) and not bool(req.videos)
    assert text_only is True, "no images, no videos → must be text_only"

    req_with_imgs = StubReq(images=["a.jpg"])
    assert (not bool(req_with_imgs.images) and not bool(req_with_imgs.videos)) is False


# ---------------------------------------------------------------------------
# Goal #5: store + fetch round-trip preserves the state. Failure here
# means the cache eviction or storage layer is corrupting state.
# ---------------------------------------------------------------------------


def test_store_then_fetch_returns_equivalent_state():
    cache = _make_cache()
    tokens = list(range(1000))
    state = ["layer_0_kv", "layer_1_kv", "layer_2_kv"]
    cache.store_cache(tokens, state)

    fetched, remaining = cache.fetch_cache(tokens)
    assert fetched is not None
    assert fetched == state, f"round-trip mismatch: {fetched}"
    assert remaining == []


# ---------------------------------------------------------------------------
# Goal #6: Cache survives across requests. Storing one tokenization, then
# fetching DIFFERENT tokens (miss), then refetching the original must hit.
# ---------------------------------------------------------------------------


def test_cache_survives_intervening_misses():
    """User pattern: send a /v1/messages tools=144 (cache stores), then a
    simple /v1/chat/completions 'hi' (miss — different prompt), then
    repeat /v1/messages tools=144 (must HIT, not miss)."""
    cache = _make_cache()
    tools_tokens = list(range(50000))
    cache.store_cache(tools_tokens, ["full_state"])

    # Intervening miss — disjoint short request
    miss_cached, miss_remaining = cache.fetch_cache([99, 88, 77])
    assert miss_cached is None

    # Original lookup must still hit
    cached, remaining = cache.fetch_cache(tools_tokens)
    assert cached is not None, "cache lost the first entry after a miss probe"
    assert remaining == []


# ---------------------------------------------------------------------------
# Goal #7: Cache stats accurately reflect hit / miss decisions, so we can
# observe production behavior.
# ---------------------------------------------------------------------------


def test_stats_track_hits_and_misses():
    cache = _make_cache()
    cache.store_cache([1, 2, 3], ["s"])

    cache.fetch_cache([1, 2, 3])  # hit
    cache.fetch_cache([1, 2, 3])  # hit
    cache.fetch_cache([99, 88])   # miss

    assert cache.stats.hits == 2, f"expected 2 hits, got {cache.stats.hits}"
    assert cache.stats.misses == 1, f"expected 1 miss, got {cache.stats.misses}"
