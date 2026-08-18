# SPDX-License-Identifier: Apache-2.0
"""Tests for memory-aware prefix cache."""

from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.memory_cache import (
    CacheStats,
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
    _CacheEntry,
    _array_memory,
    _get_available_memory,
    estimate_kv_cache_memory,
)


class TestMemoryCacheConfig:
    """Tests for MemoryCacheConfig."""

    def test_default_config(self):
        config = MemoryCacheConfig()
        assert config.max_memory_mb is None
        assert config.max_memory_percent == 0.20
        assert config.max_entries == 1000
        assert config.enable_memory_tracking is True

    def test_custom_config(self):
        config = MemoryCacheConfig(
            max_memory_mb=2048,
            max_memory_percent=0.5,
            max_entries=100,
        )
        assert config.max_memory_mb == 2048
        assert config.max_memory_percent == 0.5
        assert config.max_entries == 100

    def test_invalid_memory_percent_zero(self):
        with pytest.raises(ValueError, match="max_memory_percent"):
            MemoryCacheConfig(max_memory_percent=0.0)

    def test_invalid_memory_percent_negative(self):
        with pytest.raises(ValueError, match="max_memory_percent"):
            MemoryCacheConfig(max_memory_percent=-0.1)

    def test_invalid_memory_percent_over_one(self):
        with pytest.raises(ValueError, match="max_memory_percent"):
            MemoryCacheConfig(max_memory_percent=1.5)

    def test_invalid_max_entries(self):
        with pytest.raises(ValueError, match="max_entries"):
            MemoryCacheConfig(max_entries=0)

    def test_compute_memory_limit_explicit(self):
        config = MemoryCacheConfig(max_memory_mb=1024)
        assert config.compute_memory_limit() == 1024 * 1024 * 1024

    def test_compute_memory_limit_auto(self):
        with patch(
            "vllm_mlx.memory_cache._get_available_memory",
            return_value=8 * 1024 * 1024 * 1024,  # 8GB
        ):
            config = MemoryCacheConfig(max_memory_percent=0.25)
            limit = config.compute_memory_limit()
            assert limit == 2 * 1024 * 1024 * 1024  # 25% of 8GB = 2GB

    def test_compute_memory_limit_fallback(self):
        with patch(
            "vllm_mlx.memory_cache._get_available_memory",
            return_value=0,  # Detection failed
        ):
            config = MemoryCacheConfig(max_memory_percent=0.25)
            limit = config.compute_memory_limit()
            # Fallback: 25% of 8GB = 2GB
            assert limit == 2 * 1024 * 1024 * 1024


class TestCacheStats:
    """Tests for CacheStats."""

    def test_initial_stats(self):
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        stats = CacheStats(hits=3, misses=1)
        assert stats.hit_rate == 0.75

    def test_hit_rate_no_queries(self):
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0

    def test_memory_utilization(self):
        stats = CacheStats(
            current_memory_bytes=500 * 1024 * 1024,
            max_memory_bytes=1000 * 1024 * 1024,
        )
        assert stats.memory_utilization == 0.5

    def test_to_dict(self):
        stats = CacheStats(hits=10, misses=5, evictions=2)
        d = stats.to_dict()
        assert d["hits"] == 10
        assert d["misses"] == 5
        assert d["evictions"] == 2
        assert "hit_rate" in d
        assert "memory_utilization" in d


class MockArray:
    """Mock array with nbytes attribute."""

    def __init__(self, nbytes: int):
        self.nbytes = nbytes


class MockDtype:
    """Mock dtype with size attribute."""

    def __init__(self, size: int):
        self.size = size


class MockShapeArray:
    """Mock array with shape and dtype (like MLX arrays) but no nbytes."""

    def __init__(self, shape: tuple, dtype_size: int):
        self.shape = shape
        self.dtype = MockDtype(dtype_size)


class MockKVCache:
    """Mock KV cache with keys/values attributes."""

    def __init__(self, key_bytes: int, value_bytes: int):
        self.keys = MockArray(key_bytes)
        self.values = MockArray(value_bytes)


class MockStateCache:
    """Mock cache with state property."""

    def __init__(self, key_bytes: int, value_bytes: int):
        self._keys = MockArray(key_bytes)
        self._values = MockArray(value_bytes)

    @property
    def state(self):
        return (self._keys, self._values)


class TestArrayMemory:
    """Tests for _array_memory helper (shape-based, no lazy eval trigger)."""

    def test_shape_dtype_estimation(self):
        """Verify shape*dtype.size computation without .nbytes access."""
        arr = MockShapeArray(shape=(2, 16, 128, 64), dtype_size=2)
        # 2 * 16 * 128 * 64 * 2 = 524288
        assert _array_memory(arr) == 2 * 16 * 128 * 64 * 2

    def test_fallback_to_nbytes(self):
        """Verify fallback to .nbytes when shape/dtype not available."""
        arr = MockArray(nbytes=4096)
        assert _array_memory(arr) == 4096

    def test_zero_for_unknown_object(self):
        """Return 0 for objects without shape/dtype/nbytes."""
        assert _array_memory(42) == 0
        assert _array_memory("string") == 0

    def test_shape_dtype_preferred_over_nbytes(self):
        """When both shape+dtype and nbytes exist, shape+dtype is used."""

        class DualArray:
            def __init__(self):
                self.shape = (10,)
                self.dtype = MockDtype(4)
                self.nbytes = 9999  # should NOT be used

        arr = DualArray()
        assert _array_memory(arr) == 40  # 10 * 4, not 9999

    def test_estimate_uses_shape_based_for_dict_state(self):
        """estimate_kv_cache_memory uses _array_memory (shape-based) for dicts."""
        keys = MockShapeArray(shape=(1, 8, 100, 64), dtype_size=2)
        values = MockShapeArray(shape=(1, 8, 100, 64), dtype_size=2)
        layer = {"state": (keys, values)}
        expected = 2 * (1 * 8 * 100 * 64 * 2)
        assert estimate_kv_cache_memory([layer]) == expected


class TestEstimateKvCacheMemory:
    """Tests for estimate_kv_cache_memory function."""

    def test_empty_cache(self):
        assert estimate_kv_cache_memory([]) == 0
        assert estimate_kv_cache_memory(None) == 0

    def test_cache_with_nbytes_attribute(self):
        layer = MockKVCache(1000, 1000)
        assert estimate_kv_cache_memory([layer]) == 2000

    def test_cache_with_state_property(self):
        layer = MockStateCache(500, 500)
        assert estimate_kv_cache_memory([layer]) == 1000

    def test_cache_with_dict_state(self):
        keys = MockArray(300)
        values = MockArray(300)
        layer = {"state": (keys, values)}
        assert estimate_kv_cache_memory([layer]) == 600

    def test_multiple_layers(self):
        layers = [MockKVCache(100, 100) for _ in range(4)]
        assert estimate_kv_cache_memory(layers) == 800


class TestCacheEntry:
    """Tests for _CacheEntry."""

    def test_create_entry(self):
        cache = [MockKVCache(100, 100)]
        entry = _CacheEntry.create([1, 2, 3], cache)
        assert entry.tokens == (1, 2, 3)
        assert entry.cache is cache
        assert entry.memory_bytes == 200


class TestMemoryAwarePrefixCache:
    """Tests for MemoryAwarePrefixCache."""

    @pytest.fixture
    def model(self):
        return MagicMock()

    @pytest.fixture
    def small_cache(self, model):
        """Cache with 1MB limit."""
        config = MemoryCacheConfig(max_memory_mb=1, max_entries=10)
        return MemoryAwarePrefixCache(model, config)

    @pytest.fixture
    def mock_kv_cache(self):
        """Create a mock KV cache with known size."""

        def _create(size_bytes: int):
            return [MockKVCache(size_bytes // 2, size_bytes // 2)]

        return _create

    def test_initialization(self, model):
        config = MemoryCacheConfig(max_memory_mb=100)
        cache = MemoryAwarePrefixCache(model, config)
        assert len(cache) == 0
        assert cache.memory_limit_mb == 100.0

    def test_store_and_fetch_exact_match(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3, 4, 5]
        kv = mock_kv_cache(1000)

        # Store
        assert small_cache.store(tokens, kv) is True
        assert len(small_cache) == 1

        # Fetch exact match
        result, remaining = small_cache.fetch(tokens)
        assert result is kv  # Same reference, no copy
        assert remaining == []

    def test_fetch_prefix_match(self, small_cache, mock_kv_cache):
        # Store shorter sequence
        short_tokens = [1, 2, 3]
        kv = mock_kv_cache(1000)
        small_cache.store(short_tokens, kv)

        # Fetch longer sequence that starts with cached prefix
        long_tokens = [1, 2, 3, 4, 5, 6]
        result, remaining = small_cache.fetch(long_tokens)

        assert result is kv
        assert remaining == [4, 5, 6]

    def test_fetch_miss(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3]
        kv = mock_kv_cache(1000)
        small_cache.store(tokens, kv)

        # Fetch completely different sequence
        result, remaining = small_cache.fetch([7, 8, 9])
        assert result is None
        assert remaining == [7, 8, 9]

    def test_lru_eviction_on_memory_pressure(self, model, mock_kv_cache):
        # Create cache with 500KB limit
        config = MemoryCacheConfig(max_memory_mb=0.5, max_entries=100)
        cache = MemoryAwarePrefixCache(model, config)

        # Store entries that together exceed limit
        # Each is ~200KB
        for i in range(5):
            tokens = list(range(i * 10, (i + 1) * 10))
            kv = mock_kv_cache(200 * 1024)
            cache.store(tokens, kv)

        # Should have evicted older entries
        assert cache.memory_usage_mb <= 0.5
        stats = cache.get_stats()
        assert stats["evictions"] > 0

    def test_lru_order_updated_on_fetch(self, small_cache, mock_kv_cache):
        # Store two entries
        tokens1 = [1, 2, 3]
        tokens2 = [4, 5, 6]
        kv1 = mock_kv_cache(100 * 1024)
        kv2 = mock_kv_cache(100 * 1024)

        small_cache.store(tokens1, kv1)
        small_cache.store(tokens2, kv2)

        # Fetch first entry (moves it to end of LRU)
        small_cache.fetch(tokens1)

        # Now tokens2 should be evicted first if we need space
        # Store a large entry to trigger eviction
        big_kv = mock_kv_cache(900 * 1024)
        small_cache.store([7, 8, 9], big_kv)

        # tokens1 should still be there (was recently accessed)
        # tokens2 should be evicted
        assert tokens1 in small_cache or len(small_cache) == 1

    def test_entry_too_large_rejected(self, small_cache, mock_kv_cache):
        # Try to store entry larger than cache limit
        tokens = [1, 2, 3]
        huge_kv = mock_kv_cache(10 * 1024 * 1024)  # 10MB, limit is 1MB

        result = small_cache.store(tokens, huge_kv)
        assert result is False
        assert len(small_cache) == 0

    def test_store_empty_rejected(self, small_cache, mock_kv_cache):
        assert small_cache.store([], mock_kv_cache(100)) is False
        assert small_cache.store([1, 2, 3], []) is False
        assert small_cache.store([1, 2, 3], None) is False

    def test_remove_entry(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3]
        kv = mock_kv_cache(1000)
        small_cache.store(tokens, kv)
        assert len(small_cache) == 1

        assert small_cache.remove(tokens) is True
        assert len(small_cache) == 0
        assert small_cache.remove(tokens) is False  # Already removed

    def test_clear(self, small_cache, mock_kv_cache):
        for i in range(3):
            small_cache.store([i], mock_kv_cache(1000))

        assert len(small_cache) == 3
        small_cache.clear()
        assert len(small_cache) == 0
        assert small_cache.memory_usage_mb == 0

    def test_contains(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3]
        assert tokens not in small_cache
        small_cache.store(tokens, mock_kv_cache(1000))
        assert tokens in small_cache

    def test_stats_tracking(self, small_cache, mock_kv_cache):
        tokens1 = [1, 2, 3]
        tokens2 = [4, 5, 6]
        kv = mock_kv_cache(1000)

        small_cache.store(tokens1, kv)
        small_cache.fetch(tokens1)  # Hit
        small_cache.fetch(tokens2)  # Miss

        stats = small_cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entry_count"] == 1

    def test_reset_stats(self, small_cache, mock_kv_cache):
        small_cache.store([1, 2, 3], mock_kv_cache(1000))
        small_cache.fetch([1, 2, 3])
        small_cache.fetch([4, 5, 6])

        small_cache.reset_stats()
        stats = small_cache.get_stats()

        # Stats reset but entry count preserved
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["entry_count"] == 1

    def test_duplicate_store_updates_lru(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3]
        kv = mock_kv_cache(1000)

        small_cache.store(tokens, kv)
        initial_len = len(small_cache)

        # Store same tokens again
        small_cache.store(tokens, kv)

        # Should not create duplicate
        assert len(small_cache) == initial_len

    def test_max_entries_limit(self, model, mock_kv_cache):
        # Create cache with low entry limit
        config = MemoryCacheConfig(max_memory_mb=100, max_entries=3)
        cache = MemoryAwarePrefixCache(model, config)

        # Store 5 entries (only 3 should remain)
        for i in range(5):
            cache.store([i], mock_kv_cache(100))

        assert len(cache) <= 3


class TestGetAvailableMemory:
    """Tests for _get_available_memory helper."""

    def test_with_psutil(self):
        try:
            from importlib.util import find_spec

            if find_spec("psutil") is None:
                pytest.skip("psutil not installed")
            mem = _get_available_memory()
            assert mem > 0
        except ImportError:
            pytest.skip("psutil not installed")

    def test_without_psutil(self):
        with patch.dict("sys.modules", {"psutil": None}):
            # Should return 0 when psutil not available
            # Note: This test may not work as expected due to import caching
            pass


class TestClearInFlightGuard:
    """PR-A Task A.1 — in-flight guard for DELETE /v1/cache.

    Mirrors the PagedCacheManager.reset_prefix_cache() pattern in
    paged_cache.py:1149-1156 ("refuse when entries are in use"). The
    adapter caller aggregates refusals so DELETE /v1/cache returns HTTP
    409 Conflict instead of proceeding and crashing mid-decode.
    """

    @pytest.fixture
    def model(self):
        return MagicMock()

    @pytest.fixture
    def cache(self, model):
        config = MemoryCacheConfig(max_memory_mb=1, max_entries=10)
        return MemoryAwarePrefixCache(model, config)

    def test_clear_refuses_when_entries_in_use(self, cache):
        """clear() must return False when entries are held by in-flight
        requests, so the endpoint can return 409 instead of wiping state
        that live decode still reads."""
        cache._mark_in_use_for_test()
        result = cache.clear()
        assert result is False, "clear() must refuse when entries are in use"

    def test_clear_returns_true_when_idle(self, cache):
        """Idle cache clears normally and reports success."""
        assert cache.clear() is True

    def test_clear_wipes_entries_when_idle(self, cache):
        """Idle clear still performs the full wipe (no regression)."""

        def _mock_kv(size_bytes: int):
            return [MockKVCache(size_bytes // 2, size_bytes // 2)]

        cache.store([1, 2, 3], _mock_kv(1000))
        assert len(cache) == 1
        assert cache.clear() is True
        assert len(cache) == 0

    def test_clear_preserves_entries_when_refused(self, cache):
        """A refused clear must NOT wipe state — that's the whole point."""

        def _mock_kv(size_bytes: int):
            return [MockKVCache(size_bytes // 2, size_bytes // 2)]

        cache.store([1, 2, 3], _mock_kv(1000))
        cache._mark_in_use_for_test()
        assert cache.clear() is False
        assert (
            len(cache) == 1
        ), "refused clear must leave entries intact for live decoders"


# ---- acquire/release production API ----


def test_acquire_blocks_clear_release_allows_it():
    from vllm_mlx.memory_cache import MemoryAwarePrefixCache

    cache = MemoryAwarePrefixCache(model=None)
    cache.acquire("req-A")

    assert cache.clear() is False  # refused while req-A holds

    cache.release("req-A")
    assert cache.clear() is True  # allowed after release


def test_acquire_is_idempotent():
    """Double-acquire of same request_id is a no-op (set semantics)."""
    from vllm_mlx.memory_cache import MemoryAwarePrefixCache

    cache = MemoryAwarePrefixCache(model=None)
    cache.acquire("req-A")
    cache.acquire("req-A")
    cache.acquire("req-A")
    assert cache._in_flight_count == 1

    cache.release("req-A")
    assert cache._in_flight_count == 0


def test_release_unknown_id_is_silent_noop():
    """Release of id never acquired doesn't error or underflow."""
    from vllm_mlx.memory_cache import MemoryAwarePrefixCache

    cache = MemoryAwarePrefixCache(model=None)
    cache.release("never-acquired")
    assert cache._in_flight_count == 0
    assert cache.clear() is True


def test_acquire_release_multiple_requests_independent():
    from vllm_mlx.memory_cache import MemoryAwarePrefixCache

    cache = MemoryAwarePrefixCache(model=None)
    cache.acquire("req-A")
    cache.acquire("req-B")
    assert cache._in_flight_count == 2

    cache.release("req-A")
    assert cache._in_flight_count == 1
    assert cache.clear() is False  # req-B still holds

    cache.release("req-B")
    assert cache.clear() is True


def test_compute_model_fingerprint_is_deterministic_and_config_sensitive():
    """#29 — fingerprint helper must (1) round-trip identically for the same
    config and (2) differ when any of the load-bearing config attrs change.
    """
    from vllm_mlx.memory_cache import _compute_model_fingerprint

    def _mk(**overrides):
        cfg = MagicMock()
        defaults = dict(
            num_hidden_layers=28,
            hidden_size=1024,
            vocab_size=151936,
            num_key_value_heads=8,
            head_dim=128,
            intermediate_size=3072,
            model_type="qwen3",
        )
        defaults.update(overrides)
        for k, v in defaults.items():
            setattr(cfg, k, v)
        m = MagicMock()
        m.config = cfg
        return m

    base = _compute_model_fingerprint(_mk())
    assert isinstance(base, str) and len(base) == 16
    # Same config -> same fingerprint.
    assert _compute_model_fingerprint(_mk()) == base
    # Each of these MUST flip the fingerprint.
    for field in (
        "num_hidden_layers",
        "hidden_size",
        "vocab_size",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "model_type",
    ):
        other_val = "xxx" if field == "model_type" else 9999
        assert (
            _compute_model_fingerprint(_mk(**{field: other_val})) != base
        ), f"fingerprint should differ when {field} changes"


def test_save_then_load_roundtrip_with_fingerprint(tmp_path):
    """#29 — save-then-load roundtrip on an empty cache must succeed
    (returns False for save-nothing, 0 for load-nothing) and must not
    crash from the new version/fingerprint plumbing.
    """
    from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

    model = MagicMock()
    model.config = MagicMock(
        num_hidden_layers=28,
        hidden_size=1024,
        vocab_size=151936,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=3072,
        model_type="qwen3",
    )
    cache = MemoryAwarePrefixCache(
        model, MemoryCacheConfig(max_memory_mb=8, max_entries=4)
    )

    d = str(tmp_path)
    assert cache.save_to_disk(d) is False
    assert cache.load_from_disk(d) == 0


def test_load_rejects_older_version(tmp_path):
    """#29 — a cache written with an older version must be discarded on load.

    v3 used positional entry_<i> filenames; v4 is content-keyed. A v3 index
    read by the v4 loader would name files that don't exist, so the version
    gate has to reject the whole directory and let the cache re-warm.
    """
    import json

    from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

    d = tmp_path
    (d / "index.json").write_text(
        json.dumps(
            {
                "version": 3,
                "num_entries": 1,
                "entries": [{"index": 0, "num_tokens": 5, "memory_bytes": 1024}],
            }
        )
    )

    model = MagicMock()
    model.config = MagicMock(num_hidden_layers=1)
    cache = MemoryAwarePrefixCache(model, MemoryCacheConfig(max_memory_mb=8))

    assert cache.load_from_disk(str(d)) == 0


def test_persist_version_is_pinned_at_4():
    """The on-disk format is v4 (content-keyed filenames). Bumping this
    without bumping the loader silently resurrects incompatible directories.
    """
    from vllm_mlx.memory_cache import _CACHE_PERSIST_VERSION

    assert _CACHE_PERSIST_VERSION == 4


def test_load_rejects_fingerprint_mismatch(tmp_path):
    """#29 — a current-version cache whose fingerprint differs from the
    current model's must be rejected.
    """
    import json

    from vllm_mlx.memory_cache import (
        _CACHE_PERSIST_VERSION,
        MemoryAwarePrefixCache,
        MemoryCacheConfig,
    )

    d = tmp_path
    (d / "index.json").write_text(
        json.dumps(
            {
                "version": _CACHE_PERSIST_VERSION,
                "model_fingerprint": "deadbeefdeadbeef",
                "num_entries": 0,
                "entries": [],
            }
        )
    )

    model = MagicMock()
    model.config = MagicMock(
        num_hidden_layers=28,
        hidden_size=1024,
        vocab_size=151936,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=3072,
        model_type="qwen3",
    )
    cache = MemoryAwarePrefixCache(model, MemoryCacheConfig(max_memory_mb=8))
    assert cache.load_from_disk(str(d)) == 0


# ---------------------------------------------------------------------------
# save_to_disk Metal-pressure retry (2026-08-17). At shutdown the dying
# engine's state plus MLX's buffer cache can sit near the Metal resource
# limit, and save_prompt_cache's .state materialization then fails with
# "[metal::malloc] Resource limit exceeded" — observed live: 86/88 entries
# (59 GB of warm prefixes) lost on one restart. save_to_disk must return the
# buffer cache's blocks and retry once before giving up on an entry.
# ---------------------------------------------------------------------------


def _persist_cache_with_entries(n, sizes=None, **config_overrides):
    """Build a cache pre-populated with `n` synthetic entries.

    Entry i has token key `(10i .. 10i+4)`; insertion order is oldest-first,
    so entry n-1 is the MRU. `sizes` overrides per-entry memory_bytes.
    """
    from vllm_mlx.memory_cache import (
        MemoryAwarePrefixCache,
        MemoryCacheConfig,
        _CacheEntry,
    )

    model = MagicMock()
    model.config = MagicMock(
        num_hidden_layers=2,
        hidden_size=64,
        vocab_size=1000,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=128,
        model_type="qwen3",
    )
    kwargs = dict(max_memory_mb=64, max_entries=8)
    kwargs.update(config_overrides)
    cache = MemoryAwarePrefixCache(model, MemoryCacheConfig(**kwargs))
    for i in range(n):
        key = _entry_key(i)
        cache._entries[key] = _CacheEntry(
            tokens=key,
            cache=[MagicMock()],
            memory_bytes=1024 if sizes is None else sizes[i],
        )
    return cache


def _entry_key(i):
    """Token key for synthetic entry `i`."""
    return tuple(range(10 * i, 10 * i + 5))


def _entry_hash(i):
    """On-disk content hash for synthetic entry `i`."""
    from vllm_mlx.memory_cache import _tokens_hash

    return _tokens_hash(_entry_key(i))


def _hashes_on_disk(d):
    """Hashes with a .safetensors file in directory `d`."""
    return {
        p.name[len("entry_") : -len(".safetensors")]
        for p in d.iterdir()
        if p.name.startswith("entry_") and p.name.endswith(".safetensors")
    }


def _fake_saver(events=None):
    """A save_prompt_cache stand-in that writes a stub file."""

    def _save(path, cache_obj, metadata=None):
        if events is not None:
            events.append(path)
        with open(path, "wb") as f:
            f.write(b"x")

    return _save


def test_save_to_disk_retries_a_failed_entry_after_clearing_the_buffer_cache(
    tmp_path, monkeypatch
):
    """An entry whose first save dies on Metal pressure must be retried after
    mx.clear_cache() — and the retry must come AFTER a clear, not blind."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")

    events = []
    attempts = {"n": 0}

    def fake_save(path, cache_obj, metadata=None):
        attempts["n"] += 1
        if attempts["n"] % 2 == 1:  # first attempt per entry fails
            events.append("save-fail")
            raise RuntimeError("[metal::malloc] Resource limit (499000) exceeded.")
        events.append("save-ok")
        with open(path, "wb") as f:
            f.write(b"x")

    monkeypatch.setattr("mlx_lm.models.cache.save_prompt_cache", fake_save)
    monkeypatch.setattr("mlx.core.clear_cache", lambda: events.append("clear"))

    cache = _persist_cache_with_entries(2)
    assert cache.save_to_disk(str(tmp_path)) is True

    saved = [e for e in events if e == "save-ok"]
    assert len(saved) == 2, (
        f"both entries must survive a first-attempt Metal failure via the "
        f"retry; events={events}"
    )
    # Every failure is followed by a clear and then the retry.
    for i, e in enumerate(events):
        if e == "save-fail":
            assert events[i + 1 : i + 3] == [
                "clear",
                "save-ok",
            ], f"retry must run after mx.clear_cache(), got {events}"
    # And one proactive clear happens before any save attempt at all.
    assert events[0] == "clear", f"expected an up-front clear, got {events}"


def test_save_to_disk_gives_up_on_an_entry_after_exactly_one_retry(
    tmp_path, monkeypatch
):
    """A persistently-failing entry is skipped after two attempts — the retry
    must not loop, and the other entries must still be saved."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    calls = {"n": 0}

    doomed = _entry_hash(0)

    def always_fail_first_key(path, cache_obj, metadata=None):
        calls["n"] += 1
        if doomed in path:
            raise RuntimeError("[metal::malloc] Resource limit (499000) exceeded.")
        with open(path, "wb") as f:
            f.write(b"x")

    monkeypatch.setattr("mlx_lm.models.cache.save_prompt_cache", always_fail_first_key)
    monkeypatch.setattr("mlx.core.clear_cache", lambda: None)

    cache = _persist_cache_with_entries(2)
    assert cache.save_to_disk(str(tmp_path)) is True  # entry 1 still saved

    # entry 0: initial + one retry = 2 attempts; entry 1: 1 attempt.
    assert (
        calls["n"] == 3
    ), f"expected exactly one retry for the bad entry, got {calls['n']}"
    import json as _json

    index = _json.loads((tmp_path / "index.json").read_text())
    # Only the entry that actually landed may be indexed.
    assert index["num_entries"] == 1
    assert [e["hash"] for e in index["entries"]] == [_entry_hash(1)]


# ---------------------------------------------------------------------------
# Periodic incremental top-K persistence (flush_to_disk, format v4).
#
# Persistence used to happen only at shutdown, so a kernel panic — or a
# shutdown save that died under Metal pressure — threw away every warm
# prefix, including the multi-minute Claude Code system-prompt prefill.
# flush_to_disk() keeps the K most-recent prefixes durable WHILE serving,
# writing only what isn't already on disk.
# ---------------------------------------------------------------------------


def test_flush_is_idempotent_and_does_no_mlx_work_when_nothing_is_new(
    tmp_path, monkeypatch
):
    """A steady-state flush must save nothing and touch save_prompt_cache
    zero times — otherwise every interval re-serializes tens of GB of KV."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")

    saves = []
    monkeypatch.setattr("mlx_lm.models.cache.save_prompt_cache", _fake_saver(saves))
    monkeypatch.setattr("mlx.core.clear_cache", lambda: None)

    cache = _persist_cache_with_entries(3)
    d = str(tmp_path)

    assert cache.flush_to_disk(d) == 3
    assert len(saves) == 3

    saves.clear()
    assert cache.flush_to_disk(d) == 0, "second flush must write nothing new"
    assert saves == [], f"no MLX serialization may happen, got {saves}"
    # ...and the durable set is unchanged.
    assert _hashes_on_disk(tmp_path) == {_entry_hash(i) for i in range(3)}


def test_flush_keeps_the_newest_k_entries_and_prunes_the_rest(tmp_path, monkeypatch):
    """Selection walks the LRU NEWEST-first: with 5 entries and K=3, the 3
    most-recently-used survive on disk and the two oldest are pruned."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")

    monkeypatch.setattr("mlx_lm.models.cache.save_prompt_cache", _fake_saver())
    monkeypatch.setattr("mlx.core.clear_cache", lambda: None)

    cache = _persist_cache_with_entries(5)  # entry 4 is MRU
    d = str(tmp_path)

    assert cache.flush_to_disk(d, max_entries=3) == 3

    assert _hashes_on_disk(tmp_path) == {_entry_hash(i) for i in (2, 3, 4)}, (
        "the three MOST RECENT entries must be the durable set — walking "
        "oldest-first would persist 0,1,2 instead"
    )
    for i in (0, 1):
        assert not (tmp_path / f"entry_{_entry_hash(i)}.safetensors").exists()
        assert not (tmp_path / f"entry_{_entry_hash(i)}_tokens.bin").exists()


def test_flush_packs_the_byte_budget_instead_of_stopping_at_the_first_misfit(
    tmp_path, monkeypatch
):
    """An entry too big for the remaining budget is SKIPPED, not a stop
    signal — a single huge MRU prefix must not starve the smaller ones."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")

    monkeypatch.setattr("mlx_lm.models.cache.save_prompt_cache", _fake_saver())
    monkeypatch.setattr("mlx.core.clear_cache", lambda: None)

    # entry 2 (the MRU) is far bigger than the whole budget.
    cache = _persist_cache_with_entries(3, sizes=[100, 100, 10_000])
    d = str(tmp_path)

    saved = cache.flush_to_disk(d, max_entries=10, max_bytes=500)

    assert saved == 2, f"the two small older entries must still land, got {saved}"
    assert _hashes_on_disk(tmp_path) == {_entry_hash(0), _entry_hash(1)}
    assert _entry_hash(2) not in _hashes_on_disk(tmp_path)


def test_flush_prunes_an_entry_that_left_memory_between_flushes(tmp_path, monkeypatch):
    """Once an entry is evicted from memory it can never be re-selected, so
    the next flush must delete its files — disk holds exactly the top-K."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")

    monkeypatch.setattr("mlx_lm.models.cache.save_prompt_cache", _fake_saver())
    monkeypatch.setattr("mlx.core.clear_cache", lambda: None)

    cache = _persist_cache_with_entries(3)
    d = str(tmp_path)
    cache.flush_to_disk(d)
    assert _entry_hash(0) in _hashes_on_disk(tmp_path)

    # Entry 0 falls out of the in-memory cache (LRU eviction).
    cache.remove(list(_entry_key(0)))

    assert cache.flush_to_disk(d) == 0  # nothing new to write
    assert _entry_hash(0) not in _hashes_on_disk(
        tmp_path
    ), "an entry no longer in memory must be pruned from disk"
    assert not (tmp_path / f"entry_{_entry_hash(0)}_tokens.bin").exists()
    assert _hashes_on_disk(tmp_path) == {_entry_hash(1), _entry_hash(2)}

    import json as _json

    index = _json.loads((tmp_path / "index.json").read_text())
    assert [e["hash"] for e in index["entries"]] == [_entry_hash(1), _entry_hash(2)]


def test_flush_then_load_roundtrips_token_keys_in_recency_order(tmp_path, monkeypatch):
    """A flushed cache must come back with the same prefixes AND the same
    recency order — the MRU has to stay last so it's evicted last."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")

    stored = {}

    def fake_save(path, cache_obj, metadata=None):
        stored[path] = cache_obj
        with open(path, "wb") as f:
            f.write(b"x")

    monkeypatch.setattr("mlx_lm.models.cache.save_prompt_cache", fake_save)
    monkeypatch.setattr(
        "mlx_lm.models.cache.load_prompt_cache", lambda path: stored[path]
    )
    monkeypatch.setattr("mlx.core.clear_cache", lambda: None)
    # Synthetic MagicMock caches have no measurable size; pin it so the
    # loader's memory-limit gate doesn't fire on garbage estimates.
    monkeypatch.setattr(
        "vllm_mlx.memory_cache.estimate_kv_cache_memory", lambda cache: 1024
    )

    src = _persist_cache_with_entries(3)
    d = str(tmp_path)
    assert src.flush_to_disk(d) == 3

    fresh = _persist_cache_with_entries(0)
    assert fresh.load_from_disk(d) == 3

    assert list(fresh._entries.keys()) == [
        _entry_key(0),
        _entry_key(1),
        _entry_key(2),
    ], (
        "entries must rebuild oldest -> newest so the OrderedDict's LRU end "
        "matches what was flushed"
    )
    # The sorted prefix index has to be rebuilt too, or fetch() goes blind.
    assert fresh._sorted_keys == sorted(fresh._entries.keys())


def test_a_serialization_failure_cannot_corrupt_the_existing_index(
    tmp_path, monkeypatch
):
    """The index must be built in a temp file and committed by rename. If
    serialization dies halfway (full disk, killed process), a directly-
    written index.json would be left truncated — and a truncated index is a
    boot that silently throws away every persisted prefix."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")

    monkeypatch.setattr("mlx_lm.models.cache.save_prompt_cache", _fake_saver())
    monkeypatch.setattr("mlx.core.clear_cache", lambda: None)

    cache = _persist_cache_with_entries(2)
    d = str(tmp_path)
    cache.flush_to_disk(d)
    good = (tmp_path / "index.json").read_text()

    import json as _json

    def half_written_then_boom(obj, fp, **kw):
        fp.write('{"version": 4, "entries": [{"hash": "aaaa')
        raise OSError("no space left on device")

    monkeypatch.setattr(_json, "dump", half_written_then_boom)

    cache3 = _persist_cache_with_entries(3)
    with pytest.raises(OSError):
        cache3.flush_to_disk(d)

    assert (
        tmp_path / "index.json"
    ).read_text() == good, (
        "a torn write must land in the temp file, never in index.json"
    )
    assert _json.loads((tmp_path / "index.json").read_text())["version"] == 4


def test_flush_cleans_up_the_temp_index_when_the_commit_fails(tmp_path, monkeypatch):
    """A failed rename must leave the previous index.json intact and not
    litter the cache dir with a stale index.json.tmp."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")

    monkeypatch.setattr("mlx_lm.models.cache.save_prompt_cache", _fake_saver())
    monkeypatch.setattr("mlx.core.clear_cache", lambda: None)

    cache = _persist_cache_with_entries(2)
    d = str(tmp_path)
    cache.flush_to_disk(d)
    good = (tmp_path / "index.json").read_text()

    import os as _os

    real_rename = _os.rename

    def boom(src, dst, *a, **kw):
        if str(dst).endswith("index.json"):
            raise OSError("disk full")
        return real_rename(src, dst, *a, **kw)

    monkeypatch.setattr("os.rename", boom)

    cache3 = _persist_cache_with_entries(3)
    with pytest.raises(OSError):
        cache3.flush_to_disk(d)

    assert (
        tmp_path / "index.json"
    ).read_text() == good, "the pre-existing index must survive a failed commit"
    assert not (tmp_path / "index.json.tmp").exists(), "temp file must be cleaned up"


def test_flush_retries_a_metal_pressure_failure_after_clearing_the_buffer_cache(
    tmp_path, monkeypatch
):
    """The Metal-pressure retry (2026-08-17) must still hold through the
    flush path, and the up-front clear must happen before any save."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")

    events = []
    attempts = {"n": 0}

    def fake_save(path, cache_obj, metadata=None):
        attempts["n"] += 1
        if attempts["n"] % 2 == 1:  # first attempt per entry fails
            events.append("save-fail")
            raise RuntimeError("[metal::malloc] Resource limit (499000) exceeded.")
        events.append("save-ok")
        with open(path, "wb") as f:
            f.write(b"x")

    monkeypatch.setattr("mlx_lm.models.cache.save_prompt_cache", fake_save)
    monkeypatch.setattr("mlx.core.clear_cache", lambda: events.append("clear"))

    cache = _persist_cache_with_entries(2)
    assert cache.flush_to_disk(str(tmp_path)) == 2

    assert events[0] == "clear", f"expected an up-front clear, got {events}"
    for i, e in enumerate(events):
        if e == "save-fail":
            assert events[i + 1 : i + 3] == [
                "clear",
                "save-ok",
            ], f"retry must run after mx.clear_cache(), got {events}"


def test_flush_does_not_clear_the_buffer_cache_when_nothing_needs_saving(
    tmp_path, monkeypatch
):
    """mx.clear_cache() drops MLX's whole buffer pool; a no-op flush every
    5 minutes must not pay that cost."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")

    clears = []
    monkeypatch.setattr("mlx_lm.models.cache.save_prompt_cache", _fake_saver())
    monkeypatch.setattr("mlx.core.clear_cache", lambda: clears.append(1))

    cache = _persist_cache_with_entries(2)
    d = str(tmp_path)
    cache.flush_to_disk(d)
    clears.clear()

    cache.flush_to_disk(d)
    assert clears == [], "a no-op flush must not clear the MLX buffer cache"


def test_flush_defaults_come_from_the_configured_budgets(tmp_path, monkeypatch):
    """flush_to_disk() with no args must honour the config, so the periodic
    flush and the shutdown save persist the same set."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")

    monkeypatch.setattr("mlx_lm.models.cache.save_prompt_cache", _fake_saver())
    monkeypatch.setattr("mlx.core.clear_cache", lambda: None)

    cache = _persist_cache_with_entries(5, persist_max_entries=2)
    assert cache.flush_to_disk(str(tmp_path)) == 2
    assert _hashes_on_disk(tmp_path) == {_entry_hash(3), _entry_hash(4)}


def test_save_to_disk_delegates_to_flush_and_keeps_its_bool_contract(
    tmp_path, monkeypatch
):
    """The shutdown path is a thin delegate: False on an empty cache, True
    once something is durable."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")

    monkeypatch.setattr("mlx_lm.models.cache.save_prompt_cache", _fake_saver())
    monkeypatch.setattr("mlx.core.clear_cache", lambda: None)

    empty = _persist_cache_with_entries(0)
    assert empty.save_to_disk(str(tmp_path)) is False

    cache = _persist_cache_with_entries(2)
    assert cache.save_to_disk(str(tmp_path)) is True
    assert _hashes_on_disk(tmp_path) == {_entry_hash(0), _entry_hash(1)}


def test_persist_config_validation():
    """Nonsense budgets must be rejected at config construction, not
    discovered as an empty cache directory hours later."""
    from vllm_mlx.memory_cache import MemoryCacheConfig

    with pytest.raises(ValueError, match="persist_max_entries"):
        MemoryCacheConfig(persist_max_entries=0)
    with pytest.raises(ValueError, match="persist_max_bytes"):
        MemoryCacheConfig(persist_max_bytes=0)
    # Sane values are accepted.
    cfg = MemoryCacheConfig(persist_max_entries=3, persist_max_bytes=1024)
    assert cfg.persist_max_entries == 3
    assert cfg.persist_max_bytes == 1024
