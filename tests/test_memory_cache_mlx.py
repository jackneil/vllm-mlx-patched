# SPDX-License-Identifier: Apache-2.0
"""MLX-dependent regression tests for the KV cache LCP contamination fix.

Ported from waybarrios/vllm-mlx#385. These tests use real
mlx_lm.models.cache.KVCache objects backed by mlx.core arrays and
therefore run only on Apple Silicon. The Linux test-matrix CI job
excludes this file because MLX has no Linux distribution.
"""

from unittest.mock import MagicMock


class TestTrimCacheOffset:
    """Regression: LCP / supersequence trim used to shrink .offset while
    sharing the oversized arrays, letting attention paths that read
    cache.state directly (Gemma 4 KV-shared, Qwen3 kickoff) see stale
    tokens from a prior owner. See waybarrios#384 and jackneil#29.
    """

    def test_plain_kv_cache_array_sliced_to_new_offset(self):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        layer = KVCache()
        layer.keys = mx.arange(1 * 4 * 500 * 8, dtype=mx.float32).reshape(1, 4, 500, 8)
        layer.values = mx.arange(1 * 4 * 500 * 8, dtype=mx.float32).reshape(
            1, 4, 500, 8
        )
        layer.offset = 500

        trim_by = 500 - 60
        trimmed = _trim_cache_offset([layer], trim_by)
        tc = trimmed[0]

        assert tc.offset == 60
        assert tc.keys.shape[-2] == 60
        assert tc.values.shape[-2] == 60

    def test_quantized_kv_cache_offset_shrinks(self):
        """Regression guard for the QuantizedKVCache branch of
        _trim_cache_offset. Our fork's branch shrinks offset correctly
        and preserves group_size/bits. The dequantize-time slice in a
        later commit depends on this invariant being stable.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache, QuantizedKVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        base = KVCache()
        base.keys = mx.ones((1, 4, 128, 64), dtype=mx.float32)
        base.values = mx.ones((1, 4, 128, 64), dtype=mx.float32)
        base.offset = 128
        qcache = base.to_quantized(group_size=64, bits=8)

        trimmed = _trim_cache_offset([qcache], 68)
        tc = trimmed[0]

        assert isinstance(tc, QuantizedKVCache)
        assert tc.offset == 60
        assert tc.group_size == qcache.group_size
        assert tc.bits == qcache.bits
        # Source entry untouched.
        assert qcache.offset == 128

    def test_fetch_returns_sliced_cache_on_lcp_match(self):
        """End-to-end: MemoryAwarePrefixCache.fetch on a request that shares
        only a prefix with a longer stored entry must return a cache whose
        arrays are already sliced. This is the full regression of #29
        above the unit level.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

        model = MagicMock()
        pc = MemoryAwarePrefixCache(
            model, MemoryCacheConfig(max_memory_mb=64, max_entries=10)
        )

        stored_layer = KVCache()
        shared = mx.ones((1, 2, 60, 4), dtype=mx.float32)
        private = mx.full((1, 2, 60, 4), 7.0, dtype=mx.float32)
        stored_layer.keys = mx.concatenate([shared, private], axis=2)
        stored_layer.values = stored_layer.keys
        stored_layer.offset = 120
        pc.store(list(range(1, 121)), [stored_layer])

        new_tokens = list(range(1, 60)) + [999, 1000, 1001]
        fetched, remaining = pc.fetch(new_tokens)

        assert fetched is not None
        tc = fetched[0]
        assert tc.offset == 59
        assert tc.keys.shape[-2] == 59
        # The 7.0 private content from the stored entry must NOT be visible.
        assert float(mx.max(tc.keys).item()) == 1.0
        assert remaining == [999, 1000, 1001]
        # Source entry unaffected (fetched is a new wrapper over sliced arrays).
        assert stored_layer.keys.shape[-2] == 120
        assert stored_layer.offset == 120

    def test_rotating_kv_cache_slice_applies_but_type_stripped(self):
        """RotatingKVCache has ``.offset`` and array-typed ``.keys`` so it
        enters the KVCache branch of ``_trim_cache_offset`` (not the
        catch-all else). The slice fix from #29 therefore applies — no
        stale tokens past the trimmed offset — but a separate pre-existing
        quirk is that the returned wrapper is a plain ``KVCache`` rather
        than a ``RotatingKVCache``. Locking that in here so a future
        refactor either preserves the type (fixing the pre-existing loss)
        or fails this test loudly.

        Tracking for proper RotatingKVCache handling (max_size / keep /
        _idx preservation): follow-up PR, referencing
        waybarrios/vllm-mlx#296 commit b61f57c.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        layer = RotatingKVCache(max_size=128, keep=0)
        layer.keys = mx.ones((1, 4, 128, 8), dtype=mx.float32)
        layer.values = mx.ones((1, 4, 128, 8), dtype=mx.float32)
        layer.offset = 128

        trimmed = _trim_cache_offset([layer], 50)
        tc = trimmed[0]

        # Slice fix applies: shape matches new offset (the #29 invariant).
        assert tc.offset == 78
        assert tc.keys.shape[-2] == 78
        # Pre-existing quirk: type-stripped to plain KVCache. When
        # waybarrios/vllm-mlx#296 is ported, this may change to preserve
        # RotatingKVCache — at which point update this assertion.
        assert isinstance(tc, KVCache)


class TestDequantizeCacheSlice:
    """Regression tests for _dequantize_cache slicing after dequantization.

    When KV quantization is enabled, the prefix cache stores
    QuantizedKVCache layers. After LCP trim reduces the offset,
    _dequantize_cache must slice the dequantized arrays down to offset
    to prevent readers that bypass offset from seeing stale tokens.
    Mirrors the plain-KVCache slice in _trim_cache_offset.
    """

    def test_dequantize_slices_to_offset(self):
        """After trim + dequantize, keys/values shape[-2] == offset."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _dequantize_cache, _trim_cache_offset

        base = KVCache()
        base.keys = mx.ones((1, 4, 512, 64), dtype=mx.float32)
        base.values = mx.ones((1, 4, 512, 64), dtype=mx.float32)
        base.offset = 512

        qcache = base.to_quantized(group_size=64, bits=8)
        trimmed = _trim_cache_offset([qcache], 512 - 60)
        result = _dequantize_cache(trimmed)

        tc = result[0]
        assert tc.offset == 60
        assert tc.keys.shape[-2] == 60
        assert tc.values.shape[-2] == 60

    def test_dequantize_no_stale_tokens_via_state(self):
        """Stale tokens past offset must not be visible via cache.state."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _dequantize_cache, _trim_cache_offset

        base = KVCache()
        shared = mx.ones((1, 4, 64, 64), dtype=mx.float32)
        private = mx.full((1, 4, 448, 64), 7.0, dtype=mx.float32)
        base.keys = mx.concatenate([shared, private], axis=2)
        base.values = mx.concatenate([shared, private], axis=2)
        base.offset = 512

        qcache = base.to_quantized(group_size=64, bits=8)
        trimmed = _trim_cache_offset([qcache], 512 - 64)
        result = _dequantize_cache(trimmed)

        tc = result[0]
        keys_view, _ = tc.state
        assert keys_view.shape[-2] == 64
        # Dequantized values are approximate; must stay near 1.0 (shared),
        # never near 7.0 (private past offset).
        assert float(mx.max(keys_view).item()) < 2.0

    def test_dequantize_no_trim_preserves_full_array(self):
        """When offset == shape[-2], no slicing occurs."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _dequantize_cache

        base = KVCache()
        base.keys = mx.ones((1, 4, 128, 64), dtype=mx.float32)
        base.values = mx.ones((1, 4, 128, 64), dtype=mx.float32)
        base.offset = 128

        qcache = base.to_quantized(group_size=64, bits=8)
        result = _dequantize_cache([qcache])

        tc = result[0]
        assert tc.offset == 128
        assert tc.keys.shape[-2] == 128
        assert tc.values.shape[-2] == 128
