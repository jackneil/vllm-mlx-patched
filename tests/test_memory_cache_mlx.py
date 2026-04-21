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
