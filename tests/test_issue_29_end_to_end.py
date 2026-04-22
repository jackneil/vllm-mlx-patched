# SPDX-License-Identifier: Apache-2.0
"""End-to-end regression for jackneil/vllm-mlx-patched#29.

Symptom: three sequential chat requests against a server that loaded a
persisted MemoryAwarePrefixCache produced coherent output on request 1
and degenerate token-loop garbage ('ongo', 'diễn', ...) on requests
2+. Root cause: LCP / supersequence matches returned a cache whose
arrays still contained data past the trimmed offset; the kickoff
prefill exposed that stale data to attention.
"""

from unittest.mock import MagicMock


def test_issue_29_supersequence_fetch_no_stale_kv():
    """Shape of the #29 bug: stored supersequence entry with private data
    past the shared prefix. Fetch on the shared prefix must not expose
    the private data via cache.state or the underlying array shape.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

    model = MagicMock()
    model.config = MagicMock(
        num_hidden_layers=2,
        hidden_size=64,
        vocab_size=100,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=128,
        model_type="qwen3",
    )
    pc = MemoryAwarePrefixCache(
        model, MemoryCacheConfig(max_memory_mb=8, max_entries=4)
    )

    layers = []
    for _ in range(2):
        shared = mx.ones((1, 2, 14, 32), dtype=mx.float32)
        private = mx.full((1, 2, 80, 32), 7.0, dtype=mx.float32)
        layer = KVCache()
        layer.keys = mx.concatenate([shared, private], axis=2)
        layer.values = mx.concatenate([shared, private], axis=2)
        layer.offset = 94
        layers.append(layer)

    pc.store(list(range(1, 95)), layers)

    fetched, remaining = pc.fetch(list(range(1, 15)))

    assert fetched is not None
    for layer in fetched:
        keys_view, values_view = layer.state
        assert keys_view.shape[-2] == 14
        assert values_view.shape[-2] == 14
        assert float(mx.max(keys_view).item()) == 1.0
        assert float(mx.max(values_view).item()) == 1.0
    assert remaining == []


def test_issue_29_pre_fix_persisted_cache_is_discarded(tmp_path):
    """A persisted cache written at version=2 (pre-fix) must be
    discarded on load so pre-fix entries cannot contaminate a fresh run.
    """
    import json

    from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

    d = tmp_path
    (d / "index.json").write_text(
        json.dumps(
            {
                "version": 2,
                "num_entries": 5,
                "entries": [{"index": i, "num_tokens": 50 + i} for i in range(5)],
            }
        )
    )

    model = MagicMock()
    model.config = MagicMock(num_hidden_layers=28)
    pc = MemoryAwarePrefixCache(model, MemoryCacheConfig(max_memory_mb=8))

    assert pc.load_from_disk(str(d)) == 0
