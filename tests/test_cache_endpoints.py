# SPDX-License-Identifier: Apache-2.0
"""Cache endpoint integration tests.

Covers GET /v1/cache/stats and DELETE /v1/cache, specifically the
extension that surfaces / clears the MemoryAwarePrefixCache (held on
app.state.prefix_cache) alongside the mlx_vlm multimodal tiers.
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient


def _build_app_with_prefix_cache(prefix_cache):
    """Attach prefix_cache to app.state.prefix_cache for the endpoint to read."""
    from vllm_mlx.server import app

    app.state.prefix_cache = prefix_cache
    return app


def test_clear_cache_calls_prefix_cache_clear_when_present():
    prefix_cache = MagicMock()
    prefix_cache.clear = MagicMock()

    app = _build_app_with_prefix_cache(prefix_cache)
    client = TestClient(app)

    resp = client.delete("/v1/cache")
    assert resp.status_code == 200
    body = resp.json()
    assert "llm_prefix" in body["caches"]
    prefix_cache.clear.assert_called_once()


def test_clear_cache_skips_prefix_cache_when_absent():
    """On a server without prefix cache."""
    from vllm_mlx.server import app

    app.state.prefix_cache = None
    client = TestClient(app)

    resp = client.delete("/v1/cache")
    assert resp.status_code == 200
    body = resp.json()
    assert "llm_prefix" not in body["caches"]


def test_cache_stats_includes_prefix_cache_when_present():
    prefix_cache = MagicMock()
    prefix_cache.get_stats = MagicMock(
        return_value={
            "entries": 42,
            "memory_bytes": 1_000_000,
        }
    )

    app = _build_app_with_prefix_cache(prefix_cache)
    client = TestClient(app)

    resp = client.get("/v1/cache/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "llm_prefix_cache" in body
    assert body["llm_prefix_cache"]["entries"] == 42


def test_cache_stats_omits_prefix_cache_when_absent():
    from vllm_mlx.server import app

    app.state.prefix_cache = None
    client = TestClient(app)

    resp = client.get("/v1/cache/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "llm_prefix_cache" not in body
