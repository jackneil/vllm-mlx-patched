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
    """Adapter now returns (all_cleared, refused_tiers). Handler surfaces
    both to the response body."""
    from vllm_mlx.server import app

    adapter = MagicMock()
    adapter.clear = MagicMock(return_value=(True, []))  # new contract
    app.state.prefix_cache = adapter

    client = TestClient(app)

    resp = client.delete("/v1/cache")
    assert resp.status_code == 200
    body = resp.json()
    caches = {c["name"]: c for c in body["caches"]}
    assert "llm_prefix" in caches
    assert caches["llm_prefix"]["cleared"] is True
    assert caches["llm_prefix"]["refused_tiers"] == []
    adapter.clear.assert_called_once()


def test_clear_cache_returns_409_when_any_tier_refuses():
    """Returns HTTP 409 when at least one LLM prefix tier refuses."""
    from vllm_mlx.server import app

    adapter = MagicMock()
    adapter.clear = MagicMock(return_value=(False, ["memory_aware_cache"]))
    app.state.prefix_cache = adapter

    client = TestClient(app)

    resp = client.delete("/v1/cache")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["status"] == "refused"
    assert detail["refused_tiers"] == ["memory_aware_cache"]
    assert (
        "in-flight" in detail["message"].lower()
        or "blocks" in detail["message"].lower()
        or "traffic" in detail["message"].lower()
    )


def test_clear_cache_skips_prefix_cache_when_absent():
    """On a server without prefix cache."""
    from vllm_mlx.server import app

    app.state.prefix_cache = None
    client = TestClient(app)

    resp = client.delete("/v1/cache")
    assert resp.status_code == 200
    body = resp.json()
    names = {c["name"] for c in body["caches"]} if body["caches"] else set()
    assert "llm_prefix" not in names


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


def test_adapter_clear_returns_empty_refused_when_no_scheduler():
    """No scheduler wired -> clear returns (True, []) — nothing to refuse."""
    from vllm_mlx.server import _PrefixCacheEndpointAdapter

    class _NoSchedEngine:
        _engine = None

    adapter = _PrefixCacheEndpointAdapter(_NoSchedEngine())
    assert adapter.clear() == (True, [])


def test_adapter_clear_aggregates_refusals_from_three_tiers():
    """Adapter walks memory_aware_cache, block_aware_cache, prefix_cache and
    collects tier names whose clear() returned False."""
    from vllm_mlx.server import _PrefixCacheEndpointAdapter

    scheduler = MagicMock()
    scheduler.memory_aware_cache.clear = MagicMock(return_value=False)
    scheduler.block_aware_cache.clear = MagicMock(return_value=True)
    scheduler.prefix_cache.clear = MagicMock(return_value=False)

    class _Engine:
        pass

    engine = _Engine()
    # Compose the nesting the adapter walks: engine._engine.engine.scheduler
    core = MagicMock()
    core.engine.scheduler = scheduler
    engine._engine = core

    adapter = _PrefixCacheEndpointAdapter(engine)
    all_cleared, refused = adapter.clear()
    assert all_cleared is False
    assert set(refused) == {"memory_aware_cache", "prefix_cache"}


def test_adapter_clear_treats_none_return_as_success():
    """Legacy tiers predating the bool contract return None; treat as success."""
    from vllm_mlx.server import _PrefixCacheEndpointAdapter

    scheduler = MagicMock()
    scheduler.memory_aware_cache.clear = MagicMock(return_value=None)
    scheduler.block_aware_cache.clear = MagicMock(return_value=True)
    # No prefix_cache attribute.
    del scheduler.prefix_cache

    class _Engine:
        pass

    engine = _Engine()
    core = MagicMock()
    core.engine.scheduler = scheduler
    engine._engine = core

    adapter = _PrefixCacheEndpointAdapter(engine)
    all_cleared, refused = adapter.clear()
    assert all_cleared is True
    assert refused == []
