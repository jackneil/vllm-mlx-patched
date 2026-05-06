# SPDX-License-Identifier: Apache-2.0
"""Tests for the EngineCore idle-branch cache reap.

Background: vllm-mlx processes accumulate Metal-pinned (wired) pages over
hours of uptime even when idle (no requests). The Metal allocator only
returns pages on `mx.clear_cache()`, but the existing call sites at
engine_core.py:229 and :269 live on the ACTIVE branch of the main loop.
An idle process never calls `mx.clear_cache()` and slowly leaks wired
pages — the proximate cause of the cascading hang documented in
mlx-lm/issues/883 and /1015.

These tests pin the idle-branch reap behavior:

  - reap fires when enough time has passed since the last reap
  - reap is a no-op when called more frequently than the interval
  - the interval is env-configurable via VLLM_MLX_IDLE_REAP_INTERVAL_S

The reap is gated on `mx.metal.is_available()` so the tests mock that
plus `mx.clear_cache` to avoid requiring a Metal device.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx import engine_core


# ---------------------------------------------------------------------------
# _maybe_reap_idle_cache predicate
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Minimal stand-in for EngineCore exposing only the reap state.

    Tests call _maybe_reap_idle_cache as an unbound method against this.
    """
    def __init__(self):
        self._last_idle_reap_ts: float = 0.0


def test_idle_reap_fires_after_interval():
    fake = _FakeEngine()
    with patch.object(engine_core.mx.metal, "is_available", return_value=True), \
         patch.object(engine_core.mx, "clear_cache") as clear_mock:
        result = engine_core.EngineCore._maybe_reap_idle_cache(fake, now=100.0)
    assert result is True
    assert clear_mock.call_count == 1
    assert fake._last_idle_reap_ts == 100.0


def test_idle_reap_skips_within_interval(monkeypatch):
    monkeypatch.setattr(engine_core, "IDLE_REAP_INTERVAL_S", 30.0)
    fake = _FakeEngine()
    fake._last_idle_reap_ts = 100.0
    with patch.object(engine_core.mx.metal, "is_available", return_value=True), \
         patch.object(engine_core.mx, "clear_cache") as clear_mock:
        result = engine_core.EngineCore._maybe_reap_idle_cache(fake, now=120.0)
    assert result is False, "second call within interval should be a no-op"
    assert clear_mock.call_count == 0


def test_idle_reap_fires_again_after_interval_elapses(monkeypatch):
    monkeypatch.setattr(engine_core, "IDLE_REAP_INTERVAL_S", 30.0)
    fake = _FakeEngine()
    fake._last_idle_reap_ts = 100.0
    with patch.object(engine_core.mx.metal, "is_available", return_value=True), \
         patch.object(engine_core.mx, "clear_cache") as clear_mock:
        # 100.0 + 31s > interval — should fire
        result = engine_core.EngineCore._maybe_reap_idle_cache(fake, now=131.0)
    assert result is True
    assert clear_mock.call_count == 1
    assert fake._last_idle_reap_ts == 131.0


def test_idle_reap_no_op_when_metal_unavailable():
    fake = _FakeEngine()
    with patch.object(engine_core.mx.metal, "is_available", return_value=False), \
         patch.object(engine_core.mx, "clear_cache") as clear_mock:
        result = engine_core.EngineCore._maybe_reap_idle_cache(fake, now=100.0)
    # Returning True only means "the time gate fired"; we expect clear_cache
    # to be skipped when metal isn't available, but the timestamp still
    # advances so we don't keep spinning the gate every tick.
    assert clear_mock.call_count == 0
    assert fake._last_idle_reap_ts == 100.0


def test_idle_reap_interval_env_configurable(monkeypatch):
    """IDLE_REAP_INTERVAL_S is set from VLLM_MLX_IDLE_REAP_INTERVAL_S env."""
    monkeypatch.setenv("VLLM_MLX_IDLE_REAP_INTERVAL_S", "5")
    # Force re-read of the constant (module-scope env reads happen at import)
    import importlib
    importlib.reload(engine_core)
    try:
        assert engine_core.IDLE_REAP_INTERVAL_S == 5.0
    finally:
        # Restore default for other tests
        monkeypatch.delenv("VLLM_MLX_IDLE_REAP_INTERVAL_S", raising=False)
        importlib.reload(engine_core)


def test_idle_reap_clear_cache_exception_does_not_propagate():
    """If clear_cache raises (rare but possible during heavy contention),
    the loop must not crash. Swallow + log."""
    fake = _FakeEngine()
    with patch.object(engine_core.mx.metal, "is_available", return_value=True), \
         patch.object(engine_core.mx, "clear_cache",
                      side_effect=RuntimeError("metal busy")):
        # Should not raise.
        result = engine_core.EngineCore._maybe_reap_idle_cache(fake, now=100.0)
    # Whether result is True or False is implementation-defined; the
    # important contract is no exception escapes.
    assert isinstance(result, bool)
