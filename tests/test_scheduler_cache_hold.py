# SPDX-License-Identifier: Apache-2.0
"""Integration test: Scheduler acquire/release wiring produces the right
behavior on the active defensive cache tier.

Pre-fix, MemoryAwarePrefixCache and PrefixCacheManager had a defensive
guard but no production callers — the guard could only be exercised via
the test-helper. Now add_request / remove_finished_request / abort
drive acquire / release, so clear() reflects real scheduler state.

These tests use mock model + tokenizer (no MLX/Metal). They exercise
Scheduler.add_request → cache.clear() refuses → Scheduler.remove_finished_request
→ cache.clear() allowed, through the same code paths used in production.
"""

from unittest.mock import MagicMock

import pytest

from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode = lambda x: list(range(len(x.split())))
    tokenizer.decode = lambda x: " ".join(str(t) for t in x)
    tokenizer.eos_token_id = 0
    tokenizer.eos_token_ids = {0}
    return tokenizer


@pytest.fixture
def mock_model():
    return MagicMock()


def _make_scheduler(mock_model, mock_tokenizer, *, use_memory_aware: bool):
    """Build a scheduler forced onto either the memory-aware or legacy
    prefix-cache tier. Paged/block-aware tiers have their own guards and
    are skipped for this test."""
    config = SchedulerConfig(
        enable_prefix_cache=True,
        use_paged_cache=False,
        use_memory_aware_cache=use_memory_aware,
    )
    return Scheduler(model=mock_model, tokenizer=mock_tokenizer, config=config)


def _submit(scheduler: Scheduler, request_id: str) -> Request:
    req = Request(
        request_id=request_id,
        prompt="hello world test prompt",
        sampling_params=SamplingParams(max_tokens=10),
    )
    scheduler.add_request(req)
    return req


def test_memory_aware_cache_clear_refused_while_request_in_flight(
    mock_model, mock_tokenizer
):
    """Scheduler.add_request must acquire on memory_aware_cache so clear()
    refuses, and remove_finished_request must release so clear() recovers."""
    scheduler = _make_scheduler(mock_model, mock_tokenizer, use_memory_aware=True)
    cache = scheduler.memory_aware_cache
    assert cache is not None, "expected memory-aware tier to be active"

    _submit(scheduler, "req-A")
    assert cache.clear() is False  # refused — req-A is live

    scheduler.remove_finished_request("req-A")
    assert cache.clear() is True   # released — clear succeeds


def test_prefix_cache_manager_clear_refused_while_request_in_flight(
    mock_model, mock_tokenizer
):
    """Same contract on the legacy PrefixCacheManager tier."""
    scheduler = _make_scheduler(mock_model, mock_tokenizer, use_memory_aware=False)
    cache = scheduler.prefix_cache
    assert cache is not None, "expected legacy prefix cache tier to be active"

    _submit(scheduler, "req-B")
    assert cache.clear() is False

    scheduler.remove_finished_request("req-B")
    assert cache.clear() is True


def test_abort_path_releases_cache_hold(mock_model, mock_tokenizer):
    """_do_abort_request must release so operators can clear after an
    abort even if the engine loop hasn't popped the finished request yet."""
    scheduler = _make_scheduler(mock_model, mock_tokenizer, use_memory_aware=True)
    cache = scheduler.memory_aware_cache
    assert cache is not None

    _submit(scheduler, "req-C")
    assert cache.clear() is False

    # Defer path: abort_request enqueues, then executor drains via
    # _process_pending_aborts. The public abort_request -> _process ->
    # _do_abort_request pipeline is what fires in production.
    scheduler.abort_request("req-C")
    scheduler._process_pending_aborts()

    assert cache.clear() is True


def test_multiple_requests_released_independently(mock_model, mock_tokenizer):
    scheduler = _make_scheduler(mock_model, mock_tokenizer, use_memory_aware=True)
    cache = scheduler.memory_aware_cache
    assert cache is not None

    _submit(scheduler, "req-D")
    _submit(scheduler, "req-E")
    assert cache._in_flight_count == 2
    assert cache.clear() is False

    scheduler.remove_finished_request("req-D")
    assert cache._in_flight_count == 1
    assert cache.clear() is False  # req-E still holds

    scheduler.remove_finished_request("req-E")
    assert cache.clear() is True


def test_paged_cache_path_does_not_require_acquire_release(
    mock_model, mock_tokenizer
):
    """When paged cache is active, memory_aware_cache and prefix_cache are
    both None, so the acquire/release helpers must no-op cleanly. The
    paged tier's guard is authoritative via block refcounts."""
    config = SchedulerConfig(
        enable_prefix_cache=True,
        use_paged_cache=True,
    )
    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer, config=config)
    assert scheduler.memory_aware_cache is None
    assert scheduler.prefix_cache is None
    # Must not raise — helpers tolerate missing tiers.
    scheduler._acquire_cache_hold("req-F")
    scheduler._release_cache_hold("req-F")
