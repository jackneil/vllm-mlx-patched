# SPDX-License-Identifier: Apache-2.0
"""Server-side periodic prefix-cache flush: the background loop and the
CLI flags that configure it.

Persistence used to happen only at shutdown. A kernel panic — or a shutdown
save that failed under Metal pressure — threw away every warm prefix,
including the multi-minute Claude Code system-prompt prefill. The loop below
is what makes those prefixes durable while the server is still up, so its
two load-bearing properties are (a) it actually fires repeatedly and (b) a
raising flush never kills it.
"""

import ast
import asyncio
import inspect
import sys
import textwrap
import threading
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx import cli, server

# ---------------------------------------------------------------------------
# The periodic loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_periodic_loop_fires_repeatedly(monkeypatch):
    """The loop must keep flushing on its interval, not just once."""
    calls = []
    monkeypatch.setattr(server, "_cache_persist_interval_seconds", 0.01)
    monkeypatch.setattr(server, "_flush_prefix_cache_to_disk", lambda: calls.append(1))

    task = asyncio.create_task(server._periodic_cache_flush_loop())
    for _ in range(200):
        await asyncio.sleep(0.01)
        if len(calls) >= 3:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(calls) >= 3, f"loop must fire repeatedly, saw {len(calls)} flushes"


@pytest.mark.asyncio
async def test_a_raising_flush_does_not_kill_the_loop(monkeypatch):
    """One bad flush must not end persistence for the rest of the process's
    life — the next prefix is exactly the one worth saving."""
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("[metal::malloc] Resource limit (499000) exceeded.")

    monkeypatch.setattr(server, "_cache_persist_interval_seconds", 0.01)
    monkeypatch.setattr(server, "_flush_prefix_cache_to_disk", boom)

    task = asyncio.create_task(server._periodic_cache_flush_loop())
    for _ in range(200):
        await asyncio.sleep(0.01)
        if len(calls) >= 3:
            break
    still_alive = not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(calls) >= 3, f"loop must keep flushing after a raise, saw {len(calls)}"
    assert still_alive, "an exception in the flush must not terminate the loop"


@pytest.mark.asyncio
async def test_flush_helper_logs_the_saved_count(monkeypatch, caplog):
    """The operator's only visibility into incremental persistence is this
    log line, so it must carry the count the engine actually reported."""
    engine = MagicMock()
    engine.flush_cache_to_disk.return_value = 4
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_get_cache_dir", lambda: "/tmp/does-not-matter")

    with caplog.at_level("INFO"):
        server._flush_prefix_cache_to_disk()

    assert "[cache_persist] periodic flush: saved=4" in caplog.text
    engine.flush_cache_to_disk.assert_called_once_with("/tmp/does-not-matter")


@pytest.mark.asyncio
async def test_flush_helper_swallows_engine_errors(monkeypatch):
    """The helper mirrors _save_prefix_cache_to_disk: a failure is logged,
    never propagated into the caller."""
    engine = MagicMock()
    engine.flush_cache_to_disk.side_effect = RuntimeError("boom")
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_get_cache_dir", lambda: "/tmp/does-not-matter")

    server._flush_prefix_cache_to_disk()  # must not raise


@pytest.mark.asyncio
async def test_the_periodic_flush_runs_on_the_event_loop_thread(monkeypatch):
    """The flush must NOT be offloaded with asyncio.to_thread. Two reasons,
    both observed:

    * task.cancel() cannot stop a worker thread, so the lifespan shutdown save
      ran CONCURRENTLY with an in-flight flush over the same cache directory —
      interleaved, corrupt entry files that index.json recorded as durable.
    * MLX evaluates lazily against a per-thread stream, so serializing a KV
      cache off the loop thread raises "There is no Stream(gpu, N) in current
      thread" and the save silently fails.
    """
    seen = []
    loop_thread = threading.current_thread()
    monkeypatch.setattr(server, "_cache_persist_interval_seconds", 0.01)
    monkeypatch.setattr(
        server,
        "_flush_prefix_cache_to_disk",
        lambda: seen.append(threading.current_thread()),
    )

    task = asyncio.create_task(server._periodic_cache_flush_loop())
    for _ in range(200):
        await asyncio.sleep(0.01)
        if seen:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert seen, "the loop never flushed"
    assert all(t is loop_thread for t in seen), (
        f"the flush ran off the event-loop thread: "
        f"{[t.name for t in seen]} != {loop_thread.name}"
    )


def test_the_loop_does_not_offload_the_flush_to_a_worker_thread():
    """Source-level guard for the same regression, so re-introducing the
    offload is caught even if the timing test happens to pass. The overlap
    guard goes with it: with the flush on the loop thread, a tick physically
    cannot start while the previous one is still running."""
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(server._periodic_cache_flush_loop))
    )
    fn = tree.body[0]
    first = fn.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        fn.body = fn.body[1:]  # the docstring EXPLAINS the offload; skip it
    src = ast.unparse(fn)
    assert "to_thread" not in src, (
        "the periodic flush must run on the event-loop thread — see "
        "test_the_periodic_flush_runs_on_the_event_loop_thread"
    )
    assert not hasattr(
        server, "_cache_flush_running"
    ), "the overlap guard is dead code once the flush is synchronous"


# ---------------------------------------------------------------------------
# CLI threading — follows tests/test_cli_max_thinking_token_budget.py
# ---------------------------------------------------------------------------


def _parse_serve_args(*argv_extra):
    with (
        patch("vllm_mlx.server.load_model"),
        patch("uvicorn.run"),
        patch.object(sys, "argv", ["vllm-mlx", "serve", "some-model", *argv_extra]),
    ):
        try:
            cli.main()
        except SystemExit:
            pass


def test_interval_defaults_to_the_server_module_default():
    _parse_serve_args()
    assert (
        server._cache_persist_interval_seconds
        == server.CACHE_PERSIST_INTERVAL_SECONDS_DEFAULT
    )


def test_valid_interval_threads_into_the_server_global():
    _parse_serve_args("--cache-persist-interval-seconds", "45")
    assert server._cache_persist_interval_seconds == 45.0


def test_zero_interval_is_accepted_and_disables_periodic_flush():
    _parse_serve_args("--cache-persist-interval-seconds", "0")
    assert server._cache_persist_interval_seconds == 0.0


def test_negative_interval_is_rejected(capsys):
    _parse_serve_args("--cache-persist-interval-seconds", "-1")
    err = capsys.readouterr().err
    assert "--cache-persist-interval-seconds must be >= 0" in err


def test_zero_max_entries_is_rejected(capsys):
    _parse_serve_args("--cache-persist-max-entries", "0")
    err = capsys.readouterr().err
    assert "--cache-persist-max-entries must be >= 1" in err


def test_zero_max_gb_is_rejected(capsys):
    _parse_serve_args("--cache-persist-max-gb", "0")
    err = capsys.readouterr().err
    assert "--cache-persist-max-gb must be > 0" in err


def test_budgets_thread_into_the_scheduler_config():
    """The periodic flush and the shutdown save must share one budget, so
    the flags have to reach the MemoryCacheConfig the scheduler builds."""
    from vllm_mlx.scheduler import SchedulerConfig

    captured = {}
    real_init = SchedulerConfig.__init__

    def spy(self, *a, **kw):
        captured.update(kw)
        return real_init(self, *a, **kw)

    with patch.object(SchedulerConfig, "__init__", spy):
        _parse_serve_args(
            "--continuous-batching",
            "--cache-persist-max-entries",
            "7",
            "--cache-persist-max-gb",
            "2",
        )

    assert captured.get("cache_persist_max_entries") == 7
    assert captured.get("cache_persist_max_bytes") == 2 * 1024**3


def test_scheduler_config_budgets_reach_the_memory_cache_config():
    """SchedulerConfig's persist budgets must land on MemoryCacheConfig —
    otherwise the flags are inert."""
    from vllm_mlx.memory_cache import MemoryCacheConfig

    cfg = MemoryCacheConfig(persist_max_entries=7, persist_max_bytes=2 * 1024**3)
    assert cfg.persist_max_entries == 7
    assert cfg.persist_max_bytes == 2 * 1024**3
