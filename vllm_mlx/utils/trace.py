# SPDX-License-Identifier: Apache-2.0
"""Env-gated diagnostic trace for scheduler + batch-generator boundaries.

Set ``VLLM_MLX_SCHEDULER_TRACE=1`` to enable.  Optionally cap list-field
render via ``VLLM_MLX_SCHEDULER_TRACE_MAXLIST`` (default 20).

Lines emitted on the ``vllm_mlx.trace`` child logger at INFO with prefix
``[trace:sched]`` and a JSON body containing both:
- ``t``: ``time.perf_counter_ns()`` (in-process monotonic, for ordering)
- ``wall_ns``: ``time.time_ns()`` (wall clock, for cross-process correlation)

Trace I/O is routed through a ``QueueHandler`` whose underlying queue is
sized generously (100_000 slots) so the single-worker scheduler executor
thread never blocks on log flushing.  At emit-rate ~1000 lines/s with a
10-minute burst, total volume is ~600k — but the listener thread drains
continuously to the target handler, so steady-state queue depth stays
near-zero.  Queue fills only if the target handler itself blocks.

Error-path logging (trace serialization failures) uses the
``vllm_mlx.utils.trace`` logger (via ``__name__``) — a DIFFERENT logger
from ``vllm_mlx.trace``.  This is intentional: trace-emit errors should
go to normal server logs, not through the trace channel.

On queue overflow, ``QueueHandler`` drops records via ``handleError``
(stdlib behaviour: ``put_nowait`` raises ``queue.Full``, ``Handler.emit``
catches it, ``handleError`` writes to stderr).  The scheduler thread
never blocks on trace I/O.

``_TRACE_ENABLED`` and ``_MAX_LIST_RENDER`` are read once at module
import; set the env vars BEFORE ``import vllm_mlx`` or they will be
stale for the lifetime of the process.
"""

from __future__ import annotations

import contextlib
import json
import logging
import logging.handlers
import os
import queue
import time
from typing import Any

_TRACE_ENABLED: bool = os.environ.get("VLLM_MLX_SCHEDULER_TRACE") == "1"

_logger = logging.getLogger("vllm_mlx.trace")

_PREFIX = "[trace:sched]"

_MAX_LIST_RENDER = int(os.environ.get("VLLM_MLX_SCHEDULER_TRACE_MAXLIST", "20"))

_QUEUE_CAPACITY = 100_000


def is_trace_enabled() -> bool:
    """Return True iff VLLM_MLX_SCHEDULER_TRACE=1.

    PINNED AT IMPORT.  If the env var is set AFTER ``import vllm_mlx``, this
    still returns False.  Tests can monkey-patch ``_TRACE_ENABLED`` directly.
    """
    return _TRACE_ENABLED


def _truncate_list(v: Any) -> Any:
    if isinstance(v, list) and len(v) > _MAX_LIST_RENDER:
        return v[:_MAX_LIST_RENDER] + [f"...<+{len(v) - _MAX_LIST_RENDER}>"]
    return v


def _safe_default(o: Any) -> str:
    try:
        return repr(o)
    except Exception:
        return f"<unrepr-able {type(o).__name__} id={id(o):#x}>"


def emit(point: str, **fields: Any) -> None:
    """Emit one structured trace line.  No-op when trace disabled.

    Body always contains:
      t        — perf_counter_ns, in-process ordering
      wall_ns  — time.time_ns(), cross-process correlation
    Plus caller-supplied kwargs.
    """
    if not _TRACE_ENABLED:
        return
    try:
        body: dict[str, Any] = {
            "t": time.perf_counter_ns(),
            "wall_ns": time.time_ns(),
        }
        for k, v in fields.items():
            body[k] = _truncate_list(v)
        _logger.info(
            "%s %s %s", _PREFIX, point, json.dumps(body, default=_safe_default)
        )
    # Outer safety net: handler I/O failure during logger.info, or
    # time.* / json.dumps monkeypatched broken in tests.  _safe_default
    # already catches individual bad reprs, so this path fires only for
    # whole-emit failures.
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "trace_emit_error point=%s error=%s", point, e
        )


_queue_listener: logging.handlers.QueueListener | None = None


def install_trace_queue_handler(
    propagate: bool = False, handler: logging.Handler | None = None
) -> None:
    """Route ``vllm_mlx.trace`` through a QueueHandler + QueueListener.

    100k-slot queue; drops become impossible in practice at the emission
    rates this subsystem produces (~1k/s peak).  Idempotent-under-
    sequential-calls: prior listener is stopped before replacement.

    **Call during server startup before tracing begins.  Not safe to
    call concurrently with active trace emissions** — a small number of
    records in flight on the old handler may be orphaned in the stopped
    listener's queue.
    """
    global _queue_listener
    if _queue_listener is not None:
        with contextlib.suppress(Exception):
            _queue_listener.stop()
        _queue_listener = None

    target = handler or logging.StreamHandler()
    q: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=_QUEUE_CAPACITY)
    qh = logging.handlers.QueueHandler(q)
    listener = logging.handlers.QueueListener(q, target, respect_handler_level=True)
    for h in list(_logger.handlers):
        _logger.removeHandler(h)
    _logger.addHandler(qh)
    _logger.setLevel(logging.INFO)
    _logger.propagate = propagate
    listener.start()
    _queue_listener = listener
