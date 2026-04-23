# SPDX-License-Identifier: Apache-2.0
"""Pass-through trace shim over any mlx-lm-style BatchGenerator.

Field extraction is SHAPE-AWARE across the three real response types
(verified against mlx_lm 0.31.3 + vllm_mlx.mllm_batch_generator source):

- GenerationBatch.Response:  uid, token, finish_reason (plus heavy
  fields we intentionally drop: logprobs=mx.array, current_state,
  match_sequence, prompt_cache, all_tokens).
- PromptProcessingBatch.Response:  uid, progress, end_of_segment,
  end_of_prompt.
- MLLMBatchResponse:  uid, request_id, token, finish_reason.

Never calls repr() on raw items — logprobs contains mx.array and
repr() would force Metal sync, introducing the exact race class under
investigation.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any

from .trace import emit

_logger = logging.getLogger(__name__)

_MISSING = object()

# Whitelist of scalar fields extracted per response item.  Single source of
# truth — scheduler.py and mllm_scheduler.py's step.exit dict-comprehensions
# reference this same tuple so adding a new field is a one-place edit.  Never
# add fields that might hold mx.array (logprobs, current_state, etc.) — repr
# would force Metal sync and mask the race class under investigation.
SAFE_OUTPUT_FIELDS: tuple[str, ...] = (
    "uid",
    "request_id",
    "token",
    "finish_reason",
    "end_of_segment",
    "end_of_prompt",
    "progress",
)


def _extract_request_summary(request: Any) -> Any:
    try:
        if isinstance(request, dict):
            return {
                "uid": request.get("uid", request.get("request_id")),
                "tokens": (
                    len(request.get("prompt"))
                    if isinstance(request.get("prompt"), (list, tuple))
                    else None
                ),
            }
        return {
            "uid": getattr(request, "uid", getattr(request, "request_id", None)),
            "tokens": getattr(request, "num_prompt_tokens", None),
        }
    except Exception:
        return {"extract_err": True}


def _extract_output_summary(output: Any) -> dict:
    out: dict[str, Any] = {}
    for f in SAFE_OUTPUT_FIELDS:
        try:
            v = getattr(output, f, _MISSING)
            if v is _MISSING:
                continue
            out[f] = v
        except Exception:
            out[f] = "<get-err>"
    return out


class BatchGeneratorTraceShim:
    def __init__(self, bg: Any):
        self.__dict__["_bg"] = bg

    def insert(self, *args: Any, **kwargs: Any) -> Any:
        summaries = None
        if args and isinstance(args[0], (list, tuple)):
            try:
                summaries = [_extract_request_summary(r) for r in args[0]]
            except Exception:
                summaries = None
        try:
            emit(
                "bg.insert.enter",
                count=(
                    len(args[0])
                    if args and isinstance(args[0], (list, tuple))
                    else None
                ),
                requests=summaries,
                kwarg_keys=sorted(kwargs.keys()),
            )
        except Exception as e:  # noqa: BLE001
            _logger.warning("bg_trace_shim_enter_error point=bg.insert err=%s", e)

        t0 = time.perf_counter()
        try:
            result = self.__dict__["_bg"].insert(*args, **kwargs)
        except Exception as e:
            with contextlib.suppress(Exception):
                emit(
                    "bg.insert.exit",
                    raised=True,
                    error=repr(e),
                    elapsed_ms=round((time.perf_counter() - t0) * 1000, 3),
                )
            raise
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        try:
            uids_returned = list(result) if isinstance(result, (list, tuple)) else None
            emit("bg.insert.exit", uids_returned=uids_returned, elapsed_ms=elapsed_ms)
        except Exception as e:  # noqa: BLE001
            _logger.warning("bg_trace_shim_exit_error point=bg.insert err=%s", e)
        return result

    def next(self) -> Any:  # noqa: A003
        try:
            emit("bg.next.enter")
        except Exception as e:  # noqa: BLE001
            _logger.warning("bg_trace_shim_enter_error point=bg.next err=%s", e)

        t0 = time.perf_counter()
        try:
            result = self.__dict__["_bg"].next()
        except Exception as e:
            with contextlib.suppress(Exception):
                emit(
                    "bg.next.exit",
                    raised=True,
                    error=repr(e),
                    elapsed_ms=round((time.perf_counter() - t0) * 1000, 3),
                )
            raise
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        try:
            if isinstance(result, tuple) and len(result) == 2:
                prompt_part, gen_part = result
                emit(
                    "bg.next.exit",
                    prompt_outputs=[
                        _extract_output_summary(o) for o in (prompt_part or [])
                    ],
                    gen_outputs=[_extract_output_summary(o) for o in (gen_part or [])],
                    elapsed_ms=elapsed_ms,
                )
            elif result is None:
                emit("bg.next.exit", outputs=None, elapsed_ms=elapsed_ms)
            else:
                emit(
                    "bg.next.exit",
                    outputs=[_extract_output_summary(o) for o in result],
                    elapsed_ms=elapsed_ms,
                )
        except Exception as e:  # noqa: BLE001
            _logger.warning("bg_trace_shim_exit_error point=bg.next err=%s", e)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_bg"], name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.__dict__["_bg"], name, value)
