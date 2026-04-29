# Stream Affinity Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore Qwen3.6 (and all post-mlx-lm-0.31+) inference in `vllm-mlx-patched` by binding `mlx_lm.generate.generation_stream` into the `mlx-step` worker thread, with a fallback path for cases where binding fails. Mirrors upstream PR #421 (waybarrios/vllm-mlx) without taking the full 221-commit rebase.

**Architecture:** Surgical forward-port from upstream's PR #421. Add `vllm_mlx/mlx_streams.py` (binding helper) and wire `ThreadPoolExecutor(initializer=bind_generation_streams)` in `engine_core._engine_loop`. Fallback to inline event-loop `scheduler.step` on stream-thread-mismatch errors. Add observability (distinct error code, response header, INFO log, breadcrumb log for the still-deferred Reg-A path). Lock the contract with a new UPSTREAM_PIN invariant #20 + AST sentinel + flipped-on H1 heterogeneous test.

**Tech Stack:** Python 3.13, mlx_lm 0.31+ (BatchGenerator API), vllm-mlx-patched fork at upstream pin `b4fa030`. Tests via pytest with anyio. AST sentinels via `ast.walk`. Mirror existing pattern in `tests/test_mlx_lm_api_contract.py`.

**Spec:** `/Users/jackneil/Github/hank-llm-arena/docs/superpowers/specs/2026-04-29-stream-affinity-fix-design.md`

---

## File Structure

| Action | Path | Responsibility |
|---|---|---|
| Create | `vllm_mlx/mlx_streams.py` | `bind_generation_streams()` helper — registers `mlx_lm.generate.generation_stream` in the calling thread |
| Modify | `vllm_mlx/engine_core.py` | Wire `ThreadPoolExecutor(initializer=...)` + `_is_stream_thread_error` fallback + preflight + positive telemetry log |
| Modify | `vllm_mlx/scheduler.py` | Add `_scheduler_async_eval` helper, refactor 8 direct async-eval sites onto it; add Reg-A breadcrumb log; add `error_code`/`error_message` to error path |
| Modify | `vllm_mlx/api/utils.py` (or wherever response headers are set on the chat-completion path) | Add `x-mlx-thread-local-stream: bound` response header |
| Create | `tests/test_engine_core_stream_safety.py` | Regression guard for the worker-thread stream-mismatch (cherry-picked from upstream `986dda9`, adapted for our hybrid executor) |
| Create | `tests/test_mlx_streams.py` | Unit test for `bind_generation_streams()` — both happy path and "raises in thread" failure mode |
| Modify | `tests/test_mlx_lm_api_contract.py` | Add `test_no_direct_async_eval_in_scheduler` AST sentinel mirroring the existing `_ALLOWLIST_FUNCS` pattern |
| Modify | `tests/test_scheduler_heterogeneous_logits_processors.py` | Flip from opt-in (env-gated) to default-on by replacing real-model load with stubbed BatchGenerator |
| Modify | `UPSTREAM_PIN.md` | Add invariant #20 documenting the new contract |

Total: 1 new module, 4 new/modified tests, 4 modified source files, 1 modified doc.

---

## Task 0: Branch + freeze

**Files:**
- Status check, no edits yet

- [ ] **Step 1: Confirm branch state**

```bash
cd /Users/jackneil/Github/vllm-mlx-patched
git branch --show-current  # expect: fix/stream-affinity-port-pr421
git status --short          # expect: clean
git log --oneline -3        # expect: HEAD = 44357a9 (test deepseek_v4 sentinel)
```

- [ ] **Step 2: Verify Python + dependency state**

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/bin/python -c "
import sys, mlx, mlx_lm
print('Python:', sys.version_info)
print('mlx:', mlx.__version__)
print('mlx_lm:', mlx_lm.__version__)
"
```
Expected: Python 3.12+/3.13, mlx_lm 0.31.x, mlx 0.x. If anything else, abort and fix the env first.

- [ ] **Step 3: Suggest /freeze for sensitive paths**

Run in your editor session: `/freeze /Users/jackneil/Github/vllm-mlx-patched/vllm_mlx/scheduler.py /Users/jackneil/Github/vllm-mlx-patched/vllm_mlx/engine_core.py /Users/jackneil/Github/vllm-mlx-patched/UPSTREAM_PIN.md`

This task is a no-op for code — proceed when steps 1 and 2 pass.

---

## Task 1: Cherry-pick the regression test (RED)

**Files:**
- Create: `tests/test_engine_core_stream_safety.py`

- [ ] **Step 1: Copy the upstream test into our fork**

```bash
cd /Users/jackneil/Github/vllm-mlx-patched
git show 986dda9:tests/test_engine_core_stream_safety.py > tests/test_engine_core_stream_safety.py
```

- [ ] **Step 2: Verify the test file looks right**

```bash
head -10 tests/test_engine_core_stream_safety.py
```
Expected: starts with `# SPDX-License-Identifier: Apache-2.0` and `"""Regression guard for issue #407.`

- [ ] **Step 3: Run the test against current HEAD — verify it FAILS with the expected error**

```bash
cd /Users/jackneil/Github/vllm-mlx-patched
/opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/bin/python -m pytest tests/test_engine_core_stream_safety.py -v 2>&1 | tail -30
```
Expected: FAIL. Look for either:
- `AssertionError: scheduler logged cross-thread stream errors: [...'no Stream(gpu, ...']`
- `AssertionError: batch generator was None after generation, meaning the scheduler's error-recovery path fired`

If the test ERRORS instead of FAILS (e.g. ImportError, model download failure), fix the import or model availability before proceeding. The test must demonstrably fail with the stream-thread error, not skip.

- [ ] **Step 4: Commit the failing test**

```bash
git add tests/test_engine_core_stream_safety.py
git commit -m "test(stream): regression guard for cross-thread stream error (RED)

Cherry-pick of waybarrios/vllm-mlx commit 986dda9 — pins the contract that
EngineCore must not log 'no Stream(gpu, N) in current thread' errors during
prefill+decode on Llama-3.2-1B (which reliably surfaces the mismatch where
Qwen3 is variable). Currently FAILS on this branch — Task 3 wires the fix."
```

---

## Task 2: Add `mlx_streams.py` (still RED — no wiring yet)

**Files:**
- Create: `vllm_mlx/mlx_streams.py`

- [ ] **Step 1: Cherry-pick the file from upstream**

```bash
cd /Users/jackneil/Github/vllm-mlx-patched
git show 0d8c6d1:vllm_mlx/mlx_streams.py > vllm_mlx/mlx_streams.py
```

- [ ] **Step 2: Verify the file content matches the spec design**

```bash
cat vllm_mlx/mlx_streams.py
```
Expected: imports `mlx.core as mx` + `importlib`, defines `bind_generation_streams(module_names: Iterable[str] = ("mlx_lm.generate", "mlx_vlm.generate"))`. Body: creates `default_stream = mx.new_stream(mx.default_device())`, calls `mx.set_default_stream(default_stream)`, monkey-patches `module.generation_stream = default_stream` on each named module. Returns the stream.

- [ ] **Step 3: Add a unit test for the helper itself**

Create `tests/test_mlx_streams.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for vllm_mlx.mlx_streams.bind_generation_streams.

Validates that the helper:
  - Returns a stream object.
  - Patches mlx_lm.generate.generation_stream on the module.
  - Tolerates missing modules (mlx_vlm not installed in some envs).
  - Works idempotently when called twice in the same thread.
"""
from __future__ import annotations

import importlib
import threading
import pytest


def test_bind_returns_a_stream():
    from vllm_mlx.mlx_streams import bind_generation_streams
    s = bind_generation_streams()
    assert s is not None


def test_bind_patches_mlx_lm_generation_stream():
    from vllm_mlx.mlx_streams import bind_generation_streams
    import mlx_lm.generate as gen
    original = gen.generation_stream
    new = bind_generation_streams()
    assert gen.generation_stream is new
    # idempotent
    again = bind_generation_streams()
    assert gen.generation_stream is again


def test_bind_tolerates_missing_module():
    from vllm_mlx.mlx_streams import bind_generation_streams
    # Use a definitely-not-importable name — must not raise.
    result = bind_generation_streams(module_names=("not_a_real_mlx_module_xyz",))
    assert result is not None


def test_bind_runs_in_worker_thread():
    """The helper must not crash when invoked from a non-main thread.
    This is the actual production scenario (mlx-step ThreadPoolExecutor)."""
    from vllm_mlx.mlx_streams import bind_generation_streams
    errors = []
    def worker():
        try:
            bind_generation_streams()
        except Exception as e:
            errors.append(e)
    t = threading.Thread(target=worker, name="test-worker")
    t.start()
    t.join(timeout=5)
    assert not errors, f"bind_generation_streams raised in worker thread: {errors}"
```

- [ ] **Step 4: Run the unit test — verify all 4 cases PASS**

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/bin/python -m pytest tests/test_mlx_streams.py -v 2>&1 | tail -20
```
Expected: 4 passed.

- [ ] **Step 5: Run the regression test from Task 1 again — should STILL FAIL**

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/bin/python -m pytest tests/test_engine_core_stream_safety.py -v 2>&1 | tail -10
```
Expected: STILL FAIL. The helper exists but isn't wired into the engine loop yet.

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/mlx_streams.py tests/test_mlx_streams.py
git commit -m "feat(streams): add bind_generation_streams helper from upstream PR #421

Forward-ports vllm_mlx/mlx_streams.py from waybarrios/vllm-mlx commit 0d8c6d1.
Helper registers mlx_lm.generate.generation_stream into the calling thread
by creating an mx.new_stream, calling mx.set_default_stream, and
monkey-patching module.generation_stream on mlx_lm.generate (and mlx_vlm.generate
when present).

Standalone unit test confirms: returns a stream, patches the module, tolerates
missing modules (mlx_vlm), and works in a non-main thread (the actual production
scenario via mlx-step ThreadPoolExecutor).

Helper is not yet wired into engine_core; Task 3 wires the ThreadPoolExecutor
initializer. The cross-thread regression test from Task 1 still FAILS."
```

---

## Task 3: Wire ThreadPoolExecutor initializer (GREEN)

**Files:**
- Modify: `vllm_mlx/engine_core.py:140-160` (the `_engine_loop` ThreadPoolExecutor construction)

- [ ] **Step 1: Read the current executor construction site**

```bash
grep -n "ThreadPoolExecutor\|mlx-step" /Users/jackneil/Github/vllm-mlx-patched/vllm_mlx/engine_core.py
```
Locate line ~157 where `_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-step")` is constructed.

- [ ] **Step 2: Add `from .mlx_streams import bind_generation_streams` to engine_core imports**

Locate the existing imports near the top of `engine_core.py` (around line 1-30). Add immediately after the other relative imports:

```python
from .mlx_streams import bind_generation_streams
```

- [ ] **Step 3: Add `initializer=bind_generation_streams` to the ThreadPoolExecutor construction**

Replace this block in `_engine_loop`:

```python
        _executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx-step"
        )
```

with:

```python
        # initializer runs once on the worker thread when it spawns. mlx-lm
        # 0.31+ (PR #1090) made generation_stream thread-local, so the worker
        # MUST register the stream before any scheduler.step / async-eval runs
        # there. See UPSTREAM_PIN.md invariant #20 + tests/test_engine_core_stream_safety.py.
        _executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mlx-step",
            initializer=bind_generation_streams,
        )
```

- [ ] **Step 4: Run the regression test — should now PASS**

```bash
cd /Users/jackneil/Github/vllm-mlx-patched
/opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/bin/python -m pytest tests/test_engine_core_stream_safety.py tests/test_mlx_streams.py -v 2>&1 | tail -15
```
Expected: PASS. If it still fails with `no Stream(gpu, ...)`, the initializer isn't running on the right thread — investigate `concurrent.futures.ThreadPoolExecutor` semantics or fall back to Pattern B (eager bind on first dispatch).

- [ ] **Step 5: Run the existing test suite — verify no regressions**

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/bin/python -m pytest tests/ -x --ignore=tests/test_scheduler_heterogeneous_logits_processors.py 2>&1 | tail -10
```
Expected: All passing (or any failures pre-exist, not introduced by this change). If any new failure references stream/thread/executor, investigate.

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/engine_core.py
git commit -m "fix(engine_core): bind generation streams in mlx-step worker thread

Closes the cross-thread stream-mismatch caused by mlx-lm PR #1090
(commit ed1fca4) making generation_stream thread-local. The mlx-step
ThreadPoolExecutor's initializer now calls bind_generation_streams() once
when the worker thread spawns, registering mlx_lm.generate.generation_stream
into that thread before any scheduler.step runs.

Mirrors waybarrios/vllm-mlx PR #421 commit 986dda9, but keeps our hybrid
executor pattern (worker thread for prefill, inline for generation-only)
intact rather than removing the worker thread entirely.

tests/test_engine_core_stream_safety.py now PASSES."
```

---

## Task 4: Add stream-thread-error fallback path

**Files:**
- Modify: `vllm_mlx/engine_core.py` (the `_engine_loop` worker-dispatch block, lines ~196-205)
- Create: test addition in `tests/test_engine_core_stream_safety.py`

- [ ] **Step 1: Add a new test that forces the bind to fail and asserts the fallback runs**

Append to `tests/test_engine_core_stream_safety.py`:

```python
@pytest.mark.anyio
async def test_engine_core_falls_back_when_bind_raises(
    model_and_tokenizer, caplog, monkeypatch
):
    """When the worker-thread bind fails (e.g. future MLX API drift), the
    engine loop must detect the stream-thread error and fall back to inline
    event-loop step instead of returning empty error responses to users.

    Tests UPSTREAM_PIN invariant #20 fallback contract.
    """
    import vllm_mlx.mlx_streams as mlx_streams
    from vllm_mlx import AsyncEngineCore, SamplingParams

    # Make bind a no-op so the worker thread runs WITHOUT registering the
    # stream — this is the post-#1090 broken state.
    monkeypatch.setattr(mlx_streams, "bind_generation_streams", lambda *a, **kw: None)

    model, tokenizer = model_and_tokenizer
    params = SamplingParams(max_tokens=5, temperature=0.0)

    caplog.set_level(logging.WARNING, logger="vllm_mlx.engine_core")

    engine = AsyncEngineCore(model, tokenizer)
    await engine.__aenter__()
    await asyncio.sleep(0.05)

    rid = await engine.add_request("Hello", params)
    tokens = 0
    async for out in engine.stream_outputs(rid, timeout=30):
        tokens += 1
        if out.finished:
            break
    await engine.__aexit__(None, None, None)

    assert tokens > 0, (
        "fallback path did not produce any tokens — the request was abandoned "
        "instead of being re-dispatched on the event-loop thread."
    )
    fallback_warns = [
        r.message for r in caplog.records
        if "stream_thread_fallback" in r.message or "scheduler.thread_local_stream_missing" in r.message
    ]
    assert fallback_warns, (
        "expected a one-shot WARN naming the fallback path; saw none."
    )
```

- [ ] **Step 2: Run the new test — verify it FAILS (no fallback wired yet)**

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/bin/python -m pytest tests/test_engine_core_stream_safety.py::test_engine_core_falls_back_when_bind_raises -v 2>&1 | tail -15
```
Expected: FAIL — `tokens` will be 0 because the request errors out instead of falling back.

- [ ] **Step 3: Add the `_is_stream_thread_error` helper to engine_core.py**

Locate the imports section near the top of `engine_core.py`. Just below the existing `from .mlx_streams import bind_generation_streams` line, add:

```python
def _is_stream_thread_error(error: Exception) -> bool:
    """True when MLX reports stream-ownership mismatch across threads.

    Mirrors waybarrios/vllm-mlx PR #421 commit 526f46c. Used by the engine
    loop to detect and fall back from the mlx-step worker thread to inline
    event-loop dispatch when the worker's stream registration is missing.
    """
    message = str(error)
    return "no Stream(" in message or "no Stream(gpu" in message
```

- [ ] **Step 4: Wrap the worker-thread `run_in_executor` call with try/except + fallback**

Locate the worker-dispatch block in `_engine_loop` (around line 196-205, the `if needs_executor: output = await loop.run_in_executor(...)` branch). Add a one-shot fallback flag in the loop's local scope (just after `_executor = ...`, around line 161):

```python
        _stream_thread_fallback_used = False
```

Then replace the worker-dispatch block:

```python
                    if needs_executor:
                        output = await loop.run_in_executor(
                            _executor, self.scheduler.step
                        )
                    else:
                        output = self.scheduler.step()
                        # Yield to event loop after inline step
                        await asyncio.sleep(0)
```

with:

```python
                    if needs_executor and not _stream_thread_fallback_used:
                        try:
                            output = await loop.run_in_executor(
                                _executor, self.scheduler.step
                            )
                        except RuntimeError as exc:
                            if not _is_stream_thread_error(exc):
                                raise
                            # MLX stream binding failed in the worker. Fall
                            # back to event-loop dispatch (where the stream
                            # was registered at module import). One-shot —
                            # subsequent calls go inline.
                            logger.warning(
                                "stream_thread_fallback engaging — error_code="
                                "scheduler.thread_local_stream_missing exc=%r. "
                                "Subsequent steps run inline. See UPSTREAM_PIN "
                                "invariant #20.",
                                exc,
                            )
                            _stream_thread_fallback_used = True
                            output = self.scheduler.step()
                            await asyncio.sleep(0)
                    else:
                        output = self.scheduler.step()
                        # Yield to event loop after inline step
                        await asyncio.sleep(0)
```

- [ ] **Step 5: Run the fallback test — verify it now PASSES**

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/bin/python -m pytest tests/test_engine_core_stream_safety.py -v 2>&1 | tail -15
```
Expected: 2 passed (the original + the new fallback test).

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/engine_core.py tests/test_engine_core_stream_safety.py
git commit -m "feat(engine_core): stream-thread-error fallback to event-loop dispatch

If bind_generation_streams fails or future MLX API drift breaks the
worker-thread registration, RuntimeError 'no Stream(gpu, N) in current
thread' from worker dispatch is now caught and the engine loop
re-dispatches on the event-loop thread (where the stream was registered
at module import). One-shot fallback flag prevents repeated retries.

Mirrors waybarrios/vllm-mlx PR #421 commit 526f46c. Pinned by new test
test_engine_core_falls_back_when_bind_raises."
```

---

## Task 5: Add `error_code` / `error_message` fields to error path

**Files:**
- Modify: `vllm_mlx/scheduler.py` (find `RequestOutput` construction in `_recover_from_generation_error` near line 2810)

- [ ] **Step 1: Locate where `RequestOutput(finish_reason="error")` is constructed**

```bash
grep -n "finish_reason=\"error\"\|finish_reason='error'" /Users/jackneil/Github/vllm-mlx-patched/vllm_mlx/scheduler.py
```
Locate the construction sites (likely around lines 2811-2818 in `_recover_from_generation_error`).

- [ ] **Step 2: Inspect `RequestOutput` definition**

```bash
grep -rn "class RequestOutput\b" /Users/jackneil/Github/vllm-mlx-patched/vllm_mlx/
```
Locate the class definition (probably in `vllm_mlx/api/types.py` or `vllm_mlx/types.py` or similar). Note its current fields.

- [ ] **Step 3: Write a failing test that asserts `error_code` is set when stream errors fire**

Create `tests/test_scheduler_error_codes.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Pin scheduler error_code population for regression-class diagnosis.

When _recover_from_generation_error fires, the resulting RequestOutput
must carry a distinct error_code so on-call can grep arena logs and tell
thread_local_stream_missing apart from cache_corruption_unrecoverable
apart from oom apart from generic.
"""
from __future__ import annotations

import pytest


def test_request_output_has_error_code_field():
    """RequestOutput must expose error_code (default empty string)
    so downstream emitters can carry the field through to API responses."""
    from vllm_mlx.api.types import RequestOutput  # adjust path if needed
    # Construct a minimal RequestOutput. Required fields will vary;
    # use whatever the existing __init__ requires + check error_code.
    # Adjust the constructor call to match the actual signature.
    # ...
    out = RequestOutput(request_id="test", finish_reason="error")
    assert hasattr(out, "error_code"), (
        "RequestOutput missing error_code field. Add per UPSTREAM_PIN invariant #20."
    )
    assert out.error_code in ("", None), (
        "error_code default must be empty/None — populated only on error paths."
    )


def test_recovery_path_populates_error_code():
    """_recover_from_generation_error must tag the RequestOutput with a
    distinct error_code reflecting the originating exception class."""
    pytest.skip(
        "TODO: requires synthesizing a Scheduler instance — test once "
        "RequestOutput.error_code field exists. Tracked as follow-up."
    )
```

(The second test is intentionally skipped — surfaces the gap but doesn't block this task.)

- [ ] **Step 4: Run the test — see it FAIL**

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/bin/python -m pytest tests/test_scheduler_error_codes.py -v 2>&1 | tail -10
```
Expected: FAIL on `assert hasattr(out, "error_code")`.

- [ ] **Step 5: Add `error_code` and `error_message` fields to `RequestOutput`**

Locate the `RequestOutput` dataclass / class definition. Add (post existing required fields):

```python
    error_code: str = ""
    error_message: str = ""
```

- [ ] **Step 6: Populate the fields in `_recover_from_generation_error`**

In `scheduler.py`, locate `_recover_from_generation_error` (around line 2811). Where it constructs `RequestOutput(..., finish_reason="error")`, classify the exception class and set `error_code`:

```python
    # Classify the recovery-triggering exception so callers can tell
    # thread_local_stream_missing from cache_corruption_unrecoverable
    # from generic.
    error_code = "scheduler.generic_recovery"
    error_message = f"{type(exc).__name__}: {exc}"
    msg = str(exc)
    if "no Stream(" in msg or "no Stream(gpu" in msg:
        error_code = "scheduler.thread_local_stream_missing"
    elif self._is_cache_corruption_error(exc):
        error_code = "scheduler.cache_corruption_unrecoverable"
    elif "out of memory" in msg.lower() or isinstance(exc, MemoryError):
        error_code = "scheduler.oom"
```

Then where `RequestOutput(...)` is constructed in this function, add `error_code=error_code, error_message=error_message`.

- [ ] **Step 7: Re-run the failing test — verify it now PASSES**

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/bin/python -m pytest tests/test_scheduler_error_codes.py -v 2>&1 | tail -10
```
Expected: 1 passed, 1 skipped.

- [ ] **Step 8: Commit**

```bash
git add vllm_mlx/scheduler.py vllm_mlx/api/types.py tests/test_scheduler_error_codes.py
git commit -m "feat(scheduler): distinct error_code in RequestOutput recovery path

Adds error_code + error_message fields to RequestOutput, populated by
_recover_from_generation_error with classified codes:
  - scheduler.thread_local_stream_missing
  - scheduler.cache_corruption_unrecoverable
  - scheduler.oom
  - scheduler.generic_recovery (default)

Lets arena's logs grep on a single token to distinguish regression classes
that all currently surface as finish_reason=error with empty content.

Pinned by tests/test_scheduler_error_codes.py."
```

(If the actual `RequestOutput` definition lives somewhere different, adjust the test import + the file modified in this task. Don't fight the existing structure.)

---

## Task 6: Add preflight self-test at scheduler startup

**Files:**
- Modify: `vllm_mlx/engine_core.py` (add a preflight method called from `_engine_loop` before the main loop body)

- [ ] **Step 1: Write the preflight test (failing) — assert preflight exists and aborts loud on failure**

Append to `tests/test_engine_core_stream_safety.py`:

```python
@pytest.mark.anyio
async def test_engine_core_preflight_aborts_when_bind_raises(
    model_and_tokenizer, caplog, monkeypatch
):
    """Engine_core preflight must run a no-op async-evaluation in the worker
    thread BEFORE accepting requests. If it raises, abort startup loud
    rather than failing on first user request."""
    import vllm_mlx.mlx_streams as mlx_streams
    from vllm_mlx import AsyncEngineCore

    def boom():
        raise RuntimeError("Simulated MLX init failure for test")
    monkeypatch.setattr(mlx_streams, "bind_generation_streams", boom)

    caplog.set_level(logging.ERROR, logger="vllm_mlx.engine_core")

    model, tokenizer = model_and_tokenizer
    engine = AsyncEngineCore(model, tokenizer)
    with pytest.raises((RuntimeError, SystemExit)) as exc_info:
        await engine.__aenter__()
        await asyncio.sleep(0.5)  # give preflight time to run + abort

    msg_text = str(exc_info.value).lower()
    fatal_logs = [r.message for r in caplog.records if "preflight" in r.message.lower()]
    assert fatal_logs, (
        "expected a fatal log line mentioning 'preflight' on stream init failure"
    )
```

- [ ] **Step 2: Run the test — verify it FAILS (no preflight yet)**

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/bin/python -m pytest tests/test_engine_core_stream_safety.py::test_engine_core_preflight_aborts_when_bind_raises -v 2>&1 | tail -10
```
Expected: FAIL — engine starts despite the broken bind.

- [ ] **Step 3: Add a `_preflight_executor_streams` async method on the engine class**

Locate the engine class (likely `EngineCore` or `AsyncEngineCore` in `engine_core.py`). Add a method (location: after `__init__`, before `_engine_loop`):

```python
    async def _preflight_executor_streams(self, executor) -> None:
        """Verify the worker thread can execute mlx async ops before serving.

        Submits a no-op async-evaluation to the executor and awaits its result.
        On RuntimeError matching the stream-thread-mismatch pattern, log fatal
        and re-raise so startup aborts with a clear pointer to UPSTREAM_PIN
        invariant #20 instead of failing on first user request.
        """
        import mlx.core as mx
        loop = asyncio.get_running_loop()

        def _probe():
            x = mx.array([1.0])
            mx.async_eval(x)
            mx.eval(x)
        try:
            await loop.run_in_executor(executor, _probe)
        except RuntimeError as exc:
            logger.error(
                "preflight FAILED on mlx-step worker thread: %r. "
                "This indicates the bind_generation_streams initializer did "
                "not register mlx_lm.generate.generation_stream in the worker. "
                "See UPSTREAM_PIN.md invariant #20 + vllm_mlx/mlx_streams.py.",
                exc,
            )
            raise
```

- [ ] **Step 4: Call the preflight from the engine loop's startup**

In `_engine_loop`, immediately after `_executor = concurrent.futures.ThreadPoolExecutor(...)` and before the `while self._running:` loop, add:

```python
        await self._preflight_executor_streams(_executor)
        logger.info(
            "action=stream_bound thread=mlx-step "
            "mlx_lm_version=%s preflight=ok",
            getattr(__import__("mlx_lm"), "__version__", "unknown"),
        )
```

- [ ] **Step 5: Re-run the preflight test — verify it now PASSES**

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/bin/python -m pytest tests/test_engine_core_stream_safety.py -v 2>&1 | tail -15
```
Expected: 3 tests passing in this file.

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/engine_core.py tests/test_engine_core_stream_safety.py
git commit -m "feat(engine_core): preflight stream binding + positive INFO log

Engine startup now submits a no-op async-evaluation to the mlx-step worker
thread before accepting requests. If the binding failed (e.g. future MLX
API drift), preflight aborts with a clear ERROR log naming UPSTREAM_PIN
invariant #20 + the mlx_streams.py fix path.

On success, emits a one-shot 'action=stream_bound thread=mlx-step
mlx_lm_version=<x> preflight=ok' INFO line so on-call can confirm via
grep that the patched build is live (mirrors invariant #10's 'invariant
#10 upheld' pattern)."
```

---

## Task 7: Add `x-mlx-thread-local-stream` response header

**Files:**
- Modify: wherever response headers are set on the chat-completion path. Most likely `vllm_mlx/server.py` or `vllm_mlx/api/utils.py`.

- [ ] **Step 1: Locate where existing response headers (e.g. `x-thinking-budget-applied`) are set**

```bash
grep -rn "x-thinking-budget-applied\|x-thinking-budget-resolved" /Users/jackneil/Github/vllm-mlx-patched/vllm_mlx/ --include="*.py"
```
Find the response object construction site for chat-completion non-stream and stream paths. Note the file:line.

- [ ] **Step 2: Add a module-global flag for "preflight done"**

In `engine_core.py`, near the top:

```python
# Set to True after the engine loop's preflight binding succeeds.
# Read by the response-header emitter to confirm the fix is live.
_stream_preflight_succeeded: bool = False
```

In `_engine_loop`, after `await self._preflight_executor_streams(...)` succeeds:

```python
        global _stream_preflight_succeeded
        _stream_preflight_succeeded = True
```

- [ ] **Step 3: Add the response header at each emitter site**

In the response header construction code identified in Step 1, alongside `x-thinking-budget-applied`, add:

```python
        from .engine_core import _stream_preflight_succeeded
        if _stream_preflight_succeeded:
            response.headers["x-mlx-thread-local-stream"] = "bound"
```

(Adjust import + response object reference to match the actual file.)

- [ ] **Step 4: Add a test that asserts the header is present**

Append to `tests/test_engine_core_stream_safety.py`:

```python
@pytest.mark.anyio
async def test_response_header_advertises_stream_binding(
    model_and_tokenizer
):
    """First successful response after preflight must carry
    x-mlx-thread-local-stream: bound so smoke runbook + arena can
    confirm the fix is live."""
    pytest.skip(
        "TODO: integration-level test — requires HTTP harness. "
        "Verify manually via curl. The unit-level proof is the flag."
    )
```

(Skip — the assertion is integration-level. The flag's value is the unit-level signal.)

- [ ] **Step 5: Run the unit-level path — verify the flag flips**

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/bin/python -c "
import vllm_mlx.engine_core as ec
print('flag default:', ec._stream_preflight_succeeded)
"
```
Expected: `flag default: False`. Then `Task 6`'s test exercises the flag — it should flip to True there.

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/engine_core.py vllm_mlx/server.py tests/test_engine_core_stream_safety.py
git commit -m "feat(observability): x-mlx-thread-local-stream response header

After preflight succeeds, every chat-completion response carries
x-mlx-thread-local-stream: bound. Lets the arena smoke runbook and
operators confirm via curl that the patched build is live without
reading source code or process logs.

Mirrors the existing x-thinking-budget-applied header pattern."
```

---

## Task 8: Reg-A breadcrumb log

**Files:**
- Modify: `vllm_mlx/scheduler.py` near the long-prompt path (the `_install_chunked_prefill` short-circuit warning is at lines 343-361; the per-request breadcrumb goes near where prompts are admitted in `Scheduler.add_request` or similar)

- [ ] **Step 1: Locate where prompt_tokens and prefill_step_size are visible**

```bash
grep -n "prefill_step_size\|prompt_tokens" /Users/jackneil/Github/vllm-mlx-patched/vllm_mlx/scheduler.py | head -20
```
Find the per-request handler (likely `add_request` or wherever a new request is admitted).

- [ ] **Step 2: Determine whether `_install_chunked_prefill` short-circuited at startup**

Read `scheduler.py:343-361`. The function returns early after logging the WARN, but doesn't set a flag. Add a Scheduler attribute:

```python
        # Set False if _install_chunked_prefill short-circuited due to
        # missing 'Batch' import (mlx_lm 0.31+ API drift). Used by the
        # add_request breadcrumb to log when long prompts hit the
        # disabled-chunked-prefill single-pass path. See UPSTREAM_PIN
        # invariant #10 + Reg-A documented in
        # /Users/jackneil/Github/hank-llm-arena/docs/superpowers/specs/
        # 2026-04-29-stream-affinity-fix-design.md
        self._chunked_prefill_enabled: bool = True
```

In `_install_chunked_prefill`'s ImportError branch (line 351-360), add as the last statement before `return`:

```python
        self._chunked_prefill_enabled = False
```

- [ ] **Step 3: Emit the per-request breadcrumb in the request-admission path**

Find `Scheduler.add_request` or equivalent. After the prompt_tokens length is computed, add:

```python
        if (
            not self._chunked_prefill_enabled
            and prompt_tokens is not None
            and len(prompt_tokens) > self.config.prefill_step_size
        ):
            logger.warning(
                "[chunked_prefill_disabled_path] request=%s prompt_tokens=%d "
                "going through single-pass prefill (chunked-prefill disabled "
                "since 2026-04-01: see scheduler.py:344 WARN at startup). "
                "May OOM on prompts >32k.",
                request_id, len(prompt_tokens),
            )
```

- [ ] **Step 4: Add a unit test that confirms the breadcrumb fires**

Create `tests/test_scheduler_chunked_prefill_breadcrumb.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Pin the Reg-A breadcrumb log: when chunked prefill is disabled
(_install_chunked_prefill ImportError'd because mlx_lm dropped Batch
in 0.31+) AND a long prompt arrives, scheduler emits a per-request WARN
so on-call can correlate downstream OOMs to the disabled feature.
"""
from __future__ import annotations

import logging
import pytest


def test_long_prompt_emits_disabled_path_warning(caplog, monkeypatch):
    from vllm_mlx.scheduler import Scheduler  # adjust if needed
    # Construct or stub a Scheduler with _chunked_prefill_enabled=False.
    # Submit a request with prompt_tokens > prefill_step_size.
    # Assert WARN log line contains "chunked_prefill_disabled_path" and
    # the prompt token count.
    pytest.skip(
        "TODO: requires Scheduler-construction helper. "
        "The breadcrumb is wired; integration-test via smoke runbook."
    )
```

(Skip the unit test — Scheduler construction is heavy; integration-test via smoke runbook is sufficient. The wiring is the contract.)

- [ ] **Step 5: Verify by inspection**

```bash
grep -n "chunked_prefill_disabled_path\|_chunked_prefill_enabled" /Users/jackneil/Github/vllm-mlx-patched/vllm_mlx/scheduler.py
```
Expected: 3 hits — the attribute init, the False-set in the WARN path, and the WARN at request admission.

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/scheduler.py tests/test_scheduler_chunked_prefill_breadcrumb.py
git commit -m "feat(scheduler): Reg-A breadcrumb log for disabled chunked prefill

When _install_chunked_prefill short-circuits at scheduler startup
(mlx_lm 0.31+ removed Batch — see UPSTREAM_PIN invariant #10 history),
sets self._chunked_prefill_enabled=False. Per-request handler emits a
WARN with prompt_tokens whenever a long prompt would have used chunked
prefill but is now going through single-pass.

Lets on-call grep correlation when a long-prompt OOM lands later. The
real fix (port _install_chunked_prefill to GenerationBatch /
PromptProcessingBatch) is deferred to a follow-up PR; this is the
breadcrumb until then."
```

---

## Task 9: AST sentinel — no direct async-eval calls outside scheduler helper

**Files:**
- Modify: `tests/test_mlx_lm_api_contract.py` (append a new test mirroring the existing `_ALLOWLIST_FUNCS` pattern)
- Modify: `vllm_mlx/scheduler.py` (add `_scheduler_async_eval` helper, refactor 8 sites)

- [ ] **Step 1: Read the existing AST sentinel pattern**

```bash
head -120 /Users/jackneil/Github/vllm-mlx-patched/tests/test_mlx_lm_api_contract.py
```
Note: `_collect_allowlist_line_ranges` walks the AST and returns line ranges of allowlisted function bodies. `_is_in_allowlist` checks if a given line is inside one. Mirror this pattern for the new sentinel.

- [ ] **Step 2: Append the new sentinel test**

Append to `tests/test_mlx_lm_api_contract.py`:

```python
# === New sentinel — no direct async-eval call outside the helper ===

# Allowlist: only the canonical scheduler helper may call mx.async_eval
# directly. Every other site MUST route through it. See UPSTREAM_PIN
# invariant #20 + 2026-04-29-stream-affinity-fix-design.md.
_ASYNC_EVAL_ALLOWLIST_FUNCS = {"_scheduler_async_eval"}


def test_no_direct_async_eval_in_scheduler():
    """scheduler.py must not call mx.async_eval directly outside the
    canonical _scheduler_async_eval helper.

    Regression-class guard: each direct call site is one more place a
    future mlx_lm threading change can resurface the 'no Stream(gpu, N)
    in current thread' bug. Routing through one helper means future
    threading-context changes touch one site, not eight.
    """
    src = _SCHEDULER.read_text()
    tree = ast.parse(src)

    # Compute allowlist line ranges for the new helper.
    allow_ranges = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _ASYNC_EVAL_ALLOWLIST_FUNCS:
                end = node.end_lineno or node.lineno
                allow_ranges.append((node.lineno, end))

    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match mx.async_eval(...)
        func = node.func
        is_target = (
            isinstance(func, ast.Attribute)
            and func.attr == "async_eval"
            and isinstance(func.value, ast.Name)
            and func.value.id == "mx"
        )
        if not is_target:
            continue
        line = node.lineno
        in_allowlist = any(start <= line <= end for start, end in allow_ranges)
        if not in_allowlist:
            violations.append(line)

    assert not violations, (
        "Direct mx.async_eval(...) call(s) found in scheduler.py outside "
        f"the {_ASYNC_EVAL_ALLOWLIST_FUNCS} helper(s) at lines: {violations}. "
        "Route through _scheduler_async_eval — see UPSTREAM_PIN invariant #20."
    )
```

- [ ] **Step 3: Run the new test — verify it FAILS**

```bash
cd /Users/jackneil/Github/vllm-mlx-patched
/opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/bin/python -m pytest tests/test_mlx_lm_api_contract.py::test_no_direct_async_eval_in_scheduler -v 2>&1 | tail -15
```
Expected: FAIL — message lists 8 violation line numbers (e.g. 408, 520, 584, 699, 967, 996, 1040, 1060).

- [ ] **Step 4: Add the `_scheduler_async_eval` helper at the top of `scheduler.py`**

Locate a good place near the top of `scheduler.py` (after imports, near other module-level helpers):

```python
def _scheduler_async_eval(*arrays):
    """Single canonical site for mx.async_eval inside vllm_mlx.scheduler.

    All async-eval call sites in this module MUST route through this helper.
    Routing through one site lets future MLX threading-context changes touch
    one place rather than eight. Pinned by
    tests/test_mlx_lm_api_contract.py::test_no_direct_async_eval_in_scheduler
    + UPSTREAM_PIN invariant #20.
    """
    import mlx.core as mx
    return mx.async_eval(*arrays)
```

- [ ] **Step 5: Refactor each direct `mx.async_eval(...)` call in scheduler.py to use the helper**

Replace each of the 8 occurrences. Pattern:
- `mx.async_eval(batch.y, batch.logprobs)` → `_scheduler_async_eval(batch.y, batch.logprobs)`
- Apply across all 8 sites. Use search-and-replace; verify each replacement keeps the same args.

```bash
grep -n "mx\.async_eval(" /Users/jackneil/Github/vllm-mlx-patched/vllm_mlx/scheduler.py
```
After refactor, this should ONLY show one match — inside `_scheduler_async_eval`.

- [ ] **Step 6: Re-run the AST sentinel — verify it now PASSES**

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/bin/python -m pytest tests/test_mlx_lm_api_contract.py -v 2>&1 | tail -15
```
Expected: All sentinel tests pass (the original `test_no_active_batch_or_split_batch_refs_outside_allowlist` + the new one).

- [ ] **Step 7: Run the broader test suite — verify no regressions from the refactor**

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/bin/python -m pytest tests/ -x --ignore=tests/test_scheduler_heterogeneous_logits_processors.py 2>&1 | tail -10
```
Expected: same pass/fail as Task 3 step 5.

- [ ] **Step 8: Commit**

```bash
git add vllm_mlx/scheduler.py tests/test_mlx_lm_api_contract.py
git commit -m "feat(scheduler): single _scheduler_async_eval helper + AST sentinel

Routes all 8 direct mx.async_eval call sites in vllm_mlx/scheduler.py
through one helper. Future MLX threading-context changes now touch one
site rather than scaling with the call count.

AST sentinel test_no_direct_async_eval_in_scheduler walks scheduler.py
and fails CI if any direct mx.async_eval call appears outside the
helper. Mirrors the existing _ALLOWLIST_FUNCS pattern for invariant #10
in the same file.

Pinned by UPSTREAM_PIN invariant #20."
```

---

## Task 10: Promote H1 heterogeneous test to default-on

**Files:**
- Modify: `tests/test_scheduler_heterogeneous_logits_processors.py`

- [ ] **Step 1: Read the current test gate**

```bash
head -80 /Users/jackneil/Github/vllm-mlx-patched/tests/test_scheduler_heterogeneous_logits_processors.py
```
Note the env-gate at line ~33 (`if os.environ.get("VLLM_MLX_H1_REGRESSION_INTEG") != "1": pytest.skip(...)`). Also note the `mlx-community/Qwen3-0.6B-8bit` model load.

- [ ] **Step 2: Replace the model load with a stub BatchGenerator**

Locate the `from mlx_lm import load` + `model, tokenizer = load("mlx-community/Qwen3-0.6B-8bit")` lines. Replace with a stub:

```python
    # Stub BatchGenerator — exercises the insert kwarg contract without
    # downloading a model. The H1 deadlock manifests on the kwarg-handling
    # path (mlx_lm BatchGenerator's `for p in None`), not on actual
    # generation tokens. Stubbing keeps this in default CI.
    class _StubModel:
        def __call__(self, *args, **kwargs):
            import mlx.core as mx
            # Return a zero-logit tensor sized to the batch.
            return mx.zeros((1, 32_000))
    class _StubTokenizer:
        def encode(self, text, add_special_tokens=False):
            # Map each character to a fake token id.
            return [ord(c) for c in text]

    model = _StubModel()
    tokenizer = _StubTokenizer()
```

- [ ] **Step 3: Remove the env-var gate**

Delete the lines:

```python
    if os.environ.get("VLLM_MLX_H1_REGRESSION_INTEG") != "1":
        pytest.skip("set VLLM_MLX_H1_REGRESSION_INTEG=1 to run (downloads model)")
```

(Keep the `pytest.importorskip("mlx_lm")` line at the top — that's environment-defensive, not opt-in.)

- [ ] **Step 4: Run the test — verify it still triggers the H1 contract**

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/bin/python -m pytest tests/test_scheduler_heterogeneous_logits_processors.py -v 2>&1 | tail -15
```
Expected: PASS (the kwarg contract is what's tested; the stub model is enough to exercise the BatchGenerator's insert path). If it FAILS for other reasons (e.g. the stub doesn't satisfy BatchGenerator's contract), iterate the stub until the test exercises the original H1 fix path. The goal is the test still PROVES the contract, not just runs.

- [ ] **Step 5: Commit**

```bash
git add tests/test_scheduler_heterogeneous_logits_processors.py
git commit -m "test(qwen3): flip H1 heterogeneous test to default-on with stubbed model

Replaces real-model load (mlx-community/Qwen3-0.6B-8bit, requires CI download
+ VLLM_MLX_H1_REGRESSION_INTEG=1 env gate) with a stub model/tokenizer that
satisfies BatchGenerator's insert kwarg contract — the actual subject of
UPSTREAM_PIN invariant #18.

Test now runs in default CI without downloads, mirroring the pattern at
tests/test_mlx_lm_arrays_cache_concurrent.py per invariant #17.

Closes Pre-Mortem Scenario 3 from the 2026-04-29 design review:
'5% concurrent regression — heterogeneous test was opt-in.'"
```

---

## Task 11: Add UPSTREAM_PIN invariant #20

**Files:**
- Modify: `UPSTREAM_PIN.md`

- [ ] **Step 1: Read the existing invariants to match style**

```bash
grep -n "^[0-9]\+\." /Users/jackneil/Github/vllm-mlx-patched/UPSTREAM_PIN.md | head -25
```
Find the highest existing invariant number (should be 19). The new one is #20.

- [ ] **Step 2: Append invariant #20 to UPSTREAM_PIN.md**

Append to UPSTREAM_PIN.md (after invariant #19):

```markdown

### Added 2026-04-29: thread-local generation stream binding

20. **Worker thread stream binding.** The `mlx-step` `ThreadPoolExecutor` in
    `vllm_mlx/engine_core.py::_engine_loop` MUST initialize each worker thread
    via `mlx_streams.bind_generation_streams` (passed as the `initializer=`
    kwarg). Required because mlx-lm 0.31+ (PR #1090, commit `ed1fca4`)
    made `mlx_lm.generate.generation_stream` thread-local — workers that
    don't bind it raise `RuntimeError: There is no Stream(gpu, N) in
    current thread` on the first `mx.async_eval`.

    The fallback at `engine_core.py::_is_stream_thread_error` MUST detect
    `RuntimeError` messages matching `"no Stream(gpu, "` and re-dispatch
    the failed request on the event-loop thread (one-shot WARN with
    `error_code=scheduler.thread_local_stream_missing`).

    All `mx.async_eval` call sites in `vllm_mlx/scheduler.py` MUST route
    through `_scheduler_async_eval`. Direct calls outside that helper
    are an immediate regression of this invariant.

    Pinned by:
    - `tests/test_engine_core_stream_safety.py` (3 tests: cross-thread error
      regression guard, fallback-on-bind-failure, preflight-aborts-loud)
    - `tests/test_mlx_streams.py` (4 unit tests for the helper)
    - `tests/test_mlx_lm_api_contract.py::test_no_direct_async_eval_in_scheduler`
      (AST sentinel — fails on any direct call outside the helper)

    References:
    - ml-explore/mlx-lm PR #1090 (commit `ed1fca4`): introduced thread-local
    - waybarrios/vllm-mlx PR #421: fix shipped 2026-04-24, commits
      `0d8c6d1` (mlx_streams.py), `986dda9` (test + initial wiring),
      `526f46c` (fallback path), `3f1377a` (mllm_scheduler follow-up)
    - Design spec: `hank-llm-arena/docs/superpowers/specs/2026-04-29-stream-affinity-fix-design.md`
```

- [ ] **Step 3: Run all sentinel tests — verify still passing**

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/bin/python -m pytest tests/test_mlx_lm_api_contract.py tests/test_mlx_streams.py tests/test_engine_core_stream_safety.py -v 2>&1 | tail -20
```
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add UPSTREAM_PIN.md
git commit -m "docs(UPSTREAM_PIN): add invariant #20 — thread-local stream binding

Documents the contract added by Tasks 1-10 of the stream-affinity fix:
- mlx-step ThreadPoolExecutor MUST initialize via bind_generation_streams
- _is_stream_thread_error MUST detect 'no Stream(gpu' and fall back inline
- All mx.async_eval calls in scheduler.py MUST route through
  _scheduler_async_eval (AST-sentinel-pinned)

References upstream PR #421 commits (waybarrios) + mlx-lm PR #1090
(ml-explore) so the next rebase has a clear pointer to what changed
and why."
```

---

## Task 12: Final smoke + push

**Files:**
- (No code changes)

- [ ] **Step 1: Run the full vllm-mlx-patched test suite**

```bash
cd /Users/jackneil/Github/vllm-mlx-patched
/opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/bin/python -m pytest tests/ 2>&1 | tail -15
```
Expected: all green (modulo any pre-existing skipped/integration tests).

- [ ] **Step 2: Visual check the diff scope**

```bash
git diff --stat 44357a9..HEAD
```
Expected: changes confined to `vllm_mlx/{mlx_streams.py, engine_core.py, scheduler.py, server.py}`, `tests/test_*.py` (new files + 1 modified), `UPSTREAM_PIN.md`.

- [ ] **Step 3: Push the branch**

```bash
git push -u origin fix/stream-affinity-port-pr421
```

- [ ] **Step 4: Open the PR**

```bash
gh pr create --title "fix(stream): port upstream PR #421 — bind generation_stream in mlx-step worker thread" --body "$(cat <<'EOF'
## Summary

Restores Qwen3.6 (and all post-mlx-lm-0.31+) inference. Forward-ports upstream waybarrios/vllm-mlx PR #421 (commits 0d8c6d1, 986dda9, 526f46c, 3f1377a, merged 2026-04-24) without taking the full 221-commit rebase.

**Root cause:** mlx-lm PR #1090 (commit ed1fca4, 2026-04-22) made generation streams thread-local. Our fork's hybrid-executor `_engine_loop` hands prefill to a `mlx-step` worker thread that never registered the stream → every Qwen3 chat call dies at the first `mx.async_eval` with `RuntimeError: There is no Stream(gpu, N) in current thread`.

**Fix:** Add `vllm_mlx/mlx_streams.py` with `bind_generation_streams()`. Wire `ThreadPoolExecutor(initializer=bind_generation_streams)` in engine_core. Add fallback path on stream-thread-error detection. Plus observability and pre-mortem mitigations.

## Changes

- New `vllm_mlx/mlx_streams.py` — binding helper (cherry-picked from upstream 0d8c6d1)
- `vllm_mlx/engine_core.py` — wire initializer, add `_is_stream_thread_error` + fallback, preflight self-test, positive INFO log + module flag for response header
- `vllm_mlx/scheduler.py` — `_scheduler_async_eval` helper + 8-site refactor, `error_code` + `error_message` on RequestOutput recovery path, Reg-A breadcrumb log
- `vllm_mlx/server.py` (or wherever response headers are set) — `x-mlx-thread-local-stream: bound` header
- `tests/test_engine_core_stream_safety.py` — 3 regression tests (cherry-picked from 986dda9 + 2 new)
- `tests/test_mlx_streams.py` — 4 unit tests for the helper
- `tests/test_mlx_lm_api_contract.py` — AST sentinel: no direct `mx.async_eval` in scheduler.py outside helper
- `tests/test_scheduler_heterogeneous_logits_processors.py` — flipped from opt-in to default-on with stubbed BatchGenerator
- `UPSTREAM_PIN.md` — invariant #20

## Test plan
- [x] All new tests pass
- [x] Existing test suite passes (no regressions)
- [x] AST sentinel + invariant #20 documented
- [ ] Manual: arena `cross-model-smoke-runbook.md` Smoke A green on Qwen3.6 across all 5 effort levels
- [ ] Manual: arena Smoke A on DeepSeek-V4-Flash-4bit (added in companion arena PR)
- [ ] Manual: response header `x-mlx-thread-local-stream: bound` present on first response

## Related
- Spec: `hank-llm-arena/docs/superpowers/specs/2026-04-29-stream-affinity-fix-design.md`
- Plan: `docs/superpowers/plans/2026-04-29-stream-affinity-fix.md`
- Upstream PR: https://github.com/waybarrios/vllm-mlx/pull/421
- Upstream mlx-lm PR: https://github.com/ml-explore/mlx-lm/pull/1090
EOF
)"
```

- [ ] **Step 5: Smoke against running arena**

After PR is open, manually run the smoke runbook from `hank-llm-arena/docs/cross-model-smoke-runbook.md` Smoke A on Qwen3.6 (kill the running serve, restart via arena admin so the new code loads). Verify all 5 effort cells green and `x-mlx-thread-local-stream: bound` header present.

If green, the fix is shippable. If anything fails, the smoke output is the next-task input.

