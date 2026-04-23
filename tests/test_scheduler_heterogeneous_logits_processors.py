# SPDX-License-Identifier: Apache-2.0
"""Regression test for the ``logits_processors=None per-row`` crash in
mlx_lm.BatchGenerator when admitting heterogeneous requests (some with
processors, some without) and merging their GenerationBatches.

Root cause documented in ``/tmp/h1-repro/localization.md`` (2026-04-23).
The test exercises the mlx-lm API directly — no HTTP, no vllm_mlx
scheduler — so regressions here are bounded to the insert-kwarg
contract pinned by UPSTREAM_PIN.md invariant #18.
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.integration
def test_insert_without_logits_processors_does_not_leave_none_row():
    """When one request is inserted WITHOUT ``logits_processors=`` kwarg
    and another WITH a processor, running decode steps must not crash on
    ``for p in None`` inside ``GenerationBatch._step``.

    This pins UPSTREAM_PIN invariant #18: callers MUST pass
    ``logits_processors=[[per_row]]`` with ``per_row`` being a
    (possibly empty) list, never omit the kwarg.  Omitting lets
    mlx-lm's default slot in the BG-level None, which crashes
    ``GenerationBatch._step`` at line 1346 in mlx_lm 0.31.3.
    """
    pytest.importorskip("mlx_lm")
    if os.environ.get("VLLM_MLX_H1_REGRESSION_INTEG") != "1":
        pytest.skip("set VLLM_MLX_H1_REGRESSION_INTEG=1 to run (downloads model)")

    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator

    model, tokenizer = load("mlx-community/Qwen3-0.6B-8bit")
    bg = BatchGenerator(model, max_tokens=8)

    tokens_a = tokenizer.encode("Hello ", add_special_tokens=False)
    tokens_b = tokenizer.encode("World ", add_special_tokens=False)

    # Identity processor for request A.
    def noop_processor(token_context, logits):
        return logits

    # Request A — WITH per-row processor.
    uids_a = bg.insert([tokens_a], logits_processors=[[noop_processor]])

    # Request B — WITH empty per-row list (post-fix caller contract).
    # Pre-fix callers OMITTED the kwarg here, which is exactly what
    # triggered the crash documented in UPSTREAM_PIN invariant #18.
    uids_b = bg.insert([tokens_b], logits_processors=[[]])
    assert uids_a and uids_b

    # Drive steps until both prefills complete and at least one
    # generation step runs on the merged batch.  Pre-fix this would
    # have crashed; post-fix it must produce tokens cleanly.
    saw_gen_tokens = False
    for _ in range(64):
        result = bg.next()
        gens = (
            result[1]
            if isinstance(result, tuple) and len(result) == 2
            else (result or [])
        )
        if gens:
            saw_gen_tokens = True
        if gens and any(getattr(g, "finish_reason", None) for g in gens):
            break

    assert saw_gen_tokens, "drain produced no generation tokens — fixture broken"


@pytest.mark.integration
def test_insert_with_none_logits_processor_row_demonstrates_crash():
    """Negative-control / documentation test: directly passing
    ``logits_processors=[None]`` reproduces the mlx-lm crash that our
    caller-side contract prevents.

    Documented here (not asserted-to-pass) so future debuggers can see
    exactly what the broken call-shape looks like.  xfail because the
    mlx-lm bug itself has not been upstream-fixed yet — if a future
    mlx-lm release closes this, the test will XPASS and we can delete
    this case.
    """
    pytest.importorskip("mlx_lm")
    if os.environ.get("VLLM_MLX_H1_REGRESSION_INTEG") != "1":
        pytest.skip("set VLLM_MLX_H1_REGRESSION_INTEG=1 to run (downloads model)")

    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator

    model, tokenizer = load("mlx-community/Qwen3-0.6B-8bit")
    bg = BatchGenerator(model, max_tokens=8)

    tokens_a = tokenizer.encode("Hello ", add_special_tokens=False)
    tokens_b = tokenizer.encode("World ", add_special_tokens=False)

    def noop_processor(token_context, logits):
        return logits

    bg.insert([tokens_a], logits_processors=[[noop_processor]])
    # This is the broken call-shape: None at the per-row slot.  Pre-
    # caller-fix, this is what mlx-lm's default `self.logits_processors`
    # produced when the kwarg was omitted.
    bg.insert([tokens_b], logits_processors=[None])

    with pytest.raises(TypeError, match="'NoneType' object is not iterable"):
        for _ in range(64):
            result = bg.next()
            gens = (
                result[1]
                if isinstance(result, tuple) and len(result) == 2
                else (result or [])
            )
            if gens and any(getattr(g, "finish_reason", None) for g in gens):
                break
