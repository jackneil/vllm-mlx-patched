# SPDX-License-Identifier: Apache-2.0
"""Live-server integration tests for Qwen3.x hybrid-cache concurrent-prefill bugs.

Regression lockdown for the bug family documented in
``docs/testing/2026-04-21-qwen3-35b-a3b-concurrent-heavy-payload-deadlock.md``.
Pinned by UPSTREAM_PIN.md invariant #17.

This file covers THREE concurrency shapes. Each is an independent test so
a failure names the failing shape in the pytest output:

    test_homogeneous_heavy_pair_regression_guard
        Two IDENTICAL heavy payloads fired concurrently. Regression guard
        for the partial fix shipped in PR #31 (mlx-lm#1169 + #1177 /
        ArraysCache.extend batch-dim). This was the shape verified
        fixed on 2026-04-22.

    test_heterogeneous_pair_current_failure_mode
        One heavy (~4.5k tokens, 3 tools, effort=high, thinking.adaptive)
        + one light (no system, no tools, "Reply OK") fired concurrently.
        This is Claude Code's real-world pattern (haiku tools=0 +
        sonnet tools=3 per scenario). As of 2026-04-22 this is
        **STILL BROKEN** — both requests hang after ``message_start``
        with no content_block_delta. Marked xfail with a pointer to
        the remaining H1 hypothesis (CB scheduler mispack of
        heterogeneous prefill batches). Remove the xfail when H1
        is closed.

    test_heterogeneous_pair_cross_family_gemma
        Same heterogeneous pair but targeted at
        ``mlx-community/gemma-4-26b-a4b-it-4bit`` (set via
        ``QWEN3_CONCURRENT_TEST_GEMMA_MODEL``). Gemma-4 routes through
        the MLLM scheduler with a different cache and is unaffected by
        the bug class. This test ensures any Qwen-targeted fix does
        not regress Gemma's CB path.

Run explicitly (requires a live vllm-mlx server)::

    # Start the server:
    vllm-mlx serve mlx-community/Qwen3.6-35B-A3B-4bit \\
        --host 127.0.0.1 --port 19001 --continuous-batching \\
        --enable-auto-tool-choice --tool-call-parser qwen3 \\
        --reasoning-parser qwen3 --max-thinking-token-budget 2048

    # Required env vars for Qwen tests:
    QWEN3_CONCURRENT_TEST_URL=http://127.0.0.1:19001 \\
    QWEN3_CONCURRENT_TEST_MODEL=mlx-community/Qwen3.6-35B-A3B-4bit \\
        pytest tests/test_qwen3_concurrent_heavy_payload.py -m integration -v

    # Optional: to also run the Gemma cross-family guard, either run the
    # arena's Gemma-4 server on its own port and set
    #   QWEN3_CONCURRENT_TEST_GEMMA_URL=http://127.0.0.1:8005
    #   QWEN3_CONCURRENT_TEST_GEMMA_MODEL=mlx-community/gemma-4-26b-a4b-it-4bit
    # OR set QWEN3_CONCURRENT_TEST_GEMMA_MODEL alone if a single arena proxy
    # routes models by ID (uses QWEN3_CONCURRENT_TEST_URL in that case).
    # Test skips with a clear message if either is missing.

Substitute ``Qwen3.5-35B-A3B-4bit`` freely — both were verified on the
original repro. Any hybrid-cache model (uses ArraysCache for
linear-attn / Gated-DeltaNet layers) reproduces the bug class.

Optional env knobs:
    QWEN3_CONCURRENT_TEST_CONCURRENCY=8  # default for the homogeneous
                                         # multi-request test only
    QWEN3_CONCURRENT_TEST_TIMEOUT_S=60   # per-request budget
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

URL = os.getenv("QWEN3_CONCURRENT_TEST_URL")
MODEL = os.getenv("QWEN3_CONCURRENT_TEST_MODEL")
GEMMA_URL = os.getenv("QWEN3_CONCURRENT_TEST_GEMMA_URL") or URL
GEMMA_MODEL = os.getenv("QWEN3_CONCURRENT_TEST_GEMMA_MODEL")
CONCURRENCY = int(os.getenv("QWEN3_CONCURRENT_TEST_CONCURRENCY", "8"))
TIMEOUT_S = float(os.getenv("QWEN3_CONCURRENT_TEST_TIMEOUT_S", "60"))

_HEAVY_PAYLOAD = (
    Path(__file__).parent.parent
    / "docs"
    / "testing"
    / "claude-shape-heavy-payload.json"
)

_VALID_STOP_REASONS = {"end_turn", "tool_use", "max_tokens", "stop_sequence"}

# Per bug doc success criteria: each request in a concurrent pair must
# complete within this budget. Pre-fix hangs until client timeout (30s+);
# post-fix on homogeneous completes in <4s; heterogeneous (when fixed)
# should also come in well under this budget.
_PAIR_COMPLETION_BUDGET_S = 15.0


def _skip_if_not_configured(require_gemma: bool = False) -> None:
    if not URL or not MODEL:
        pytest.skip(
            "Set QWEN3_CONCURRENT_TEST_URL=http://host:port and "
            "QWEN3_CONCURRENT_TEST_MODEL=mlx-community/Qwen3.6-35B-A3B-4bit "
            "(or another hybrid-cache model) to run."
        )
    if require_gemma and not GEMMA_MODEL:
        pytest.skip(
            "Set QWEN3_CONCURRENT_TEST_GEMMA_MODEL=mlx-community/gemma-4-26b-a4b-it-4bit "
            "(and optionally QWEN3_CONCURRENT_TEST_GEMMA_URL for a different host/port) "
            "to run the Gemma cross-family non-regression guard."
        )
    if not _HEAVY_PAYLOAD.exists():
        pytest.skip(f"Heavy payload fixture missing: {_HEAVY_PAYLOAD}")


def _load_base_heavy(model_id: str | None = None) -> dict:
    with _HEAVY_PAYLOAD.open() as f:
        base = json.load(f)
    base["model"] = model_id or MODEL
    return base


def _build_light_payload(model_id: str | None = None) -> dict:
    """Minimal, tool-less, system-less light payload — mirrors Claude Code's
    haiku-shape request that triggers the heterogeneous deadlock when
    paired with the heavy payload.
    """
    return {
        "model": model_id or MODEL,
        "max_tokens": 200,
        "stream": True,
        "output_config": {"effort": "high"},
        "thinking": {"type": "adaptive"},
        "messages": [{"role": "user", "content": "Reply with OK"}],
    }


def _inject_nonce(payload: dict, idx: int) -> dict:
    """Return a deep-copied payload with a unique-per-request nonce injected
    into BOTH the system prompt (if present) and the user message.

    Used for homogeneous N-concurrent tests where all requests share the
    same fixture. Guarantees prefix-cache divergence at token 0. For the
    heterogeneous pair, the two requests are already distinguishable by
    their differing shapes, so nonce injection is not required.
    """
    d = copy.deepcopy(payload)
    nonce = uuid.uuid4().hex
    tag = f"[req-{idx} nonce={nonce}] "

    system = d.get("system")
    if isinstance(system, list) and system:
        block = system[0]
        if isinstance(block, dict) and "text" in block:
            block["text"] = tag + block["text"]
        else:
            system.insert(0, {"type": "text", "text": tag})
    elif isinstance(system, str):
        d["system"] = tag + system
    elif system is not None:
        d["system"] = tag.rstrip()

    if d.get("messages"):
        msg = d["messages"][0]
        content = msg.get("content")
        if isinstance(content, list) and content:
            block = content[0]
            if isinstance(block, dict) and "text" in block:
                block["text"] = tag + block["text"]
            else:
                content.insert(0, {"type": "text", "text": tag})
        else:
            msg["content"] = tag + str(content or "")
    return d


async def _one_request(session, idx: int, url: str, payload: dict) -> dict:
    """Fire one streaming POST to `url`/v1/messages, accumulate events,
    return a per-request summary."""
    summary = {
        "idx": idx,
        "http_status": None,
        "events": 0,
        "message_stop_count": 0,
        "content_block_delta_count": 0,
        "stop_reason": None,
        "output_tokens": None,
        "elapsed_s": None,
        "exception": None,
    }
    loop = asyncio.get_event_loop()
    start = loop.time()
    try:
        async with session.stream(
            "POST",
            f"{url.rstrip('/')}/v1/messages",
            json=payload,
            timeout=TIMEOUT_S,
            headers={"Content-Type": "application/json"},
        ) as resp:
            summary["http_status"] = resp.status_code
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    summary["events"] += 1
                    name = line.split(":", 1)[1].strip()
                    if name == "message_stop":
                        summary["message_stop_count"] += 1
                    elif name == "content_block_delta":
                        summary["content_block_delta_count"] += 1
                elif line.startswith("data:"):
                    blob = line[len("data:") :].strip()
                    if not blob:
                        continue
                    try:
                        obj = json.loads(blob)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") == "message_delta":
                        delta = obj.get("delta") or {}
                        if "stop_reason" in delta:
                            summary["stop_reason"] = delta["stop_reason"]
                        usage = obj.get("usage") or {}
                        if "output_tokens" in usage:
                            summary["output_tokens"] = usage["output_tokens"]
    except Exception as exc:  # noqa: BLE001 — diagnostic surface
        summary["exception"] = repr(exc)
    summary["elapsed_s"] = loop.time() - start
    return summary


def _validate_request_result(r: dict, label: str) -> list[str]:
    """Return a list of failure strings for the result. Empty list = pass."""
    failures: list[str] = []
    if r["exception"]:
        failures.append(
            f"{label}-req-{r['idx']}: raised {r['exception']} "
            f"(http_status={r['http_status']}, elapsed={r['elapsed_s']:.2f}s)"
        )
        return failures
    if r["http_status"] != 200:
        failures.append(f"{label}-req-{r['idx']}: HTTP {r['http_status']}")
        return failures
    if r["content_block_delta_count"] < 1:
        failures.append(
            f"{label}-req-{r['idx']}: ZERO content_block_delta events "
            f"(events={r['events']}, output_tokens={r['output_tokens']}, "
            f"stop_reason={r['stop_reason']}, elapsed={r['elapsed_s']:.2f}s). "
            "Matches the bug doc's 'hang after message_start' signature — "
            "see H1 (CB scheduler mispack of mixed-shape prefill)."
        )
        return failures
    if not r["output_tokens"] or r["output_tokens"] < 1:
        failures.append(
            f"{label}-req-{r['idx']}: output_tokens={r['output_tokens']} "
            f"(expected >=1); events={r['events']}, stop_reason={r['stop_reason']}."
        )
        return failures
    if r["message_stop_count"] < 1:
        failures.append(
            f"{label}-req-{r['idx']}: missing message_stop event "
            f"(events={r['events']}, stop_reason={r['stop_reason']})"
        )
        return failures
    if r["stop_reason"] not in _VALID_STOP_REASONS:
        failures.append(
            f"{label}-req-{r['idx']}: stop_reason {r['stop_reason']!r} not in "
            f"{sorted(_VALID_STOP_REASONS)}."
        )
    if r["elapsed_s"] > _PAIR_COMPLETION_BUDGET_S:
        failures.append(
            f"{label}-req-{r['idx']}: elapsed={r['elapsed_s']:.2f}s exceeded "
            f"budget {_PAIR_COMPLETION_BUDGET_S}s. Pre-fix hangs up to 30s; "
            "post-fix homogeneous completes in <4s; a slow-but-not-hung result "
            "suggests partial degradation."
        )
    return failures


@pytest.mark.asyncio
async def test_homogeneous_heavy_pair_regression_guard():
    """Regression guard for the partial fix shipped in PR #31.

    Two IDENTICAL heavy payloads fired concurrently must both complete
    with proper streaming output inside the pair completion budget.
    Before PR #31 (mlx-lm#1169 + #1177 ArraysCache.extend fix) these
    produced ``content_block_delta_count=0`` degenerate responses.

    If this test fails, the mlx-lm pin has regressed — check the unit
    sentinel ``test_mlx_lm_arrays_cache_concurrent.py`` first.
    """
    _skip_if_not_configured()
    httpx = pytest.importorskip("httpx")

    base = _load_base_heavy()
    # Two IDENTICAL payloads (nonce injected to defeat prefix-cache dedup
    # without changing shape).
    payloads = [_inject_nonce(base, idx=i) for i in range(1, 3)]

    async with httpx.AsyncClient() as session:
        results = await asyncio.gather(
            *(
                _one_request(session, i, URL, p)
                for i, p in enumerate(payloads, start=1)
            ),
            return_exceptions=False,
        )

    failures: list[str] = []
    for r in results:
        failures.extend(_validate_request_result(r, "homogeneous-heavy"))

    assert not failures, (
        f"{len(failures)} homogeneous-heavy pair failures:\n  "
        + "\n  ".join(failures)
        + f"\n\nModel: {MODEL}  URL: {URL}\n"
        "This is the REGRESSION GUARD for PR #31. If this fails, the "
        "ArraysCache.extend pin has regressed — run "
        "pytest tests/test_mlx_lm_arrays_cache_concurrent.py first."
    )


@pytest.mark.asyncio
async def test_heterogeneous_pair_current_failure_mode():
    """Heterogeneous pair: one heavy + one light payload at the SAME Qwen
    hybrid-cache model. Closed by PR #31 (mlx-lm 0.31.3 pin).

    This is Claude Code's real-world pattern: every scenario fires one
    haiku request (``tools=0``, minimal) plus one sonnet request
    (``tools=3``, full system + tool schemas) concurrently at the same
    model after proxy rewrite. Pre-PR-#31, both hung at ``message_start``
    or produced ``!!!`` degenerate output via the mlx-lm#1169 middle-wave
    neighbour state contamination path. Verified fixed via 10-pair burst
    on mlx-lm 0.31.3 (20/20 requests pass with distinct proper output).

    Kept as a hard regression guard — if the mlx-lm pin regresses below
    0.31.3 (invariant #17) or the deployment env reverts, this will fail
    loudly.
    """
    _skip_if_not_configured()
    httpx = pytest.importorskip("httpx")

    heavy = _load_base_heavy()
    light = _build_light_payload()

    async with httpx.AsyncClient() as session:
        results = await asyncio.gather(
            _one_request(session, 1, URL, heavy),
            _one_request(session, 2, URL, light),
            return_exceptions=False,
        )

    labels = ["heavy", "light"]
    failures: list[str] = []
    for r, label in zip(results, labels):
        failures.extend(_validate_request_result(r, f"heterogeneous-{label}"))

    assert not failures, (
        f"{len(failures)} heterogeneous-pair failures:\n  "
        + "\n  ".join(failures)
        + f"\n\nModel: {MODEL}  URL: {URL}\n"
        "Expected path (once H1 is fixed): both requests complete in <15s "
        "with stop_reason=end_turn and >=1 content_block_delta each. "
        "Current failure signature (pre-H1-fix): both hang 30s, events=1, "
        "stops=0, content_block_delta_count=0."
    )


@pytest.mark.asyncio
async def test_staggered_pair_50ms_current_failure_mode():
    """PRIMARY open-bug repro: heavy fires at T+0, light fires at T+50ms at
    the SAME Qwen hybrid-cache model. Both must complete with proper
    streaming output.

    This is Claude Code's actual real-world pattern — `hank-secure-llm`
    proxy logs show haiku + sonnet arriving 30-70ms apart at the arena
    during every `claude --bare -p` invocation. The same pair fired
    simultaneously (asyncio.gather, no sleep) passes after PR #31, which
    is why the simultaneous test `test_heterogeneous_pair_current_failure_mode`
    is already green. The 50ms stagger puts the second request into a
    LATER CB scheduler step, exercising a join-mid-batch path PR #31
    didn't fix.

    Pre-fix failure signature (observed on prod post-PR-#31, 3/3 runs):
        - heavy (started T+0):  3 events (message_start + content_block_start),
                                 0 message_stop, 0 content_block_delta
        - light (started T+50): 1 event  (message_start only),
                                 0 message_stop, 0 content_block_delta

    Passing criterion: both requests produce >=1 content_block_delta,
    a valid stop_reason, and a message_stop within the per-pair budget.
    """
    _skip_if_not_configured()
    httpx = pytest.importorskip("httpx")

    heavy = _load_base_heavy()
    light = _build_light_payload()

    async with httpx.AsyncClient() as session:
        # Fire heavy first; wait 50ms so it lands in CB step N and is
        # mid-prefill when light arrives and tries to join step N+1.
        heavy_task = asyncio.create_task(_one_request(session, 1, URL, heavy))
        await asyncio.sleep(0.05)
        light_task = asyncio.create_task(_one_request(session, 2, URL, light))
        results = await asyncio.gather(heavy_task, light_task)

    labels = ["heavy", "light"]
    failures: list[str] = []
    for r, label in zip(results, labels):
        failures.extend(_validate_request_result(r, f"staggered-{label}"))

    assert not failures, (
        f"{len(failures)} staggered-pair (50ms) failures:\n  "
        + "\n  ".join(failures)
        + f"\n\nModel: {MODEL}  URL: {URL}\n"
        "This is the PRIMARY open bug as of 2026-04-22 evening. Claude Code's "
        "haiku+sonnet fan-out hits this shape deterministically. PR #31 closed "
        "the simultaneous case but a second shared-state bug remains in the "
        "scheduler join-mid-batch path. See "
        "docs/testing/2026-04-21-qwen3-35b-a3b-concurrent-heavy-payload-deadlock.md "
        '"Symptom (refined 2026-04-22 evening, post-PR-#31 deployment)".'
    )


@pytest.mark.asyncio
async def test_staggered_pair_50ms_cross_family_gemma():
    """Cross-family guard: the 50ms-staggered pair against Gemma-4-26b-a4b
    must continue to work (it does — Gemma routes through MLLMScheduler).

    Same pattern as ``test_staggered_pair_50ms_current_failure_mode`` but
    targeted at Gemma. Any Qwen-targeted fix must not regress this path.
    Per bug doc: Gemma completes the staggered pair in ~0.4s + ~0.9s.
    """
    _skip_if_not_configured(require_gemma=True)
    httpx = pytest.importorskip("httpx")

    heavy = _load_base_heavy(model_id=GEMMA_MODEL)
    light = _build_light_payload(model_id=GEMMA_MODEL)

    async with httpx.AsyncClient() as session:
        heavy_task = asyncio.create_task(_one_request(session, 1, GEMMA_URL, heavy))
        await asyncio.sleep(0.05)
        light_task = asyncio.create_task(_one_request(session, 2, GEMMA_URL, light))
        results = await asyncio.gather(heavy_task, light_task)

    labels = ["heavy", "light"]
    failures: list[str] = []
    for r, label in zip(results, labels):
        failures.extend(_validate_request_result(r, f"gemma-staggered-{label}"))

    assert not failures, (
        f"{len(failures)} Gemma cross-family staggered-pair failures:\n  "
        + "\n  ".join(failures)
        + f"\n\nModel: {GEMMA_MODEL}  URL: {GEMMA_URL}\n"
        "Gemma-4 should be unaffected — regression means the Qwen fix "
        "leaked into MLLMScheduler."
    )


@pytest.mark.asyncio
async def test_heterogeneous_pair_cross_family_gemma():
    """Cross-family non-regression guard: the same heterogeneous pair
    pattern against Gemma-4-26b-a4b-it-4bit must continue to work.

    Gemma-4 routes through the MLLM scheduler with a different cache
    (``MLLMCacheManager``, not ``ArraysCache``) and is unaffected by
    the Qwen3.x hybrid-cache bug family. Any Qwen-specific fix must
    not regress this path. Per bug doc: Gemma completes the same pair
    in ~2.6s (homogeneous was ~4.8s).

    Skips if ``QWEN3_CONCURRENT_TEST_GEMMA_MODEL`` is not set.
    """
    _skip_if_not_configured(require_gemma=True)
    httpx = pytest.importorskip("httpx")

    heavy = _load_base_heavy(model_id=GEMMA_MODEL)
    light = _build_light_payload(model_id=GEMMA_MODEL)

    async with httpx.AsyncClient() as session:
        results = await asyncio.gather(
            _one_request(session, 1, GEMMA_URL, heavy),
            _one_request(session, 2, GEMMA_URL, light),
            return_exceptions=False,
        )

    labels = ["heavy", "light"]
    failures: list[str] = []
    for r, label in zip(results, labels):
        failures.extend(_validate_request_result(r, f"gemma-heterogeneous-{label}"))

    assert not failures, (
        f"{len(failures)} Gemma cross-family heterogeneous-pair failures:\n  "
        + "\n  ".join(failures)
        + f"\n\nModel: {GEMMA_MODEL}  URL: {GEMMA_URL}\n"
        "Gemma-4 should be unaffected — if this fails the Qwen-specific "
        "fix has regressed Gemma's MLLM scheduler path. Bisect against "
        "the last known-good mlx_lm pin and UPSTREAM_PIN.md invariant #17."
    )


@pytest.mark.asyncio
async def test_homogeneous_multi_concurrent_heavy_payload():
    """Extended regression guard: N concurrent UNIQUE-prefix heavy payloads
    (default 8) all complete with proper output.

    This is the original PR #31 integration test — stressing the
    ArraysCache.extend path with many simultaneous prefills. Kept as a
    broader stress test; tunable via ``QWEN3_CONCURRENT_TEST_CONCURRENCY``.
    """
    _skip_if_not_configured()
    if CONCURRENCY < 2:
        pytest.skip(
            f"QWEN3_CONCURRENT_TEST_CONCURRENCY={CONCURRENCY} is below 2; "
            "set >=2 to exercise batched prefill."
        )
    httpx = pytest.importorskip("httpx")

    base = _load_base_heavy()
    payloads = [_inject_nonce(base, idx=i) for i in range(1, CONCURRENCY + 1)]

    async with httpx.AsyncClient() as session:
        results = await asyncio.gather(
            *(
                _one_request(session, i, URL, p)
                for i, p in enumerate(payloads, start=1)
            ),
            return_exceptions=False,
        )

    failures: list[str] = []
    for r in results:
        failures.extend(_validate_request_result(r, "homogeneous-multi"))

    assert not failures, (
        f"{len(failures)}/{CONCURRENCY} homogeneous-multi failures:\n  "
        + "\n  ".join(failures)
        + f"\n\nModel: {MODEL}  URL: {URL}"
    )
