# SPDX-License-Identifier: Apache-2.0
"""Live-server integration test for the Qwen3.x hybrid-cache concurrent-prefill bug.

Regression lockdown for the bug documented in
``docs/testing/2026-04-21-qwen3-35b-a3b-concurrent-heavy-payload-deadlock.md``
and the upstream fix in ml-explore/mlx-lm#1169 + #1177 (shipped in
mlx_lm 0.31.3, git SHA 3cd9a52d). Pinned by UPSTREAM_PIN.md invariant #17.

The ``ArraysCache.extend`` contract is unit-tested in
``tests/test_mlx_lm_arrays_cache_concurrent.py`` without any model.
This file tests the end-to-end behaviour — that a running vllm-mlx
server serving a Qwen3.x hybrid-cache model (Qwen3.5-35B-A3B,
Qwen3.6-35B-A3B, Qwen3-Next variants, etc.) does NOT produce
degenerate zero-token responses under concurrent heavy-payload load.

Pre-fix symptom (mlx_lm 0.31.2 and earlier):
    - 8 concurrent unique prefixes → all complete in ~0.8s with
      ``content_block_delta`` count == 0 and ``completion=0 tokens``
      logged server-side. Degenerate output per mlx-lm#1169's
      "neighbour's conv/SSM state contamination" description.

Post-fix behaviour (mlx_lm 0.31.3+):
    - 8 concurrent unique prefixes → all complete in 10-25s with
      proper ``content_block_delta`` events and non-zero output tokens.

Run explicitly (requires a live vllm-mlx server on a reachable URL)::

    # Start the server:
    vllm-mlx serve mlx-community/Qwen3.5-35B-A3B-4bit \\
        --host 127.0.0.1 --port 19001 --continuous-batching \\
        --enable-auto-tool-choice --tool-call-parser qwen3 \\
        --reasoning-parser qwen3 --max-thinking-token-budget 2048

    # Run:
    QWEN3_CONCURRENT_TEST_URL=http://127.0.0.1:19001 \\
    QWEN3_CONCURRENT_TEST_MODEL=mlx-community/Qwen3.5-35B-A3B-4bit \\
        pytest tests/test_qwen3_concurrent_heavy_payload.py -m integration -v

Optional env knobs:
    QWEN3_CONCURRENT_TEST_CONCURRENCY=8  # default; 2 is insufficient to
                                         # force concurrent prefill on a
                                         # cold cache
    QWEN3_CONCURRENT_TEST_TIMEOUT_S=60   # per-request curl-style budget
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

URL = os.getenv("QWEN3_CONCURRENT_TEST_URL")
MODEL = os.getenv("QWEN3_CONCURRENT_TEST_MODEL")
CONCURRENCY = int(os.getenv("QWEN3_CONCURRENT_TEST_CONCURRENCY", "8"))
TIMEOUT_S = float(os.getenv("QWEN3_CONCURRENT_TEST_TIMEOUT_S", "60"))

_HEAVY_PAYLOAD = (
    Path(__file__).parent.parent
    / "docs"
    / "testing"
    / "claude-shape-heavy-payload.json"
)


def _skip_if_not_configured() -> None:
    if not URL or not MODEL:
        pytest.skip(
            "Set QWEN3_CONCURRENT_TEST_URL=http://host:port and "
            "QWEN3_CONCURRENT_TEST_MODEL=mlx-community/Qwen3.5-35B-A3B-4bit "
            "(or another hybrid-cache model) to run."
        )
    if not _HEAVY_PAYLOAD.exists():
        pytest.skip(f"Heavy payload fixture missing: {_HEAVY_PAYLOAD}")


def _load_base_payload() -> dict:
    with _HEAVY_PAYLOAD.open() as f:
        base = json.load(f)
    base["model"] = MODEL
    return base


def _unique_payload(idx: int, base: dict) -> dict:
    """Return a deep-copied payload with a unique user-message prefix.

    The bug reproduces only when requests don't share prefix-cache
    entries — so we inject a unique nonce into the first content
    block's text. Identical prompts get deduped on the second-arriving
    request and don't force concurrent prefill.
    """
    d = json.loads(json.dumps(base))
    nonce = uuid.uuid4().hex
    msg = d["messages"][0]
    content = msg.get("content")
    tag = f"[req-{idx} nonce={nonce}] "
    if isinstance(content, list) and content:
        block = content[0]
        if isinstance(block, dict) and "text" in block:
            block["text"] = tag + block["text"]
        else:
            content.insert(0, {"type": "text", "text": tag})
    else:
        msg["content"] = tag + str(content or "")
    return d


async def _one_request(session, idx: int, payload: dict) -> dict:
    """Fire one streaming POST, accumulate events, return a per-request summary."""
    import httpx

    summary = {
        "idx": idx,
        "http_status": None,
        "events": 0,
        "message_stop_count": 0,
        "content_block_delta_count": 0,
        "total_bytes": 0,
        "stop_reason": None,
        "output_tokens": None,
        "exception": None,
    }
    try:
        async with session.stream(
            "POST",
            f"{URL.rstrip('/')}/v1/messages",
            json=payload,
            timeout=TIMEOUT_S,
            headers={"Content-Type": "application/json"},
        ) as resp:
            summary["http_status"] = resp.status_code
            async for line in resp.aiter_lines():
                if not line:
                    continue
                summary["total_bytes"] += len(line) + 1
                if line.startswith("event:"):
                    summary["events"] += 1
                    if "message_stop" in line:
                        summary["message_stop_count"] += 1
                    elif "content_block_delta" in line:
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
    except Exception as exc:  # noqa: BLE001 — test surfaces exception class + msg
        summary["exception"] = f"{type(exc).__name__}: {exc}"
    return summary


@pytest.mark.asyncio
async def test_concurrent_heavy_payload_produces_nonzero_output():
    """Regression lockdown: N concurrent unique-prefix heavy payloads must
    each emit at least one content_block_delta and finish with end_turn.

    Pre-fix: all N complete in <1s with ``content_block_delta_count == 0``
    and ``output_tokens == 0`` — the ``ArraysCache.extend`` batch-dim bug
    corrupts conv/SSM state across concurrent prefills and the model
    exits without generating anything usable.

    Post-fix: each request emits proper streaming deltas and the
    completion counter is positive.
    """
    _skip_if_not_configured()
    httpx = pytest.importorskip("httpx")

    base = _load_base_payload()
    payloads = [_unique_payload(i, base) for i in range(1, CONCURRENCY + 1)]

    async with httpx.AsyncClient() as session:
        results = await asyncio.gather(
            *(_one_request(session, i, p) for i, p in enumerate(payloads, start=1)),
            return_exceptions=False,
        )

    failures: list[str] = []
    for r in results:
        if r["exception"]:
            failures.append(f"req-{r['idx']}: raised {r['exception']}")
            continue
        if r["http_status"] != 200:
            failures.append(f"req-{r['idx']}: HTTP {r['http_status']}")
            continue
        if r["content_block_delta_count"] < 1:
            failures.append(
                f"req-{r['idx']}: ZERO content_block_delta events "
                f"(events={r['events']}, output_tokens={r['output_tokens']}, "
                f"stop_reason={r['stop_reason']}, bytes={r['total_bytes']}). "
                "This is the pre-fix mlx_lm#1169 regression — "
                "ArraysCache.extend corrupted batch-dim under concurrent prefill."
            )
            continue
        if r["message_stop_count"] < 1:
            failures.append(
                f"req-{r['idx']}: missing message_stop event "
                f"(events={r['events']}, stop_reason={r['stop_reason']})"
            )
            continue
        if r["stop_reason"] not in (None, "end_turn", "tool_use", "max_tokens"):
            failures.append(
                f"req-{r['idx']}: unexpected stop_reason {r['stop_reason']!r}"
            )

    assert not failures, (
        f"{len(failures)}/{CONCURRENCY} concurrent requests failed the "
        f"concurrent-prefill regression check:\n  "
        + "\n  ".join(failures)
        + f"\n\nModel under test: {MODEL}\nTarget URL: {URL}\n"
        "See docs/testing/2026-04-21-qwen3-35b-a3b-concurrent-heavy-payload-deadlock.md"
        " and docs/superpowers/plans/2026-04-22-qwen3-concurrent-deadlock.md."
    )
