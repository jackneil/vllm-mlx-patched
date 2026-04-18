#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Manual matrix runner for thinking_token_budget across live servers.

Pairs with tests/test_thinking_budget_matrix.py but prints a human-readable
markdown report instead of asserting. Useful for:
    - Ad-hoc "is the stack alive" checks after a deploy
    - Generating evidence for PR bodies / incident writeups
    - Spot-checking a new model before adding it to the pytest matrix

Usage:
    # Config file form (same schema as $THINKING_BUDGET_MATRIX):
    python scripts/thinking_budget_matrix.py --config matrix.json

    # Inline form for a single model:
    python scripts/thinking_budget_matrix.py \\
        --url http://127.0.0.1:8099 \\
        --model mlx-community/Qwen3-0.6B-8bit \\
        --name qwen3-0.6b --family supported

    # Write report to disk:
    python scripts/thinking_budget_matrix.py --config matrix.json \\
        --out /tmp/thinking-budget-report.md

Exit codes:
    0 = all cells finished, report written
    1 = one or more cells failed (HTTP error, timeout) — report still written
    2 = config error (missing file, bad JSON)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import requests


PROMPT_REASONING = (
    "Solve step by step: what is the sum of all prime numbers less than 50? "
    "Show your reasoning thoroughly."
)
PROMPT_TRIVIAL = "What is 2+2?"

DEFAULT_TIMEOUT = 180


@dataclass
class Cell:
    model: str
    family: str
    budget: int | None
    prompt_label: str
    max_tokens: int
    finish_reason: str | None
    completion_tokens: int | None
    reasoning_chars: int
    content_chars: int
    header: str | None
    elapsed_s: float
    error: str | None = None


def _chat(
    url: str,
    model: str,
    budget: int | None,
    prompt: str,
    max_tokens: int,
    message: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[dict, dict, float]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    if budget is not None:
        body["thinking_token_budget"] = budget
    if message is not None:
        body["thinking_budget_message"] = message

    t0 = time.time()
    resp = requests.post(
        f"{url}/v1/chat/completions", json=body, timeout=timeout
    )
    elapsed = time.time() - t0
    resp.raise_for_status()
    return dict(resp.headers), resp.json(), elapsed


def _run_cell(
    spec: dict,
    budget: int | None,
    prompt_label: str,
    prompt: str,
    max_tokens: int,
) -> Cell:
    try:
        headers, data, elapsed = _chat(
            spec["url"], spec["model"], budget, prompt, max_tokens
        )
    except requests.exceptions.RequestException as exc:
        return Cell(
            model=spec["name"],
            family=spec["family"],
            budget=budget,
            prompt_label=prompt_label,
            max_tokens=max_tokens,
            finish_reason=None,
            completion_tokens=None,
            reasoning_chars=0,
            content_chars=0,
            header=None,
            elapsed_s=0.0,
            error=str(exc)[:200],
        )

    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = data.get("usage") or {}
    reasoning = msg.get("reasoning") or ""
    content = msg.get("content") or ""
    return Cell(
        model=spec["name"],
        family=spec["family"],
        budget=budget,
        prompt_label=prompt_label,
        max_tokens=max_tokens,
        finish_reason=choice.get("finish_reason"),
        completion_tokens=usage.get("completion_tokens"),
        reasoning_chars=len(reasoning),
        content_chars=len(content),
        header=headers.get("x-thinking-budget-applied"),
        elapsed_s=round(elapsed, 2),
    )


def _load_matrix(args) -> list[dict]:
    if args.config:
        try:
            return json.loads(open(args.config).read())
        except Exception as exc:
            print(f"ERROR: {args.config}: {exc}", file=sys.stderr)
            sys.exit(2)
    if args.url and args.model:
        return [
            {
                "name": args.name or args.model.split("/")[-1],
                "url": args.url,
                "model": args.model,
                "family": args.family or "supported",
            }
        ]
    print(
        "ERROR: must provide either --config or (--url AND --model)",
        file=sys.stderr,
    )
    sys.exit(2)


def _run_matrix(matrix: list[dict]) -> list[Cell]:
    cells: list[Cell] = []
    # For each model, run a budget sweep on the reasoning prompt (main
    # invariant check) + a budget=0 trivial cell (fast-path sanity).
    budget_sweep = [None, 0, 64, 512, 2048]
    for spec in matrix:
        print(f"\n=== {spec['name']} ({spec['family']}) ===", file=sys.stderr)
        for b in budget_sweep:
            cell = _run_cell(
                spec, b, "reasoning", PROMPT_REASONING, max_tokens=2048
            )
            _log_cell(cell)
            cells.append(cell)
        trivial = _run_cell(
            spec, 0, "trivial", PROMPT_TRIVIAL, max_tokens=256
        )
        _log_cell(trivial)
        cells.append(trivial)
    return cells


def _log_cell(c: Cell) -> None:
    if c.error:
        print(
            f"  budget={c.budget!s:>5} {c.prompt_label:<9} ERROR: {c.error}",
            file=sys.stderr,
        )
        return
    print(
        f"  budget={c.budget!s:>5} {c.prompt_label:<9} "
        f"tokens={c.completion_tokens!s:>4} "
        f"rc={c.reasoning_chars:>5} cc={c.content_chars:>5} "
        f"header={c.header!s:<5} t={c.elapsed_s}s",
        file=sys.stderr,
    )


def _make_markdown(cells: list[Cell]) -> str:
    """Render a markdown report grouped by model."""
    lines = [
        "# Thinking-budget matrix results",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')}_",
        "",
        "## Legend",
        "- `header` column: value of `x-thinking-budget-applied` response header.",
        "  `true` = processor attached and enforced; `false` = loud no-op; "
        "`None` = header absent (budget was not set or streaming).",
        "- `reasoning_chars` / `content_chars`: length of the reasoning and "
        "content fields in the response (proxies for thinking/final tokens).",
        "- `elapsed_s`: wall clock for the single HTTP round trip.",
        "",
    ]
    by_model: dict[str, list[Cell]] = {}
    for c in cells:
        by_model.setdefault(c.model, []).append(c)

    for name, group in by_model.items():
        family = group[0].family
        lines.append(f"## {name} — family=`{family}`")
        lines.append("")
        lines.append(
            "| budget | prompt | finish | tokens | r_chars | c_chars | "
            "header | elapsed | error |"
        )
        lines.append(
            "|--------|--------|--------|--------|---------|---------|"
            "--------|---------|-------|"
        )
        for c in group:
            err = c.error or ""
            if len(err) > 40:
                err = err[:37] + "..."
            lines.append(
                f"| {c.budget!s} | {c.prompt_label} | "
                f"{c.finish_reason or '-'} | "
                f"{c.completion_tokens if c.completion_tokens is not None else '-'} | "
                f"{c.reasoning_chars} | {c.content_chars} | "
                f"`{c.header}` | {c.elapsed_s}s | {err} |"
            )
        lines.append("")
        lines.extend(_evaluate(name, family, group))
        lines.append("")
    return "\n".join(lines)


def _evaluate(name: str, family: str, group: list[Cell]) -> list[str]:
    """Emit pass/fail bullets per model based on family-appropriate
    invariants. Does NOT raise — this is a report, not an assertion."""
    out: list[str] = ["**Evaluation:**", ""]
    errors = [c for c in group if c.error]
    if errors:
        out.append(
            f"- ⚠️ {len(errors)}/{len(group)} cells failed with HTTP errors"
        )
        return out

    by_budget = {c.budget: c for c in group if c.prompt_label == "reasoning"}
    trivial = next(
        (c for c in group if c.prompt_label == "trivial"), None
    )

    if family == "supported":
        # Header must be true for every cell where budget is set
        for b in (0, 64, 512, 2048):
            cell = by_budget.get(b)
            if cell and cell.header != "true":
                out.append(
                    f"- ❌ budget={b}: header=`{cell.header}` (expected "
                    f"`true`) — processor attach failed"
                )
            elif cell:
                out.append(f"- ✅ budget={b}: header=`true`")
        # Unbounded should NOT carry the header
        r_none = by_budget.get(None)
        if r_none and r_none.header is not None:
            out.append(
                f"- ⚠️ budget=None: header=`{r_none.header}` "
                "(expected absent)"
            )
        # Ordering: r(0) <= r(64) <= r(512), r(None) >> r(0)
        r0 = by_budget[0].reasoning_chars if 0 in by_budget else None
        r64 = by_budget[64].reasoning_chars if 64 in by_budget else None
        r512 = by_budget[512].reasoning_chars if 512 in by_budget else None
        rN = by_budget[None].reasoning_chars if None in by_budget else None
        if r0 is not None and r64 is not None and r0 > r64 + 200:
            out.append(
                f"- ❌ budget=0 produced MORE reasoning ({r0}) than "
                f"budget=64 ({r64}) — check force-close"
            )
        elif r0 is not None and r64 is not None:
            out.append(f"- ✅ budget=0 reasoning ≤ budget=64 reasoning")
        if r64 is not None and r512 is not None and r64 > r512 + 200:
            out.append(
                f"- ❌ budget=64 reasoning ({r64}) exceeds budget=512 "
                f"({r512})"
            )
        if rN is not None and r0 is not None and rN <= r0 + 100:
            out.append(
                f"- ⚠️ unbounded reasoning ({rN}) not substantially "
                f"larger than budget=0 ({r0}) — prompt may be trivial"
            )
        elif rN is not None and r0 is not None:
            out.append(
                f"- ✅ unbounded reasoning ({rN}) >> budget=0 ({r0})"
            )
        # Latency: budget=0 should be faster than unbounded
        if r_none and 0 in by_budget:
            z = by_budget[0].elapsed_s
            n = r_none.elapsed_s
            if z < n * 0.7:
                out.append(
                    f"- ✅ budget=0 ({z}s) << unbounded ({n}s)"
                )
            else:
                out.append(
                    f"- ⚠️ budget=0 ({z}s) not materially faster than "
                    f"unbounded ({n}s) — MLX warm-up variance likely"
                )
        # Trivial budget=0 sanity
        if trivial:
            if trivial.content_chars == 0:
                out.append(
                    "- ❌ trivial/budget=0: empty content — force-close "
                    "may have suppressed the answer"
                )
            elif trivial.reasoning_chars > 50:
                out.append(
                    f"- ⚠️ trivial/budget=0: reasoning was "
                    f"{trivial.reasoning_chars} chars (expected ~0)"
                )
            else:
                out.append("- ✅ trivial/budget=0: fast, minimal thinking")
    elif family.startswith("noop"):
        # All cells with a budget set must report header=false
        for b in (0, 64, 512, 2048):
            cell = by_budget.get(b)
            if cell and cell.header != "false":
                out.append(
                    f"- ❌ budget={b}: header=`{cell.header}` (expected "
                    f"`false` for noop family)"
                )
            elif cell:
                out.append(f"- ✅ budget={b}: header=`false` (noop)")
        # Generation must still work
        empties = [
            c for c in group
            if not c.error and (c.content_chars + c.reasoning_chars) == 0
        ]
        if empties:
            out.append(
                f"- ❌ {len(empties)} cells returned empty output — "
                "no-op must still generate"
            )
        else:
            out.append("- ✅ all cells produced output despite no-op budget")
    else:
        out.append(f"- ⚠️ unknown family `{family}` — no evaluation applied")
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Thinking-budget matrix runner (live HTTP).",
    )
    p.add_argument("--config", help="Path to matrix JSON")
    p.add_argument("--url", help="Server base URL (single-model mode)")
    p.add_argument("--model", help="Model ID (single-model mode)")
    p.add_argument("--name", help="Display name (single-model mode)")
    p.add_argument(
        "--family",
        choices=["supported", "noop-mllm", "noop-simple", "noop-parser"],
        default="supported",
    )
    p.add_argument("--out", help="Write markdown report to this path")
    p.add_argument(
        "--json-out",
        help="Write raw per-cell results as JSON to this path",
    )
    args = p.parse_args()

    matrix = _load_matrix(args)
    cells = _run_matrix(matrix)
    report = _make_markdown(cells)

    if args.out:
        open(args.out, "w").write(report)
        print(f"\nReport written to {args.out}", file=sys.stderr)
    else:
        print(report)

    if args.json_out:
        raw = [asdict(c) for c in cells]
        open(args.json_out, "w").write(json.dumps(raw, indent=2))
        print(f"JSON written to {args.json_out}", file=sys.stderr)

    # Exit nonzero if any cell had an HTTP error.
    return 1 if any(c.error for c in cells) else 0


if __name__ == "__main__":
    sys.exit(main())
