#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cross-model thinking-budget benchmark with accuracy on ground-truth tasks.

For each (model, budget) cell this reports:
    - tokens generated
    - wall-clock seconds
    - tokens/second
    - x-thinking-budget-applied header value
    - accuracy (model's answer matches the known correct answer?)

Budgets tested: None (natural), 512 (low), 8192 (high). The "OFF" case
(budget=0) is excluded by default — different models have different
opinions about "no thinking" and a dead-silent immediate force is noisy
to compare across families. Re-add --include-off for the strict test.

Usage (one model at a time — recommended while validating):

    python scripts/thinking_budget_benchmark.py \\
        --url http://127.0.0.1:8010 \\
        --model mlx-community/Qwen3-0.6B-8bit \\
        --name "qwen3-0.6b" \\
        --out /tmp/bench-qwen3-0.6b.md

Or multiple at once (after each is proven to work):

    python scripts/thinking_budget_benchmark.py \\
        --config bench-models.json \\
        --out /tmp/bench-all.md

Config schema (same as the matrix test):
    [{"name": "...", "url": "http://...", "model": "mlx-community/..."}, ...]

Exit non-zero if any cell errored.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import requests


# ----- Ground-truth tasks (known correct answers) -----

@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    answer_patterns: tuple[str, ...]   # regex alternatives that all count as correct
    kind: str  # "arithmetic" | "factual" | "reasoning"


TASKS: list[Task] = [
    Task(
        id="arith_13x17",
        prompt="What is 13 multiplied by 17? Show your work and give the final numeric answer.",
        # 221 is the only right answer. Accept surrounded by word boundaries or
        # common punctuation. Reject if wrapped in a larger number.
        answer_patterns=(r"\b221\b",),
        kind="arithmetic",
    ),
    Task(
        id="arith_primes_under_50",
        prompt="Find the sum of all prime numbers less than 50. Show step by step. What is the final sum?",
        # 2+3+5+7+11+13+17+19+23+29+31+37+41+43+47 = 328
        answer_patterns=(r"\b328\b",),
        kind="arithmetic",
    ),
    Task(
        id="factual_capital_france",
        prompt="What is the capital of France? Answer in one short sentence.",
        answer_patterns=(r"\bParis\b",),
        kind="factual",
    ),
]


# ----- Per-cell measurement -----

@dataclass
class Cell:
    model_name: str
    task_id: str
    task_kind: str
    budget_label: str  # "none" | "low" | "high" | "off"
    budget_val: int | None
    max_tokens: int
    completion_tokens: int | None
    prompt_tokens: int | None
    reasoning_chars: int  # separated so budget enforcement is visible
    content_chars: int
    finish_reason: str | None
    elapsed_s: float
    tokens_per_sec: float
    header: str | None
    correct: bool
    output_first_60: str
    error: str | None = None


_TIMEOUT = 300


def _chat(
    url: str,
    model: str,
    budget: int | None,
    prompt: str,
    max_tokens: int,
) -> tuple[dict, dict, float]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    if budget is not None:
        body["thinking_token_budget"] = budget
    t0 = time.time()
    resp = requests.post(
        f"{url}/v1/chat/completions", json=body, timeout=_TIMEOUT
    )
    elapsed = time.time() - t0
    resp.raise_for_status()
    return dict(resp.headers), resp.json(), elapsed


def _check_answer(text: str, patterns: tuple[str, ...]) -> bool:
    for pat in patterns:
        if re.search(pat, text):
            return True
    return False


def _combined_output(msg: dict) -> str:
    """Merge reasoning + content into one string for answer extraction.
    Most reasoning models put the final answer in content, but if the
    parser mis-routed (or the model didn't tag), the answer might be in
    reasoning. Check both."""
    parts = []
    r = msg.get("reasoning") or ""
    c = msg.get("content") or ""
    if r:
        parts.append(r)
    if c:
        parts.append(c)
    return "\n".join(parts)


def _run_cell(
    model_spec: dict,
    task: Task,
    budget_label: str,
    budget_val: int | None,
    max_tokens: int,
) -> Cell:
    try:
        headers, data, elapsed = _chat(
            model_spec["url"], model_spec["model"], budget_val,
            task.prompt, max_tokens,
        )
    except requests.exceptions.RequestException as exc:
        return Cell(
            model_name=model_spec["name"],
            task_id=task.id,
            task_kind=task.kind,
            budget_label=budget_label,
            budget_val=budget_val,
            max_tokens=max_tokens,
            completion_tokens=None,
            prompt_tokens=None,
            reasoning_chars=0,
            content_chars=0,
            finish_reason=None,
            elapsed_s=0.0,
            tokens_per_sec=0.0,
            header=None,
            correct=False,
            output_first_60="",
            error=str(exc)[:120],
        )

    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = data.get("usage") or {}
    comp_toks = usage.get("completion_tokens") or 0
    reasoning_raw = msg.get("reasoning") or ""
    content_raw = msg.get("content") or ""
    combined = _combined_output(msg)
    correct = _check_answer(combined, task.answer_patterns)
    tps = (comp_toks / elapsed) if elapsed > 0 else 0.0

    return Cell(
        model_name=model_spec["name"],
        task_id=task.id,
        task_kind=task.kind,
        budget_label=budget_label,
        budget_val=budget_val,
        reasoning_chars=len(reasoning_raw),
        content_chars=len(content_raw),
        finish_reason=choice.get("finish_reason"),
        max_tokens=max_tokens,
        completion_tokens=comp_toks,
        prompt_tokens=usage.get("prompt_tokens"),
        elapsed_s=round(elapsed, 2),
        tokens_per_sec=round(tps, 1),
        header=headers.get("x-thinking-budget-applied"),
        correct=correct,
        output_first_60=combined[:60].replace("\n", " "),
    )


# ----- Invocation -----

def _load_specs(args) -> list[dict]:
    if args.config:
        try:
            return json.loads(open(args.config).read())
        except Exception as exc:
            print(f"ERROR reading {args.config}: {exc}", file=sys.stderr)
            sys.exit(2)
    if args.url and args.model:
        return [
            {
                "name": args.name or args.model.split("/")[-1],
                "url": args.url,
                "model": args.model,
            }
        ]
    print("ERROR: need --config or (--url + --model)", file=sys.stderr)
    sys.exit(2)


def _run(
    specs: list[dict],
    include_off: bool,
    max_tokens: int,
) -> list[Cell]:
    budget_list: list[tuple[str, int | None]] = [
        ("none", None),
        ("low", 512),
        ("high", 8192),
    ]
    if include_off:
        budget_list.insert(0, ("off", 0))

    cells: list[Cell] = []
    for spec in specs:
        print(f"\n=== {spec['name']} ===", file=sys.stderr)
        for task in TASKS:
            for label, val in budget_list:
                print(
                    f"  {task.id:30s} budget={label:4s} ... ",
                    end="", flush=True, file=sys.stderr,
                )
                cell = _run_cell(spec, task, label, val, max_tokens)
                cells.append(cell)
                _log_cell(cell)
    return cells


def _log_cell(c: Cell) -> None:
    if c.error:
        print(f"ERROR ({c.error})", file=sys.stderr)
        return
    verdict = "✓" if c.correct else "✗"
    print(
        f"{verdict} tok={c.completion_tokens:>4} "
        f"r_c={c.reasoning_chars:>5} c_c={c.content_chars:>5} "
        f"t={c.elapsed_s:>5.1f}s tps={c.tokens_per_sec:>5.1f} "
        f"fin={c.finish_reason!s:<6} hdr={c.header!s:<5} "
        f"out={c.output_first_60!r}",
        file=sys.stderr,
    )


# ----- Reporting -----

def _make_markdown(cells: list[Cell]) -> str:
    lines = [
        "# Thinking-budget benchmark",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')}_",
        "",
        "Budgets tested: `none` (natural), `low` (512), `high` (8192). "
        "Correctness = answer_patterns matched in merged reasoning+content.",
        "",
    ]

    # Group by model, summarize per (model, budget) across tasks
    by_model: dict[str, list[Cell]] = {}
    for c in cells:
        by_model.setdefault(c.model_name, []).append(c)

    # Per-model detail tables
    for name, group in by_model.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(
            "| task | budget | header | ✓ | tokens | r_chars | c_chars | finish | elapsed | tok/s | preview |"
        )
        lines.append(
            "|------|--------|--------|---|--------|---------|---------|--------|---------|-------|---------|"
        )
        for c in group:
            err = c.error or ""
            if err:
                lines.append(
                    f"| {c.task_id} | {c.budget_label} | — | ⚠ | — | — | — | — | — | — | "
                    f"ERR: {err[:50]} |"
                )
                continue
            mark = "✅" if c.correct else "❌"
            preview = c.output_first_60.replace("|", "\\|")
            lines.append(
                f"| {c.task_id} | {c.budget_label} | `{c.header}` | {mark} | "
                f"{c.completion_tokens} | {c.reasoning_chars} | {c.content_chars} | "
                f"{c.finish_reason} | {c.elapsed_s}s | {c.tokens_per_sec} | `{preview}` |"
            )
        lines.append("")

    # Cross-model summary
    lines.append("## Summary across all models")
    lines.append("")
    lines.append(
        "**Budget enforcement signal:** compare `avg r_chars` (reasoning chars)"
        " across `off` / `low` / `high` / `none`. A working budget should"
        " produce monotonically increasing reasoning: `off < low < high`,"
        " with `none` roughly matching `high` (natural length)."
    )
    lines.append("")
    lines.append(
        "| model | budget | cells | ✓ | avg tokens | avg r_chars | avg c_chars | avg elapsed | avg tok/s | header |"
    )
    lines.append(
        "|-------|--------|-------|---|------------|-------------|-------------|-------------|-----------|--------|"
    )
    for name, group in by_model.items():
        by_budget: dict[str, list[Cell]] = {}
        for c in group:
            by_budget.setdefault(c.budget_label, []).append(c)
        # Order budgets consistently: off, low, high, none
        order = ["off", "low", "high", "none"]
        for label in order:
            if label not in by_budget:
                continue
            cells_in = by_budget[label]
            ok = [c for c in cells_in if not c.error]
            if not ok:
                lines.append(
                    f"| {name} | {label} | {len(cells_in)} | - | - | - | - | - | - | all errors |"
                )
                continue
            n_correct = sum(1 for c in ok if c.correct)
            avg_toks = sum(c.completion_tokens or 0 for c in ok) / len(ok)
            avg_r = sum(c.reasoning_chars for c in ok) / len(ok)
            avg_c = sum(c.content_chars for c in ok) / len(ok)
            avg_elapsed = sum(c.elapsed_s for c in ok) / len(ok)
            avg_tps = sum(c.tokens_per_sec for c in ok) / len(ok)
            headers_seen = sorted(set(c.header or "absent" for c in ok))
            lines.append(
                f"| {name} | {label} | {len(ok)} | {n_correct}/{len(ok)} | "
                f"{avg_toks:.0f} | {avg_r:.0f} | {avg_c:.0f} | "
                f"{avg_elapsed:.1f}s | {avg_tps:.1f} | {','.join(headers_seen)} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cross-model thinking-budget benchmark."
    )
    ap.add_argument("--config", help="JSON list of {name,url,model}")
    ap.add_argument("--url", help="Server URL (single-model mode)")
    ap.add_argument("--model", help="Model ID (single-model mode)")
    ap.add_argument("--name", help="Display name (single-model mode)")
    ap.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Max tokens per request (default 2048)",
    )
    ap.add_argument(
        "--include-off", action="store_true",
        help="Also test budget=0 (force-close immediately). Off by default "
             "because it's flaky across model families and dominates the "
             "signal. Add only if specifically testing the OFF path.",
    )
    ap.add_argument(
        "--out", help="Write markdown report to path (default stdout)",
    )
    ap.add_argument(
        "--json-out", help="Write raw per-cell data to path (JSON)",
    )
    args = ap.parse_args()

    specs = _load_specs(args)
    cells = _run(specs, include_off=args.include_off, max_tokens=args.max_tokens)
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

    return 1 if any(c.error for c in cells) else 0


if __name__ == "__main__":
    sys.exit(main())
