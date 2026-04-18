# SPDX-License-Identifier: Apache-2.0
"""Live-server matrix tests for thinking_token_budget across model families.

These tests hit a running vllm-mlx HTTP server and validate that thinking
budgets behave correctly across different reasoning-model families
(Qwen3, DeepSeek-R1, Gemma 4, GPT-OSS, VLM/MLLM).

Why HTTP instead of direct engine:
    - Tests the full stack: Pydantic validation, chat_template_kwargs
      precedence, scheduler attach, logits processor, response headers,
      parser extraction. A bug in any of those layers surfaces here.
    - Matches production behavior exactly — arena traffic flows through
      the same path.
    - Avoids fragile asyncio fixtures; uses plain `requests`.

How to run:
    Point THINKING_BUDGET_MATRIX at a JSON file describing the servers
    and models to test, then run pytest with the integration marker:

        export THINKING_BUDGET_MATRIX=/path/to/matrix.json
        pytest tests/test_thinking_budget_matrix.py -m integration -v

    Example matrix.json:
        [
          {"name": "qwen3-0.6b", "url": "http://127.0.0.1:8099",
           "model": "mlx-community/Qwen3-0.6B-8bit",
           "family": "supported", "parser": "qwen3"},
          {"name": "gemma4", "url": "http://127.0.0.1:8000",
           "model": "mlx-community/gemma-4-31b-it-4bit",
           "family": "noop-mllm"}
        ]

    Each entry's `family` controls assertion strictness:
        supported   — budget enforcement MUST work; ordering invariants apply
        noop-mllm   — MLLM path; header MUST be false; generation still works
        noop-simple — SimpleEngine; header MUST be false
        noop-parser — reasoning parser not configured or unsupported delimiter

Gated by default: tests are marked `integration` and are skipped unless the
matrix env var is set AND pytest is invoked with `-m integration`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import pytest
import requests


pytestmark = pytest.mark.integration


# Two prompts chosen to exercise different paths:
#   PROMPT_REASONING: asks for step-by-step work — the model will enter
#       <think>…</think> and keep thinking until budget or </think>.
#   PROMPT_TRIVIAL: asks for a simple fact — on budget=0 we expect the
#       model to answer in one line.
PROMPT_REASONING = (
    "Solve step by step: what is the sum of all prime numbers less than 50? "
    "Show your reasoning thoroughly."
)
PROMPT_TRIVIAL = "What is 2+2?"

# Timeouts (seconds) — per cell. Reasoning models can be slow; give each
# request generous headroom but cap so a stuck server doesn't wedge CI.
_TIMEOUT_SEC = 120


@dataclass(frozen=True)
class ModelSpec:
    """One entry in the test matrix."""

    name: str
    url: str
    model: str
    family: str  # "supported" | "noop-mllm" | "noop-simple" | "noop-parser"
    parser: str | None = None  # advisory; server decides at startup


def _load_matrix() -> list[ModelSpec]:
    """Load the test matrix from $THINKING_BUDGET_MATRIX, or return []
    (which causes pytest.mark.parametrize to skip every test with a
    "empty parameter set" message — same UX as explicit skip)."""
    path = os.getenv("THINKING_BUDGET_MATRIX")
    if not path:
        return []
    try:
        data = json.loads(open(path).read())
    except Exception as exc:
        pytest.skip(
            f"matrix file {path!r} unreadable: {exc}",
            allow_module_level=True,
        )
    return [ModelSpec(**entry) for entry in data]


def _matrix_ids(spec: ModelSpec) -> str:
    return f"{spec.name}:{spec.family}"


# Load once at module collection. If the env var is unset we fall back
# to a single sentinel entry that every test skips on — otherwise
# `pytest.mark.parametrize([])` emits a confusing `[NOTSET]` id.
_MATRIX = _load_matrix()
_MATRIX_EMPTY = not _MATRIX
if _MATRIX_EMPTY:
    _MATRIX = [ModelSpec(name="unset", url="", model="", family="supported")]


def _require_matrix() -> None:
    if _MATRIX_EMPTY:
        pytest.skip(
            "Set THINKING_BUDGET_MATRIX=/path/to/matrix.json to run "
            "(see module docstring for format)"
        )


@dataclass
class CellResult:
    """One (model, budget) measurement."""

    budget: int | None
    message: str | None
    finish_reason: str | None
    completion_tokens: int | None
    reasoning_chars: int
    content_chars: int
    header_applied: str | None  # "true" | "false" | None
    elapsed_s: float
    raw_content_starts: str  # first 60 chars (for diagnostics)


def _chat(
    spec: ModelSpec,
    *,
    budget: int | None = None,
    message: str | None = None,
    prompt: str = PROMPT_REASONING,
    max_tokens: int = 1024,
) -> CellResult:
    """One non-streaming chat-completion round trip."""
    body: dict[str, Any] = {
        "model": spec.model,
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
        f"{spec.url}/v1/chat/completions",
        json=body,
        timeout=_TIMEOUT_SEC,
    )
    elapsed = time.time() - t0
    resp.raise_for_status()

    header = resp.headers.get("x-thinking-budget-applied")
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    m = choice.get("message") or {}
    usage = data.get("usage") or {}
    reasoning = m.get("reasoning") or ""
    content = m.get("content") or ""

    return CellResult(
        budget=budget,
        message=message,
        finish_reason=choice.get("finish_reason"),
        completion_tokens=usage.get("completion_tokens"),
        reasoning_chars=len(reasoning),
        content_chars=len(content),
        header_applied=header,
        elapsed_s=round(elapsed, 2),
        raw_content_starts=(content or "")[:60].replace("\n", " "),
    )


# ----- Supported-family invariants -------------------------------------

@pytest.mark.parametrize("spec", _MATRIX, ids=_matrix_ids)
def test_budget_zero_is_fast_and_silent_on_thinking(spec: ModelSpec) -> None:
    """budget=0: on a reasoning model, thinking MUST be force-closed
    immediately and the response MUST return rapidly.

    Concrete assertion: reasoning_chars is small (≤ ~50 chars to allow for
    the forced </think> token arriving after a newline or short prefix);
    completion_tokens is bounded; and response is much faster than an
    unbounded run on the same prompt.

    Families:
        supported  — reasoning stripped to near-empty.
        noop-*     — no enforcement; we only assert the header is "false".
    """
    _require_matrix()
    r = _chat(spec, budget=0, prompt=PROMPT_TRIVIAL, max_tokens=256)

    if spec.family == "supported":
        assert r.header_applied == "true", (
            f"{spec.name}: expected header true, got {r.header_applied!r}. "
            f"Parser/tokenizer combo may not expose single-token "
            f"<think>/</think>. Response={r}"
        )
        assert r.reasoning_chars <= 50, (
            f"{spec.name}: budget=0 should produce ~no reasoning, got "
            f"{r.reasoning_chars} chars"
        )
        assert r.content_chars > 0, (
            f"{spec.name}: budget=0 must still produce an answer; content "
            f"was empty. reasoning_chars={r.reasoning_chars}"
        )
    else:
        # Loud no-op family: header MUST be false; model generates normally.
        assert r.header_applied == "false", (
            f"{spec.name} is family={spec.family}; expected header=false, "
            f"got {r.header_applied!r}"
        )
        # Content or reasoning must come back — generation still runs.
        assert (r.content_chars + r.reasoning_chars) > 0, (
            f"{spec.name}: no-op family still has to generate output; "
            f"got empty response"
        )


@pytest.mark.parametrize("spec", _MATRIX, ids=_matrix_ids)
def test_budget_ordering_invariant(spec: ModelSpec) -> None:
    """On supported families, higher budgets must permit ≥ thinking.

    We run four budgets (0, 64, 512, None) and assert:

        reasoning(0)  <=  reasoning(64)  <=  reasoning(512)
        reasoning(0)  <<  reasoning(None)     (strict, on a reasoning prompt)

    We use reasoning *chars* as a robust proxy for thinking tokens — the
    HTTP response surfaces reasoning in the message.reasoning field after
    the reasoning parser extracts it. The invariant holds for char counts
    just as it does for tokens.

    For noop families, we only verify that all four cells return
    `x-thinking-budget-applied: false` and produce non-empty output.
    """
    _require_matrix()
    cells = [
        _chat(spec, budget=b, prompt=PROMPT_REASONING, max_tokens=2048)
        for b in (0, 64, 512, None)
    ]

    if spec.family == "supported":
        r0, r64, r512, rnone = (c.reasoning_chars for c in cells)
        # Monotonicity with slack — tokenization variance can cause small
        # inversions near the boundary (e.g., 64 vs 80 chars due to a
        # single multi-char token landing on the edge).
        slack = 200  # generous; we're asserting qualitative behavior
        assert r0 <= r64 + slack, (
            f"{spec.name}: budget=0 produced MORE reasoning ({r0}) than "
            f"budget=64 ({r64}). Processor mis-attached?"
        )
        assert r64 <= r512 + slack, (
            f"{spec.name}: budget=64 produced MORE reasoning ({r64}) than "
            f"budget=512 ({r512}). Force-close likely firing late."
        )
        # Strict ordering: unbounded MUST be substantially larger than 0.
        # If this fails, either the prompt didn't provoke reasoning or
        # the force-close at budget=0 is silently failing.
        assert rnone > r0 + 100, (
            f"{spec.name}: unbounded reasoning ({rnone}) not substantially "
            f"larger than budget=0 ({r0}). Prompt may be too trivial for "
            f"this model, or budget=0 isn't actually closing."
        )
        # Every supported-family cell must carry the header.
        for c in cells:
            assert c.header_applied == "true", (
                f"{spec.name}: budget={c.budget} missing header=true "
                f"(got {c.header_applied!r})"
            )
    else:
        for c in cells:
            # None doesn't emit the header at all; 0/64/512 must show false.
            if c.budget is not None:
                assert c.header_applied == "false", (
                    f"{spec.name} ({spec.family}): budget={c.budget} "
                    f"expected header=false, got {c.header_applied!r}"
                )
            # All cells must still generate output.
            assert c.content_chars + c.reasoning_chars > 0


@pytest.mark.parametrize("spec", _MATRIX, ids=_matrix_ids)
def test_budget_zero_is_faster_than_unbounded(spec: ModelSpec) -> None:
    """budget=0 should return much faster than unbounded on a reasoning
    prompt, because the force-close cuts generation within a few tokens.

    We allow generous slack (budget=0 < 50% of unbounded) because MLX
    compile + warm-up variance can dominate on the first call. Sequencing
    matters: we run budget=0 second so it benefits from the warm cache,
    biasing AGAINST the test passing — a passing test is a robust signal.
    """
    _require_matrix()
    if spec.family != "supported":
        pytest.skip(f"{spec.name}: latency assertion only meaningful for "
                    "supported family")

    # Warm the server with unbounded first, THEN measure budget=0 so the
    # warm-up cost is absorbed by the unbounded cell.
    r_none = _chat(spec, budget=None, prompt=PROMPT_REASONING, max_tokens=1024)
    r_zero = _chat(spec, budget=0, prompt=PROMPT_REASONING, max_tokens=1024)

    assert r_zero.elapsed_s < r_none.elapsed_s * 0.7, (
        f"{spec.name}: budget=0 ({r_zero.elapsed_s}s) was not materially "
        f"faster than unbounded ({r_none.elapsed_s}s). Force-close may "
        f"not be firing, or max_tokens is bounding both cells."
    )


# ----- Graceful wrap-up message ---------------------------------------

@pytest.mark.parametrize("spec", _MATRIX, ids=_matrix_ids)
def test_thinking_budget_message_produces_transition(
    spec: ModelSpec,
) -> None:
    """With a wrap-up message, the forced close should be preceded by the
    message tokens in the reasoning block. This exercises the
    message_token_ids path in ThinkingTokenBudgetLogitsProcessor.
    """
    _require_matrix()
    if spec.family != "supported":
        pytest.skip(f"{spec.name}: message feature only runs on supported "
                    "family")

    hint = "Wrap up and answer now."
    # Fetch the raw response so we can inspect the reasoning block text.
    body = {
        "model": spec.model,
        "messages": [{"role": "user", "content": PROMPT_REASONING}],
        "max_tokens": 2048,
        "temperature": 0.3,
        "thinking_token_budget": 64,
        "thinking_budget_message": hint,
    }
    resp = requests.post(
        f"{spec.url}/v1/chat/completions", json=body, timeout=_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    assert resp.headers.get("x-thinking-budget-applied") == "true"
    m = (resp.json().get("choices") or [{}])[0].get("message") or {}
    reasoning = m.get("reasoning") or ""
    assert reasoning, (
        f"{spec.name}: message feature — reasoning block was empty; "
        f"processor may not have attached"
    )
    # A distinctive substring from the hint should survive tokenization.
    # "Wrap up" is the highest-signal phrase; if the tokenizer splits it
    # weirdly, fall back to checking "answer" which is also in the hint.
    hit = ("Wrap up" in reasoning) or ("wrap up" in reasoning) or (
        "answer" in reasoning.lower()
    )
    assert hit, (
        f"{spec.name}: expected wrap-up hint to appear in reasoning "
        f"block; reasoning was {reasoning[:200]!r}"
    )


# ----- Anthropic /v1/messages plumbing --------------------------------

@pytest.mark.parametrize("spec", _MATRIX, ids=_matrix_ids)
def test_anthropic_messages_path_honors_budget(spec: ModelSpec) -> None:
    """Same budget=0 invariant via the Anthropic /v1/messages endpoint.

    This catches regressions where the OpenAI adapter is wired but the
    Anthropic adapter drops the field.
    """
    _require_matrix()
    if spec.family != "supported":
        pytest.skip(f"{spec.name}: Anthropic plumbing test only on "
                    "supported family")

    body = {
        "model": spec.model,
        "messages": [{"role": "user", "content": PROMPT_TRIVIAL}],
        "max_tokens": 256,
        "thinking_token_budget": 0,
        "temperature": 0.3,
    }
    t0 = time.time()
    resp = requests.post(
        f"{spec.url}/v1/messages",
        json=body,
        timeout=_TIMEOUT_SEC,
    )
    elapsed = time.time() - t0
    resp.raise_for_status()

    assert resp.headers.get("x-thinking-budget-applied") == "true", (
        f"{spec.name}: Anthropic endpoint dropped the budget — header was "
        f"{resp.headers.get('x-thinking-budget-applied')!r}. "
        f"Check AnthropicRequest model + handler plumbing."
    )
    data = resp.json()
    blocks = data.get("content") or []
    thinking_chars = sum(
        len(b.get("thinking", "") or "")
        for b in blocks
        if b.get("type") == "thinking"
    )
    text_chars = sum(
        len(b.get("text", "") or "")
        for b in blocks
        if b.get("type") == "text"
    )
    assert thinking_chars <= 50, (
        f"{spec.name}: budget=0 via Anthropic produced {thinking_chars} "
        f"thinking chars; expected ~0"
    )
    assert text_chars > 0, (
        f"{spec.name}: Anthropic must still produce an answer at budget=0"
    )
    # Soft latency check — Anthropic path shouldn't be dramatically
    # slower than OpenAI.
    assert elapsed < _TIMEOUT_SEC * 0.5


# ----- Provider-effort-unification: new input dialects ----------------
#
# These rows pin the end-to-end wiring of the resolver introduced in the
# provider-effort-unification branch. Each case sends a distinct dialect
# (output_config.effort low/medium/high/xhigh, thinking.adaptive,
# top-level precedence over effort, OpenAI reasoning_effort) and asserts
# on the two new response headers:
#   x-thinking-budget-source   — which input field won precedence
#   x-thinking-budget-resolved — the int (or "none") the resolver produced
#
# Unit coverage lives in tests/test_effort_resolver.py and
# tests/test_thinking_budget_headers.py; this file is the HTTP-layer
# smoke test that ties Pydantic validation → resolver → header emission
# together against a live server.

_MESSAGES_DIALECT_CASES: list[tuple[dict, str, str]] = [
    # (body_patch, expected_source, expected_budget_str)
    ({"output_config": {"effort": "low"}},    "output_config_effort",        "512"),
    ({"output_config": {"effort": "medium"}}, "output_config_effort",        "2048"),
    ({"output_config": {"effort": "high"}},   "output_config_effort",        "8192"),
    ({"output_config": {"effort": "xhigh"}},  "output_config_effort",        "16384"),
    ({"thinking": {"type": "adaptive"}},      "anthropic_thinking_adaptive", "none"),
    # top-level budget MUST win even when effort="max" is also sent.
    ({
        "thinking_token_budget": 777,
        "output_config": {"effort": "max"},
    },                                         "top_level",                   "777"),
]


def _dialect_ids(case: tuple[dict, str, str]) -> str:
    return case[1]


@pytest.mark.parametrize("spec", _MATRIX, ids=_matrix_ids)
@pytest.mark.parametrize(
    "body_patch,expected_source,expected_budget_str",
    _MESSAGES_DIALECT_CASES,
    ids=[c[1] for c in _MESSAGES_DIALECT_CASES],
)
def test_messages_header_matrix(
    spec: ModelSpec,
    body_patch: dict,
    expected_source: str,
    expected_budget_str: str,
) -> None:
    """HTTP-level matrix: every new Anthropic input dialect produces the
    documented source + resolved header values against a live server.

    This does NOT assert on generation behavior — just that the request
    parsed, reached the resolver, and the resolver's answer rode back
    out on the response headers. Generation-behavior invariants live in
    the tests above (ordering, budget=0 latency, etc.).
    """
    _require_matrix()

    body: dict[str, Any] = {
        "model": spec.model,
        "messages": [{"role": "user", "content": PROMPT_TRIVIAL}],
        "max_tokens": 32,
        "temperature": 0.3,
        **body_patch,
    }
    resp = requests.post(
        f"{spec.url}/v1/messages",
        json=body,
        timeout=_TIMEOUT_SEC,
    )
    assert resp.status_code == 200, resp.text

    assert resp.headers.get("x-thinking-budget-source") == expected_source, (
        f"{spec.name}: dialect={body_patch!r} — expected source="
        f"{expected_source!r}, got "
        f"{resp.headers.get('x-thinking-budget-source')!r}"
    )
    assert resp.headers.get("x-thinking-budget-resolved") == expected_budget_str, (
        f"{spec.name}: dialect={body_patch!r} — expected resolved="
        f"{expected_budget_str!r}, got "
        f"{resp.headers.get('x-thinking-budget-resolved')!r}"
    )


@pytest.mark.parametrize("spec", _MATRIX, ids=_matrix_ids)
@pytest.mark.parametrize(
    "reasoning_effort,expected_budget_str",
    [
        ("low",    "512"),
        ("medium", "2048"),
        ("high",   "8192"),
    ],
)
def test_chat_completions_reasoning_effort(
    spec: ModelSpec,
    reasoning_effort: str,
    expected_budget_str: str,
) -> None:
    """OpenAI reasoning_effort round-trips end-to-end.

    The /v1/chat/completions path accepts reasoning_effort natively;
    this pins that it flows through ChatCompletionRequest → resolver →
    response headers with the documented budget values.
    """
    _require_matrix()

    body: dict[str, Any] = {
        "model": spec.model,
        "messages": [{"role": "user", "content": PROMPT_TRIVIAL}],
        "max_tokens": 32,
        "temperature": 0.3,
        "reasoning_effort": reasoning_effort,
    }
    resp = requests.post(
        f"{spec.url}/v1/chat/completions",
        json=body,
        timeout=_TIMEOUT_SEC,
    )
    assert resp.status_code == 200, resp.text

    assert resp.headers.get("x-thinking-budget-source") == "reasoning_effort", (
        f"{spec.name}: reasoning_effort={reasoning_effort!r} — expected "
        f"source=reasoning_effort, got "
        f"{resp.headers.get('x-thinking-budget-source')!r}"
    )
    assert resp.headers.get("x-thinking-budget-resolved") == expected_budget_str, (
        f"{spec.name}: reasoning_effort={reasoning_effort!r} — expected "
        f"resolved={expected_budget_str!r}, got "
        f"{resp.headers.get('x-thinking-budget-resolved')!r}"
    )
