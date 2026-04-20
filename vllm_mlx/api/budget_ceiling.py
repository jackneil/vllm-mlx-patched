# SPDX-License-Identifier: Apache-2.0
"""Server-side ceiling for resolved thinking-token budgets.

See docs/superpowers/specs/2026-04-20-qwen3-runaway-fix-design.md.

The helper is invoked at every place a `ResolvedBudget` flows into
`chat_kwargs["thinking_token_budget"]`:
  1. anthropic_adapter.py — before ChatCompletionRequest construction
  2. server.py _resolve_thinking_budget — top-level + chat_template_kwargs path
  3. server.py OpenAI handler — after resolve_effort()
  4. server.py Anthropic handler — defensive re-clamp (redundant, idempotent)

Sites 1 and 4 overlap on the Anthropic path — the helper is idempotent
(resolved.budget <= ceiling returns input unchanged) so this is safe.
"""

from __future__ import annotations

import dataclasses

from ..metrics import thinking_budget_clamp_fired_total
from .effort import ResolvedBudget


def apply_server_thinking_token_budget_ceiling(
    resolved: ResolvedBudget | None,
    ceiling: int | None,
    *,
    engine_supports_processor: bool,
) -> tuple[ResolvedBudget | None, int | None, str | None]:
    """Clamp a resolved thinking-token budget down to the server ceiling.

    Returns ``(resolved_or_clamped, clamped_from, skip_reason)``.

    Rules (in order):
      1. No-op when ``ceiling`` is ``None`` — feature disabled.
      2. No-op when ``resolved`` is ``None`` — no budget to clamp.
      3. No-op when ``resolved.budget`` is ``None`` — resolver returned
         "no budget," nothing to clamp.
      4. No-op when ``resolved.budget == 0`` — client explicitly disabled
         thinking (``ANTHROPIC_THINKING_DISABLED``). Never raise.
      5. No-op when ``resolved.budget <= ceiling`` — already under cap.
      6. Skip with ``skip_reason="engine-no-op"`` when the budget WOULD
         clamp but the engine doesn't support the logits processor
         (MLLM, SimpleEngine, missing reasoning_parser). Returns unclamped
         resolved so the ``x-thinking-budget-clamp-skipped`` header reflects
         reality.
      7. Otherwise: returns ``dataclasses.replace(resolved,
         budget=ceiling, max_tokens_floor=None)`` — the floor was computed
         for the pre-clamp budget and is now stale/contradictory, so we
         drop it. Source and effort_label preserved. Increments the
         ``thinking_budget_clamp_fired_total`` counter.
    """
    if ceiling is None:
        return resolved, None, None
    if resolved is None:
        return None, None, None
    if resolved.budget is None:
        return resolved, None, None
    if resolved.budget == 0:
        return resolved, None, None
    if resolved.budget <= ceiling:
        return resolved, None, None
    if not engine_supports_processor:
        return resolved, None, "engine-no-op"

    clamped_from = resolved.budget
    clamped = dataclasses.replace(resolved, budget=ceiling, max_tokens_floor=None)
    thinking_budget_clamp_fired_total.inc()
    return clamped, clamped_from, None
