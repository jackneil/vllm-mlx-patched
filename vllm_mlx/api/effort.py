# SPDX-License-Identifier: Apache-2.0
"""Dialect-agnostic resolver for thinking_token_budget.

Five provider-dialect signals can appear on a request:
  1. Top-level `thinking_token_budget` (vllm-mlx extension, all paths)
  2. Anthropic-native `thinking: {"type": ..., "budget_tokens": ...}`
  3. Anthropic `output_config: {"effort": "low"|"medium"|"high"|"xhigh"|"max"}`
  4. OpenAI-native `reasoning_effort: "low"|"medium"|"high"`
  5. Nothing — natural behavior

This module collapses all five into a single `ResolvedBudget`. See
`docs/superpowers/specs/2026-04-18-provider-effort-budget-unification-design.md`
for the precedence table and rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class EffortSource(StrEnum):
    """Which input field produced the resolved budget. Emitted as the
    `x-thinking-budget-source` response header."""
    TOP_LEVEL = "top_level"
    ANTHROPIC_THINKING_DISABLED = "anthropic_thinking_disabled"
    ANTHROPIC_THINKING_ENABLED = "anthropic_thinking_enabled"
    ANTHROPIC_THINKING_ADAPTIVE = "anthropic_thinking_adaptive"
    OUTPUT_CONFIG_EFFORT = "output_config_effort"
    REASONING_EFFORT = "reasoning_effort"
    DEFAULT = "default"


@dataclass(frozen=True)
class ResolvedBudget:
    """Result of resolving all budget-ish signals on a single request."""
    budget: int | None
    source: EffortSource
    max_tokens_floor: int | None
    effort_label: str | None


# Canonical effort → (budget, max_tokens_floor) table.
# max_tokens_floor is a client-side *hint* returned via
# `x-thinking-budget-max-tokens-floor`; it is not enforced server-side.
_EFFORT_TABLE: dict[str, tuple[int, int]] = {
    "low":    (512,   2048),
    "medium": (2048,  4096),
    "high":   (8192,  16384),
    "xhigh":  (16384, 32768),
    # "max" is computed dynamically in resolve_effort — see below.
}

# Hard cap for "max" effort: min(context_window // 2, _MAX_BUDGET_CAP).
# Keeps 1M-context models from burning 500k tokens on one reasoning pass.
_MAX_BUDGET_CAP: int = 65536

# Synonyms — client vocabularies that mean the same thing.
_EFFORT_ALIASES: dict[str, str] = {
    "minimal": "low",
    "normal":  "medium",
}


def resolve_effort(
    *,
    top_level_budget: int | None = None,
    anthropic_thinking: dict | None = None,
    output_config: dict | None = None,
    reasoning_effort: str | None = None,
    context_window: int = 131072,
) -> ResolvedBudget:
    """Collapse all dialect signals into a single ResolvedBudget.

    Pure function. No side effects. No exceptions on unknown effort strings —
    those fall through to DEFAULT with a WARN log.

    See module docstring for precedence order.
    """
    # 1. Top-level explicit budget wins over everything.
    if top_level_budget is not None:
        return ResolvedBudget(
            budget=top_level_budget,
            source=EffortSource.TOP_LEVEL,
            max_tokens_floor=None,
            effort_label=None,
        )

    # 2. Anthropic thinking dict.
    if isinstance(anthropic_thinking, dict):
        thinking_type = anthropic_thinking.get("type")
        if thinking_type == "disabled":
            return ResolvedBudget(
                budget=0,
                source=EffortSource.ANTHROPIC_THINKING_DISABLED,
                max_tokens_floor=None,
                effort_label=None,
            )
        if thinking_type == "enabled":
            return ResolvedBudget(
                budget=anthropic_thinking.get("budget_tokens"),
                source=EffortSource.ANTHROPIC_THINKING_ENABLED,
                max_tokens_floor=None,
                effort_label=None,
            )
        if thinking_type == "adaptive":
            return ResolvedBudget(
                budget=None,
                source=EffortSource.ANTHROPIC_THINKING_ADAPTIVE,
                max_tokens_floor=None,
                effort_label=None,
            )
        if thinking_type is not None:
            # Unknown type — log and fall through to lower-precedence signals
            # or to DEFAULT. Matches the old adapter's WARN-on-unknown.
            logger.warning(
                "Unknown anthropic_thinking.type=%r; falling through to "
                "lower-precedence signals or default.",
                thinking_type,
            )

    # 5. output_config.effort and 6. reasoning_effort handled in Tasks 3-4.

    # 7. Default.
    return ResolvedBudget(
        budget=None,
        source=EffortSource.DEFAULT,
        max_tokens_floor=None,
        effort_label=None,
    )
