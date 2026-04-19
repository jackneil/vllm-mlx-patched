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
from enum import Enum

logger = logging.getLogger(__name__)


class EffortSource(str, Enum):
    """Which input field produced the resolved budget. Emitted as the
    `x-thinking-budget-source` response header.

    Subclasses ``(str, Enum)`` rather than ``StrEnum`` for Python 3.10 compat
    (StrEnum is 3.11+ only). Behaviorally equivalent: ``EffortSource.DEFAULT
    == "default"`` is True, and the server's header emission uses
    ``source.value`` so the wire format is identical either way.
    """

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
    "low": (512, 2048),
    "medium": (2048, 4096),
    "high": (8192, 16384),
    "xhigh": (16384, 32768),
    # "max" is computed dynamically in resolve_effort — see below.
}

# Hard cap for "max" effort: min(context_window // 2, _MAX_BUDGET_CAP).
# Keeps 1M-context models from burning 500k tokens on one reasoning pass.
_MAX_BUDGET_CAP: int = 65536

# Synonyms — client vocabularies that mean the same thing.
_EFFORT_ALIASES: dict[str, str] = {
    "minimal": "low",
    "normal": "medium",
}

# Public export: the canonical set of effort levels accepted by the resolver.
# Pydantic validators on ChatCompletionRequest.reasoning_effort and
# AnthropicRequest.output_config.effort import this to avoid drift if
# _EFFORT_TABLE or _EFFORT_ALIASES grows a new level. "max" is in
# _EFFORT_TABLE implicitly (handled dynamically) so we add it explicitly.
ALLOWED_EFFORT_LEVELS: frozenset[str] = frozenset(
    set(_EFFORT_TABLE.keys()) | set(_EFFORT_ALIASES.keys()) | {"max"}
)


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
        # Unknown value for `type` (includes non-string types).
        if thinking_type is not None:
            logger.warning(
                "[thinking-budget-resolver] Unknown anthropic_thinking.type=%r; "
                "falling through to lower-precedence signals.",
                thinking_type,
            )
        # Missing `type` key on a non-empty dict = malformed shape.
        # (Empty dict `{}` is DELIBERATELY treated as "no signal" — same
        # as thinking=None — so clients that use `thinking={}` as a
        # sentinel don't get spurious WARNs.)
        elif anthropic_thinking:
            logger.warning(
                "[thinking-budget-resolver] anthropic_thinking dict is "
                "missing `type` key (keys=%s); falling through to "
                "lower-precedence signals. Clients should include "
                "`type: \"enabled\"|\"disabled\"|\"adaptive\"` explicitly.",
                sorted(anthropic_thinking.keys()),
            )

    # 5. Anthropic output_config.effort (Claude Code wire format).
    if isinstance(output_config, dict):
        raw_effort = output_config.get("effort")
        if raw_effort is not None:
            resolved = _resolve_effort_string(
                raw_effort,
                context_window,
                EffortSource.OUTPUT_CONFIG_EFFORT,
            )
            if resolved is not None:
                return resolved

    # 6. OpenAI reasoning_effort.
    if reasoning_effort is not None:
        resolved = _resolve_effort_string(
            reasoning_effort,
            context_window,
            EffortSource.REASONING_EFFORT,
        )
        if resolved is not None:
            return resolved

    # 7. Default.
    return ResolvedBudget(
        budget=None,
        source=EffortSource.DEFAULT,
        max_tokens_floor=None,
        effort_label=None,
    )


def _resolve_effort_string(
    raw_effort: str,
    context_window: int,
    source: EffortSource,
) -> ResolvedBudget | None:
    """Look up an effort-style string in the table. Returns None if the string
    is unknown (caller falls through to lower precedence / DEFAULT).

    Preserves the raw client string in `effort_label` so the header reflects
    what the client actually sent (e.g., "minimal" not "low").
    """
    canonical = _EFFORT_ALIASES.get(raw_effort, raw_effort)

    if canonical == "max":
        # Dynamic: half the context, capped at _MAX_BUDGET_CAP to keep
        # 1M-context models from burning 500k tokens per reasoning pass.
        budget = min(context_window // 2, _MAX_BUDGET_CAP)
        # Cap the floor at a serving-realistic ceiling and leave at least
        # 1024 tokens of prompt headroom. Without this cap, a 1M-context
        # model returned floor=131072 which most serving topologies reject
        # outright. The `max(budget, ...)` guarantees the floor never
        # undershoots the budget itself (pointless to request a floor
        # below the very budget we're setting).
        _FLOOR_CEILING = 32768
        _PROMPT_HEADROOM = 1024
        floor = min(
            budget * 2,
            _FLOOR_CEILING,
            max(budget, context_window - _PROMPT_HEADROOM),
        )
        return ResolvedBudget(
            budget=budget,
            source=source,
            max_tokens_floor=floor,
            effort_label=raw_effort,
        )

    if canonical in _EFFORT_TABLE:
        budget, floor = _EFFORT_TABLE[canonical]
        return ResolvedBudget(
            budget=budget,
            source=source,
            max_tokens_floor=floor,
            effort_label=raw_effort,
        )

    logger.warning(
        "Unknown effort=%r (source=%s); falling through to DEFAULT.",
        raw_effort,
        source,
    )
    return None
