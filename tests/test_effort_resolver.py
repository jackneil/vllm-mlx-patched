"""Unit tests for vllm_mlx.api.effort — dialect-agnostic budget resolver."""

import pytest

from vllm_mlx.api.effort import (
    EffortSource,
    ResolvedBudget,
    resolve_effort,
    _EFFORT_TABLE,
    _MAX_BUDGET_CAP,
)


def test_module_imports_and_exports_public_api():
    """Scaffold test: the module loads and exposes the documented surface."""
    assert EffortSource.DEFAULT == "default"
    assert ResolvedBudget(budget=None, source=EffortSource.DEFAULT,
                          max_tokens_floor=None, effort_label=None)
    assert callable(resolve_effort)
    assert isinstance(_EFFORT_TABLE, dict)
    assert isinstance(_MAX_BUDGET_CAP, int)


# -----------------------------------------------------------------------------
# Precedence: top-level `thinking_token_budget` (highest)
# -----------------------------------------------------------------------------

def test_top_level_int_wins_over_everything():
    """If the caller set thinking_token_budget=N, use N regardless of others."""
    result = resolve_effort(
        top_level_budget=1234,
        anthropic_thinking={"type": "enabled", "budget_tokens": 9999},
        output_config={"effort": "max"},
        reasoning_effort="high",
    )
    assert result.budget == 1234
    assert result.source == EffortSource.TOP_LEVEL


def test_top_level_zero_is_honored():
    """0 is a first-class value — force-close thinking at step 1."""
    result = resolve_effort(top_level_budget=0)
    assert result.budget == 0
    assert result.source == EffortSource.TOP_LEVEL


# -----------------------------------------------------------------------------
# Precedence: Anthropic `thinking.type`
# -----------------------------------------------------------------------------

def test_thinking_disabled_means_budget_zero():
    result = resolve_effort(anthropic_thinking={"type": "disabled"})
    assert result.budget == 0
    assert result.source == EffortSource.ANTHROPIC_THINKING_DISABLED


def test_thinking_enabled_with_budget_tokens():
    result = resolve_effort(
        anthropic_thinking={"type": "enabled", "budget_tokens": 256}
    )
    assert result.budget == 256
    assert result.source == EffortSource.ANTHROPIC_THINKING_ENABLED


def test_thinking_enabled_without_budget_tokens_is_none():
    """Enabled but no cap = None (natural behavior, same as adaptive)."""
    result = resolve_effort(anthropic_thinking={"type": "enabled"})
    assert result.budget is None
    assert result.source == EffortSource.ANTHROPIC_THINKING_ENABLED


def test_thinking_adaptive_is_none_not_warning():
    """adaptive was previously a WARN-and-ignore case; now it's first-class."""
    result = resolve_effort(anthropic_thinking={"type": "adaptive"})
    assert result.budget is None
    assert result.source == EffortSource.ANTHROPIC_THINKING_ADAPTIVE


def test_thinking_unknown_type_falls_through_to_default():
    """Old behavior only WARNed; new behavior still WARNs but returns DEFAULT
    so the request doesn't error out."""
    result = resolve_effort(anthropic_thinking={"type": "bogus"})
    assert result.budget is None
    assert result.source == EffortSource.DEFAULT


def test_default_when_no_signals():
    result = resolve_effort()
    assert result.budget is None
    assert result.source == EffortSource.DEFAULT
    assert result.max_tokens_floor is None
    assert result.effort_label is None
