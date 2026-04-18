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


# -----------------------------------------------------------------------------
# Precedence: Anthropic output_config.effort
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("effort,expected_budget,expected_floor", [
    ("low",    512,   2048),
    ("medium", 2048,  4096),
    ("high",   8192,  16384),
    ("xhigh",  16384, 32768),
])
def test_output_config_effort_table_lookup(effort, expected_budget, expected_floor):
    result = resolve_effort(output_config={"effort": effort})
    assert result.budget == expected_budget
    assert result.source == EffortSource.OUTPUT_CONFIG_EFFORT
    assert result.max_tokens_floor == expected_floor
    assert result.effort_label == effort


def test_output_config_effort_max_with_small_context():
    """max = min(context_window // 2, _MAX_BUDGET_CAP). 32k ctx → 16k budget."""
    result = resolve_effort(
        output_config={"effort": "max"},
        context_window=32768,
    )
    assert result.budget == 16384
    assert result.max_tokens_floor == 32768
    assert result.source == EffortSource.OUTPUT_CONFIG_EFFORT
    assert result.effort_label == "max"


def test_output_config_effort_max_capped_at_65k():
    """max = min(context_window // 2, 65536). 1M ctx → still 65536."""
    result = resolve_effort(
        output_config={"effort": "max"},
        context_window=1_000_000,
    )
    assert result.budget == 65536
    assert result.max_tokens_floor is not None
    assert result.effort_label == "max"


@pytest.mark.parametrize("alias,canonical", [
    ("minimal", "low"),
    ("normal",  "medium"),
])
def test_output_config_effort_synonyms(alias, canonical):
    """minimal → low, normal → medium."""
    result = resolve_effort(output_config={"effort": alias})
    expected_budget, expected_floor = _EFFORT_TABLE[canonical]
    assert result.budget == expected_budget
    assert result.max_tokens_floor == expected_floor
    assert result.effort_label == alias  # preserve the raw client string


def test_output_config_effort_unknown_falls_through_to_default():
    """Unknown effort string logs WARN, returns DEFAULT — does not error."""
    result = resolve_effort(output_config={"effort": "absurdly_high"})
    assert result.budget is None
    assert result.source == EffortSource.DEFAULT


def test_output_config_without_effort_key_is_default():
    """output_config present but no effort key = default."""
    result = resolve_effort(output_config={"other": "thing"})
    assert result.budget is None
    assert result.source == EffortSource.DEFAULT


# -----------------------------------------------------------------------------
# Precedence: output_config LOSES to higher-precedence signals
# -----------------------------------------------------------------------------

def test_output_config_effort_loses_to_top_level():
    result = resolve_effort(
        top_level_budget=111,
        output_config={"effort": "max"},
        context_window=32768,
    )
    assert result.budget == 111
    assert result.source == EffortSource.TOP_LEVEL


def test_output_config_effort_loses_to_thinking_enabled():
    result = resolve_effort(
        anthropic_thinking={"type": "enabled", "budget_tokens": 222},
        output_config={"effort": "high"},
    )
    assert result.budget == 222
    assert result.source == EffortSource.ANTHROPIC_THINKING_ENABLED


# -----------------------------------------------------------------------------
# Precedence: OpenAI reasoning_effort
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("effort,expected_budget", [
    ("low",    512),
    ("medium", 2048),
    ("high",   8192),
])
def test_reasoning_effort_openai_levels(effort, expected_budget):
    result = resolve_effort(reasoning_effort=effort)
    assert result.budget == expected_budget
    assert result.source == EffortSource.REASONING_EFFORT
    assert result.effort_label == effort


def test_reasoning_effort_accepts_xhigh_via_same_table():
    """Permissive: OpenAI spec only defines low/medium/high, but we accept
    xhigh from OpenAI-extension clients via the same table."""
    result = resolve_effort(reasoning_effort="xhigh")
    assert result.budget == 16384
    assert result.source == EffortSource.REASONING_EFFORT


def test_reasoning_effort_unknown_falls_through_to_default():
    result = resolve_effort(reasoning_effort="bogus")
    assert result.source == EffortSource.DEFAULT
    assert result.budget is None


def test_reasoning_effort_loses_to_output_config():
    """output_config.effort comes first in precedence (Anthropic-path caller),
    reasoning_effort is OpenAI-path only — but the resolver doesn't know the
    path, so precedence decides."""
    result = resolve_effort(
        output_config={"effort": "low"},
        reasoning_effort="high",
    )
    assert result.budget == 512
    assert result.source == EffortSource.OUTPUT_CONFIG_EFFORT
