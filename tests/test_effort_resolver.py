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
    assert ResolvedBudget(
        budget=None,
        source=EffortSource.DEFAULT,
        max_tokens_floor=None,
        effort_label=None,
    )
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


@pytest.mark.parametrize(
    "effort,expected_budget,expected_floor",
    [
        ("low", 512, 2048),
        ("medium", 2048, 4096),
        ("high", 8192, 16384),
        ("xhigh", 16384, 32768),
    ],
)
def test_output_config_effort_table_lookup(effort, expected_budget, expected_floor):
    result = resolve_effort(output_config={"effort": effort})
    assert result.budget == expected_budget
    assert result.source == EffortSource.OUTPUT_CONFIG_EFFORT
    assert result.max_tokens_floor == expected_floor
    assert result.effort_label == effort


def test_output_config_effort_max_with_small_context():
    """`max` effort on a 32k-context model produces budget=16384 (half the
    context) and a floor clamped to leave 1024 tokens of prompt headroom.
    Pre-v2, the floor was unclamped and could equal context_window exactly —
    leaving no room for the prompt itself."""
    from vllm_mlx.api.effort import EffortSource, resolve_effort

    result = resolve_effort(
        output_config={"effort": "max"},
        context_window=32768,
    )
    assert result.source == EffortSource.OUTPUT_CONFIG_EFFORT
    assert result.budget == 16384  # half of 32768, under the 65536 cap
    # Floor math (from the new formula):
    #   min(budget*2, 32768 ceiling, max(budget, ctx - 1024 headroom))
    #   = min(32768, 32768, max(16384, 31744))
    #   = min(32768, 32768, 31744)
    #   = 31744
    assert result.max_tokens_floor == 31744, (
        f"expected 31744 (ctx - 1024 prompt headroom); "
        f"got {result.max_tokens_floor}"
    )


def test_output_config_effort_max_floor_capped_at_ceiling():
    """On a 1M-context model, `max` floor must NOT exceed 32768 (serving-
    realistic ceiling). Pre-fix, 1M context produced floor=131072 which
    most serving topologies reject."""
    from vllm_mlx.api.effort import resolve_effort

    result = resolve_effort(
        reasoning_effort="max",
        context_window=1_000_000,
    )
    assert result.budget == 65536
    assert result.max_tokens_floor is not None
    assert result.max_tokens_floor <= 32768, (
        f"max_tokens_floor={result.max_tokens_floor} exceeds 32768 ceiling"
    )


def test_output_config_effort_max_floor_respects_prompt_headroom():
    """Tiny-context model: floor must leave 1024 tokens for the prompt."""
    from vllm_mlx.api.effort import resolve_effort

    result = resolve_effort(
        reasoning_effort="max",
        context_window=4096,
    )
    assert result.budget == 2048  # 4096 // 2
    assert result.max_tokens_floor is not None
    assert result.max_tokens_floor <= 4096 - 1024, (
        f"floor={result.max_tokens_floor} leaves no prompt headroom"
    )


def test_output_config_effort_max_capped_at_65k():
    """max = min(context_window // 2, 65536). 1M ctx → still 65536."""
    result = resolve_effort(
        output_config={"effort": "max"},
        context_window=1_000_000,
    )
    assert result.budget == 65536
    assert result.max_tokens_floor is not None
    assert result.effort_label == "max"


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("minimal", "low"),
        ("normal", "medium"),
    ],
)
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


@pytest.mark.parametrize(
    "effort,expected_budget",
    [
        ("low", 512),
        ("medium", 2048),
        ("high", 8192),
    ],
)
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


# ---- PR-C Task C.1: ALLOWED_EFFORT_LEVELS export ----

def test_allowed_effort_levels_covers_table_and_aliases():
    """ALLOWED_EFFORT_LEVELS must be the union of _EFFORT_TABLE keys and
    _EFFORT_ALIASES keys plus "max" (handled dynamically). Adding a new
    level to either must automatically expand the validated vocabulary."""
    from vllm_mlx.api.effort import (
        ALLOWED_EFFORT_LEVELS,
        _EFFORT_ALIASES,
        _EFFORT_TABLE,
    )

    expected = (
        set(_EFFORT_TABLE.keys()) | set(_EFFORT_ALIASES.keys()) | {"max"}
    )
    assert set(ALLOWED_EFFORT_LEVELS) == expected


def test_allowed_effort_levels_includes_core_synonyms():
    """Sanity: the current canonical levels and synonyms are present."""
    from vllm_mlx.api.effort import ALLOWED_EFFORT_LEVELS

    for level in ("minimal", "low", "normal", "medium", "high", "xhigh", "max"):
        assert level in ALLOWED_EFFORT_LEVELS, f"{level!r} missing"


# ---- PR-C Task C.2: malformed anthropic_thinking ----


def test_anthropic_thinking_missing_type_key_logs_warn(caplog):
    """Non-empty dict missing `type` must WARN and fall through to
    lower-precedence signals. Pre-fix, this silently returned DEFAULT."""
    import logging
    from vllm_mlx.api.effort import EffortSource, resolve_effort

    with caplog.at_level(logging.WARNING, logger="vllm_mlx.api.effort"):
        result = resolve_effort(anthropic_thinking={"budget_tokens": 100})
    assert result.source == EffortSource.DEFAULT
    msgs = [r.message for r in caplog.records]
    assert any(
        "[thinking-budget-resolver]" in m and "missing `type`" in m.lower()
        for m in msgs
    ), f"expected WARN; got {msgs}"


def test_anthropic_thinking_unknown_type_logs_warn_with_prefix(caplog):
    """Unknown `type` value already WARNed pre-fix but without the
    [thinking-budget-resolver] prefix. Ensure the prefix is present now."""
    import logging
    from vllm_mlx.api.effort import EffortSource, resolve_effort

    with caplog.at_level(logging.WARNING, logger="vllm_mlx.api.effort"):
        result = resolve_effort(anthropic_thinking={"type": "invalidtype"})
    assert result.source == EffortSource.DEFAULT
    assert any(
        "[thinking-budget-resolver]" in r.message for r in caplog.records
    )


def test_anthropic_thinking_non_str_type_logs_warn(caplog):
    """`type` as non-string (e.g. int) also malformed."""
    import logging
    from vllm_mlx.api.effort import EffortSource, resolve_effort

    with caplog.at_level(logging.WARNING, logger="vllm_mlx.api.effort"):
        result = resolve_effort(anthropic_thinking={"type": 42})
    assert result.source == EffortSource.DEFAULT
    assert any(
        "[thinking-budget-resolver]" in r.message for r in caplog.records
    )


def test_anthropic_thinking_none_does_not_warn(caplog):
    """thinking=None is the normal 'no signal' case — no WARN."""
    import logging
    from vllm_mlx.api.effort import EffortSource, resolve_effort

    with caplog.at_level(logging.WARNING, logger="vllm_mlx.api.effort"):
        result = resolve_effort(anthropic_thinking=None)
    assert result.source == EffortSource.DEFAULT
    assert not any(
        "[thinking-budget-resolver]" in r.message for r in caplog.records
    )


def test_anthropic_thinking_empty_dict_is_silent(caplog):
    """Empty dict {} is treated as 'no signal' (same as None). No WARN.

    Rationale: some clients construct thinking={} as a sentinel meaning
    'let defaults decide'. Treating it as malformed would break them. This
    is a deliberate design choice — documented here so future reviewers
    don't 'fix' it."""
    import logging
    from vllm_mlx.api.effort import EffortSource, resolve_effort

    with caplog.at_level(logging.WARNING, logger="vllm_mlx.api.effort"):
        result = resolve_effort(anthropic_thinking={})
    assert result.source == EffortSource.DEFAULT
    assert not any(
        "[thinking-budget-resolver]" in r.message for r in caplog.records
    )


# ---- "adaptive + explicit effort" ceiling respect (Qwen3.6 first-turn runaway) ----


def test_adaptive_plus_output_config_effort_uses_effort_ceiling():
    """CRITICAL: `thinking.type=adaptive` is model-specific trained
    behavior (Claude 4.5+ self-regulate). Open-weight models like
    Qwen3.6 don't honor it — so when `adaptive` lands alone, budget=None
    produces runaway thinking until the streaming cap fires.

    Claude Code sends BOTH `thinking.type=adaptive` AND
    `output_config.effort=high`. When both are present, use the effort
    level as a ceiling so non-Claude models can't run away."""
    from vllm_mlx.api.effort import EffortSource, resolve_effort

    result = resolve_effort(
        anthropic_thinking={"type": "adaptive"},
        output_config={"effort": "high"},
    )

    assert result.budget == 8192, (
        f"adaptive + effort=high must use effort's budget 8192; got {result.budget}"
    )
    assert result.source == EffortSource.OUTPUT_CONFIG_EFFORT


def test_adaptive_plus_reasoning_effort_uses_effort_ceiling():
    """Same for the OpenAI dialect path."""
    from vllm_mlx.api.effort import EffortSource, resolve_effort

    result = resolve_effort(
        anthropic_thinking={"type": "adaptive"},
        reasoning_effort="medium",
    )

    assert result.budget == 2048
    assert result.source == EffortSource.REASONING_EFFORT


def test_adaptive_alone_still_returns_none_budget():
    """When `adaptive` is the ONLY signal (no effort ceiling), preserve
    the existing `budget=None` natural behavior — we have no explicit
    ceiling to apply."""
    from vllm_mlx.api.effort import EffortSource, resolve_effort

    result = resolve_effort(anthropic_thinking={"type": "adaptive"})

    assert result.budget is None
    assert result.source == EffortSource.ANTHROPIC_THINKING_ADAPTIVE


def test_adaptive_plus_effort_precedence_prefers_output_config():
    """When `adaptive` + BOTH `output_config.effort` AND `reasoning_effort`
    are present, prefer output_config.effort (Anthropic path wins over
    OpenAI path, matching the top-level precedence table)."""
    from vllm_mlx.api.effort import EffortSource, resolve_effort

    result = resolve_effort(
        anthropic_thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        reasoning_effort="high",
    )

    assert result.budget == 512  # low
    assert result.source == EffortSource.OUTPUT_CONFIG_EFFORT


def test_disabled_plus_effort_still_disabled():
    """Regression: `type=disabled` must continue winning over effort —
    a client sending disabled means "no thinking," even if they also
    misguidedly include an effort level."""
    from vllm_mlx.api.effort import EffortSource, resolve_effort

    result = resolve_effort(
        anthropic_thinking={"type": "disabled"},
        output_config={"effort": "high"},
    )

    assert result.budget == 0
    assert result.source == EffortSource.ANTHROPIC_THINKING_DISABLED


def test_enabled_with_budget_plus_effort_uses_enabled_budget():
    """Regression: explicit `type=enabled` + `budget_tokens` must win
    over a coexisting effort level (explicit > heuristic)."""
    from vllm_mlx.api.effort import EffortSource, resolve_effort

    result = resolve_effort(
        anthropic_thinking={"type": "enabled", "budget_tokens": 4096},
        output_config={"effort": "high"},
    )

    assert result.budget == 4096
    assert result.source == EffortSource.ANTHROPIC_THINKING_ENABLED
