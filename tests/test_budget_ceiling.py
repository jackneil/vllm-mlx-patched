# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Layer 2 clamp helper."""

from vllm_mlx.api.budget_ceiling import apply_server_thinking_token_budget_ceiling
from vllm_mlx.api.effort import EffortSource, ResolvedBudget


def _resolved(budget, source=EffortSource.OUTPUT_CONFIG_EFFORT, floor=16384):
    return ResolvedBudget(
        budget=budget, source=source, max_tokens_floor=floor, effort_label="high"
    )


class TestNoopCases:
    def test_ceiling_none_is_noop(self):
        r = _resolved(8192)
        out, clamped_from, skip = apply_server_thinking_token_budget_ceiling(
            r, ceiling=None, engine_supports_processor=True
        )
        assert out is r
        assert clamped_from is None
        assert skip is None

    def test_resolved_none_is_noop(self):
        out, clamped_from, skip = apply_server_thinking_token_budget_ceiling(
            None, ceiling=2048, engine_supports_processor=True
        )
        assert out is None
        assert clamped_from is None
        assert skip is None

    def test_resolved_budget_none_is_noop(self):
        r = _resolved(None)
        out, clamped_from, skip = apply_server_thinking_token_budget_ceiling(
            r, ceiling=2048, engine_supports_processor=True
        )
        assert out is r
        assert clamped_from is None
        assert skip is None

    def test_resolved_budget_zero_is_noop(self):
        """budget=0 means client explicitly disabled thinking — never raise."""
        r = _resolved(0, source=EffortSource.ANTHROPIC_THINKING_DISABLED, floor=None)
        out, clamped_from, skip = apply_server_thinking_token_budget_ceiling(
            r, ceiling=2048, engine_supports_processor=True
        )
        assert out is r
        assert out.budget == 0
        assert clamped_from is None
        assert skip is None

    def test_budget_below_ceiling_is_noop(self):
        r = _resolved(1024)
        out, clamped_from, skip = apply_server_thinking_token_budget_ceiling(
            r, ceiling=2048, engine_supports_processor=True
        )
        assert out is r
        assert out.budget == 1024
        assert clamped_from is None
        assert skip is None

    def test_budget_equal_to_ceiling_is_noop(self):
        r = _resolved(2048)
        out, clamped_from, skip = apply_server_thinking_token_budget_ceiling(
            r, ceiling=2048, engine_supports_processor=True
        )
        assert out.budget == 2048
        assert clamped_from is None
        assert skip is None


class TestClamping:
    def test_budget_above_ceiling_is_clamped(self):
        r = _resolved(8192, source=EffortSource.OUTPUT_CONFIG_EFFORT, floor=16384)
        out, clamped_from, skip = apply_server_thinking_token_budget_ceiling(
            r, ceiling=2048, engine_supports_processor=True
        )
        assert out.budget == 2048
        assert out.source == EffortSource.OUTPUT_CONFIG_EFFORT  # source preserved
        assert out.max_tokens_floor is None  # stale floor cleared
        assert clamped_from == 8192
        assert skip is None
        # `replace` returns a new instance — original unchanged (frozen dataclass)
        assert r.budget == 8192
        assert r.max_tokens_floor == 16384

    def test_clamp_preserves_effort_label(self):
        r = _resolved(8192, source=EffortSource.OUTPUT_CONFIG_EFFORT)
        out, _, _ = apply_server_thinking_token_budget_ceiling(
            r, ceiling=2048, engine_supports_processor=True
        )
        assert out.effort_label == "high"

    def test_clamp_very_low_ceiling(self):
        r = _resolved(65536, source=EffortSource.REASONING_EFFORT)
        out, clamped_from, _ = apply_server_thinking_token_budget_ceiling(
            r, ceiling=128, engine_supports_processor=True
        )
        assert out.budget == 128
        assert clamped_from == 65536


class TestEngineNoopSkip:
    def test_skip_when_engine_unsupported(self):
        r = _resolved(8192)
        out, clamped_from, skip = apply_server_thinking_token_budget_ceiling(
            r, ceiling=2048, engine_supports_processor=False
        )
        # Helper returns unclamped resolved + skip_reason so header tells truth.
        assert out is r
        assert out.budget == 8192
        assert clamped_from is None
        assert skip == "engine-no-op"

    def test_skip_returns_original_on_equal_below_ceiling_too(self):
        """If resolved.budget <= ceiling, the skip path is NOT taken even when
        engine is unsupported — there was nothing to clamp in the first place."""
        r = _resolved(1024)
        out, clamped_from, skip = apply_server_thinking_token_budget_ceiling(
            r, ceiling=2048, engine_supports_processor=False
        )
        assert out is r
        assert clamped_from is None
        assert skip is None
