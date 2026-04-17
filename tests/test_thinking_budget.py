# SPDX-License-Identifier: Apache-2.0
"""Unit tests for per-request thinking token budget."""

import pytest

from vllm_mlx.request import SamplingParams


class TestSamplingParamsThinkingBudget:
    def test_defaults(self):
        sp = SamplingParams()
        assert sp.thinking_token_budget is None
        assert sp.thinking_budget_message is None

    def test_accepts_values(self):
        sp = SamplingParams(thinking_token_budget=512, thinking_budget_message="Wrap it up.")
        assert sp.thinking_token_budget == 512
        assert sp.thinking_budget_message == "Wrap it up."

    def test_zero_budget_is_valid(self):
        sp = SamplingParams(thinking_token_budget=0)
        assert sp.thinking_token_budget == 0

    def test_negative_budget_rejected(self):
        """Validate at construction — negative budgets are nonsense."""
        with pytest.raises(ValueError, match="thinking_token_budget"):
            SamplingParams(thinking_token_budget=-1)
