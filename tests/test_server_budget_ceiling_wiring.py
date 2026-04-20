# SPDX-License-Identifier: Apache-2.0
"""Integration tests: --max-thinking-token-budget clamp at every resolve site."""

import pytest

from vllm_mlx import server
from vllm_mlx.api.effort import EffortSource, ResolvedBudget


@pytest.fixture(autouse=True)
def reset_server_state():
    """Ensure test isolation — each test sees a clean ceiling."""
    original_ceiling = server._max_thinking_token_budget
    original_disable = server._disable_qwen3_first_turn_no_think
    yield
    server._max_thinking_token_budget = original_ceiling
    server._disable_qwen3_first_turn_no_think = original_disable


def test_openai_handler_clamps_reasoning_effort_high():
    """Sanity: helper is callable from a server state with ceiling set."""
    server._max_thinking_token_budget = 2048

    from vllm_mlx.api.budget_ceiling import (
        apply_server_thinking_token_budget_ceiling,
    )

    resolved = ResolvedBudget(
        budget=8192,
        source=EffortSource.REASONING_EFFORT,
        max_tokens_floor=16384,
        effort_label="high",
    )
    clamped, clamped_from, skip = apply_server_thinking_token_budget_ceiling(
        resolved,
        ceiling=server._max_thinking_token_budget,
        engine_supports_processor=True,
    )
    assert clamped.budget == 2048
    assert clamped_from == 8192
    assert skip is None
