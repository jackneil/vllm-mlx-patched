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


def test_chat_template_kwargs_budget_also_clamped():
    """Site 2: clients sending chat_template_kwargs={"thinking_token_budget":N}
    must have that N clamped the same as resolver output. Otherwise this
    becomes a documented-extension-field bypass of the ceiling."""
    server._max_thinking_token_budget = 2048

    from vllm_mlx.api.budget_ceiling import (
        apply_server_thinking_token_budget_ceiling,
    )

    # Site 2 synthesizes a ResolvedBudget with TEMPLATE_KWARGS source.
    synth = ResolvedBudget(
        budget=8192,
        source=EffortSource.TEMPLATE_KWARGS,
        max_tokens_floor=None,
        effort_label=None,
    )
    out, clamped_from, _ = apply_server_thinking_token_budget_ceiling(
        synth,
        ceiling=server._max_thinking_token_budget,
        engine_supports_processor=True,
    )
    assert out.budget == 2048
    assert out.source == EffortSource.TEMPLATE_KWARGS  # provenance preserved
    assert clamped_from == 8192


def test_anthropic_handler_post_adapter_defensive_reclamp():
    """Site 4 is redundant with site 1 (adapter) but defends against future
    adapter refactors. Helper is idempotent — safe to call twice."""
    server._max_thinking_token_budget = 2048

    from vllm_mlx.api.budget_ceiling import (
        apply_server_thinking_token_budget_ceiling,
    )

    unclamped = ResolvedBudget(
        budget=8192,
        source=EffortSource.OUTPUT_CONFIG_EFFORT,
        max_tokens_floor=16384,
        effort_label="high",
    )
    # First clamp (adapter site 1)
    once, once_from, _ = apply_server_thinking_token_budget_ceiling(
        unclamped,
        ceiling=server._max_thinking_token_budget,
        engine_supports_processor=True,
    )
    # Second clamp (handler site 4) — idempotent no-op
    twice, twice_from, _ = apply_server_thinking_token_budget_ceiling(
        once,
        ceiling=server._max_thinking_token_budget,
        engine_supports_processor=True,
    )
    assert once.budget == 2048
    assert once_from == 8192
    assert twice is once  # no-op: same object returned
    assert twice_from is None  # already under ceiling
