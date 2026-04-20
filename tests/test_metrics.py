# SPDX-License-Identifier: Apache-2.0
"""Smoke test: Prometheus counters for Qwen3 runaway mitigation exist and inc()."""

from vllm_mlx.metrics import (
    qwen3_first_turn_no_think_applied_total,
    thinking_budget_clamp_fired_total,
)


def test_new_counters_start_at_zero():
    # Baseline — captured relative to whatever prior tests did.
    b1 = thinking_budget_clamp_fired_total.value
    b2 = qwen3_first_turn_no_think_applied_total.value
    thinking_budget_clamp_fired_total.inc()
    qwen3_first_turn_no_think_applied_total.inc(2)
    assert thinking_budget_clamp_fired_total.value == b1 + 1
    assert qwen3_first_turn_no_think_applied_total.value == b2 + 2
