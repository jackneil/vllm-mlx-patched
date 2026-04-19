"""Unit tests for the response-header builder helper."""

from vllm_mlx.api.effort import EffortSource, ResolvedBudget


def test_helper_all_fields_on_top_level_budget():
    from vllm_mlx.server import _build_thinking_budget_headers

    resolved = ResolvedBudget(
        budget=512,
        source=EffortSource.TOP_LEVEL,
        max_tokens_floor=2048,
        effort_label=None,
    )
    headers = _build_thinking_budget_headers(resolved, applied=True)
    assert headers["x-thinking-budget-applied"] == "true"
    assert headers["x-thinking-budget-resolved"] == "512"
    assert headers["x-thinking-budget-source"] == "top_level"
    assert headers["x-thinking-budget-max-tokens-floor"] == "2048"


def test_helper_default_source_omits_floor():
    from vllm_mlx.server import _build_thinking_budget_headers

    resolved = ResolvedBudget(
        budget=None,
        source=EffortSource.DEFAULT,
        max_tokens_floor=None,
        effort_label=None,
    )
    headers = _build_thinking_budget_headers(resolved, applied=None)
    assert "x-thinking-budget-applied" not in headers  # applied=None → absent
    assert headers["x-thinking-budget-resolved"] == "none"
    assert headers["x-thinking-budget-source"] == "default"
    assert "x-thinking-budget-max-tokens-floor" not in headers


def test_helper_applied_false():
    from vllm_mlx.server import _build_thinking_budget_headers

    resolved = ResolvedBudget(
        budget=8192,
        source=EffortSource.OUTPUT_CONFIG_EFFORT,
        max_tokens_floor=16384,
        effort_label="high",
    )
    headers = _build_thinking_budget_headers(resolved, applied=False)
    assert headers["x-thinking-budget-applied"] == "false"
    assert headers["x-thinking-budget-resolved"] == "8192"
    assert headers["x-thinking-budget-source"] == "output_config_effort"
    assert headers["x-thinking-budget-max-tokens-floor"] == "16384"


def test_helper_zero_budget_omits_floor():
    """budget=0 means 'no thinking' — max_tokens_floor doesn't apply."""
    from vllm_mlx.server import _build_thinking_budget_headers

    resolved = ResolvedBudget(
        budget=0,
        source=EffortSource.ANTHROPIC_THINKING_DISABLED,
        max_tokens_floor=None,
        effort_label=None,
    )
    headers = _build_thinking_budget_headers(resolved, applied=True)
    assert headers["x-thinking-budget-resolved"] == "0"
    assert "x-thinking-budget-max-tokens-floor" not in headers


def test_noop_reason_emitted_when_applied_false():
    from vllm_mlx.server import _build_thinking_budget_headers

    resolved = ResolvedBudget(
        budget=512,
        source=EffortSource.TOP_LEVEL,
        max_tokens_floor=2048,
        effort_label=None,
    )
    headers = _build_thinking_budget_headers(
        resolved, applied=False, noop_reason="mllm_path",
    )
    assert headers["x-thinking-budget-applied"] == "false"
    assert headers["x-thinking-budget-noop-reason"] == "mllm_path"


def test_noop_reason_omitted_when_applied_true():
    from vllm_mlx.server import _build_thinking_budget_headers

    resolved = ResolvedBudget(
        budget=512,
        source=EffortSource.TOP_LEVEL,
        max_tokens_floor=2048,
        effort_label=None,
    )
    headers = _build_thinking_budget_headers(resolved, applied=True)
    assert "x-thinking-budget-noop-reason" not in headers


def test_noop_reason_omitted_when_none_even_on_false():
    from vllm_mlx.server import _build_thinking_budget_headers

    resolved = ResolvedBudget(
        budget=512,
        source=EffortSource.TOP_LEVEL,
        max_tokens_floor=2048,
        effort_label=None,
    )
    headers = _build_thinking_budget_headers(
        resolved, applied=False, noop_reason=None,
    )
    assert "x-thinking-budget-noop-reason" not in headers


def test_warn_when_max_tokens_below_floor(caplog):
    """When client sets max_tokens < resolver's max_tokens_floor, log a WARN
    so operators can diagnose the truncation mistake. Sibling of PR #12's
    WARN (max_tokens <= budget)."""
    import logging
    from vllm_mlx.server import _warn_if_max_tokens_below_floor
    from vllm_mlx.api.effort import EffortSource, ResolvedBudget

    resolved = ResolvedBudget(
        budget=8192,
        source=EffortSource.REASONING_EFFORT,
        max_tokens_floor=16384,
        effort_label="high",
    )
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.server"):
        _warn_if_max_tokens_below_floor(resolved, max_tokens=4000)
    msgs = [r.message for r in caplog.records]
    assert any(
        "[thinking-budget-resolver]" in m
        and "4000" in m and "16384" in m
        for m in msgs
    )


def test_no_warn_when_max_tokens_meets_floor(caplog):
    import logging
    from vllm_mlx.server import _warn_if_max_tokens_below_floor
    from vllm_mlx.api.effort import EffortSource, ResolvedBudget

    resolved = ResolvedBudget(
        budget=8192, source=EffortSource.REASONING_EFFORT,
        max_tokens_floor=16384, effort_label="high",
    )
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.server"):
        _warn_if_max_tokens_below_floor(resolved, max_tokens=20000)
    assert not caplog.records
