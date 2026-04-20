"""Integration tests for anthropic_adapter's use of the effort resolver."""

from vllm_mlx.api.anthropic_adapter import anthropic_to_openai
from vllm_mlx.api.anthropic_models import AnthropicRequest
from vllm_mlx.api.effort import EffortSource


def _mk(body: dict) -> AnthropicRequest:
    """Build an AnthropicRequest with sensible defaults for testing."""
    return AnthropicRequest(
        model="any-model",
        messages=[],
        max_tokens=1024,
        **body,
    )


def test_adapter_returns_tuple_with_resolved_budget():
    req = _mk({})
    result = anthropic_to_openai(req, context_window=131072)
    assert isinstance(result, tuple)
    assert len(result) == 2
    oa_req, resolved = result
    assert resolved.source == EffortSource.DEFAULT


def test_adapter_forwards_top_level_budget():
    req = _mk({"thinking_token_budget": 777})
    oa_req, resolved = anthropic_to_openai(req, context_window=131072)
    assert oa_req.thinking_token_budget == 777
    assert resolved.budget == 777
    assert resolved.source == EffortSource.TOP_LEVEL


def test_adapter_thinking_disabled_forces_zero():
    req = _mk({"thinking": {"type": "disabled"}})
    oa_req, resolved = anthropic_to_openai(req, context_window=131072)
    assert oa_req.thinking_token_budget == 0
    assert resolved.source == EffortSource.ANTHROPIC_THINKING_DISABLED


def test_adapter_thinking_adaptive_forwards_none():
    req = _mk({"thinking": {"type": "adaptive"}})
    oa_req, resolved = anthropic_to_openai(req, context_window=131072)
    assert oa_req.thinking_token_budget is None
    assert resolved.source == EffortSource.ANTHROPIC_THINKING_ADAPTIVE


def test_adapter_output_config_effort_high():
    req = _mk({"output_config": {"effort": "high"}})
    oa_req, resolved = anthropic_to_openai(req, context_window=131072)
    assert oa_req.thinking_token_budget == 8192
    assert resolved.source == EffortSource.OUTPUT_CONFIG_EFFORT
    assert resolved.effort_label == "high"


def test_adapter_output_config_effort_max_dynamic():
    """Context window influences the 'max' budget."""
    req = _mk({"output_config": {"effort": "max"}})
    _, small_ctx = anthropic_to_openai(req, context_window=32768)
    _, large_ctx = anthropic_to_openai(req, context_window=1_000_000)
    assert small_ctx.budget == 16384
    assert large_ctx.budget == 65536  # capped


def test_adapter_top_level_beats_output_config():
    """Precedence — top-level thinking_token_budget wins."""
    req = _mk(
        {
            "thinking_token_budget": 333,
            "output_config": {"effort": "max"},
        }
    )
    oa_req, resolved = anthropic_to_openai(req, context_window=131072)
    assert oa_req.thinking_token_budget == 333
    assert resolved.source == EffortSource.TOP_LEVEL


# =============================================================================
# C.3 — Pydantic validator tests for output_config.effort and thinking
# =============================================================================


def test_output_config_rejects_unknown_effort():
    import pytest
    from pydantic import ValidationError
    from vllm_mlx.api.anthropic_models import AnthropicRequest

    with pytest.raises(ValidationError):
        AnthropicRequest(
            model="t",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            output_config={"effort": "hgih"},
        )


def test_output_config_accepts_allowed_efforts():
    from vllm_mlx.api.anthropic_models import AnthropicRequest
    from vllm_mlx.api.effort import ALLOWED_EFFORT_LEVELS

    for level in sorted(ALLOWED_EFFORT_LEVELS):
        req = AnthropicRequest(
            model="t",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            output_config={"effort": level},
        )
        assert req.output_config["effort"] == level


def test_thinking_budget_tokens_rejects_string():
    import pytest
    from pydantic import ValidationError
    from vllm_mlx.api.anthropic_models import AnthropicRequest

    with pytest.raises(ValidationError):
        AnthropicRequest(
            model="t",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            thinking={"type": "enabled", "budget_tokens": "8192"},
        )


def test_thinking_budget_tokens_rejects_negative():
    import pytest
    from pydantic import ValidationError
    from vllm_mlx.api.anthropic_models import AnthropicRequest

    with pytest.raises(ValidationError):
        AnthropicRequest(
            model="t",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            thinking={"type": "enabled", "budget_tokens": -1},
        )


def test_thinking_budget_tokens_accepts_positive_int():
    from vllm_mlx.api.anthropic_models import AnthropicRequest

    req = AnthropicRequest(
        model="t",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        thinking={"type": "enabled", "budget_tokens": 8192},
    )
    assert req.thinking["budget_tokens"] == 8192


def test_thinking_budget_tokens_accepts_zero():
    """budget_tokens=0 is valid — means 'no thinking'. Don't reject it."""
    from vllm_mlx.api.anthropic_models import AnthropicRequest

    req = AnthropicRequest(
        model="t",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        thinking={"type": "enabled", "budget_tokens": 0},
    )
    assert req.thinking["budget_tokens"] == 0
