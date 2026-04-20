"""Smoke test: SimpleEngine paths set thinking_budget_noop_reason='simple_engine'
on GenerationOutput when thinking_token_budget was requested but can't be enforced.

This is a unit-level pin -- not an integration test. We construct a minimal
GenerationOutput and verify the fields flow through correctly. Paired with the
fix that propagates _tb_* kwargs through _stream_generate_specprefill and
_stream_generate_text so the markers survive the inner-helper yields.
"""

from vllm_mlx.engine.base import GenerationOutput


def test_generation_output_carries_simple_engine_reason():
    """GenerationOutput accepts and stores thinking_budget_noop_reason='simple_engine'."""
    out = GenerationOutput(
        text="hi",
        tokens=[1, 2],
        finish_reason="stop",
        thinking_budget_applied=False,
        thinking_budget_noop_reason="simple_engine",
    )
    assert out.thinking_budget_applied is False
    assert out.thinking_budget_noop_reason == "simple_engine"


def test_generation_output_carries_mllm_path_reason():
    """GenerationOutput accepts and stores thinking_budget_noop_reason='mllm_path'."""
    out = GenerationOutput(
        text="hi",
        tokens=[1, 2],
        finish_reason="stop",
        thinking_budget_applied=False,
        thinking_budget_noop_reason="mllm_path",
    )
    assert out.thinking_budget_noop_reason == "mllm_path"


def test_generation_output_default_noop_reason_is_none():
    """When no thinking budget was requested, both fields default to None."""
    out = GenerationOutput(text="hi", tokens=[1])
    assert out.thinking_budget_applied is None
    assert out.thinking_budget_noop_reason is None


def test_stream_generate_specprefill_accepts_tb_kwargs():
    """_stream_generate_specprefill signature accepts tb_applied / tb_noop_reason."""
    import inspect

    from vllm_mlx.engine.simple import SimpleEngine

    sig = inspect.signature(SimpleEngine._stream_generate_specprefill)
    assert "tb_applied" in sig.parameters
    assert "tb_noop_reason" in sig.parameters


def test_stream_generate_text_accepts_tb_kwargs():
    """_stream_generate_text signature accepts tb_applied / tb_noop_reason."""
    import inspect

    from vllm_mlx.engine.simple import SimpleEngine

    sig = inspect.signature(SimpleEngine._stream_generate_text)
    assert "tb_applied" in sig.parameters
    assert "tb_noop_reason" in sig.parameters
