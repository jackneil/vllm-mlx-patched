# SPDX-License-Identifier: Apache-2.0
"""Live-model integration tests for thinking_token_budget.

Run explicitly:
    THINKING_BUDGET_TEST_MODEL=mlx-community/Qwen3-8B-Instruct-MLX \
        pytest tests/test_thinking_budget_integration.py -m integration -v
"""

import os

import pytest

pytestmark = pytest.mark.integration

MODEL = os.getenv("THINKING_BUDGET_TEST_MODEL")


def _skip_if_no_model():
    if not MODEL:
        pytest.skip("Set THINKING_BUDGET_TEST_MODEL=<hf-id> to run")


PROMPT = "Solve step by step: what is the sum of all prime numbers under 50?"


@pytest.fixture(scope="module")
async def engine():
    _skip_if_no_model()
    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.reasoning import get_parser

    eng = BatchedEngine(
        model_name=MODEL,
        reasoning_parser=get_parser("qwen3")(),
    )
    yield eng


async def _drain(engine, **kw):
    out = ""
    async for chunk in engine.stream_generate(prompt=PROMPT, max_tokens=1024, **kw):
        # chunk.text is accumulated per vllm_mlx/engine/base.py.
        out = chunk.text
    return out


async def test_budget_zero_closes_immediately(engine):
    out = await _drain(engine, thinking_token_budget=0)
    close = out.find("</think>")
    assert close >= 0 and close < 50


async def test_budget_512_caps_thinking(engine):
    out = await _drain(engine, thinking_token_budget=512)
    open_i = out.find("<think>")
    close_i = out.find("</think>")
    assert open_i >= 0 and close_i > open_i
    # Rough token proxy: whitespace-split.
    approx = len(out[open_i + len("<think>") : close_i].split())
    assert approx <= 700  # 512 + slack


async def test_budget_none_unbounded(engine):
    out = await _drain(engine, thinking_token_budget=None)
    assert out


async def test_message_produces_hint_in_output(engine):
    """With a message, the forced close is preceded by the hint tokens."""
    out = await _drain(
        engine,
        thinking_token_budget=64,
        thinking_budget_message="Wrap up now and answer.",
    )
    close_i = out.find("</think>")
    assert close_i > 0
    # The message should appear right before </think>.
    pre = out[max(0, close_i - 100) : close_i]
    assert "Wrap up now" in pre or "answer" in pre
