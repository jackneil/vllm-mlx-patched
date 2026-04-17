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


from vllm_mlx.api.models import ChatCompletionRequest


class TestChatCompletionRequestThinkingBudget:
    def _base(self, **overrides):
        payload = {
            "model": "qwen3",
            "messages": [{"role": "user", "content": "hi"}],
            **overrides,
        }
        return ChatCompletionRequest.model_validate(payload)

    def test_defaults(self):
        req = self._base()
        assert req.thinking_token_budget is None
        assert req.thinking_budget_message is None

    def test_both_fields(self):
        req = self._base(thinking_token_budget=512, thinking_budget_message="wrap up")
        assert req.thinking_token_budget == 512
        assert req.thinking_budget_message == "wrap up"

    def test_zero_accepted(self):
        req = self._base(thinking_token_budget=0)
        assert req.thinking_token_budget == 0

    def test_chat_template_kwargs_passthrough(self):
        req = self._base(chat_template_kwargs={"thinking_token_budget": 256})
        assert req.chat_template_kwargs == {"thinking_token_budget": 256}


import mlx.core as mx


def _make_processor(
    budget=5,
    start_ids=(100,),
    end_ids=(200,),
    message_ids=None,
    prompt_ids=None,
):
    from vllm_mlx.logits_processors.thinking_budget import (
        ThinkingTokenBudgetLogitsProcessor,
    )

    return ThinkingTokenBudgetLogitsProcessor(
        budget=budget,
        start_token_ids=list(start_ids),
        end_token_ids=list(end_ids),
        message_token_ids=list(message_ids) if message_ids else None,
        prompt_token_ids=list(prompt_ids) if prompt_ids else None,
    )


def _forced_token(processor, tokens):
    """Run the processor against a zero-logits tensor and return the
    argmax (the token it's forcing), or None if nothing was forced."""
    tok_arr = mx.array(tokens, dtype=mx.int32)
    vocab = 300
    logits = mx.zeros((1, vocab), dtype=mx.float32)
    out = processor(tok_arr, logits)
    forced = int(mx.argmax(out, axis=-1)[0])
    # If the peak is barely above zero, nothing was forced.
    if float(out[0, forced]) < 1e8:
        return None
    return forced


class TestProcessorStateMachine:
    def test_no_think_no_force(self):
        """Without <think> in output, no forcing happens regardless of tokens."""
        p = _make_processor(budget=3)
        assert _forced_token(p, [1, 2, 3, 4, 5]) is None

    def test_enters_think_and_forces_at_budget(self):
        p = _make_processor(budget=3)
        # Emit <think> then 3 thinking tokens — budget hit on the 3rd.
        _forced_token(p, [100])             # enter think
        _forced_token(p, [100, 1])          # count=1
        _forced_token(p, [100, 1, 2])       # count=2
        forced = _forced_token(p, [100, 1, 2, 3])  # count=3, should force
        assert forced == 200

    def test_budget_zero_forces_immediately(self):
        p = _make_processor(budget=0)
        forced = _forced_token(p, [100])  # <think> just emitted
        assert forced == 200

    def test_natural_end_no_force(self):
        """Model closes </think> on its own — no force."""
        p = _make_processor(budget=100)
        _forced_token(p, [100])         # enter
        _forced_token(p, [100, 1])      # thinking
        _forced_token(p, [100, 1, 200]) # model closes
        # After natural close, subsequent tokens produce no force.
        assert _forced_token(p, [100, 1, 200, 5]) is None

    def test_prompt_pre_injection_detected(self):
        """Prompt has <think> at end; first output counts as thinking.

        mlx_lm.BatchGenerator passes FULL history (prompt + output). Test
        inputs below include the prompt [5, 6, 100] followed by the
        output tokens — matching the production contract.
        """
        prompt = [5, 6, 100]  # prompt ends in <think> (id 100)
        p = _make_processor(budget=2, prompt_ids=prompt)
        # In _init_from_prompt: think_count = 3 - (2+1) = 0 (post-<think> is empty).
        # Step 1: output=[1] → think_count=1.
        # Step 2: output=[1, 2] → think_count=2, budget hit, force.
        _forced_token(p, prompt + [1])            # think_count=1
        forced = _forced_token(p, prompt + [1, 2])  # think_count=2, force
        assert forced == 200

    def test_multi_token_end_sequence(self):
        """End delimiter is multiple tokens — processor forces each in turn."""
        p = _make_processor(budget=1, end_ids=(200, 201, 202))
        _forced_token(p, [100])          # enter think, count=0
        forced1 = _forced_token(p, [100, 1])  # count=1, hit budget → force 200
        assert forced1 == 200
        forced2 = _forced_token(p, [100, 1, 200])  # force 201
        assert forced2 == 201
        forced3 = _forced_token(p, [100, 1, 200, 201])  # force 202
        assert forced3 == 202
        # After full sequence, no more forcing.
        assert _forced_token(p, [100, 1, 200, 201, 202]) is None

    def test_message_injection_prepends_message_tokens(self):
        """thinking_budget_message tokenizes to tokens that force BEFORE end_ids."""
        p = _make_processor(
            budget=1, end_ids=(200,), message_ids=[50, 51]
        )
        _forced_token(p, [100])              # enter think
        f1 = _forced_token(p, [100, 1])      # budget hit → first force = first message tok
        assert f1 == 50
        f2 = _forced_token(p, [100, 1, 50])  # second message tok
        assert f2 == 51
        f3 = _forced_token(p, [100, 1, 50, 51])  # then </think>
        assert f3 == 200
