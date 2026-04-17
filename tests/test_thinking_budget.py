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


class _FakeTokenizer:
    """Minimal tokenizer stand-in that encodes strings to fixed ids."""
    _MAP = {
        "<think>": [100],
        "</think>": [200],
        "Wrap it up.": [50, 51],
    }

    def encode(self, text, add_special_tokens=False):
        return _FakeTokenizer._MAP[text]


class _FakeParser:
    start_token = "<think>"
    end_tokens = ["</think>"]


class TestAttachProcessor:
    def _call(self, **overrides):
        from vllm_mlx.scheduler import _attach_thinking_budget_processor

        base = dict(
            tokenizer=_FakeTokenizer(),
            reasoning_parser=_FakeParser(),
            budget=512,
            message=None,
            prompt_token_ids=None,
        )
        base.update(overrides)
        return _attach_thinking_budget_processor(**base)

    def test_returns_processor_when_valid(self):
        proc = self._call()
        assert proc is not None

    def test_none_when_budget_is_none(self):
        assert self._call(budget=None) is None

    def test_none_when_no_parser(self):
        assert self._call(reasoning_parser=None) is None

    def test_none_when_start_tokenize_fails(self):
        class BrokenTok:
            def encode(self, text, add_special_tokens=False):
                raise RuntimeError("oops")

        assert self._call(tokenizer=BrokenTok()) is None

    def test_message_tokenized_when_provided(self):
        proc = self._call(message="Wrap it up.")
        assert proc is not None
        # Force sequence includes message ids (50, 51) then end ids (200).
        assert proc._force_sequence == [50, 51, 200]


class TestServerResolver:
    def _resolve(self, top_level=None, template_kwargs=None):
        from vllm_mlx.server import _resolve_thinking_budget

        return _resolve_thinking_budget(top_level, template_kwargs)

    def test_none(self):
        assert self._resolve() == (None, None)

    def test_top_level(self):
        assert self._resolve(top_level={"b": 512, "m": "wrap"}) == (512, "wrap")

    def test_template_kwargs_only(self):
        assert self._resolve(
            template_kwargs={"thinking_token_budget": 512, "thinking_budget_message": "wrap"}
        ) == (512, "wrap")

    def test_top_level_wins(self):
        assert self._resolve(
            top_level={"b": 1024, "m": None},
            template_kwargs={"thinking_token_budget": 512, "thinking_budget_message": "wrap"},
        ) == (1024, "wrap")  # budget from top, message fills from template

    def test_zero_is_a_real_value(self):
        assert self._resolve(top_level={"b": 0, "m": None}) == (0, None)


class TestNoopCounter:
    def test_counter_increments(self):
        from vllm_mlx.metrics import thinking_budget_noop_total

        before = thinking_budget_noop_total.value
        thinking_budget_noop_total.inc()
        thinking_budget_noop_total.inc(2)
        assert thinking_budget_noop_total.value == before + 3


class TestApplyPropagation:
    """End-to-end: Request.thinking_budget_applied → RequestOutput →
    GenerationOutput. This test caught the CRITICAL-1 regression that the
    existence-only sentinel missed."""

    def test_merge_preserves_applied(self):
        from vllm_mlx.output_collector import RequestOutputCollector
        from vllm_mlx.request import RequestOutput

        collector = RequestOutputCollector()
        existing = RequestOutput(
            request_id="x",
            new_text="hello",
            thinking_budget_applied=True,
        )
        new = RequestOutput(
            request_id="x",
            new_text=" world",
            thinking_budget_applied=True,
        )
        merged = collector._merge_outputs(existing, new)
        assert merged.thinking_budget_applied is True

    def test_merge_with_false_applied(self):
        from vllm_mlx.output_collector import RequestOutputCollector
        from vllm_mlx.request import RequestOutput

        collector = RequestOutputCollector()
        existing = RequestOutput(request_id="x", thinking_budget_applied=False)
        new = RequestOutput(request_id="x", thinking_budget_applied=False)
        assert collector._merge_outputs(existing, new).thinking_budget_applied is False

    def test_scheduler_output_carries_applied_flag(self):
        """Scheduler's output-producing method must copy
        Request.thinking_budget_applied to the RequestOutput it produces."""
        from vllm_mlx.request import Request, RequestOutput, SamplingParams
        import inspect
        # Read _process_batch_responses source (the method that constructs
        # RequestOutput during the generation step); assert it references
        # thinking_budget_applied so this test breaks if someone deletes the
        # field propagation without replacing it with another mechanism.
        from vllm_mlx.scheduler import Scheduler
        source = inspect.getsource(Scheduler._process_batch_responses)
        assert "thinking_budget_applied" in source, (
            "Scheduler._process_batch_responses must reference thinking_budget_applied "
            "or the server header will always be 'false'"
        )

    def test_batched_engine_text_generate_constructs_with_applied(self):
        """BatchedEngine.generate text branch must set thinking_budget_applied
        on its GenerationOutput (text branch, non-streaming)."""
        import inspect
        from vllm_mlx.engine.batched import BatchedEngine
        source = inspect.getsource(BatchedEngine.generate)
        # Verify both the text-branch construction (line ~570) AND the return
        # reference output.thinking_budget_applied.
        assert source.count("thinking_budget_applied") >= 2, (
            "BatchedEngine.generate should reference thinking_budget_applied at "
            "least twice (MLLM branch and text branch GenerationOutput)"
        )

    def test_batched_engine_text_stream_constructs_with_applied(self):
        import inspect
        from vllm_mlx.engine.batched import BatchedEngine
        source = inspect.getsource(BatchedEngine.stream_generate)
        assert source.count("thinking_budget_applied") >= 2, (
            "BatchedEngine.stream_generate should reference "
            "thinking_budget_applied in both branches"
        )


class TestStreamingHeader:
    """The streaming path can't read output.thinking_budget_applied, so
    it uses a pre-flight check. M1: check _reasoning_parser presence too."""

    def _compute(self, is_mllm, reasoning_parser):
        from vllm_mlx.server import _streaming_header_value

        return _streaming_header_value(is_mllm=is_mllm, reasoning_parser=reasoning_parser)

    def test_mllm_always_false(self):
        assert self._compute(is_mllm=True, reasoning_parser=object()) == "false"

    def test_text_no_parser_false(self):
        assert self._compute(is_mllm=False, reasoning_parser=None) == "false"

    def test_text_with_parser_true(self):
        assert self._compute(is_mllm=False, reasoning_parser=object()) == "true"


class TestAnthropicPlumbing:
    """M3: /v1/messages accepts thinking_token_budget and plumbs it through
    the OpenAI adapter."""

    def _base(self, **overrides):
        payload = {
            "model": "qwen3",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "hi"}],
            **overrides,
        }
        from vllm_mlx.api.anthropic_models import AnthropicRequest

        return AnthropicRequest(**payload)

    def test_top_level_field(self):
        req = self._base(thinking_token_budget=512)
        assert req.thinking_token_budget == 512

    def test_thinking_nested_budget_tokens(self):
        """Anthropic-native form: {"thinking": {"budget_tokens": N}}."""
        req = self._base(thinking={"budget_tokens": 256})
        assert req.thinking == {"budget_tokens": 256}

    def test_adapter_translates_top_level(self):
        from vllm_mlx.api.anthropic_adapter import anthropic_to_openai

        req = self._base(thinking_token_budget=512, thinking_budget_message="wrap")
        openai = anthropic_to_openai(req)
        assert openai.thinking_token_budget == 512
        assert openai.thinking_budget_message == "wrap"

    def test_adapter_translates_nested_thinking(self):
        """Nested thinking.budget_tokens becomes OpenAI's
        thinking_token_budget."""
        from vllm_mlx.api.anthropic_adapter import anthropic_to_openai

        req = self._base(thinking={"budget_tokens": 256})
        openai = anthropic_to_openai(req)
        assert openai.thinking_token_budget == 256

    def test_adapter_top_level_wins_over_nested(self):
        from vllm_mlx.api.anthropic_adapter import anthropic_to_openai

        req = self._base(
            thinking_token_budget=1024,
            thinking={"budget_tokens": 256},
        )
        openai = anthropic_to_openai(req)
        assert openai.thinking_token_budget == 1024

    def test_adapter_thinking_disabled_forces_zero(self):
        """Anthropic type='disabled' means "don't think at all" —
        map to budget=0 (our immediate-close semantics)."""
        from vllm_mlx.api.anthropic_adapter import anthropic_to_openai

        req = self._base(thinking={"type": "disabled", "budget_tokens": 1024})
        openai = anthropic_to_openai(req)
        assert openai.thinking_token_budget == 0  # disabled wins over budget_tokens

    def test_adapter_thinking_enabled_uses_budget_tokens(self):
        from vllm_mlx.api.anthropic_adapter import anthropic_to_openai

        req = self._base(thinking={"type": "enabled", "budget_tokens": 512})
        openai = anthropic_to_openai(req)
        assert openai.thinking_token_budget == 512

    def test_adapter_unknown_type_ignores(self, caplog):
        """Unknown type values are ignored with a WARN log, not silently
        enforced."""
        import logging

        from vllm_mlx.api.anthropic_adapter import anthropic_to_openai

        req = self._base(thinking={"type": "unknown", "budget_tokens": 256})
        with caplog.at_level(logging.WARNING):
            openai = anthropic_to_openai(req)
        assert openai.thinking_token_budget is None
        assert any("not recognized" in rec.message for rec in caplog.records)

    def test_anthropic_budget_requested_helper(self):
        from vllm_mlx.server import _anthropic_budget_requested

        req_no_budget = self._base()
        req_top_level = self._base(thinking_token_budget=512)
        req_nested_enabled = self._base(thinking={"type": "enabled", "budget_tokens": 256})
        req_nested_disabled = self._base(thinking={"type": "disabled"})
        req_nested_only_budget = self._base(thinking={"budget_tokens": 256})

        assert _anthropic_budget_requested(req_no_budget) is False
        assert _anthropic_budget_requested(req_top_level) is True
        assert _anthropic_budget_requested(req_nested_enabled) is True
        assert _anthropic_budget_requested(req_nested_disabled) is True
        assert _anthropic_budget_requested(req_nested_only_budget) is True


class TestMessageTokenCoincidence:
    """Regression test for pre-mortem scenario 1: if the model naturally
    emits one of the message tokens during thinking BEFORE the budget fires,
    does the processor force-emit it a second time?"""

    def test_message_token_in_natural_thinking_is_ok(self):
        """Budget=5. Message tokens are [50, 51]. Model emits token 50
        naturally at count=2, then continues thinking until count=5.
        At budget, force sequence [50, 51, 200] kicks in.
        Accepted behavior: the natural 50 at count=2 is counted as a
        thinking token; the forced 50 at count=5 is the message prefix;
        user sees two 50s separated by tokens. Not pretty, but not
        corrupted."""
        p = _make_processor(
            budget=5,
            end_ids=(200,),
            message_ids=[50, 51],
        )
        _forced_token(p, [100])                # <think>
        _forced_token(p, [100, 1])             # count=1
        _forced_token(p, [100, 1, 50])         # count=2, natural 50
        _forced_token(p, [100, 1, 50, 2])      # count=3
        _forced_token(p, [100, 1, 50, 2, 3])   # count=4
        # count=5 hits budget, force sequence begins with token 50
        forced1 = _forced_token(p, [100, 1, 50, 2, 3, 4])
        assert forced1 == 50
        # Then 51, then 200
        forced2 = _forced_token(p, [100, 1, 50, 2, 3, 4, 50])
        assert forced2 == 51
        forced3 = _forced_token(p, [100, 1, 50, 2, 3, 4, 50, 51])
        assert forced3 == 200

    def test_end_token_in_natural_thinking_closes_early(self):
        """Edge case: if the model naturally emits </think> (token 200)
        while thinking, the processor exits think mode — model closed on
        its own. Subsequent tokens must NOT be forced."""
        p = _make_processor(budget=100, end_ids=(200,))
        _forced_token(p, [100])              # <think>
        _forced_token(p, [100, 1])           # count=1
        _forced_token(p, [100, 1, 200])      # model closed
        # Post-close: no force regardless of counter
        assert _forced_token(p, [100, 1, 200, 5]) is None
        assert _forced_token(p, [100, 1, 200, 5, 6]) is None
