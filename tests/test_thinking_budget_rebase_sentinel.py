# SPDX-License-Identifier: Apache-2.0
"""Arena-critical rebase sentinel for thinking-budget plumbing.

See UPSTREAM_PIN.md invariant #9. If this file fails after a rebase,
stop — do not merge. A silent merge can break the feature without
surfacing a conflict.
"""

import inspect

import pytest


class TestThinkingBudgetSentinel:
    def test_sampling_params_has_fields(self):
        from vllm_mlx.request import SamplingParams

        sp = SamplingParams(thinking_token_budget=512, thinking_budget_message="x")
        assert sp.thinking_token_budget == 512
        assert sp.thinking_budget_message == "x"

    def test_chat_completion_request_has_fields(self):
        from vllm_mlx.api.models import ChatCompletionRequest

        req = ChatCompletionRequest.model_validate(
            {
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "thinking_token_budget": 1,
                "thinking_budget_message": "y",
            }
        )
        assert req.thinking_token_budget == 1
        assert req.thinking_budget_message == "y"

    def test_processor_class_exists_and_callable(self):
        from vllm_mlx.logits_processors import ThinkingTokenBudgetLogitsProcessor

        p = ThinkingTokenBudgetLogitsProcessor(
            budget=1, start_token_ids=[100], end_token_ids=[200]
        )
        # Must be callable with (tokens, logits) -> logits (mlx_lm contract).
        sig = inspect.signature(p.__call__)
        assert list(sig.parameters.keys()) == ["tokens", "logits"]

    def test_bg_kwargs_includes_logits_processors(self):
        """The _bg_kwargs filter must forward logits_processors — if a rebase
        of mlx_lm drops it, the processor never runs."""
        from vllm_mlx.scheduler import _BG_INIT_PARAMS

        assert "logits_processors" in _BG_INIT_PARAMS

    def test_attach_helper_exists(self):
        from vllm_mlx.scheduler import _attach_thinking_budget_processor

        assert callable(_attach_thinking_budget_processor)

    def test_metrics_counter_exists(self):
        from vllm_mlx.metrics import thinking_budget_noop_total

        assert hasattr(thinking_budget_noop_total, "inc")
        assert hasattr(thinking_budget_noop_total, "value")

    def test_request_output_has_applied_field(self):
        from vllm_mlx.request import RequestOutput

        ro = RequestOutput(request_id="x")
        assert hasattr(ro, "thinking_budget_applied")

    def test_attach_helper_builds_processor(self):
        """Exercises the delimiter resolution + processor construction
        happy path with a minimal fake tokenizer/parser."""
        from vllm_mlx.logits_processors import ThinkingTokenBudgetLogitsProcessor

        class _FakeTok:
            _MAP = {"<think>": [100], "</think>": [200]}

            def encode(self, text, add_special_tokens=False):
                return _FakeTok._MAP.get(text, [])

        class _FakeParser:
            start_token = "<think>"
            end_tokens = ["</think>"]

        from vllm_mlx.scheduler import _attach_thinking_budget_processor

        proc = _attach_thinking_budget_processor(
            tokenizer=_FakeTok(),
            reasoning_parser=_FakeParser(),
            budget=512,
            message=None,
            prompt_token_ids=None,
        )
        assert isinstance(proc, ThinkingTokenBudgetLogitsProcessor)

    def test_scheduler_output_propagates_applied(self):
        """Pin the CRITICAL-1 fix: scheduler's _process_batch_responses must
        construct RequestOutput with thinking_budget_applied populated
        from Request.thinking_budget_applied. Use a regex-tolerant source
        match so reformatting (spaces, intermediate variables, comments)
        doesn't trip the test — only deletion does.

        NOTE: plan referenced _generate_outputs; the actual method name in
        this codebase is _process_batch_responses."""
        import inspect
        import re

        from vllm_mlx.scheduler import Scheduler

        source = inspect.getsource(Scheduler._process_batch_responses)
        # Accept: `thinking_budget_applied=request.thinking_budget_applied`
        # OR:    `thinking_budget_applied=applied` after `applied = request.thinking_budget_applied`
        # OR:    any other reference that keeps the field wired through.
        assert re.search(
            r"thinking_budget_applied\s*=\s*(?:request\.thinking_budget_applied|\w+)",
            source,
        ), (
            "Regression: Scheduler._process_batch_responses no longer propagates "
            "Request.thinking_budget_applied → RequestOutput. Server header "
            "will always be 'false' on the happy-path text branch."
        )

    def test_batched_engine_text_branch_propagates_applied(self):
        """Pin the CRITICAL-1 fix on BatchedEngine's text branch."""
        import inspect
        from vllm_mlx.engine.batched import BatchedEngine

        gen_src = inspect.getsource(BatchedEngine.generate)
        stream_src = inspect.getsource(BatchedEngine.stream_generate)

        # Each method must reference thinking_budget_applied >=2 times:
        # once in MLLM branch, once in text branch.
        assert gen_src.count("thinking_budget_applied") >= 2, (
            "BatchedEngine.generate regression: text branch no longer "
            "propagates thinking_budget_applied"
        )
        assert stream_src.count("thinking_budget_applied") >= 2, (
            "BatchedEngine.stream_generate regression: text branch no longer "
            "propagates thinking_budget_applied"
        )

    def test_merge_outputs_preserves_applied(self):
        """output_collector._merge_outputs must preserve the field."""
        from vllm_mlx.output_collector import RequestOutputCollector
        from vllm_mlx.request import RequestOutput

        coll = RequestOutputCollector()
        a = RequestOutput(request_id="x", thinking_budget_applied=True)
        b = RequestOutput(request_id="x", thinking_budget_applied=True)
        assert coll._merge_outputs(a, b).thinking_budget_applied is True

    def test_insert_call_accepts_logits_processors(self):
        """CRITICAL-2 (pre-mortem S2): _BG_INIT_PARAMS is derived from
        BatchGenerator.__init__ but the production path forwards
        logits_processors via .insert(). A rebase that moves the kwarg
        between __init__ and insert must fail this sentinel."""
        import inspect

        try:
            from mlx_lm.generate import BatchGenerator
        except ImportError:
            import pytest

            pytest.skip("mlx_lm not installed")

        insert_sig = inspect.signature(BatchGenerator.insert)
        assert "logits_processors" in insert_sig.parameters, (
            "Upstream mlx_lm.BatchGenerator.insert no longer accepts "
            "logits_processors. The feature will silently stop working. "
            "Update vllm_mlx/scheduler.py to match the new API surface."
        )
