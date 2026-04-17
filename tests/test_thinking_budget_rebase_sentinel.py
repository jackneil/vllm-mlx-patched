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
