# SPDX-License-Identifier: Apache-2.0
"""Arena-critical rebase sentinel for thinking-budget plumbing.

See UPSTREAM_PIN.md invariant #9. If this file fails after a rebase,
stop — do not merge. A silent merge can break the feature without
surfacing a conflict.
"""

import inspect


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

    def test_rebase_sentinel_noop_reason_field_on_request(self):
        """noop_reason field on Request is pinned to survive rebase."""
        from vllm_mlx.request import Request

        assert hasattr(Request, "__dataclass_fields__")
        assert "thinking_budget_noop_reason" in Request.__dataclass_fields__, (
            "Request.thinking_budget_noop_reason removed — rebase regression"
        )
        f = Request.__dataclass_fields__["thinking_budget_noop_reason"]
        assert f.default is None, (
            "Default for thinking_budget_noop_reason changed; "
            "header emission depends on None sentinel."
        )

    def test_rebase_sentinel_noop_reason_field_on_request_output(self):
        """noop_reason field on RequestOutput is pinned to survive rebase."""
        from vllm_mlx.request import RequestOutput

        assert "thinking_budget_noop_reason" in RequestOutput.__dataclass_fields__, (
            "RequestOutput.thinking_budget_noop_reason removed — rebase regression"
        )
        f = RequestOutput.__dataclass_fields__["thinking_budget_noop_reason"]
        assert f.default is None, (
            "Default for thinking_budget_noop_reason changed; "
            "header emission depends on None sentinel."
        )

    def test_generation_output_has_applied_field(self):
        """Pin the api-surface dataclass: server.py reads
        GenerationOutput.thinking_budget_applied via getattr(..., None).
        A silent rename here would cause the x-thinking-budget-applied
        header to return 'false' for every successful request (DCR
        Scenario 3: 5% retry storm)."""
        from vllm_mlx.engine.base import GenerationOutput

        go = GenerationOutput(text="")
        assert hasattr(go, "thinking_budget_applied")

    def test_anthropic_handlers_forward_budget_to_chat_kwargs(self):
        """Pin the DCR-CRITICAL-1 fix: both Anthropic handlers must add
        thinking_token_budget AND thinking_budget_message to chat_kwargs
        when the openai_request has them set. A rebase that deletes this
        forwarding would make the feature inert on /v1/messages while
        still emitting misleading response headers.

        Regex is intentionally lenient (matches intermediate-variable
        refactors) to match the sibling sentinel's drift policy."""
        import inspect
        import re

        from vllm_mlx import server

        _tok_re = r'chat_kwargs\[["\']thinking_token_budget["\']\]\s*=\s*\w+'
        _msg_re = r'chat_kwargs\[["\']thinking_budget_message["\']\]\s*=\s*\w+'

        for fn_name in ("create_anthropic_message", "_stream_anthropic_messages"):
            src = inspect.getsource(getattr(server, fn_name))
            assert re.search(_tok_re, src), (
                f"{fn_name} regression: no longer forwards thinking_token_budget "
                f"to chat_kwargs — /v1/messages will be inert for budget requests."
            )
            assert re.search(_msg_re, src), (
                f"{fn_name} regression: no longer forwards thinking_budget_message "
                f"to chat_kwargs — wrap-up hint feature broken on /v1/messages."
            )

    def test_openai_chat_completion_emits_budget_header(self):
        """Pin the response-header emission on the OpenAI handler too.
        DCR Wave-3: Wave-2 sentinel only covered the Anthropic handler,
        but invariant #9 claims both chat-completion handlers emit the
        header. Deleting the OpenAI emission silently passed all 63
        tests before this assertion.

        Task 9 refactored emission to go through `_build_thinking_budget_headers`,
        which also emits x-thinking-budget-resolved/source/max-tokens-floor.
        The sentinel now counts calls to that helper (>=2 = streaming +
        non-streaming branches)."""
        import inspect

        from vllm_mlx import server

        src = inspect.getsource(server.create_chat_completion)
        occurrences = src.count("_build_thinking_budget_headers")
        assert occurrences >= 2, (
            f"create_chat_completion regression: _build_thinking_budget_headers "
            f"appears only {occurrences} time(s) — expected >=2 (streaming + "
            f"non-streaming branches). Clients lose budget-enforcement signal."
        )

    def test_streaming_header_call_sites_bind_engine_type(self):
        """Pin the Wave-2 O-1 call-site wiring. The helper _streaming_header_value
        has `engine_supports_budget` as a required kwarg; both call sites
        must pass `isinstance(engine, BatchedEngine)` so SimpleEngine gets
        "false". A rebase that replaces this with a literal `True` would
        re-introduce the Wave-2 CRITICAL (simple-mode streaming lies)
        without any unit test failing."""
        import inspect
        import re

        from vllm_mlx import server

        pattern = (
            r"engine_supports_budget\s*=\s*isinstance\(\s*engine\s*,"
            r"\s*BatchedEngine\s*\)"
        )
        for fn_name in ("create_chat_completion", "create_anthropic_message"):
            src = inspect.getsource(getattr(server, fn_name))
            assert re.search(pattern, src), (
                f"{fn_name} regression: _streaming_header_value is called "
                f"without engine_supports_budget=isinstance(engine, "
                f"BatchedEngine). SimpleEngine streaming will lie to clients."
            )

    def test_anthropic_handler_emits_budget_header(self):
        """Pin the response-header emission. DCR Wave-2 CRITICAL T-1: if
        someone deletes the `x-thinking-budget-applied` header line from
        the Anthropic handler, the client has no signal about budget
        enforcement even though the engine correctly applies or rejects.
        Wave-1 sentinel caught forwarding deletion; this one catches
        emission deletion.

        Note: both streaming and non-streaming header logic lives in
        `create_anthropic_message` (the outer handler builds the
        StreamingResponse headers dict AND the non-streaming
        Response/JSONResponse headers). `_stream_anthropic_messages` is
        the generator that yields SSE event bodies — it doesn't set
        HTTP headers.

        Task 9 refactored emission to go through `_build_thinking_budget_headers`,
        which also emits x-thinking-budget-resolved/source/max-tokens-floor.
        The sentinel now counts calls to that helper (>=2 = streaming +
        non-streaming branches).
        """
        import inspect

        from vllm_mlx import server

        src = inspect.getsource(server.create_anthropic_message)
        # The helper call must appear at least twice: once for the
        # streaming branch (StreamingResponse headers dict) and once for
        # the non-streaming branch (Response headers dict).
        occurrences = src.count("_build_thinking_budget_headers")
        assert occurrences >= 2, (
            f"create_anthropic_message regression: _build_thinking_budget_headers "
            f"appears only {occurrences} time(s) — expected >=2 (streaming + "
            f"non-streaming branches). Clients lose budget-enforcement signal."
        )

    def test_streaming_header_distinguishes_simple_engine(self):
        """DCR Wave-2 CRITICAL O-1: SimpleEngine ignores budgets but
        carries a reasoning_parser. _streaming_header_value must return
        'false' when engine_supports_budget=False regardless of parser
        presence, to avoid lying to streaming clients in simple mode."""
        from vllm_mlx.server import _streaming_header_value

        # Simple mode with a parser set: pre-fix this returned "true" (lied).
        assert (
            _streaming_header_value(
                is_mllm=False,
                reasoning_parser=object(),
                engine_supports_budget=False,
            )
            == "false"
        )
        # BatchedEngine text path with parser: "true" (honest).
        assert (
            _streaming_header_value(
                is_mllm=False,
                reasoning_parser=object(),
                engine_supports_budget=True,
            )
            == "true"
        )

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

        proc, reason = _attach_thinking_budget_processor(
            tokenizer=_FakeTok(),
            reasoning_parser=_FakeParser(),
            budget=512,
            message=None,
            prompt_token_ids=None,
        )
        assert isinstance(proc, ThinkingTokenBudgetLogitsProcessor)
        assert reason is None

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

    def test_thinking_message_token_cap_drops_oversize_message(self):
        """DCR Wave-3: the defense-in-depth token-count cap on
        thinking_budget_message must be enforced. If someone deletes
        the `len(message_ids) > _THINKING_MESSAGE_MAX_TOKENS` check, a
        malicious client can force the decode hot path to emit a huge
        forced sequence — cheap DoS.

        Fake tokenizer returns a list longer than the cap for the message
        text; assert the processor is built but the force sequence does
        NOT include message tokens (only end_token_ids)."""
        from vllm_mlx.logits_processors import ThinkingTokenBudgetLogitsProcessor
        from vllm_mlx.scheduler import (
            _THINKING_MESSAGE_MAX_TOKENS,
            _attach_thinking_budget_processor,
        )

        class _OversizeTok:
            def encode(self, text, add_special_tokens=False):
                if text == "<think>":
                    return [100]
                if text == "</think>":
                    return [200]
                # Message text: emit a token list larger than the cap.
                return list(range(1000, 1000 + _THINKING_MESSAGE_MAX_TOKENS + 50))

        class _FakeParser:
            start_token = "<think>"
            end_tokens = ["</think>"]

        proc, reason = _attach_thinking_budget_processor(
            tokenizer=_OversizeTok(),
            reasoning_parser=_FakeParser(),
            budget=512,
            message="hypothetical DoS payload",
            prompt_token_ids=None,
        )
        assert isinstance(proc, ThinkingTokenBudgetLogitsProcessor)
        assert reason is None
        # Force sequence must be end_token_ids only — the oversized
        # message was dropped. If the cap check regresses, the sequence
        # would be 1000..1549+[200] = 551 tokens.
        assert proc._force_sequence == [200], (
            f"Wave-3 regression: oversized thinking_budget_message was not "
            f"dropped by the _THINKING_MESSAGE_MAX_TOKENS cap. "
            f"Force sequence has {len(proc._force_sequence)} tokens; expected 1."
        )

    def test_merge_outputs_prefers_non_none(self):
        """DCR Wave-1 CRITICAL-3: an abort-path RequestOutput may carry
        thinking_budget_applied=None (request already popped from
        scheduler tracking). The merge rule must prefer the non-None
        value so a prior mid-stream True is not silently overwritten."""
        from vllm_mlx.output_collector import RequestOutputCollector
        from vllm_mlx.request import RequestOutput

        coll = RequestOutputCollector()
        # existing=True (happy path) + new=None (abort) → True preserved.
        existing_true = RequestOutput(request_id="x", thinking_budget_applied=True)
        new_none = RequestOutput(request_id="x", thinking_budget_applied=None)
        assert (
            coll._merge_outputs(existing_true, new_none).thinking_budget_applied is True
        )
        # Symmetric: existing=None + new=True → True wins.
        existing_none = RequestOutput(request_id="x", thinking_budget_applied=None)
        new_true = RequestOutput(request_id="x", thinking_budget_applied=True)
        assert (
            coll._merge_outputs(existing_none, new_true).thinking_budget_applied is True
        )

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
