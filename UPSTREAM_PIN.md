# Upstream Pin for vllm-mlx-patched

This is a **hard fork** of vllm-mlx, not a patch overlay. The canonical upstream
is https://github.com/waybarrios/vllm-mlx (see `git remote -v`). This file
documents the upstream commit the patched fork is rebased against and the
arena-critical invariants that must survive a rebase.

## Current upstream pin

- **Upstream commit:** `b4fa03080ce18280ac9916d19a690a13aa45f904` (Merge pull request #232 from sjswerdloff/fix/streaming-tool-call-content-leak)
- **Rebased on:** `2026-04-11`
- **Rebased by:** Jack Neil

## Arena-critical invariants

These are things the `hank-llm-arena` arena's `models_process.py` depends on.
If a rebase breaks any of them, the arena's Gemma 4 or Qwen3 tool calling will
silently degrade to raw-text output.

1. **`@ToolParserManager.register_module("gemma4")`** must fire at import time in
   `vllm_mlx/tool_parsers/gemma4_tool_parser.py`. The registration must succeed
   before the CLI argparse runs `ToolParserManager.list_registered()`.

2. **`@ToolParserManager.register_module(["hermes", "nous", "qwen3_coder"])`** must
   register the `qwen3_coder` alias under HermesToolParser in
   `vllm_mlx/tool_parsers/hermes_tool_parser.py`. The arena's
   `_TOOL_PARSER_PATTERNS` expects this exact string.

3. **`ToolParserManager.list_registered()`** must return a list of strings that
   includes at minimum: `gemma4`, `qwen`, `qwen3`, `qwen3_coder`. Adding more
   parsers is fine; removing or renaming any of these breaks the arena.

4. **`--tool-call-parser` argparse action** in `vllm_mlx/cli.py` must accept any
   registered parser name as a valid choice. The argparse choices list must be
   driven dynamically by `ToolParserManager.list_registered()` (NOT hardcoded).
   Tests in `tests/test_cli_argparse.py` pin this.

### Added 2026-04-12: reasoning-parser streaming integration

5. **`ReasoningParser` base class exposes these properties** (`vllm_mlx/reasoning/base.py`):
   - `start_token: str` (default `"<think>"`)
   - `end_tokens: list[str]` (default `["</think>"]`)
   - `channel_strip_prefix: str | None` (default `None`)

   These are read verbatim by `_stream_anthropic_messages` in `vllm_mlx/server.py`
   behind a `hasattr` precondition to configure `StreamingThinkRouter`. If any
   property is renamed, the server logs an error and falls back to default
   router behavior. Pin: these three exact names with those exact defaults.
   `tests/test_reasoning_parser_properties.py::TestRebaseBreakageSentinel` fires
   in CI if any property name drifts.

6. **`Gemma4ReasoningParser` overrides** `end_tokens` as `["<channel|>", "<|channel>response"]`
   and `channel_strip_prefix` as `"thought\n"` (`vllm_mlx/reasoning/gemma4_parser.py`).
   Note: the legacy singular `end_token` property is RETAINED because
   `BaseThinkingReasoningParser` still declares it abstractmethod and uses it
   in `extract_reasoning_streaming`. Do not remove without untangling the parent.

7. **`StreamingThinkRouter.__init__` accepts** `start_token: str`,
   `end_tokens: Iterable[str]`, `channel_strip_prefix: str | None` kwargs with
   backward-compatible defaults (`vllm_mlx/api/utils.py`). Rejects prefix
   collisions in `end_tokens` at construction. Uses an integer counter
   `_strip_remaining` for channel-strip state (not buffer mutation). Pin:
   the kwarg names, the prefix-collision invariant, and the counter-based
   (non-buffer-mutating) design.

8. **`MLLMScheduler.add_request` precomputes `num_prompt_tokens`** via
   `self.processor.tokenizer.encode(prompt)` in `vllm_mlx/mllm_scheduler.py`.
   Continuous-batching MLLM requests flow through this path — without it,
   `message_delta.usage.input_tokens` leaks as 0 to Anthropic clients.
   `MLLM.stream_chat` has a matching precompute in `vllm_mlx/models/mllm.py`
   for the non-batched MLLM path. Both prefer a truthy chunk-reported value
   when present.

### Added 2026-04-17: thinking-budget plumbing

9. **Thinking-token-budget plumbing**
   (`vllm_mlx/request.py`, `vllm_mlx/api/models.py`,
   `vllm_mlx/logits_processors/thinking_budget.py`,
   `vllm_mlx/scheduler.py`, `vllm_mlx/server.py`,
   `vllm_mlx/metrics.py`):
   - `SamplingParams.thinking_token_budget` and `SamplingParams.thinking_budget_message`
     exist and round-trip through the Pydantic API.
   - `ChatCompletionRequest` accepts both fields at top level (via OpenAI
     SDK extra_body flattening) and a `chat_template_kwargs` passthrough
     for the nested parse path.
   - `vllm_mlx.logits_processors.ThinkingTokenBudgetLogitsProcessor`
     exists and its `__call__(tokens, logits) -> logits` signature
     matches the mlx_lm.generate.BatchGenerator processor contract.
   - `vllm_mlx.scheduler._attach_thinking_budget_processor` exists and
     returns a processor or None.
   - `vllm_mlx.scheduler._BG_INIT_PARAMS` includes `logits_processors`
     (the `_bg_kwargs` filter must not drop it).
   - `vllm_mlx.metrics.thinking_budget_noop_total` counter exists.
   - `RequestOutput.thinking_budget_applied: Optional[bool]` exists.
   - The chat-completion handler emits `x-thinking-budget-applied:
     true|false` as a response header whenever a request sets
     thinking_token_budget.

   `tests/test_thinking_budget_rebase_sentinel.py` fires in CI if any of
   these drift. The sentinel asserts BOTH existence (names exist) AND
   wiring (propagation through the RequestOutput → GenerationOutput →
   server-header chain; mlx_lm.BatchGenerator.insert accepts
   logits_processors). If a rebase brings an upstream-vLLM-compatible
   thinking budget implementation, reconcile naming and drop our port —
   but keep the sentinel.

   API promise: `thinking_token_budget` and `thinking_budget_message`
   match the vllm-project/vllm PR #20859 (merged 2026-03-24) public
   contract. Any client that uses vLLM upstream's field names works
   unchanged against vllm-mlx.

10. **mlx_lm >= 0.31.2 `BatchGenerator` split batches.**
    vllm_mlx requires `mlx_lm.generate.BatchGenerator` to expose
    `_prompt_batch` and `_generation_batch` (the 0.31.2+ structure).
    The pre-0.31.2 single `active_batch` attribute is no longer
    supported. The helper `vllm_mlx.scheduler._active_batches(bg)` is
    the canonical way to access active batches — direct references to
    `bg.active_batch`, `bg._prompt_batch`, or `bg._generation_batch`
    are forbidden outside the helper itself (and inside
    `_install_mtp` / `_install_chunked_prefill`, which remain as
    legacy code gated at their call sites; see `_ALLOWLIST_FUNCS`
    in `tests/test_mlx_lm_api_contract.py` for the exact allowlist).
    The sentinel at `tests/test_mlx_lm_api_contract.py` walks
    `scheduler.py`'s AST to enforce this.
    `Scheduler._create_batch_generator` fails fast at startup with
    `RuntimeError` if the installed mlx_lm does not expose
    `_generation_batch`, and logs
    `"[BatchGenerator] invariant #10 upheld"` on success.
    MTP (`_install_mtp`) is version-gated at the call site and emits
    a WARN on 0.31.2+ — speculative decoding is disabled there until
    the MTP port lands in a follow-up plan.

### Invariant 11: Effort resolver is the single source of truth

Any code path that reads `thinking_token_budget`, `thinking`, `output_config`, or `reasoning_effort` MUST call `vllm_mlx.api.effort.resolve_effort` — either directly (for OpenAI `/v1/chat/completions` endpoint) or via `anthropic_to_openai` (for `/v1/messages` endpoint, which returns a `(ChatCompletionRequest, ResolvedBudget)` tuple).

Adding a new provider dialect means extending `EffortSource` + the resolver. It does NOT mean adding parallel resolution logic in an adapter, the server handler, or the logits processor.

Test guard: `tests/test_effort_resolver.py` + `tests/test_anthropic_adapter_effort.py`.

### Invariant 12: Five-header contract on budget-resolved responses

Every chat-completion or messages response that went through the resolver MUST emit:
- `x-thinking-budget-applied` (`true` | `false`, absent when no budget requested)
- `x-thinking-budget-resolved` (int as string, or `"none"`)
- `x-thinking-budget-source` (one of the `EffortSource` enum values)
- `x-thinking-budget-max-tokens-floor` (int as string, absent when source=default or budget=0)
- `x-thinking-budget-noop-reason` (string; emitted ONLY when `applied=false`; one of `parser_not_configured` / `tokenizer_encode_failed` / `multi_token_delimiter` / `mllm_path` / `simple_engine`)

Downstream consumer assumption: `hank-llm-arena::proxy.py::_FORWARDED_HEADERS` forwards the `x-thinking-` prefix to clients. Arena's admin UI reads these headers to show a per-request "effort honored" indicator. The `noop-reason` header is what operators use to diagnose why the feature is a no-op (e.g., `simple_engine` means the server needs `--continuous-batching`).

Test guard: `tests/test_thinking_budget_headers.py` + matrix rows in `tests/test_thinking_budget_matrix.py`.

### Invariant 13: Non-streaming Anthropic thinking-block ordering and schema

`vllm_mlx.api.anthropic_adapter.openai_to_anthropic` MUST emit a `type: "thinking"` content block BEFORE the `type: "text"` content block when `choice.message.reasoning` is populated. The ordering matches Anthropic's public API and the streaming path (`server.py:1991-2032`).

Every `type: "thinking"` block MUST also be schema-conformant with the Anthropic SDK's `ThinkingBlock` model: it MUST include a `signature: str` field. We populate it with `"vllm-mlx:" + sha256(thinking_text).hexdigest()[:32]` — an opaque, deterministic hash so identical reasoning text produces identical signatures across requests. The prefix distinguishes our signatures from Anthropic's own server-signed values.

Test guards: `tests/test_anthropic_adapter_thinking_block.py` (ordering), `tests/test_anthropic_thinking_block_schema.py` (signature contract).

## Rebase checklist

When merging upstream changes into this fork:

1. [ ] `git fetch upstream && git log upstream/main..HEAD --oneline` to see fork-local commits
2. [ ] `git merge-base HEAD upstream/main` to confirm the current pin
3. [ ] Attempt rebase: `git rebase upstream/main`
4. [ ] Run `pytest tests/test_gemma4_tool_parser.py tests/test_tool_parsers.py tests/test_cli_argparse.py` — ALL must pass
5. [ ] Verify `gemma4` is in `--tool-call-parser` help:
       `vllm-mlx serve --help 2>&1 | grep -A2 tool-call-parser`
6. [ ] Update the "Current upstream pin" section above with the new SHA and date
7. [ ] Commit the UPSTREAM_PIN.md update as part of the rebase merge commit

## If a rebase breaks an invariant

Fall back to these recovery steps in order:

1. **Gemma 4 parser disappeared.** Check if upstream added their own Gemma 4
   parser under a different name (e.g., `gemma`, `google_gemma`). If so, update
   the arena's `_TOOL_PARSER_PATTERNS` to point at the new name AND add a
   compatibility alias `@ToolParserManager.register_module("gemma4")` back in a
   fork-local file.

2. **`list_registered()` was renamed.** Update both `vllm_mlx/cli.py` (the
   dynamic choices call) and `hank-llm-arena/tests/test_models_process.py` (the
   drift test).

3. **`--tool-call-parser` was renamed or removed.** Update the arena's
   `_tool_parser_flags` helper to use the new flag name. Add a compatibility
   shim in `vllm_mlx/cli.py` that accepts both names during a transition.

4. **`qwen3_coder` alias was moved out of HermesToolParser.** Update the
   `_TOOL_PARSER_PATTERNS` comment in `hank-llm-arena/models_process.py` to
   point at the new location.
