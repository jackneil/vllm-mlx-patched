# Changelog

Fork-local changes on top of [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx). Entries are grouped by feature area and sorted newest-first within each release bucket.

The `Unreleased` section captures work that has merged to `main` but isn't tagged yet.

Format: entries cite the PR number + a one-line summary. See the PR body for the full context. See [`UPSTREAM_PIN.md`](UPSTREAM_PIN.md) for the invariants that must survive an upstream rebase.

---

## Unreleased

### Thinking budget — API correctness and observability

- **#22** `fix(anthropic)`: close the Qwen3.x "interleaved thinking" streaming trap. When the Anthropic conversation history carries `type: "thinking"` content blocks on prior assistant turns, the adapter now round-trips them back into the assistant's OpenAI content as inline `<think>…</think>` (Qwen-gated). Qwen3.x models were interpreting thinking-less history as "we're mid tool-call loop; don't close </think> until final answer" and generating reasoning indefinitely — surfacing as a 5-minute Claude Code hang on multi-turn + tool requests. Also adds `--streaming-max-seconds` CLI flag (default 260s) — server-side wall-clock cap that terminates streams with clean `finish_reason=length`/`stop_reason=max_tokens` + tail frames instead of running until the client aborts. **Security**: client-supplied `thinking` blocks are sanitized against prompt-injection via chat-template markers (`<think>`/`</think>`, `<|im_end|>`/`<|endoftext|>`, `<|channel>`/`<|channel|>…<|message|>`, Llama `<|eot_id|>`/`<|start_header_id|>`, Gemma `<start_of_turn>`/`<end_of_turn>`) plus Bidi/isolate control stripping — blocks matching any marker are dropped with WARN. **Observability**: new `streaming_cap_fired_total` metric + cap-fire WARNs carry `cap=prologue|wait_for` path discriminator, `msg_id`/`response_id`, and model name for alert correlation. **Safety**: cap implementation uses explicit task-cancellation (not `break`-in-`async for`) to avoid Metal/SIGABRT risk from implicit `aclose()`; bounded 2s cleanup wait on cancellation; defensive `finally` cancel for outer-task-cancelled-during-shield orphan case. Cross-refs: llama.cpp#21118, litellm#19849. Shipped after 5 waves of /dcr recursive review.
- **#20** `fix(anthropic)`: wire `extract_reasoning` + `thinking_tokens` on non-streaming `/v1/messages`. Pre-fix, the non-streaming Anthropic path never split reasoning from content — the `type:"thinking"` content block (from #13) and its `signature` field (from #14) were dead code. Also adds `AnthropicUsage.thinking_tokens` (vllm-mlx extension) so operators can measure thinking depth without re-tokenizing.
- **#14** `fix(anthropic)`: required `signature` field on thinking content blocks. Schema-conformant with the Anthropic SDK's `ThinkingBlock` model. Populated deterministically with `"vllm-mlx:" + sha256(thinking_text)[:32]`.

### Response headers and diagnostics

- **#15** `feat(server)`: `x-thinking-budget-noop-reason` header (5 enumerated values: `parser_not_configured`, `tokenizer_encode_failed`, `multi_token_delimiter`, `mllm_path`, `simple_engine`) + WARN when `max_tokens < max_tokens_floor`. The `noop-reason` header closes a diagnosability gap: operators no longer have to guess why `applied=false`.

### Cache safety

- **#19** `feat(cache)`: wire production `acquire(request_id)` / `release(request_id)` on `MemoryAwarePrefixCache` and `PrefixCacheManager`. Before this, the #16 guards existed but never fired under real load — only `_mark_in_use_for_test()` bumped the counter. Now all three clear-path tiers refuse consistently when requests are in flight.
- **#16** `fix(cache)`: `DELETE /v1/cache` returns HTTP 409 when any tier refuses. Three cache tiers (`memory_aware_cache`, `block_aware_cache`, `prefix_cache`) gained matching in-flight guards. Also fixed a pre-clear ordering bug in `BlockAwarePrefixCache` that could leave partial corruption on delegate refusal.

### Resolver hardening

- **#17** `fix(effort)`: resolver hardening. Exports `ALLOWED_EFFORT_LEVELS` as single source of truth for validators. Pydantic `field_validator` on `reasoning_effort`, `output_config.effort`, and `thinking.budget_tokens` rejects malformed input at HTTP 422 instead of silently falling through to DEFAULT. WARNs on non-empty `thinking` dict missing `type`. Caps the `max` effort floor at 32768 with prompt headroom (down from 131072 on 1M-ctx models).

### Test coverage

- **#18** `test`: adapter scheduler-walking chain + cross-dialect parity matrix (M-9 fix). Parity test now requires every dialect to emit thinking > 0 — pre-fix `count == 0 or ratio_ok` could mask 2/4 dialects silently dropping to zero.

### Provider dialect unification

- **#13** `feat`: provider effort/budget unification. New `vllm_mlx/api/effort.py` resolver collapses 5 provider-dialect signals (top-level `thinking_token_budget`, Anthropic `thinking.type` disabled/enabled/adaptive, `output_config.effort`, OpenAI `reasoning_effort`) into one `ResolvedBudget`. Emits 3 new response headers (`x-thinking-budget-resolved`, `x-thinking-budget-source`, `x-thinking-budget-max-tokens-floor`). Extends `DELETE /v1/cache` + `GET /v1/cache/stats` to cover `MemoryAwarePrefixCache`. Non-streaming `/v1/messages` now emits `type:"thinking"` content blocks (completed by #20 which wired `extract_reasoning`).

### Sizing guardrail

- **#12** `feat(server)`: WARN when `thinking_token_budget >= max_tokens`. Catches the common truncation mistake where clients set `effort=high` but leave `max_tokens` at the SDK default.

### `mlx_lm` 0.31.x compatibility

- **#8** `fix`: `mlx_lm` 0.31.2+ `BatchGenerator.active_batch` drift (hard-cut). The `active_batch` attribute was removed; scheduler now walks `_prompt_batch` + `_generation_batch` via `_active_batches(bg)` helper. Includes AST sentinel so a rebase that reintroduces `active_batch` fails loudly. Adds `UPSTREAM_PIN.md` invariant #10 + one-shot breadcrumb. Fixes crash-every-step on Qwen3-0.6B observed after mlx_lm upgrade.
- **#7** `fix(thinking-budget)`: `tokens` arg is generated-only in `mlx_lm` 0.31+. Drops the `prompt_len` slice in the logits processor that was correct for 0.30.x but over-slices on 0.31+.
- **#3** `fix`: tolerate `mlx_lm` 0.31+ API drift (`Batch` split + `BatchGenerator.next` tuple). Chunked prefill becomes a no-op on unsupported versions.

### Reasoning parsers

- **#10** `fix(thinking-budget)`: `budget=0` enforcement with prompt-injected `<think>`. Qwen3.6's chat template pre-injects `<think>` into the assistant turn, which tripped a state-machine ordering bug in the logits processor — `_end_count` was consumed on the model's freely-sampled first token. Fix: `_end_count` management moved to `__call__`, advanced in lockstep with the force bias.
- **Q1 fix (55d25cf)** `fix(qwen3-parser)`: surface truncated `<think>` as reasoning, not content. When `max_tokens` cuts generation before `</think>`, the parser now returns `(reasoning_before_cut, None)` instead of `(None, content_with_<think>_tag)`.
- **#1** `fix`: Gemma 4 broken streaming text in Anthropic Messages API.

### Thinking token budget (feature)

- **#2** `feat`: `thinking_token_budget` for the text path — port of vLLM #20859. Per-request logits processor that force-biases the `</think>` token's logit to `+1e9` once the budget is hit. Works where start/end delimiters tokenize to single tokens (Qwen3, DeepSeek-R1). Multi-token delimiter families (Gemma 4 channel protocol, GPT-OSS Harmony) are a loud no-op.

### Documentation

- **#11** `docs`: answer-quality validation — OFF still correct on multi-step + CRT problems.
- **#9** `docs`: post-merge validation of `active_batch` drift fix.
- **#6** `test`: HTTP matrix for `thinking_token_budget` across model families.
- **#5** `docs`: MLLM extension playbook for `thinking_token_budget`.
- **#4** `docs`: mark Quirk #1 as fixed; correct `_find_last_subsequence` perf note.
- `docs`: add vllm-mlx vs upstream vLLM relationship + capabilities reference (b9e9343).

---

## Contributing to this file

When you merge a PR to `main`, add a one-line entry under `Unreleased` in the appropriate section. Keep it to the PR number + a summary of *what changed* and *why it matters* (not "what we did"). When we cut a release, move everything under `Unreleased` into a new `## X.Y.Z (date)` section.
