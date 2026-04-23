# Changelog

Fork-local changes on top of [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx). Entries are grouped by feature area and sorted newest-first within each release bucket.

The `Unreleased` section captures work that has merged to `main` but isn't tagged yet.

Format: entries cite the PR number + a one-line summary. See the PR body for the full context. See [`UPSTREAM_PIN.md`](UPSTREAM_PIN.md) for the invariants that must survive an upstream rebase.

---

## Unreleased

### Qwen3.x heterogeneous 50ms-staggered prefill deadlock — CLOSED (H1)

- **H1 root cause closed (2026-04-23, commit 312de2f).**
  `Scheduler._schedule_waiting` now passes
  `logits_processors=[[per_row]]` with a (possibly empty) per-row
  list instead of omitting the kwarg when the request has no
  processor.  Omitting caused mlx-lm's default to slot `None` into
  the per-row position, which crashed `GenerationBatch._step()` on
  `for p in None` when two heterogeneous requests' batches were
  merged.  See `UPSTREAM_PIN.md` invariant #18 for the full
  contract.
- **Defense in depth:** `CACHE_CORRUPTION_PATTERNS` widened to
  include `"'NoneType' object is not iterable"` alongside the
  existing `"not subscriptable"` pattern.  `engine_core._engine_loop`'s
  generic-exception catch is now bounded — 10 consecutive identical
  errors aborts running requests with `finish_reason=error` instead
  of retrying forever.
- **Trace subsystem (commits f58aae5 + f1b3158) used to localize.**
  `VLLM_MLX_SCHEDULER_TRACE=1` env-gates structured JSON emits at
  the scheduler ↔ BatchGenerator boundary on both Qwen and Gemma
  paths, plus a shape-aware pass-through shim over mlx-lm's
  BatchGenerator that never `repr()`s output items (avoids Metal
  sync).  Queue-handler-backed so trace I/O never blocks the
  scheduler thread.
- **Regression guards:**
  - `tests/test_scheduler_heterogeneous_logits_processors.py` — unit
    test against real mlx-lm 0.31.3 at the BatchGenerator API level.
  - `tests/test_qwen3_concurrent_heavy_payload.py::test_staggered_pair_50ms_current_failure_mode`
    flipped from xfail to hard regression guard.
- **Verification:**
  - Qwen3.6-35B-A3B: 0/90 HANG across 3 × 30-pair bursts.
  - Gemma-4-26b-a4b-it-4bit: 0/15 HANG (non-regression control).
  - Pre-fix baseline (from the same instrumented prod): 6/10 HANG,
    `stall_streak_max=1797` on the deadlocked request.

### Qwen3.x hybrid-cache concurrent-prefill fix (mlx-lm#1169 + #1177) — PARTIAL

- **#31** `fix(mlx-lm)`: pin to mlx-lm 0.31.3 (git SHA `3cd9a52d`) for the
  ArraysCache.extend concurrent-prefill fix. **Partial fix** — closes the
  *homogeneous* concurrent-pair case; Claude Code's real-world
  *heterogeneous* pair (heavy sonnet + light haiku at the same model)
  still deadlocks on Qwen3.5/3.6-35B-A3B and is tracked under refined
  hypothesis H1 (CB scheduler mispack of mixed-shape prefill batches)
  in `docs/testing/2026-04-21-qwen3-35b-a3b-concurrent-heavy-payload-deadlock.md`.
  The heterogeneous case has an xfail integration test
  (`tests/test_qwen3_concurrent_heavy_payload.py::test_heterogeneous_pair_current_failure_mode`)
  which flips to a hard regression guard when the fix lands.
- **Fix concurrent heavy-payload deadlock / degenerate-response on
  Qwen3.5-35B-A3B, Qwen3.6-35B-A3B, Qwen3-Next and other hybrid-cache
  models.** Bug doc: [`docs/testing/2026-04-21-qwen3-35b-a3b-concurrent-heavy-payload-deadlock.md`](docs/testing/2026-04-21-qwen3-35b-a3b-concurrent-heavy-payload-deadlock.md).
  Root cause lives entirely in mlx-lm, not this fork: `ArraysCache.extend`
  silently dropped the batch dimension when one side had a None slot
  (fresh, never-used per-sequence cache). Under `--continuous-batching`,
  two concurrent prefills of large prompts would land in the same
  `BatchGenerator`, the `extend` call would leave batch dimensions
  mismatched across cache slots, and the downstream generation path
  would see one sequence's conv/SSM state fed into the other's attention
  layers — producing either degenerate zero-token responses (8-concurrent
  unique-prefix repro) or indefinite stalls after `message_start`
  (bug doc's original symptom via arena-proxied requests). Gemma 4 on CB
  was unaffected because the Gemma-4 path routes through the MLLM
  scheduler with `MLLMCacheManager`, not `ArraysCache`.
- **Fix path:** port by pinning `mlx-lm` to `3cd9a52df261edbcfd74ba8f72ca345380bb1bbd`
  on `ml-explore/mlx-lm` main (first SHA containing both #1169 and #1177).
  `pyproject.toml` uses `[tool.uv] override-dependencies` because the
  `[audio]` extra's `mlx-audio>=0.4.1` transitively pins `mlx-lm==0.31.1`.
  When mlx-lm 0.31.3 reaches PyPI and mlx-audio releases a compatible
  version, drop the git URL and the override entry.
- **Regression coverage:** `tests/test_mlx_lm_arrays_cache_concurrent.py`
  (7 behavioral sentinels covering BOTH regression vectors separately —
  5 of 7 fail on mlx-lm 0.31.2 (catches vector #1169 and #1177 together),
  2 of 7 fail on post-#1169-pre-#1177 intermediate SHAs
  (catches vector #1177 alone — previously undetected by version-string
  checks since mlx-lm bumped `__version__` to "0.31.3" before either
  fix landed). Added to `.github/workflows/ci.yml` `test-apple-silicon`
  job so the sentinel actually runs on PRs.
  `tests/test_qwen3_concurrent_heavy_payload.py` (live-server
  integration test gated by `QWEN3_CONCURRENT_TEST_URL` +
  `QWEN3_CONCURRENT_TEST_MODEL` env vars; injects nonces into BOTH
  system and user blocks to force concurrent prefill).
- **UPSTREAM_PIN.md invariant 17** added to pin the dependency floor
  and document the drop-back path for when a PyPI release ships.
- **Verification (on this fork's main at 1f333f1):** isolated test
  server on `:19001` with pre-fix 0.31.2 produced
  `content_block_delta_count=0, completion=0 tokens` on 8 concurrent
  unique-prefix heavy payloads; post-fix 0.31.3 produced
  `content_block_delta_count=2, completion=2 tokens` per request.
  Gemma-4-26b-a4b-it-4bit non-regression passed on 0.31.3.

### Anthropic streaming thinking-signature fix (PR #34)

- **Fix Claude Code silently dropping text after thinking blocks on Qwen3.5/3.6 streaming.** Bug doc: [`docs/testing/2026-04-23-qwen3-streaming-thinking-missing-signature-silent-text-drop.md`](docs/testing/2026-04-23-qwen3-streaming-thinking-missing-signature-silent-text-drop.md). Non-streaming path was already correct (PR #14); this PR extends the fix to streaming — every thinking content block now carries a `signature_delta` event before `content_block_stop`, both on mid-stream transitions and at final-close.
- Both paths share `vllm_mlx.api.anthropic_adapter.compute_thinking_signature()` with a byte-parity guard test (`tests/test_anthropic_streaming_thinking_signature.py::test_streaming_signature_matches_non_streaming_byte_for_byte`).
- Shared close-event emitter `vllm_mlx.server._emit_block_close` dispatches on block type; unknown types raise `ValueError` so a future signable block type cannot silently drop the contract.
- Adds `thinking_signature_emitted_total` counter (local `_Counter` pattern; `/metrics` endpoint wiring is a follow-up per metrics.py docstring).
- `[streaming-signature]` INFO log lines include `msg_id` for on-call correlation (both mid-stream and final-close empty-buffer paths).
- UPSTREAM_PIN.md **invariant 13 extended** to cover streaming + byte-parity. Rebase-sentinel tests in `tests/test_streaming_signature_rebase_sentinel.py` fire loudly if a future rebase drops the dispatcher or its invocation.
- **Explicitly rejected: `--anthropic-emit-thinking-signatures` feature flag.** Speculative risk to hypothetical non-Claude-Code downstream SDKs pinned to pre-extended-thinking spec versions. `signature_delta` only emits when the model produces thinking content, which only happens when the client opts into extended thinking (via `thinking.type: "adaptive"` or a chat-template `enable_thinking=True`). Non-opt-in clients never see thinking blocks and therefore never see `signature_delta`. Rollback via `git revert` is the accepted safety net.

### KV cache LCP contamination fix (issue #29)

- **Fix `--continuous-batching` producing garbage on request 2+** ([#29](https://github.com/jackneil/vllm-mlx-patched/issues/29)) — sessions that loaded a persisted prefix cache could return degenerate repetitive output (`ongo`, `diễn`, ...) for the second and later requests. Root cause was twofold:
  1. `_trim_cache_offset` and `_dequantize_cache` returned caches that shrank `.offset` but reused the over-sized source `keys`/`values` arrays. Attention paths that read `cache.state` directly (Gemma 4 KV-shared layers upstream, Qwen3 kickoff on supersequence matches in our fork) saw stale tokens from the previous owner of the buffer.
  2. Disk-persisted entries from older fork versions had no consistency gate at load time.
- Ports `waybarrios/vllm-mlx#385` (`9630c5d`, array-slice fix) and `waybarrios/vllm-mlx#365` (`01261c1`, persist-format v3 + model fingerprint). Existing caches at `~/.cache/vllm-mlx/prefix_cache/` auto-discarded and rebuilt on the next server start.
- Regression coverage: `tests/test_memory_cache_mlx.py` (MLX-backed slice tests), `tests/test_memory_cache.py` (fingerprint helper + load gate), `tests/test_issue_29_end_to_end.py` (end-to-end lockdown).

### Qwen3 first-turn runaway mitigation

- **Qwen3 first-turn runaway — three-layer defense shipped.** Qwen3.x models on first-turn-with-tools requests can enter an "interleaved thinking" trap — generating reasoning indefinitely without emitting `</think>`. Mitigations:
  - **Layer 1** (surgical auto-disable, default-on): when serving `--reasoning-parser qwen3` AND the request has `tools: [...]` AND no prior assistant message, the adapter injects `chat_template_kwargs={"enable_thinking": False}`. Uses Qwen's own chat-template branch → model skips think mode. Client-explicit `enable_thinking` or `thinking.type` always wins. Opt-out via `--disable-qwen3-first-turn-no-think`.
  - **Layer 2** (opt-in `--max-thinking-token-budget N` CLI flag): operator-tunable ceiling on resolver output. Clamps any resolved budget > N down to N at all 4 resolve-sites. Respects client `budget=0` (never raises). Recommended `2048` for Qwen3.x + agentic clients.
  - **Layer 3** (existing `--streaming-max-seconds 260`): remains as third-line wall-clock backstop.
- **New response headers**: `x-thinking-budget-ceiling`, `x-thinking-budget-clamped-to`, `x-thinking-budget-clamp-skipped`, `x-thinking-qwen3-auto-disabled`.
- **New counters**: `thinking_budget_clamp_fired_total`, `qwen3_first_turn_no_think_applied_total`.
- **Changed**: `x-thinking-budget-resolved` now reflects the POST-clamp value when Layer 2 fires. `x-thinking-budget-max-tokens-floor` absent when Layer 2 clamped (pre-clamp floor is stale).
- **Known limitation**: multi-turn in-context-learning back-fire. When Layer 1 fires on turn 1, Qwen3 may produce shallow thinking on turn 2+ even with explicit client thinking request. Root fix (synthesize `<think></think>` in prior-turn reconstruction) deferred. Skip-marked regression test documents the vector.
- Investigation: [docs/testing/2026-04-19-qwen36-first-turn-runaway-under-claude-code-payload.md](docs/testing/2026-04-19-qwen36-first-turn-runaway-under-claude-code-payload.md). Upstream refs: [ml-explore/mlx#3267](https://github.com/ml-explore/mlx/issues/3267), [ggml-org/llama.cpp#21118](https://github.com/ggml-org/llama.cpp/issues/21118), [vllm-project/vllm#39103](https://github.com/vllm-project/vllm/issues/39103).

### Thinking budget — API correctness and observability

- **#24** `fix(effort)`: `thinking.type="adaptive"` + explicit effort ceiling. When Claude Code (and other clients) send BOTH `thinking.type=adaptive` AND `output_config.effort=high` (or `reasoning_effort=high`), the resolver previously discarded the effort signal and returned `budget=None` — because `adaptive` short-circuited precedence. On Claude 4.5+ this works (self-regulation); on Qwen3.6 and other open-weight models it produced a first-turn runaway (thinking indefinitely until the 260s streaming cap fired). Fix: when `adaptive` is accompanied by an explicit effort signal, fall through to the effort branch so the model gets a ceiling. Closes the Qwen3.6 first-turn runaway documented in `docs/testing/2026-04-19-qwen36-first-turn-runaway-under-claude-code-payload.md` (H4).
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
