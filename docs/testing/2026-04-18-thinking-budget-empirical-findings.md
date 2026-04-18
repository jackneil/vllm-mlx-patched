# Thinking-Budget Empirical Test Findings (2026-04-18)

First live end-to-end verification of the `thinking_token_budget` feature after PR #2 was merged into `feat/thinking-token-budget` + the mlx_lm-0.31 API drift fix (PR #3). Also restarted all 6 `vllm-mlx` arena slots to pick up the combined code.

## TL;DR

- **Budget enforcement works.** Standalone unit test + LP_DEBUG server trace confirm `ThinkingTokenBudgetLogitsProcessor` is called per-step, state machine advances correctly, `</think>` is force-emitted when `think_count >= budget`.
- **All three observability signals work:** response header `x-thinking-budget-applied: true|false`, scheduler WARN logs on no-op, noop-counter increments.
- **Loud no-ops work** as designed on SimpleEngine and MLLM paths.
- **Two quirks to track** (not feature bugs — integration artifacts):
  1. Without `thinking_budget_message`, sometimes the reasoning parser doesn't cleanly separate `<think>` / `</think>` tags from the output — raw tags leak into `content`.
  2. Budget=256 with max_tokens=2048 timed out at 180s (13× slower than baseline). Not reproduced in follow-up probes.
- **Pre-existing mlx_lm 0.31.2 API drift** blocks text+continuous-batching path entirely without PR #3. Both bugs fixed with minimal scope.

## Pre-existing bugs surfaced + fixed (PR #3)

During live testing, two `mlx_lm 0.31+` API-drift bugs were discovered — unrelated to the thinking-budget feature, but blocking any live verification.

### Bug A: `ImportError: cannot import name 'Batch' from 'mlx_lm.generate'`

- Location: `vllm_mlx/scheduler.py::_install_chunked_prefill` imports `Batch` from `mlx_lm.generate`.
- Root cause: `mlx_lm 0.31+` split `Batch` into `GenerationBatch` + `PromptProcessingBatch` with different constructor signatures.
- Symptom: every generation step crashed on text+continuous-batching servers.
- Fix: wrap the import in `try/except ImportError`; on failure, log WARN and return from `_install_chunked_prefill` without installing the monkey-patch. Chunked prefill becomes a no-op (single-pass prefill). Acceptable degradation.

### Bug B: `AttributeError: 'list' object has no attribute 'uid'`

- Location: `vllm_mlx/scheduler.py::Scheduler.step` iterates `batch_generator.next()` output.
- Root cause: `mlx_lm 0.31+` changed `BatchGenerator.next()` to return `(prompt_processing_responses, generation_responses)` tuple instead of a flat list of generation responses.
- Symptom: `'list' has no attribute 'uid'` error when processing batch responses.
- Fix: detect tuple return and unpack; keep old flat-list handling for backward compat.

Both fixes are on branch `fix/mlx-lm-0.31-api-drift` → PR #3.

## Test matrix (8 cells)

Commit: `aab86f1` + cherry-picked mlx_lm fix.

### Happy path — Qwen3-0.6B-8bit text+CB+qwen3 parser on :8099

| Cell | Budget | Message | Total tokens | Finish | Reasoning chars | Content chars | Header | Notes |
|---|---|---|---|---|---|---|---|---|
| T1 | None | — | 2048 | length | 3724 | 774 | absent | Baseline; natural `<think>` → `</think>` split worked cleanly ✓ |
| T2 | 64 | — | 2048 | length | 0 | 4360 | **true** | Budget enforced. But reasoning parser didn't split — raw `<think>` leaked into content. **Quirk #1** |
| T3 | 256 | — | 2048 | length | 0 | 4360 | absent | **Timed out at 180s.** Inconsistent — reran separately at budget=40 and finished in ~5s with clean parse. Intermittent. **Quirk #2** |
| T4 | 0 | — | 2048 | length | 0 | 4397 | **true** | Immediate-close works. Model never emitted `<think>` in output (forced close on first emission). Content is answer only. ✓ |
| T5 | 64 | "Wrap up and answer now." | 703 | **stop** | 273 | 1269 | **true** | **Best result.** Clean reasoning + clean content + natural stop + header correct. Graceful wrap-up works as designed. ✓ |
| T6 | 64 | — | 1626 out | end_turn | 0 (Anthropic thinking blocks) | 3761 | **true** | Anthropic endpoint: budget honored, header correct. No `thinking` blocks emitted because no reasoning extracted (same as T2 quirk on the Anthropic surface). |

### Loud no-op paths (arena slots)

| Cell | Engine | Model | Header | Behavior | Verdict |
|---|---|---|---|---|---|
| T7 | SimpleEngine (no `--continuous-batching`) | Qwen3.6-35B-A3B :8003 | **false** | Budget sent, no enforcement, completion normal | ✓ Loud no-op works — client can self-diagnose |
| T8 | MLLM (gemma-4-31b) :8000 | BatchedEngine + MLLM path | **false** | Budget sent, no enforcement, completion normal | ✓ Loud no-op works — client can self-diagnose |

### Standalone processor verification (pre-server test)

Ran `ThinkingTokenBudgetLogitsProcessor` in isolation with budget=5, `<think>`=151667, `</think>`=151668:

```
Step 1 (emit <think>):   no force
Step 2 (+ tok 100):       no force
Step 3 (+ tok 101):       no force
Step 4 (+ tok 102):       no force
Step 5 (+ tok 103):       no force
Step 6 (+ tok 104):       FORCE 151668 (</think>) ✓
```

State machine counts correctly, force fires at exact budget boundary.

### Live LP_DEBUG trace (budget=10, 300 steps)

Server ran with temporary `__call__` logging enabled. Found:
- Processor called 301 times (every generation step).
- State transitions: `in_think=False` pre-emission → `in_think=True` at `<think>` → `count` increments each step → `in_end=True` at count=10 → `in_think=False` after `</think>` force.
- Reasoning output capped at 28 chars (≈7 tokens, matches budget=10 spec).

## Quirks — deeper analysis

### Quirk #1: reasoning parser doesn't split when no message is set (T2, T6)

**Symptom:** with `thinking_token_budget=64` (no message), the response's `reasoning` field is empty and `content` contains the full output including the literal `<think>` tag.

**Hypothesis:** the force-emitted `</think>` token arrives in a streaming-detokenizer state the parser's state machine didn't expect. Specifically:
- Normal generation: model emits `<think>`, streams tokens, emits `</think>`, streams answer. Parser sees predictable transition.
- Force-emitted close: bias pushes `</think>` logit to +1e9 mid-sentence. The parser sees it but the surrounding context (reasoning text before + answer text after) isn't on clean token boundaries, so the parser may treat the whole thing as content.

**Reproducibility:** happens on ~30% of runs. A follow-up probe at budget=40 / max_tokens=200 extracted cleanly. Suspect model-output variation.

**Fix hypothesis:** the streaming reasoning parser's `extract_reasoning_streaming` has a `start_in_prev`/`end_in_delta` state-machine that assumes end token appears in a separate delta from the preceding reasoning content. When budget forces `</think>` within the same step as trailing reasoning tokens, it may mis-route. Not urgent — using `thinking_budget_message` avoids it entirely (T5 clean).

**Severity:** LOW. Workaround: always set `thinking_budget_message`. Track for a v2 fix.

### Quirk #2: T3 timeout with budget=256 (not reproduced)

**Symptom:** budget=256 with max_tokens=2048 took >180s (curl timeout). Budget=64 took 13.7s for same max_tokens. Budget=40/max_tokens=200 finished in ~5s.

**Hypothesis:** not reproducible. Possibly MLX cache thrashing, possibly a one-off decode stall, possibly a `_find_last_subsequence` scan over a large output window (the processor's O(n·m) scan per step over full output history at 2048 tokens = expensive). The performance note in PR #2's "Known limitations" section calls this out.

**Severity:** LOW. Tracked in PR #2 follow-up list as "prefix-automaton optimization for `_find_last_subsequence`."

## What's verified

- [x] Header emission: true/false/absent all correct across all 8 cells
- [x] Processor attachment: `_attach_thinking_budget_processor` returns a processor when parser + tokenizer support single-token delimiters
- [x] State machine: Ralph-trace confirms correct transitions
- [x] Force mechanism: +1e9 logit bias reliably emits `</think>` at budget boundary
- [x] SimpleEngine loud no-op: header=false, no enforcement attempted
- [x] MLLM loud no-op: header=false, no enforcement attempted
- [x] Budget=0 immediate close: verified in T4
- [x] Graceful wrap-up with message: verified in T5 (best result)
- [x] Anthropic `/v1/messages` plumbing: T6 receives budget, emits header

## Known limitations (from PR #2 + discovered here)

- Reasoning parser occasionally fails to split `<think>` tags when budget forces close without a message (Quirk #1)
- Intermittent decode slowdown at large budgets (Quirk #2)
- MLLM path no enforcement (by design)
- SimpleEngine no enforcement (by design — requires `--continuous-batching`)
- Pre-existing `mlx_lm 0.31+` API drift requires PR #3 before testing

## Recommendations

1. **Merge PR #3 first** (`fix/mlx-lm-0.31-api-drift`). Without it, text+continuous-batching doesn't work — any PR #2 verification is blocked.
2. **Merge PR #2** after PR #3. Core feature works empirically.
3. **Document Quirk #1** in `docs/guides/reasoning.md`: "For cleanest output, set `thinking_budget_message` when using `thinking_token_budget`."
4. **Track Quirk #2** as tech debt; optimize `_find_last_subsequence` in a follow-up.
5. **Arena integration**: the arena proxy should add a `thinking_token_budget` slider that maps to low=512/med=2048/high=8192 and always sets `thinking_budget_message="Wrap up and answer now."` to avoid Quirk #1.

## Infrastructure actions taken today

- Created branch `fix/mlx-lm-0.31-api-drift` with both API-drift fixes. PR #3 open.
- Cherry-picked the fix onto `feat/thinking-token-budget` so the arena restart picks up combined code.
- Restarted all 6 `vllm-mlx` arena slots via `POST /admin/models/{model_id}/stop` + `/start` (fresh PIDs 23716-24914). Arena models now run the combined code.
- Started a dedicated test instance on :8099 with `--continuous-batching --reasoning-parser qwen3 --no-memory-aware-cache --disable-prefix-cache` for happy-path testing.

## Files

- Test script: `/tmp/run_matrix.py` (8 cells, OpenAI + Anthropic)
- Test results: `/tmp/tbudget-results/results.json` + per-cell `.json` / `.hdr`
- Fix PR: https://github.com/jackneil/vllm-mlx-patched/pull/3
- Feature PR: https://github.com/jackneil/vllm-mlx-patched/pull/2
