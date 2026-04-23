# Qwen3.5 / Qwen3.6 35B-A3B: concurrent heavy-payload requests deadlock

**Status:** **CLOSED 2026-04-23.** Root cause: `vllm_mlx.scheduler` omitted the `logits_processors=` kwarg on `BatchGenerator.insert()` for requests without a processor, letting mlx-lm slot `None` into the per-row slot. When a heterogeneous pair merged via `GenerationBatch.extend()`, the merged list had `[None, [proc]]` and `_step()` crashed on `for p in None` at `mlx_lm/generate.py:1346`. Fix commit: **312de2f** (caller-side contract + defense-in-depth + unit regression). Verification: 0/90 HANG on Qwen3.6-35B-A3B across three 30-pair bursts, 0/15 on Gemma cross-family. See `UPSTREAM_PIN.md` invariant #18 for the contract. See the **Closure (2026-04-23)** section at the end of this file for the full writeup.

**Historical status (kept for archaeology):** REOPENED 2026-04-22 evening. PR #31's mlx-lm 0.31.3 pin (`ArraysCache.extend` #1169 + #1177) is deployed in prod. Verified fix for the **simultaneous** pair case: 20/20 across a 10-pair same-millisecond burst pass. But Claude Code's real-world timing is **not** simultaneous — `hank-secure-llm` proxy logs show the haiku + sonnet requests arrive at the arena ~50ms apart. Under that stagger, the deadlock reproduces deterministically on Qwen3.5/3.6 via the same `llm.hank.ai` prod path that the burst test used. Gemma-4-26b is unaffected. The `ArraysCache.extend` fix closed one instance of the shared-state class but a second instance remains — likely a scheduler path that batches two requests differently when they arrive in separate CB steps.
**Reported:** 2026-04-21, refined 2026-04-22 after partial fix shipped. Surfaced via `hank-secure-llm`'s `model_qa` harness run.
**Impact:** High for Claude Code users on Qwen-35B-A3B. Claude Code always fires haiku + sonnet requests in parallel on every scenario; both deadlock → every request times out. Single-request access works fine, so a naive single-curl test from a developer laptop masks the bug completely.
**Scope:** `vllm_mlx/` — one of `StreamingThinkRouter` (`api/utils.py:240`), the continuous-batching scheduler `step()` (`scheduler.py:2588` or `mllm_scheduler.py:595`), or a sibling LCP-contamination path that PR #30's slice fix didn't cover.

## Symptom (refined 2026-04-22 evening, post-PR-#31 deployment)

**The trigger is not "heterogeneous payloads" or "concurrency" generically — it's specifically a ~50ms inter-arrival stagger between two requests to the same model.** Once the second request arrives AFTER the first has begun prefill (rather than in the same CB step), the model's KV/scheduler state decoheres and both streams hang at `message_start`.

Measured on `llm.hank.ai` prod (mlx-lm 0.31.3 pinned via PR #31):

| case | Qwen3.6-35B-A3B | Gemma-4-26b-a4b |
|---|---|---|
| 1 request in isolation | ✅ ~2s | ✅ |
| 2 simultaneous requests (same millisecond) | ✅ ~3.7s (**PR #31 fix**) | ✅ |
| **2 requests with 50ms stagger** (Claude Code's real behavior) | ❌ **both hang 30s+** | ✅ 0.4s + 0.9s |
| Same stagger but bodies are identical heavy | ✅ 3.7s | ✅ |

The 50ms-stagger case is deterministic across multiple runs (reproduced 3/3 today). It's also what Claude Code produces in normal use — `hank-secure-llm` proxy logs show haiku arriving at T+0ms and sonnet at T+50ms during every `claude --bare -p` invocation.

Behavior on failure:
- Both requests receive HTTP 200 with `text/event-stream` and TTFB ~10-20ms — upstream accepts and opens both streams.
- Haiku emits `message_start` + `content_block_start` (3 events, 0 `message_stop`); sonnet emits only `message_start` (1 event, 0 stops).
- Neither emits `content_block_delta`, `message_delta`, or `message_stop` within 30s. The `--streaming-max-seconds 260` cap eventually fires but not before client timeouts.

## Payload that triggers it

Stored at `docs/testing/claude-shape-heavy-payload.json` in this repo (~21 KB, ~4.5k tokens of system prompt + 3 tool definitions + `output_config.effort=high` + `thinking.type=adaptive` + `stream=true`). The shape matches Claude Code's real request.

The single-request **minimal** payload in `docs/testing/2026-04-19-qwen36-first-turn-runaway-under-claude-code-payload.md` (no system prompt, no tools) works fine on the current build — that earlier runaway bug is fixed. This is a distinct, concurrency-only issue.

## Reproduction — 50ms-staggered concurrent pair (current failure mode)

The precise trigger is two requests to the same model where the second arrives ~50ms AFTER the first (not simultaneously). Any fire-and-background shell script achieves this naturally:

```bash
# Real Claude Code captured payloads: cc-cap-01-qwen.json is haiku (tools=0,
# max_tokens=32000, ~1.5KB), cc-cap-02-qwen.json is sonnet (tools=3,
# max_tokens=32000, ~5.3KB), both with model=mlx-community/Qwen3.6-35B-A3B-4bit.
# Capture method: run `claude --bare -p` against a pass-through proxy that
# dumps request bodies (see capture-proxy.py in hank-secure-llm scratch).

( curl -sS -m 30 -N -o /tmp/h.sse -w "haiku t=%{time_total}s\n" \
    -X POST https://llm.hank.ai/v1/messages \
    -H "Authorization: Bearer $NEURAL_ARENA_KEY" -H "Content-Type: application/json" \
    -d @cc-cap-01-qwen.json ) &
sleep 0.05                                             # <-- THE TRIGGER
( curl -sS -m 30 -N -o /tmp/s.sse -w "sonnet t=%{time_total}s\n" \
    -X POST https://llm.hank.ai/v1/messages \
    -H "Authorization: Bearer $NEURAL_ARENA_KEY" -H "Content-Type: application/json" \
    -d @cc-cap-02-qwen.json ) &
wait

echo "haiku: $(grep -c '^event:' /tmp/h.sse) events, $(grep -c message_stop /tmp/h.sse) stops"
echo "sonnet: $(grep -c '^event:' /tmp/s.sse) events, $(grep -c message_stop /tmp/s.sse) stops"

# Observed 2026-04-22 evening, post-PR-#31 prod deploy (3/3 runs):
#   haiku t=30.00s   HTTP 200  → 3 events, 0 stops (message_start + content_block_start only)
#   sonnet t=30.00s  HTTP 200  → 1 event,  0 stops (message_start only)
```

Remove the `sleep 0.05` and both requests complete in ~1.8s with clean `message_stop`s — confirming this is not a general concurrency bug but specifically a same-millisecond-vs-staggered-arrival distinction.

Swap both `model` fields to `mlx-community/gemma-4-26b-a4b-it-4bit` — completes in ~0.4s + ~0.9s with 6 events + 2 `message_stop`s each. Gemma is the control and remains unaffected.

### Why the "simultaneous" tests pass but real traffic fails

The regression harness added with PR #31 (the 10-pair burst in `vllm-mlx-test`) uses `asyncio.gather` on pre-constructed requests, which typically dispatches within the same event-loop tick — all N requests land on the scheduler within <5ms. That pattern successfully hits the CB fast path where the entire pair is packed into a single prefill step and the `ArraysCache.extend` fix makes that step correct. Real-world callers (Claude Code, browser chat UIs, anyone firing the second request only after the first returns its HTTP headers) stagger by tens of milliseconds, which puts the second request into a LATER scheduler step — exactly the path PR #31 didn't exercise.

`hank-secure-llm` proxy logs consistently show the two Claude Code requests arriving 30-70ms apart at `llm.hank.ai`. Arena's `/v1/messages` proxy (`hank_llm_arena/proxy.py:proxy_v1`) is a dumb `httpx.AsyncClient.stream()` passthrough with no concurrency guards or queueing — it forwards the 50ms gap through to vllm-mlx untouched. So arena isn't introducing the gap or the bug; the gap exists in real client traffic and vllm-mlx's scheduler doesn't handle it on Qwen3.5/3.6.

## Hypotheses for investigation (refined 2026-04-22 evening)

Since the repro is now known to be **timing-staggered**, not shape-dependent, the bug is in the scheduler path that handles a second request arriving after the first has started generating. Candidates in descending likelihood:

**H1 (most likely). Continuous-batching `add_new_request` path corrupts KV state when a second sequence joins mid-step on Qwen3-MoE.** In `scheduler.py:2588` / `mllm_scheduler.py:595`'s `step()`, when a new request enters the batch while another is already prefilling, the slot allocation / KV-offset recompute for the existing sequence may be miscomputed under Qwen3's GQA head layout. The first sequence's generation state gets invalidated (explains haiku stalling after `content_block_start`) and the new sequence never gets a valid KV slot (explains sonnet stalling at `message_start`). PR #31's `ArraysCache.extend` fix handled the case where both sequences start together; a symmetric bug exists for the "one sequence already running" case.

**H2. `StreamingThinkRouter` carries state from the first request when the second joins.** `vllm_mlx/api/utils.py:240`. If the router instance is bound to a scheduler slot that gets reassigned when a new request joins mid-step, the second router could inherit or share state, leaving both streams unable to advance their `<think>` / `</think>` detection.

**H3. Prefix cache allocation race when writing the first request's prefill output AND the second request's prefill input simultaneously.** PR #30 sliced the single-request LCP fetch; PR #31 fixed `ArraysCache.extend` for coincident arrivals. A third race exists when the first request's cache slab is still being written while the second request is trying to allocate. MoE weights (Qwen3-MoE routes tokens to different experts per step) widen the window where the slab is being updated.

**H4. Qwen3-MoE expert-routing state leaks across staggered requests.** Qwen3-MoE has top-k expert selection per token. If the expert-routing module retains per-step state (e.g., a routing table or gate output cache) keyed by batch position rather than request id, the position reassignment on step 2 (when the second request joins) could corrupt routing for both.

The "both hang at `message_start` but one reaches `content_block_start`" asymmetry and the exact-50ms trigger strongly suggest H1 — the first request is partway through content generation when the second's join event corrupts its state, while the second never escapes the join.

Diagnostic ask: add logging at the scheduler step where a new request joins an in-progress batch, specifically dumping the KV offset table and slot assignments before and after the join. The log diff between the Gemma case (works) and Qwen3-MoE case (deadlocks) should localize the corruption.

## Observable signals while reproducing

During the hang, in `vllm-mlx` server logs look for:
- Two concurrent `[REQUEST] POST /v1/messages` arriving within ~100ms of each other
- Does either emit `[stream_outputs] <req-id> first token after <N>s`? If neither, scheduler or CB is stuck. If one emits and the other doesn't, it's router/cache state-sharing.
- Log the per-request `router_id`, `kv_cache_slab_id`, `scheduler_queue_position` — does one request reference another's id?
- Is the `260s --streaming-max-seconds` cap from PR #23 firing and closing the stream? If so, CB step() is live but never produces output for either.

## Success criteria for the fix

This file should be kept in-repo and a regression test added. The new test must cover **three arrival patterns**, not just the simultaneous one — PR #31 slipped through because only the simultaneous case was exercised:

1. **Staggered pair at 50ms** (primary — this is the open bug and matches real Claude Code traffic): fire the heavy request, `await asyncio.sleep(0.05)`, fire the light request, then `await asyncio.gather(h, s)`. Both must complete in <15s with `stop_reason: end_turn` and ≥1 `content_block_delta` each.
2. **Simultaneous pair** (regression guard for PR #31): `asyncio.gather(h, s)` with no sleep between. Must continue to complete in <15s each.
3. **Same tests against Gemma-4-26b-a4b** (cross-family non-regression): both patterns must complete — ensures any fix doesn't regress CB for other families.

Fail-mode assertion: when a test fails, dump per-request `events_received` and `last_event_type`. The failure signature for this bug is haiku at 3 events / `content_block_start` and sonnet at 1 event / `message_start` (or symmetric — whichever started first gets further). A test that just checks "total duration <15s" is insufficient — it must observe that each request independently reached a terminal event.

Diagnostic instrumentation to add alongside the test: log the CB scheduler's pre-step and post-step KV slot tables with request ids visible, so a regression produces an actionable log diff pointing at where the join-time corruption happens. This data is what the 50ms-stagger case needs that the simultaneous case didn't expose.

## Why this was hard to catch

Single-request curl succeeds. The previous runaway doc (`2026-04-19-qwen36-first-turn-runaway-under-claude-code-payload.md`) was also single-request and its minimal payload is now fixed — that checks out, PR #26 + PR #27 closed the tool+big-payload path for single requests. But the existing tests in `tests/test_issue_29_end_to_end.py` and `tests/test_memory_cache_mlx.py` don't exercise concurrency. Claude Code being the canary here is load-bearing: it's the only real-world caller that reliably fires parallel haiku + sonnet requests against the same model, which is exactly the trigger. A CI addition of a concurrent-request test on Qwen3.x would have caught this.

## Related

- PR #30 (`1f333f1`) — fixed KV-cache LCP contamination single-request; this is the concurrent analog.
- Upstream PR #384 (`9630c5d`) — "gemma4 CB leaks content across requests via prefix cache LCP" — same class of bug, Gemma variant already fixed.
- Upstream PR #380 (`608ac4d`) — "gemma4 streaming reasoning leak" — Gemma-specific router state leak.
- `docs/testing/2026-04-19-qwen36-first-turn-runaway-under-claude-code-payload.md` — the single-request Qwen runaway, **now closed** by PR #26 + #27.
- `hank-secure-llm` `BLOCKED_MODEL_IDS` currently holds both 35B-A3B Qwens pending this fix.

## Closure (2026-04-23)

### Root cause

`vllm_mlx.scheduler.Scheduler._schedule_waiting` was calling
`BatchGenerator.insert()` WITHOUT passing `logits_processors=` when a
request had no per-request logits processor.  mlx-lm's
`BatchGenerator.insert_segments` normalized the missing kwarg via:

```python
logits_processors = logits_processors or (
    [self.logits_processors] * len(segments)
)
```

`self.logits_processors` is `None` at the BatchGenerator level for our
construction, so the per-row slot became `None`.

When two heterogeneous requests (one WITH a `thinking_budget` processor,
one WITHOUT) completed their prompt phases and merged into a shared
`GenerationBatch` via `extend()`, the merged `logits_processors` became
`[None, [proc]]`.

`GenerationBatch._step()` at `mlx_lm/generate.py:1346`:
```python
for processor in self.logits_processors[e]:   # e=row_with_None
```
crashed with `TypeError: 'NoneType' object is not iterable`.

The error propagated up to our scheduler at `scheduler.py:2759`, which
caught it as a TypeError — but `_is_cache_corruption_error(e)` returned
False because `CACHE_CORRUPTION_PATTERNS` only matched
`"not subscriptable"` (not the error's actual wording `"not iterable"`).
Re-raise fell through to `engine_core._engine_loop`'s bare
`except Exception`, which just logged and slept 100ms in an infinite
loop — turning a one-step bug into a 35s client timeout.

### Fix (three layers)

1. **Caller-side contract** (`vllm_mlx/scheduler.py`): always pass
   `logits_processors=[[per_row]]` with `per_row = list(request.logits_processors or [])`.
   The per-row value is a (possibly empty) list — never omitted, never
   `None`.  See `UPSTREAM_PIN.md` invariant #18.
2. **Defense in depth — scheduler** (`vllm_mlx/scheduler.py`):
   `CACHE_CORRUPTION_PATTERNS` widened to include
   `"'NoneType' object is not iterable"` alongside the existing
   `"not subscriptable"`.  Future variants of the same class of bug
   route through `_recover_from_cache_error()` instead of escalating
   to engine_core.
3. **Defense in depth — engine** (`vllm_mlx/engine_core.py`): bounded
   retry on consecutive identical errors.  After 10 identical error
   reprs in a row, the engine loop calls
   `scheduler._recover_from_generation_error()`, emits
   `finish_reason=error` terminal outputs, and resets the streak
   counter.  Prevents any future similar class from becoming an
   infinite loop.

### Verification (full matrix)

| Condition | Result |
|---|---|
| Pre-fix Qwen3.6-35B-A3B, 10 staggered pairs (with trace on) | **6/10 HANG** |
| Pre-fix stall_streak_max (deadlocked request) | 1797 steps |
| Post-fix Qwen3.6-35B-A3B, 30 pairs × 3 bursts | **0/90 HANG** ✅ |
| Post-fix Gemma-4-26b-a4b-it-4bit, 15 pairs (non-regression) | 0/15 HANG ✅ |
| Post-fix unit regression (`test_scheduler_heterogeneous_logits_processors.py`) | 2/2 PASS ✅ |

### Instrumentation residue

The scheduler-trace subsystem (`VLLM_MLX_SCHEDULER_TRACE=1`, commits
f58aae5 + f1b3158) that localized this bug remains in the codebase as
a disabled-by-default diagnostic.  It's env-gated, does NOT affect
prod-default behavior, and is available for future investigations of
similar heterogeneous-admission classes.  Emit-point documentation
lives in the commit messages + `vllm_mlx/utils/bg_trace.py` docstring.

### Evidence archive

- `/tmp/h1-repro/localization.md` — root-cause walkthrough with
  full traceback and stall-detector output.
- `/tmp/h1-repro/trace-diff.txt` — Qwen-vs-Gemma diff showing matrix
  row 3 signal pre-fix.
- `/tmp/h1-repro/stalls-qwen.txt` — per-uid stall detection showing
  pair=10 heavy at `stall_streak_max=1797`.
- `/tmp/h1-verify-results.txt` — post-fix acceptance evidence.
- `docs/testing/h1-verify.sh` — committed repro/verify script.

### Follow-up

- **48h prod observation window** before this entry is archived.
  Watch `server.log` for any `"not iterable"` or `"not subscriptable"`
  patterns; any hit indicates a regression of a sibling code path.
- **Upstream mlx-lm issue** to be filed proposing defensive
  `for p in (self.logits_processors[e] or []):` in
  `GenerationBatch._step()` at line 1346.  Our caller-side contract
  remains load-bearing until that lands.
- **Remove 35B-A3B Qwens from `hank-secure-llm BLOCKED_MODEL_IDS`**
  after the 48h window passes (separate repo, separate PR).
