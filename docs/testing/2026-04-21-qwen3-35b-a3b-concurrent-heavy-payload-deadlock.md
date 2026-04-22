# Qwen3.5 / Qwen3.6 35B-A3B: concurrent heavy-payload requests deadlock

**Status:** TODO — concurrent-request shared-state bug, Qwen3.x-specific. PR #30 (issue #29) fixed KV-cache LCP contamination in the single-request path but did not cover the concurrent-prefill case.
**Reported:** 2026-04-21, surfaced via `hank-secure-llm`'s `model_qa` harness run against post-PR-#30 `origin/main`.
**Impact:** High for Claude Code users on Qwen-35B-A3B. Claude Code always fires haiku + sonnet requests in parallel on every scenario; both deadlock → every request times out. Single-request access works fine, so a naive single-curl test from a developer laptop masks the bug completely.
**Scope:** `vllm_mlx/` — one of `StreamingThinkRouter` (`api/utils.py:240`), the continuous-batching scheduler `step()` (`scheduler.py:2588` or `mllm_scheduler.py:595`), or a sibling LCP-contamination path that PR #30's slice fix didn't cover.

## Symptom

Fire **two identical heavy-payload requests concurrently** at Qwen3.5-35B-A3B-4bit or Qwen3.6-35B-A3B-4bit — both hang after emitting only `message_start`. No chunks after that, no `content_block_start`, no `message_stop`. The backend's HTTP response is 200 with SSE content-type, TTFB ~10ms — so the upstream accepted the request and opened the stream, but no tokens are produced for either.

- 1 request in isolation: completes in ~2s (`end_turn`, 2 output tokens), works fine.
- 2 concurrent requests: **both hang** until connection timeout. Deterministic.
- Same test against Gemma-4-26b-a4b-it-4bit: both complete in ~4.8s. Gemma is unaffected.

## Payload that triggers it

Stored at `docs/testing/claude-shape-heavy-payload.json` in this repo (~21 KB, ~4.5k tokens of system prompt + 3 tool definitions + `output_config.effort=high` + `thinking.type=adaptive` + `stream=true`). The shape matches Claude Code's real request.

The single-request **minimal** payload in `docs/testing/2026-04-19-qwen36-first-turn-runaway-under-claude-code-payload.md` (no system prompt, no tools) works fine on the current build — that earlier runaway bug is fixed. This is a distinct, concurrency-only issue.

## Reproduction

```bash
# Fire two heavy curls in parallel. Both will hang 30s+.
for i in 1 2; do
  curl -sS -m 30 -N -o /tmp/c$i.sse -w "c$i HTTP %{http_code} t=%{time_total}s\n" \
    -X POST https://llm.hank.ai/v1/messages \
    -H "Authorization: Bearer $NEURAL_ARENA_KEY" \
    -H "Content-Type: application/json" \
    -d @docs/testing/claude-shape-heavy-payload.json &
done
wait

# Check what each received:
for i in 1 2; do
  echo "c$i events: $(grep -c '^event:' /tmp/c$i.sse) stops: $(grep -c message_stop /tmp/c$i.sse)"
done

# Failing output (both hang):
#   c1 HTTP 200 t=30.00s
#   c2 HTTP 200 t=30.00s
#   c1 events: 1 stops: 0
#   c2 events: 1 stops: 0
```

Run the same command swapping `d['model'] = 'mlx-community/gemma-4-26b-a4b-it-4bit'` — completes in ~4.8s, both 6 events + 2 `message_stop`s. Gemma is the control.

The bug reproduces through any path that lets two concurrent requests hit vllm-mlx — arena's `/v1/messages` proxy is a dumb `httpx.AsyncClient.stream()` passthrough (verified in `hank_llm_arena/proxy.py:proxy_v1`, no concurrency guards).

## Hypotheses for investigation

**H1. `StreamingThinkRouter` shared state leak across concurrent requests.** `vllm_mlx/api/utils.py:240` — if router state (token accumulators, `<think>` detection position, end-gate flag) isn't strictly per-request, two concurrent Qwen streams could have one router's state poison the other's end-token detection, so both stall waiting for `</think>`. Matches symptom: `message_start` emits, then nothing.

**H2. Continuous-batching scheduler (`scheduler.py:2588` `step()` or `mllm_scheduler.py:595`) mishandles concurrent prefills of large prompts.** Both requests enter prefill at nearly the same time. If the scheduler has a state field that isn't properly keyed per-request — or a serialized critical section that deadlocks when two large prefills are queued — both would stall before generating any token.

**H3. Prefix cache / LCP contamination path that PR #30's slice fix didn't cover.** PR #30 fixed the dequant + rotating KVCache trim branches in `vllm_mlx/memory_cache.py`. If a sibling code path (e.g., memory_cache write-back for concurrent allocations, or a CB-specific LCP fetch used only when batches > 1) still leaks, the second-arriving request could grab a corrupted cache view and never progress. PR #29 (`9630c5d`, upstream #384) covered Gemma-4 specifically — the Qwen equivalent may need the analogous fix.

**H4. Qwen3 reasoning parser (`vllm_mlx/reasoning/qwen3_parser.py`) loses its `</think>` detection mid-stream under concurrent input contexts.** Parser state is per-request in the happy path but might reference a shared tokenizer state or module-level cache. Under concurrent heavy prefills, the parser for both requests might stay in "thinking" mode indefinitely.

## Observable signals while reproducing

During the hang, in `vllm-mlx` server logs look for:
- Two concurrent `[REQUEST] POST /v1/messages` arriving within ~100ms of each other
- Does either emit `[stream_outputs] <req-id> first token after <N>s`? If neither, scheduler or CB is stuck. If one emits and the other doesn't, it's router/cache state-sharing.
- Log the per-request `router_id`, `kv_cache_slab_id`, `scheduler_queue_position` — does one request reference another's id?
- Is the `260s --streaming-max-seconds` cap from PR #23 firing and closing the stream? If so, CB step() is live but never produces output for either.

## Success criteria for the fix

This file should be kept in-repo and a regression test added. The new test must:

1. Spin up a single Qwen3.5 or Qwen3.6 model in a local pytest fixture.
2. Fire two concurrent POSTs to `/v1/messages` with the heavy payload (or a smaller functional equivalent that still triggers).
3. Assert both complete within a budget (e.g., 15s) with `stop_reason: end_turn` and at least one `content_block_delta` each.
4. Run the same test against Gemma-4-26b-a4b as a non-regression sanity (must also pass — ensures the fix doesn't regress continuous batching in general).

A shell-level equivalent that wraps the curl block above is acceptable, but a Python `httpx.AsyncClient` variant with both requests awaited via `asyncio.gather` is preferred — it lets the test log per-request state machine progression (router id, cache slab id, scheduler queue position) for diagnostic value if a future regression appears.

## Why this was hard to catch

Single-request curl succeeds. The previous runaway doc (`2026-04-19-qwen36-first-turn-runaway-under-claude-code-payload.md`) was also single-request and its minimal payload is now fixed — that checks out, PR #26 + PR #27 closed the tool+big-payload path for single requests. But the existing tests in `tests/test_issue_29_end_to_end.py` and `tests/test_memory_cache_mlx.py` don't exercise concurrency. Claude Code being the canary here is load-bearing: it's the only real-world caller that reliably fires parallel haiku + sonnet requests against the same model, which is exactly the trigger. A CI addition of a concurrent-request test on Qwen3.x would have caught this.

## Related

- PR #30 (`1f333f1`) — fixed KV-cache LCP contamination single-request; this is the concurrent analog.
- Upstream PR #384 (`9630c5d`) — "gemma4 CB leaks content across requests via prefix cache LCP" — same class of bug, Gemma variant already fixed.
- Upstream PR #380 (`608ac4d`) — "gemma4 streaming reasoning leak" — Gemma-specific router state leak.
- `docs/testing/2026-04-19-qwen36-first-turn-runaway-under-claude-code-payload.md` — the single-request Qwen runaway, **now closed** by PR #26 + #27.
- `hank-secure-llm` `BLOCKED_MODEL_IDS` currently holds both 35B-A3B Qwens pending this fix.
