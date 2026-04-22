# Qwen3.5 / Qwen3.6 35B-A3B: concurrent heavy-payload requests deadlock

**Status:** **CLOSED PENDING PROD DEPLOYMENT** as of 2026-04-22 post-PR-#31. PR #31's mlx-lm 0.31.3 pin (`ArraysCache.extend` fixes #1169+#1177) resolves both homogeneous AND heterogeneous concurrent-pair cases. Empirical verification on the isolated `vllm-mlx-test` env (post-fix): 20/20 requests across a 10-pair heterogeneous burst pass with proper distinct output. The "heterogeneous still fails" signal observed via `llm.hank.ai` was measured against prod `:8000` / `:8001` which are still running the **vllm-mlx conda env with mlx-lm 0.31.2 (pre-PR-#31 pin)**. Prod needs deployment of the upgraded mlx-lm to pick up PR #31's pin. No additional code fix required. See `docs/superpowers/plans/2026-04-22-qwen3-heterogeneous-prefill-deadlock-h1.md` "Phase 3 execution" section for the full test matrix.
**Reported:** 2026-04-21, refined 2026-04-22 after partial fix shipped. Surfaced via `hank-secure-llm`'s `model_qa` harness run.
**Impact:** High for Claude Code users on Qwen-35B-A3B. Claude Code always fires haiku + sonnet requests in parallel on every scenario; both deadlock → every request times out. Single-request access works fine, so a naive single-curl test from a developer laptop masks the bug completely.
**Scope:** `vllm_mlx/` — one of `StreamingThinkRouter` (`api/utils.py:240`), the continuous-batching scheduler `step()` (`scheduler.py:2588` or `mllm_scheduler.py:595`), or a sibling LCP-contamination path that PR #30's slice fix didn't cover.

## Symptom (refined 2026-04-22)

The current trigger is **heterogeneous concurrent payloads** — one request with tools + large system prompt, one without, arriving within ~50ms. Both hang after emitting only `message_start`.

After the partial fix (build post-1f333f1):
- 1 request in isolation: ✅ ~2s
- 2 concurrent **identical** heavy payloads: ✅ ~3.7s (homogeneous pair — **fixed**)
- 2 concurrent **heterogeneous** payloads (heavy + light): ❌ **still hangs 30s+**
- Same heterogeneous pair against Gemma-4-26b-a4b: ✅ 2.6s (unaffected)

Claude Code's default behavior hits the remaining trigger: every scenario fires one haiku request (`tools=0`, minimal messages) plus one sonnet request (`tools=3`, full tool schemas + long system prompt) **concurrently** at the same model after proxy rewrite. The harness shows every Qwen scenario timing out even though bare single curls succeed. No chunks after that, no `content_block_start`, no `message_stop`. The backend's HTTP response is 200 with SSE content-type, TTFB ~10ms — so the upstream accepted the request and opened the stream, but no tokens are produced for either.

- 1 request in isolation: completes in ~2s (`end_turn`, 2 output tokens), works fine.
- 2 concurrent requests: **both hang** until connection timeout. Deterministic.
- Same test against Gemma-4-26b-a4b-it-4bit: both complete in ~4.8s. Gemma is unaffected.

## Payload that triggers it

Stored at `docs/testing/claude-shape-heavy-payload.json` in this repo (~21 KB, ~4.5k tokens of system prompt + 3 tool definitions + `output_config.effort=high` + `thinking.type=adaptive` + `stream=true`). The shape matches Claude Code's real request.

The single-request **minimal** payload in `docs/testing/2026-04-19-qwen36-first-turn-runaway-under-claude-code-payload.md` (no system prompt, no tools) works fine on the current build — that earlier runaway bug is fixed. This is a distinct, concurrency-only issue.

## Reproduction (heterogeneous pair — current failure mode)

Build a light payload (no tools, no cache_control system block) alongside the heavy one and fire them concurrently:

```bash
python3 -c "
import json
light = {
    'model': 'mlx-community/Qwen3.6-35B-A3B-4bit',
    'max_tokens': 200, 'stream': True,
    'output_config': {'effort':'high'}, 'thinking':{'type':'adaptive'},
    'messages': [{'role':'user','content':'Reply with OK'}],
}
json.dump(light, open('/tmp/shape-light.json','w'))
"

( curl -sS -m 30 -N -o /tmp/h1.sse -w "heavy t=%{time_total}s\n" \
    -X POST https://llm.hank.ai/v1/messages \
    -H "Authorization: Bearer $NEURAL_ARENA_KEY" -H "Content-Type: application/json" \
    -d @docs/testing/claude-shape-heavy-payload.json ) &
( curl -sS -m 30 -N -o /tmp/h2.sse -w "light t=%{time_total}s\n" \
    -X POST https://llm.hank.ai/v1/messages \
    -H "Authorization: Bearer $NEURAL_ARENA_KEY" -H "Content-Type: application/json" \
    -d @/tmp/shape-light.json ) &
wait

echo "heavy: $(grep -c '^event:' /tmp/h1.sse) events, $(grep -c message_stop /tmp/h1.sse) stops"
echo "light: $(grep -c '^event:' /tmp/h2.sse) events, $(grep -c message_stop /tmp/h2.sse) stops"

# Failing output (both hang, observed 2026-04-22):
#   heavy t=30.00s   HTTP 200
#   light t=30.00s   HTTP 200
#   heavy: 1 events, 0 stops
#   light: 1 events, 0 stops
```

Swap both `model` fields to `mlx-community/gemma-4-26b-a4b-it-4bit` — completes in ~2.6s with 6 events + 2 `message_stop`s each. Gemma is the control and remains unaffected.

### Homogeneous pair — now passes (partial fix landed 2026-04-22)

Running the same command with **both** requests pointing at `docs/testing/claude-shape-heavy-payload.json` (identical payloads) now completes in ~3.7s on Qwen3.5/3.6. That code path is resolved; the remaining bug is specifically about the two queued requests having **different shapes** (different `tools` count, different system block, different message length) at prefill time.

The bug reproduces through any path that lets two concurrent requests hit vllm-mlx — arena's `/v1/messages` proxy is a dumb `httpx.AsyncClient.stream()` passthrough (verified in `hank_llm_arena/proxy.py:proxy_v1`, no concurrency guards).

## Hypotheses for investigation (refined after partial fix)

Since the partial fix resolved homogeneous pairs but not heterogeneous, the remaining bug is most likely in code that treats a batch of two same-shape requests correctly but mishandles a batch where shapes differ. Candidates:

**H1 (most likely). Continuous-batching prefill scheduler mispacks heterogeneous batches.** In `scheduler.py:2588` or `mllm_scheduler.py:595`'s `step()`, when two requests enter the prefill queue with very different token counts (e.g., heavy=4500 tokens vs light=30 tokens), the padding / slot allocation / attention mask construction for the mixed batch may be computing an incorrect sequence boundary. One request's tokens get masked away, it never gets to generate a logit, and its stream hangs. The other request may be stuck waiting on a shared step barrier.

**H2. `StreamingThinkRouter` state-sharing specifically on mixed-length batches.** `vllm_mlx/api/utils.py:240`. If each request's router is a per-request instance but they share a tokenizer decode pipeline keyed by the batch index (not the request id), a mixed batch could route the wrong token stream into each router.

**H3. Prefix cache LCP fetch incorrectly applied across requests with different prompts.** PR #30's fix sliced dequant + rotating KVCache trim. For a mixed batch, one request's prompt has no shared prefix with the other, but the CB LCP pass might still be computing a common prefix and serving stale tokens to one side. PR #384 (`9630c5d`) handled Gemma specifically; Qwen-3_5_moe has different attention group sizes and may hit a separate code path.

**H4. Attention group / KV head sizing for Qwen3-MoE in mixed batches.** Qwen3-MoE has GQA with specific group counts. If the KV head packing / unpacking assumes uniform sequence length across a batch and the mixed-shape case computes an off-by-one slice, reads return garbage and generation stalls.

The symptom (both requests stall at `message_start`, no tokens emitted) and the shape-sensitivity (same shape OK, different shape breaks) together strongly suggest H1 — a CB scheduler step that only proceeds when both concurrent requests reach some expected alignment that mixed-shape batches never satisfy.

## Observable signals while reproducing

During the hang, in `vllm-mlx` server logs look for:
- Two concurrent `[REQUEST] POST /v1/messages` arriving within ~100ms of each other
- Does either emit `[stream_outputs] <req-id> first token after <N>s`? If neither, scheduler or CB is stuck. If one emits and the other doesn't, it's router/cache state-sharing.
- Log the per-request `router_id`, `kv_cache_slab_id`, `scheduler_queue_position` — does one request reference another's id?
- Is the `260s --streaming-max-seconds` cap from PR #23 firing and closing the stream? If so, CB step() is live but never produces output for either.

## Success criteria for the fix

This file should be kept in-repo and a regression test added. The new test must cover **three** concurrency shapes, not just one — the partial fix passed the homogeneous case but missed heterogeneous:

1. **Heterogeneous concurrent pair** (primary — this is the open bug): one heavy request (~4.5k token system + 3 tools + `effort=high` + `thinking.adaptive`) and one light request (no system, no tools, `Reply OK` prompt) via `asyncio.gather` at the SAME model. Both must complete in <15s with `stop_reason: end_turn` and ≥1 `content_block_delta` each.
2. **Homogeneous heavy pair** (regression guard for the partial fix): two identical heavy requests. Must still complete in <15s each.
3. **Same tests against Gemma-4-26b-a4b** (cross-family non-regression): both homogeneous and heterogeneous pairs must complete — ensures the Qwen-specific fix doesn't regress CB for other families.

A Python `httpx.AsyncClient` variant is preferred — it lets the test log per-request state machine progression (`router_id`, `kv_cache_slab_id`, `scheduler_queue_position`, and crucially the batch slot assignment + prefill sequence boundaries) for diagnostic value. Because the partial fix slipped through with only the homogeneous case covered, the new test should explicitly name the heterogeneous case and assert both requests made forward progress independently (not just one of them).

## Why this was hard to catch

Single-request curl succeeds. The previous runaway doc (`2026-04-19-qwen36-first-turn-runaway-under-claude-code-payload.md`) was also single-request and its minimal payload is now fixed — that checks out, PR #26 + PR #27 closed the tool+big-payload path for single requests. But the existing tests in `tests/test_issue_29_end_to_end.py` and `tests/test_memory_cache_mlx.py` don't exercise concurrency. Claude Code being the canary here is load-bearing: it's the only real-world caller that reliably fires parallel haiku + sonnet requests against the same model, which is exactly the trigger. A CI addition of a concurrent-request test on Qwen3.x would have caught this.

## Related

- PR #30 (`1f333f1`) — fixed KV-cache LCP contamination single-request; this is the concurrent analog.
- Upstream PR #384 (`9630c5d`) — "gemma4 CB leaks content across requests via prefix cache LCP" — same class of bug, Gemma variant already fixed.
- Upstream PR #380 (`608ac4d`) — "gemma4 streaming reasoning leak" — Gemma-specific router state leak.
- `docs/testing/2026-04-19-qwen36-first-turn-runaway-under-claude-code-payload.md` — the single-request Qwen runaway, **now closed** by PR #26 + #27.
- `hank-secure-llm` `BLOCKED_MODEL_IDS` currently holds both 35B-A3B Qwens pending this fix.
