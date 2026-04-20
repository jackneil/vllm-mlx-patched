# Qwen3.6-35B-A3B: first-turn runaway under Claude Code payload shape

**Status:** Fix landed (2026-04-20). Design at `docs/superpowers/specs/2026-04-20-qwen3-runaway-fix-design.md` (local/gitignored). Layer 1 (surgical auto-disable) is default-on; Layer 2 (`--max-thinking-token-budget N` ceiling) is opt-in. The existing `--streaming-max-seconds 260` cap remains as third-line backstop. Rest of this document stays as historical record of the investigation.
**Reported:** 2026-04-19, surfaced via `hank-secure-llm` `model_qa` harness after restarting with the post-`fa66768` build.
**Investigated:** 2026-04-20.
**Impact:** High for Claude-Code users of Qwen3.6 on desktop. Every first-turn request stalls until the 260s `--streaming-max-seconds` cap fires. Observable as 4:25 wall-clock for a "say OK" prompt.
**Scope:** Model generation behavior, not `vllm_mlx/reasoning/qwen3_parser.py` (falsified — see [Investigation](#investigation-update--2026-04-20) below). Existing 260s cap (commit `64a8085`) catches it cleanly but with poor UX; defensive fix brings the bound in to ~1s via a properly-sized thinking-token budget.

## Investigation update — 2026-04-20

After pulling external evidence ([`ggml-org/llama.cpp#21118`](https://github.com/ggml-org/llama.cpp/issues/21118), [`vllm-project/vllm#17655`](https://github.com/vllm-project/vllm/issues/17655)) and running falsification tests against the live fleet, here's what's actually happening:

### Mechanism (from upstream)

Per the llama.cpp thread: "Qwen3.5 models are in-context learning to **not emit a `</think>` tag** when the reasoning from previous turns in the same `tool_call → tool → tool_call → ...` loop is not present. Most of the newer reasoning models expect this, and it was coined 'interleaved thinking.'"

Translation for our single-turn case: when the request includes `tools: [...]` plus a large Claude-Code-style system prompt, Qwen3.6's sampler enters interleaved-thinking mode (expecting an agent loop is about to begin) and **never emits `</think>`** because the model is "waiting for the tool-result round-trip" that never happens on first turn. The stream runs until our server-side 260s cap fires.

### Falsification attempts — synthetic repro failed (0/168 hangs)

To verify the mechanism I ran 168 streaming trials against the running serve (port 8000, Qwen3.6-35B-A3B-4bit, same flags as production: `--continuous-batching --enable-auto-tool-choice --tool-call-parser qwen3 --reasoning-parser qwen3`):

**Matrix phase — 8 cells isolating `tools` × `effort` × `long_system` (1 trial each):** wall-clock 1.36-3.44s, all ended `end_turn`.

**Stress phase — 4 Claude-Code-shaped variants × 30 trials = 120, with `DELETE /v1/cache` between every trial (cold prefix cache):**

| Variant | Payload | N | timeouts | natural `end_turn` | wall median | wall max |
|---|---|---|---|---|---|---|
| A | tools only | 30 | 0 | 30 | 1.20s | 4.11s |
| B | tools + long system + `output_config.effort=high` | 30 | 0 | 29 (1 `tool_use`) | 2.50s | 2.92s |
| C | tools + long system + `thinking.type=enabled` | 30 | 0 | 29 (1 `tool_use`) | 2.50s | 2.93s |
| D | tools + long system + `effort=high` + `thinking.type=enabled` | 30 | 0 | 30 | 2.50s | 2.99s |

**Zero hangs across 168 trials.** So the synthetic payload — 3 tool definitions, a 100-line system array with `cache_control.type=ephemeral`, both thinking flags set, trivial "reply OK" prompt — does not reproduce. The hang requires something specific in Claude Code's actual payload that we haven't captured (candidate diffs: specific tool descriptions/schemas, system-reminder block contents, or the exact token sequence interaction with the chat template).

### H1/H2/H3 from the original hypotheses

- **H1 (parser loses `</think>` detection mid-stream):** not supported. `vllm_mlx/reasoning/qwen3_parser.py` is 69 lines of simple string-match logic; no stateful regression path visible. Commit `55d25cf` only changed the NO-tags short-circuit — can't affect streams where tags are (or aren't) present.
- **H2 (chat template pre-injects `<think>` without `</think>`):** partially correct but the mechanism per llama.cpp#21118 is richer — it's in-context learning from tools + system-prompt shape, not just a template defect.
- **H3 (StreamingThinkRouter state leaks across requests):** **falsified by code inspection.** `StreamingThinkRouter` is instantiated per request at `vllm_mlx/server.py:2548`, inside the `_stream_anthropic_messages` function scope. Each stream gets a fresh router.
- **H4 (`thinking.type=adaptive` × `output_config.effort=high` interact):** falsified by variant D — both set simultaneously, 30/30 clean `end_turn`.

The upstream-mechanism explanation from llama.cpp#21118 remains the best-supported reading. We just can't hit it via synthetic curl.

### Defensive fix — bound thinking tokens, not wall-clock

The current safety net (`--streaming-max-seconds 260`, commit `64a8085`) works but is the wrong unit: it lets the model think for 260 seconds before giving up. The right unit is **thinking tokens**. The `thinking_token_budget` infrastructure (commit `7be87fa`, port of vLLM #20859) is already wired into both streaming OpenAI and Anthropic paths — server.py:1840, 2143, 2472. It's reachable per-request via the `thinking_token_budget` field or the effort-resolver system, but there's **no server-side default floor**.

Empirical measurement against live port 8000, fresh cache, complex multi-step prompt ("train catch-up problem, show reasoning"):

| `thinking_token_budget` | thinking words emitted | text chars | wall | stop |
|---|---|---|---|---|
| unset (baseline) | 749 | 1,345 | 18.25s | end_turn |
| 1000 | 602 | 1,216 | 14.35s | end_turn |
| 500 | 270 | 3,149 | 24.66s | end_turn |
| 200 | 88 | 2,923 | 12.93s | end_turn |
| 100 | 34 | 5,410 | 21.21s | end_turn |
| 50 | 0 (force-close before first flush) | 1,988 | 7.14s | end_turn |

Monotonic, deterministic bound. At `budget=100` the model gets 34 words of think and then emits text; at `budget=0` thinking is skipped entirely. `end_turn` fires in every case — no cap, no truncation. **This is the right knob.**

### Initial fix proposal — superseded by three-layer design (2026-04-20)

The proposal in this section was the starting point. After swarm-research, firecrawl investigation of upstream projects, and a three-round /dc review, the final design is a **three-layer defense** rather than a single flag:

1. **Layer 1 (surgical, default-on):** detect `reasoning_parser="qwen3" + tools + no prior assistant` → inject `chat_template_kwargs.enable_thinking=False`. Uses Qwen's own chat-template mechanism.
2. **Layer 2 (opt-in ceiling):** `--max-thinking-token-budget N` CLI flag clamps resolved budget down via shared helper at 4 resolve-sites.
3. **Layer 3 (existing):** `--streaming-max-seconds 260` wall-clock cap as backstop.

Full design + verified-bound measurements + upstream references at `docs/superpowers/specs/2026-04-20-qwen3-runaway-fix-design.md` (local, gitignored per repo convention). Implementation plan at `docs/superpowers/plans/2026-04-20-qwen3-runaway-fix.md` (local, 3237 lines, 23 tasks, reviewed across 3 /dc rounds). Execution pending.

The original proposal below kept for historical record:

### Initial proposal — `--default-thinking-token-budget` CLI flag (superseded)

Mirror the existing `--streaming-max-seconds` pattern in `cli.py`:

1. Add `--default-thinking-token-budget N` to the serve subparser (default: unset = no floor, preserving current behavior for operators who don't opt in).
2. When set: any request that reaches `_chat_completions_dispatch_kwargs` (server.py ~line 1817) **without** an explicit `thinking_token_budget` (request field, `chat_template_kwargs`, or effort-resolver result) inherits the server default.
3. Operators serving Qwen3.6 to Claude Code users set `--default-thinking-token-budget 2048` (or similar) — tight enough to bound first-turn runaway to ~5-10s, loose enough to not starve legitimate reasoning.

This is orthogonal to `--streaming-max-seconds` — the wall-clock cap remains as a second line of defense for any other model-side non-termination.

### Why not file upstream against ml-explore/mlx-lm

The mechanism is **model behavior** (Qwen3.6 weights + chat template in-context-learning to withhold `</think>`), not an MLX or vllm-mlx bug. llama.cpp#21118 ran into the same thing with a different runtime. An upstream MLX or vllm-mlx issue asking them to force-close `</think>` would be reinventing what our `thinking_token_budget` already does — correctly, we just haven't defaulted it.

### Next diagnostic (if needed for fix verification)

Because 168 synthetic trials failed to reproduce, any fix we ship can't be deterministically end-to-end verified against the bug on this box — only against the bound mechanism. To close the loop:

1. **Add request-body capture to `hank-secure-llm/app/src-tauri/src/proxy.rs`** under a conditional: dump full body to `/tmp/cc-hang-<timestamp>.json` when proxy observes `chunks_sent > 1000 AND message_stop_count == 0 AND duration_ms > 200000`. Negligible overhead (writes only on hang), fires only on the specific symptom.
2. **Wait for next hang in the wild.** Per the sister arena doc, hangs are "non-deterministic" — some runs pass, some hang. Don't need to force it.
3. **Replay captured body via `curl`** — deterministic reproducer. Then toggle `--default-thinking-token-budget` and measure end-to-end fix.

Until that capture exists, the fix ships on the basis of the measured bound mechanism + the upstream-validated root cause.

## Symptom

With today's post-restart vllm-mlx build (PR #13 resolver + PR #14 signature + PR #20 non-streaming adapter + PR #22 interleaved-thinking round-trip + PR #23 260s cap all present), Qwen3.6 on a direct curl with a trivial prompt finishes in 2.6s:

```bash
curl -N -X POST https://llm.hank.ai/v1/messages \
  -d '{"model":"mlx-community/Qwen3.6-35B-A3B-4bit","max_tokens":200,"stream":true,
       "output_config":{"effort":"high"},
       "messages":[{"role":"user","content":"Reply with exactly OK"}]}' \
  | grep -E 'content_block_start|stop_reason|message_stop'
# → content_block_start type: thinking
# → stop_reason: end_turn
# → event: message_stop
# Wall clock: 2.6s
```

The stream terminates cleanly with `message_stop`. Thinking-only response is a separate "max_tokens < floor" issue (see `docs/testing/2026-04-19-max-tokens-floor-not-enforced-on-anthropic-messages.md` or equivalent).

**Same model, same upstream, through Claude Code CLI with `--bare -p "Reply with exactly OK"`**:

| Measurement | Value |
|---|---|
| Wall clock | 4:25.81 (265s) |
| Implied cap-hit count | 1 × 260s cap |
| Result | Claude Code emits empty `.result` string (non-streaming fallback also empty or similar) |

The 260s cap is firing — that's what's bounding the run, not an infinite hang. But the cap firing means `stop_reason:"max_tokens"` is being emitted artificially, not as a natural end-of-turn.

Gemma-4-31b under the exact same Claude Code payload completes all 6 scenarios in 16–46s each. Issue is Qwen3.6-specific.

## Claude Code's payload shape (what differs from direct curl)

Captured earlier via a mock-upstream mirror on 2026-04-18:

- System prompt: ~4,500 tokens of `system-reminder`, tool instructions, context-management hints (`cache_control: ephemeral` on parts)
- 3 tool definitions (`Bash`, `Read`, `Write` or similar in `--bare` default)
- `thinking.type = "adaptive"` (Claude Code's default when `--effort` flag is set)
- `output_config.effort = "high"` (Claude Code's default when `--effort` unspecified)
- Actual user message is tiny (the prompt text)
- **No prior assistant messages** on first turn — so PR #22's interleaved-thinking round-trip doesn't apply

## Reproduction status

- **Synthetic (curl, 168 trials across 8 factor-matrix cells and 4 Claude-Code-shape variants with cache clear):** negative. All trials ended `end_turn` under 4.11s wall-clock.
- **Wild (hank-secure-llm model_qa harness via Claude Code):** positive, non-deterministic. "Same prompt may succeed or hang across runs" per the sister arena doc.
- **Next step to bridge the gap:** capture a real Claude Code request body (add a body-dump on-hang-signature path in `hank-secure-llm/app/src-tauri/src/proxy.rs`), then replay via curl. See [Next diagnostic](#next-diagnostic-if-needed-for-fix-verification).

## What this blocks downstream

- `hank-secure-llm` `model_qa` harness can't complete most scenarios on Qwen3.6 (all tool-use scenarios time out at 60–180s; only trivial `simple_reply` sometimes squeaks through).
- Desktop-app users who pick Qwen3.6 hit 4+ minute hangs per request. The 260s cap makes it recoverable but not usable.
- `effort_gradient` observability cell in the harness false-fails on Qwen because `effort_low` times out before the gradient can be computed.

## Requested outcome

A follow-up that either:
- **Identifies the parser/stop-gate regression** specific to Qwen3.6 under heavy single-turn input and fixes it (preferred), OR
- **Documents Qwen3.6 as a known-incompatible model with Claude Code under effort=high** and moves it to `BLOCKED_FAMILIES` or equivalent client-facing filter, with a note pointing to the upstream bug.

Cross-check: Qwen3.5-35B-A3B on the same payload — does it exhibit the same bug or is this strictly Qwen3.6? Qwen3.5 didn't exhibit this in my 2026-04-17 runs but hasn't been re-tested post-restart.

## Related

- `docs/testing/2026-04-18-thinking-budget-empirical-findings.md` — earlier smaller-payload validation.
- `hank-llm-arena/docs/2026-04-19-qwen36-stream-runaway.md` — the hank-side summary of the symptom before the 260s cap landed.
- PR #22 (`22c9c6a`) — interleaved-thinking fix for multi-turn; orthogonal to this single-turn case.
- PR #23 (`fa66768`) — `--streaming-max-seconds` cap; the reason this is bounded at 265s instead of hanging forever.
