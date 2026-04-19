# Qwen3.6-35B-A3B: first-turn runaway under Claude Code payload shape

**Status:** TODO — likely a qwen3 reasoning parser or stop-gate issue specific to heavy single-turn input (no prior assistant history).
**Reported:** 2026-04-19, surfaced via `hank-secure-llm` `model_qa` harness after restarting with the post-`fa66768` build.
**Impact:** High for Claude-Code users of Qwen3.6 on desktop. Every first-turn request stalls until the 260s `--streaming-max-seconds` cap fires. Observable as 4:25 wall-clock for a "say OK" prompt.
**Scope:** `vllm_mlx/reasoning/qwen3_parser.py` and/or `vllm_mlx/server.py:_stream_anthropic_messages` stop detection. Not the `anthropic_adapter.py` round-trip fix (`22c9c6a`, PR #22) — that fix addresses multi-turn history; this is single-turn.

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

## Hypotheses

**H1. Long system prompt triggers a parser state where `</think>` detection degrades.** `qwen3_parser.py` tracks `<think>` start detection — if the parser is sensitive to the large ephemeral-cached system prompt (which contains unusual tokens like `<system-reminder>`), it may fail to recognize or emit `</think>` at the right moment, keeping thinking budget uncapped until the 260s server wall-cap fires.

**H2. Chat template + Claude Code's `thinking.type: "adaptive"` combine to pre-inject `<think>` but no `</think>`.** The PR #10 fix (`23cecdc`) addresses budget=0 with pre-injected `<think>`. If `adaptive` mode pre-injects differently and the end-token detection isn't wired, the model never gets force-closed via the logits processor, runs to cap.

**H3. Large tool-definition payload (3 tools, ~4KB JSON schema total) triggers a threshold where MoE expert routing degrades on Qwen3.6 specifically.** MoE-3B-active might allocate reasoning to experts that don't converge on `</think>` emission under heavy instruction-tuning-prompt stress.

**H4. `thinking.type = "adaptive"` interacts with `output_config.effort = "high"` in a way that breaks one or both.** Only one should win per the effort-resolver precedence table (`vllm_mlx/api/effort.py:57-63`); but if both are applied against the same model, the budget/thinking pipeline may double-configure and break.

## Minimal reproduction (Python, no Claude Code needed)

The Claude Code payload can be captured once and replayed as a fixture. Target: find the minimal payload that reproduces the >60s wall-clock on Qwen3.6. Bisect by progressively stripping:
1. Start with Claude Code's full captured body (paste from a mirror upstream — see `hank-llm-arena/docs/2026-04-19-qwen36-stream-runaway.md` for capture technique).
2. Binary-chop the system prompt length.
3. Drop tool definitions one by one.
4. Swap `thinking.type: "adaptive"` → removed, → `"enabled"`, etc.
5. Vary effort between `low` and `high`.

Whichever combination still reproduces identifies the trigger.

## Observable signals during the hang (for log bisection)

During the 265s period, `vllm-mlx` server logs should show:
- `[REQUEST] POST /v1/messages` with the right payload dimensions
- `[stream_outputs] <req-id> first token after <N>s` — is N reasonable?
- Streaming token generation — does `chunks_sent` monotonically grow?
- Does the thinking budget logits processor attach? (`x-thinking-budget-applied: true/false` header on the response)
- When the 260s cap fires: look for the log line added by PR #23 that announces the cap and emits the tail frame. Confirms the cap is the terminator, not a natural stop.

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
