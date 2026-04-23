# Qwen3.x streaming: thinking content blocks emit no `signature`, causing Claude Code to silently drop subsequent text

**Status:** TODO — streaming-path regression on the Anthropic adapter. Non-streaming responses are correct (include `signature` on thinking blocks); streaming responses omit it entirely. Claude Code's client silently discards the text block that follows an unsigned thinking block, making non-tool responses appear empty to the application even though the wire bytes contain the text.
**Reported:** 2026-04-23, surfaced via `hank-secure-llm`'s `model_qa` harness running against prod `llm.hank.ai` after the concurrent-prefill deadlock (PR #33 / commit 312de2f) closed. Once the deadlock was fixed, the harness stopped timing out and started returning fast empty-result failures on every non-tool scenario.
**Impact:** High for Claude Code users on Qwen3.5/3.6. Every prompt that returns "thinking → text" produces an empty `.result` in `--output-format=json`. The terminal UI looks like the model failed, even though the inference ran successfully and the upstream stream contains the answer.
**Scope:** `vllm_mlx/api/anthropic_adapter.py` / `vllm_mlx/server.py` streaming emitter — specifically the path that emits `content_block_start` / `content_block_delta` / `content_block_stop` events for thinking content. The non-streaming path (PR #14, `6b580ba`) already attaches `signature`; the streaming path was never updated to match.

## Symptom (harness)

| scenario (Qwen3.6-35B-A3B-4bit) | result |
|---|---|
| `simple_reply` — "Reply with exactly 'OK' and nothing else." | ✗ **empty `.result`**, 3.4s, `stop_reason: end_turn` |
| `list_output` — "List the numbers 1 to 10..." | ✗ empty, 3.6s |
| `effort_low` / `effort_high` — "Think step by step: what is 13 × 17?" | ✗ empty, 3-5s |
| `tool_read` — "Read /tmp/hank-qa/fixture.txt..." | ✅ 6.8s, 2 turns |
| `tool_write` — "Write 'qa-test-ok' to..." | ✅ 7.0s, 2 turns |

Pattern: any response where the model emits `thinking` followed by `text` produces empty `.result`. Responses where the second block is `tool_use` work fine. Gemma-4-26b passes all seven scenarios because its reasoning parser is a documented no-op (no thinking block emitted).

## Proof: the bug is streaming-path only

Identical payload, two requests, one streaming one not. Same model instance, same PID, seconds apart:

```bash
# Non-streaming — CORRECT
$ curl -sS ... -d '{..., "stream": false}' | jq '.content[].type'
"thinking"
"text"
$ curl -sS ... -d '{..., "stream": false}' | jq '.content[0].signature'
"vllm-mlx:c31d43cb017a637728906835423c96f2"       # ← present

# Streaming — MISSING signature
$ curl -sS -N ... -d '{..., "stream": true}' | grep -c '"signature"'
0                                                  # ← never emitted
```

Full streaming event trace looks like:

```
event: message_start
event: content_block_start    {"index":0,"content_block":{"type":"thinking","thinking":""}}
event: content_block_delta    {"index":0,"delta":{"type":"thinking_delta","thinking":"..."}}
... (many thinking_deltas) ...
event: content_block_stop     {"index":0}
                              ← NO signature_delta here, NO signature on the stop
event: content_block_start    {"index":1,"content_block":{"type":"text","text":""}}
event: content_block_delta    {"index":1,"delta":{"type":"text_delta","text":"OK"}}
event: content_block_stop     {"index":1}
event: message_delta          {"delta":{"stop_reason":"end_turn"}}
event: message_stop
```

The thinking block opens, streams content, then closes without ever carrying a signature. On the non-streaming path, the accumulated response object includes `content[0].signature = "vllm-mlx:<hash>"`. PR #14 (`6b580ba fix(anthropic): required signature field on thinking content blocks`) added this for non-streaming; the streaming emitter was not updated.

## Why Claude Code silently drops the text

Anthropic's `messages` API requires that thinking content blocks carry a `signature` field in both streaming and non-streaming responses (so a client can verify the reasoning wasn't tampered with en route). Claude Code's client-side stream aggregator treats an unsigned thinking block as either (a) an incomplete / malformed response, or (b) a signal that subsequent content blocks in the same message should be discarded (presumably to prevent prompt-injection attacks via unsigned reasoning).

In the failure mode observed:

```json
{
  "is_error": false,
  "result": "",                // ← empty
  "stop_reason": "end_turn",   // ← clean completion
  "num_turns": 1,
  "duration_api_ms": 1877      // ← inference actually succeeded
}
```

`is_error` is false and `stop_reason` is `end_turn`, so from the client's perspective nothing failed — but the text that was on the wire never made it into `.result`. Users see a hang-equivalent in their terminal UX even though no timeout occurred.

## Why tool scenarios still pass

When the second block is `tool_use` (not `text`), Claude Code's client parses it as a tool-call intent and dispatches the tool regardless of the thinking block's signature state. The tool response comes back, the model produces a second turn with the tool output included, and the final assistant text block (no preceding thinking) appears in `.result`. This is why the `tool_read` and `tool_write` harness scenarios show 2 turns and populate `.result` correctly, while `simple_reply` / `list_output` / effort-pair (all single-turn, thinking+text shape) show empty.

## Reproduction

No concurrent traffic needed. Single request is sufficient:

```bash
curl -sS -N -X POST https://llm.hank.ai/v1/messages \
  -H "Authorization: Bearer $NEURAL_ARENA_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
    "max_tokens": 32000,
    "stream": true,
    "output_config": {"effort": "high"},
    "thinking": {"type": "adaptive"},
    "messages": [{"role": "user", "content": "Reply with exactly OK"}]
  }' > /tmp/stream.sse

grep -c '"signature"' /tmp/stream.sse      # → 0 (bug)
grep -c '"text_delta"' /tmp/stream.sse     # → ≥1 (text IS on the wire)
```

And the end-to-end through Claude Code:

```bash
ANTHROPIC_BASE_URL=http://<your-proxy>:<port> \
ANTHROPIC_API_KEY=<proxy-token> \
claude --bare -p --output-format json "Reply with exactly 'OK' and nothing else." | jq '.result'
# → "" (empty, reproduced 5/5 on Qwen3.5 and Qwen3.6 this session)
```

## Fix direction

The Anthropic streaming spec (as of the 2024 thinking-blocks SDK update) carries signatures via either:

1. A `signature_delta` event in the same pattern as `thinking_delta`, accumulated client-side and finalized on `content_block_stop`, **or**
2. A `signature` field on the `content_block_stop` event for the thinking block

Option (2) is simpler for the emitter — the server already has the full thinking text at close time, so it can compute the signature then and attach it to the stop event. This matches how the non-streaming path works today (compute signature after the full thinking is accumulated).

Concrete edit location candidates:
- `vllm_mlx/api/anthropic_adapter.py` — the streaming adapter that converts OpenAI-shaped model output into Anthropic SSE events. The `emit_content_block_stop()` or equivalent function needs to attach the signature when the closing block type is `thinking`.
- `vllm_mlx/server.py:_stream_anthropic_messages` — the actual SSE writer. If the signature is computed upstream and passed down, this function needs to include it in the event payload.

Reference the non-streaming path's signature computation (from PR #14) — same hash, same prefix (`"vllm-mlx:"`), same input (the concatenated thinking content). Consistency between streaming and non-streaming signatures lets clients reuse validation logic.

**Chosen fix (shipped 2026-04-23):** Option (1) — `signature_delta` delta events before `content_block_stop`. This matches Anthropic's canonical streaming spec and lets clients beyond Claude Code (any Anthropic-SDK-compliant client) reuse their aggregation logic. See plan `docs/superpowers/plans/2026-04-23-qwen3-streaming-thinking-signature.md` and UPSTREAM_PIN.md invariant 13 for details.

## Success criteria

1. Streaming response on Qwen3.5/3.6 emits a non-empty signature on the thinking block — either via `signature_delta` events or as a `signature` field on `content_block_stop` (whichever matches Anthropic's current spec; non-streaming already uses the latter via the response object).
2. Signature matches what the non-streaming path computes for the same prompt/output (hash prefix `vllm-mlx:`, same bytes as non-stream would produce).
3. `hank-secure-llm`'s `model_qa` harness running `claude --bare -p` on Qwen3.6 produces non-empty `.result` for the five non-tool scenarios (`simple_reply`, `list_output`, `effort_low`, `effort_high`, `effort_gradient`) — they should flip from ✗ empty to ✓ with content. Tool scenarios should stay ✓.
4. Regression test: a unit test that fires a streaming thinking+text request at the Anthropic adapter and asserts the `content_block_stop` event for the thinking block contains a signature matching `^vllm-mlx:[0-9a-f]{32}$`. Today this test would fail (signature never emitted); after the fix it must pass.
5. Cross-family non-regression: Gemma-4 (which doesn't emit thinking) still works — signature emission should only fire on blocks where `type == "thinking"`.

## Related

- PR #14 (`6b580ba`) — fix(anthropic): required signature field on thinking content blocks (**non-streaming only**; this bug is the streaming gap)
- PR #33 / commit `312de2f` — the concurrent-prefill deadlock fix that unblocked seeing this bug at all (previously these requests just timed out)
- `docs/testing/2026-04-21-qwen3-35b-a3b-concurrent-heavy-payload-deadlock.md` — the prior deadlock doc; this bug was hidden behind the deadlock until it landed
- `hank-secure-llm`'s `app/src-tauri/src/proxy.rs` — ruled out as the cause. The proxy is a pure passthrough for thinking/text content blocks (verified by tapping the SSE stream through `dogfood_proxy`: 1 text_delta arrives at Claude Code, matching the direct upstream curl)
- Arena `hank_llm_arena/proxy.py:proxy_v1` — also ruled out (raw-bytes passthrough after model rewrite + optional system injection)

## Workaround until fix lands

None clean on the client side — Claude Code's signature check is not configurable. Options with trade-offs:

- Strip the `thinking.type` field from outgoing requests so the model doesn't emit thinking blocks at all (loses reasoning quality; user-visible tok/s speedup from skipping thinking might be mistaken for a fix rather than a workaround).
- Add a proxy-layer streaming transformer that stamps a placeholder `signature` on `content_block_stop` events where `content_block.type == "thinking"`. Works around Claude Code's silent-drop but would cause real signature validation to fail if Claude Code ever adds it — fragile.
- Ship the real fix upstream. Preferred.
