# Effort and Thinking Budget

vllm-mlx supports five provider-dialect ways to ask for a reasoning-token budget. All of them normalize to a single internal value (`thinking_token_budget`) via the resolver in `vllm_mlx/api/effort.py`. This page documents every knob, their precedence, and what clients can expect back.

## The knobs

| Where sent | Field | Example |
|---|---|---|
| `/v1/chat/completions` + `/v1/messages` | top-level `thinking_token_budget` (int) | `{"thinking_token_budget": 512}` |
| `/v1/messages` | `thinking.type` + `budget_tokens` | `{"thinking": {"type": "enabled", "budget_tokens": 256}}` |
| `/v1/messages` | `thinking.type: "disabled"` | `{"thinking": {"type": "disabled"}}` |
| `/v1/messages` | `thinking.type: "adaptive"` | `{"thinking": {"type": "adaptive"}}` |
| `/v1/messages` | `output_config.effort` | `{"output_config": {"effort": "high"}}` |
| `/v1/chat/completions` | `reasoning_effort` | `{"reasoning_effort": "high"}` |

## Effort level → budget

| Effort | budget | max_tokens_floor (hint) |
|---|---|---|
| `low` / `minimal` | 512 | 2048 |
| `medium` / `normal` | 2048 | 4096 |
| `high` | 8192 | 16384 |
| `xhigh` | 16384 | 32768 |
| `max` | min(context_window ÷ 2, 65536) | 2× budget |

`max_tokens_floor` is a **client hint**, returned via the `x-thinking-budget-max-tokens-floor` response header. Clients should ensure `max_tokens >= max_tokens_floor` to avoid mid-thought truncation. The server does not enforce this — it's up to the caller.

## Precedence

The resolver evaluates signals in this order; first match wins:

1. Top-level `thinking_token_budget` (int, including 0)
2. `thinking.type: "disabled"` → budget = 0
3. `thinking.type: "enabled"` + `budget_tokens` → budget = `budget_tokens`
4. `thinking.type: "adaptive"` → budget = None (natural behavior)
5. `output_config.effort` (table lookup)
6. `reasoning_effort` (table lookup)
7. Nothing → budget = None

## Response headers

Every response that went through the resolver emits these headers:

| Header | Values | Meaning |
|---|---|---|
| `x-thinking-budget-applied` | `true` / `false` | Whether the logits processor attached AND the model family supports enforcement. Absent when no budget was requested. |
| `x-thinking-budget-resolved` | int as string, or `"none"` | The resolver's output — what the server actually tried to enforce. |
| `x-thinking-budget-source` | `top_level` \| `anthropic_thinking_enabled` \| `anthropic_thinking_disabled` \| `anthropic_thinking_adaptive` \| `output_config_effort` \| `reasoning_effort` \| `default` | Which input field won precedence. |
| `x-thinking-budget-max-tokens-floor` | int as string | Recommended minimum `max_tokens` for this effort level. Absent when source is `default` or budget is 0. |
| `x-thinking-budget-noop-reason` | `parser_not_configured` \| `tokenizer_encode_failed` \| `multi_token_delimiter` \| `mllm_path` \| `simple_engine` | Machine-readable reason the processor didn't attach. Only present when `applied=false`. See **Noop reasons** below. |
| `x-thinking-budget-ceiling` | int as string | Configured `--max-thinking-token-budget` value. Emitted on every response when the flag is set, regardless of whether Layer 2 fired. Absent when the flag is unset. |
| `x-thinking-budget-clamped-to` | int as string | Post-clamp budget. Emitted ONLY when Layer 2 actually clamped. Matches `x-thinking-budget-resolved`. |
| `x-thinking-budget-clamp-skipped` | `mllm_path` \| `simple_engine` \| `parser_not_configured` \| `engine-no-op` | Emitted when ceiling is set but the engine/parser combination cannot enforce the clamp. Matches `noop-reason` vocabulary. |
| `x-thinking-qwen3-auto-disabled` | `"true"` | Emitted when Layer 1 (Qwen3 surgical first-turn-no-think) fired on this request — `chat_template_kwargs.enable_thinking=False` was injected automatically. |

### Server-side ceiling behavior (Layer 2)

The `--max-thinking-token-budget N` flag applies AFTER the resolver precedence chain above. Clamping only raises `clamped_from`/`clamped_to` headers when the resolver output actually exceeded the ceiling; budgets below the ceiling pass through unchanged. The ceiling NEVER raises `budget=0` (client explicitly disabled thinking) — clamps are lower-bound only. When the engine/parser can't enforce the clamp (MLLM path, SimpleEngine, missing reasoning parser), `clamp-skipped` is emitted with the specific reason so operators can tell clamp-fired-truthfully from clamp-was-asked-for-but-couldn't-deliver.

## Per-family enforceability

Not every model family can enforce the budget. The logits processor biases the `</think>` token's logit when the budget is hit — this only works if `</think>` tokenizes to a single token AND the engine runs logits processors at all.

| Family | Thinking protocol | Enforceable? | Typical noop reason |
|---|---|---|---|
| Qwen3 / Qwen3.5 / Qwen3.6 | `<think>…</think>` (single token) | ✓ | — |
| DeepSeek-R1 | `<think>…</think>` | ✓ | — |
| Generic models with `think_parser` | `<think>…</think>` | ✓ | — |
| Gemma 4 | channel protocol, also auto-detected as MLLM | ✗ — loud noop | `mllm_path` |
| GPT-OSS / Harmony | channel protocol (multi-token) | ✗ — loud noop | `multi_token_delimiter` |

When enforcement fails, the request still succeeds — the server generates normally and reports `applied=false` + the reason header. Nothing silent.

## Noop reasons

| Value | What it means | How to fix |
|---|---|---|
| `parser_not_configured` | Server started without `--reasoning-parser`. | Restart the server with e.g. `--reasoning-parser qwen3`. |
| `tokenizer_encode_failed` | The parser's `start_token` / `end_tokens` raised when encoded, or returned an empty list. | Usually means the parser is mismatched with the tokenizer. Check parser/model compatibility. |
| `multi_token_delimiter` | Delimiters exist but encode to >1 token — the force-bias logic only hits single-token `</think>`. | Inherent to the model family. No fix; choose a family with single-token delimiters or accept unbounded thinking. |
| `mllm_path` | Request went through the multimodal path, which skips logits processors entirely. | Inherent to the path. For text-only requests you can force the LLM path if the model supports it; otherwise accept unbounded thinking. |
| `simple_engine` | Server is running in SimpleEngine mode, which doesn't support logits processors. | Restart the server with `--continuous-batching`. This is the most common misconfiguration. |

## `AnthropicUsage.thinking_tokens` (vllm-mlx extension)

Non-streaming `/v1/messages` responses include a `usage.thinking_tokens` field carrying the token count of the reasoning portion alone (Anthropic's public API only surfaces a single `output_tokens`). `None` (excluded from JSON) when no reasoning was produced. Best-effort — computed by tokenizing the extracted reasoning text with the reasoning parser's tokenizer; a tokenizer failure logs a WARNING and leaves the field as `None`. Streaming does not yet emit this field.

## Examples

### Claude Code's `--effort` flag (Anthropic path)

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "content-type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-35B-A3B-4bit",
    "messages": [{"role": "user", "content": "Think step by step: 13*17"}],
    "max_tokens": 4000,
    "output_config": {"effort": "high"}
  }' -i | head -20
```

Response headers include:
```
x-thinking-budget-applied: true
x-thinking-budget-resolved: 8192
x-thinking-budget-source: output_config_effort
x-thinking-budget-max-tokens-floor: 16384
```

### OpenAI o1-style `reasoning_effort`

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-35B-A3B-4bit",
    "messages": [{"role": "user", "content": "Solve: 13*17"}],
    "max_tokens": 4000,
    "reasoning_effort": "medium"
  }' -i | head -20
```

### Force no thinking on any path

```json
{"thinking_token_budget": 0}
```
or (Anthropic only)
```json
{"thinking": {"type": "disabled"}}
```

## Sizing rule

When setting a non-zero budget, ensure `max_tokens >= thinking_token_budget + content_headroom` (roughly `budget + 1000` for typical chat responses). If `max_tokens` is too tight, the model hits `max_tokens` while still inside `<think>` and the caller sees `finish_reason: "length"` with an empty or truncated content block.

Clients can trust `x-thinking-budget-max-tokens-floor` as a floor — it is always at least `budget + content_headroom`.

The server logs a WARNING with prefix `[thinking-budget-resolver]` when `max_tokens < max_tokens_floor` at request ingress, naming the effort level, the floor, and the received `max_tokens` — grep operator logs for this marker when debugging truncated output on `effort=high` requests.
