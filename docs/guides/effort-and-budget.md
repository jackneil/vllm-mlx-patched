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
| `x-thinking-budget-noop-reason` | string (e.g., `channel_protocol_not_supported`) | Explains why `applied=false`. Only present when applied=false. |

## Per-family enforceability

Not every model family can enforce the budget. The logits processor biases the `</think>` token's logit when the budget is hit — this only works if `</think>` tokenizes to a single token.

| Family | Thinking protocol | Enforceable? |
|---|---|---|
| Qwen3 / Qwen3.5 / Qwen3.6 | `<think>…</think>` (single token) | ✓ |
| DeepSeek-R1 | `<think>…</think>` | ✓ |
| Generic models with `think_parser` | `<think>…</think>` | ✓ |
| Gemma 4 | channel protocol (multi-token) | ✗ — loud noop |
| GPT-OSS / Harmony | channel protocol | ✗ — loud noop |

When a model family cannot enforce, the server attaches the processor anyway (so `applied=false` is a legitimate signal rather than a silent failure), emits `x-thinking-budget-noop-reason`, and generates normally. The request succeeds with no cap applied.

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
