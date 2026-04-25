---
topic: Upstream waybarrios/vllm-mlx fixes for continuous-batching KV cache bugs
sources:
  - https://github.com/waybarrios/vllm-mlx/issues/380
  - https://github.com/waybarrios/vllm-mlx/issues/384
  - https://github.com/waybarrios/vllm-mlx/pull/385
  - https://github.com/waybarrios/vllm-mlx/pull/365
  - https://github.com/waybarrios/vllm-mlx/pull/296
  - https://github.com/waybarrios/vllm-mlx/pull/286
  - https://github.com/waybarrios/vllm-mlx/pull/217
  - /opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/lib/python3.12/site-packages/mlx_lm/models/cache.py
  - /opt/homebrew/Caskroom/miniconda/base/envs/vllm-mlx/lib/python3.12/site-packages/mlx_lm/generate.py
date: 2026-04-21
checkpoint: 20260421-203033-issue-29-kv-cache-lcp-contamination
---

# Upstream waybarrios/vllm-mlx fixes for continuous-batching KV cache bugs

## Findings

### The bug we were chasing (issue #29 = waybarrios#384)

`--continuous-batching` returns degenerate repetitive tokens on request 2+
of a session that loaded a persisted prefix cache. Same mechanism upstream
hit on Gemma 4 (content bleed across concurrent requests), different
surface symptom.

**Root cause (proven by PR #385, commit `9630c5d`, merged 2026-04-21):**
`_trim_cache_offset` in `memory_cache.py` created a shallow wrapper that
shrank `.offset` but kept `.keys`/`.values` pointing at the source's
over-sized arrays. Consumer paths that read `cache.state` directly
(Gemma 4 KV-shared attention layers, Qwen3 kickoff on supersequence
match) saw stale tokens beyond the new offset.

**Fix:** slice the arrays to `new_offset` when returning the trimmed
wrapper. Two hunks — one in `_trim_cache_offset` (plain KVCache branch),
one in `_dequantize_cache` (the `_QuantizedCacheWrapper` equivalent
upstream — for us, a parallel slice after dequantize).

### The amplifier (PR #365, commit `01261c1`, merged 2026-04-17)

Disk-persisted prefix caches had no consistency gate at load. A cache
written by an older fork version with the slice bug could be re-loaded
into a fresh session and contaminate it even if the current code is
fixed. Upstream bumped `_CACHE_PERSIST_VERSION` to 3 and added a
`_compute_model_fingerprint()` helper (SHA over 7 architecture fields).
`load_from_disk` rejects any entry whose version or fingerprint differs
from the current run.

### The deferred 4 (deliberately NOT ported in PR #30)

- `09cc64e`, `b61f57c` (#296) — RotatingKVCache post-restore trim. Our
  `_trim_cache_offset` catches RotatingKVCache via the KVCache branch
  (it has `.offset` and array `.keys`), so the slice fix already covers
  it for #29. Missing bits: Rotating-specific `max_size`/`keep`/`_idx`
  attrs are lost when we convert to plain KVCache. Cosmetic for now.
- `d38b978` (#286) — 3D KV tensor handling in `prefix_cache.py`. Our
  paged/block-aware tier. Not active on the text path that #29 hits.
- `5d93852` (#217) — preserve hybrid recurrent state across blocks.
  Not relevant to the pure-transformer Qwen3-0.6B repro.

### Adjacent drift worth auditing (not in scope for #29)

PRs touching `memory_cache.py` / `scheduler.py` / `prefix_cache.py` in
the 30 days before 2026-04-21:
- PR #373 — `--warm-prompts`
- PR #303 — respect `--chunked-prefill-tokens 0` with memory cache
- PR #295 — GLM-4 parser + think/prefix cache bugs
- PR #221 — preserve prompt checkpoints across chunked prefill resume
- PR #294 — mlx-lm 0.31.x compat in scheduler

## Source Notes

### mlx_lm 0.31.2 KVCache semantics

- `KVCache.offset` tracks filled-positions count. After `update_and_fetch`
  with N tokens: `self.offset += keys.shape[2]`.
- `keys.shape[2]` is the BUFFER size, which grows in `step=256`
  increments. NOT equal to offset when the buffer has spare capacity.
- `state` getter returns `(keys[..., :offset, :], values[..., :offset, :])`
  — BUT has a fast-path: if `offset == keys.shape[2]` (buffer exactly
  filled), it returns the full tensors un-sliced. **This is the exact
  path that leaks stale tokens when `_trim_cache_offset` shrinks offset
  but leaves the buffer untouched.** Our pre-fix code hit this
  fast-path in the supersequence-match scenario: after trim, the wrapper
  has `offset=14` but the source's `keys.shape[2]=94`, so the fast-path
  doesn't fire and `cache.state` slices correctly — HOWEVER, any
  subsequent path that reads the arrays via their shape (not via
  state) still saw all 94 positions. That's how the kickoff prefill
  exposed stale data.
- `save_prompt_cache` serializes `state` (sliced) + `meta_state` +
  class names. On load, `from_state(state, meta_state)` sets offset
  from the loaded tensor shape. So a correctly-saved cache round-trips
  with offset == shape[2]. Stale offset-vs-tensor-size inconsistencies
  on disk came from OLDER fork code that saved in the bad state.

### BatchGenerator 0.31.2 split

`PromptProcessingBatch` (generate.py:1004) and `GenerationBatch` (:1229)
are split classes (different from pre-0.31 single `Batch`). Our fork's
`_install_chunked_prefill` still targets the pre-split API — it
early-returns with a WARN when mlx_lm ≥ 0.31, disabling chunked prefill
and the prompt-cache-save callback. That was an unrelated observation
during this research but worth remembering: chunked prefill is
effectively dead in our fork today. Not why #29 happened, but it is
why the `_prompt_cache_save` callback never fires in our live sessions.

### Why Gemma 4 was the canary (waybarrios#384)

Gemma 4 has KV-shared attention layers that read `cache.state` directly
to implement the sharing — they bypass `update_and_fetch` and the
offset pointer. That's why upstream hit the bug first on Gemma 4's
Hammer eval (22 corrupted outputs / 204 total → 0 / 204 after #385).
Qwen3 models normally use `update_and_fetch` correctly, which is why
our #29 only surfaced after a specific sequence: supersequence match
→ trim-by-N → kickoff with `tokens_to_process=prompt[-1:]` → prefill
writes at offset, but the subsequent step reads the arrays via shape.
