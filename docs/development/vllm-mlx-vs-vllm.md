# vllm-mlx vs upstream vLLM — What this is, what it isn't, and why

This document answers: "What's the relationship between this project and `vllm-project/vllm`?" It's reference for new contributors and for future rebase/alignment decisions.

## Short answer

`vllm-mlx` is **not a fork of `vllm-project/vllm`.** It is an **independent reimplementation** of vLLM's external API using Apple's MLX framework under the hood. The two codebases share design vocabulary (SamplingParams, continuous batching, paged KV cache, reasoning parsers) but no source code.

This specific repo — `vllm-mlx-patched` — is a hard fork of `waybarrios/vllm-mlx` with arena-critical patches. See [`UPSTREAM_PIN.md`](../../UPSTREAM_PIN.md).

## Why two projects exist

vLLM was designed for NVIDIA GPUs and uses CUDA-only kernels like FlashAttention and PagedAttention. It has since added support for AMD (ROCm), TPUs, and CPUs, but **not Apple Silicon MLX natively**. MLX is Apple's research framework built around unified memory and Metal kernels — a different primitive than what vLLM's architecture assumes.

Rather than retrofit vLLM's CUDA-centric internals onto MLX, the community built `vllm-mlx`: a standalone project that reimplements vLLM's user-facing contract (server, API, scheduler, batching) on top of MLX + `mlx-lm` + `mlx-vlm` + `mlx-audio` + `mlx-embeddings`.

**Who maintains what:**
- `vllm-project/vllm` — core vLLM team, NVIDIA-adjacent, 77k+ stars.
- `waybarrios/vllm-mlx` — community project, Apple Silicon only.
- `vllm-mlx-patched` (this repo) — our hard fork with arena-specific patches that upstream hasn't absorbed yet.

## Concrete differences

| | vllm-project/vllm | vllm-mlx (this repo) |
|---|---|---|
| **Hardware** | NVIDIA (primary), AMD, TPU, CPU | Apple Silicon M1/M2/M3/M4 only |
| **Backend** | PyTorch + CUDA/ROCm kernels | MLX + Metal |
| **Codebase relationship** | Upstream | **Independent source tree**. Not a patch overlay. |
| **`import vllm`** | Required | **Optional**. This repo runs standalone; `vllm>=0.4.0` is in `[project.optional-dependencies]` only. |
| **Core classes** | `vllm.SamplingParams`, `vllm.AsyncLLMEngine` | `vllm_mlx.SamplingParams`, `vllm_mlx.engine.*` — same public shape, separate impls. |
| **Scheduler** | vLLM V1 engine w/ GPU model runner | `vllm_mlx/scheduler.py` wrapping `mlx_lm.generate.BatchGenerator` |
| **Attention** | Custom PagedAttention CUDA kernel | MLX built-in attention + `vllm_mlx/paged_cache.py` for cache management |
| **Logits processors** | Batch-wide with `update_state(BatchUpdate)` / `apply(logits)` | Per-request `Callable[[mx.array, mx.array], mx.array]` (mlx_lm's contract) |
| **Tensor parallelism** | Yes (multi-GPU) | No (single-device, but benefits from Apple Silicon's unified memory) |
| **OpenAI API** | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` | Same |
| **Anthropic API** | No | **Yes** — `/v1/messages` is native (vllm-mlx addition for Claude Code / OpenCode) |
| **MCP tool calling** | No first-class support | **Yes** — `vllm_mlx/mcp/` |
| **Audio models** | Limited | **Yes** — mlx-audio integration for TTS/STT |
| **Plugin hook** | Extensible via platform plugins | Declares `vllm.platform_plugins` entry point so vLLM auto-detects and hands off to MLX on Macs |

## What's shared (vocabulary, not code)

Both projects converge on the same solution ideas, independently implemented:

- **Continuous batching** — prefill + decode in the same scheduler, different LoC but same concept.
- **Paged KV cache with prefix sharing** — both have it. vllm-mlx's is at `vllm_mlx/paged_cache.py`.
- **Reasoning-parser registry** — vllm-mlx reimplements this at `vllm_mlx/reasoning/` matching vLLM's `ReasoningParserManager` public API. Property names (`start_token`, `end_tokens`, `channel_strip_prefix`) are pinned in [`UPSTREAM_PIN.md`](../../UPSTREAM_PIN.md) invariant #5.
- **Tool-parser registry** — same pattern. Pinned as invariants #1–#4.
- **OpenAI-compatible HTTP server** — same external contract; FastAPI in both cases.
- **`SamplingParams` shape** — we match field names (`max_tokens`, `temperature`, `top_p`, `stop`, `stop_token_ids`, recently `thinking_token_budget`) so clients are portable.

## Capability parity — what works on each side

### Text generation
- **vLLM:** Deeply optimized for throughput at scale. Tensor parallelism, speculative decoding across devices, chunked prefill, prefix caching, structured output (guidance, outlines), custom kernels.
- **vllm-mlx:** Unified-memory benefits (no PCIe crossing for KV cache), MLX-optimized inference paths, good single-device throughput. SpecPrefill, MLX compile, paged cache, prefix caching. Structured output via `response_format` injection.

### Multimodal (VLM)
- **vLLM:** Strong support for LLaVA-class models and a growing list of VLMs. Uses its own multimodal preprocessor.
- **vllm-mlx:** Via `mlx-vlm`. Covers Qwen-VL, Gemma 4 VLM, and others. Continuous batching for MLLM is in `vllm_mlx/mllm_batch_generator.py`.

### Audio
- **vLLM:** Not a core focus.
- **vllm-mlx:** First-class TTS/STT via `mlx-audio`. Native voices for 9+ languages. `vllm_mlx/audio/` module.

### Embeddings
- **vLLM:** Yes, via embedding models loaded like any other.
- **vllm-mlx:** Yes, via `mlx-embeddings`. OpenAI-compatible `/v1/embeddings`.

### Reasoning / thinking
- **vLLM:** Reasoning parsers for DeepSeek-R1, Qwen3. `thinking_token_budget` merged 2026-03-24 (PR #20859) as a LogitsProcessor. Follow-up PR #37112 adds `thinking_budget_message` for graceful wrap-up.
- **vllm-mlx (this repo, feat/thinking-token-budget branch):** Ports PR #20859 + PR #37112's message feature. Text-only path initially; MLLM path follow-up.

### Tool calling
- **vLLM:** Extensive tool-parser registry (Hermes, Mistral, Qwen, etc.).
- **vllm-mlx:** Ported + extended — `ToolParserManager` with `gemma4`, `qwen`, `qwen3`, `qwen3_coder` aliases. Arena invariants #1–#4 pin these.

### Tensor parallelism / distributed inference
- **vLLM:** Multi-GPU, multi-node. This is the reason vLLM exists at scale.
- **vllm-mlx:** Single-device. A Mac Studio with 192GB unified memory is the ceiling.

### Hardware targets
- **vLLM:** Any datacenter hardware that can run CUDA/ROCm.
- **vllm-mlx:** Any Apple Silicon Mac. Runs on laptops (M1 Air, 8GB) to Mac Studios (M3 Ultra, 512GB) with model-appropriate sizing.

## Why we're using vllm-mlx instead of vLLM

This is **the arena's production inference server** for Apple Silicon hardware. vLLM proper won't run on a Mac at any meaningful performance — there are no CUDA GPUs and no ROCm, and the PyTorch-CPU path is orders of magnitude slower than MLX's Metal-accelerated path.

Concrete reasons:
1. **Apple Silicon performance.** MLX's Metal kernels + unified memory give 1.4–1.8× decode throughput over `llama.cpp` on Apple Silicon, and 2–3× on MoE models. vLLM-on-CPU is not competitive.
2. **Unified memory architecture.** On Apple Silicon, GPU and CPU share RAM. Large KV caches don't cross a bus — they just are. Paged caching still matters for prefix sharing, but the constant factor is different and favorable.
3. **Audio + vision built in.** mlx-audio and mlx-vlm are first-class dependencies. vLLM audio support is nascent.
4. **Anthropic Messages API.** We talk to Claude Code and OpenCode through `/v1/messages`. vLLM doesn't ship this natively.
5. **Single-Mac deployment model.** Our production topology is one Mac per arena slot, not a Kubernetes cluster of H100s. vLLM's distributed features are cost we don't need.

## How we stay aligned with upstream vLLM

Since the code isn't shared, alignment is done at three layers:

1. **API surface parity.** When vLLM adds a new field to `SamplingParams` or `ChatCompletionRequest`, we match the field name and default exactly. A client that works against `vllm-project/vllm` must work unchanged against `vllm-mlx`. Example: the current `thinking_token_budget` work is a name-exact port of vLLM PR #20859.
2. **Algorithm ports.** When vLLM merges an algorithm (logits processor, scheduling heuristic, cache policy), we port it by reading the upstream diff and adapting to MLX primitives. We don't "merge" commits — we translate.
3. **Registry naming.** Reasoning parsers, tool parsers, and platform plugins use the same names vLLM uses (`qwen3`, `deepseek_r1`, `hermes`, etc.). Arena routing depends on these names — they're pinned in `UPSTREAM_PIN.md`.

**Rebase cadence:**
- Against `waybarrios/vllm-mlx` (the fork's actual upstream): periodic, driven by meaningful improvements or arena-impacting fixes. See `UPSTREAM_PIN.md` for the current pin SHA and rebase checklist.
- Against `vllm-project/vllm`: **no direct rebase possible** (different code). Instead, when upstream ships a feature we want, we open a tracked PR here that ports it. We name API surfaces identically so a future rebase from `waybarrios/vllm-mlx` that absorbs upstream behaves as an incremental refinement rather than a breaking change.

## Decision rubric — which project should a change land in?

- **Apple Silicon performance fix / MLX optimization** → this repo.
- **CUDA kernel / tensor-parallel scheduling** → upstream vLLM (not our concern).
- **New API field or public contract** → **both**, coordinated. Open a PR here mirroring the upstream name and defaults. If upstream hasn't landed yet, we can lead, but we MUST take upstream's name when it merges.
- **New reasoning parser / tool parser** → this repo, matching upstream's registered names.
- **Bug in an MLX-specific code path** → this repo.
- **Bug in shared vocabulary (API shape mismatch)** → fix here; file the matching issue/PR against upstream vLLM if the bug affects portability.

## Future alignment opportunities

- **Upstream vllm-mlx may at some point merge a subset of upstream vLLM's v1 engine** (the platform-plugin hook exists for this). If/when that happens, this repo's `vllm_mlx/plugin.py` and `vllm_platform.py` are the integration point. Until then, we run standalone.
- **Apple's `mlx-lm` could absorb more of what `vllm_mlx/scheduler.py` does** (continuous batching at the mlx-lm layer). The `_bg_kwargs` filter (`vllm_mlx/scheduler.py:42-63`) exists because mlx-lm's `BatchGenerator` signature moves between releases — we're already tolerant of that churn.
- **`thinking_token_budget` on the MLLM path** is the first follow-up to the current feature branch: plumb `logits_processors` through `MLLMBatchGenerator._step()` so the feature works for VLMs too.

## References

- Upstream vLLM: https://github.com/vllm-project/vllm
- Upstream vllm-mlx: https://github.com/waybarrios/vllm-mlx
- Upstream vLLM PR #20859 (thinking_token_budget, merged 2026-03-24): https://github.com/vllm-project/vllm/pull/20859
- Upstream vLLM PR #37112 (thinking_budget_message, in review): https://github.com/vllm-project/vllm/pull/37112
- MLX: https://github.com/ml-explore/mlx
- mlx-lm: https://github.com/ml-explore/mlx-lm
- `UPSTREAM_PIN.md` — rebase invariants for this fork.
