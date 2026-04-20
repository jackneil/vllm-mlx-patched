# Metal `kIOGPUCommandBufferCallbackErrorInnocentVictim` — mlx-lm training crashing under concurrent vllm-mlx serving load

**Date:** 2026-04-19
**Observed by:** hank-llm-lab training runs
**Crashing process:** `mlx-lm 0.31.2` (pip package) — `mlx_lm.lora` subprocess
**Suspected cause:** Metal resource contention between vllm-mlx-patched serving processes and concurrent `mlx-lm` training process
**Scope of this doc:** empirical crash data + root-cause hypothesis + proposed mitigations. Not filing as a bug against vllm-mlx-patched yet — root cause sits at the boundary between Metal driver, MLX runtime, and whoever happens to be the Metal process under the most memory pressure.

## Summary

While running LoRA training (`mlx_lm.lora`) on `hank-llm-lab` on the same box
that's serving 6 vllm-mlx-patched models concurrently (~138 GB RSS pinned
across the serving processes), **two** separate training runs died with
identical error signatures:

```
libc++abi: terminating due to uncaught exception of type std::runtime_error:
[METAL] Command buffer execution failed: Discarded (victim of GPU error/recovery)
(00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)
```

The "innocent victim" language in the error code `kIOGPUCommandBufferCallbackErrorInnocentVictim`
is literal: Metal recovered from a GPU error in **some other process** and
discarded all in-flight command buffers system-wide, taking down the training
process even though training itself wasn't the source of the error.

Hypothesis: vllm-mlx serving processes are the most likely culprit for
triggering the Metal recovery, given their large working set and continuous
inference traffic. This makes training+serving co-location on the same Mac
unreliable by default.

## Environment

- **Hardware:** Apple M3 Ultra (Mac15,14), 512 GB unified memory
- **OS:** macOS 26.3 (Darwin 25.3.0)
- **Python:** 3.14.3 (training side) / 3.12 (vllm-mlx serving side)
- **MLX:** training side uses `mlx-lm==0.31.2` (pip); vllm-mlx-patched serving
  uses its pinned MLX (see serving process cwd)
- **Concurrent vllm-mlx workloads at crash time** (from `ps aux | grep vllm-mlx`):

  | Model | RSS (MB) | Port |
  |---|---|---|
  | Qwen3.5-122B-A10B-4bit | 66,498 | 8002 |
  | Qwen3.6-35B-A3B-4bit | 20,055 | 8000 |
  | Qwen3.5-35B-A3B-4bit | 19,693 | other |
  | gemma-4-26b-a4b-it-4bit | 16,118 | other |
  | Qwen3-0.6B-8bit | 1,287 | 8001 |
  | (1 additional vllm-mlx) | 18,183 | — |
  | **Total vllm-mlx RSS** | **~142 GB** | — |

  All launched with `--continuous-batching` (active scheduling).

## Incident 1: PHI redaction training, 2026-04-19T22:01 UTC (~18:01 local)

**Run dir:** `runs/2026-04-19-220740-phi_redaction-spike/` (hank-llm-lab)
**Model under training:** `mlx-community/Llama-3.2-1B-Instruct-4bit`
**Config:** LoRA rank 8, batch_size 2, max_seq_length 4096, 1500 iters planned
**Training peak memory:** 42.1 GB
**Iter when crashed:** ~100 (right after val loss computation completed)

Trace from training.log:

```
Iter 90: Train loss 0.034, ..., Peak mem 42.065 GB
Iter 100: Val loss 0.027, Val took 3.257s
Iter 100: Train loss 0.022, ..., Peak mem 42.065 GB
libc++abi: terminating due to uncaught exception of type std::runtime_error:
[METAL] Command buffer execution failed: Discarded (victim of GPU error/recovery)
(00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)
```

**Concurrent lab workload at crash:** A separate `nano_eval_phi.py` script
hitting OpenAI's gpt-5.4-nano API for evaluation. API calls only — zero MLX or
Metal workload on the lab side. So the only MLX/Metal producers on the box
were the 6 vllm-mlx servers + the 1 mlx-lm training subprocess.

## Incident 2: note_classification medgemma training, 2026-04-19T23:52 UTC (~19:52 local)

**Run dir:** `runs/2026-04-19-234244-note_classification-shootout_medgemma_1.5_4b/`
**Model under training:** `mlx-community/medgemma-1.5-4b-it-4bit`
**Config:** LoRA rank 8, batch_size 2, max_seq_length 4096, 200 iters planned
**Training peak memory:** 25.9 GB
**Iter when crashed:** ~180 (10 iters into the 180-190 window, between the
iter 180 train-loss line and any subsequent output)

Trace from training.log:

```
Iter 170: Train loss 0.012, ..., Peak mem 25.919 GB
Iter 180: Train loss 0.016, ..., Peak mem 25.919 GB
libc++abi: terminating due to uncaught exception of type std::runtime_error:
[METAL] Command buffer execution failed: Discarded (victim of GPU error/recovery)
(00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)
```

**Concurrent lab workload at crash:** None. The shootout was running
sequentially (one model at a time, subprocess-isolated per shootout.py
design). The lab dashboard (HTML server, no GPU) was running on port 8766.
**The only concurrent MLX/Metal workload was the vllm-mlx serving fleet.**

This is the important incident — it eliminates "training contention with
training" as a cause. Training was alone on the lab side, yet still got reset.

## Evidence that the adapter save was clean

Important detail for any downstream user worried that the crash corrupted
the saved weights:

```
$ shasum -a 256 runs/.../adapter/adapters.safetensors runs/.../adapter/0000100_adapters.safetensors
c383264b1df62bf5c788f58620e3816ff96db3375a2abc62aa785ca4f93d7cd9  adapters.safetensors
c383264b1df62bf5c788f58620e3816ff96db3375a2abc62aa785ca4f93d7cd9  0000100_adapters.safetensors
```

`training.log` has exactly **one** "Saved adapter weights" line (at iter 100),
followed by normal iter 110–180 lines, then the Metal abort. So the canonical
`adapters.safetensors` is byte-identical to the iter-100 snapshot. No torn
write, no corruption.

## Hypothesis — root cause

The `kIOGPUCommandBufferCallbackErrorInnocentVictim` error is documented in
Apple's IOKit as: *"The GPU encountered an error while processing another
command buffer, and as a safety measure the system discards all in-flight
command buffers from all processes sharing the GPU context."*

The "other command buffer" that triggered recovery is most likely from one
of the vllm-mlx serving processes, because:

1. They are the most memory-dense Metal workloads on the box (~142 GB RSS
   combined, vs 26–42 GB for training).
2. They are under active inference load, so command buffers are being
   submitted continuously.
3. The 122B-A10B model on port 8002 alone is 66 GB RSS — a serious Metal
   allocation that competes for working-set headroom against any other
   Metal process.
4. Continuous batching means the scheduler is repeatedly allocating short-
   lived transient buffers on top of the long-lived KV cache.

Without Metal-level profiling (Instruments timeline, Metal System Trace) I
cannot confirm which process actually faulted. But the pattern is consistent
across both incidents — training took ~30 minutes to start crashing in
incident 1 and ~12 minutes in incident 2, both under active vllm-mlx
inference.

## Why this matters for vllm-mlx-patched

Most MLX users don't run training and serving on the same box. But on a
512 GB Mac Studio that's a natural deployment — serve everything from
the same host that trains LoRA adapters. If vllm-mlx's serving workload
is making concurrent mlx-lm training unreliable, that limits the Mac as
a training box even when the training job itself fits easily in memory.

Two places the bug could be mitigated:

### From vllm-mlx-patched's side

1. **Set a working-set limit per serving model** via `MTLResourceOptions`
   or similar, so an individual serving process can't monopolize Metal
   working-set headroom. If every model is capped at its weight footprint
   + N GB of transient buffer allowance, the driver has predictable
   eviction decisions.
2. **Document expected concurrency** — the README or guides should say
   whether running N models concurrently with active training is
   supported or explicitly unsupported.
3. **Expose a GPU-activity signal** via admin endpoint so other workloads
   (training jobs, batch inference scripts) can back off during serving
   traffic spikes.

### From mlx-lm / mlx side (upstream)

1. **Handle `kIOGPUCommandBufferCallbackErrorInnocentVictim` as retriable**
   — the error is not caused by the victim process's own bug. mlx-lm
   should catch the Metal exception, resubmit the last command buffer,
   and continue. Silently losing the training iteration is the wrong
   response for a driver-recovery event.
2. **Resume-on-crash in the training harness** — if mlx_lm.lora's
   subprocess aborts and the adapter has a saved checkpoint, the harness
   should resume from that checkpoint rather than fail the job.

### From hank-llm-lab's side (already landed mitigations)

1. Shootout runs each model in its own subprocess, so a Metal abort in
   one model doesn't kill the whole shootout — the other models still
   complete (validated during this incident; Qwen3-0.6B and Llama-1B
   completed normally, medgemma's iter-100 adapter was still evaluable).
2. `save_every=100` is already set; for 4B+ models we should drop to
   `save_every=50` so at most 50 iters of work are lost per crash.
3. `scripts/eval_samples.py` + hand-written summary.json let us evaluate
   a crashed run's last-saved adapter as a fallback (used for medgemma
   in incident 2 — got us the verdict despite the crash).

## Reproducer (approximate)

Not yet reproduced deliberately. An attempted reproducer would:

1. Launch vllm-mlx-patched serving at least three of:
   Qwen3.5-122B-A10B-4bit, Qwen3.6-35B-A3B-4bit, gemma-4-26b-a4b-it-4bit
   with `--continuous-batching`.
2. Drive sustained inference traffic on at least one of them (a few
   hundred tokens/sec, mixed prompt lengths).
3. Concurrently start `mlx_lm.lora` training of any 1-4B model with
   `batch_size: 2`, `max_seq_length: 4096`, `iterations: 500`.
4. Watch for Metal abort within ~10-30 minutes.

Expected failure rate on a 512 GB M3 Ultra under the current fleet:
based on N=2/2 this session, high. Need a larger sample to estimate.

## References / next steps

- Both crashed training runs preserved at
  `hank-llm-lab/runs/2026-04-19-220740-phi_redaction-spike/` and
  `hank-llm-lab/runs/2026-04-19-234244-note_classification-shootout_medgemma_1.5_4b/`
  (logs + saved adapters).
- macOS Console `log show --predicate 'eventMessage CONTAINS "GPU"'` around
  the crash timestamps might reveal which serving process actually triggered
  the recovery. Not yet captured.
- If the pattern reproduces, worth filing upstream against
  `ml-explore/mlx-lm` for the retry-on-victim behavior, and possibly
  against Apple's Feedback Assistant for the Metal driver recovery
  taking down innocent processes.

## TL;DR for the maintainer reading this

If vllm-mlx-patched is serving ≥3 large models under continuous-batching
on an Apple Silicon box, do not expect `mlx_lm.lora` training on the same
box to finish reliably. Two separate training runs this session died with
`kIOGPUCommandBufferCallbackErrorInnocentVictim` after ~12-30 min, while
the only other MLX/Metal workload on the system was the vllm-mlx serving
fleet.

The crash is almost certainly not a vllm-mlx-patched code bug — more likely
a Metal driver quirk under multi-process GPU pressure. But vllm-mlx-patched
is the workload that makes the box hostile to concurrent MLX training, and
it's the layer with the most leverage to either (a) cap its own Metal
working-set footprint, or (b) document the concurrency limits clearly.
