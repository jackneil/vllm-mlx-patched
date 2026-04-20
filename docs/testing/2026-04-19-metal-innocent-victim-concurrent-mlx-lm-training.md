# Metal firmware-detected lockup during `mlx-lm` LoRA training — with concurrent vllm-mlx-patched serving on an M3 Ultra

**Date:** 2026-04-19 (updated with evidence-based rewrite same day)
**Observed by:** hank-llm-lab training runs (incidents on 2026-04-19 and 2026-04-14)
**Crashing processes:** `mlx-lm 0.31.2` training subprocess (python3.14); on 2026-04-19 also one of the `vllm-mlx-patched` serving processes (python3.12) on 3 of 4 incidents
**Mechanism (confirmed by kernel gpuEvent diagnostics):** Apple GPU firmware compute-engine watchdog declared a "firmware-detected lockup" (restart_reason=4, signature 579, guilty_dm=3). Driver reset the whole GPU and discarded every in-flight Metal command buffer system-wide as part of recovery. Every MLX process with a buffer in flight aborts because `mlx::core::gpu::check_error` raises an uncaught `std::runtime_error` on the resulting completion callback.

> **Status note (final):** After two rewrites — the first on internal kernel evidence, the second after web research — this turns out to be a **known upstream issue with a known workaround**. MLX issue [ml-explore/mlx#3267](https://github.com/ml-explore/mlx/issues/3267) ("[BUG] Metal GPU watchdog kills LoRA training when display is active") documents the exact same failure mode on macOS 26.2/26.3.1 Tahoe, on someone else's hardware, with the same `guilty_dm: 3` compute-engine watchdog mechanism. MLX maintainer [@zcbenz](https://github.com/zcbenz) closed it as **`wontfix`** with the comment that the behavior is controlled by the OS and that an MLX-side retry "would be a complicated solution and might not work for larger works." The workaround @zcbenz suggested — and which the reporter confirmed "completely resolved" their crashes — is a single environment variable:
>
> ```
> AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1
> ```
>
> (CDM = Compute Data Master, which matches our `guilty_dm: 3` attribution. This env var relaxes the AGX compute-context-store timeout that trips the firmware watchdog.)
>
> The related issue [ml-explore/mlx#3302](https://github.com/ml-explore/mlx/issues/3302) ("GPU watchdog kills process during long-context SDPA prefill (65K+ keys)") independently confirms the same env var resolves the same class of failure during long-context inference. A proper upstream fix for the long-context case — chunked SDPA via [ml-explore/mlx#3307](https://github.com/ml-explore/mlx/pull/3307) — is in review.
>
> So the action item is no longer "investigate further" but "**set `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` everywhere MLX is invoked**" — on the `vllm-mlx-patched` fleet launch environment, on `mlx_lm.lora` training launches, and arguably as a default in the hank-llm-lab training harness. This doc retains the incident narrative and kernel evidence as a worked example, but the recommended primary mitigation is now that one env var.
>
> Earlier incorrect framings in this doc (since superseded):
> 1. First version: "vllm-mlx is the most likely culprit triggering the Metal recovery, training is innocent victim." Wrong — kernel attributes the lockup to training, not serving.
> 2. Second version: "kernel attribution proves training is the cause." Too strong — see [ml-explore/mlx#3267](https://github.com/ml-explore/mlx/issues/3267): the watchdog fires whenever *any* compute dispatch exceeds the AGX context-store timeout; attribution names the stalled process, not necessarily a buggy one. Both MLX training and Chrome's GPU renderer (see April 14 cluster) can trip it.

## Summary

Between 16:13 and 19:55 local on 2026-04-19, the Apple GPU on a Mac Studio (M3 Ultra, 512 GB unified memory) entered "firmware-detected lockup" four separate times. Each time the kernel driver reset the entire GPU and discarded every in-flight Metal command buffer across every process sharing the GPU. Each recovery took ~12 ms from signal to end.

Every `mlx_lm.lora` training subprocess that was running during any of these recoveries died with:

```
libc++abi: terminating due to uncaught exception of type std::runtime_error:
[METAL] Command buffer execution failed: Discarded (victim of GPU error/recovery)
(00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)
```

On 3 of the 4 incidents a concurrent `vllm-mlx-patched` serving process (python3.12) *also* died with the identical message, at the same microsecond. On the 4th incident the co-victim was `ScreensharingAgent` instead. Neither vllm-mlx nor ScreensharingAgent is the attributed trigger — see [Evidence inventory](#evidence-inventory).

The user-facing error string `kIOGPUCommandBufferCallbackErrorInnocentVictim` is per-process: from each victim's own vantage point its own buffer was discarded by "some other process's" error. But the kernel's GPU event reports attribute every one of today's 4 events to the *training* process (python3.14).

### This isn't new — it started at least by 2026-04-14

Walking the DiagnosticReports folder further, **this phenomenon has been occurring for at least five days**. Across 2026-04-14 (5 days ago) there were **five** python3.14 training crashes with identical `check_error → MTL::CommandBuffer::addCompletedHandler` stacks — three attributed in gpuEvent reports to python3.14 itself, and two attributed to **Google Chrome Helper** (`gpuEvent-Google Chrome He-2026-04-14-*.ips`). On those latter two events, the firmware watchdog fingered Chrome's GPU renderer as the stalling compute dispatch, and the training process was the innocent victim alongside Chrome.

That's a significant correction to the original framing in this doc. The attributed trigger can be **any** compute-intensive Metal client — sometimes MLX training, sometimes a browser renderer. Today it happened to attribute to MLX training on all 4 events; on 2026-04-14 the attribution alternated. The `log show --last 30d` query earlier showed zero events only because the unified-log retention window is shorter than five days on this box — the .ips DiagnosticReports go back further and tell the real story.

Summary of all 12 MLX-callback-path crashes on record (stack: `mlx::core::gpu::check_error`):

| Day | python3.14 (training) crashes | python3.12 (serving) crashes | gpuEvent attributions |
|---|---|---|---|
| 2026-04-14 | 5 | 0 (1 unrelated `end_encoding` segfault) | 3× python3.14, **2× Google Chrome Helper** |
| 2026-04-18 | 0 | 0 (1 unrelated `__assert_rtn` abort) | none |
| 2026-04-19 | 4 | 3 | 4× python3.14 |

Two conclusions:

1. The baseline is **not zero**. The firmware watchdog has been tripping for at least 5 days. Today's 4 events are an unusually dense cluster, but the class of failure is a week old.
2. Attribution is not a clean "who caused it" signal. It's more like "whose dispatch was in flight when the watchdog fired." On April 14 the compute-bound Metal process swung between MLX training and Chrome; today it's been MLX training every time.

## Environment

- **Hardware:** Apple M3 Ultra (Mac15,14), 512 GB unified memory
- **OS:** macOS 26.3 (25D125) / Darwin 25.3.0
- **Python:** 3.14 on training side, 3.12 on serving side
- **Training stack:** `mlx-lm==0.31.2` (pip)
- **Serving stack:** `vllm-mlx-patched` @ commit `52136ee` (this repo, pinned MLX per `UPSTREAM_PIN.md`)
- **Serving fleet at crash time (10 processes, ~151 GB combined RSS):**

  | Port | Model | RSS |
  |---|---|---|
  | 8002 | Qwen3.5-122B-A10B-4bit | 65 GB |
  | 8000 | Qwen3.6-35B-A3B-4bit | 20 GB |
  | 8005 | Qwen3.5-35B-A3B-4bit | 20 GB |
  | 8006 | Qwen3-VL-30B-A3B-Instruct-4bit | 18 GB |
  | 8004 | gemma-4-26b-a4b-it-4bit | 16 GB |
  | 8007 | mlx-vlm medgemma-1.5-4b-it-4bit | 4 GB |
  | 8010 | mlx-audio Qwen3-ASR-1.7B-8bit | 3 GB |
  | 8009 | mlx-audio Qwen3-TTS-12Hz-0.6B | 3 GB |
  | 8008 | mlx-audio whisper-large-v3-turbo | 2 GB |
  | 8001 | Qwen3-0.6B-8bit | 1.3 GB |
  | | **Total** | **~151 GB** |

  All six `vllm-mlx serve` processes launched with `--continuous-batching`. All 10 processes share a single Metal device.

## Evidence inventory

Two independent kernel/driver artifacts exist for every one of today's 4 incidents. These are not derived from the training or serving processes' own logs — they come from `IOGPUFamily` in the kernel and from `ReportCrash`.

### kernel GPU-event reports (system-level)

Four files in `/Library/Logs/DiagnosticReports/` (owned by `root:_analyticsusers`, 847 bytes each), written by `ReportCrash` in response to kernel `GPURestartReport` events:

| File | captureTime (local) |
|---|---|
| `gpuEvent-python3.14-2026-04-19-161322.ips` | 16:13:22 |
| `gpuEvent-python3.14-2026-04-19-161951.ips` | 16:19:51 |
| `gpuEvent-python3.14-2026-04-19-181122.ips` | 18:11:22 |
| `gpuEvent-python3.14-2026-04-19-195508.ips` | 19:55:08 |

All four have identical structure:

```json
{
  "bug_type": "284",
  "process_name": "python3.14",         // kernel attribution = training
  "analysis": {
    "signature": 579,
    "restart_reason": 4,
    "restart_reason_desc": "firmware-detected lockup",
    "guilty_dm": 3,                     // DM 3 = compute data-master
    "fw_cl_state": {"slot0":1,"slot1":1,"slot2":1},
    "fw_3d_state": {"slot0":0,"slot1":0,"slot2":0},
    "fw_ta_state": {"slot0":0,"slot1":0},
    "command_buffer_trace_id": …        // unique per incident
  }
}
```

All four crashes share `signature=579`, `restart_reason=4`, `guilty_dm=3`, and all three compute slots active with the 3D and tile-accelerator engines idle. That's a fingerprint of a **long-running compute kernel that the GPU firmware gave up on**.

The filename convention (`gpuEvent-<process_name>-<timestamp>.ips`) is the kernel's attribution. All four name `python3.14` — the training process. **None** name `python3.12` (serving) or any other process.

### per-process crash reports (user-level)

Seven files in `~/Library/Logs/DiagnosticReports/`, all dated 2026-04-19:

| File | pid | coalition | captureTime (local) |
|---|---|---|---|
| `python3.14-2026-04-19-161322.000.ips` | 81499 | `com.googlecode.iterm2` | 16:13:22.4648 |
| `python3.12-2026-04-19-161323.ips`     | 80783 | `com.hank.arena`        | 16:13:22.4649 |
| `python3.14-2026-04-19-161951.000.ips` | 83574 | `com.googlecode.iterm2` | 16:19:51.3081 |
| `python3.12-2026-04-19-161952.ips`     | 80784 | `com.hank.arena`        | 16:19:51.3082 |
| `python3.14-2026-04-19-181123.ips`     | 49981 | `com.googlecode.iterm2` | 18:11:22.7889 |
| `python3.14-2026-04-19-195508.000.ips` | 7431  | `com.googlecode.iterm2` | 19:55:08.4499 |
| `python3.12-2026-04-19-195509.ips`     | 98643 | `com.hank.arena`        | 19:55:08.4499 |

The training side (python3.14) is in the iTerm coalition (LoRA launched from a terminal). The serving side (python3.12) is in `com.hank.arena` (the lab's vllm-mlx fleet manager).

Every one of these 7 .ips files has the **same faulting-thread stack** on queue `com.Metal.CompletionQueueDispatch`:

```
mlx::core::gpu::check_error(MTL::CommandBuffer*)
  ← block in MTL::CommandBuffer::addCompletedHandler(std::__1::function<void(MTL::CommandBuffer*)> const&)
    ← MTLDispatchListApply
      ← -[_MTLCommandBuffer didCompleteWithStartTime:endTime:error:]
        ← -[IOGPUMetalCommandBuffer didCompleteWithStartTime:endTime:error:]
          ← -[_MTLCommandQueue commandBufferDidComplete:startTime:completionTime:error:]
            ← IOGPUNotificationQueueDispatchAvailableCompletionNotifications
              ← libdispatch serial-drain
```

Translation: the Metal driver invoked the command-buffer completion callback with an error code; MLX's `check_error` raised an uncaught `std::runtime_error`; `std::terminate` → `abort` → `SIGABRT`. **Same source line, same symbol, every time, on every victim.**

### kernel + Metal log correlation (example: incident 4, 19:55:08)

```
19:55:08.433 kernel IOGPUScheduler::signalHardwareError(eRestartRequest): GPURestartSignaled stampIdx=61 type=1
19:55:08.433 kernel IOGPUScheduler::hardware_error_interrupt: setting channel 61 restart type to 1
19:55:08.433 kernel GPURestartBegin stampIdx=61
19:55:08.445 kernel GPURestartEnd                                      (≈ 12 ms of recovery)
19:55:08.445 ReportCrash Handling GPU event version 101, 920 bytes
19:55:08.445 ReportCrash GPURestartReport: event with 4 keys
19:55:08.445 python3.14[7431]  IOGPUMetalError: <private>
19:55:08.445 python3.14[7431]  (Metal) … Discarded (victim of GPU error/recovery) (…InnocentVictim)
19:55:08.445 python3.12[98643] IOGPUMetalError: <private>
19:55:08.445 python3.12[98643] (Metal) … Discarded (victim of GPU error/recovery) (…InnocentVictim)
```

Channel/stamp numbers across today's incidents: `60, 60, 96, 61`. The 16:13 and 16:19 events landed on the **same channel** (60), 6 minutes apart. The `IOGPUMetalError: <private>` strings carry the actual per-process error detail but are **redacted** by macOS unified logging defaults; see [Digging deeper](#digging-deeper-next-time) for unmasking instructions.

### baseline rate

```sh
/usr/bin/log show --last 30d --predicate \
  'eventMessage CONTAINS "GPURestartSignaled" OR eventMessage CONTAINS "signalHardwareError"'
```

Returns only the 4 events from 2026-04-19 — but only because the unified-log retention window on this box doesn't actually cover 30 days. The `.ips` DiagnosticReports in `/Library/Logs/DiagnosticReports/` go back further and show this is not a one-day event:

- **2026-04-14** — 3 gpuEvent reports attributing firmware lockup to python3.14, plus 2 gpuEvent reports attributing to **Google Chrome Helper** (all five with identical `restart_reason_desc: "firmware-detected lockup"`, same signature 579). Five MLX-callback-path crashes on the user side. Chrome was the attributed trigger on 2 of 5.
- **2026-04-15 through 2026-04-18** — zero gpuEvent reports in DiagnosticReports. No MLX-callback crashes. (Two unrelated crashes exist for other causes: an `EXC_BAD_ACCESS` segfault in `mlx::core::metal::Device::end_encoding` on 2026-04-14, and an `__assert_rtn` abort on 2026-04-18 — neither is this failure mode.)
- **2026-04-19 (today)** — 4 gpuEvent reports, all attributing to python3.14. 7 MLX-callback-path crashes across training + serving.

The firmware-lockup events cluster on days with heavy GPU activity and happen zero times on other days. The 2026-04-14 cluster (5 events in ~3 hours, 12:59 to 15:53) parallels the 2026-04-19 cluster (4 events in ~4 hours, 16:13 to 19:55). In between, four quiet days.

## Incident detail

### Incident 1 — 2026-04-19 16:13:22 local

- **Training:** python3.14 pid 81499 (mlx-lm LoRA subprocess)
- **Co-victim:** python3.12 pid 80783 (`com.hank.arena`, one of the vllm-mlx serves)
- **Kernel event:** stampIdx=60 on channel 60, `firmware-detected lockup`

### Incident 2 — 2026-04-19 16:19:51 local

- **Training:** python3.14 pid 83574
- **Co-victim:** python3.12 pid 80784 (consecutive pid to incident 1's co-victim — same coalition, re-spawned after incident 1)
- **Kernel event:** stampIdx=60 on channel 60 again, 6 minutes after incident 1 on the same channel
- **Note:** Channel 60 re-fault strongly suggests the same compute dispatch / same model / same gradient kernel was re-submitted after incident 1 and hit the firmware watchdog again.

### Incident 3 — 2026-04-19 18:11:22 local — PHI redaction LoRA

- **Run dir:** `runs/2026-04-19-220740-phi_redaction-spike/` (hank-llm-lab)
- **Model:** `mlx-community/Llama-3.2-1B-Instruct-4bit`, LoRA rank 8, batch_size 2, max_seq_length 4096
- **Peak training RSS:** 42.1 GB. Crashed at ~iter 100, immediately after val-loss computation.
- **Training:** python3.14 pid 49981
- **Co-victim:** `ScreensharingAgent[11796]` — **not** a vllm-mlx process. The screen-sharing daemon happened to have a Metal buffer in flight at the recovery instant and caught the discard.
- **Kernel event:** stampIdx=96 on channel 96, `firmware-detected lockup`
- **Concurrent lab workload:** `nano_eval_phi.py` hitting OpenAI's API (network-only, zero local Metal work).

### Incident 4 — 2026-04-19 19:55:08 local — medgemma shootout

- **Run dir:** `runs/2026-04-19-234244-note_classification-shootout_medgemma_1.5_4b/`
- **Model:** `mlx-community/medgemma-1.5-4b-it-4bit`, LoRA rank 8, batch_size 2, max_seq_length 4096
- **Peak training RSS:** 25.9 GB. Crashed at ~iter 180.
- **Training:** python3.14 pid 7431
- **Co-victim:** python3.12 pid 98643 (vllm-mlx serve)
- **Kernel event:** stampIdx=61 on channel 61
- **Concurrent lab workload:** none other than vllm-mlx serving. Shootout was sequential; lab dashboard is pure HTML with no GPU load.

### Saved adapters are clean

For both incidents 3 and 4 the last-saved adapter checkpoint is byte-identical to the canonical `adapters.safetensors`:

```
$ shasum -a 256 runs/.../adapter/adapters.safetensors runs/.../adapter/0000100_adapters.safetensors
c383264b1df62bf5c788f58620e3816ff96db3375a2abc62aa785ca4f93d7cd9  adapters.safetensors
c383264b1df62bf5c788f58620e3816ff96db3375a2abc62aa785ca4f93d7cd9  0000100_adapters.safetensors
```

The log shows exactly one "Saved adapter weights" line followed by more normal iters, then the Metal abort — so the canonical file *is* the last-snapshot file. No torn write, no corruption.

## Root cause

With the kernel's GPU-event reports in hand, the actual sequence is:

1. The training process submits a Metal compute dispatch (LoRA gradient / matmul / optimizer step).
2. The GPU firmware's watchdog for data-master 3 (compute) expects the dispatch to make progress within some internal timeout.
3. On each of today's 4 incidents, the dispatch **did not make progress fast enough**, and firmware declared `restart_reason=4` / signature 579 / "firmware-detected lockup".
4. The kernel `IOGPUScheduler` raised `signalHardwareError(eRestartRequest)`. The full GPU engine was quiesced and all in-flight command buffers across every process sharing the GPU were aborted. Recovery took ~12 ms.
5. Every victim process (training, plus whichever other Metal clients had buffers in flight at that instant) received the `kIOGPUCommandBufferCallbackErrorInnocentVictim` error on its completion handler. MLX's `mlx::core::gpu::check_error` raises `std::runtime_error` on that error code, and MLX does not catch it — so every affected MLX process aborts.

The kernel's own attribution (`process_name: "python3.14"` in the gpuEvent report) names **the training process** as the associated process for all four events. The vllm-mlx serving process that died alongside training on 3 of 4 incidents **was not the attributed trigger** — it was simply another Metal client that had a buffer in flight at the recovery instant.

### Why the original hypothesis was backwards

The first cut of this doc hypothesized that vllm-mlx-patched's large working set and continuous-batching throughput were triggering the Metal recovery, with training as the innocent victim. That's not what the driver says. Kernel attribution today is unambiguous: training is the named process for every one of today's events; vllm-mlx is a co-victim on 3 of 4 and **not a co-victim at all on the 4th** (screen-sharing daemon took that slot). So vllm-mlx is neither the attributed trigger nor a consistent companion of the event.

### Attribution is "who stalled first," not "who caused it"

The 2026-04-14 history complicates the "training caused the lockup" reading. On that day the firmware watchdog attributed 2 of 5 MLX-callback crashes to **Google Chrome Helper**, not to training — and those 2 still took down the training process as a co-victim. On 3 of 5 it attributed to training; today it attributed to training on all 4. Same box, same user, same OS build.

A plausible reading: the watchdog fires when *any* compute dispatch exceeds the firmware's internal timeout, and the gpuEvent's `process_name` field names whichever process owned the stalling dispatch at that instant. That's a report of symptom, not of cause. Whatever's driving the box into the regime where long compute stalls become common is the actual upstream factor. Concurrent heavy MLX load is one known way to get there; Chrome GPU rendering is apparently another.

### What's proven vs. what's speculation

Per the evidence and the confirming upstream issues [ml-explore/mlx#3267](https://github.com/ml-explore/mlx/issues/3267) and [ml-explore/mlx#3302](https://github.com/ml-explore/mlx/issues/3302):

**Proven:**
- The fault is the Apple GPU AGX firmware's Compute Data Master (CDM) context-store-timeout watchdog. The watchdog fires when a GPU compute context store exceeds some internal deadline (estimated ~5 seconds from #3302's analysis of long-context SDPA dispatches).
- When it fires, the driver resets all in-flight Metal command buffers system-wide and attributes the lockup to whichever process owned the stalling dispatch.
- Every MLX process with a buffer in flight aborts because `mlx::core::gpu::check_error` treats the resulting completion-handler error as uncaught `std::runtime_error`. This is the same code path MLX takes for any Metal error — it's not InnocentVictim-specific.
- The env var `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` relaxes that timeout and resolves the crashes. Confirmed by the original MLX#3267 reporter ("Completely resolved"), confirmed for long-context SDPA in MLX#3302, and independently confirmed by a Qwen3.6-35B llama.cpp user at 69K context in [ggml-org/llama.cpp#20141](https://github.com/ggml-org/llama.cpp/issues/20141).

**Speculation (consistent with evidence but not directly proven for today's incidents):**
- Why the timeout kept firing on our box specifically, 4 times in 4 hours: most plausibly working-set pressure + scheduler contention between 10 concurrent Metal processes (~151 GB combined RSS), which stretches compute-context-store time past the deadline. But we don't have Instruments Metal System Trace timelines to confirm the mechanism. The env var fix doesn't depend on confirming this — it relaxes the timeout regardless of why it was firing.

**Notable:** the 2026-04-14 events attributed to Google Chrome Helper likely share root cause with the well-known [macOS Tahoe 26 + Electron GPU slowdown bug](https://9to5mac.com/2025/10/11/macos-26-tahoe-electron-gpu-slowdown-bug-fix-rollout/) — an unrelated Electron/macOS 26 interaction that Apple patched in macOS 26.2 as a workaround and that Electron fixed client-side for Slack/Discord/VS Code around the same time. Since this box is on 26.3 the Apple-side workaround is present, but it's possible older Chrome/Chromium builds or extension pages still occasionally trip the CDM watchdog on the GPU renderer path. That's separate from our MLX problem but explains the co-incidence of Chrome-attributed events alongside MLX ones.

### Parallel: the existing Gemma 3 sliding-window Metal timeout in README

`README.md` already documents a related class of failure:

> Gemma 3's default `sliding_window=1024` limits context to ~10K tokens on Apple Silicon (Metal GPU timeout at higher context). To enable longer context (up to ~50K tokens), patch mlx-vlm: …

That's the same Metal GPU firmware watchdog mechanism, triggered by a natively long attention kernel over a large KV cache. The fix there is a workload-side cap (`RotatingKVCache(max_size=sliding_window)` keeps the attention kernel bounded). The training crashes reported here are the same watchdog tripping, just on a compute dispatch whose stall is caused by system-level GPU pressure rather than kernel length.

The README section is worth reading as a precedent — the repo already ships a mitigation for *this exact firmware timeout phenomenon*, just for a different input pattern.

## Primary mitigation (landed upstream as wontfix + env-var workaround)

**Set `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` in every process that invokes MLX.**

That's the @zcbenz-confirmed workaround in [ml-explore/mlx#3267](https://github.com/ml-explore/mlx/issues/3267). It relaxes the AGX compute-context-store timeout so the firmware watchdog no longer fires on long dispatches. Independently confirmed for long-context SDPA inference in [ml-explore/mlx#3302](https://github.com/ml-explore/mlx/issues/3302) and for Qwen3.6-35B at 69K context in [ggml-org/llama.cpp#20141](https://github.com/ggml-org/llama.cpp/issues/20141). A third-party Python package "MetalGuard" (referenced in the MLX#3267 thread) sets this env var at import time as a drop-in fix — vllm-mlx-patched could do the same thing at import (see Secondary mitigation #2 below).

Concrete places to set it:

1. **`vllm-mlx-patched` serve launchers.** If the fleet is started via a launchd plist, systemd unit, shell wrapper, or `launchctl setenv`, add `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` to the environment. Anywhere a `vllm-mlx serve …` command is spawned.
2. **`mlx_lm.lora` training launches.** Same, in the training harness (hank-llm-lab) wrapper that spawns the LoRA subprocess.
3. **Any ad-hoc `python -m mlx_lm …` / `mlx_embeddings` / `mlx_vlm` / `mlx_audio` invocation on this box.**

One cost: the env var disables the AGX compute-context-store watchdog entirely (per #3302's comment: "it's a system-wide env var that disables the watchdog entirely — not appropriate as a permanent solution"). The watchdog exists to kill runaway compute kernels that would otherwise hang indefinitely. On a dedicated ML host with trusted workloads the risk is low; on a multi-tenant or interactive machine you're trading safety against reliability. Our M3 Ultra is dedicated, so the trade is fine.

## Secondary mitigations (defense in depth)

Even with the env var, other failure modes exist (OOM, memory pressure, hard kernel crashes unrelated to the watchdog). So:

1. **Document the concurrency envelope in README.** State that running ≥4 MoE serves under continuous-batching on a single Apple Silicon box, concurrent with `mlx_lm.lora` training, requires `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` set in both the serving and training environments. Link this doc + MLX#3267 + MLX#3302.
2. **Hint at the env var at process start.** Before importing MLX, vllm-mlx-patched could `os.environ.setdefault("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1")`. Setting it in-process only works if it's set *before* the Metal driver is initialized; setting it via a shell wrapper or launchd plist is safer. But having it in code as a default is the belt-and-suspenders option. Same pattern as MetalGuard.
3. **Consider vendoring the chunked-SDPA fix for long-context inference.** [ml-explore/mlx#3307](https://github.com/ml-explore/mlx/pull/3307) is the *proper* fix for long-context SDPA (independent of the env-var workaround) — it breaks the attention dispatch into ≤65K-key chunks so each one stays under the watchdog deadline. Worth pinning once merged, since the existing README workaround for Gemma 3 (`sliding_window=1024`) becomes unnecessary. Until it merges, keep the sliding-window doc.
4. **Expose a per-process working-set cap knob (nice-to-have).** `engine_core.py:163`, `batched.py:341`, `mllm_batch_generator.py:370` already query `max_recommended_working_set_size`. Exposing a server-level flag that limits a single serve to `weights_size + N * continuous_batch_transient_size` would protect against a different failure mode (true OOM under co-tenant pressure), orthogonal to the watchdog issue.

## For `mlx-lm` / upstream MLX

The "retry on InnocentVictim" approach I originally proposed was already considered by @zcbenz in #3267 and rejected as impractical:

> "we could try to re-submit the command buffer but it would be a complicated solution and might not work for larger works. So I'm marking this as won't fix because we don't really have a fix."

So don't file a new issue asking for that. The upstream path forward is the chunked-SDPA PR [ml-explore/mlx#3307](https://github.com/ml-explore/mlx/pull/3307), which targets the root cause for the long-context-prefill subset (by ensuring no single SDPA dispatch exceeds the watchdog deadline). A parallel effort to split long LoRA-training compute dispatches into smaller chunks would help the training-side case, but I don't see an upstream issue yet — possibly worth filing.

The most practical training-side improvement is still:

1. **Resume-on-crash in the training harness.** Even with the env var, hard crashes from other causes will happen. Check for a saved adapter checkpoint on startup and resume from it automatically.

## Already-landed mitigations (hank-llm-lab)

1. `shootout.py` runs each model in its own subprocess, so a Metal abort in one shootout model does not take down the other models. Validated during incident 4: Qwen3-0.6B and Llama-1B completed normally, medgemma's iter-100 adapter was evaluable via the fallback path.
2. `save_every=100` is set in the LoRA config; for 4B+ models it will drop to `save_every=50` so at most 50 iters of work are lost per recovery event.
3. `scripts/eval_samples.py` plus a hand-written `summary.json` lets a crashed run's last-saved adapter be evaluated as a fallback. Used successfully for medgemma after incident 4 — got us the verdict despite the abort.

## Reproducer (approximate)

1. Launch vllm-mlx-patched serving at least 4 of the following concurrently with `--continuous-batching`:
   - `Qwen3.5-122B-A10B-4bit` (biggest single contributor)
   - `Qwen3.6-35B-A3B-4bit`
   - `Qwen3.5-35B-A3B-4bit`
   - `gemma-4-26b-a4b-it-4bit`
2. Drive sustained inference traffic on at least one of them (a few hundred tokens/sec, mixed prompt lengths).
3. Concurrently start `mlx_lm.lora` training of any 1-4B model with `batch_size: 2`, `max_seq_length: 4096`, `iterations: 500`.
4. Watch for a kernel `GPURestartSignaled` event within ~10-30 minutes:
   ```
   /usr/bin/log stream --predicate 'eventMessage CONTAINS "GPURestartSignaled"'
   ```

Empirical rate so far: 4 events in ~4 hours today (16:13, 16:19, 18:11, 19:55), 5 events in ~3 hours on 2026-04-14 (12:59, 13:30, 14:25, 15:00, 15:53). Zero events on 2026-04-15 through 2026-04-18. Need a larger sample to estimate probability as a function of concurrent fleet size and GPU utilization.

## TL;DR for maintainers

**This is a known, duplicate issue upstream.** On this M3 Ultra Mac Studio the Apple GPU AGX compute-engine firmware watchdog (Compute Data Master context-store timeout) has tripped at least 9 times across 2026-04-14 and 2026-04-19 with identical kernel signatures, discarding all in-flight Metal command buffers system-wide each time and hard-killing every MLX process that had a buffer in flight (all 12 matching crashes abort at `mlx::core::gpu::check_error`). The attributed-trigger process rotates between MLX training (most of our events) and Chrome Helper (2 events on April 14, which align with the macOS Tahoe 26 Electron GPU bug).

MLX [issue #3267](https://github.com/ml-explore/mlx/issues/3267) tracks this exact failure mode and is closed `wontfix` by the MLX maintainer, with the confirmed workaround being one env var:

```
AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1
```

Set that env var in every process that invokes MLX (vllm-mlx-patched serves, mlx-lm training, mlx-vlm/audio/embeddings), and the firmware watchdog stops firing. The long-context-inference variant ([MLX#3302](https://github.com/ml-explore/mlx/issues/3302)) has the same workaround and a proper fix in review at [MLX#3307](https://github.com/ml-explore/mlx/pull/3307) (chunked SDPA).

Secondary:
- **vllm-mlx-patched:** document the env var in README and optionally `os.environ.setdefault` it at import time; once #3307 merges, consider pinning the MLX version that includes it and dropping the Gemma 3 `sliding_window` workaround.
- **hank-llm-lab:** set the env var in the training harness env; add resume-on-crash for defense in depth; keep `save_every ≤ 50` for 4B+ models.

## References

### Today's incidents (2026-04-19)

- System gpuEvent reports (all attributed to python3.14 / training):
  - `/Library/Logs/DiagnosticReports/gpuEvent-python3.14-2026-04-19-161322.ips`
  - `/Library/Logs/DiagnosticReports/gpuEvent-python3.14-2026-04-19-161951.ips`
  - `/Library/Logs/DiagnosticReports/gpuEvent-python3.14-2026-04-19-181122.ips`
  - `/Library/Logs/DiagnosticReports/gpuEvent-python3.14-2026-04-19-195508.ips`
- Per-process crashes (user):
  - `~/Library/Logs/DiagnosticReports/python3.14-2026-04-19-161322.000.ips` (training, incident 1)
  - `~/Library/Logs/DiagnosticReports/python3.12-2026-04-19-161323.ips` (vllm-mlx serve co-victim, incident 1)
  - `~/Library/Logs/DiagnosticReports/python3.14-2026-04-19-161951.000.ips` (training, incident 2)
  - `~/Library/Logs/DiagnosticReports/python3.12-2026-04-19-161952.ips` (vllm-mlx serve co-victim, incident 2)
  - `~/Library/Logs/DiagnosticReports/python3.14-2026-04-19-181123.ips` (training, incident 3 — co-victim was ScreensharingAgent)
  - `~/Library/Logs/DiagnosticReports/python3.14-2026-04-19-195508.000.ips` (training, incident 4)
  - `~/Library/Logs/DiagnosticReports/python3.12-2026-04-19-195509.ips` (vllm-mlx serve co-victim, incident 4)

### 2026-04-14 precedent cluster

- System gpuEvent reports:
  - `/Library/Logs/DiagnosticReports/gpuEvent-python3.14-2026-04-14-125928.ips` (attributed python3.14)
  - `/Library/Logs/DiagnosticReports/gpuEvent-Google Chrome He-2026-04-14-133022.ips` (attributed **Google Chrome Helper**)
  - `/Library/Logs/DiagnosticReports/gpuEvent-python3.14-2026-04-14-142530.ips` (attributed python3.14)
  - `/Library/Logs/DiagnosticReports/gpuEvent-python3.14-2026-04-14-150030.ips` (attributed python3.14)
  - `/Library/Logs/DiagnosticReports/gpuEvent-Google Chrome He-2026-04-14-155328.ips` (attributed **Google Chrome Helper**)
- Per-process crashes (user, all training, all `check_error` stack):
  - `~/Library/Logs/DiagnosticReports/python3.14-2026-04-14-125929.ips`
  - `~/Library/Logs/DiagnosticReports/python3.14-2026-04-14-133023.ips`
  - `~/Library/Logs/DiagnosticReports/python3.14-2026-04-14-142531.ips`
  - `~/Library/Logs/DiagnosticReports/python3.14-2026-04-14-150032.ips`
  - `~/Library/Logs/DiagnosticReports/python3.14-2026-04-14-155329.ips`

### Upstream references (the reason this doc ends with a one-line fix)

- **[ml-explore/mlx#3267](https://github.com/ml-explore/mlx/issues/3267)** — "[BUG] Metal GPU watchdog kills LoRA training when display is active." Same mechanism, same user-visible error, macOS 26.2/26.3.1 Tahoe. Closed `wontfix` by @zcbenz with `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` as the confirmed workaround. **Read this first if re-visiting this issue.**
- **[ml-explore/mlx#3302](https://github.com/ml-explore/mlx/issues/3302)** — "GPU watchdog kills process during long-context SDPA prefill (65K+ keys)." Same class, different trigger (long SDPA dispatch rather than compute contention). Env var is the same workaround. Real fix in review at #3307.
- **[ml-explore/mlx#3307](https://github.com/ml-explore/mlx/pull/3307)** — "Chunked full-attention SDPA for long key sequences." The proper fix for the long-context-inference subset. Validated on M3 Ultra (our hardware class) at 65K/128K/262K contexts. Not yet merged as of this writing.
- **[ggml-org/llama.cpp#20141](https://github.com/ggml-org/llama.cpp/issues/20141)** — llama.cpp users hitting the same error family (InnocentVictim + ImpactingInteractivity) on Mac M4 Pro / M1 Max / M2 Max Tahoe 26.3+, also resolved by `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1`. Independent confirmation beyond the MLX repo.
- **[electron/electron#48311](https://github.com/electron/electron/issues/48311)** + [9to5Mac coverage](https://9to5mac.com/2025/10/11/macos-26-tahoe-electron-gpu-slowdown-bug-fix-rollout/) — background on the macOS Tahoe 26 + Electron GPU slowdown bug, which is the most plausible explanation for the 2026-04-14 Chrome-Helper attributions in our history. Patched in macOS 26.2 + Electron client-side, though older Chromium builds may still trigger CDM-watchdog events on the renderer path.

### Local artifacts

- Both crashed training runs from 2026-04-19 preserved with logs + saved adapters at
  `hank-llm-lab/runs/2026-04-19-220740-phi_redaction-spike/` and
  `hank-llm-lab/runs/2026-04-19-234244-note_classification-shootout_medgemma_1.5_4b/`
- README precedent for the same watchdog class: `README.md` Gemma 3 `sliding_window` section
- Code sites already querying `max_recommended_working_set_size`:
  - `vllm_mlx/engine_core.py:163`
  - `vllm_mlx/engine/batched.py:341`
  - `vllm_mlx/mllm_batch_generator.py:370`
- MLX source (upstream, `ml-explore/mlx`): `mlx::core::gpu::check_error(MTL::CommandBuffer*)` in `mlx/backend/metal/metal.cpp` — the symbol at the top of every faulting stack. The installed `mlx==0.31.1` wheel ships only compiled `libmlx.dylib`; no patch is needed there (env var suffices).
