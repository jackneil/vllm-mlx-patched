# Lessons from past sessions

[1x] ALWAYS verify external library API signatures (help(), source, docs) before writing code in plans. The bench-compile plan used `temp=0.0` and `mx.metal.clear_cache()` — both wrong. Subagents faithfully implement whatever the plan says.

[1x] ALWAYS check locally cached models (`~/.cache/huggingface/hub/`) before running benchmarks. Don't assume or try to download models — use what's already available.

[1x] ALWAYS check the user's existing benchmark infrastructure (Hank Arena at localhost:6969, bench_runner.py) before proposing new benchmark approaches. The user already has speed benchmarks — test against those, not in isolation.

[1x] NEVER use jargon without a plain-English explanation first. "Prefill" means nothing to the user — say "the wait before tokens start flowing" then use the technical term.

[1x] ALWAYS set short timeouts (5-10s) on curl/network status checks. A hanging curl locks up the session.

[1x] ALWAYS use Firecrawl for web searching, not raw curl or WebSearch.

[1x] NEVER leave uncommitted changes or stale worktrees behind. User said "You should not be leaving uncommitted stash or stale worktrees." Before declaring any work complete, check `git status` and handle everything — commit it, gitignore it, or explain it.

[1x] When user says "look at the fix on main" or "look at what they merged upstream", actually go to GitHub and read the upstream code/PRs — don't just read a local summary doc. The user had to redirect twice before I checked the actual mlx-lm PRs on GitHub.

[1x] Think about fork maintenance. When working on a fork, always consider: has upstream already merged similar work? Can we reduce our patch delta? The user asked "does this allow us to get closer to main" — a question about fork health, not just features.

[1x] ALWAYS verify model chat template behavior by reading the actual Jinja template (from HuggingFace or the model files), not by assuming. Initially set SUPPORTS_NATIVE_TOOL_FORMAT=True for Gemma 4, then discovered via web research that Gemma 4 uses `tool_responses` on assistant messages, not `role="tool"`. Had to change to False.

[1x] Before adopting patterns from a reference implementation, check that its dependencies exist in our project. The mlx-lm Gemma 4 parser uses the `regex` module (recursive patterns). We don't have it — had to rewrite with stdlib `re` and a manual balanced-brace walker.

[1x] Benchmark warmup must actually be warm. The first bench-compile baseline run showed 553 tok/s while runs 2-3 showed 4000+. One warmup generation isn't enough — the first measurement run is still cold. Consider discarding the first measurement or doing 2+ warmup runs.

[1x] Apple Silicon decode speed is memory-bandwidth-bound, not compute-bound. mx.compile helps prefill (compute-bound, +33%) but not decode (bandwidth-bound, +0%). The things that move decode tok/s are: speculative decoding, async eval pipelining, or faster memory hardware. Don't waste time optimizing compute for decode.

[1x] MLX ecosystem context (April 2026): MLX decode is 1.4-1.8x faster than raw llama.cpp on Apple Silicon. Prefill is SLOWER than llama.cpp (no flash attention). MoE models show the biggest MLX advantage (up to 3x). Ollama switched to MLX backend in v0.19. M5 Neural Accelerators give 4x TTFT speedup. See docs/research/performance-optimization-research.md for full analysis and sources.

[1x] The user trusts technical decisions but wants honest results. Don't oversell optimizations. When bench-compile showed +0.3% decode, report that honestly — don't spin prefill improvement as the headline. The user's benchmark (Arena tok/s) measures decode, and that's what matters to them.
