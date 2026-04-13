# Upstream Pin for vllm-mlx-patched

This is a **hard fork** of vllm-mlx, not a patch overlay. The canonical upstream
is https://github.com/waybarrios/vllm-mlx (see `git remote -v`). This file
documents the upstream commit the patched fork is rebased against and the
arena-critical invariants that must survive a rebase.

## Current upstream pin

- **Upstream commit:** `b4fa03080ce18280ac9916d19a690a13aa45f904` (Merge pull request #232 from sjswerdloff/fix/streaming-tool-call-content-leak)
- **Rebased on:** `2026-04-11`
- **Rebased by:** Jack Neil

## Arena-critical invariants

These are things the `hank-llm-arena` arena's `models_process.py` depends on.
If a rebase breaks any of them, the arena's Gemma 4 or Qwen3 tool calling will
silently degrade to raw-text output.

1. **`@ToolParserManager.register_module("gemma4")`** must fire at import time in
   `vllm_mlx/tool_parsers/gemma4_tool_parser.py`. The registration must succeed
   before the CLI argparse runs `ToolParserManager.list_registered()`.

2. **`@ToolParserManager.register_module(["hermes", "nous", "qwen3_coder"])`** must
   register the `qwen3_coder` alias under HermesToolParser in
   `vllm_mlx/tool_parsers/hermes_tool_parser.py`. The arena's
   `_TOOL_PARSER_PATTERNS` expects this exact string.

3. **`ToolParserManager.list_registered()`** must return a list of strings that
   includes at minimum: `gemma4`, `qwen`, `qwen3`, `qwen3_coder`. Adding more
   parsers is fine; removing or renaming any of these breaks the arena.

4. **`--tool-call-parser` argparse action** in `vllm_mlx/cli.py` must accept any
   registered parser name as a valid choice. The argparse choices list must be
   driven dynamically by `ToolParserManager.list_registered()` (NOT hardcoded).
   Tests in `tests/test_cli_argparse.py` pin this.

### Added 2026-04-12: reasoning-parser streaming integration

5. **`ReasoningParser` base class exposes these properties** (`vllm_mlx/reasoning/base.py`):
   - `start_token: str` (default `"<think>"`)
   - `end_tokens: list[str]` (default `["</think>"]`)
   - `channel_strip_prefix: str | None` (default `None`)

   These are read verbatim by `_stream_anthropic_messages` in `vllm_mlx/server.py`
   behind a `hasattr` precondition to configure `StreamingThinkRouter`. If any
   property is renamed, the server logs an error and falls back to default
   router behavior. Pin: these three exact names with those exact defaults.
   `tests/test_reasoning_parser_properties.py::TestRebaseBreakageSentinel` fires
   in CI if any property name drifts.

6. **`Gemma4ReasoningParser` overrides** `end_tokens` as `["<channel|>", "<|channel>response"]`
   and `channel_strip_prefix` as `"thought\n"` (`vllm_mlx/reasoning/gemma4_parser.py`).
   Note: the legacy singular `end_token` property is RETAINED because
   `BaseThinkingReasoningParser` still declares it abstractmethod and uses it
   in `extract_reasoning_streaming`. Do not remove without untangling the parent.

7. **`StreamingThinkRouter.__init__` accepts** `start_token: str`,
   `end_tokens: Iterable[str]`, `channel_strip_prefix: str | None` kwargs with
   backward-compatible defaults (`vllm_mlx/api/utils.py`). Rejects prefix
   collisions in `end_tokens` at construction. Uses an integer counter
   `_strip_remaining` for channel-strip state (not buffer mutation). Pin:
   the kwarg names, the prefix-collision invariant, and the counter-based
   (non-buffer-mutating) design.

8. **`MLLMScheduler.add_request` precomputes `num_prompt_tokens`** via
   `self.processor.tokenizer.encode(prompt)` in `vllm_mlx/mllm_scheduler.py`.
   Continuous-batching MLLM requests flow through this path — without it,
   `message_delta.usage.input_tokens` leaks as 0 to Anthropic clients.
   `MLLM.stream_chat` has a matching precompute in `vllm_mlx/models/mllm.py`
   for the non-batched MLLM path. Both prefer a truthy chunk-reported value
   when present.

## Rebase checklist

When merging upstream changes into this fork:

1. [ ] `git fetch upstream && git log upstream/main..HEAD --oneline` to see fork-local commits
2. [ ] `git merge-base HEAD upstream/main` to confirm the current pin
3. [ ] Attempt rebase: `git rebase upstream/main`
4. [ ] Run `pytest tests/test_gemma4_tool_parser.py tests/test_tool_parsers.py tests/test_cli_argparse.py` — ALL must pass
5. [ ] Verify `gemma4` is in `--tool-call-parser` help:
       `vllm-mlx serve --help 2>&1 | grep -A2 tool-call-parser`
6. [ ] Update the "Current upstream pin" section above with the new SHA and date
7. [ ] Commit the UPSTREAM_PIN.md update as part of the rebase merge commit

## If a rebase breaks an invariant

Fall back to these recovery steps in order:

1. **Gemma 4 parser disappeared.** Check if upstream added their own Gemma 4
   parser under a different name (e.g., `gemma`, `google_gemma`). If so, update
   the arena's `_TOOL_PARSER_PATTERNS` to point at the new name AND add a
   compatibility alias `@ToolParserManager.register_module("gemma4")` back in a
   fork-local file.

2. **`list_registered()` was renamed.** Update both `vllm_mlx/cli.py` (the
   dynamic choices call) and `hank-llm-arena/tests/test_models_process.py` (the
   drift test).

3. **`--tool-call-parser` was renamed or removed.** Update the arena's
   `_tool_parser_flags` helper to use the new flag name. Add a compatibility
   shim in `vllm_mlx/cli.py` that accepts both names during a transition.

4. **`qwen3_coder` alias was moved out of HermesToolParser.** Update the
   `_TOOL_PARSER_PATTERNS` comment in `hank-llm-arena/models_process.py` to
   point at the new location.
