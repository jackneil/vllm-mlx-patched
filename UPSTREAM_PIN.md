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
