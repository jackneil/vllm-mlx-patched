"""Tests for vllm-mlx CLI argparse wiring.

These tests pin the binding between `ToolParserManager.list_registered()`
and the `--tool-call-parser` argparse choices list. If a refactor hardcodes
the list or misses a new parser, these tests fail loudly.

Uses subprocess to invoke `vllm-mlx serve --help` because the parser is
built inside `main()` and not exposed as a standalone function.
"""

import subprocess
import sys

import pytest

from vllm_mlx.tool_parsers import ToolParserManager


@pytest.fixture(scope="module")
def serve_help_output():
    """Run `vllm-mlx serve --help` once and cache the output."""
    result = subprocess.run(
        [sys.executable, "-m", "vllm_mlx.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"serve --help failed: {result.stderr}"
    return result.stdout


def _parse_choices_from_help(help_text: str, option: str) -> set[str]:
    """Extract the {choice1,choice2,...} set for a given option from --help output."""
    import re
    # argparse formats choices as: --option {a,b,c}
    # Match the first occurrence after the option name
    pattern = rf"{re.escape(option)}\s+\{{([^}}]+)\}}"
    match = re.search(pattern, help_text)
    if not match:
        return set()
    return {c.strip() for c in match.group(1).split(",")}


def test_tool_call_parser_choices_match_registry(serve_help_output):
    """--tool-call-parser choices must be ToolParserManager.list_registered()."""
    cli_choices = _parse_choices_from_help(serve_help_output, "--tool-call-parser")
    registry = set(ToolParserManager.list_registered())
    assert cli_choices == registry, (
        f"CLI choices {sorted(cli_choices)} do not match "
        f"ToolParserManager.list_registered() {sorted(registry)}. "
        f"Someone may have hardcoded the choices list again."
    )


def test_gemma4_is_in_tool_call_parser_choices(serve_help_output):
    """Regression guard: gemma4 must specifically be an accepted CLI choice."""
    assert "gemma4" in serve_help_output, (
        "gemma4 tool-call parser is not a valid CLI choice. "
        "Check that @ToolParserManager.register_module('gemma4') fires at "
        "import time in vllm_mlx/tool_parsers/gemma4_tool_parser.py and that "
        "tool_parsers/__init__.py imports Gemma4ToolParser."
    )


def test_qwen3_coder_is_in_tool_call_parser_choices(serve_help_output):
    """Regression guard: qwen3_coder must be a valid choice (Hermes alias)."""
    assert "qwen3_coder" in serve_help_output, (
        "qwen3_coder is not a valid --tool-call-parser choice. "
        "It is registered as a Hermes alias in hermes_tool_parser.py — "
        "check that the alias is still present."
    )
