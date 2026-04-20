# SPDX-License-Identifier: Apache-2.0
"""CLI parsing tests for --max-thinking-token-budget and
--disable-qwen3-first-turn-no-think flags.

Drives cli.main() with sys.argv mocking up to the point where
server.* attrs are set, then short-circuits via SystemExit from
uvicorn.run patch. Follows the pattern in tests/test_cli_argparse.py.
"""

import sys
from unittest.mock import patch

from vllm_mlx import cli, server


def _parse_serve_args(*argv_extra):
    """Drive cli.main's argparse up to the server attribute assignments.
    Patch load_model and uvicorn.run so we don't actually boot."""
    with (
        patch("vllm_mlx.server.load_model"),
        patch("uvicorn.run"),
        patch.object(sys, "argv", ["vllm-mlx", "serve", "some-model", *argv_extra]),
    ):
        try:
            cli.main()
        except SystemExit:
            pass  # argparse.exit on error / uvicorn no-op is expected
    return


def test_accepts_positive_int():
    _parse_serve_args("--max-thinking-token-budget", "2048")
    assert server._max_thinking_token_budget == 2048


def test_defaults_to_none_when_omitted():
    _parse_serve_args()  # no flag
    assert server._max_thinking_token_budget is None


def test_rejects_zero_with_clear_error(capsys):
    _parse_serve_args("--max-thinking-token-budget", "0")
    err = capsys.readouterr().err
    assert "Omit the flag" in err


def test_rejects_negative(capsys):
    _parse_serve_args("--max-thinking-token-budget", "-1")
    err = capsys.readouterr().err
    assert "must be > 0" in err


def test_rejects_non_integer(capsys):
    _parse_serve_args("--max-thinking-token-budget", "abc")
    err = capsys.readouterr().err
    assert "integer" in err or "invalid int" in err


def test_server_attribute_defaults_to_none():
    _parse_serve_args()
    assert server._max_thinking_token_budget is None


def test_disable_qwen3_flag_defaults_false():
    _parse_serve_args()
    assert server._disable_qwen3_first_turn_no_think is False


def test_disable_qwen3_flag_can_be_set():
    _parse_serve_args("--disable-qwen3-first-turn-no-think")
    assert server._disable_qwen3_first_turn_no_think is True


def test_disable_qwen3_server_attribute_is_bool_type():
    # Module-level default is a bool; CLI drives it via store_true.
    _parse_serve_args()
    assert isinstance(server._disable_qwen3_first_turn_no_think, bool)
    assert server._disable_qwen3_first_turn_no_think is False
