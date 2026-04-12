# SPDX-License-Identifier: Apache-2.0
"""Unit tests for streaming-router integration properties on reasoning parsers.

These tests pin the property names and values that `server.py`'s
`_stream_anthropic_messages` reads to configure `StreamingThinkRouter`.
If any property is renamed or its default changes, the Anthropic
streaming path for that parser silently breaks — these tests catch it.

The ``TestRebaseBreakageSentinel`` class is the sanity check that will fire
in CI if an upstream rebase renames one of these properties. Keep it.
"""

from __future__ import annotations

import unittest

from vllm_mlx.reasoning.base import ReasoningParser
from vllm_mlx.reasoning.gemma4_parser import Gemma4ReasoningParser
from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser


class _ConcreteMinimalParser(ReasoningParser):
    """Minimal subclass to exercise base-class defaults."""

    def extract_reasoning(self, model_output):
        return None, model_output

    def extract_reasoning_streaming(self, previous_text, current_text, delta_text):
        return None


class TestBaseDefaults(unittest.TestCase):
    def test_start_token_default(self):
        self.assertEqual(_ConcreteMinimalParser().start_token, "<think>")

    def test_end_tokens_default(self):
        self.assertEqual(_ConcreteMinimalParser().end_tokens, ["</think>"])

    def test_channel_strip_prefix_default_is_none(self):
        self.assertIsNone(_ConcreteMinimalParser().channel_strip_prefix)


class TestGemma4Overrides(unittest.TestCase):
    def test_start_token(self):
        self.assertEqual(Gemma4ReasoningParser().start_token, "<|channel>")

    def test_end_tokens_list(self):
        self.assertEqual(
            Gemma4ReasoningParser().end_tokens,
            ["<channel|>", "<|channel>response"],
        )

    def test_channel_strip_prefix(self):
        self.assertEqual(Gemma4ReasoningParser().channel_strip_prefix, "thought\n")


class TestQwen3DefaultsUnchanged(unittest.TestCase):
    """Qwen3 should inherit base defaults — no overrides needed."""

    def test_qwen3_uses_default_tokens(self):
        p = Qwen3ReasoningParser()
        self.assertEqual(p.start_token, "<think>")
        self.assertEqual(p.end_tokens, ["</think>"])
        self.assertIsNone(p.channel_strip_prefix)


class TestRebaseBreakageSentinel(unittest.TestCase):
    """If any expected property gets renamed upstream, this blows up in CI.

    The server's ``_stream_anthropic_messages`` gates on ``hasattr``; without
    this test, a rename would silently fall back to defaults and the Anthropic
    thinking-block stream would regress with no failing test.
    """

    EXPECTED = ("start_token", "end_tokens", "channel_strip_prefix")

    def test_base_class_exposes_expected_properties(self):
        for name in self.EXPECTED:
            self.assertTrue(
                hasattr(_ConcreteMinimalParser(), name),
                f"ReasoningParser base missing property '{name}' — "
                "upstream rebase drift. Fix the server's hasattr gate too.",
            )

    def test_gemma4_exposes_expected_properties(self):
        for name in self.EXPECTED:
            self.assertTrue(
                hasattr(Gemma4ReasoningParser(), name),
                f"Gemma4ReasoningParser missing property '{name}'.",
            )


if __name__ == "__main__":
    unittest.main()
