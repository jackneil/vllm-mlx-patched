"""Tests for _detect_starts_thinking helper."""

import unittest
from unittest.mock import MagicMock


class TestDetectStartsThinking(unittest.TestCase):
    """Test the render-and-inspect approach for _starts_thinking."""

    def _make_tokenizer(self, rendered_suffix: str):
        """Create a mock tokenizer whose apply_chat_template returns a prompt ending with rendered_suffix."""
        tok = MagicMock()
        tok.chat_template = "template with <|channel> and add_generation_prompt"
        tok.apply_chat_template = MagicMock(
            return_value=f"<bos><|turn>user\nx<turn|>\n<|turn>model\n{rendered_suffix}"
        )
        return tok

    def test_closed_thinking_block_returns_false(self):
        """Gemma4 without enable_thinking: <|channel>thought\\n<channel|> is CLOSED."""
        from vllm_mlx.server import _detect_starts_thinking

        tok = self._make_tokenizer("<|channel>thought\n<channel|>")
        result = _detect_starts_thinking(
            tok,
            start_token="<|channel>",
            end_tokens=["<channel|>", "<|channel>response"],
        )
        self.assertFalse(result)

    def test_open_thinking_block_returns_true(self):
        """Qwen3-style: template injects <think> with no </think> -> open."""
        from vllm_mlx.server import _detect_starts_thinking

        tok = self._make_tokenizer("<think>")
        tok.chat_template = "template with <think> and add_generation_prompt"
        result = _detect_starts_thinking(
            tok, start_token="<think>", end_tokens=["</think>"]
        )
        self.assertTrue(result)

    def test_no_start_token_in_template_returns_false(self):
        """Template doesn't contain start token at all -> False."""
        from vllm_mlx.server import _detect_starts_thinking

        tok = self._make_tokenizer("")
        tok.chat_template = "template with add_generation_prompt but no channel token"
        result = _detect_starts_thinking(
            tok, start_token="<|channel>", end_tokens=["<channel|>"]
        )
        self.assertFalse(result)

    def test_no_add_generation_prompt_in_template_returns_false(self):
        """Template doesn't contain add_generation_prompt -> False."""
        from vllm_mlx.server import _detect_starts_thinking

        tok = self._make_tokenizer("<|channel>")
        tok.chat_template = "template with <|channel> but no gen prompt"
        result = _detect_starts_thinking(
            tok, start_token="<|channel>", end_tokens=["<channel|>"]
        )
        self.assertFalse(result)

    def test_no_tokenizer_returns_false(self):
        """No tokenizer available -> False."""
        from vllm_mlx.server import _detect_starts_thinking

        result = _detect_starts_thinking(
            None, start_token="<think>", end_tokens=["</think>"]
        )
        self.assertFalse(result)

    def test_tokenizer_without_chat_template_returns_false(self):
        """Tokenizer exists but has no chat_template attr -> False."""
        from vllm_mlx.server import _detect_starts_thinking

        tok = MagicMock(spec=[])
        result = _detect_starts_thinking(
            tok, start_token="<think>", end_tokens=["</think>"]
        )
        self.assertFalse(result)

    def test_template_render_exception_falls_back_to_naive_check(self):
        """If apply_chat_template raises, fall back to naive text check (True)."""
        from vllm_mlx.server import _detect_starts_thinking

        tok = MagicMock()
        tok.chat_template = "template with <think> and add_generation_prompt"
        tok.apply_chat_template = MagicMock(side_effect=Exception("template error"))
        result = _detect_starts_thinking(
            tok, start_token="<think>", end_tokens=["</think>"]
        )
        self.assertTrue(result)

    def test_multiple_end_tokens_checks_all(self):
        """With multiple end tokens, any closing after start -> False."""
        from vllm_mlx.server import _detect_starts_thinking

        tok = self._make_tokenizer("<|channel>thought\n<|channel>response\nsome text")
        tok.chat_template = "has <|channel> and add_generation_prompt"
        result = _detect_starts_thinking(
            tok,
            start_token="<|channel>",
            end_tokens=["<channel|>", "<|channel>response"],
        )
        self.assertFalse(result)


class TestDetectStartsThinkingHonorsTemplateKwargs(unittest.TestCase):
    """The probe must render with THIS request's chat_template_kwargs.

    Qwen3.x: default tail is OPEN (`<think>\\n`); with enable_thinking=False
    (Layer 1 first-turn-with-tools auto-disable, or client-set) the tail is
    CLOSED (`<think>\\n\\n</think>\\n\\n`). Probing without the kwarg reported
    "open" for such requests -> router started in thinking mode -> a tagless
    answer streamed entirely as a thinking block (Claude Code: no visible
    output). Verified tails from the served Qwen3.8 tokenizer, 2026-08-16.
    """

    OPEN = "<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n<think>\n"
    CLOSED = (
        "<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    def _qwen3_tokenizer(self):
        tok = MagicMock()
        tok.chat_template = "... add_generation_prompt ... enable_thinking ... <think>"

        def _render(messages, **kwargs):
            if kwargs.get("enable_thinking") is False:
                return self.CLOSED
            return self.OPEN

        tok.apply_chat_template = MagicMock(side_effect=_render)
        return tok

    def test_default_kwargs_open(self):
        from vllm_mlx.server import _detect_starts_thinking

        tok = self._qwen3_tokenizer()
        self.assertTrue(
            _detect_starts_thinking(tok, start_token="<think>", end_tokens=["</think>"])
        )

    def test_none_kwargs_same_as_default(self):
        from vllm_mlx.server import _detect_starts_thinking

        tok = self._qwen3_tokenizer()
        self.assertTrue(
            _detect_starts_thinking(
                tok,
                start_token="<think>",
                end_tokens=["</think>"],
                chat_template_kwargs=None,
            )
        )

    def test_enable_thinking_false_closes(self):
        """The Layer 1 / client-set case: kwarg reaches the render and the
        probe reports CLOSED."""
        from vllm_mlx.server import _detect_starts_thinking

        tok = self._qwen3_tokenizer()
        result = _detect_starts_thinking(
            tok,
            start_token="<think>",
            end_tokens=["</think>"],
            chat_template_kwargs={"enable_thinking": False},
        )
        self.assertFalse(result)
        _, kwargs = tok.apply_chat_template.call_args
        self.assertEqual(kwargs.get("enable_thinking"), False)
        # Engine defaults still present (same merge as BatchedEngine).
        self.assertFalse(kwargs.get("tokenize"))
        self.assertTrue(kwargs.get("add_generation_prompt"))

    def test_enable_thinking_true_stays_open(self):
        from vllm_mlx.server import _detect_starts_thinking

        tok = self._qwen3_tokenizer()
        self.assertTrue(
            _detect_starts_thinking(
                tok,
                start_token="<think>",
                end_tokens=["</think>"],
                chat_template_kwargs={"enable_thinking": True},
            )
        )

    def test_caller_kwargs_override_engine_defaults(self):
        """Merge order mirrors BatchedEngine._apply_chat_template: caller
        kwargs LAST, so they win over the probe's own defaults."""
        from vllm_mlx.server import _detect_starts_thinking

        tok = self._qwen3_tokenizer()
        _detect_starts_thinking(
            tok,
            start_token="<think>",
            end_tokens=["</think>"],
            chat_template_kwargs={"add_generation_prompt": True, "custom_flag": 1},
        )
        _, kwargs = tok.apply_chat_template.call_args
        self.assertEqual(kwargs.get("custom_flag"), 1)
        self.assertTrue(kwargs.get("add_generation_prompt"))

    def test_kwargs_do_not_leak_between_calls(self):
        """A closed probe for one request must not affect the next request's
        probe (no shared/mutated dict)."""
        from vllm_mlx.server import _detect_starts_thinking

        tok = self._qwen3_tokenizer()
        req_kwargs = {"enable_thinking": False}
        self.assertFalse(
            _detect_starts_thinking(
                tok, "<think>", ["</think>"], chat_template_kwargs=req_kwargs
            )
        )
        self.assertEqual(req_kwargs, {"enable_thinking": False})  # not mutated
        self.assertTrue(_detect_starts_thinking(tok, "<think>", ["</think>"]))


class TestStartsThinkingIntegration(unittest.TestCase):
    """Verify the old inline code path is replaced."""

    def test_gemma4_closed_block_produces_text_not_thinking(self):
        """End-to-end: when starts_thinking is False, plain text routes as text."""
        from vllm_mlx.api.utils import StreamingThinkRouter

        # Simulate Gemma4 with _starts_thinking=False (the fix)
        router = StreamingThinkRouter(
            start_in_thinking=False,
            start_token="<|channel>",
            end_tokens=["<channel|>", "<|channel>response"],
            channel_strip_prefix="thought\n",
        )
        pieces = router.process("Paris") + router.flush()

        # Should be text, not thinking
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0], ("text", "Paris"))

    def test_gemma4_true_starts_thinking_strips_and_misclassifies(self):
        """Demonstrates the bug: starts_thinking=True eats short responses."""
        from vllm_mlx.api.utils import StreamingThinkRouter

        # The old broken behavior
        router = StreamingThinkRouter(
            start_in_thinking=True,
            start_token="<|channel>",
            end_tokens=["<channel|>", "<|channel>response"],
            channel_strip_prefix="thought\n",
        )
        pieces = router.process("Paris") + router.flush()

        # "Paris" (5 chars) is entirely consumed by the 8-char strip counter
        self.assertEqual(pieces, [])


if __name__ == "__main__":
    unittest.main()
