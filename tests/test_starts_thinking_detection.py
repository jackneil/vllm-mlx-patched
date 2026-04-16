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
            tok, start_token="<|channel>", end_tokens=["<channel|>", "<|channel>response"]
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
            tok, start_token="<|channel>", end_tokens=["<channel|>", "<|channel>response"]
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
