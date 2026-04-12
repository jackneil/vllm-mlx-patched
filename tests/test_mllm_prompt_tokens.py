# SPDX-License-Identifier: Apache-2.0
"""Tests for MLLM.stream_chat prompt_tokens fallback + chunk-preference logic."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class _FakeChunk:
    def __init__(self, text: str, prompt_tokens: int = 0, finish_reason=None):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.finish_reason = finish_reason


class _FakeTokenizer:
    """Tokenizer returns one token per character (easy math)."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(self, *args, **kwargs):
        return "system: hello\nuser: hi"


def _build_mllm_with_processor(processor):
    from vllm_mlx.models.mllm import MLXMultimodalLM
    mllm = MLXMultimodalLM.__new__(MLXMultimodalLM)
    mllm.model = MagicMock()
    mllm.processor = processor
    # Minimum attrs needed for stream_chat to reach the generate loop.
    mllm._loaded = True
    mllm._video_native = False
    mllm._cache_manager = None
    return mllm


def _fixed_prompt(*_args, **_kwargs):
    """Replacement for mlx_vlm.prompt_utils.get_chat_template so tests have
    a deterministic formatted prompt regardless of processor.chat_template."""
    return "system: hello\nuser: hi"


class TestMLLMPromptTokensPrecompute(unittest.TestCase):
    def test_precompute_used_when_chunks_report_zero(self):
        chunks = [
            _FakeChunk("Hello", prompt_tokens=0),
            _FakeChunk(" world", prompt_tokens=0, finish_reason="stop"),
        ]
        expected = len("system: hello\nuser: hi")

        with patch("mlx_vlm.stream_generate", return_value=iter(chunks)), patch(
            "mlx_vlm.prompt_utils.get_chat_template", side_effect=_fixed_prompt
        ):
            mllm = _build_mllm_with_processor(_FakeProcessor())
            outputs = list(
                mllm.stream_chat(
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=10,
                )
            )
        for out in outputs:
            self.assertEqual(out.prompt_tokens, expected)

    def test_nonzero_chunk_value_wins_over_precompute(self):
        chunks = [
            _FakeChunk("Hello", prompt_tokens=42),
            _FakeChunk(" world", prompt_tokens=42, finish_reason="stop"),
        ]
        with patch("mlx_vlm.stream_generate", return_value=iter(chunks)), patch(
            "mlx_vlm.prompt_utils.get_chat_template", side_effect=_fixed_prompt
        ):
            mllm = _build_mllm_with_processor(_FakeProcessor())
            outputs = list(
                mllm.stream_chat(
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=10,
                )
            )
        for out in outputs:
            self.assertEqual(out.prompt_tokens, 42)

    def test_mismatch_warning_fires_once(self):
        chunks = [
            _FakeChunk("hi", prompt_tokens=5),
            _FakeChunk("", prompt_tokens=5, finish_reason="stop"),
        ]
        import logging
        logger = logging.getLogger("vllm_mlx.models.mllm")
        with patch("mlx_vlm.stream_generate", return_value=iter(chunks)), patch(
            "mlx_vlm.prompt_utils.get_chat_template", side_effect=_fixed_prompt
        ):
            mllm = _build_mllm_with_processor(_FakeProcessor())
            with self.assertLogs(logger, level="WARNING") as cm:
                list(
                    mllm.stream_chat(
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=5,
                    )
                )
        warning_msgs = [r for r in cm.output if "mismatch" in r]
        self.assertEqual(len(warning_msgs), 1, cm.output)

    def test_zero_when_processor_has_no_tokenizer(self):
        chunks = [_FakeChunk("Hi", prompt_tokens=0, finish_reason="stop")]

        class _NoTokProcessor:
            def apply_chat_template(self, *a, **k):
                return "prompt"

        with patch("mlx_vlm.stream_generate", return_value=iter(chunks)), patch(
            "mlx_vlm.prompt_utils.get_chat_template", side_effect=_fixed_prompt
        ):
            mllm = _build_mllm_with_processor(_NoTokProcessor())
            outputs = list(
                mllm.stream_chat(
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=5,
                )
            )
        for out in outputs:
            self.assertEqual(out.prompt_tokens, 0)


if __name__ == "__main__":
    unittest.main()
