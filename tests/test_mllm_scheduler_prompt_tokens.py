# SPDX-License-Identifier: Apache-2.0
"""Tests for MLLMScheduler.add_request precomputing num_prompt_tokens.

Root cause: MLLMRequest.num_prompt_tokens defaulted to 0 and was never
assigned, so message_delta.usage.input_tokens leaked 0 into Anthropic
streaming responses (breaking Claude Code context tracking).

These tests lock in the precompute behavior, including the graceful
fallback when tokenizer.encode raises.
"""

from __future__ import annotations

import logging
import unittest
from unittest.mock import MagicMock

from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig


class _FakeTokenizer:
    """One token per character — makes assertions trivial."""

    def __init__(self, fail: bool = False):
        self._fail = fail

    def encode(self, text: str) -> list[int]:
        if self._fail:
            raise RuntimeError("boom")
        return list(range(len(text)))


class _FakeProcessor:
    def __init__(self, fail: bool = False):
        self.tokenizer = _FakeTokenizer(fail=fail)


def _build_scheduler(processor) -> MLLMScheduler:
    """Construct a scheduler without running __init__ side effects.

    We bypass __init__ because it pulls in MultimodalProcessor / caches /
    stop-token resolution that require a real model. Only the attrs
    add_request touches need to be set.
    """
    sched = MLLMScheduler.__new__(MLLMScheduler)
    sched.model = MagicMock()
    sched.processor = processor
    sched.config = MLLMSchedulerConfig()
    sched.requests = {}
    from collections import deque

    sched.waiting = deque()
    return sched


class TestSchedulerPrecomputesPromptTokens(unittest.TestCase):
    def test_precompute_sets_num_prompt_tokens(self):
        sched = _build_scheduler(_FakeProcessor())
        prompt = "hello world"
        request_id = sched.add_request(prompt=prompt)

        req = sched.requests[request_id]
        self.assertEqual(req.num_prompt_tokens, len(prompt))

    def test_precompute_handles_tokenizer_directly_on_processor(self):
        """If processor has no .tokenizer, fall back to processor.encode."""
        sched = _build_scheduler(_FakeTokenizer())  # tokenizer IS processor
        request_id = sched.add_request(prompt="abcd")
        self.assertEqual(sched.requests[request_id].num_prompt_tokens, 4)

    def test_precompute_falls_back_to_zero_and_warns_on_error(self):
        sched = _build_scheduler(_FakeProcessor(fail=True))

        with self.assertLogs("vllm_mlx.mllm_scheduler", level="WARNING") as cm:
            request_id = sched.add_request(prompt="anything")

        req = sched.requests[request_id]
        self.assertEqual(req.num_prompt_tokens, 0)
        self.assertTrue(
            any("prompt_tokens precompute failed" in m for m in cm.output),
            cm.output,
        )

    def test_precompute_zero_when_no_encode_method(self):
        class _NoEncode:
            pass

        class _Proc:
            tokenizer = _NoEncode()

        sched = _build_scheduler(_Proc())
        request_id = sched.add_request(prompt="whatever")
        # Should remain at default 0 without raising.
        self.assertEqual(sched.requests[request_id].num_prompt_tokens, 0)


if __name__ == "__main__":
    unittest.main()
