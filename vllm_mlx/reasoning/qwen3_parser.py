# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for Qwen3 models.

Qwen3 uses <think>...</think> tags for reasoning content and supports
a strict switch via 'enable_thinking=False' in chat template kwargs.

Supports implicit reasoning mode where <think> is injected in the prompt
by AI agents (e.g., OpenCode) and only </think> appears in the output.
"""

from .think_parser import BaseThinkingReasoningParser


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Qwen3 models.

    Qwen3 uses <think>...</think> tokens to denote reasoning text.

    Supports three scenarios:
    1. Both tags in output: <think>reasoning</think>content
    2. Only closing tag (think in prompt): reasoning</think>content
    3. No tags: pure content

    Example (normal):
        Input: "<think>Let me analyze this...</think>The answer is 42."
        Output: reasoning="Let me analyze this...", content="The answer is 42."

    Example (think in prompt):
        Input: "Let me analyze this...</think>The answer is 42."
        Output: reasoning="Let me analyze this...", content="The answer is 42."
    """

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from Qwen3 output.

        Handles explicit <think>...</think> tags, implicit mode
        (only </think> in output), and truncated reasoning
        (<think> present but </think> missing because budget/max_tokens
        cut generation before the close tag was emitted).

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple.
        """
        # Only short-circuit to "pure content" when NEITHER tag is present.
        # If <think> is present without </think>, the base class's Case 3
        # correctly surfaces the partial thinking as reasoning (not content).
        # Returning (None, text) here would leak truncated reasoning into the
        # user-visible content field — see thinking_token_budget regression.
        if self.start_token not in model_output and self.end_token not in model_output:
            return None, model_output

        return super().extract_reasoning(model_output)
