# SPDX-License-Identifier: Apache-2.0
"""
Base classes for reasoning content extraction.

This module provides the abstract base class for reasoning parsers that extract
thinking/reasoning content from model outputs (e.g., <think>...</think> tags).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class DeltaMessage:
    """
    Delta message for streaming reasoning output.

    Contains either reasoning content, regular content, or both when
    transitioning from reasoning to content phase.

    Note: reasoning and content should typically not both be non-None
    except during the transition chunk.
    """

    role: str | None = None
    content: str | None = None
    reasoning: str | None = None

    @property
    def reasoning_content(self) -> str | None:
        """Deprecated: use reasoning instead. Maintained for backward compatibility."""
        return self.reasoning


class ReasoningParser(ABC):
    """
    Abstract base class for reasoning content extraction.

    Reasoning parsers extract thinking/reasoning content from model outputs,
    separating it from the final response content. This is useful for models
    like DeepSeek-R1, Qwen3, etc. that use special tokens to denote reasoning.

    Example:
        Input: "<think>Let me solve this step by step...</think>The answer is 42."
        Output: reasoning="Let me solve this step by step...", content="The answer is 42."
    """

    def __init__(self, tokenizer: Any | None = None):
        """
        Initialize parser with optional tokenizer.

        Args:
            tokenizer: Optional tokenizer for token-based parsing. For vllm-mlx,
                      text-based parsing is sufficient, so this is optional.
        """
        self.tokenizer = tokenizer

    # -- Streaming router integration ---------------------------------------
    # The Anthropic streaming path in server.py uses a StreamingThinkRouter
    # that needs to know the reasoning delimiters. Expose them as properties
    # so subclasses override only what they need. Defaults match the
    # "<think>/</think>" convention that the router hardcoded before this
    # refactor — safe fallback for any parser that doesn't override.

    @property
    def start_token(self) -> str:
        """Opening delimiter for reasoning content. Default: ``<think>``."""
        return "<think>"

    @property
    def end_tokens(self) -> list[str]:
        """List of closing delimiters that terminate reasoning content.

        The streaming router scans for the earliest occurrence of any entry
        in this list and treats it as the close. Default is a single-entry
        list matching ``</think>``. Override to supply protocol-specific
        alternatives (e.g. parsers whose format has more than one valid
        transition marker out of reasoning mode).
        """
        return ["</think>"]

    @property
    def channel_strip_prefix(self) -> str | None:
        """Optional prefix stripped from the first reasoning emission.

        Parsers whose protocol prefixes reasoning content with channel or
        metadata tokens should override this to return that prefix string.
        The router drops exactly ``len(prefix)`` characters from the very
        start of the emitted reasoning content. Default ``None`` means
        "strip nothing."
        """
        return None

    @abstractmethod
    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from complete model output.

        Args:
            model_output: Complete text output from the model.

        Returns:
            Tuple of (reasoning_content, final_content).
            Either may be None if not present.
        """
        pass

    @abstractmethod
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming delta.

        Uses the "previous + delta = current" model where:
        - previous_text: All text accumulated before this delta
        - current_text: All text including this delta (previous + delta)
        - delta_text: Just the new text in this chunk

        Args:
            previous_text: Accumulated text before this delta.
            current_text: Accumulated text including this delta.
            delta_text: The new text in this streaming chunk.

        Returns:
            DeltaMessage with reasoning and/or content populated,
            or None if this delta should be skipped (e.g., special tokens).
        """
        pass

    def reset_state(self):  # noqa: B027
        """
        Reset any internal state for a new request.

        Called before starting to process a new streaming request.
        Override in subclasses if stateful parsing is needed.
        This is intentionally a default no-op implementation.
        """
        pass
