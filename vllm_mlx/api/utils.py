# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for text processing and model detection.
"""

import logging
import re
from typing import Iterable

from .models import Message

logger = logging.getLogger(__name__)

# =============================================================================
# Special Token Patterns
# =============================================================================

# Pattern to match special tokens that should be removed from output
# Keeps <think>...</think> blocks intact for reasoning models.
#
# NOTE: a distinct inbound-reject pattern lives in
# `api/anthropic_adapter.py::_THINKING_INJECTION_PATTERNS` — it UNIONS
# this pattern with `<think>`/`</think>` to catch client-supplied
# prompt-injection attempts in assistant-history thinking blocks.
# Keep that pattern in sync with this one (it derives from `.pattern`).
SPECIAL_TOKENS_PATTERN = re.compile(
    r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|"
    r"<\|end\|>|<\|eot_id\|>|<\|start_header_id\|>|<\|end_header_id\|>|"
    r"<\|channel\|>|<\|message\|>|<\|start\|>|<\|return\|>|<\|call\|>|<\|constrain\|>|"
    r"</s>|<s>|<pad>|\[PAD\]|\[SEP\]|\[CLS\]"
)


# Regex for matching final channel marker with optional constrain token:
#   <|channel|>final<|message|>
#   <|channel|>final <|constrain|>JSON<|message|>
_FINAL_CHANNEL_RE = re.compile(
    r"<\|channel\|>final[^<]*(?:<\|constrain\|>[^<]*)?<\|message\|>"
)


def _clean_gpt_oss_output(text: str) -> str:
    """
    Extract final channel content from GPT-OSS channel-based output.

    When reasoning parser is not enabled, this provides a fallback that
    extracts the 'final' channel content so the API response is usable.

    Handles both standard and extended format with constrain token:
        <|channel|>final<|message|>...
        <|channel|>final <|constrain|>JSON<|message|>...

    Args:
        text: Raw model output containing channel tokens.

    Returns:
        Extracted final content, or text with channel tokens stripped.
    """
    match = _FINAL_CHANNEL_RE.search(text)
    if match:
        content = text[match.end() :]
        # Strip trailing structural tokens (including <|constrain|>)
        content = re.sub(
            r"<\|start\|>|<\|end\|>|<\|channel\|>|<\|return\|>|<\|call\|>|<\|message\|>|<\|constrain\|>",
            "",
            content,
        )
        return content.strip()

    # No final channel — strip all channel/structural tokens (including constrain)
    cleaned = re.sub(
        r"<\|channel\|>[^<]*(?:<\|constrain\|>[^<]*)?<\|message\|>|<\|start\|>[^<]*|<\|return\|>|<\|call\|>|<\|constrain\|>[^<]*",
        "",
        text,
    )
    return cleaned.strip()


def clean_output_text(text: str) -> str:
    """
    Clean model output by removing special tokens.

    Keeps <think>...</think> blocks intact for reasoning models.
    Adds opening <think> tag if missing (happens when thinking is enabled
    in the prompt template but the tag is part of the prompt, not output).
    Handles GPT-OSS channel-based format as fallback when reasoning parser
    is not enabled.

    Args:
        text: Raw model output

    Returns:
        Cleaned text with special tokens removed
    """
    if not text:
        return text

    # GPT-OSS channel format — extract final content before general stripping
    if "<|channel|>" in text and "<|message|>" in text:
        text = _clean_gpt_oss_output(text)
        return text

    text = SPECIAL_TOKENS_PATTERN.sub("", text)
    text = text.strip()

    # Add opening <think> tag if response has closing but not opening
    # This happens when enable_thinking=True in the chat template
    if "</think>" in text and not text.lstrip().startswith("<think>"):
        text = "<think>" + text

    return text


# =============================================================================
# Streaming Tool Call Filter
# =============================================================================

# Safety cap for tool call buffer (bytes). If a tool call block never closes,
# the buffer is capped to prevent unbounded memory growth. In practice, the
# buffer is bounded by max_tokens (~100KB at 32768 tokens), but this cap
# protects against pathological cases.
_MAX_TOOL_BUFFER_BYTES = 1_048_576  # 1 MB

# Tags that delimit tool call blocks in streaming output.
# Content inside these tags should be suppressed during streaming because
# it will be re-emitted as structured tool_use blocks after parsing.
_TOOL_CALL_TAGS = [
    ("<minimax:tool_call>", "</minimax:tool_call>"),
    ("<tool_call>", "</tool_call>"),
    ("<function=", "</function>"),
    ("<|tool_call>", "<tool_call|>"),
    ("[TOOL_CALL]", "[/TOOL_CALL]"),
    ("[Calling tool", "]\n"),  # Qwen3 bracket-style: [Calling tool: func({...})]\n
]


class StreamingToolCallFilter:
    """Buffer streaming text to suppress tool call markup.

    Tool call XML (e.g. <minimax:tool_call>...</minimax:tool_call>) arrives
    split across multiple streaming deltas. This filter detects entry into a
    tool call block, suppresses all output until the block closes, and emits
    only non-tool-call text.

    The full unfiltered text is still accumulated separately for tool call
    parsing at stream end.
    """

    def __init__(self):
        self._buffer = ""
        self._in_block = False
        self._close_tag = ""
        # Longest open tag - used to determine how much buffer to hold back
        self._max_open_len = max(len(t[0]) for t in _TOOL_CALL_TAGS)

    def process(self, delta: str) -> str:
        """Process a streaming delta. Returns text to emit (may be empty)."""
        self._buffer += delta

        if self._in_block:
            return self._consume_block()
        else:
            return self._scan_for_open()

    def _scan_for_open(self) -> str:
        """Scan buffer for tool call open tags. Emit safe text."""
        # Check for complete open tags
        for open_tag, close_tag in _TOOL_CALL_TAGS:
            idx = self._buffer.find(open_tag)
            if idx >= 0:
                # Found an open tag - emit text before it, enter block mode
                emit = self._buffer[:idx]
                self._buffer = self._buffer[idx + len(open_tag) :]
                self._in_block = True
                self._close_tag = close_tag
                # Process remainder in case close tag is already in buffer
                after = self._consume_block()
                return emit + after

        # No complete open tag found. Check if buffer ends with a partial
        # match of any open tag - hold that back to avoid emitting a fragment.
        hold_back = 0
        for open_tag, _ in _TOOL_CALL_TAGS:
            for prefix_len in range(min(len(open_tag), len(self._buffer)), 0, -1):
                if self._buffer.endswith(open_tag[:prefix_len]):
                    hold_back = max(hold_back, prefix_len)
                    break

        if hold_back > 0:
            emit = self._buffer[:-hold_back]
            self._buffer = self._buffer[-hold_back:]
            return emit

        # No partial match - safe to emit everything
        emit = self._buffer
        self._buffer = ""
        return emit

    def _consume_block(self) -> str:
        """Consume content inside a tool call block. Returns empty string
        unless the block closes and there's text after it."""
        idx = self._buffer.find(self._close_tag)
        if idx >= 0:
            # Block closed - discard content up to and including close tag
            self._buffer = self._buffer[idx + len(self._close_tag) :]
            self._in_block = False
            self._close_tag = ""
            # Process remainder - might have more text or another tool call
            if self._buffer:
                return self._scan_for_open()
            return ""
        # Still inside block - suppress everything but cap buffer size
        if len(self._buffer) > _MAX_TOOL_BUFFER_BYTES:
            logger.warning(
                f"Tool call buffer exceeded {_MAX_TOOL_BUFFER_BYTES} bytes, "
                f"discarding and exiting block"
            )
            self._buffer = ""
            self._in_block = False
            self._close_tag = ""
        return ""

    def flush(self) -> str:
        """Flush remaining buffer at end of stream."""
        if self._in_block:
            # Unterminated tool call block - discard
            self._buffer = ""
            self._in_block = False
            return ""
        emit = self._buffer
        self._buffer = ""
        return emit


# =============================================================================
# Streaming Think Block Router
# =============================================================================


class StreamingThinkRouter:
    """Route <think>...</think> content to separate Anthropic thinking blocks.

    Instead of emitting thinking content as plain text (where it's
    indistinguishable from the response), this router yields tagged
    pieces that the streaming handler can emit as proper Anthropic
    content block types.

    Each call to process() returns a list of (block_type, text) tuples:
    - ("thinking", text) for content inside <think>...</think>
    - ("text", text) for content outside think blocks

    Args:
        start_in_thinking: If True, assume the model starts in thinking
            mode (e.g. MiniMax adds <think> to the generation prompt,
            so the tag never appears in the output stream).
    """

    def __init__(
        self,
        start_in_thinking: bool = False,
        start_token: str = "<think>",
        end_tokens: "Iterable[str]" = ("</think>",),
        channel_strip_prefix: str | None = None,
    ):
        """Route reasoning content (delimited by start/end tokens) into
        Anthropic thinking blocks.

        Args:
            start_in_thinking: the model's chat template injects the start
                token into the generation prompt, so only the close tag
                appears in output (e.g. MiniMax/Nemotron pattern).
            start_token: opening delimiter. Default ``<think>`` works for
                Qwen3 and other Anthropic-friendly models. Pass ``<|channel>``
                for Gemma 4.
            end_tokens: iterable of closing delimiters. The earliest match
                in the buffer wins. Default ``("</think>",)`` preserves
                legacy behavior. For Gemma 4 pass
                ``("<channel|>", "<|channel>response")``.
            channel_strip_prefix: optional string stripped from the start of
                the first thinking emission. Gemma 4 emits
                ``<|channel>thought\n<real reasoning>``; pass ``"thought\n"``
                to drop the channel name from the Anthropic thinking delta.
                None (default) strips nothing.
        """
        self._buffer = ""
        self._in_think = start_in_thinking
        self._start_token = start_token

        # Normalize end_tokens to a tuple. Reject empty (ambiguous intent)
        # and reject entries where one is a strict prefix of another —
        # otherwise a long marker gets spuriously closed by its own prefix.
        end_tokens_tuple: tuple[str, ...] = tuple(end_tokens)
        if not end_tokens_tuple:
            raise ValueError("StreamingThinkRouter requires at least one end token")
        for a in end_tokens_tuple:
            for b in end_tokens_tuple:
                if a is not b and a != b and b.startswith(a):
                    raise ValueError(
                        f"end_tokens contains prefix collision: {a!r} is a prefix "
                        f"of {b!r}; the shorter would spuriously close first"
                    )
        self._end_tokens: tuple[str, ...] = end_tokens_tuple

        self._channel_strip_prefix = channel_strip_prefix

        # Integer counter for channel-name strip. Set when we enter thinking
        # mode (either via start_in_thinking=True or by matching the start
        # token later). Each thinking emission drops up to _strip_remaining
        # characters from the front of the piece and decrements. At 0, emits
        # pass through. NOT buffer mutation — the counter survives buffer
        # resets.
        self._strip_remaining: int = (
            len(channel_strip_prefix)
            if (start_in_thinking and channel_strip_prefix)
            else 0
        )

    def _enter_thinking(self) -> None:
        """Transition from text mode to thinking mode.

        Resets the channel-strip counter on every entry so multiple
        <think>...</think> blocks within one stream each strip their
        channel name. Defensive — most streams have a single block.
        """
        self._in_think = True
        if self._channel_strip_prefix is not None:
            self._strip_remaining = len(self._channel_strip_prefix)
        else:
            self._strip_remaining = 0

    def _consume_strip(self, text: str) -> str:
        """Drop up to ``self._strip_remaining`` characters from the front.

        Called on every thinking-mode emission. Never mutates ``self._buffer``.
        When ``_strip_remaining`` reaches 0 this is a no-op pass-through.
        """
        if self._strip_remaining <= 0 or not text:
            return text
        drop = min(self._strip_remaining, len(text))
        self._strip_remaining -= drop
        return text[drop:]

    def _find_end_marker(self) -> tuple[int, int] | None:
        """Find the earliest end-of-thinking marker in ``self._buffer``.

        Returns (index, length) of the first match among ``self._end_tokens``.
        If multiple markers are present, the earliest wins. Returns ``None``
        if no marker is present in the buffer.
        """
        best: tuple[int, int] | None = None
        for tok in self._end_tokens:
            idx = self._buffer.find(tok)
            if idx < 0:
                continue
            if best is None or idx < best[0]:
                best = (idx, len(tok))
        return best

    def process(self, delta: str) -> list[tuple[str, str]]:
        """Process a delta. Returns list of (block_type, text) pieces."""
        self._buffer += delta
        pieces = []
        self._extract_pieces(pieces)
        return pieces

    def _extract_pieces(self, pieces: list[tuple[str, str]]) -> None:
        """Extract all complete pieces from the buffer."""
        while True:
            if self._in_think:
                end_marker = self._find_end_marker()
                if end_marker is not None:
                    end_idx, end_len = end_marker
                    thinking = self._buffer[:end_idx]
                    self._buffer = self._buffer[end_idx + end_len :]
                    self._in_think = False
                    thinking = self._consume_strip(thinking)
                    if thinking:
                        pieces.append(("thinking", thinking))
                    continue  # Process remainder
                else:
                    # Hold back any suffix that is a partial match of ANY
                    # end token. Use the longest configured end token as the
                    # upper bound on hold-back length.
                    longest = max(len(t) for t in self._end_tokens)
                    held = 0
                    for plen in range(min(longest, len(self._buffer)), 0, -1):
                        suffix = self._buffer[-plen:]
                        if any(t.startswith(suffix) for t in self._end_tokens):
                            held = plen
                            break
                    if held > 0:
                        emit = self._buffer[:-held]
                        self._buffer = self._buffer[-held:]
                        emit = self._consume_strip(emit)
                        if emit:
                            pieces.append(("thinking", emit))
                        return
                    if self._buffer:
                        emit = self._consume_strip(self._buffer)
                        self._buffer = ""
                        if emit:
                            pieces.append(("thinking", emit))
                    return
            else:
                idx = self._buffer.find(self._start_token)
                if idx >= 0:
                    before = self._buffer[:idx]
                    self._buffer = self._buffer[idx + len(self._start_token) :]
                    self._enter_thinking()
                    if before:
                        pieces.append(("text", before))
                    continue  # Process remainder
                else:
                    for plen in range(
                        min(len(self._start_token), len(self._buffer)), 0, -1
                    ):
                        if self._buffer.endswith(self._start_token[:plen]):
                            emit = self._buffer[:-plen]
                            self._buffer = self._buffer[-plen:]
                            if emit:
                                pieces.append(("text", emit))
                            return
                    if self._buffer:
                        pieces.append(("text", self._buffer))
                        self._buffer = ""
                    return

    def flush(self) -> list[tuple[str, str]]:
        """Flush remaining buffer at end of stream."""
        pieces = []
        if self._buffer:
            block_type = "thinking" if self._in_think else "text"
            text = self._buffer
            self._buffer = ""
            if block_type == "thinking":
                text = self._consume_strip(text)
            if text:
                pieces.append((block_type, text))
        self._in_think = False
        self._strip_remaining = 0
        return pieces


# =============================================================================
# Model Detection
# =============================================================================

# Patterns that indicate a multimodal language model (MLLM/VLM)
MLLM_PATTERNS = [
    "-VL-",
    "-VL/",
    "VL-",  # Qwen-VL, Qwen2-VL, Qwen3-VL, etc.
    "llava",
    "LLaVA",  # LLaVA models
    "idefics",
    "Idefics",  # Idefics models
    "paligemma",
    "PaliGemma",  # PaliGemma
    "gemma-3",
    "gemma3",  # Gemma 3 (multimodal)
    "gemma-4",
    "gemma4",  # Gemma 4 (multimodal: vision + audio)
    "medgemma",
    "MedGemma",  # MedGemma (medical multimodal with SigLIP vision encoder)
    "pixtral",
    "Pixtral",  # Pixtral
    "molmo",
    "Molmo",  # Molmo
    "phi3-vision",
    "phi-3-vision",  # Phi-3 Vision
    "cogvlm",
    "CogVLM",  # CogVLM
    "internvl",
    "InternVL",  # InternVL
    "deepseek-vl",
    "DeepSeek-VL",  # DeepSeek-VL
]


def is_mllm_model(model_name: str) -> bool:
    """
    Check if model name indicates a multimodal language model.

    Args:
        model_name: HuggingFace model name or local path

    Returns:
        True if model is detected as MLLM/VLM
    """
    model_lower = model_name.lower()
    for pattern in MLLM_PATTERNS:
        if pattern.lower() in model_lower:
            return True
    return False


# Backwards compatibility alias
is_vlm_model = is_mllm_model


# =============================================================================
# Multimodal Content Extraction
# =============================================================================


def _content_to_text(content) -> str:
    """Extract text from content that can be str, list[ContentPart], or None."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if hasattr(item, "model_dump"):
                item = item.model_dump(exclude_none=True)
            elif hasattr(item, "dict"):
                item = {k: v for k, v in item.dict().items() if v is not None}
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return str(content)


def extract_multimodal_content(
    messages: list[Message],
    preserve_native_format: bool = False,
) -> tuple[list[dict], list[str], list[str]]:
    """
    Extract text content, images, and videos from OpenAI-format messages.

    Handles:
    - Simple text messages
    - Multimodal messages with images/videos
    - Tool call messages (assistant with tool_calls)
    - Tool response messages (role="tool")

    Args:
        messages: List of Message objects
        preserve_native_format: If True, preserve native tool message format
            (role="tool", tool_calls field) instead of converting to text.
            Required for models with native tool support in chat templates
            (e.g., Mistral, Llama 3+, DeepSeek V3).

    Returns:
        Tuple of (processed_messages, images, videos)
        - processed_messages: List of {"role": str, "content": str}
        - images: List of image URLs/paths/base64
        - videos: List of video URLs/paths/base64
    """
    processed_messages = []
    images = []
    videos = []

    for msg in messages:
        # Handle both dict and Pydantic model messages
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content")
        else:
            role = msg.role
            content = msg.content

        # Handle tool response messages (role="tool")
        if role == "tool":
            if isinstance(msg, dict):
                tool_call_id = msg.get("tool_call_id", "") or ""
            else:
                tool_call_id = getattr(msg, "tool_call_id", None) or ""
            tool_content = content if content else ""

            if preserve_native_format:
                # Preserve native tool format for models that support it
                processed_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_content,
                    }
                )
            else:
                # Convert to user role for models without native support
                processed_messages.append(
                    {
                        "role": "user",
                        "content": f"[Tool Result ({tool_call_id})]: {tool_content}",
                    }
                )
            continue

        # Handle assistant messages with tool_calls
        if isinstance(msg, dict):
            tool_calls = msg.get("tool_calls")
        else:
            tool_calls = getattr(msg, "tool_calls", None)

        if role == "assistant" and tool_calls:
            if preserve_native_format:
                # Preserve native tool_calls format
                tool_calls_list = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tc_copy = tc
                    elif hasattr(tc, "model_dump"):
                        tc_copy = tc.model_dump()
                    elif hasattr(tc, "dict"):
                        tc_copy = tc.dict()
                    else:
                        continue

                    # Chat templates (e.g. Qwen3) iterate arguments|items,
                    # but OpenAI API sends arguments as a JSON string.
                    # Parse it into a dict so the template can iterate it.
                    func = tc_copy.get("function") or {}
                    args = func.get("arguments")
                    if isinstance(args, str):
                        try:
                            import json

                            func["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            pass

                    tool_calls_list.append(tc_copy)

                msg_dict = {"role": role, "content": _content_to_text(content)}
                if tool_calls_list:
                    msg_dict["tool_calls"] = tool_calls_list
                processed_messages.append(msg_dict)
            else:
                # Convert tool calls to text for models without native support
                tool_calls_text = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        name = func.get("name", "unknown")
                        args = func.get("arguments", "{}")
                        tool_calls_text.append(f"[Calling tool: {name}({args})]")

                text = _content_to_text(content)
                if tool_calls_text:
                    text = (text + "\n" if text else "") + "\n".join(tool_calls_text)

                processed_messages.append({"role": role, "content": text})
            continue

        # Handle None content
        if content is None:
            processed_messages.append({"role": role, "content": ""})
            continue

        if isinstance(content, str):
            # Simple text message
            processed_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Multimodal message - extract text and media
            text_parts = []
            for item in content:
                # Handle both Pydantic models and dicts
                if hasattr(item, "model_dump"):
                    item = item.model_dump(exclude_none=True)
                elif hasattr(item, "dict"):
                    item = {k: v for k, v in item.dict().items() if v is not None}

                item_type = item.get("type", "")

                if item_type == "text":
                    text_parts.append(item.get("text", ""))

                elif item_type == "image_url":
                    img_url = item.get("image_url", {})
                    if isinstance(img_url, str):
                        images.append(img_url)
                    elif isinstance(img_url, dict):
                        images.append(img_url.get("url", ""))

                elif item_type == "image":
                    images.append(item.get("image", item.get("url", "")))

                elif item_type == "video":
                    videos.append(item.get("video", item.get("url", "")))

                elif item_type == "video_url":
                    vid_url = item.get("video_url", {})
                    if isinstance(vid_url, str):
                        videos.append(vid_url)
                    elif isinstance(vid_url, dict):
                        videos.append(vid_url.get("url", ""))

            # Combine text parts
            combined_text = "\n".join(text_parts) if text_parts else ""
            processed_messages.append({"role": role, "content": combined_text})
        else:
            # Unknown format, try to convert
            processed_messages.append({"role": role, "content": str(content)})

    return processed_messages, images, videos
