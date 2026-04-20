# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for Anthropic Messages API.

These models define the request and response schemas for the
Anthropic-compatible /v1/messages endpoint, enabling clients like
Claude Code to communicate with vllm-mlx.
"""

import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator

from vllm_mlx.api.effort import ALLOWED_EFFORT_LEVELS

# =============================================================================
# Request Models
# =============================================================================


class AnthropicContentBlock(BaseModel):
    """A content block in an Anthropic message."""

    type: str  # "text", "image", "tool_use", "tool_result", "thinking"
    # text block
    text: str | None = None
    # thinking block (assistant history — clients echo prior reasoning here)
    thinking: str | None = None
    # Optional opaque signature that accompanied our emitted thinking block
    # (PR #14). We don't verify it; accepting it lets typed Anthropic SDK
    # clients round-trip cleanly.
    signature: str | None = None
    # tool_use block
    id: str | None = None
    name: str | None = None
    input: dict | None = None
    # tool_result block
    tool_use_id: str | None = None
    content: str | list | None = None
    is_error: bool | None = None
    # image block
    source: dict | None = None


class AnthropicMessage(BaseModel):
    """A message in an Anthropic conversation."""

    role: str  # "user" | "assistant"
    content: str | list[AnthropicContentBlock]


class AnthropicToolDef(BaseModel):
    """Definition of a tool in Anthropic format."""

    name: str
    description: str | None = None
    input_schema: dict | None = None


class AnthropicRequest(BaseModel):
    """Request for Anthropic Messages API."""

    model: str
    messages: list[AnthropicMessage]
    system: str | list[dict] | None = None
    max_tokens: int  # Required in Anthropic API
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    stop_sequences: list[str] | None = None
    tools: list[AnthropicToolDef] | None = None
    tool_choice: dict | None = None
    metadata: dict | None = None
    top_k: int | None = None
    # Anthropic-native extended-thinking: {"thinking": {"budget_tokens": N}}.
    # Takes precedence only over its absence — the top-level
    # thinking_token_budget below wins when both are set.
    thinking: dict | None = None
    # vllm-mlx extension matching the OpenAI chat-completion surface.
    # Clients that target vLLM upstream's PR #20859 use this name.
    thinking_token_budget: int | None = None
    # Optional wrap-up hint injected before </think> when budget is hit.
    # Capped at 2048 chars — see ChatCompletionRequest for rationale
    # (DoS prevention — DCR Wave-3 finding).
    thinking_budget_message: str | None = Field(default=None, max_length=2048)
    # Claude Code's wire format for effort-based thinking budget selection.
    # Normalized by vllm_mlx.api.effort.resolve_effort. Accepts
    # {"effort": "low"|"medium"|"high"|"xhigh"|"max"}.
    # PR-C C.3: Pydantic validator now rejects unknown effort values at
    # request ingress (HTTP 422) using ALLOWED_EFFORT_LEVELS. Likewise the
    # `thinking` validator rejects non-int / negative budget_tokens.
    output_config: dict | None = None

    @field_validator("output_config")
    @classmethod
    def _validate_output_config(cls, v):
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("output_config must be a dict or null")
        effort = v.get("effort")
        if effort is not None:
            if not isinstance(effort, str) or effort not in ALLOWED_EFFORT_LEVELS:
                raise ValueError(
                    f"output_config.effort={effort!r} is not a recognized "
                    f"level. Allowed: {sorted(ALLOWED_EFFORT_LEVELS)}"
                )
        return v

    @field_validator("thinking")
    @classmethod
    def _validate_thinking(cls, v):
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("thinking must be a dict or null")
        bt = v.get("budget_tokens")
        if bt is not None:
            # bool is a subclass of int in Python — reject explicitly.
            if not isinstance(bt, int) or isinstance(bt, bool):
                raise ValueError(
                    f"thinking.budget_tokens must be int, got "
                    f"{type(bt).__name__}: {bt!r}"
                )
            if bt < 0:
                raise ValueError(f"thinking.budget_tokens must be >= 0, got {bt}")
        return v


# =============================================================================
# Response Models
# =============================================================================


class AnthropicUsage(BaseModel):
    """Token usage for Anthropic response."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    # vllm-mlx extension: token count for the reasoning/thinking portion
    # alone. Anthropic's public API lumps thinking into output_tokens;
    # this field breaks it out so operators can measure thinking depth
    # without re-tokenizing the response. None when no thinking content
    # was produced (non-reasoning model or reasoning parser not attached).
    thinking_tokens: int | None = None


class AnthropicResponseContentBlock(BaseModel):
    """A content block in the Anthropic response."""

    type: str  # "text" | "thinking" | "tool_use"
    text: str | None = None
    # Populated when type == "thinking" — the reasoning content from the
    # attached reasoning parser's extracted <think>...</think> span.
    # Distinct from text so clients can segment reasoning from the answer.
    thinking: str | None = None
    # Required by Anthropic SDK's ThinkingBlock schema: an opaque signature
    # string clients can use to verify/echo the thinking block. We emit a
    # deterministic hash of the thinking text so same-text → same-signature.
    signature: str | None = None
    # tool_use fields
    id: str | None = None
    name: str | None = None
    input: Any | None = None


class AnthropicResponse(BaseModel):
    """Response for Anthropic Messages API."""

    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:24]}")
    type: str = "message"
    role: str = "assistant"
    model: str
    content: list[AnthropicResponseContentBlock]
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)
