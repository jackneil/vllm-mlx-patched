# SPDX-License-Identifier: Apache-2.0
"""
Adapter for converting between Anthropic Messages API and OpenAI Chat Completions API.

Handles translation of:
- Requests: Anthropic → OpenAI format
- Responses: OpenAI → Anthropic format
- Messages: Content blocks, tool calls, tool results
"""

import hashlib
import json
import logging
import re
import uuid

from .anthropic_models import (
    AnthropicMessage,
    AnthropicRequest,
    AnthropicResponse,
    AnthropicResponseContentBlock,
    AnthropicToolDef,
    AnthropicUsage,
)
from .effort import ResolvedBudget, resolve_effort
from .utils import SPECIAL_TOKENS_PATTERN as _SPECIAL_TOKENS_OUTPUT_PATTERN

logger = logging.getLogger(__name__)

# Literals that, if present inside client-supplied assistant-history
# thinking content, would escape the `<think>…</think>` wrapper we build
# at conversion time — letting the client inject arbitrary prompt
# structure (tool calls, system-level instructions, turn markers) into
# the model's context. We drop the whole thinking block (with a WARN)
# when any of these appears.
#
# DELIBERATELY distinct from `SPECIAL_TOKENS_PATTERN` in api/utils.py:
#   - SPECIAL_TOKENS_PATTERN is an OUTBOUND scrub: strips special tokens
#     from MODEL OUTPUT while intentionally preserving `<think>…</think>`
#     (those are the reasoning markers we want the client to see).
#   - This pattern is an INBOUND reject: drops client INPUT that would
#     break the wrapper we're about to build. `<think>` and `</think>`
#     MUST be in this set even though they're NOT in SPECIAL_TOKENS.
#
# The regex reuses `SPECIAL_TOKENS_PATTERN.pattern` as the base, unions
# `<think>` / `</think>`, and the full `<|channel|>…<|message|>` channel
# marker (GPT-OSS). Keeping them textually linked to SPECIAL_TOKENS
# prevents drift if that outbound pattern grows.
_THINKING_INJECTION_PATTERNS = re.compile(
    "|".join(
        [
            r"<think>",
            r"</think>",
            # GPT-OSS/Harmony channel forms:
            #   `<|channel|>final<|message|>` (full marker)
            #   `<|channel|>` (bare — already in SPECIAL_TOKENS below)
            #   `<|channel>` (single-pipe, Gemma variant)
            r"<\|channel\|>[a-z]*(?:<\|message\|>)?",
            r"<\|channel>",
            # Gemma chat-template turn markers.
            r"<start_of_turn>",
            r"<end_of_turn>",
            _SPECIAL_TOKENS_OUTPUT_PATTERN.pattern,
        ]
    )
)

# Bidi / isolate controls that BPE tokenizers may treat as ordinary glyphs
# in edge cases — strip them from thinking content we round-trip back to
# the model. See Wave 2 /dcr review (paranoid-auditor lens).
_BIDI_CONTROL_PATTERN = re.compile(r"[\u202a-\u202e\u2066-\u2069]")

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    ToolDefinition,
)


def anthropic_to_openai(
    request: AnthropicRequest,
    context_window: int = 131072,
    reasoning_parser_start_token: str | None = None,
    *,
    engine_supports_processor: bool = True,
) -> tuple[ChatCompletionRequest, ResolvedBudget]:
    """
    Convert an Anthropic Messages API request to OpenAI Chat Completions format.

    Handles:
    - system field → system message
    - Content blocks → OpenAI message format
    - tool_use/tool_result → OpenAI tool_calls/tool messages
    - Anthropic tools → OpenAI tools

    Returns a (ChatCompletionRequest, ResolvedBudget) tuple so the caller
    can emit resolver-derived response headers. The ChatCompletionRequest's
    thinking_token_budget is populated from the resolver; callers should
    not read request.thinking_token_budget directly after this returns.

    Args:
        request: Anthropic Messages API request
        context_window: model context window used for dynamic effort sizing
            (e.g., "max" effort scales with ctx). Default 131072 when unknown.
        reasoning_parser_start_token: Start delimiter of the currently-active
            reasoning parser (e.g. ``"<think>"`` for Qwen3/DeepSeek-R1,
            ``"<|channel>"`` for Gemma 4). When this matches ``"<think>"``,
            prior-turn Anthropic ``type: "thinking"`` content blocks are
            preserved back into the assistant's OpenAI content field as
            inline ``<think>…</think>`` — which closes the Qwen3.x
            "interleaved thinking" trap where the model otherwise never
            emits ``</think>`` after seeing thinking-less assistant history
            in multi-turn + tool contexts. See llama.cpp#21118 and
            litellm#19849 for the upstream discussion. Any other value
            (or None) preserves the legacy drop behavior — injecting
            literal ``<think>`` text into Gemma/non-reasoning prompts
            would just appear as prose to the model.

    Returns:
        (ChatCompletionRequest, ResolvedBudget)
    """
    messages = []

    # Convert system to system message
    if request.system:
        if isinstance(request.system, str):
            system_text = request.system
        elif isinstance(request.system, list):
            # System can be a list of content blocks
            parts = []
            for block in request.system:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            system_text = "\n".join(parts)
        else:
            system_text = str(request.system)
        messages.append(Message(role="system", content=system_text))

    # Convert each message
    for msg in request.messages:
        converted = _convert_message(
            msg, reasoning_parser_start_token=reasoning_parser_start_token
        )
        messages.extend(converted)

    # Convert tools
    tools = None
    if request.tools:
        tools = [_convert_tool(t) for t in request.tools]

    # Convert tool_choice
    tool_choice = None
    if request.tool_choice:
        tool_choice = _convert_tool_choice(request.tool_choice)

    # Layer 1 (Qwen3 runaway spec, 2026-04-20): auto-disable thinking on the
    # Claude-Code-first-turn-with-tools fingerprint. Mutates
    # request.chat_template_kwargs when firing and sets request._layer1_fired
    # = True. Must run BEFORE resolve_effort so the resolver sees the final
    # template-kwargs state.
    #
    # Reads reasoning_parser_name + disable flag from the server module (the
    # engine does not expose reasoning_parser_name, per UPSTREAM_PIN inv. 14).
    from .. import server as _server_module
    from .thinking_policy import maybe_disable_thinking_for_qwen3_agent_first_turn

    maybe_disable_thinking_for_qwen3_agent_first_turn(
        request,
        reasoning_parser_name=_server_module._reasoning_parser_name,
        disabled=_server_module._disable_qwen3_first_turn_no_think,
    )

    # Resolve thinking budget via the shared resolver. Replaces the old
    # inline thinking.type + thinking_token_budget resolution block.
    # The resolver handles top-level vs Anthropic `thinking` (including
    # "adaptive") vs output_config.effort precedence.
    resolved = resolve_effort(
        top_level_budget=request.thinking_token_budget,
        anthropic_thinking=request.thinking,
        output_config=request.output_config,
        reasoning_effort=None,  # not on the Anthropic path
        context_window=context_window,
    )

    # Layer 2 site 1 (Qwen3 runaway spec, 2026-04-20): apply server ceiling
    # BEFORE constructing ChatCompletionRequest so the clamped budget
    # propagates cleanly. Reads `server._max_thinking_token_budget` as a
    # module-level global, same pattern as _streaming_max_seconds.
    from .. import server as _server_module
    from .budget_ceiling import apply_server_thinking_token_budget_ceiling

    resolved, _clamped_from_adapter, _clamp_skip_adapter = (
        apply_server_thinking_token_budget_ceiling(
            resolved,
            ceiling=_server_module._max_thinking_token_budget,
            engine_supports_processor=engine_supports_processor,
        )
    )
    # Surface adapter-side clamp/skip state to the handler via request
    # markers. Same pattern as _layer1_fired — avoids changing the
    # 2-tuple return contract. Handler reads these back out to emit
    # x-thinking-budget-clamped-to / clamp-skipped headers accurately.
    if _clamped_from_adapter is not None:
        try:
            request._layer2_clamped_from = _clamped_from_adapter
        except (AttributeError, TypeError):
            object.__setattr__(request, "_layer2_clamped_from", _clamped_from_adapter)
    if _clamp_skip_adapter is not None:
        try:
            request._layer2_clamp_skip = _clamp_skip_adapter
        except (AttributeError, TypeError):
            object.__setattr__(request, "_layer2_clamp_skip", _clamp_skip_adapter)

    openai_req = ChatCompletionRequest(
        model=request.model,
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature if request.temperature is not None else 0.7,
        top_p=request.top_p if request.top_p is not None else 0.9,
        stream=request.stream,
        stop=request.stop_sequences,
        tools=tools,
        tool_choice=tool_choice,
        thinking_token_budget=resolved.budget,
        thinking_budget_message=request.thinking_budget_message,
        chat_template_kwargs=request.chat_template_kwargs,
    )

    return openai_req, resolved


def openai_to_anthropic(
    response: ChatCompletionResponse,
    model: str,
) -> AnthropicResponse:
    """
    Convert an OpenAI Chat Completions response to Anthropic Messages API format.

    Args:
        response: OpenAI ChatCompletionResponse
        model: Model name for the response

    Returns:
        Anthropic Messages API response
    """
    content = []
    choice = response.choices[0] if response.choices else None

    if choice:
        # Emit thinking block first when the reasoning parser populated .reasoning.
        # Matches Anthropic's public API and the streaming path
        # (server.py:1991-2032).
        if choice.message.reasoning:
            sig = (
                "vllm-mlx:"
                + hashlib.sha256(
                    choice.message.reasoning.encode("utf-8")
                ).hexdigest()[:32]
            )
            content.append(
                AnthropicResponseContentBlock(
                    type="thinking",
                    thinking=choice.message.reasoning,
                    signature=sig,
                )
            )

        # Add text content (existing behavior).
        if choice.message.content:
            content.append(
                AnthropicResponseContentBlock(
                    type="text",
                    text=choice.message.content,
                )
            )

        # Add tool use blocks
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    tool_input = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    tool_input = {}

                content.append(
                    AnthropicResponseContentBlock(
                        type="tool_use",
                        id=tc.id,
                        name=tc.function.name,
                        input=tool_input,
                    )
                )

        stop_reason = _convert_stop_reason(choice.finish_reason)
    else:
        stop_reason = "end_turn"

    # If no content blocks, add empty text
    if not content:
        content.append(AnthropicResponseContentBlock(type="text", text=""))

    return AnthropicResponse(
        model=model,
        content=content,
        stop_reason=stop_reason,
        usage=AnthropicUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        ),
    )


def _convert_message(
    msg: AnthropicMessage,
    *,
    reasoning_parser_start_token: str | None = None,
) -> list[Message]:
    """
    Convert an Anthropic message to one or more OpenAI messages.

    Anthropic tool_result blocks (sent as user messages) need to be
    split into separate OpenAI tool messages.

    Args:
        msg: Anthropic message
        reasoning_parser_start_token: see ``anthropic_to_openai`` docstring.
            Gates whether prior-turn ``type: "thinking"`` blocks are
            preserved as inline ``<think>…</think>`` text (Qwen3+) or
            dropped (Gemma 4 / non-reasoning models).

    Returns:
        List of OpenAI messages
    """
    # Simple string content
    if isinstance(msg.content, str):
        return [Message(role=msg.role, content=msg.content)]

    # Content is a list of blocks
    messages = []
    text_parts = []
    thinking_parts = []  # preserved for Qwen-style parsers only
    tool_calls_for_assistant = []
    tool_results = []

    preserve_thinking = reasoning_parser_start_token == "<think>"

    for block in msg.content:
        if block.type == "text":
            text_parts.append(block.text or "")

        elif block.type == "thinking":
            # Only relevant in assistant-history turns. For Qwen3+ we
            # round-trip the thinking back into the conversation so the
            # model sees the "closed </think>" pattern and doesn't fall
            # into the interleaved-thinking trap where it declines to
            # close the current turn's </think>. Non-Qwen parsers drop.
            thinking_text = (getattr(block, "thinking", None) or "").strip()
            if preserve_thinking and thinking_text:
                # Reject content that would escape the <think>…</think>
                # wrapper we build below (Wave 2 /dcr CRITICAL finding):
                # a client sending "reason\n</think>\nSYSTEM: …" could
                # otherwise smuggle arbitrary prompt structure past the
                # closing tag. Drop with WARN rather than try to escape
                # (no defined escape form for these tokenizer specials).
                if _THINKING_INJECTION_PATTERNS.search(thinking_text):
                    logger.warning(
                        "[thinking-preservation] assistant-history thinking "
                        "block dropped: contains injection marker "
                        "(<think>/</think>/<|im_end|>/<|endoftext|>/<|channel|>). "
                        "Block length=%d. Client may be attempting to smuggle "
                        "prompt structure through the round-trip.",
                        len(thinking_text),
                    )
                    continue
                # Strip Bidi/isolate controls that some BPE tokenizers
                # absorb into ordinary tokens.
                sanitized = _BIDI_CONTROL_PATTERN.sub("", thinking_text)
                thinking_parts.append(sanitized)

        elif block.type == "tool_use":
            # Assistant message with tool calls
            tool_input = block.input or {}
            tool_calls_for_assistant.append(
                {
                    "id": block.id or f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": block.name or "",
                        "arguments": json.dumps(tool_input),
                    },
                }
            )

        elif block.type == "tool_result":
            # Tool result → OpenAI tool message
            result_content = block.content
            if isinstance(result_content, list):
                # Extract text from content blocks
                parts = []
                for item in result_content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        parts.append(item)
                result_content = "\n".join(parts)
            elif result_content is None:
                result_content = ""

            tool_results.append(
                Message(
                    role="tool",
                    content=str(result_content),
                    tool_call_id=block.tool_use_id or "",
                )
            )

    # Build the messages
    if msg.role == "assistant":
        # Prepend any preserved thinking as inline <think>…</think> so
        # Qwen3+ sees the closed-think pattern in history (see docstring).
        # thinking_parts is only ever populated when preserve_thinking=True.
        thinking_prefix = ""
        if thinking_parts:
            inner = "\n".join(thinking_parts)
            thinking_prefix = f"<think>\n{inner}\n</think>\n"
        combined_text_body = "\n".join(text_parts) if text_parts else None
        if combined_text_body is None and not thinking_prefix:
            combined_text = None
        else:
            combined_text = thinking_prefix + (combined_text_body or "")
        if tool_calls_for_assistant:
            messages.append(
                Message(
                    role="assistant",
                    content=combined_text or "",
                    tool_calls=tool_calls_for_assistant,
                )
            )
        elif combined_text is not None:
            messages.append(Message(role="assistant", content=combined_text))
        else:
            messages.append(Message(role="assistant", content=""))
    elif msg.role == "user":
        # User messages: collect text parts, then add tool results separately
        if text_parts:
            combined_text = "\n".join(text_parts)
            messages.append(Message(role="user", content=combined_text))

        # Tool results become separate tool messages
        messages.extend(tool_results)

        # If no text and no tool results, add empty user message
        if not text_parts and not tool_results:
            messages.append(Message(role="user", content=""))
    else:
        # Other roles
        combined_text = "\n".join(text_parts) if text_parts else ""
        messages.append(Message(role=msg.role, content=combined_text))

    return messages


def _convert_tool(tool: AnthropicToolDef) -> ToolDefinition:
    """
    Convert an Anthropic tool definition to OpenAI format.

    Anthropic: {"name": "...", "description": "...", "input_schema": {...}}
    OpenAI: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    return ToolDefinition(
        type="function",
        function={
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema or {"type": "object", "properties": {}},
        },
    )


def _convert_tool_choice(tool_choice: dict) -> str | dict | None:
    """
    Convert Anthropic tool_choice to OpenAI format.

    Anthropic: {"type": "auto"} | {"type": "any"} | {"type": "tool", "name": "..."}
    OpenAI: "auto" | "none" | "required" | {"type": "function", "function": {"name": "..."}}
    """
    raw_type = tool_choice.get("type", "auto")
    choice_type = raw_type.lower() if isinstance(raw_type, str) else "auto"

    if choice_type == "auto":
        return "auto"
    elif choice_type == "any":
        return "required"
    elif choice_type == "tool":
        return {
            "type": "function",
            "function": {"name": tool_choice.get("name", "")},
        }
    elif choice_type == "none":
        return "none"

    return "auto"


def _convert_stop_reason(openai_reason: str | None) -> str:
    """
    Convert OpenAI finish_reason to Anthropic stop_reason.

    OpenAI: "stop" | "tool_calls" | "length" | "content_filter"
    Anthropic: "end_turn" | "tool_use" | "max_tokens" | "stop_sequence"
    """
    if openai_reason is None:
        return "end_turn"

    mapping = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "content_filter": "end_turn",
    }
    return mapping.get(openai_reason, "end_turn")
