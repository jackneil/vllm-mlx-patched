"""Non-streaming /v1/messages response must emit type:'thinking' content
blocks in addition to type:'text', matching Anthropic's public API and
the streaming path's behavior."""

from vllm_mlx.api.anthropic_adapter import openai_to_anthropic
from vllm_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    Usage,
)


def _mk_openai_response(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    finish_reason: str = "stop",
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="test-id",
        object="chat.completion",
        created=0,
        model="any-model",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(
                    role="assistant",
                    content=content,
                    reasoning=reasoning,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        ),
    )


def test_reasoning_emits_thinking_block_before_text():
    resp = _mk_openai_response(
        reasoning="Let me think step by step...",
        content="The answer is 42.",
    )
    out = openai_to_anthropic(resp, model="any-model")

    # Two content blocks, thinking-first.
    assert len(out.content) == 2
    assert out.content[0].type == "thinking"
    assert out.content[0].thinking == "Let me think step by step..."
    assert out.content[1].type == "text"
    assert out.content[1].text == "The answer is 42."


def test_reasoning_only_no_text():
    """Pathological model output: reasoning but no final answer. Emit just
    the thinking block — don't fabricate an empty text block."""
    resp = _mk_openai_response(reasoning="Still thinking...", content=None)
    out = openai_to_anthropic(resp, model="any-model")

    assert len(out.content) == 1
    assert out.content[0].type == "thinking"
    assert out.content[0].thinking == "Still thinking..."


def test_text_only_no_reasoning():
    """Non-reasoning model path: single text block, unchanged behavior."""
    resp = _mk_openai_response(content="Hello there.")
    out = openai_to_anthropic(resp, model="any-model")

    assert len(out.content) == 1
    assert out.content[0].type == "text"
    assert out.content[0].text == "Hello there."


def test_empty_preserves_placeholder_text_block():
    """Pre-existing fallback: empty content + no reasoning = empty text block."""
    resp = _mk_openai_response(content=None, reasoning=None)
    out = openai_to_anthropic(resp, model="any-model")

    assert len(out.content) == 1
    assert out.content[0].type == "text"
    assert out.content[0].text == ""
