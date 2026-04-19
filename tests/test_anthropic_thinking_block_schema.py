"""Anthropic SDK's ThinkingBlock model requires a `signature` field on
every type:"thinking" content block. Our adapter must populate it with a
deterministic, opaque string so same-text always produces same-signature.

Contract: we emit `"vllm-mlx:" + sha256(thinking_text).hexdigest()[:32]`.
The prefix distinguishes our signatures from Anthropic's own server-signed
values, and the hex digest makes repeated requests on the same reasoning
text stable (useful for caching/replay)."""

import hashlib

from vllm_mlx.api.anthropic_adapter import openai_to_anthropic
from vllm_mlx.api.anthropic_models import AnthropicResponseContentBlock
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
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


def test_content_block_model_accepts_signature_field():
    """Schema: the Pydantic model has a `signature` field (str | None)."""
    block = AnthropicResponseContentBlock(
        type="thinking",
        thinking="some reasoning",
        signature="vllm-mlx:deadbeef",
    )
    assert block.signature == "vllm-mlx:deadbeef"


def test_thinking_block_has_deterministic_signature():
    """Signature must be populated on thinking blocks and must be a stable
    function of the thinking text so identical reasoning → identical sig."""
    text = "Let me think step by step about this problem..."
    resp = _mk_openai_response(reasoning=text, content="42")

    out = openai_to_anthropic(resp, model="any-model")

    thinking_blocks = [b for b in out.content if b.type == "thinking"]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0].signature is not None

    expected = "vllm-mlx:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    assert thinking_blocks[0].signature == expected

    # Stability: a second adapter call on the same reasoning text yields the
    # same signature.
    out2 = openai_to_anthropic(resp, model="any-model")
    assert out2.content[0].signature == thinking_blocks[0].signature


def test_non_thinking_blocks_have_no_signature():
    """signature is only meaningful on thinking blocks; text/tool_use blocks
    leave it as None (field exists but is unset)."""
    resp = _mk_openai_response(reasoning="x", content="answer")
    out = openai_to_anthropic(resp, model="any-model")

    for block in out.content:
        if block.type != "thinking":
            assert block.signature is None, (
                f"{block.type} block should not have a signature"
            )


def test_different_reasoning_produces_different_signatures():
    """Sanity: the hash is not a constant."""
    r1 = openai_to_anthropic(_mk_openai_response(reasoning="one"), model="m")
    r2 = openai_to_anthropic(_mk_openai_response(reasoning="two"), model="m")
    assert r1.content[0].signature != r2.content[0].signature


# ---- Inbound schema sentinels (round-trip protection) ----


def test_inbound_content_block_accepts_thinking_type():
    """Pins that AnthropicContentBlock accepts `type: "thinking"` with a
    `thinking` field on inbound. If an upstream rebase tightens `type` to
    a Literal enum excluding "thinking", this sentinel catches the
    regression that would silently 422 every Qwen3+ assistant-history
    thinking block in multi-turn conversations."""
    from vllm_mlx.api.anthropic_models import AnthropicContentBlock

    block = AnthropicContentBlock(
        type="thinking",
        thinking="prior reasoning from this assistant turn",
        signature="vllm-mlx:deadbeef",
    )
    assert block.type == "thinking"
    assert block.thinking == "prior reasoning from this assistant turn"
    assert block.signature == "vllm-mlx:deadbeef"


def test_inbound_content_block_thinking_defaults_to_none():
    """Fields default to None so old clients that don't send them still
    parse cleanly."""
    from vllm_mlx.api.anthropic_models import AnthropicContentBlock

    block = AnthropicContentBlock(type="text", text="hello")
    assert block.thinking is None
    assert block.signature is None
