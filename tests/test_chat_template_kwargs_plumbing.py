# SPDX-License-Identifier: Apache-2.0
"""Verifies the chat_template_kwargs plumbing that Layer 1 depends on.

If this test fails, Layer 1 is a no-op — a client-set (or Layer-1-mutated)
chat_template_kwargs dict must reach the tokenizer's apply_chat_template
call. Without this plumbing, Layer 1 is silent.
"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from vllm_mlx import server as srv
from vllm_mlx.api.anthropic_adapter import anthropic_to_openai
from vllm_mlx.api.anthropic_models import AnthropicRequest


def test_anthropic_request_declares_chat_template_kwargs():
    """Foundational: the Pydantic model must accept the field."""
    req = AnthropicRequest(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert req.chat_template_kwargs == {"enable_thinking": False}


def test_anthropic_adapter_forwards_chat_template_kwargs():
    """Adapter must copy chat_template_kwargs from AnthropicRequest →
    ChatCompletionRequest so it survives the adapter round-trip."""
    req = AnthropicRequest(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        chat_template_kwargs={"enable_thinking": False, "tool_choice": "auto"},
    )
    openai_req, _ = anthropic_to_openai(req, context_window=131072)
    assert openai_req.chat_template_kwargs == {
        "enable_thinking": False,
        "tool_choice": "auto",
    }


def test_batched_engine_apply_chat_template_merges_chat_template_kwargs():
    """BatchedEngine._apply_chat_template must merge a passed
    chat_template_kwargs dict into its internal template_kwargs before
    calling the tokenizer's apply_chat_template."""
    from vllm_mlx.engine.batched import BatchedEngine

    fake_tokenizer = MagicMock()
    fake_tokenizer.apply_chat_template = MagicMock(return_value="prompt text")
    engine = MagicMock(spec=BatchedEngine)
    engine.tokenizer = fake_tokenizer
    # Force the text/tokenizer branch (not the MLLM/processor branch).
    engine._is_mllm = False
    engine._processor = None

    BatchedEngine._apply_chat_template(
        engine,
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        chat_template_kwargs={"enable_thinking": False},
    )

    fake_tokenizer.apply_chat_template.assert_called_once()
    call_kwargs = fake_tokenizer.apply_chat_template.call_args.kwargs
    assert call_kwargs.get("enable_thinking") is False
    # Existing kwargs still present
    assert call_kwargs.get("tokenize") is False
    assert call_kwargs.get("add_generation_prompt") is True


def test_server_reasoning_parser_name_defaults_to_none():
    """Module-level attribute exists with correct default."""
    assert srv._reasoning_parser_name is None


def test_server_forwards_chat_template_kwargs_through_engine_on_anthropic_stream():
    """Integration: a client-set chat_template_kwargs on /v1/messages
    must appear in the captured engine.stream_chat kwargs."""
    captured = []

    def stream_chat(**kwargs):
        captured.append(kwargs)

        async def _gen():
            yield MagicMock(
                new_text="hi",
                text="hi",
                prompt_tokens=10,
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
                thinking_budget_applied=None,
                thinking_budget_noop_reason=None,
            )

        return _gen()

    fake = MagicMock()
    fake.preserve_native_tool_format = False
    fake.model_name = "test-model"
    fake.tokenizer = None
    fake._is_mllm = False
    fake.is_mllm = False
    from vllm_mlx.engine.batched import BatchedEngine

    fake.__class__ = BatchedEngine
    fake._reasoning_parser = MagicMock()
    fake._reasoning_parser.start_token = "<think>"
    fake._reasoning_parser.end_tokens = ["</think>"]
    fake._reasoning_parser.channel_strip_prefix = None
    fake.stream_chat = stream_chat

    with (
        patch.object(srv, "_engine", fake),
        patch.object(srv, "_model_name", fake.model_name),
        patch.object(srv, "_reasoning_parser", fake._reasoning_parser),
        patch.object(srv, "_reasoning_parser_name", "qwen3"),
    ):
        client = TestClient(srv.app)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"enable_thinking": False},
                "max_tokens": 200,
                "stream": True,
            },
        )
        for _ in resp.iter_lines():
            pass

    assert len(captured) == 1
    assert captured[0].get("chat_template_kwargs") == {"enable_thinking": False}
