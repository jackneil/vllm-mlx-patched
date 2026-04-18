"""Schema-level tests for ChatCompletionRequest.reasoning_effort.

Server-side integration (resolver call + thinking_token_budget population)
is covered by the HTTP matrix test in Task 11.
"""

import pytest

from vllm_mlx.api.models import ChatCompletionRequest


def test_chat_request_accepts_reasoning_effort():
    req = ChatCompletionRequest(
        model="any",
        messages=[],
        reasoning_effort="high",
    )
    assert req.reasoning_effort == "high"


def test_chat_request_reasoning_effort_optional():
    req = ChatCompletionRequest(model="any", messages=[])
    assert req.reasoning_effort is None


def test_chat_request_reasoning_effort_passes_any_string_through():
    """Validation is intentionally permissive — the resolver handles
    unknown strings with a WARN + fallthrough to DEFAULT."""
    req = ChatCompletionRequest(
        model="any",
        messages=[],
        reasoning_effort="some_future_level",
    )
    assert req.reasoning_effort == "some_future_level"
