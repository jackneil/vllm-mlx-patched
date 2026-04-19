"""Schema-level tests for ChatCompletionRequest.reasoning_effort.

Server-side integration (resolver call + thinking_token_budget population)
is covered by the HTTP matrix test in Task 11.
"""

import pytest
from pydantic import ValidationError

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


def test_reasoning_effort_rejects_unknown_level():
    """C.3 — validator rejects unknown effort levels at request ingress.

    Pre-C.3 behavior was to silently fall through to DEFAULT. The Pydantic
    validator now raises ValidationError (HTTP 422) for typos like "hgih".
    """
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest(
            model="t",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="hgih",
        )
    assert "reasoning_effort" in str(exc_info.value)


def test_reasoning_effort_accepts_all_allowed_levels():
    from vllm_mlx.api.effort import ALLOWED_EFFORT_LEVELS

    for level in sorted(ALLOWED_EFFORT_LEVELS):
        req = ChatCompletionRequest(
            model="t",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort=level,
        )
        assert req.reasoning_effort == level


def test_reasoning_effort_allows_none():
    req = ChatCompletionRequest(
        model="t",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert req.reasoning_effort is None
