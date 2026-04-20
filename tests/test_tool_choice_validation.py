"""Test tool_choice validation and normalization."""

import pytest
from pydantic import ValidationError

from vllm_mlx.api.models import ChatCompletionRequest


def test_tool_choice_none_lowercase():
    req = ChatCompletionRequest(model="test", messages=[], tool_choice="none")
    assert req.tool_choice == "none"


def test_tool_choice_none_uppercase():
    req = ChatCompletionRequest(model="test", messages=[], tool_choice="None")
    assert req.tool_choice == "none"


def test_tool_choice_none_allcaps():
    req = ChatCompletionRequest(model="test", messages=[], tool_choice="NONE")
    assert req.tool_choice == "none"


def test_tool_choice_none_with_whitespace():
    req = ChatCompletionRequest(model="test", messages=[], tool_choice="  none  ")
    assert req.tool_choice == "none"


def test_tool_choice_auto():
    req = ChatCompletionRequest(model="test", messages=[], tool_choice="auto")
    assert req.tool_choice == "auto"


def test_tool_choice_required():
    req = ChatCompletionRequest(model="test", messages=[], tool_choice="required")
    assert req.tool_choice == "required"


def test_tool_choice_absent():
    req = ChatCompletionRequest(model="test", messages=[])
    assert req.tool_choice is None


def test_tool_choice_dict_function_passthrough():
    tc = {"type": "function", "function": {"name": "get_weather"}}
    req = ChatCompletionRequest(model="test", messages=[], tool_choice=tc)
    assert req.tool_choice == tc


def test_tool_choice_dict_type_none_converted():
    """Dict-form {"type": "none"} should be normalized to string "none"."""
    req = ChatCompletionRequest(model="test", messages=[], tool_choice={"type": "none"})
    assert req.tool_choice == "none"


def test_tool_choice_dict_type_none_uppercase_converted():
    req = ChatCompletionRequest(model="test", messages=[], tool_choice={"type": "None"})
    assert req.tool_choice == "none"


def test_tool_choice_invalid_string_rejected():
    with pytest.raises(ValidationError, match="Invalid tool_choice string"):
        ChatCompletionRequest(model="test", messages=[], tool_choice="maybe")


def test_tool_choice_integer_rejected():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="test", messages=[], tool_choice=0)


def test_tool_choice_boolean_rejected():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="test", messages=[], tool_choice=True)


# --- Anthropic adapter tests ---

from vllm_mlx.api.anthropic_adapter import _convert_tool_choice


def test_anthropic_tool_choice_none_lowercase():
    assert _convert_tool_choice({"type": "none"}) == "none"


def test_anthropic_tool_choice_none_uppercase():
    assert _convert_tool_choice({"type": "None"}) == "none"


def test_anthropic_tool_choice_auto():
    assert _convert_tool_choice({"type": "auto"}) == "auto"


def test_anthropic_tool_choice_any():
    assert _convert_tool_choice({"type": "any"}) == "required"
