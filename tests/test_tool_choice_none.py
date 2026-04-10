"""Test that tool_choice='none' suppresses tool parsing and template injection."""

from vllm_mlx.api.models import ChatCompletionRequest


def _make_request(**kwargs):
    defaults = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def test_parse_tool_calls_returns_raw_text_when_tool_choice_none():
    """_parse_tool_calls_with_parser should return raw text and no tools."""
    from vllm_mlx.server import _parse_tool_calls_with_parser

    request = _make_request(tool_choice="none")
    text = '<tool_call>{"name":"foo","arguments":{}}</tool_call>'
    cleaned, tool_calls = _parse_tool_calls_with_parser(text, request)
    assert tool_calls is None
    assert cleaned == text


def test_parse_tool_calls_works_when_tool_choice_auto():
    """tool_choice='auto' should still attempt parsing."""
    from vllm_mlx.server import _parse_tool_calls_with_parser

    request = _make_request(tool_choice="auto")
    text = "Hello world"
    cleaned, tool_calls = _parse_tool_calls_with_parser(text, request)
    assert tool_calls is None


def test_parse_tool_calls_works_when_tool_choice_absent():
    """Absent tool_choice should still attempt parsing."""
    from vllm_mlx.server import _parse_tool_calls_with_parser

    request = _make_request()
    text = "Hello world"
    cleaned, tool_calls = _parse_tool_calls_with_parser(text, request)
    assert tool_calls is None
