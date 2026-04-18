"""Test streaming with reasoning parser and tool calls coexisting."""

from unittest.mock import MagicMock
from vllm_mlx.server import _process_streaming_tool_delta, ToolDeltaResult


def test_reasoning_content_fed_to_tool_parser():
    """Content from reasoning parser should be fed through tool parser."""
    parser = MagicMock()
    parser.extract_tool_calls_streaming.return_value = None  # inside markup

    result = _process_streaming_tool_delta(
        content="<tool_call>start",
        tool_parser=parser,
        tool_accumulated_text="",
        tool_markup_possible=False,
    )

    # Content should be suppressed (inside markup)
    assert result.content is None
    assert result.markup_possible is True
    # Parser was called with content
    parser.extract_tool_calls_streaming.assert_called_once()


def test_reasoning_none_content_not_fed_to_parser():
    """None content (reasoning-only delta) should not be fed to tool parser."""
    parser = MagicMock()

    result = _process_streaming_tool_delta(
        content=None,
        tool_parser=parser,
        tool_accumulated_text="prev",
        tool_markup_possible=False,
    )

    # Passthrough — parser not called
    assert result.content is None
    assert result.accumulated_text == "prev"
    parser.extract_tool_calls_streaming.assert_not_called()


def test_tool_delta_result_named_tuple_access():
    """ToolDeltaResult fields are accessible by name."""
    result = ToolDeltaResult(
        content="hello",
        accumulated_text="hello",
        markup_possible=False,
        tools_found=False,
        tool_result=None,
    )
    assert result.content == "hello"
    assert result.tools_found is False
    assert result.tool_result is None
