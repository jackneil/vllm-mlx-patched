"""Test _process_streaming_tool_delta helper function."""

from unittest.mock import MagicMock


def test_no_tool_parser_passthrough():
    from vllm_mlx.server import _process_streaming_tool_delta

    result = _process_streaming_tool_delta(
        content="Hello", tool_parser=None,
        tool_accumulated_text="", tool_markup_possible=False,
    )
    assert result.content == "Hello"
    assert result.tools_found is False
    assert result.tool_result is None


def test_none_content_passthrough():
    from vllm_mlx.server import _process_streaming_tool_delta

    result = _process_streaming_tool_delta(
        content=None, tool_parser="dummy",
        tool_accumulated_text="", tool_markup_possible=False,
    )
    assert result.content is None
    assert result.tools_found is False


def test_fast_path_no_angle_bracket():
    from vllm_mlx.server import _process_streaming_tool_delta

    parser = MagicMock()
    result = _process_streaming_tool_delta(
        content="Hello world", tool_parser=parser,
        tool_accumulated_text="prev", tool_markup_possible=False,
    )
    assert result.content == "Hello world"
    assert result.accumulated_text == "prevHello world"
    assert result.markup_possible is False
    parser.extract_tool_calls_streaming.assert_not_called()


def test_tool_markup_detected_and_suppressed():
    from vllm_mlx.server import _process_streaming_tool_delta

    parser = MagicMock()
    parser.extract_tool_calls_streaming.return_value = None  # inside markup

    result = _process_streaming_tool_delta(
        content="<tool_call>", tool_parser=parser,
        tool_accumulated_text="", tool_markup_possible=False,
    )
    assert result.content is None  # suppressed
    assert result.markup_possible is True
    assert result.tools_found is False


def test_tool_calls_complete():
    from vllm_mlx.server import _process_streaming_tool_delta

    parser = MagicMock()
    tool_calls_data = {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                        "function": {"name": "foo", "arguments": "{}"}}]}
    parser.extract_tool_calls_streaming.return_value = tool_calls_data

    result = _process_streaming_tool_delta(
        content="</tool_call>", tool_parser=parser,
        tool_accumulated_text="<tool_call>{}", tool_markup_possible=True,
    )
    assert result.content is None
    assert result.tools_found is True
    assert result.tool_result == tool_calls_data


def test_content_passthrough_from_parser():
    from vllm_mlx.server import _process_streaming_tool_delta

    parser = MagicMock()
    parser.extract_tool_calls_streaming.return_value = {"content": "normal text"}

    result = _process_streaming_tool_delta(
        content="<some text", tool_parser=parser,
        tool_accumulated_text="", tool_markup_possible=False,
    )
    assert result.content == "normal text"
    assert result.tools_found is False
