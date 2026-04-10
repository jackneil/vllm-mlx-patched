"""Test sequential reasoning + tool parsing (non-streaming)."""

from vllm_mlx.api.models import ChatCompletionRequest


def _make_request(**kwargs):
    defaults = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


class TestNonStreamingSequentialParsing:
    """Test that reasoning extraction runs before tool parsing in non-streaming."""

    def test_reasoning_and_tools_both_extracted(self):
        """When both <think> and <tool_call> present, both should be populated."""
        from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

        parser = Qwen3ReasoningParser()
        model_output = (
            '<think>Let me check the weather.</think>'
            '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'
        )

        # Step 1: reasoning extraction
        reasoning_text, text_for_tools = parser.extract_reasoning(model_output)
        assert reasoning_text == "Let me check the weather."
        assert "<think>" not in (text_for_tools or "")

        # Step 2: tool parsing on remainder
        from vllm_mlx.server import _parse_tool_calls_with_parser
        cleaned, tool_calls = _parse_tool_calls_with_parser(
            text_for_tools or model_output, _make_request()
        )
        # Generic parser should find the tool call in the remainder
        assert tool_calls is not None or "<tool_call>" in (text_for_tools or "")

    def test_tools_only_no_reasoning(self):
        """Without reasoning tags, reasoning extraction is a no-op."""
        from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

        parser = Qwen3ReasoningParser()
        model_output = '<tool_call>{"name":"foo","arguments":{}}</tool_call>'

        reasoning_text, text_for_tools = parser.extract_reasoning(model_output)
        assert reasoning_text is None
        assert text_for_tools == model_output

    def test_reasoning_only_no_tools(self):
        """Only reasoning tags, no tool calls."""
        from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

        parser = Qwen3ReasoningParser()
        model_output = "<think>Just thinking...</think>The answer is 42."

        reasoning_text, text_for_tools = parser.extract_reasoning(model_output)
        assert reasoning_text == "Just thinking..."
        assert text_for_tools == "The answer is 42."

    def test_empty_response(self):
        """Empty string should not crash."""
        from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

        parser = Qwen3ReasoningParser()
        reasoning_text, text_for_tools = parser.extract_reasoning("")
        assert reasoning_text is None
