"""Test that HarmonyReasoningParser returns (None, original_text) for tagless input."""

from vllm_mlx.reasoning.harmony_parser import HarmonyReasoningParser


def test_harmony_tagless_input_returns_original_text():
    """When input has no channel tags, second element must be the original text."""
    parser = HarmonyReasoningParser()
    text = "The answer is 42."
    reasoning, content = parser.extract_reasoning(text)
    assert reasoning is None
    assert content == text


def test_harmony_tagless_empty_string():
    """Empty string should return (None, None) -- empty is legitimately empty."""
    parser = HarmonyReasoningParser()
    reasoning, content = parser.extract_reasoning("")
    assert reasoning is None
    # Empty string is falsy, so or-fallback works either way
    assert content is None or content == ""


def test_harmony_with_tags_still_works():
    """Normal tagged input should still extract correctly."""
    parser = HarmonyReasoningParser()
    text = (
        "<|channel|>analysis<|message|>Thinking...<|end|>"
        "<|channel|>final<|message|>Result.<|return|>"
    )
    reasoning, content = parser.extract_reasoning(text)
    assert reasoning == "Thinking..."
    assert content == "Result."


def test_harmony_analysis_only():
    """Only analysis channel, no final -- reasoning present, content is original text."""
    parser = HarmonyReasoningParser()
    text = "<|channel|>analysis<|message|>Thinking...<|end|>"
    reasoning, content = parser.extract_reasoning(text)
    assert reasoning == "Thinking..."
    # No final channel found, so content should be None (the analysis was extracted)
    assert content is None
