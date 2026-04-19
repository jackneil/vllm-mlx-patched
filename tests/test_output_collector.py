"""Pinning test for _merge_outputs's noop_reason rule.

When merging two outputs, prefer non-None for thinking_budget_noop_reason so
that a mid-stream marker (set by the happy-path scheduler/engine) survives a
later abort/error output that carries None in the field.
"""

from vllm_mlx.output_collector import RequestOutputCollector
from vllm_mlx.request import RequestOutput


def _mk_output(noop_reason, applied=None):
    # Construct with only the fields _merge_outputs actually reads.
    # Use defaults for everything else; the collector's own tests are
    # source of truth on the full signature.
    return RequestOutput(
        request_id="req-1",
        new_token_ids=[],
        new_text="",
        output_token_ids=[],
        prompt_tokens=0,
        completion_tokens=0,
        thinking_budget_applied=applied,
        thinking_budget_noop_reason=noop_reason,
    )


def test_merge_outputs_preserves_noop_reason_through_abort():
    """Mid-stream noop_reason must survive a later None-carrying abort."""
    collector = RequestOutputCollector()
    mid = _mk_output(noop_reason="mllm_path", applied=False)
    abort = _mk_output(noop_reason=None, applied=None)

    merged = collector._merge_outputs(mid, abort)
    assert merged.thinking_budget_noop_reason == "mllm_path"


def test_merge_outputs_later_non_none_overwrites_earlier_none():
    """A later populated noop_reason replaces a prior None."""
    collector = RequestOutputCollector()
    early = _mk_output(noop_reason=None, applied=None)
    later = _mk_output(noop_reason="simple_engine", applied=False)

    merged = collector._merge_outputs(early, later)
    assert merged.thinking_budget_noop_reason == "simple_engine"
