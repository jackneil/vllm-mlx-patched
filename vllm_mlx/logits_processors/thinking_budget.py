# SPDX-License-Identifier: Apache-2.0
"""Thinking-token budget logits processor.

Ported from vllm-project/vllm PR #20859 (merged 2026-03-24):
    vllm/v1/sample/logits_processor/builtin.py::ThinkingTokenBudgetLogitsProcessor

Adapted for mlx_lm.generate.BatchGenerator's per-request Callable interface:
    processor(tokens: mx.array, logits: mx.array [1, vocab]) -> mx.array

The state machine, multi-token-ID handling, and prompt-pre-injection detection
are faithful to upstream. The logit-forcing mechanism uses a +1e9 bias on the
forced token (matching upstream's approach in the vLLM apply() method) so the
sampler naturally selects it regardless of temperature/top-k/top-p.

The optional thinking_budget_message feature mirrors vllm PR #37112: when set,
the tokenized message is prepended to the forced end sequence so the model
reads a wrap-up hint in its own context and writes a coherent transition to
the answer phase (instead of a potentially truncated mid-sentence cut).
"""

from __future__ import annotations

from typing import List, Optional

import mlx.core as mx

# Large positive bias — matches upstream vLLM's 1e9.
_FORCE_BIAS = 1e9


def _find_last_subsequence(haystack: List[int], needle: List[int]) -> int:
    """Return the start index of the last occurrence of needle in haystack,
    or -1 if not found. Empty needle returns -1."""
    if not needle or len(needle) > len(haystack):
        return -1
    n = len(needle)
    for i in range(len(haystack) - n, -1, -1):
        if haystack[i : i + n] == needle:
            return i
    return -1


class ThinkingTokenBudgetLogitsProcessor:
    """Per-request logits processor that caps thinking tokens.

    When the per-request token count inside <think>…</think> reaches the
    configured budget, the processor switches into 'end-forcing' mode: for
    the next len(force_sequence) steps it biases the forced token's logit
    to +1e9 so the sampler emits exactly that token. The force_sequence is
    end_token_ids, optionally preceded by the tokenized
    thinking_budget_message.

    Multi-token delimiters are handled natively.
    """

    def __init__(
        self,
        *,
        budget: int,
        start_token_ids: List[int],
        end_token_ids: List[int],
        message_token_ids: Optional[List[int]] = None,
        prompt_token_ids: Optional[List[int]] = None,
    ) -> None:
        if budget < 0:
            raise ValueError(f"budget must be >= 0, got {budget}")
        if not start_token_ids or not end_token_ids:
            raise ValueError("start_token_ids and end_token_ids must be non-empty")

        self._budget = budget
        self._start_ids = list(start_token_ids)
        self._end_ids = list(end_token_ids)
        # Force sequence = optional message prefix, then end tokens.
        self._force_sequence: List[int] = (
            list(message_token_ids) if message_token_ids else []
        ) + self._end_ids

        # Per-request state.
        self._in_think = False
        self._in_end = False
        self._think_count = 0
        self._end_count = 0  # index into _force_sequence
        self._prev_output_len = 0  # number of output tokens last seen
        self._prompt_len = len(prompt_token_ids) if prompt_token_ids else 0

        if prompt_token_ids:
            self._init_from_prompt(list(prompt_token_ids))

    def _init_from_prompt(self, prompt: List[int]) -> None:
        last_start = _find_last_subsequence(prompt, self._start_ids)
        last_end = _find_last_subsequence(prompt, self._end_ids)
        if last_start > last_end:
            self._in_think = True
            self._think_count = len(prompt) - (last_start + len(self._start_ids))
            # If prompt already burned the budget, jump straight to end mode.
            if self._budget == 0 or self._think_count >= self._budget:
                self._in_think = False
                self._in_end = True
                self._end_count = 0

    def _advance_state(self, output: List[int]) -> None:
        """Update state from the full output-tokens list (excludes prompt)."""
        cur_len = len(output)
        if cur_len <= self._prev_output_len:
            return

        new_tokens = output[self._prev_output_len : cur_len]
        self._prev_output_len = cur_len

        if self._in_end:
            # We are emitting the force sequence. Advance by len(new_tokens).
            self._end_count += len(new_tokens)
            if self._end_count >= len(self._force_sequence):
                # Full sequence emitted — leave end mode and reset.
                self._in_end = False
                self._end_count = 0
                self._think_count = 0
            return

        # Scan the recent output window for start/end occurrences.
        # Window is sized to catch sequences that might span the prev-cur
        # boundary (upstream vLLM does the same).
        start_len = len(self._start_ids)
        end_len = len(self._end_ids)
        window_start = max(0, self._prev_output_len - max(start_len, end_len))
        recent = output[window_start:]

        recent_start = _find_last_subsequence(recent, self._start_ids)
        recent_end = _find_last_subsequence(recent, self._end_ids)

        if not self._in_think:
            # Haven't entered thinking yet. Did <think> appear in recent?
            if recent_start > recent_end:
                abs_start = window_start + recent_start
                self._in_think = True
                self._think_count = cur_len - (abs_start + start_len)
        else:
            # We're in thinking mode. Did </think> appear since last step?
            # (Model closed on its own.) Or did a new <think> appear?
            if recent_end > recent_start and recent_end >= 0:
                # Model closed.
                self._in_think = False
                self._think_count = 0
            elif recent_start >= 0 and recent_start > recent_end:
                # A new <think> — reset count from there.
                abs_start = window_start + recent_start
                self._think_count = cur_len - (abs_start + start_len)
            else:
                # Still thinking; increment by whatever new tokens landed.
                self._think_count += len(new_tokens)

        # Check budget transition.
        if (
            self._in_think
            and self._think_count >= self._budget
        ):
            self._in_think = False
            self._in_end = True
            self._end_count = 0

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """Return modified logits.

        `tokens` is the FULL token history for this request (prompt + output)
        as an mx.array of ints. We separate prompt from output by the
        recorded prompt length.
        """
        # Convert to Python list for the state machine. mlx arrays don't
        # support the slicing/indexing idioms the state machine uses.
        # mlx_lm.BatchGenerator contract: tokens is the FULL history
        # (prompt + generated output). We slice by the recorded prompt
        # length to get the output-only tail.
        token_list = tokens.tolist()
        output = token_list[self._prompt_len :]

        self._advance_state(output)

        if self._in_end and self._end_count < len(self._force_sequence):
            forced_id = self._force_sequence[self._end_count]
            # +1e9 on the forced id; sampler's argmax / softmax picks it.
            # Use mx.scatter-style update: add a one-hot bias vector.
            vocab = logits.shape[-1]
            bias = mx.zeros(vocab, dtype=logits.dtype)
            bias[forced_id] = _FORCE_BIAS
            return logits + bias

        return logits
