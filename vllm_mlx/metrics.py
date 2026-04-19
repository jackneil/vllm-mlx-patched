# SPDX-License-Identifier: Apache-2.0
"""Lightweight in-process counters.

Standard-lib only. A follow-up PR wires these to Prometheus via a /metrics
endpoint.
"""

import threading


class _Counter:
    def __init__(self, name: str) -> None:
        self._name = name
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, n: int = 1) -> None:
        with self._lock:
            self._value += n

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


# Increments every time a request's thinking_token_budget cannot be
# enforced (MLLM path, unsupported parser, multi-token delimiter failure,
# broken tokenizer, etc.). Alert on rate > 0 if budget is expected to work.
thinking_budget_noop_total = _Counter("thinking_budget_noop_total")

# Increments every time a streaming response terminated because the
# --streaming-max-seconds wall-clock cap fired. Distinct from organic
# finish_reason="length" (model hit max_tokens naturally) — cap-firings
# indicate the model produced no terminating token within the time
# budget and we forced closure. Alert when the rate climbs: either the
# cap is too low for the workload OR a model is stuck (Qwen3.x
# interleaved-thinking trap or similar).
streaming_cap_fired_total = _Counter("streaming_cap_fired_total")
