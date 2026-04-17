# SPDX-License-Identifier: Apache-2.0
"""Per-request logits processors for vllm-mlx.

Callable[[mx.array, mx.array], mx.array] — compatible with
mlx_lm.generate.BatchGenerator's logits_processors interface.
"""

from .thinking_budget import ThinkingTokenBudgetLogitsProcessor

__all__ = ["ThinkingTokenBudgetLogitsProcessor"]
