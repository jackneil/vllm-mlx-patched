"""Unit tests for vllm_mlx.api.effort — dialect-agnostic budget resolver."""

import pytest

from vllm_mlx.api.effort import (
    EffortSource,
    ResolvedBudget,
    resolve_effort,
    _EFFORT_TABLE,
    _MAX_BUDGET_CAP,
)


def test_module_imports_and_exports_public_api():
    """Scaffold test: the module loads and exposes the documented surface."""
    assert EffortSource.DEFAULT == "default"
    assert ResolvedBudget(budget=None, source=EffortSource.DEFAULT,
                          max_tokens_floor=None, effort_label=None)
    assert callable(resolve_effort)
    assert isinstance(_EFFORT_TABLE, dict)
    assert isinstance(_MAX_BUDGET_CAP, int)
