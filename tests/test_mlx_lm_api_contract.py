# SPDX-License-Identifier: Apache-2.0
"""Regression sentinels for the mlx_lm BatchGenerator API contract.

vllm_mlx requires mlx_lm >= 0.31.2 and adapts to its split
_prompt_batch + _generation_batch model. These tests catch accidental
reintroduction of the removed 0.30/0.31.1 API (single `active_batch`
slot on the BatchGenerator itself), direct attribute access to the
split batches outside the canonical helper, and bypass patterns like
closures that re-bind self to the mlx_lm BatchGenerator.

Pinned by UPSTREAM_PIN.md invariant #10.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_SCHEDULER = Path(__file__).parent.parent / "vllm_mlx" / "scheduler.py"

# Functions whose body is allowed to reference the forbidden attributes.
# - _active_batches: the canonical helper itself (defines `_prompt_batch`
#   and `_generation_batch` access).
# - _install_mtp, _install_chunked_prefill: legacy closures containing
#   `self=batch_gen` / `self=bg` default-kwarg rebinds. Both are gated
#   at their call sites for mlx_lm 0.31.2+ (see version-gate in
#   Scheduler._create_batch_generator). Keeping them dead-but-visible
#   is cheaper than deleting 500+ lines in a focused drift fix; they
#   will be rewritten or removed in a follow-up plan.
_ALLOWLIST_FUNCS = {"_active_batches", "_install_mtp", "_install_chunked_prefill"}

_FORBIDDEN_ATTRS = {"active_batch", "_prompt_batch", "_generation_batch"}


def _collect_allowlist_line_ranges(tree: ast.Module) -> list[tuple[int, int]]:
    """Return (start, end) line numbers of every allowlisted function body."""
    ranges = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _ALLOWLIST_FUNCS:
                end = node.end_lineno or node.lineno
                ranges.append((node.lineno, end))
    return ranges


def _is_in_allowlist(
    lineno: int, ranges: list[tuple[int, int]]
) -> bool:
    return any(start <= lineno <= end for start, end in ranges)


def test_no_active_batch_or_split_batch_refs_outside_allowlist():
    """scheduler.py must not reference the removed `active_batch`
    attribute on an mlx_lm BatchGenerator, and must not directly access
    `_prompt_batch` or `_generation_batch` outside the canonical
    `_active_batches` helper.

    Exceptions (allowlisted by function name — see _ALLOWLIST_FUNCS):
      - `_active_batches` itself defines the direct access.
      - `_install_mtp` / `_install_chunked_prefill` are legacy closures
        gated at their call sites; their bodies are unreachable on
        mlx_lm 0.31.2+ but remain source-visible.

    The scan walks the AST so it is robust against comments, strings,
    renames, closures (including `self=batch_gen` default-kwarg rebinds),
    and whitespace.
    """
    src = _SCHEDULER.read_text()
    tree = ast.parse(src)
    allow = _collect_allowlist_line_ranges(tree)

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr not in _FORBIDDEN_ATTRS:
            continue
        if _is_in_allowlist(node.lineno, allow):
            continue
        try:
            rendered = ast.unparse(node)
        except Exception:
            rendered = f"<Attribute attr={node.attr}>"
        offenders.append(f"  line {node.lineno}: {rendered}")

    assert not offenders, (
        "scheduler.py references a forbidden BatchGenerator attribute "
        "outside the allowlisted functions. mlx_lm >= 0.31.2 replaced "
        "`active_batch` with split `_prompt_batch` + `_generation_batch`; "
        "use the `_active_batches(bg)` helper for all access. Offenders:\n"
        + "\n".join(offenders)
    )


def test_active_batches_helper_returns_empty_list_when_no_batches():
    """Helper returns [] when both split slots are None."""
    from vllm_mlx.scheduler import _active_batches

    class _FakeBG:
        _prompt_batch = None
        _generation_batch = None

    assert _active_batches(_FakeBG()) == []


def test_active_batches_helper_skips_empty_batches():
    """Batches with len == 0 are excluded (not 'active' for the
    cache-eviction / liveness-scan purposes)."""
    from vllm_mlx.scheduler import _active_batches

    class _FakeBatch:
        def __init__(self, n):
            self._n = n

        def __len__(self):
            return self._n

    class _FakeBG:
        _prompt_batch = _FakeBatch(0)
        _generation_batch = _FakeBatch(3)

    result = _active_batches(_FakeBG())
    assert len(result) == 1
    assert result[0] is _FakeBG._generation_batch


def test_active_batches_helper_returns_both_when_pipelined():
    """During pipelined prefill + decode both split batches can be
    non-empty; the helper surfaces both, in prompt-first order."""
    from vllm_mlx.scheduler import _active_batches

    class _FakeBatch:
        def __init__(self, n):
            self._n = n

        def __len__(self):
            return self._n

    class _FakeBG:
        _prompt_batch = _FakeBatch(2)
        _generation_batch = _FakeBatch(5)

    result = _active_batches(_FakeBG())
    assert len(result) == 2
    assert result[0] is _FakeBG._prompt_batch
    assert result[1] is _FakeBG._generation_batch


def test_startup_assertion_rejects_pre_0_31_2_batch_generator():
    """A stub BatchGenerator lacking `_generation_batch` must trigger a
    RuntimeError with mlx_lm >= 0.31.2 requirement in its message. This
    test mirrors the assertion shape used in
    `Scheduler._create_batch_generator` so the contract stays locked."""
    class _LegacyBG:
        """Pre-0.31.2 mlx_lm had a single `active_batch` slot."""

        def __init__(self):
            self.active_batch = None

    bg = _LegacyBG()
    assert not hasattr(bg, "_generation_batch")

    import mlx_lm as _mlx_lm_mod
    _v = getattr(_mlx_lm_mod, "__version__", "unknown")

    with pytest.raises(RuntimeError, match="mlx_lm >= 0.31.2"):
        if not hasattr(bg, "_generation_batch"):
            raise RuntimeError(
                "vllm_mlx requires mlx_lm >= 0.31.2 (BatchGenerator "
                f"must expose _prompt_batch + _generation_batch). "
                f"Installed mlx_lm version: {_v}. Upgrade with: "
                "pip install -U 'mlx-lm>=0.31.2'"
            )
