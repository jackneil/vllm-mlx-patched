# SPDX-License-Identifier: Apache-2.0
"""Rebase / env-state sentinel for the mlx_lm.models.deepseek_v4 module.

Pinned by UPSTREAM_PIN.md invariant #19.

Upstream PyPI mlx-lm 0.31.3 ships only ``deepseek``, ``deepseek_v2``,
``deepseek_v3``, and ``deepseek_v32`` model classes. The ``deepseek_v4``
class lives in jackneil/mlx-lm-1@fix/deepseek-v4-decode-cache (cloned
locally at /Users/jackneil/Github/mlx-lm-v4) and reaches the vllm-mlx
conda env via an editable install (.pth). Without this editable install
in place, ``vllm-mlx serve mlx-community/DeepSeek-V4-Flash-4bit`` (or any
other config.json with ``model_type: deepseek_v4``) hard-fails at startup:

    ModuleNotFoundError: No module named 'mlx_lm.models.deepseek_v4'
    ValueError: Model type deepseek_v4 not supported.
    ERROR:    Application startup failed. Exiting.

Forge's ``hank_llm_arena/forge/validators/deepseek_v4.py`` validator (in
the arena repo) does ``from mlx_lm.models import deepseek_v4`` at module
load time and ``from mlx_lm import load`` for artifact-parity validation.
Forge does not ship the model class — it depends on this editable install.

How this sentinel fails:

1. ``pip install -U mlx-lm`` (or any transitive install resolving mlx-lm
   from PyPI) silently overwrites the editable install. ``mlx_lm.__file__``
   moves out of /Users/jackneil/Github/mlx-lm-v4/, the deepseek_v4 module
   disappears, and DeepSeek serving regresses without any other test
   noticing.
2. The mlx-lm-v4 working tree is checked out to a branch that doesn't
   carry ``mlx_lm/models/deepseek_v4.py`` (e.g. accidental
   ``git checkout main`` in the local clone).

Fix path on either failure: re-run the editable install:

    pip install -e /Users/jackneil/Github/mlx-lm-v4

If this file fails in CI, do NOT merge — DeepSeek serving will be silently
broken until the first user request crashes.
"""
from __future__ import annotations

import importlib

import pytest


def test_mlx_lm_deepseek_v4_module_imports():
    """Module must be importable for vllm-mlx to load DeepSeek V4 weights."""
    try:
        importlib.import_module("mlx_lm.models.deepseek_v4")
    except ModuleNotFoundError as e:  # pragma: no cover
        pytest.fail(
            "mlx_lm.models.deepseek_v4 is missing — invariant #19 broken. "
            "The vllm-mlx conda env's mlx-lm install does NOT include this "
            "architecture. Re-install the editable fork: "
            "`pip install -e /Users/jackneil/Github/mlx-lm-v4`. "
            f"Underlying error: {e}"
        )


def test_mlx_lm_get_classes_resolves_deepseek_v4():
    """The exact code path that vllm-mlx hits at server startup must succeed."""
    from mlx_lm.utils import _get_classes

    try:
        cls, args_cls = _get_classes(config={"model_type": "deepseek_v4"})
    except ValueError as e:  # pragma: no cover
        pytest.fail(
            "mlx_lm.utils._get_classes failed to resolve model_type='deepseek_v4'. "
            "This is the same crash users hit at vllm-mlx serve time. "
            f"See invariant #19 in UPSTREAM_PIN.md. Underlying error: {e}"
        )

    assert cls.__module__ == "mlx_lm.models.deepseek_v4", (
        f"Expected resolved class module to be 'mlx_lm.models.deepseek_v4', "
        f"got {cls.__module__!r}. This usually means a different mlx-lm "
        f"install is being picked up — check mlx_lm.__file__ resolves to "
        f"/Users/jackneil/Github/mlx-lm-v4/mlx_lm/."
    )
    assert args_cls.__module__ == "mlx_lm.models.deepseek_v4"


def test_editable_install_points_at_mlx_lm_v4():
    """mlx_lm package must resolve from the local fork, not PyPI install.

    A passing import (test above) is necessary but not sufficient — a
    future upstream PyPI release that ships its OWN deepseek_v4 module
    would also pass the import test, but might lack the cache-parity
    fixes the fork carries (HEAD 2673e367 at the time of pinning). When
    upstream catches up, drop this test and the editable install (per
    UPSTREAM_PIN.md invariant 19's drop-back path).
    """
    import mlx_lm

    expected_marker = "/Users/jackneil/Github/mlx-lm-v4/"
    if expected_marker not in (mlx_lm.__file__ or ""):  # pragma: no cover
        pytest.fail(
            f"mlx_lm.__file__ = {mlx_lm.__file__!r} does NOT resolve under "
            f"{expected_marker}. The editable install of the mlx-lm-v4 fork "
            f"has been overwritten (likely by `pip install -U mlx-lm` or a "
            f"transitive install). Re-run: "
            f"`pip install -e /Users/jackneil/Github/mlx-lm-v4`. "
            f"See invariant #19 in UPSTREAM_PIN.md."
        )
