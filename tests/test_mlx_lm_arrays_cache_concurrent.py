# SPDX-License-Identifier: Apache-2.0
"""Rebase sentinel for the mlx_lm ArraysCache.extend() batch-dim contract.

Pinned by UPSTREAM_PIN.md invariant #17.

Qwen3.x 35B-A3B / Qwen3-Next and other hybrid-cache models use
ArraysCache for their linear-attn / Gated-DeltaNet layers. When two
concurrent requests are mid-prefill inside a BatchGenerator, the
scheduler calls ArraysCache.extend() to fold the joining sequence's
per-slot cache into the already-batched cache.

The pre-fix bug has TWO distinct regression vectors. Each test below
isolates one. A pin that covers only one vector silently re-admits the
bug — see the plan doc at
docs/superpowers/plans/2026-04-22-qwen3-concurrent-deadlock.md for the
empirical verification that both must fail on a pre-fix install.

    Vector #1169 (ml-explore/mlx-lm#1169, merged 2026-04-21):
        ArraysCache.extend's inner ``cat`` helper returned ``a`` or
        ``b`` unchanged when either was None, skipping the batch-dim
        padding step. Per-slot batch dimensions became mismatched.
        Exposed by: extend-with-None-slot tests that check final
        ``cache[i].shape[0]``.

    Vector #1177 (ml-explore/mlx-lm#1177, merged same day):
        ``cat`` reads ``self.batch_size`` from the enclosing closure.
        ``extend`` mutates ``self.cache`` via a list-comprehension
        BEFORE the post-loop ``cat(self.left_padding, other.left_padding)``
        and ``cat(self.lengths, other.lengths)`` calls. After the
        listcomp, ``self.batch_size`` reads from the mutated cache and
        returns the already-extended size — so the None-side zero-pad
        in those post-loop cat calls over-pads. Fix captures ``a_batch``
        and ``b_batch`` BEFORE the listcomp. Exposed by: extend where
        one side has ``left_padding=None`` and the other doesn't, plus
        a post-extend assertion on ``left_padding.shape``.

    The mlx_lm version string is NOT a sufficient floor check on its own:
    the version bump commit ``d9c63ff`` in mlx-lm landed the
    ``__version__ = "0.31.3"`` BEFORE #1169 and BEFORE #1177 merged.
    Any SHA in the window ``d9c63ff..3cd9a52d^`` reports 0.31.3 but
    lacks one or both fixes. A string-only version check passes.
    Behavioral probes are required. See
    ``test_arrays_cache_extend_fixes_vector_1169_with_none_slot``
    and ``test_arrays_cache_extend_fixes_vector_1177_via_left_padding``.

If this file fails after an mlx_lm rebase or downgrade, stop — do not
merge. The concurrent-prefill deadlock regresses silently (no test
failure in the basic prompt-cache suite; symptom only surfaces under
concurrent load on hybrid models in production).

**Scope limit:** these tests bind the #1177 contract on exactly two
post-listcomp ``cat`` targets — ``self.left_padding`` and
``self.lengths``. If a future mlx_lm refactor adds another post-loop
line of the shape ``self.X = cat(self.X, other.X)`` where the closure
reads ``self.batch_size``, a #1177-class regression on ``X`` would
slip past the sentinel. When rebasing, grep
``mlx_lm/models/cache.py`` for new post-``self.cache`` assignments
inside ``ArraysCache.extend`` and extend this file accordingly.
"""

from __future__ import annotations

import pytest

mlx = pytest.importorskip("mlx.core")
_cache_mod = pytest.importorskip("mlx_lm.models.cache")

try:
    ArraysCache = _cache_mod.ArraysCache
except AttributeError as exc:
    raise RuntimeError(
        "mlx_lm.models.cache.ArraysCache is missing from the installed mlx_lm "
        f"(looked in {_cache_mod.__file__}). Either mlx_lm renamed the class "
        "(update this sentinel and UPSTREAM_PIN.md invariant #17) or the pin "
        "was bumped to a version that dropped hybrid-cache support entirely. "
        "Do NOT replace this import with a broader `getattr(..., default)` — "
        "the sentinel's whole purpose is to fail loudly on API drift that "
        "would silently mask the concurrent-prefill bug."
    ) from exc


def _make_populated_cache(size: int, batch: int, dim: int = 4) -> ArraysCache:
    """Return an ArraysCache of ``size`` slots, all populated at batch=batch."""
    c = ArraysCache(size)
    for i in range(size):
        c[i] = mlx.zeros((batch, dim), dtype=mlx.float32)
    return c


class TestArraysCacheExtendBatchDim:
    """Invariant #17 — ArraysCache.extend preserves batch-size contract
    under both vector #1169 and vector #1177."""

    # ---- Vector #1169 (None-slot padding) ----

    def test_populated_by_populated_preserves_batch_sum(self):
        """Baseline: two populated caches concat to a_batch + b_batch per slot.

        This is the non-regression case — always passed, even pre-#1169.
        Kept as an anti-regression sentinel so a broken rewrite of the
        happy path is also caught.
        """
        a = _make_populated_cache(size=4, batch=3, dim=8)
        b = _make_populated_cache(size=4, batch=2, dim=8)
        a.extend(b)
        for slot, arr in enumerate(a.cache):
            assert arr is not None, f"slot {slot}: expected non-None"
            assert (
                arr.shape[0] == 3 + 2
            ), f"slot {slot}: batch dim should be 5, got {arr.shape[0]}"
            assert arr.shape[1] == 8

    def test_arrays_cache_extend_fixes_vector_1169_with_none_slot(self):
        """Vector #1169: populated `self` + all-None `other` must pad None
        slots to other.batch_size.

        Pre-#1169, the ``cat`` helper returned `a` unchanged when
        `b is None`, leaving the batch dim at `a_batch` instead of
        `a_batch + b_batch`. Concurrent middle-wave requests then saw
        their neighbour's state in the attention layers.

        Also asserts the post-loop ``left_padding`` shape to bind the
        test against vector #1177 — on a buggy install that fixed
        #1169 but not #1177, cache slots would look right but
        ``a.left_padding.shape`` would be wrong.
        """
        a = _make_populated_cache(size=4, batch=3, dim=8)
        b = ArraysCache(4)
        # Give `b` a non-trivial batch_size via left_padding so b.batch_size != 1.
        b.left_padding = mlx.zeros((2,), dtype=mlx.int32)
        a.extend(b)

        for slot, arr in enumerate(a.cache):
            assert arr is not None, f"slot {slot}: expected non-None after extend"
            assert arr.shape[0] == 3 + 2, (
                f"slot {slot}: batch dim should be 5 (a_batch=3 + b_batch=2); "
                f"got {arr.shape[0]}. Regression of mlx_lm#1169 — ArraysCache.extend "
                "is NOT padding None slots to other.batch_size. Concurrent prefill "
                "on hybrid models will corrupt neighbour state."
            )

        # Post-loop left_padding check: this is where #1177 surfaces.
        # Pre-#1177, self.batch_size inside `cat` reads the already-extended
        # cache and returns 5 instead of the captured a_batch=3 — producing
        # left_padding of shape (5 + 2,) = (7,) instead of (3 + 2,) = (5,).
        assert a.left_padding is not None, (
            "left_padding was dropped — cat should have concatenated "
            "a.left_padding (None) with b.left_padding (zeros((2,)))"
        )
        assert a.left_padding.shape == (3 + 2,), (
            f"left_padding.shape should be (5,) — a_batch=3 (captured BEFORE "
            f"cache mutation) + b_batch=2 — but got {a.left_padding.shape}. "
            "Regression of mlx_lm#1177 — `cat` closure is reading self.batch_size "
            "from the post-mutation cache. See ArraysCache.extend in "
            f"{_cache_mod.__file__}."
        )

    def test_arrays_cache_extend_fixes_vector_1169_empty_by_populated(self):
        """Vector #1169 mirror: all-None `self` + populated `other`.

        Same contract as the previous test but with `a` as the None side.
        Mirrors the symmetric case in the bug and catches half-fixes that
        only handle one direction.

        Note on #1177 coverage: the ``left_padding`` assertion below does
        NOT discriminate #1177 in this construction — the post-loop
        ``cat(a.left_padding=zeros((3,)), b.left_padding=None)`` fires
        the ``b is None`` branch, which pads using ``other.batch_size``.
        Since ``other.cache`` is not mutated by the listcomp,
        ``other.batch_size`` stays stable pre- and post-#1177, so shape
        equals ``a_batch + b_batch = 5`` in both. The #1177 coverage
        lives in ``test_arrays_cache_extend_fixes_vector_1177_via_left_padding``
        (which forces the symmetric ``a is None`` branch). Left here as a
        basic-contract sanity check.
        """
        a = ArraysCache(4)
        # Give `a` a non-trivial batch_size via left_padding.
        a.left_padding = mlx.zeros((3,), dtype=mlx.int32)
        b = _make_populated_cache(size=4, batch=2, dim=8)
        a.extend(b)
        for slot, arr in enumerate(a.cache):
            assert arr is not None, f"slot {slot}: expected non-None after extend"
            assert arr.shape[0] == 3 + 2, (
                f"slot {slot}: batch dim should be 5 (a_batch=3 + b_batch=2); "
                f"got {arr.shape[0]}. Regression of mlx_lm#1169."
            )
        # left_padding basic-shape sanity — see docstring note on coverage scope.
        assert a.left_padding.shape == (3 + 2,), (
            f"left_padding.shape should be (5,) but got {a.left_padding.shape}. "
            "Basic shape-contract failure (not a #1177 discriminator — see docstring)."
        )

    def test_empty_by_empty_stays_empty(self):
        """All-None both sides (no batch metadata): cache stays all-None.

        The ``shape is None`` branch of ``cat``. Kept to catch broken
        rewrites that spuriously materialize zero tensors.
        """
        a = ArraysCache(4)
        b = ArraysCache(4)
        a.extend(b)
        assert all(
            c is None for c in a.cache
        ), "All-None extend should leave cache all-None (shape is None branch)."

    # ---- Vector #1177 (capture-before-mutation) ----

    def test_arrays_cache_extend_fixes_vector_1177_via_left_padding(self):
        """Vector #1177: `cat` must use a_batch/b_batch captured BEFORE
        the cache-listcomp mutation, not re-read ``self.batch_size``
        after it.

        Reproduces the exact regression pattern the upstream fix PR
        #1177 was titled around: construct an extend where ``self.cache``
        is populated AND the post-loop ``cat(self.left_padding,
        other.left_padding)`` has ONE None side. Pre-#1177, the closure
        reads ``self.batch_size`` which now reflects the already-extended
        cache, over-padding the None side.

        Concretely (pre-#1177):

            a.cache     = [zeros((2, 4))]   # a_batch=2 INITIALLY
            a.left_padding = None
            b.cache     = [zeros((3, 4))]   # b_batch=3
            b.left_padding = zeros((3,))

            a.extend(b)
            # step 1: a.cache = [zeros((5, 4))]  # mutates; a.batch_size now 5
            # step 2: cat(None, zeros((3,)))
            #    pre-#1177: pads None with zeros((self.batch_size,)) = zeros((5,))
            #      returns concatenate([zeros((5,)), zeros((3,))]) = shape (8,)  # WRONG
            #    post-#1177: pads None with zeros((a_batch,)) = zeros((2,))
            #      returns concatenate([zeros((2,)), zeros((3,))]) = shape (5,)  # RIGHT
        """
        a = ArraysCache(1)
        a.cache = [mlx.zeros((2, 4), dtype=mlx.float32)]
        a.left_padding = None

        b = ArraysCache(1)
        b.cache = [mlx.zeros((3, 4), dtype=mlx.float32)]
        b.left_padding = mlx.zeros((3,), dtype=mlx.int32)

        a.extend(b)

        assert a.cache[0].shape[0] == 2 + 3, (
            f"Slot 0 cache batch dim should be a_batch+b_batch=5, got "
            f"{a.cache[0].shape[0]}."
        )
        assert a.left_padding is not None, "left_padding was dropped by extend."
        assert a.left_padding.shape == (2 + 3,), (
            f"left_padding.shape should be (5,) — a_batch=2 (captured BEFORE "
            "cache mutation) + b_batch=3 — but got "
            f"{a.left_padding.shape}. Regression of mlx_lm#1177: `cat` "
            "closure is reading self.batch_size from the post-mutation cache. "
            "This is exactly the bug that deadlocks concurrent Qwen3.5/3.6-35B-A3B "
            "serving. See ArraysCache.extend in "
            f"{_cache_mod.__file__}."
        )

    def test_arrays_cache_extend_fixes_vector_1177_via_lengths(self):
        """Vector #1177 mirror: the same regression surfaces on
        ``self.lengths`` when populated on one side only.

        Belt-and-suspenders — mlx-lm's own test exercises both
        ``left_padding`` and ``lengths`` paths since both are post-loop
        ``cat`` calls that read ``self.batch_size``.
        """
        a = ArraysCache(1)
        a.cache = [mlx.zeros((2, 4), dtype=mlx.float32)]
        a.lengths = None

        b = ArraysCache(1)
        b.cache = [mlx.zeros((3, 4), dtype=mlx.float32)]
        b.lengths = mlx.array([0, 0, 0], dtype=mlx.int32)

        a.extend(b)

        assert a.lengths is not None, "lengths was dropped by extend."
        assert a.lengths.shape == (2 + 3,), (
            f"lengths.shape should be (5,) but got {a.lengths.shape}. "
            "Regression of mlx_lm#1177 on the lengths path."
        )

    # ---- Pin floor (behavioral probe + version info on failure) ----

    def test_installed_mlx_lm_has_both_vector_fixes(self):
        """Belt-and-suspenders: probe behavior under BOTH regression
        vectors at once, and on failure, report the installed
        ``mlx_lm.__version__`` and path.

        A version-string comparison against ``>= 0.31.3`` is not
        sufficient because mlx-lm's version-bump commit (``d9c63ff``)
        landed BEFORE #1169 and BEFORE #1177 merged — any intermediate
        SHA reports 0.31.3 but lacks one or both fixes. This test
        probes the ArraysCache behavior directly: if either fix is
        missing the output shape is wrong and we fail with a
        diagnostic pointing at the installed path and version.
        """
        import mlx_lm

        # Vector #1169 probe: None-slot pad.
        a1 = _make_populated_cache(size=1, batch=3, dim=4)
        b1 = ArraysCache(1)
        b1.left_padding = mlx.zeros((2,), dtype=mlx.int32)
        a1.extend(b1)
        vector_1169_ok = a1.cache[0] is not None and a1.cache[0].shape[0] == 5

        # Vector #1177 probe: capture-before-mutation via left_padding.
        a2 = ArraysCache(1)
        a2.cache = [mlx.zeros((2, 4), dtype=mlx.float32)]
        a2.left_padding = None
        b2 = ArraysCache(1)
        b2.cache = [mlx.zeros((3, 4), dtype=mlx.float32)]
        b2.left_padding = mlx.zeros((3,), dtype=mlx.int32)
        a2.extend(b2)
        vector_1177_ok = a2.left_padding is not None and a2.left_padding.shape == (5,)

        if vector_1169_ok and vector_1177_ok:
            return  # Both fixes present.

        missing = []
        if not vector_1169_ok:
            missing.append(
                "#1169 (None-slot padding): "
                f"a1.cache[0].shape[0]={a1.cache[0].shape[0] if a1.cache[0] is not None else None}"
            )
        if not vector_1177_ok:
            missing.append(
                "#1177 (capture batch sizes before mutation): "
                f"a2.left_padding.shape={tuple(a2.left_padding.shape) if a2.left_padding is not None else None}"
            )

        pytest.fail(
            "Installed mlx_lm is missing one or both ArraysCache.extend fixes "
            "required for concurrent heavy-payload serving of Qwen3.x hybrid-cache "
            "models.\n"
            f"  Installed version: {mlx_lm.__version__}\n"
            f"  Installed path:    {_cache_mod.__file__}\n"
            f"  Missing vectors:   {', '.join(missing)}\n"
            "Required: mlx-lm at ml-explore/mlx-lm@3cd9a52d or later (first SHA "
            "with BOTH #1169 and #1177 merged). Note that mlx_lm.__version__ "
            "alone is insufficient — any SHA in d9c63ff..3cd9a52d^ reports "
            "'0.31.3' but lacks the fix. See UPSTREAM_PIN.md invariant #17 and "
            "docs/superpowers/plans/2026-04-22-qwen3-concurrent-deadlock.md."
        )
