# SPDX-License-Identifier: Apache-2.0
"""Rebase sentinel for the mlx_lm ArraysCache.extend() batch-dim contract.

Pinned by UPSTREAM_PIN.md invariant #17.

Qwen3.x 35B-A3B / Qwen3-Next and other hybrid-cache models use
ArraysCache for their linear-attn / Gated-DeltaNet layers. When two
concurrent requests are mid-prefill inside a BatchGenerator, the
scheduler calls ArraysCache.extend() to fold the joining sequence's
per-slot cache into the already-batched cache.

Pre-mlx_lm 0.31.3 (specifically pre-ml-explore/mlx-lm#1169 + #1177,
merged 2026-04-21), ArraysCache.extend() had a batch-dim bug: when
one side had ``keys=None`` (a fresh, never-used slot) and the other
had content, the ``cat`` helper returned the non-None side unchanged
instead of padding the None side to the correct batch size. This left
the per-slot batch dimension mismatched across slots, which under
concurrent heavy-payload serving of Qwen3.5/3.6-35B-A3B surfaced as
degenerate zero-token responses or deterministic deadlocks (both
requests emit ``message_start`` then never produce a token — see
docs/testing/2026-04-21-qwen3-35b-a3b-concurrent-heavy-payload-deadlock.md).

This sentinel asserts the post-#1177 contract holds: after extending a
cache-with-content by an all-None cache with the same size, every
populated slot's batch dimension equals ``a_batch + b_batch``, and
None slots on both sides stay None.

If this file fails after an mlx_lm rebase or downgrade, stop — do not
merge. The concurrent-prefill deadlock regresses silently (no test
failure in the basic prompt-cache suite; symptom only surfaces under
concurrent load on hybrid models).
"""

from __future__ import annotations

import pytest

mlx = pytest.importorskip("mlx.core")
_cache_mod = pytest.importorskip("mlx_lm.models.cache")

ArraysCache = _cache_mod.ArraysCache


def _make_populated_cache(size: int, batch: int, dim: int = 4) -> ArraysCache:
    """Return an ArraysCache of ``size`` slots, all populated at batch=batch."""
    c = ArraysCache(size)
    for i in range(size):
        c[i] = mlx.zeros((batch, dim), dtype=mlx.float32)
    return c


def _make_empty_cache(size: int) -> ArraysCache:
    """Return an ArraysCache of ``size`` slots, all None."""
    return ArraysCache(size)


class TestArraysCacheExtendBatchDim:
    """Invariant #17 — ArraysCache.extend preserves batch-size contract."""

    def test_extend_populated_by_populated(self):
        """Two populated caches concat to a_batch + b_batch per slot."""
        a = _make_populated_cache(size=4, batch=3, dim=8)
        b = _make_populated_cache(size=4, batch=2, dim=8)
        a.extend(b)
        for slot, arr in enumerate(a.cache):
            assert arr is not None, f"slot {slot}: expected non-None"
            assert (
                arr.shape[0] == 3 + 2
            ), f"slot {slot}: batch dim should be 5, got {arr.shape[0]}"
            assert arr.shape[1] == 8

    def test_extend_populated_by_empty_left(self):
        """Populated `self` + all-None `other` must pad None side to other.batch_size.

        This is THE regression case for mlx_lm#1169. Pre-fix, the `cat`
        helper returned `a` unchanged when `b is None`, leaving the batch
        dim at `a_batch` instead of `a_batch + b_batch`. Concurrent
        middle-wave requests then saw their neighbour's state.
        """
        a = _make_populated_cache(size=4, batch=3, dim=8)
        b = _make_empty_cache(size=4)
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

    def test_extend_empty_by_populated(self):
        """All-None `self` + populated `other` must pad None side to self.batch_size."""
        a = _make_empty_cache(size=4)
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

    def test_extend_empty_by_empty_stays_empty(self):
        """All-None both sides: cache stays all-None (nothing to concat)."""
        a = _make_empty_cache(size=4)
        b = _make_empty_cache(size=4)
        a.extend(b)
        assert all(
            c is None for c in a.cache
        ), "All-None extend should leave cache all-None (shape is None branch)."

    def test_extend_captures_batch_sizes_before_mutation(self):
        """Invariant #17 sub-clause: the fix from mlx_lm#1177 requires
        capturing a_batch/b_batch BEFORE the cat-loop mutates self.cache.

        Regression: if `cat` is rewritten to re-read `self.batch_size`
        inside the loop, the second and later slot assignments see the
        already-extended batch size and mis-pad. This test constructs a
        cache where the first slot is None on both sides and the second
        slot is populated, forcing the loop to rely on the captured
        batch sizes for the first slot's behavior (None+None → None, no
        padding triggered, so the mismatched `self.batch_size` read on
        the NEXT iteration would cause mis-pad of the now-None-vs-populated
        case).
        """
        a = ArraysCache(2)
        a.cache = [None, mlx.zeros((3, 4), dtype=mlx.float32)]
        b = ArraysCache(2)
        b.cache = [mlx.zeros((2, 4), dtype=mlx.float32), None]
        a.extend(b)
        # Slot 0: was None on a, populated on b with batch=2.
        # a_batch captured BEFORE loop should be 3 (from a.cache[1]).
        assert a.cache[0] is not None
        assert a.cache[0].shape[0] == 3 + 2, (
            f"slot 0: expected batch 5, got {a.cache[0].shape[0]}. "
            "Regression of mlx_lm#1177 — batch sizes not captured before cat loop."
        )
        # Slot 1: populated on a with batch=3, None on b. b_batch captured
        # before loop should be 2 (from b.cache[0]).
        assert a.cache[1] is not None
        assert a.cache[1].shape[0] == 3 + 2, (
            f"slot 1: expected batch 5, got {a.cache[1].shape[0]}. "
            "Regression of mlx_lm#1177 — batch sizes not captured before cat loop."
        )

    def test_minimum_mlx_lm_version(self):
        """The vllm_mlx pin must include the ArraysCache.extend fix.

        mlx_lm 0.31.3 (ml-explore/mlx-lm@3cd9a52d on main at 2026-04-22)
        is the first version with both #1169 and #1177 merged. Any pin
        that resolves below 0.31.3 silently reintroduces the concurrent
        prefill bug.
        """
        import mlx_lm

        version = mlx_lm.__version__
        # Parse "0.31.3" or "0.31.3.devN+gabcdef" etc.
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1])
        patch_str = parts[2].split("+")[0].split("-")[0].split("dev")[0]
        patch = int("".join(ch for ch in patch_str if ch.isdigit()) or "0")

        current = (major, minor, patch)
        minimum = (0, 31, 3)
        assert current >= minimum, (
            f"mlx_lm {version} predates the ArraysCache.extend fix "
            f"(mlx-lm#1169 + #1177, shipped in 0.31.3). Concurrent heavy-payload "
            "serving of Qwen3.5/3.6-35B-A3B and other hybrid-cache models will "
            "deadlock or produce degenerate zero-token responses. See "
            "docs/superpowers/plans/2026-04-22-qwen3-concurrent-deadlock.md."
        )
