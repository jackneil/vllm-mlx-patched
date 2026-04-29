# SPDX-License-Identifier: Apache-2.0
"""Unit tests for vllm_mlx.mlx_streams.bind_generation_streams.

Validates that the helper:
  - Returns a stream object.
  - Patches mlx_lm.generate.generation_stream on the module.
  - Tolerates missing modules (mlx_vlm not installed in some envs).
  - Works correctly when invoked from a non-main thread (the actual
    production scenario via mlx-step ThreadPoolExecutor).
"""
from __future__ import annotations

import threading


def test_bind_returns_a_stream():
    from vllm_mlx.mlx_streams import bind_generation_streams
    s = bind_generation_streams()
    assert s is not None


def test_bind_patches_mlx_lm_generation_stream():
    """mlx_lm package re-exports `generate` as a function, shadowing the
    submodule on the package surface. Access via importlib to reach the
    actual submodule object that the helper patches."""
    import importlib
    from vllm_mlx.mlx_streams import bind_generation_streams
    gen = importlib.import_module("mlx_lm.generate")
    new = bind_generation_streams()
    assert gen.generation_stream is new
    # second call writes a fresh stream into the same slot
    again = bind_generation_streams()
    assert gen.generation_stream is again


def test_bind_tolerates_missing_module():
    from vllm_mlx.mlx_streams import bind_generation_streams
    # Use a definitely-not-importable name — must not raise.
    result = bind_generation_streams(module_names=("not_a_real_mlx_module_xyz",))
    assert result is not None


def test_bind_runs_in_worker_thread():
    """The helper must not crash when invoked from a non-main thread.
    This is the actual production scenario (mlx-step ThreadPoolExecutor)."""
    from vllm_mlx.mlx_streams import bind_generation_streams
    errors = []
    def worker():
        try:
            bind_generation_streams()
        except Exception as e:
            errors.append(e)
    t = threading.Thread(target=worker, name="test-worker")
    t.start()
    t.join(timeout=5)
    assert not errors, f"bind_generation_streams raised in worker thread: {errors}"
