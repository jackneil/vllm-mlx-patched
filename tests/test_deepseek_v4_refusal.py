# SPDX-License-Identifier: Apache-2.0
"""Tests for the DeepSeek-V4 runtime refusal dial (rank-1 projection).

Covers vllm_mlx/patches/deepseek_v4_refusal.py plus its two consumers:
the ``GET``/``POST /admin/refusal_lambda`` routes in server.py and the
``--refusal-dirs`` serve flag in cli.py.

No model is loaded and nothing touches the network: the module tree is a
handful of plain objects and every sidecar is written to tmp_path with
``mx.save_safetensors``. The projection math is checked against
hand-constructed orthonormal vectors whose float32 arithmetic is exact
(all components are dyadic rationals), so the expected values are known
analytically rather than being read back out of the implementation.

Basis used throughout:

    r = [0.5, 0.5,  0.5,  0.5]   (unit)
    w = [0.5, 0.5, -0.5, -0.5]   (unit, orthogonal to r)
    v = 3*r - 2*w = [0.5, 0.5, 2.5, 2.5]   ->  v . r == 3.0 exactly
"""

import contextlib
import inspect
import logging
import math
import sys
import threading
from unittest.mock import patch

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

from vllm_mlx.compile import apply_compile, is_compiled
from vllm_mlx.patches import deepseek_v4_refusal as refusal

_LOG = refusal.__name__  # "vllm_mlx.patches.deepseek_v4_refusal"
_SERVER_LOG = "vllm_mlx.server"

# --- exact float32 basis -----------------------------------------------------

R = mx.array([0.5, 0.5, 0.5, 0.5], dtype=mx.float32)
W = mx.array([0.5, 0.5, -0.5, -0.5], dtype=mx.float32)
V = mx.array([0.5, 0.5, 2.5, 2.5], dtype=mx.float32)  # 3*R - 2*W
V_DOT_R = 3.0
V_DOT_W = -2.0


def _dot(a, b):
    return float(mx.sum(a * b).item())


# --- synthetic module tree ---------------------------------------------------


class _Block:
    """Stand-in for a DeepseekV4DecoderLayer / MTP block: owns ``.attn``."""

    def __init__(self, attn):
        self.attn = attn


class _MTPHolder:
    def __init__(self, blocks):
        self.layers = blocks


class _Inner:
    """Stand-in for DeepseekV4Model: owns the full block list."""

    def __init__(self, layers, mtp=None):
        self.layers = layers
        if mtp is not None:
            self.mtp = mtp


class _Wrapper:
    """Stand-in for an outer wrapper whose ``.layers`` is a sliced view."""

    def __init__(self, inner, layers):
        self.model = inner
        self.layers = layers


class _CallableInner(_Inner):
    """An ``_Inner`` that is callable, so ``apply_compile`` can wrap it.

    Mirrors what ``vllm_mlx.compile.apply_compile`` is pointed at in serving:
    the object whose forward pass runs every attention module.
    """

    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = layer.attn(out)
        return out


def _tree(n_layers, n_mtp=0, attn_factory=object):
    layers = [_Block(attn_factory()) for _ in range(n_layers)]
    mtp = _MTPHolder([_Block(attn_factory()) for _ in range(n_mtp)]) if n_mtp else None
    return _Inner(layers, mtp)


def _write_sidecar(tmp_path, keys, direction=R, scale=1.0, name="refusal_dirs"):
    """Write a sidecar with ``scale * direction`` under each ``<key>.attn.wo_b``."""
    data = {f"{k}.attn.wo_b": (direction * scale) for k in keys}
    p = tmp_path / f"{name}.safetensors"
    mx.save_safetensors(str(p), data)
    return p


def _make_attn_class(out):
    """A fake attention class whose ``__call__`` returns a known array."""

    class FakeV4Attention:
        def __init__(self):
            self.calls = 0

        def __call__(self, x, *args, **kwargs):
            self.calls += 1
            return out

    return FakeV4Attention


@pytest.fixture(autouse=True)
def _clean_refusal_state():
    """Process-global dial state must not leak between tests, in any order."""
    refusal.reset()
    yield
    refusal.reset()


@pytest.fixture
def attn_cls():
    """A fresh fake attention class, guaranteed unpatched at teardown."""
    cls = _make_attn_class(V)
    yield cls
    refusal.reset(cls)


# --- 1-5: the projection math ------------------------------------------------


def test_lambda_zero_is_bit_exact_passthrough():
    """lambda=0 short-circuits: same object back, bit-identical."""
    out = refusal.apply_projection(V, R, 0.0)
    assert out is V
    assert bool(mx.all(out == V).item())
    assert out.dtype == V.dtype


def test_lambda_one_removes_exactly_the_component_along_r():
    """v = 3r - 2w at lambda=1 -> -2w: r-component gone, w-component intact."""
    out = refusal.apply_projection(V, R, 1.0)
    assert _dot(out, R) == 0.0  # exact, not a tolerance
    assert _dot(out, W) == V_DOT_W
    assert bool(mx.all(out == (W * -2.0)).item())


def test_lambda_two_overshoots_and_negative_lambda_amplifies():
    """lambda=2 flips the projection's sign; lambda<0 grows it (more reticent)."""
    over = refusal.apply_projection(V, R, 2.0)
    assert _dot(over, R) == -V_DOT_R
    assert _dot(over, R) < 0 < V_DOT_R
    assert _dot(over, W) == V_DOT_W  # orthogonal part untouched

    amplified = refusal.apply_projection(V, R, -1.0)
    assert _dot(amplified, R) == 2 * V_DOT_R
    assert _dot(amplified, R) > V_DOT_R > 0
    assert _dot(amplified, W) == V_DOT_W


def test_loader_normalizes_a_non_unit_direction(tmp_path, attn_cls):
    """A sidecar direction of norm 2 must behave like its normalized form.

    Guards the loader's normalization step, not apply_projection: a non-unit
    r would silently rescale lambda by |r|^2 (here 4x).
    """
    path = _write_sidecar(tmp_path, ["layers.0"], direction=R, scale=2.0)
    model = _tree(1, attn_factory=attn_cls)
    assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 1

    stored = refusal._directions[id(model.layers[0].attn)]
    assert abs(float(mx.linalg.norm(stored).item()) - 1.0) < 1e-6
    assert bool(mx.all(stored == R).item())

    refusal.set_lambda(1.0)
    got = model.layers[0].attn(None)
    # Normalized: exactly the rank-1 removal. Un-normalized it would be
    # v - 4*(v.r)*r, i.e. r-component -9.0 instead of 0.
    assert _dot(got, R) == 0.0
    assert bool(mx.all(got == refusal.apply_projection(V, R, 1.0)).item())


def test_batched_input_is_projected_per_position_independently():
    """[B, S, H] is projected on the last axis only; positions don't mix."""
    rows = [
        V,  # 3r - 2w
        W,  # pure orthogonal: must be untouched
        R,  # pure along r
        V * 2.0,
    ]
    batched = mx.stack([mx.stack(rows[:2]), mx.stack(rows[2:])])  # [2, 2, 4]
    assert batched.shape == (2, 2, 4)

    out = refusal.apply_projection(batched, R, 1.0)
    assert out.shape == batched.shape

    expected = [
        refusal.apply_projection(row, R, 1.0)
        for row in (rows[0], rows[1], rows[2], rows[3])
    ]
    flat = out.reshape(4, 4)
    for i, exp in enumerate(expected):
        assert bool(mx.all(flat[i] == exp).item()), f"row {i} differs"

    # Position independence, spelled out: the orthogonal row survives whole,
    # the pure-r row is annihilated.
    assert bool(mx.all(out[0, 1] == W).item())
    assert _dot(out[1, 0], R) == 0.0


# --- 6-9: module-tree plumbing ----------------------------------------------


def test_iter_attention_modules_yields_expected_keys_in_order():
    model = _tree(3, n_mtp=2)
    keys = [k for k, _ in refusal._iter_attention_modules(model)]
    assert keys == ["layers.0", "layers.1", "layers.2", "mtp.0", "mtp.1"]

    mods = [m for _, m in refusal._iter_attention_modules(model)]
    assert mods[0] is model.layers[0].attn
    assert mods[3] is model.mtp.layers[0].attn


def test_iter_attention_modules_with_no_mtp_at_all():
    model = _tree(2)
    assert not hasattr(model, "mtp")
    keys = [k for k, _ in refusal._iter_attention_modules(model)]
    assert keys == ["layers.0", "layers.1"]


def test_extra_sidecar_keys_are_tolerated(tmp_path, attn_cls):
    """Sidecar carries 3 mtp directions; the checkpoint has 1. No crash."""
    path = _write_sidecar(tmp_path, ["layers.0", "layers.1", "mtp.0", "mtp.1", "mtp.2"])
    model = _tree(2, n_mtp=1, attn_factory=attn_cls)

    hooked = refusal.load_refusal_directions(model, path, attn_cls=attn_cls)
    assert hooked == 3  # 2 backbone + 1 mtp
    assert refusal.status()["modules"] == 3
    assert refusal.status()["installed"] is True


def test_unwrap_model_picks_the_deepest_full_layer_list():
    """Guards the pipeline-sliced-view bug: outer .layers is a short view."""
    deep = _tree(5, n_mtp=1)
    mid = _Wrapper(deep, layers=deep.layers[:2])
    outer = _Wrapper(mid, layers=deep.layers[:1])

    assert refusal._unwrap_model(outer) is deep
    keys = [k for k, _ in refusal._iter_attention_modules(outer)]
    assert keys == ["layers.0", "layers.1", "layers.2", "layers.3", "layers.4", "mtp.0"]


def test_unwrap_model_returns_input_when_nothing_has_layers():
    plain = object()
    assert refusal._unwrap_model(plain) is plain


# --- 10: fail-open loader ----------------------------------------------------


def test_load_returns_zero_and_installs_nothing_when_no_key_matches(tmp_path, attn_cls):
    path = _write_sidecar(tmp_path, ["layers.99", "mtp.7"])
    model = _tree(2, attn_factory=attn_cls)

    assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 0
    assert refusal.status() == {"lambda": 0.0, "installed": False, "modules": 0}
    assert refusal._directions == {}
    assert refusal._hooked_modules == []
    assert not getattr(attn_cls, refusal._PATCH_FLAG, False)


def test_load_returns_zero_when_path_does_not_exist(tmp_path, attn_cls):
    model = _tree(2, attn_factory=attn_cls)
    missing = tmp_path / "nope" / "refusal_dirs.safetensors"

    assert refusal.load_refusal_directions(model, missing, attn_cls=attn_cls) == 0
    assert refusal.status()["installed"] is False
    assert not getattr(attn_cls, refusal._PATCH_FLAG, False)


def test_load_accepts_a_directory_containing_the_sidecar(tmp_path, attn_cls):
    _write_sidecar(tmp_path, ["layers.0"])
    model = _tree(1, attn_factory=attn_cls)
    assert refusal.load_refusal_directions(model, tmp_path, attn_cls=attn_cls) == 1


# --- 11-13: the served hook -------------------------------------------------


def test_hook_is_bit_exact_at_zero_and_matches_apply_projection_at_1_5(
    tmp_path, attn_cls
):
    """What is served must BE apply_projection, not a copy of the math."""
    path = _write_sidecar(tmp_path, ["layers.0"])
    model = _tree(1, attn_factory=attn_cls)
    assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 1
    attn = model.layers[0].attn

    refusal.set_lambda(0.0)
    off = attn(None)
    assert off is V
    assert bool(mx.all(off == V).item())

    refusal.set_lambda(1.5)
    on = attn(None)
    expected = refusal.apply_projection(V, R, 1.5)
    assert bool(mx.all(on == expected).item())
    assert on.dtype == V.dtype
    assert attn.calls == 2  # the original __call__ ran both times


def test_module_without_a_direction_passes_through_untouched(tmp_path, attn_cls):
    """layers.1 has no sidecar entry: identity even at lambda != 0."""
    path = _write_sidecar(tmp_path, ["layers.0"])
    model = _tree(2, attn_factory=attn_cls)
    assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 1

    refusal.set_lambda(1.5)
    hooked_out = model.layers[0].attn(None)
    bare_out = model.layers[1].attn(None)

    assert bare_out is V
    assert bool(mx.all(bare_out == V).item())
    assert not bool(mx.all(hooked_out == V).item())  # the other one did change


def test_installing_twice_does_not_double_apply(tmp_path, attn_cls):
    path = _write_sidecar(tmp_path, ["layers.0"])
    model = _tree(1, attn_factory=attn_cls)

    assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 1
    first_wrapper = attn_cls.__call__
    pristine = getattr(attn_cls, refusal._ORIGINAL_CALL)

    assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 1
    assert attn_cls.__call__ is first_wrapper  # not re-wrapped
    assert getattr(attn_cls, refusal._ORIGINAL_CALL) is pristine

    refusal.set_lambda(1.0)
    out = model.layers[0].attn(None)
    once = refusal.apply_projection(V, R, 1.0)
    assert bool(mx.all(out == once).item())
    assert _dot(out, R) == 0.0  # double-applied would be -3.0


def test_uninstall_restores_the_original_call(tmp_path, attn_cls):
    path = _write_sidecar(tmp_path, ["layers.0"])
    model = _tree(1, attn_factory=attn_cls)
    refusal.load_refusal_directions(model, path, attn_cls=attn_cls)
    refusal.uninstall_hook(attn_cls)

    assert not getattr(attn_cls, refusal._PATCH_FLAG, False)
    refusal.set_lambda(1.5)
    assert model.layers[0].attn(None) is V


# --- 14: the dial ------------------------------------------------------------


def test_lambda_round_trip_and_status(tmp_path, attn_cls):
    assert refusal.get_lambda() == 0.0
    assert refusal.status() == {"lambda": 0.0, "installed": False, "modules": 0}

    assert refusal.set_lambda(1.25) == 1.25
    assert refusal.get_lambda() == 1.25
    assert refusal.status()["lambda"] == 1.25
    assert refusal.status()["installed"] is False

    path = _write_sidecar(tmp_path, ["layers.0", "layers.1", "mtp.0"])
    model = _tree(2, n_mtp=1, attn_factory=attn_cls)
    assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 3

    st = refusal.status()
    assert st["installed"] is True
    assert st["modules"] == 3
    assert st["lambda"] == 1.25  # install does not reset the dial

    assert refusal.set_lambda(-2.5) == -2.5
    assert refusal.status()["lambda"] == -2.5


# --- 15-16: admin endpoints -------------------------------------------------


@pytest.fixture
def client():
    from vllm_mlx.server import app

    return TestClient(app)


def test_get_admin_refusal_lambda_reports_status(client, tmp_path, attn_cls):
    resp = client.get("/admin/refusal_lambda")
    assert resp.status_code == 200
    assert resp.json() == {"lambda": 0.0, "installed": False, "modules": 0}

    path = _write_sidecar(tmp_path, ["layers.0", "layers.1"])
    model = _tree(2, attn_factory=attn_cls)
    refusal.load_refusal_directions(model, path, attn_cls=attn_cls)
    refusal.set_lambda(1.5)

    body = client.get("/admin/refusal_lambda").json()
    assert body == {"lambda": 1.5, "installed": True, "modules": 2}


def test_post_admin_refusal_lambda_409_when_nothing_installed(client):
    resp = client.post("/admin/refusal_lambda", json={"lambda": 1.5})
    assert resp.status_code == 409
    assert "--refusal-dirs" in resp.json()["detail"]
    assert refusal.get_lambda() == 0.0  # unchanged


def _install_for_endpoint(tmp_path, attn_cls, n=2):
    path = _write_sidecar(tmp_path, [f"layers.{i}" for i in range(n)])
    model = _tree(n, attn_factory=attn_cls)
    assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == n
    return model


def test_post_admin_refusal_lambda_sets_and_echoes(client, tmp_path, attn_cls):
    _install_for_endpoint(tmp_path, attn_cls)

    resp = client.post("/admin/refusal_lambda", json={"lambda": 1.5})
    assert resp.status_code == 200
    assert resp.json() == {"lambda": 1.5, "installed": True, "modules": 2}
    assert refusal.get_lambda() == 1.5

    # Negative (more reticent) and zero (stock) are both in range.
    assert (
        client.post("/admin/refusal_lambda", json={"lambda": -1.0}).json()["lambda"]
        == -1.0
    )
    assert (
        client.post("/admin/refusal_lambda", json={"lambda": 0}).json()["lambda"] == 0.0
    )
    assert refusal.get_lambda() == 0.0


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_post_admin_refusal_lambda_400_on_non_finite(client, tmp_path, attn_cls, bad):
    _install_for_endpoint(tmp_path, attn_cls)
    refusal.set_lambda(0.5)

    resp = client.post(
        "/admin/refusal_lambda",
        content=f'{{"lambda": {bad}}}',
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert "finite" in resp.json()["detail"]
    assert refusal.get_lambda() == 0.5  # not applied


@pytest.mark.parametrize("bad", [10.001, -10.001, 1e6, -1e6])
def test_post_admin_refusal_lambda_400_on_out_of_range(client, tmp_path, attn_cls, bad):
    _install_for_endpoint(tmp_path, attn_cls)
    refusal.set_lambda(0.5)

    resp = client.post("/admin/refusal_lambda", json={"lambda": bad})
    assert resp.status_code == 400
    assert "between -10 and 10" in resp.json()["detail"]
    assert refusal.get_lambda() == 0.5


def test_post_admin_refusal_lambda_accepts_the_range_boundaries(
    client, tmp_path, attn_cls
):
    _install_for_endpoint(tmp_path, attn_cls)
    for value in (10.0, -10.0):
        resp = client.post("/admin/refusal_lambda", json={"lambda": value})
        assert resp.status_code == 200, resp.text
        assert resp.json()["lambda"] == value


# --- 17: CLI flag -----------------------------------------------------------


def _parse_serve_args(*argv_extra):
    """Drive cli.main's argparse far enough to capture the load_model kwargs.

    Same short-circuit pattern as tests/test_cli_max_thinking_token_budget.py.
    """
    from vllm_mlx import cli

    with (
        patch("vllm_mlx.server.load_model") as load_model,
        patch("uvicorn.run"),
        patch.object(sys, "argv", ["vllm-mlx", "serve", "some-model", *argv_extra]),
        contextlib.suppress(SystemExit),  # argparse.exit / uvicorn no-op
    ):
        cli.main()
    assert load_model.call_args is not None, "load_model was never reached"
    return load_model.call_args.kwargs


def test_refusal_dirs_flag_is_threaded_into_load_model():
    kwargs = _parse_serve_args("--refusal-dirs", "some/repo-id")
    assert kwargs["refusal_dirs"] == "some/repo-id"


def test_refusal_dirs_defaults_to_none():
    kwargs = _parse_serve_args()
    assert kwargs["refusal_dirs"] is None


# --- server-side install wiring (fail-open) ---------------------------------


def test_install_is_a_noop_without_the_flag(monkeypatch):
    from vllm_mlx import server

    monkeypatch.setattr(server, "_refusal_dirs", None, raising=False)
    assert server._install_refusal_directions() == 0
    assert refusal.status()["installed"] is False


def test_install_fails_open_when_no_model_is_loaded(monkeypatch):
    from vllm_mlx import server

    monkeypatch.setattr(server, "_refusal_dirs", "some/repo-id", raising=False)
    monkeypatch.setattr(server, "_engine", None, raising=False)
    assert server._install_refusal_directions() == 0
    assert refusal.status()["installed"] is False


def test_install_wires_the_engines_model(monkeypatch, tmp_path, attn_cls):
    """_engine._model is the object handed to load_refusal_directions."""
    from vllm_mlx import server

    path = _write_sidecar(tmp_path, ["layers.0", "layers.1"])
    model = _tree(2, attn_factory=attn_cls)

    class _Engine:
        _model = model

    monkeypatch.setattr(server, "_refusal_dirs", str(path), raising=False)
    monkeypatch.setattr(server, "_engine", _Engine(), raising=False)

    with patch.object(refusal, "_resolve_attention_class", return_value=attn_cls):
        assert server._install_refusal_directions() == 2
    assert refusal.status() == {"lambda": 0.0, "installed": True, "modules": 2}


# --- 18: mx.compile freezes the dial, so the install must refuse ------------
#
# The headline defect of the original feature: vllm_mlx/compile.py wraps the
# forward pass in mx.compile(shapeless=True), which captures the hook's
# `lam = _state["lambda"]` read as a trace-time CONSTANT. --compile and
# --refusal-dirs are independent serve flags, so a 200-OK dial that does
# nothing was one checkbox away. Every other test in this file runs eager;
# these do not.


def _compiled_sidecar(tmp_path, attn_cls, n=1):
    path = _write_sidecar(tmp_path, [f"layers.{i}" for i in range(n)])
    model = _CallableInner([_Block(attn_cls()) for _ in range(n)])
    return path, model


def test_install_is_refused_when_the_model_is_compiled(tmp_path, attn_cls, caplog):
    """A dial that cannot move must never report itself installed."""
    path, model = _compiled_sidecar(tmp_path, attn_cls)
    apply_compile(model)
    assert is_compiled(model)

    with caplog.at_level(logging.DEBUG, logger=_LOG):
        assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 0

    assert refusal.status() == {"lambda": 0.0, "installed": False, "modules": 0}
    assert refusal._directions == {}
    assert refusal._hooked_modules == []
    assert not getattr(attn_cls, refusal._PATCH_FLAG, False)

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, caplog.text
    msg = errors[0].getMessage()
    assert "--compile" in msg and "--refusal-dirs" in msg
    assert "FROZEN" in msg


def test_install_is_refused_when_compile_is_applied_one_level_down(tmp_path, attn_cls):
    """SimpleEngine compiles ``engine._model.model``; install gets ``_model``."""
    path, inner = _compiled_sidecar(tmp_path, attn_cls)
    outer = _Wrapper(inner, layers=[])
    apply_compile(inner)

    assert refusal._find_compiled(outer) is inner
    assert refusal.load_refusal_directions(outer, path, attn_cls=attn_cls) == 0
    assert refusal.status()["installed"] is False


def test_find_compiled_returns_none_for_an_uncompiled_chain(tmp_path, attn_cls):
    """The guard must not fire on the ordinary path."""
    path, inner = _compiled_sidecar(tmp_path, attn_cls)
    outer = _Wrapper(inner, layers=[])
    assert refusal._find_compiled(outer) is None
    assert refusal.load_refusal_directions(outer, path, attn_cls=attn_cls) == 1


def test_post_409s_after_a_compile_refused_install(client, tmp_path, attn_cls):
    """The refusal keeps the route honest: 409, not a 200 for a dead dial."""
    path, model = _compiled_sidecar(tmp_path, attn_cls)
    apply_compile(model)
    assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 0

    resp = client.post("/admin/refusal_lambda", json={"lambda": 1.5})
    assert resp.status_code == 409
    assert refusal.get_lambda() == 0.0


def test_compiled_forward_really_does_freeze_the_dial(tmp_path, attn_cls):
    """Why the guard exists, demonstrated rather than asserted from docs.

    Compiles AFTER a successful install (the guard only covers install time,
    which is the only order the serve path can produce) and shows the dial
    stuck at its trace-time value while ``status()`` reports the new one.
    """
    path, model = _compiled_sidecar(tmp_path, attn_cls)
    assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 1

    x = mx.zeros((1, 4), dtype=mx.float32)
    refusal.set_lambda(0.0)
    assert bool(mx.all(model(x) == V).item())
    refusal.set_lambda(1.5)
    eager = model(x)
    assert bool(mx.all(eager == refusal.apply_projection(V, R, 1.5)).item())

    apply_compile(model)
    refusal.set_lambda(0.0)
    traced = model.__call__(x)  # traced at lambda=0
    assert bool(mx.all(traced == V).item())

    refusal.set_lambda(1.5)
    frozen = model.__call__(x)
    assert bool(mx.all(frozen == V).item()), "expected the compiled trace to freeze"
    assert refusal.status()["lambda"] == 1.5  # ...while status() says otherwise


def test_apply_compile_is_bypassed_by_implicit_calls(tmp_path, attn_cls):
    """The measurement behind the "inert today" sentence in the refusal log.

    ``apply_compile`` assigns the compiled function to ``model.__call__`` as an
    INSTANCE attribute; Python resolves ``model(x)`` against
    ``type(model).__call__``. Every serving call site uses the implicit form
    (``vllm_mlx/engine/batched.py``, ``engine/simple.py``, ``specprefill.py``,
    ``mllm_batch_generator.py``), so today the compiled trace never runs and
    dropping ``--compile`` to get the dial costs nothing.
    """
    path, model = _compiled_sidecar(tmp_path, attn_cls)
    assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 1
    x = mx.zeros((1, 4), dtype=mx.float32)

    apply_compile(model)
    refusal.set_lambda(0.0)
    assert bool(mx.all(model.__call__(x) == V).item())  # trace captured at 0

    refusal.set_lambda(1.5)
    # Explicit __call__ finds the compiled instance attribute: frozen.
    assert bool(mx.all(model.__call__(x) == V).item())
    # Implicit call — the only form the serving path uses — resolves
    # type(model).__call__ and never sees the compiled function at all.
    assert bool(
        mx.all(model(x) == refusal.apply_projection(V, R, 1.5)).item()
    ), "the implicit call should still be eager, and the dial should move"


def test_compile_refusal_says_compile_is_inert_on_the_serving_path_today(
    tmp_path, attn_cls, caplog
):
    """The refusal stays (future-proofing), but must not cost an operator a
    feature they were never actually getting.
    """
    path, model = _compiled_sidecar(tmp_path, attn_cls)
    apply_compile(model)

    with caplog.at_level(logging.DEBUG, logger=_LOG):
        assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 0

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, caplog.text
    msg = " ".join(errors[0].getMessage().split())
    assert "Dropping --compile costs you nothing today" in msg
    assert "INERT on the serving path" in msg
    assert "INSTANCE attribute" in msg
    assert "type(model).__call__" in msg


# --- 19: poisonous sidecar tensors are rejected, not counted as hooked -------


def _write_raw(tmp_path, data, name="refusal_dirs"):
    p = tmp_path / f"{name}.safetensors"
    mx.save_safetensors(str(p), data)
    return p


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_direction_is_rejected_and_left_unhooked(
    tmp_path, attn_cls, caplog, bad_value
):
    """nan/inf slip past ``norm == 0`` and poison the residual stream later."""
    bad = mx.array([bad_value] * 4, dtype=mx.float32)
    path = _write_raw(tmp_path, {"layers.0.attn.wo_b": R, "layers.1.attn.wo_b": bad})
    model = _tree(2, attn_factory=attn_cls)

    with caplog.at_level(logging.DEBUG, logger=_LOG):
        assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 1

    assert refusal.status()["modules"] == 1
    assert id(model.layers[1].attn) not in refusal._directions
    assert "layers.1.attn.wo_b" in caplog.text
    assert "non-finite" in caplog.text

    # The rejected module serves stock even when the dial is up, instead of
    # emitting NaN across the whole hidden state.
    refusal.set_lambda(1.5)
    assert model.layers[1].attn(None) is V
    assert bool(mx.all(mx.isfinite(model.layers[0].attn(None))).item())


@pytest.mark.parametrize("shape", [(1, 4), (4, 1), (2, 4)])
def test_wrong_shape_direction_is_rejected(tmp_path, attn_cls, caplog, shape):
    """[1, D] broadcasts and silently changes the math; [D, 1] corrupts it."""
    vec = mx.broadcast_to(mx.array([0.5, 0.5, 0.5, 0.5]), (4,))
    bad = mx.reshape(mx.concatenate([vec] * (shape[0] * shape[1] // 4)), shape)
    path = _write_raw(tmp_path, {"layers.0.attn.wo_b": bad})
    model = _tree(1, attn_factory=attn_cls)

    with caplog.at_level(logging.DEBUG, logger=_LOG):
        assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 0

    assert refusal.status()["installed"] is False
    assert "layers.0.attn.wo_b" in caplog.text
    assert "1-D" in caplog.text
    assert str(tuple(shape)) in caplog.text


def test_direction_of_the_wrong_width_is_rejected(tmp_path, attn_cls, caplog):
    """Hidden size is reachable here, so a mismatched width is a hard no."""
    path = _write_sidecar(tmp_path, ["layers.0"])  # 4-wide directions
    model = _tree(1, attn_factory=attn_cls)

    class _Args:
        hidden_size = 8

    model.args = _Args()
    assert refusal._hidden_size(model) == 8

    with caplog.at_level(logging.DEBUG, logger=_LOG):
        assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 0
    assert "hidden size is 8" in caplog.text


def test_zero_norm_direction_is_named_in_the_log(tmp_path, attn_cls, caplog):
    zeros = mx.zeros((4,), dtype=mx.float32)
    path = _write_raw(tmp_path, {"layers.0.attn.wo_b": R, "layers.1.attn.wo_b": zeros})
    model = _tree(2, attn_factory=attn_cls)

    with caplog.at_level(logging.DEBUG, logger=_LOG):
        assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 1
    assert "layers.1.attn.wo_b" in caplog.text
    assert "zero norm" in caplog.text


# --- 20: _resolve_path is deterministic and explains itself ------------------


def test_resolve_path_prefers_the_documented_basename(tmp_path):
    """A model snapshot's own safetensors must not win over the sidecar.

    ``aaa_other`` is created first AND sorts first, so an unsorted ``glob()[0]``
    picks it under either ordering.
    """
    _write_sidecar(tmp_path, ["layers.0"], name="aaa_other")
    named = _write_sidecar(tmp_path, ["layers.0"], name="refusal_dirs")

    assert refusal._resolve_path(tmp_path) == named


def test_resolve_path_is_deterministic_when_the_named_file_is_absent(tmp_path, caplog):
    _write_sidecar(tmp_path, ["layers.0"], name="zzz_second")
    _write_sidecar(tmp_path, ["layers.0"], name="aaa_first")

    with caplog.at_level(logging.DEBUG, logger=_LOG):
        got = refusal._resolve_path(tmp_path)

    assert got.name == "aaa_first.safetensors"
    assert "2 .safetensors candidates" in caplog.text
    assert "refusal_dirs.safetensors" in caplog.text


def test_resolve_path_warns_for_a_lone_unnamed_candidate(tmp_path, caplog):
    _write_sidecar(tmp_path, ["layers.0"], name="model-00001-of-00001")
    with caplog.at_level(logging.DEBUG, logger=_LOG):
        got = refusal._resolve_path(tmp_path)
    assert got.name == "model-00001-of-00001.safetensors"
    assert "may not be a direction sidecar" in caplog.text


def test_resolve_path_logs_the_reason_it_gave_up(tmp_path, caplog):
    """ "not in the cache" and "path typo" must not collapse into one None."""
    missing = tmp_path / "nope" / "refusal_dirs.safetensors"
    with caplog.at_level(logging.DEBUG, logger=_LOG):
        assert refusal._resolve_path(missing) is None

    assert "could not be resolved" in caplog.text
    assert "nope" in caplog.text
    assert "Traceback" in caplog.text  # the exception itself, not just a note


def test_resolve_path_names_a_directory_with_no_safetensors(tmp_path, caplog):
    empty = tmp_path / "empty"
    empty.mkdir()
    with caplog.at_level(logging.DEBUG, logger=_LOG):
        assert refusal._resolve_path(empty) is None
    assert "no .safetensors" in caplog.text


# --- 21: the install log distinguishes benign from catastrophic --------------


def test_install_log_reports_both_cardinalities_and_names_orphans(
    tmp_path, attn_cls, caplog
):
    """The real model's 3 mtp orphans, and the mis-mapping that looks the same."""
    path = _write_sidecar(tmp_path, ["layers.0", "layers.1", "mtp.0", "mtp.1", "mtp.2"])
    model = _tree(2, attn_factory=attn_cls)

    with caplog.at_level(logging.DEBUG, logger=_LOG):
        assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 2

    text = caplog.text
    assert "2 model attention modules found" in text
    assert "5 sidecar directions loaded" in text
    assert "2 modules hooked" in text
    for key in ("mtp.0.attn.wo_b", "mtp.1.attn.wo_b", "mtp.2.attn.wo_b"):
        assert key in text, f"orphan {key} never named"

    orphan_records = [
        r for r in caplog.records if "matched no model module" in r.getMessage()
    ]
    assert orphan_records, text
    assert orphan_records[0].levelno == logging.WARNING


def test_install_log_stays_info_when_every_direction_maps(tmp_path, attn_cls, caplog):
    path = _write_sidecar(tmp_path, ["layers.0", "layers.1"])
    model = _tree(2, attn_factory=attn_cls)

    with caplog.at_level(logging.DEBUG, logger=_LOG):
        assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 2

    installed = [r for r in caplog.records if "modules hooked" in r.getMessage()]
    assert installed and installed[0].levelno == logging.INFO
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_sliced_view_that_hooks_too_few_modules_warns(tmp_path, attn_cls, caplog):
    """The pipeline-sliced-view failure prints at WARNING, not INFO."""
    path = _write_sidecar(tmp_path, [f"layers.{i}" for i in range(5)])
    model = _tree(2, attn_factory=attn_cls)  # only 2 of the 5 exist

    with caplog.at_level(logging.DEBUG, logger=_LOG):
        assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 2

    installed = [r for r in caplog.records if "modules hooked" in r.getMessage()]
    assert installed and installed[0].levelno == logging.WARNING


def test_install_logs_the_resolved_file_not_the_requested_directory(
    tmp_path, attn_cls, caplog
):
    """An operator must be able to read which bytes were actually loaded."""
    resolved = _write_sidecar(tmp_path, ["layers.0"])
    model = _tree(1, attn_factory=attn_cls)

    with caplog.at_level(logging.DEBUG, logger=_LOG):
        assert refusal.load_refusal_directions(model, tmp_path, attn_cls=attn_cls) == 1

    assert str(resolved.resolve()) in caplog.text


# --- 22: the cast cache (hot-path) --------------------------------------------


def test_direction_is_cast_once_per_dtype(tmp_path, attn_cls):
    path = _write_sidecar(tmp_path, ["layers.0"])
    model = _tree(1, attn_factory=attn_cls)
    refusal.load_refusal_directions(model, path, attn_cls=attn_cls)
    mid = id(model.layers[0].attn)

    first = refusal._direction_for(mid, mx.bfloat16)
    assert first.dtype == mx.bfloat16
    assert refusal._direction_for(mid, mx.bfloat16) is first
    assert bool(mx.all(first == R.astype(mx.bfloat16)).item())
    assert refusal._direction_for(mid, mx.float32) is not first


def test_cast_cache_never_survives_a_reinstall(tmp_path, attn_cls):
    """A stale cast would silently serve the previous sidecar's direction."""
    model = _tree(1, attn_factory=attn_cls)
    first = _write_sidecar(tmp_path, ["layers.0"], direction=R, name="a")
    refusal.load_refusal_directions(model, first, attn_cls=attn_cls)
    mid = id(model.layers[0].attn)
    assert bool(mx.all(refusal._direction_for(mid, mx.float32) == R).item())

    second = _write_sidecar(tmp_path, ["layers.0"], direction=W, name="b")
    refusal.load_refusal_directions(model, second, attn_cls=attn_cls)
    assert bool(mx.all(refusal._direction_for(mid, mx.float32) == W).item())

    refusal.reset()
    assert refusal._cast_cache == {}


def test_hook_math_is_unchanged_for_a_non_float32_activation(tmp_path):
    """The cast cache must not perturb what apply_projection would produce."""
    cls = _make_attn_class(V.astype(mx.bfloat16))
    try:
        path = _write_sidecar(tmp_path, ["layers.0"])
        model = _tree(1, attn_factory=cls)
        assert refusal.load_refusal_directions(model, path, attn_cls=cls) == 1

        refusal.set_lambda(1.5)
        got = model.layers[0].attn(None)
        expected = refusal.apply_projection(V.astype(mx.bfloat16), R, 1.5)
        assert got.dtype == mx.bfloat16
        assert bool(mx.all(got == expected).item())
    finally:
        refusal.reset(cls)


# --- 23: admin route auth, audit trail and read-back --------------------------


@pytest.fixture
def api_key(monkeypatch):
    from vllm_mlx import server

    monkeypatch.setattr(server, "_api_key", "secret-key", raising=False)
    return "secret-key"


def test_admin_routes_reject_unauthenticated_calls_when_a_key_is_set(
    client, tmp_path, attn_cls, api_key
):
    _install_for_endpoint(tmp_path, attn_cls)

    assert client.get("/admin/refusal_lambda").status_code == 401
    assert client.post("/admin/refusal_lambda", json={"lambda": 1.5}).status_code == 401
    assert refusal.get_lambda() == 0.0  # the dial did not move

    wrong = {"Authorization": "Bearer nope"}
    assert client.post("/admin/refusal_lambda", json={"lambda": 1.5}, headers=wrong)
    assert (
        client.post(
            "/admin/refusal_lambda", json={"lambda": 1.5}, headers=wrong
        ).status_code
        == 401
    )
    assert refusal.get_lambda() == 0.0


def test_admin_routes_accept_the_right_key(client, tmp_path, attn_cls, api_key):
    _install_for_endpoint(tmp_path, attn_cls)
    auth = {"Authorization": f"Bearer {api_key}"}

    assert client.get("/admin/refusal_lambda", headers=auth).status_code == 200
    resp = client.post("/admin/refusal_lambda", json={"lambda": 1.5}, headers=auth)
    assert resp.status_code == 200
    assert resp.json()["lambda"] == 1.5


def test_admin_routes_are_unchanged_for_a_keyless_deployment(
    client, tmp_path, attn_cls, monkeypatch
):
    """Adding the dependency must be a no-op for today's keyless server."""
    from vllm_mlx import server

    monkeypatch.setattr(server, "_api_key", None, raising=False)
    _install_for_endpoint(tmp_path, attn_cls)

    assert client.get("/admin/refusal_lambda").status_code == 200
    resp = client.post("/admin/refusal_lambda", json={"lambda": 0.5})
    assert resp.status_code == 200
    assert refusal.get_lambda() == 0.5


def test_post_logs_actor_and_both_lambda_values(client, tmp_path, attn_cls, caplog):
    """This log line is the entire audit trail for a safety-relevant change."""
    _install_for_endpoint(tmp_path, attn_cls)
    refusal.set_lambda(0.25)

    with caplog.at_level(logging.DEBUG, logger=_SERVER_LOG):
        assert (
            client.post("/admin/refusal_lambda", json={"lambda": 1.5}).status_code
            == 200
        )

    lines = [
        r.getMessage()
        for r in caplog.records
        if "[refusal] lambda changed" in r.getMessage()
    ]
    assert lines, caplog.text
    line = lines[0]
    assert "0.250" in line, line  # previous value
    assert "1.500" in line, line  # new value
    assert "testclient" in line, line  # the actor


class _LockProbe:
    """``refusal._lock``, with a second writer landing at a chosen release.

    A writer that lands exactly at a lock RELEASE is invisible to a
    set-and-read done under one hold and fatal to one that takes the lock
    twice. That makes the difference deterministic instead of hoping a sleep
    wins a race — and, unlike stubbing ``apply_lambda``, it leaves the
    production function itself running.
    """

    def __init__(self, lock, fire_when, land):
        self._lock = lock
        self._fire_when = fire_when
        self._land = land
        self._armed = True

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *exc):
        released = self._lock.__exit__(*exc)
        if self._armed and self._fire_when():
            self._armed = False  # the racing writer must not re-trigger itself
            self._land()
        return released


def test_post_reports_a_dial_value_that_was_actually_live(
    client, tmp_path, attn_cls, monkeypatch
):
    """Production ``apply_lambda``, raced by a real thread at the lock boundary.

    The previous version of this test replaced ``refusal.apply_lambda`` with a
    stub that raced and then re-read ``status()``, and asserted on the stub's
    output — production ``apply_lambda`` never ran, so the test asserted its
    own double. Here the only thing patched is the module lock, and writer B
    is a real thread contending for it.

    What is asserted is the guarantee that actually holds. "What comes back is
    what the next forward pass will use" cannot be delivered by any read that
    completes before the response is serialized — the real production sequence
    returns ``{'lambda': 1.5}`` while the live value is already 0.0. What CAN
    be delivered, and is: the value returned was the live dial at the instant
    this request set it, so a caller is never told a number that was never
    live. A later writer may already have superseded it.
    """
    _install_for_endpoint(tmp_path, attn_cls)
    # Not a double: this is the module's own function, unwrapped.
    assert refusal.apply_lambda.__module__ == refusal.__name__

    landed: list[float] = []

    def _writer_b():
        refusal.set_lambda(0.0)
        landed.append(refusal.get_lambda())

    def _land():
        thread = threading.Thread(target=_writer_b)
        thread.start()
        thread.join(5)
        assert not thread.is_alive(), "the racing writer never landed"

    monkeypatch.setattr(
        refusal,
        "_lock",
        _LockProbe(
            refusal._lock,
            fire_when=lambda: refusal._state["lambda"] == 1.5,
            land=_land,
        ),
    )

    body = client.post("/admin/refusal_lambda", json={"lambda": 1.5}).json()

    assert landed == [0.0], "the racing writer did not run"
    # A set-then-read across two lock holds would have returned writer B's 0.0.
    assert body == {"lambda": 1.5, "installed": True, "modules": 2}
    # ...and by the time the client reads it, it is already stale.
    assert refusal.get_lambda() == 0.0

    # The route must not claim more than that in its own comments either.
    from vllm_mlx import server

    src = " ".join(
        inspect.getsource(server.post_refusal_lambda).replace("#", " ").split()
    )
    assert "what the next forward pass will actually use" not in src
    assert "WAS the live dial at the instant this request set it" in src


# --- 27: the dial is read per MODULE, not per request or per forward ---------


class _MidPassDialFlip(_CallableInner):
    """A forward pass that changes the dial partway through.

    Stands in for the admin thread landing between two attention modules,
    which is what actually happens: the hook's ``_state["lambda"]`` read is
    per module per token, so nothing holds a value steady for a whole pass.
    """

    def __init__(self, layers, flip_at, to):
        super().__init__(layers)
        self.flip_at = flip_at
        self.to = to

    def __call__(self, x):
        seen = []
        for i, layer in enumerate(self.layers):
            if i == self.flip_at:
                refusal.set_lambda(self.to)
            seen.append(layer.attn(x))
        return seen


def test_a_dial_change_lands_mid_forward_and_yields_a_hybrid_pass(tmp_path, attn_cls):
    """One forward pass, two lambdas — the semantics the docs now state."""
    path = _write_sidecar(tmp_path, [f"layers.{i}" for i in range(4)])
    model = _MidPassDialFlip([_Block(attn_cls()) for _ in range(4)], flip_at=2, to=1.5)
    assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 4

    refusal.set_lambda(0.0)
    seen = model(mx.zeros((1, 4), dtype=mx.float32))

    stock = V
    dialed = refusal.apply_projection(V, R, 1.5)
    assert not bool(mx.all(dialed == stock).item())  # the two really differ

    assert bool(mx.all(seen[0] == stock).item()), "layer 0 should predate the flip"
    assert bool(mx.all(seen[1] == stock).item()), "layer 1 should predate the flip"
    assert bool(mx.all(seen[2] == dialed).item()), "layer 2 should follow the flip"
    assert bool(mx.all(seen[3] == dialed).item()), "layer 3 should follow the flip"
    assert refusal.get_lambda() == 1.5


def test_the_dial_docs_state_the_real_concurrency_semantics():
    """Both of these used to describe the opposite of what happens.

    The hook comment claimed the worst case was a whole forward pass seeing
    the old value; the route claimed the change was "effective on the next
    request". A change actually lands mid-request, mid-token, and the pass in
    flight is a hybrid of both lambdas.
    """
    from vllm_mlx import server

    # Strip comment markers first, so a phrase that wrapped across two comment
    # lines still reads as one sentence.
    hook = " ".join(inspect.getsource(refusal._install_hook).replace("#", " ").split())
    assert "one forward pass in flight" not in hook
    assert "HYBRID" in hook
    assert "per attention module per token" in hook

    route = inspect.getdoc(server.post_refusal_lambda)
    assert "effective on the next request" not in route
    assert "HYBRID" in route
    assert "cancel the request" in " ".join(route.split())

    setter = inspect.getdoc(refusal.set_lambda)
    assert "Takes effect on the next forward pass" not in setter


def test_apply_lambda_returns_previous_and_a_coherent_snapshot(tmp_path, attn_cls):
    _install_for_endpoint(tmp_path, attn_cls)
    refusal.set_lambda(0.75)

    previous, st = refusal.apply_lambda(-2.0)
    assert previous == 0.75
    assert st == {"lambda": -2.0, "installed": True, "modules": 2}
    assert refusal.get_lambda() == -2.0


# --- 24: the production wiring (lifespan + load_model) ------------------------
#
# The `client` fixture uses a bare TestClient, so the app lifespan never runs
# in any other test here. That let three separate mutations of the production
# wiring survive the whole suite: deleting the _install_refusal_directions()
# call, dropping the refusal_dirs argument on the floor inside load_model, and
# installing BEFORE the engine has weights. These tests exercise the real
# startup path with `with TestClient(app)`.


class _LifespanEngine:
    """Stub engine whose weights only exist once ``start()`` has been awaited."""

    def __init__(self, model, events):
        self._loaded = False
        self._model = None
        self._real_model = model
        self._events = events
        self.preserve_native_tool_format = False

    async def start(self):
        self._events.append("start")
        self._loaded = True
        self._model = self._real_model

    async def stop(self):
        self._events.append("stop")


def _run_lifespan(monkeypatch, refusal_dirs, model, attn_cls):
    from vllm_mlx import server

    events: list[str] = []
    engine = _LifespanEngine(model, events)
    monkeypatch.setattr(server, "_engine", engine, raising=False)
    monkeypatch.setattr(server, "_refusal_dirs", refusal_dirs, raising=False)
    monkeypatch.delenv("VLLM_MLX_MCP_CONFIG", raising=False)

    with patch.object(refusal, "_resolve_attention_class", return_value=attn_cls):
        with TestClient(server.app):
            pass
    return events


def test_lifespan_installs_the_dial_after_the_engine_has_weights(
    monkeypatch, tmp_path, attn_cls
):
    """Fails if the install call is deleted, or moved ahead of engine.start()."""
    path = _write_sidecar(tmp_path, ["layers.0", "layers.1"])
    model = _tree(2, attn_factory=attn_cls)

    events = _run_lifespan(monkeypatch, str(path), model, attn_cls)

    assert events == ["start", "stop"]
    assert refusal.status() == {"lambda": 0.0, "installed": True, "modules": 2}
    assert refusal._directions.keys() == {
        id(model.layers[0].attn),
        id(model.layers[1].attn),
    }


def test_lifespan_does_not_install_without_the_flag(monkeypatch, tmp_path, attn_cls):
    model = _tree(2, attn_factory=attn_cls)
    _run_lifespan(monkeypatch, None, model, attn_cls)
    assert refusal.status()["installed"] is False


def test_lifespan_still_boots_when_the_sidecar_is_corrupt(
    monkeypatch, tmp_path, attn_cls
):
    """Fail-open, end to end: a bad sidecar must not take the server down."""
    bad = tmp_path / "refusal_dirs.safetensors"
    bad.write_bytes(b"this is not a safetensors file")
    model = _tree(2, attn_factory=attn_cls)

    events = _run_lifespan(monkeypatch, str(bad), model, attn_cls)

    assert events == ["start", "stop"]  # startup completed
    assert refusal.status()["installed"] is False


def test_load_model_records_the_refusal_dirs_flag(monkeypatch):
    """The CLI flag has to survive load_model, not merely reach it.

    The mock-based flag test asserts what argparse handed to load_model; this
    asserts what load_model did with it, which is the part lifespan reads.
    """
    from vllm_mlx import server

    class _StubEngine:
        def __init__(self, **kwargs):
            self.preserve_native_tool_format = False

    monkeypatch.setattr(server, "BatchedEngine", _StubEngine)
    monkeypatch.setattr(server, "_engine", None, raising=False)
    monkeypatch.setattr(server, "_refusal_dirs", "stale-value", raising=False)
    monkeypatch.setattr(server, "_model_name", None, raising=False)
    monkeypatch.setattr(server, "_model_path", None, raising=False)
    monkeypatch.setattr(server, "_tool_parser_instance", None, raising=False)
    monkeypatch.setattr(server, "_default_max_tokens", 32768, raising=False)

    server.load_model("fake-model", use_batching=True, refusal_dirs="some/repo-id")
    assert server._refusal_dirs == "some/repo-id"

    server.load_model("fake-model", use_batching=True)
    assert server._refusal_dirs is None  # absent flag clears a stale value


# --- 25: fail-open and the zero-norm guard -----------------------------------


def test_loader_fails_open_on_a_corrupt_sidecar(tmp_path, attn_cls, caplog):
    """The broad except is the ONLY thing between bad bytes and a dead boot.

    ``_install_refusal_directions`` does not wrap this call, and lifespan does
    not wrap that, so a raise here fails the whole server startup.
    """
    bad = tmp_path / "refusal_dirs.safetensors"
    bad.write_bytes(b"this is not a safetensors file")
    model = _tree(1, attn_factory=attn_cls)

    with caplog.at_level(logging.DEBUG, logger=_LOG):
        assert refusal.load_refusal_directions(model, bad, attn_cls=attn_cls) == 0

    assert refusal.status() == {"lambda": 0.0, "installed": False, "modules": 0}
    assert refusal._directions == {}
    assert refusal._hooked_modules == []
    assert not getattr(attn_cls, refusal._PATCH_FLAG, False)
    assert "failed to install" in caplog.text
    assert "Traceback" in caplog.text


def test_all_zero_direction_is_rejected_and_counted_unmatched(
    tmp_path, attn_cls, caplog
):
    """A layer whose direction was never computed: vec / 0.0 would be NaN.

    Without the zero-norm guard this hooks and every forward pass at lambda!=0
    returns NaN hidden states — silently, and only once someone turns the dial.
    """
    zeros = mx.zeros((4,), dtype=mx.float32)
    path = _write_raw(tmp_path, {"layers.0.attn.wo_b": R, "layers.1.attn.wo_b": zeros})
    model = _tree(2, attn_factory=attn_cls)

    with caplog.at_level(logging.DEBUG, logger=_LOG):
        assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 1

    assert refusal.status()["modules"] == 1
    assert id(model.layers[1].attn) not in refusal._directions
    assert "layers.1.attn.wo_b" in caplog.text
    assert "zero norm" in caplog.text

    refusal.set_lambda(1.5)
    assert model.layers[1].attn(None) is V  # untouched, not NaN
    assert bool(mx.all(mx.isfinite(model.layers[0].attn(None))).item())


def test_a_direction_whose_norm_overflows_to_inf_is_rejected(
    tmp_path, attn_cls, caplog
):
    """Finite components, INFINITE norm: ``vec / inf`` is an all-ZERO direction.

    Both guards that came before this one wave it through — a corrupt or
    truncated sidecar whose bytes reinterpret as ~1e38 floats has
    ``isfinite(vec).all() == True`` and ``norm == 0`` False — and the module
    was then counted in ``hooked``, logged as installed and reported by
    ``status()`` while projecting exactly nothing at every lambda. That is the
    "hooked but silently inert" hole this feature keeps having to close.
    """
    huge = mx.array([3.0e38] * 4, dtype=mx.float32)
    assert bool(mx.isfinite(huge).all().item())  # clears the component check
    norm = mx.linalg.norm(huge).item()
    assert norm != 0  # ...and clears a bare `norm == 0` guard
    assert not math.isfinite(norm)
    assert bool(mx.all((huge / norm) == 0.0).item())  # what it would install

    path = _write_raw(tmp_path, {"layers.0.attn.wo_b": R, "layers.1.attn.wo_b": huge})
    model = _tree(2, attn_factory=attn_cls)

    with caplog.at_level(logging.DEBUG, logger=_LOG):
        assert refusal.load_refusal_directions(model, path, attn_cls=attn_cls) == 1

    assert refusal.status()["modules"] == 1
    assert id(model.layers[1].attn) not in refusal._directions
    assert "layers.1.attn.wo_b" in caplog.text
    assert "non-finite norm" in caplog.text

    # Counted as an orphan, exactly like every other reject.
    unusable = [
        r for r in caplog.records if "got no usable direction" in r.getMessage()
    ]
    assert unusable and "layers.1" in unusable[0].getMessage()

    # The layer serves stock rather than carrying an inert hook.
    refusal.set_lambda(1.5)
    assert model.layers[1].attn(None) is V
    assert not bool(mx.all(model.layers[0].attn(None) == V).item())


# --- 26: the projection coefficient is a matvec ------------------------------


def _reduction_projection(out, r, lam):
    """The pre-matvec form of apply_projection, kept as a reference."""
    if lam == 0.0:
        return out
    r = r.astype(out.dtype)
    return out - lam * mx.sum(out * r, axis=-1, keepdims=True) * r


def test_matvec_projection_matches_the_reduction_bit_for_bit_on_this_basis():
    """The switch to ``out @ r`` must not move a single bit on the exact basis.

    Guards a botched matvec (wrong axis, missing keepdims, a broadcast that
    silently changes the math) rather than the perf win itself.
    """
    batched = mx.stack([mx.stack([V, W]), mx.stack([R, V * 2.0])])
    for lam in (1.0, 1.5, 2.0, -1.0, 0.25):
        for x in (V, W, R, batched):
            got = refusal.apply_projection(x, R, lam)
            assert got.shape == x.shape, (lam, x.shape)
            assert bool(
                mx.all(got == _reduction_projection(x, R, lam)).item()
            ), f"lam={lam} shape={x.shape}"


def test_projection_does_not_materialize_a_full_size_intermediate():
    """The reason for the matvec: ``out * r`` allocates a whole extra activation.

    Fails against the reduction form. Measured [1, 4096, 4096] bfloat16 on an
    M3 Ultra: reduction 100.7 MB, matvec 67.1 MB. The assertion carries a wide
    margin because the point is the missing allocation, not a precise number.
    """
    x = mx.random.normal((1, 4096, 4096)).astype(mx.bfloat16)
    r = mx.random.normal((4096,)).astype(mx.bfloat16)
    mx.eval(x, r)

    def peak_mb(fn):
        best = None
        for _ in range(2):
            mx.clear_cache()
            mx.reset_peak_memory()
            base = mx.get_peak_memory()
            mx.eval(fn(x, r, 1.5))
            used = (mx.get_peak_memory() - base) / 1e6
            best = used if best is None else min(best, used)
        return best

    served = peak_mb(refusal.apply_projection)
    reduction = peak_mb(_reduction_projection)
    assert served < reduction * 0.85, f"matvec {served:.1f} MB vs {reduction:.1f} MB"


def test_lambda_zero_is_where_the_exactness_guarantee_lives():
    """The short-circuit, not the arithmetic form, is what makes "off" exact."""
    src = inspect.getsource(refusal.apply_projection)
    assert "if lam == 0.0:" in src
    assert "mx.sum(out * r" not in src, "reduction form is back"
    assert "out @ r" in src

    for dtype in (mx.float32, mx.bfloat16, mx.float16):
        x = V.astype(dtype)
        assert refusal.apply_projection(x, R, 0.0) is x


def test_hooked_modules_holds_a_strong_ref_per_hooked_module(tmp_path, attn_cls):
    """The id() keys are only safe while these strong refs exist."""
    path = _write_sidecar(tmp_path, ["layers.0", "layers.1", "mtp.0"])
    model = _tree(2, n_mtp=1, attn_factory=attn_cls)

    hooked = refusal.load_refusal_directions(model, path, attn_cls=attn_cls)
    assert hooked == 3
    assert len(refusal._hooked_modules) == hooked
    assert len(refusal._directions) == hooked
    assert {id(m) for m in refusal._hooked_modules} == set(refusal._directions)
