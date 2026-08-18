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
import logging
import sys
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


def test_post_reports_the_live_dial_rather_than_echoing_the_input(
    client, tmp_path, attn_cls, monkeypatch
):
    """Two concurrent POSTs must not each be told their own input won."""
    _install_for_endpoint(tmp_path, attn_cls)
    real_apply = refusal.apply_lambda

    def _apply_with_a_racing_writer(value):
        previous, _ = real_apply(value)
        real_apply(0.0)  # writer B lands before A formats its response
        return previous, refusal.status()

    def _set_with_a_racing_writer(value):
        real_apply(value)
        real_apply(0.0)
        return float(value)

    monkeypatch.setattr(refusal, "apply_lambda", _apply_with_a_racing_writer)
    monkeypatch.setattr(refusal, "set_lambda", _set_with_a_racing_writer)

    body = client.post("/admin/refusal_lambda", json={"lambda": 1.5}).json()

    assert refusal.get_lambda() == 0.0
    assert body["lambda"] == 0.0, "response echoed the caller's input, not the dial"


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


def test_hooked_modules_holds_a_strong_ref_per_hooked_module(tmp_path, attn_cls):
    """The id() keys are only safe while these strong refs exist."""
    path = _write_sidecar(tmp_path, ["layers.0", "layers.1", "mtp.0"])
    model = _tree(2, n_mtp=1, attn_factory=attn_cls)

    hooked = refusal.load_refusal_directions(model, path, attn_cls=attn_cls)
    assert hooked == 3
    assert len(refusal._hooked_modules) == hooked
    assert len(refusal._directions) == hooked
    assert {id(m) for m in refusal._hooked_modules} == set(refusal._directions)
