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
import sys
from unittest.mock import patch

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

from vllm_mlx.patches import deepseek_v4_refusal as refusal

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
