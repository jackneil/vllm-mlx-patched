# SPDX-License-Identifier: Apache-2.0
"""Runtime refusal-direction dial for DeepSeek-V4 (rank-1 projection).

Abliteration normally ships a second full checkpoint with the refusal
behaviour burned into the weights. This does the same thing with a 757 KB
sidecar of unit direction vectors and a runtime knob, because editing a
weight and projecting the sublayer output are the same function:

    (W - lambda*r r^T W)*x  ==  W*x - lambda*r*(r^T*W*x)

The left side needs a modified ``W``; the right side needs only ``r``, one
unit vector in R^4096 per edited module, applied to the attention sublayer's
output as it enters the residual stream. lambda becomes a dial: 0 is stock,
~1.5 removes refusal, negative values make the model *more* reticent.

Directions come from ``pocharlies/deepseek-v4-flash-0731-uncensored-
abliterated-refusal-directions`` (keys ``layers.N.attn.wo_b`` /
``mtp.N.attn.wo_b``, which map 1:1 onto this architecture's module paths).
Method: Arditi et al., *Refusal in Language Models Is Mediated by a Single
Direction* (NeurIPS 2024).

Implementation notes:

  * The hook patches ``V4Attention.__call__`` at the class level and looks
    each instance's direction up by ``id()``. Nothing is written into the
    module tree, so ``parameters()`` / ``load_weights`` see an unmodified
    model and the base weights stay byte-identical to the release. (Writing
    the direction onto the module as an attribute would be simpler but MLX
    registers array attributes as *parameters*, which would change
    ``parameters()`` and break exactly that guarantee.)
  * ``id()`` keys are only safe while the objects they name are alive, so
    ``_hooked_modules`` holds a strong reference to every hooked module for
    the life of the process. Without it a freed module's id could be
    recycled by an unrelated object and that object would silently start
    getting someone else's direction applied.
  * **lambda=0 short-circuits before any arithmetic**, so "off" is bit-exact
    and free. (The reference CUDA implementation must run the projection
    unconditionally to keep graph capture stable; MLX has no such
    constraint, so we get exactness and zero cost.)
  * ``apply_projection`` is the single implementation of the math. The hook
    calls it, so a test of ``apply_projection`` is a test of what is served.
  * The dial is **incompatible with ``--compile``** and refuses to install
    when the model forward pass is already wrapped by ``mx.compile``. See
    ``_find_compiled`` for the measurement that motivates the refusal.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# lambda is read on every forward pass and written by the admin route from a
# different thread, so guard it. Floats are atomic enough in CPython that the
# lock is really about making the read-modify-write in set_lambda coherent
# and giving callers a consistent snapshot.
_lock = threading.Lock()
_state: dict[str, Any] = {"lambda": 0.0, "installed": False, "modules": 0}

# id(attention_module) -> unit direction (mx.array, shape [hidden_size])
_directions: dict[int, mx.array] = {}

# (id(attention_module), out.dtype) -> the direction pre-cast to that dtype.
# The activation dtype is fixed for the life of a served model, so this cast
# happens once per module instead of once per forward pass. Bit-identical to
# casting inline: it is the same astype on the same source array. Cleared in
# lockstep with _directions (see _clear_directions) so a re-install can never
# serve a stale cast.
_cast_cache: dict[tuple[int, Any], mx.array] = {}

# Strong references to every hooked module, so the ids used as keys in
# _directions can never be recycled while this process lives.
_hooked_modules: list[Any] = []

_PATCH_FLAG = "_vllm_mlx_refusal_patched"
_ORIGINAL_CALL = "_vllm_mlx_refusal_original_call"

# The documented sidecar filename. Preferred by name when a directory or an HF
# snapshot holds more than one .safetensors file, because the DeepSeek-V4
# checkpoint itself carries tensors under the SAME key namespace the sidecar
# uses (layers.N.attn.wo_b): pointing --refusal-dirs at a model snapshot would
# otherwise load real attention biases as "directions" and corrupt the model at
# any lambda != 0, with a cheerful "installed on N modules" in the log.
_SIDECAR_BASENAME = "refusal_dirs.safetensors"

# One float32 unit vector per attention module: 61 backbone layers + 3 MTP
# blocks at hidden 4096 is ~757 KB for the published sidecar. Anything past
# this is not a direction sidecar; warn rather than reject, because a future
# architecture could legitimately be larger.
_IMPLAUSIBLE_SIDECAR_BYTES = 64 * 1024 * 1024


def _clear_directions() -> None:
    """Drop the direction map, its cast cache, and the module strong refs.

    Single entry point so the three structures can never drift apart.
    """
    _directions.clear()
    _cast_cache.clear()
    _hooked_modules.clear()


def get_lambda() -> float:
    with _lock:
        return float(_state["lambda"])


def set_lambda(value: float) -> float:
    """Set the dial. Takes effect on the next attention-module call, no reload.

    Not "the next request" and not "the next forward pass": the hook reads
    lambda once per module per token, so a change lands mid-generation and
    even mid-forward. See ``_install_hook`` for the measurement.
    """
    return apply_lambda(value)[1]["lambda"]


def apply_lambda(value: float) -> tuple[float, dict]:
    """Set the dial and read the resulting state back under ONE lock hold.

    Returns ``(previous_lambda, status)``. Callers that report what they set
    must use this rather than ``set_lambda`` + ``status``: with two concurrent
    writers, three separate lock acquisitions let a caller be told its own
    input while a different value is live.
    """
    with _lock:
        previous = float(_state["lambda"])
        _state["lambda"] = float(value)
        return previous, {
            "lambda": float(_state["lambda"]),
            "installed": bool(_state["installed"]),
            "modules": int(_state["modules"]),
        }


def status() -> dict:
    with _lock:
        return {
            "lambda": float(_state["lambda"]),
            "installed": bool(_state["installed"]),
            "modules": int(_state["modules"]),
        }


def apply_projection(out: mx.array, direction: mx.array, lam: float) -> mx.array:
    """Apply ``(I - lambda*r r^T)`` to the last axis of ``out``.

    ``direction`` must already be a unit vector (``load_refusal_directions``
    normalizes on load). This is the only place the math lives: the served
    hook calls straight into it.

    Exactness: **the bit-exactness guarantee is scoped to lambda=0**, and it is
    delivered by the short-circuit below (measured at 0.005-0.009 ms to skip
    all 61 layers), not by the form of the arithmetic. At lambda != 0 there is
    no bit-exactness contract to keep — the model is being edited on purpose —
    so the coefficient is computed as an ``out @ r`` matvec rather than a
    ``sum(out * r)`` reduction. Measured, 61-layer chain, bf16, one ``mx.eval``:

        B=1  T=   1   reduction  18.0ms   matvec  17.0ms   (1.06x)
        B=32 T=   1   reduction  25.1ms   matvec  19.1ms   (1.32x)
        B=1  T=2048   reduction 116.6ms   matvec  51.3ms   (2.27x)
        B=1  T=8192   reduction 346.1ms   matvec  76.1ms   (4.55x)
        peak memory, one layer at T=16384:  402.7MB -> 268.5MB

    The speedup magnitude varies with concurrent GPU load — a re-run on a
    busy box measured 1.1x-1.6x on the same shapes. The direction and the
    peak-memory reduction are stable, and the memory half is pinned by a
    test (100.7MB -> 67.1MB on the test's shapes).

    and against an fp32 reference computed from the SAME rounded inputs the two
    forms are equidistant at the dtypes actually served (bf16 1.561e-02 both,
    fp16 1.876e-03 both; fp32 0.0 vs 2.384e-07, i.e. 1 ulp) — the bf16 "delta"
    between them is rounding noise both forms carry, not accuracy one of them
    has and the other lacks.

    The one-cast-per-forward cost is removed by handing this function an
    already-cast direction (see ``_direction_for``); the ``astype`` below is
    then a same-dtype no-op and is kept so that direct callers (tests, tools)
    still get a correct answer.
    """
    if lam == 0.0:
        return out
    r = direction.astype(out.dtype)
    return out - lam * mx.expand_dims(out @ r, -1) * r


def _unwrap_model(model: Any) -> Any:
    """Descend the ``.model`` chain to the object that owns the block list.

    The serve path hands us different depths depending on the engine:
    SimpleEngine holds an ``MLXLanguageModel`` wrapper whose ``.model`` is the
    mlx_lm ``Model``, whose ``.model`` is the ``DeepseekV4Model`` that actually
    owns ``layers``; BatchedEngine holds the mlx_lm ``Model`` directly. Prefer
    the *deepest* object with a usable ``layers`` list, because that one holds
    the full, unsharded block list whose indices match the sidecar keys (the
    mlx_lm ``Model.layers`` property returns a pipeline-sliced view).
    """
    best = model
    found = False
    current = model
    for _ in range(4):
        layers = getattr(current, "layers", None) or []
        if len(layers) and getattr(layers[0], "attn", None) is not None:
            best = current
            found = True
        nxt = getattr(current, "model", None)
        if nxt is None or nxt is current:
            break
        current = nxt
    return best if found else model


def _find_compiled(model: Any) -> Any | None:
    """Return the first object in the ``.model`` chain wrapped by ``mx.compile``.

    ``vllm_mlx.compile.apply_compile`` replaces an object's ``__call__`` with
    ``mx.compile(..., shapeless=True)`` and stamps ``_vllm_mlx_compiled`` on it.
    It is applied at different depths depending on the engine — BatchedEngine
    compiles ``engine._model`` itself, SimpleEngine compiles
    ``engine._model.model`` — and ``_install_refusal_directions`` always hands
    us ``engine._model``, so the whole chain has to be checked.

    Why this matters: ``mx.compile`` traces the wrapped function once and
    reuses the trace. The hook's ``lam = _state["lambda"]`` is a plain Python
    read, so it is baked in as a trace-time CONSTANT and never re-read.
    Measured on a synthetic 1-layer model, ``r = [.5,.5,.5,.5]``, out = ones:

        EAGER    lam=0    -> [ 1.0,  1.0,  1.0,  1.0]
        EAGER    lam=1.5  -> [-0.5, -0.5, -0.5, -0.5]   dial works
        COMPILED trace@0  -> [ 1.0,  1.0,  1.0,  1.0]
        COMPILED lam=1.5  -> [ 1.0,  1.0,  1.0,  1.0]   dial FROZEN

    while ``status()`` cheerfully reported ``{'lambda': 1.5,
    'installed': True}``. ``--compile`` and ``--refusal-dirs`` are independent
    serve flags, so that pairing is one checkbox away.

    The refusal is deliberately kept even though ``--compile`` cannot bite
    today: ``apply_compile`` assigns ``model.__call__`` as an *instance*
    attribute and Python resolves ``model(x)`` against ``type(model).__call__``,
    so with every serving call site using the implicit form the compiled trace
    never runs (the demonstration lives in
    ``test_apply_compile_is_bypassed_by_implicit_calls``). The day compile.py
    is fixed to patch the class, this guard is what stops a frozen dial from
    shipping with it — so this is future-proofing, not dead code.
    """
    try:
        from ..compile import is_compiled
    except Exception:  # pragma: no cover - compile module is always importable
        logger.exception("[refusal] could not import the compile probe")
        return None

    current = model
    for _ in range(5):
        if is_compiled(current):
            return current
        nxt = getattr(current, "model", None)
        if nxt is None or nxt is current:
            return None
        current = nxt
    return None


def _hidden_size(model: Any) -> int | None:
    """Best-effort hidden size from the model's config, or None if unreachable.

    Used only to reject a sidecar whose vectors are the wrong width; every
    failure mode degrades to "skip the check", never to a false rejection.
    """
    current = model
    for _ in range(5):
        for holder in ("args", "config"):
            cfg = getattr(current, holder, None)
            size = getattr(cfg, "hidden_size", None) if cfg is not None else None
            if isinstance(size, int) and size > 0:
                return size
        nxt = getattr(current, "model", None)
        if nxt is None or nxt is current:
            return None
        current = nxt
    return None


def _iter_attention_modules(model: Any):
    """Yield (sidecar_key_prefix, attention_module) for every hookable module.

    Backbone layers are ``layers.N``; multi-token-prediction blocks are
    ``mtp.N``. A checkpoint may carry fewer MTP blocks than the sidecar has
    directions (or none), so this yields only what the model actually has.
    """
    inner = _unwrap_model(model)

    for i, layer in enumerate(getattr(inner, "layers", []) or []):
        attn = getattr(layer, "attn", None)
        if attn is not None:
            yield f"layers.{i}", attn

    # The mtp branch is dead on today's mlx_lm and kept deliberately.
    # mlx-lm-v4's DeepseekV4 sanitize() drops every `mtp.` weight at load
    # ("1) Drop MTP + any layers beyond n_layers" -> `if k.startswith("mtp."):
    # continue`, mlx_lm/models/deepseek_v4.py), and its docstring lists
    # `mtp.0.* (dropped)`. So the checkpoint's MTP blocks do not exist as
    # modules at inference and the sidecar's mtp.0/1/2.attn.wo_b directions are
    # STRUCTURALLY unmatchable here — that is the expected orphan count in the
    # install log, not a mapping bug. Kept so a future mlx_lm that does keep
    # MTP blocks gets them hooked without a code change.
    mtp = getattr(inner, "mtp", None)
    if mtp is None:
        mtp = getattr(model, "mtp", None)
    if mtp is not None:
        blocks = getattr(mtp, "layers", None)
        if blocks is None:
            blocks = mtp if isinstance(mtp, (list, tuple)) else []
        for i, block in enumerate(blocks):
            attn = getattr(block, "attn", None)
            if attn is not None:
                yield f"mtp.{i}", attn


def load_refusal_directions(model: Any, path: str | Path, attn_cls: Any = None) -> int:
    """Load direction vectors and install the projection hook.

    Args:
        model: A loaded deepseek_v4 model (or any wrapper around one).
        path: ``refusal_dirs.safetensors``, or a directory / HF repo id
            containing it.
        attn_cls: Attention class to patch. Defaults to
            ``mlx_lm.models.deepseek_v4.V4Attention``; injectable for tests.

    Returns:
        Number of modules wired up. 0 means nothing was installed and the
        model is untouched — callers should treat that as "serving stock".
    """
    # Always start from a clean slate: a re-install replaces the previous
    # mapping rather than accumulating stale entries from an unloaded model.
    _clear_directions()
    with _lock:
        _state["installed"] = False
        _state["modules"] = 0

    try:
        # A dial that cannot move must never report itself installed: under
        # mx.compile the lambda read is frozen into the trace, so the hook
        # would serve one fixed value while status() and the admin route
        # reported another. Refuse, so status() stays false and POST
        # /admin/refusal_lambda 409s instead of 200-ing a no-op.
        compiled = _find_compiled(model)
        if compiled is not None:
            logger.error(
                "[refusal] refusing to install: this model's forward pass is "
                "already wrapped by mx.compile (--compile), and "
                "mx.compile(shapeless=True) captures the lambda read as a "
                "trace-time constant — the dial would be FROZEN at %.3f "
                "forever while /admin/refusal_lambda kept reporting the value "
                "you set. --compile and --refusal-dirs are mutually "
                "exclusive: restart with one or the other. Dropping --compile "
                "costs you nothing today, because --compile is currently "
                "INERT on the serving path anyway: apply_compile assigns the "
                "compiled function to model.__call__ as an INSTANCE "
                "attribute, while Python resolves model(x) against "
                "type(model).__call__, and every serving call site is the "
                "implicit model(...) form — so the compiled trace is never "
                "the thing that runs. Serving stock model.",
                get_lambda(),
            )
            return 0

        f = _resolve_path(path)
        if f is None:
            logger.warning("[refusal] no %s found at %s", _SIDECAR_BASENAME, path)
            return 0

        resolved = f.resolve()
        size = resolved.stat().st_size
        logger.info(
            "[refusal] loading directions from %s (%.1f KB)", resolved, size / 1024
        )
        if size > _IMPLAUSIBLE_SIDECAR_BYTES:
            logger.warning(
                "[refusal] %s is %.1f MB, far larger than a direction sidecar "
                "(the published one is ~757 KB). If this is a model snapshot "
                "rather than a sidecar, its own layers.N.attn.wo_b tensors "
                "will be normalized into bogus 'directions' and will corrupt "
                "generation at any lambda != 0.",
                resolved,
                size / 1e6,
            )

        raw = mx.load(str(f))
        hidden = _hidden_size(model)
        hooked = 0
        modules_seen = 0
        missing: list[str] = []
        matched: set[str] = set()
        for key, attn in _iter_attention_modules(model):
            modules_seen += 1
            sidecar_key = f"{key}.attn.wo_b"
            vec = raw.get(sidecar_key)
            if vec is None:
                missing.append(key)
                continue
            if not _is_usable_direction(sidecar_key, vec, hidden):
                # Rejected tensors stay unmatched on purpose: they are counted
                # and named as orphans below rather than silently hooked.
                missing.append(key)
                continue
            norm = mx.linalg.norm(vec).item()
            # `norm == 0` is not the only unusable norm. A finite-but-huge
            # vector — a truncated or corrupt sidecar whose bytes reinterpret
            # as ~1e38 floats — passes _is_usable_direction's isfinite check
            # and then OVERFLOWS here: norm is inf, `vec / inf` is all zeros,
            # and an all-zero direction projects nothing at every lambda.
            # Measured on [3e38]*4 float32: isfinite(vec).all() True,
            # norm inf, norm == 0 False, normalized [0, 0, 0, 0], projection
            # at lambda=1.5 unchanged. Without this branch the module is
            # counted in `hooked`, logged as installed and reported by
            # status(), while the dial does nothing for that layer forever —
            # the exact "reports success while doing nothing" hole the rest of
            # this loader exists to close. Reject unless the norm is finite
            # AND strictly positive, and count it as an orphan like every
            # other reject.
            if not math.isfinite(norm) or norm <= 0.0:
                logger.warning(
                    "[refusal] %s has %s norm (%r) — no usable direction to "
                    "project on, skipping",
                    sidecar_key,
                    "zero" if norm == 0.0 else "non-finite",
                    norm,
                )
                missing.append(key)
                continue
            # Normalize defensively: the projection identity assumes a UNIT
            # direction, and a non-unit vector would silently rescale lambda.
            _directions[id(attn)] = (vec / norm).astype(mx.float32)
            _hooked_modules.append(attn)
            matched.add(sidecar_key)
            hooked += 1

        if not hooked:
            logger.warning(
                "[refusal] sidecar has %d tensors but none matched this "
                "model's %d attention modules — hook NOT installed",
                len(raw),
                modules_seen,
            )
            _clear_directions()
            return 0

        _install_hook(attn_cls if attn_cls is not None else _resolve_attention_class())
        with _lock:
            _state["installed"] = True
            _state["modules"] = hooked

        orphans = [k for k in sorted(raw) if k not in matched]
        # A partial mapping is the dangerous case (e.g. a pipeline-sliced view
        # exposing 20 of 61 layers): it installs, serves, and looks fine. Only
        # a full 1:1 mapping is routine enough for INFO.
        level = logging.INFO if hooked == len(raw) else logging.WARNING
        logger.log(
            level,
            "[refusal] rank-1 refusal projection installed: %d model attention "
            "modules found, %d sidecar directions loaded, %d modules hooked "
            "(lambda=%.3f) from %s",
            modules_seen,
            len(raw),
            hooked,
            get_lambda(),
            resolved,
        )
        if orphans:
            logger.log(
                level,
                "[refusal] %d sidecar direction(s) matched no model module: %s "
                "— on mlx_lm's deepseek_v4 the 3 mtp.N.attn.wo_b keys are "
                "EXPECTED orphans (it drops all mtp.* weights at load); "
                "anything else here means the sidecar does not match this "
                "checkpoint.",
                len(orphans),
                orphans,
            )
        if missing:
            logger.log(
                level,
                "[refusal] %d model module(s) got no usable direction: %s",
                len(missing),
                missing[:8],
            )
        return hooked

    except Exception:
        logger.exception("[refusal] failed to install — serving stock model")
        _clear_directions()
        with _lock:
            _state["installed"] = False
            _state["modules"] = 0
        return 0


def _is_usable_direction(key: str, vec: mx.array, hidden: int | None) -> bool:
    """Reject sidecar tensors that would corrupt or poison the residual stream.

    Shape: a ``[1, D]`` sidecar broadcasts cleanly through ``apply_projection``
    and silently changes the math; ``[D, 1]`` corrupts the output outright.
    Only a 1-D ``[hidden_size]`` vector is a direction.

    Finiteness: ``mx.linalg.norm`` of a NaN vector is NaN, and ``nan == 0`` is
    False, so the zero-norm guard alone lets it through; ``vec / nan`` is an
    all-NaN "direction" that serves perfectly at lambda=0 and turns the whole
    residual stream into NaN the moment somebody dials it up, arbitrarily far
    in time from the install that caused it.

    Finite COMPONENTS are not enough on their own: a vector of ~1e38 floats
    passes this check and then overflows to an infinite norm in the caller,
    which normalizes to an all-zero direction. That case is rejected by the
    norm guard in ``load_refusal_directions``, which is the reason that guard
    tests ``math.isfinite(norm)`` and not just ``norm == 0``.
    """
    if vec.ndim != 1:
        logger.warning(
            "[refusal] %s has shape %s; a direction must be 1-D [hidden_size] "
            "— skipping",
            key,
            tuple(vec.shape),
        )
        return False
    if hidden is not None and vec.shape[0] != hidden:
        logger.warning(
            "[refusal] %s has length %d but this model's hidden size is %d "
            "— skipping",
            key,
            vec.shape[0],
            hidden,
        )
        return False
    if not bool(mx.isfinite(vec).all().item()):
        logger.warning(
            "[refusal] %s contains non-finite values (nan/inf) — skipping; a "
            "NaN direction would poison the residual stream at any lambda != 0",
            key,
        )
        return False
    return True


def _pick_sidecar(d: Path) -> Path | None:
    """Choose the sidecar inside a directory, preferring the documented name.

    ``refusal_dirs.safetensors`` is what both the CLI help and this module's
    docstring name, so it wins outright. Falling back to an unsorted
    ``glob()[0]`` made the same config load different bytes on different
    machines (and after a re-download), so the fallback is sorted and loud.
    """
    named = d / _SIDECAR_BASENAME
    if named.is_file():
        return named
    hits = sorted(d.glob("*.safetensors"))
    if not hits:
        return None
    if len(hits) > 1:
        logger.warning(
            "[refusal] %s not found in %s; %d .safetensors candidates present, "
            "using the lexicographically first (%s). Point --refusal-dirs at "
            "the sidecar file itself to remove the ambiguity.",
            _SIDECAR_BASENAME,
            d,
            len(hits),
            hits[0].name,
        )
    else:
        logger.warning(
            "[refusal] %s not found in %s; using the only .safetensors there "
            "(%s), which may not be a direction sidecar.",
            _SIDECAR_BASENAME,
            d,
            hits[0].name,
        )
    return hits[0]


def _resolve_path(path: str | Path) -> Path | None:
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        picked = _pick_sidecar(p)
        if picked is None:
            logger.warning(
                "[refusal] %s is a directory but contains no .safetensors file",
                p,
            )
        return picked
    try:
        from huggingface_hub import snapshot_download

        # local_files_only: serving never downloads; the arena's hf_catalog
        # is the only sanctioned path for pulling weights.
        d = Path(snapshot_download(str(path), local_files_only=True))
    except Exception:
        # Do not collapse "no such path", "not in the HF cache", "hub not
        # installed" and "repo id typo" into one indistinguishable None — an
        # operator has to be able to tell `hf download` from `fix the path`
        # without reading this source.
        logger.exception(
            "[refusal] %r is neither an existing file nor a directory, and "
            "could not be resolved from the local HF cache (downloads are "
            "disabled here). Pre-download it, or pass a local path.",
            str(path),
        )
        return None
    picked = _pick_sidecar(d)
    if picked is None:
        logger.warning("[refusal] HF snapshot %s contains no .safetensors file", d)
    return picked


def _resolve_attention_class() -> Any:
    """Import the real attention class. Kept separate so tests can avoid it."""
    from mlx_lm.models.deepseek_v4 import V4Attention

    return V4Attention


def _direction_for(module_id: int, dtype: Any) -> mx.array | None:
    """The module's direction, cast to ``dtype`` once instead of per forward."""
    key = (module_id, dtype)
    cached = _cast_cache.get(key)
    if cached is not None:
        return cached
    base = _directions.get(module_id)
    if base is None:
        return None
    cached = base.astype(dtype)
    _cast_cache[key] = cached
    return cached


def _install_hook(attn_cls: Any) -> None:
    """Wrap ``attn_cls.__call__`` with the projection, once per class."""
    if getattr(attn_cls, _PATCH_FLAG, False):
        return

    original = attn_cls.__call__

    def _call_with_refusal(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        # Deliberately UNLOCKED, unlike get_lambda(): this runs once per
        # attention module per token, and a single dict read is atomic under
        # the GIL. Be precise about what that costs. The read is per MODULE,
        # not per forward pass, so a dial change that lands mid-pass produces
        # a HYBRID forward: the modules already run used the old lambda, the
        # rest use the new one, and the pass as a whole corresponds to no
        # configured lambda. Measured on a 61-layer chain flipped at layer 30:
        #
        #     layer 0..3         -> [1.0, 1.0, 1.0, 1.0]     (lambda = 0)
        #     layer 29,30,31,60  -> [1.0, -0.5, 0.25, -0.0]  (lambda = 1.5)
        #
        # i.e. one forward pass mixed lambda=0 and lambda=1.5 across layers.
        # A per-forward snapshot would need a model-level hook, which is a
        # design change, not a comment fix — until then this IS the semantics,
        # and POST /admin/refusal_lambda documents it as landing mid-request.
        # Do not add a lock here; do not remove the lock from
        # get_lambda()/apply_lambda(), which need it to give callers a
        # coherent multi-field snapshot.
        lam = _state["lambda"]
        if lam == 0.0:
            # Bit-exact stock: no arithmetic, no dict lookup.
            return out
        r = _direction_for(id(self), out.dtype)
        if r is None:
            return out
        return apply_projection(out, r, lam)

    attn_cls.__call__ = _call_with_refusal
    setattr(attn_cls, _ORIGINAL_CALL, original)
    setattr(attn_cls, _PATCH_FLAG, True)


def uninstall_hook(attn_cls: Any) -> None:
    """Restore ``attn_cls.__call__``. Only used by tests and ``reset``."""
    if not getattr(attn_cls, _PATCH_FLAG, False):
        return
    attn_cls.__call__ = getattr(attn_cls, _ORIGINAL_CALL)
    try:
        delattr(attn_cls, _ORIGINAL_CALL)
        delattr(attn_cls, _PATCH_FLAG)
    except AttributeError:  # pragma: no cover - inherited flag, nothing to drop
        pass


def reset(attn_cls: Any = None) -> None:
    """Drop all process-global state (and unpatch ``attn_cls`` if given).

    The server never needs this; it exists so tests that touch the globals
    can leave the module exactly as they found it.
    """
    if attn_cls is not None:
        uninstall_hook(attn_cls)
    _clear_directions()
    with _lock:
        _state["lambda"] = 0.0
        _state["installed"] = False
        _state["modules"] = 0
