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
"""

from __future__ import annotations

import logging
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

# Strong references to every hooked module, so the ids used as keys in
# _directions can never be recycled while this process lives.
_hooked_modules: list[Any] = []

_PATCH_FLAG = "_vllm_mlx_refusal_patched"
_ORIGINAL_CALL = "_vllm_mlx_refusal_original_call"


def get_lambda() -> float:
    with _lock:
        return float(_state["lambda"])


def set_lambda(value: float) -> float:
    """Set the dial. Takes effect on the next forward pass, no reload."""
    with _lock:
        _state["lambda"] = float(value)
        return float(_state["lambda"])


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
    """
    if lam == 0.0:
        return out
    r = direction.astype(out.dtype)
    return out - lam * mx.sum(out * r, axis=-1, keepdims=True) * r


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
    _directions.clear()
    _hooked_modules.clear()
    with _lock:
        _state["installed"] = False
        _state["modules"] = 0

    try:
        f = _resolve_path(path)
        if f is None:
            logger.warning("[refusal] no refusal_dirs.safetensors found at %s", path)
            return 0

        raw = mx.load(str(f))
        hooked = 0
        missing = []
        for key, attn in _iter_attention_modules(model):
            vec = raw.get(f"{key}.attn.wo_b")
            if vec is None:
                missing.append(key)
                continue
            # Normalize defensively: the projection identity assumes a UNIT
            # direction, and a non-unit vector would silently rescale lambda.
            norm = mx.linalg.norm(vec).item()
            if norm == 0:
                missing.append(key)
                continue
            _directions[id(attn)] = (vec / norm).astype(mx.float32)
            _hooked_modules.append(attn)
            hooked += 1

        if not hooked:
            logger.warning(
                "[refusal] sidecar has %d tensors but none matched this "
                "model's modules — hook NOT installed",
                len(raw),
            )
            _directions.clear()
            _hooked_modules.clear()
            return 0

        _install_hook(attn_cls if attn_cls is not None else _resolve_attention_class())
        with _lock:
            _state["installed"] = True
            _state["modules"] = hooked
        logger.info(
            "[refusal] rank-1 refusal projection installed on %d modules "
            "(lambda=%.3f, %d sidecar tensors unmatched)",
            hooked,
            get_lambda(),
            len(raw) - hooked,
        )
        if missing:
            logger.info("[refusal] modules without a direction: %s", missing[:8])
        return hooked

    except Exception:
        logger.exception("[refusal] failed to install — serving stock model")
        _directions.clear()
        _hooked_modules.clear()
        with _lock:
            _state["installed"] = False
            _state["modules"] = 0
        return 0


def _resolve_path(path: str | Path) -> Path | None:
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        hits = list(p.glob("*.safetensors"))
        return hits[0] if hits else None
    try:
        from huggingface_hub import snapshot_download

        # local_files_only: serving never downloads; the arena's hf_catalog
        # is the only sanctioned path for pulling weights.
        d = Path(snapshot_download(str(path), local_files_only=True))
        hits = list(d.glob("*.safetensors"))
        return hits[0] if hits else None
    except Exception:
        return None


def _resolve_attention_class() -> Any:
    """Import the real attention class. Kept separate so tests can avoid it."""
    from mlx_lm.models.deepseek_v4 import V4Attention

    return V4Attention


def _install_hook(attn_cls: Any) -> None:
    """Wrap ``attn_cls.__call__`` with the projection, once per class."""
    if getattr(attn_cls, _PATCH_FLAG, False):
        return

    original = attn_cls.__call__

    def _call_with_refusal(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        lam = _state["lambda"]
        if lam == 0.0:
            # Bit-exact stock: no arithmetic, no dict lookup.
            return out
        r = _directions.get(id(self))
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
    _directions.clear()
    _hooked_modules.clear()
    with _lock:
        _state["lambda"] = 0.0
        _state["installed"] = False
        _state["modules"] = 0
