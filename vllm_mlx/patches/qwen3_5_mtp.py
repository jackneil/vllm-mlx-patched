# SPDX-License-Identifier: Apache-2.0
"""Runtime MTP (Multi-Token Prediction) support for Qwen3.5 / Qwen3.8 models.

Qwen3.5-family checkpoints declare ``mtp_num_hidden_layers`` in their
``text_config``, but mlx-community ships the MTP head as a SEPARATE repo
(e.g. ``mlx-community/Qwen3.8-27B-MTP-4bit``, 253 MB) rather than inside
the base model's safetensors — the base 4-bit weights contain zero
``mtp.*`` keys. Upstream mlx_lm has no MTP head for this architecture
either (ml-explore/mlx-lm#990 is still open), so ``TextModel`` exposes
neither ``mtp_forward`` nor ``return_hidden``.

This module closes both gaps at runtime, without patching mlx_lm:

  - builds an ``MTPModule`` matching the checkpoint's weight layout
    (``fc``, ``layers.0.*``, ``norm``, ``pre_fc_norm_{hidden,embedding}``),
  - quantizes it to match the tensors actually present in the drafter
    repo (a layer is quantized iff the checkpoint carries ``.scales``
    for it),
  - loads the drafter weights,
  - patches the model CLASS with ``return_hidden`` support in
    ``__call__`` plus ``mtp_forward`` / ``make_mtp_cache``.

Architecture mirrors ml-explore/mlx-lm#990 (the Qwen3.5 reference
implementation). Two details are load-bearing and easy to get backwards:

  1. The fused projection input order is ``concat([embedding, hidden])``
     — NOT ``[hidden, embedding]``. The weight matrix is not symmetric,
     so swapping the halves yields syntactically valid but semantically
     garbage drafts (acceptance collapses to ~0% while nothing crashes).
  2. The MTP head consumes the backbone's PRE-norm hidden state, so the
     patched ``__call__`` returns ``hidden`` from before ``model.norm``,
     while logits are still computed from the normed activations.

The decode-loop scheduling that uses these hooks lives in
``vllm_mlx/scheduler.py`` (``_install_mtp_decode``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# Default sidecar repo naming: the MTP head published alongside a base
# model as "<base>-MTP-<quant>". Resolved only as a fallback when the
# caller does not pass an explicit drafter path.
_MTP_REPO_SUFFIXES = ("-MTP-4bit", "-MTP-8bit", "-MTP-bf16")


def _resolve_drafter_path(model_name: str, drafter: str | None) -> Path | None:
    """Locate the MTP sidecar snapshot directory in the HF cache.

    Args:
        model_name: Base model id (e.g. ``mlx-community/Qwen3.8-27B-4bit``)
            or a local path.
        drafter: Explicit sidecar repo id / path. When None, the
            conventional ``<base>-MTP-<quant>`` siblings are tried.

    Returns:
        Path to a snapshot directory containing the MTP weights, or None.
    """
    candidates: list[str] = []
    if drafter:
        candidates.append(drafter)
    else:
        base = model_name.rsplit("-", 1)[0] if "-" in model_name else model_name
        candidates.extend(f"{base}{sfx}" for sfx in _MTP_REPO_SUFFIXES)

    from huggingface_hub import snapshot_download

    for cand in candidates:
        p = Path(cand)
        if p.is_dir():
            return p
        try:
            # local_files_only: never trigger a download from the serving
            # path. All weight pulls go through the arena's hf_catalog.
            return Path(snapshot_download(cand, local_files_only=True))
        except Exception:
            continue
    return None


def _load_drafter_weights(path: Path) -> dict[str, mx.array]:
    """Load every safetensors shard in the sidecar, stripping an optional
    ``mtp.`` prefix so keys match the MTPModule parameter paths."""
    weights: dict[str, mx.array] = {}
    for shard in sorted(path.glob("*.safetensors")):
        for k, v in mx.load(str(shard)).items():
            weights[k.removeprefix("mtp.")] = v
    return weights


def _build_mtp_module(args: Any, n_layers: int, weights: dict[str, mx.array]):
    """Construct + quantize an MTPModule matching the drafter checkpoint."""
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.qwen3_next import Qwen3NextAttention as Attention
    from mlx_lm.models.qwen3_next import Qwen3NextMLP as MLP
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock as SparseMoeBlock

    class _MTPDecoderLayer(nn.Module):
        """Full-attention transformer layer for the MTP head.

        Unlike the backbone's DecoderLayer this is never a GatedDeltaNet
        layer — the MTP head is attention-only regardless of where it
        would fall in the backbone's linear/full attention interleave.
        """

        def __init__(self, args):
            super().__init__()
            self.self_attn = Attention(args)
            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            if args.num_experts > 0:
                self.mlp = SparseMoeBlock(args)
            else:
                self.mlp = MLP(args.hidden_size, args.intermediate_size)

        def __call__(self, x, mask=None, cache=None):
            h = x + self.self_attn(self.input_layernorm(x), mask, cache)
            return h + self.mlp(self.post_attention_layernorm(h))

    class _MTPModule(nn.Module):
        def __init__(self, args, n_layers):
            super().__init__()
            self.pre_fc_norm_hidden = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.pre_fc_norm_embedding = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            self.layers = [_MTPDecoderLayer(args) for _ in range(n_layers)]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        def __call__(self, hidden_states, next_token_ids, embed_tokens, cache=None):
            # Fusion order is [embedding, hidden] — see module docstring.
            e = self.pre_fc_norm_embedding(embed_tokens(next_token_ids))
            h = self.pre_fc_norm_hidden(hidden_states)
            fused = self.fc(mx.concatenate([e, h], axis=-1))
            if cache is None:
                cache = [None] * len(self.layers)
            mask = create_attention_mask(fused, cache[0])
            for layer, c in zip(self.layers, cache):
                fused = layer(fused, mask, c)
            return self.norm(fused)

    mtp = _MTPModule(args, n_layers)

    # Quantize exactly the layers the checkpoint quantized. Norms and any
    # BF16-kept projection carry no `.scales`, so they stay full precision.
    quant = getattr(args, "quantization", None) or {}
    bits = quant.get("bits", 4)
    group_size = quant.get("group_size", 64)

    def _predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        return f"{path}.scales" in weights

    nn.quantize(mtp, group_size=group_size, bits=bits, class_predicate=_predicate)
    return mtp


def inject_mtp_support(
    model: Any,
    model_name: str,
    config: dict,
    drafter: str | None = None,
) -> bool:
    """Attach an MTP head to a loaded Qwen3.5/3.8 model and patch its class.

    Args:
        model: Model loaded via mlx_lm (``Model`` with ``.language_model``).
        model_name: Base model id or path (used to locate the sidecar).
        config: Parsed ``config.json`` of the base model.
        drafter: Explicit MTP sidecar repo id / path. When None, the
            conventional ``<base>-MTP-*`` sibling repos are tried.

    Returns:
        True when MTP was injected and is ready to use, False otherwise.
        Never raises — a failure here must degrade to normal decoding.
    """
    try:
        text_config = config.get("text_config", config)
        n_layers = text_config.get("mtp_num_hidden_layers", 0)
        if n_layers <= 0:
            logger.info("[MTP inject] mtp_num_hidden_layers=0 — nothing to inject")
            return False

        lm = getattr(model, "language_model", None)
        if lm is None:
            logger.warning("[MTP inject] model has no .language_model — unsupported")
            return False

        path = _resolve_drafter_path(model_name, drafter)
        if path is None:
            logger.warning(
                "[MTP inject] MTP sidecar not found in the HF cache for %s "
                "(drafter=%s). Pre-download it via the arena Catalog.",
                model_name,
                drafter,
            )
            return False

        weights = _load_drafter_weights(path)
        if not weights:
            logger.warning("[MTP inject] no safetensors weights in %s", path)
            return False

        args = lm.args
        # The sidecar is quantized independently of the base model; read its
        # own config so a 4-bit head on an 8-bit base still loads correctly.
        sidecar_cfg_file = path / "config.json"
        if sidecar_cfg_file.exists():
            sidecar_cfg = json.loads(sidecar_cfg_file.read_text())
            quant = sidecar_cfg.get("quantization") or config.get("quantization")
        else:
            quant = config.get("quantization")
        args.quantization = quant

        mtp = _build_mtp_module(args, n_layers, weights)
        mtp.load_weights(list(weights.items()), strict=False)
        mx.eval(mtp.parameters())

        loaded = {k for k, _ in weights.items()}
        logger.info(
            "[MTP inject] loaded %d tensors from %s (%d layer(s))",
            len(loaded),
            path.name,
            n_layers,
        )

        lm.mtp = mtp
        _patch_model_class(model)
        logger.info("[MTP inject] model class patched: return_hidden + mtp_forward")
        return True

    except Exception:
        logger.exception("[MTP inject] failed — decoding will run without MTP")
        return False


def _gated_delta_split_forward(mod, inputs, cache, n_confirmed):
    """GatedDeltaNet forward that snapshots recurrent state mid-sequence.

    Speculative decoding feeds ``[confirmed_token, draft_token]`` in one
    pass. Attention layers can undo a rejected draft with ``KVCache.trim``,
    but a GatedDeltaNet's conv/SSM state is recurrent — once the draft is
    folded in there is no way to subtract it, and the pollution compounds
    into visibly corrupted output.

    Processing the sequence as two chunks lets us stash the exact state
    after the confirmed prefix (``cache.rollback_state``) before the draft
    is applied, so rejection is an exact restore. Mirrors the ``n_confirmed``
    split in ml-explore/mlx-lm#990.

    Only called on the decode-time verify step, where there is no padding
    and no mask; every other path uses the module's stock forward.
    """
    from mlx_lm.models.gated_delta import gated_delta_update

    B, S, _ = inputs.shape
    qkv = mod.in_proj_qkv(inputs)
    z = mod.in_proj_z(inputs).reshape(B, S, mod.num_v_heads, mod.head_v_dim)
    b = mod.in_proj_b(inputs)
    a = mod.in_proj_a(inputs)

    conv_state = cache[0]
    if conv_state is None:
        conv_state = mx.zeros(
            (B, mod.conv_kernel_size - 1, mod.conv_dim), dtype=inputs.dtype
        )
    ssm_state = cache[1]
    n_keep = mod.conv_kernel_size - 1

    def _chunk(lo, hi, conv_in_state, ssm_in_state):
        conv_in = mx.concatenate([conv_in_state, qkv[:, lo:hi]], axis=1)
        new_conv = mx.contiguous(conv_in[:, -n_keep:, :])
        conv_out = nn.silu(mod.conv1d(conv_in))
        q, k, v = [
            t.reshape(B, hi - lo, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [mod.key_dim, 2 * mod.key_dim], -1),
                [mod.num_k_heads, mod.num_k_heads, mod.num_v_heads],
                [mod.head_k_dim, mod.head_k_dim, mod.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        out, new_ssm = gated_delta_update(
            q,
            k,
            v,
            a[:, lo:hi],
            b[:, lo:hi],
            mod.A_log,
            mod.dt_bias,
            ssm_in_state,
            None,
            use_kernel=not mod.training,
        )
        return out, new_conv, new_ssm

    out_c, conv_c, ssm_c = _chunk(0, n_confirmed, conv_state, ssm_state)
    # MLX arrays are immutable, so retaining these is a reference snapshot,
    # not a copy — rollback costs nothing per step.
    cache.rollback_state = (conv_c, ssm_c)
    out_d, conv_f, ssm_f = _chunk(n_confirmed, S, conv_c, ssm_c)

    cache[0] = conv_f
    cache[1] = ssm_f
    cache.advance(S)

    out = mx.concatenate([out_c, out_d], axis=1)
    out = mod.norm(out, z)
    return mod.out_proj(out.reshape(B, S, -1))


def rollback_draft(model_cache) -> None:
    """Undo one speculative draft token across a hybrid model's caches.

    Recurrent layers restore the snapshot taken by
    ``_gated_delta_split_forward``; attention layers trim their last entry.
    """
    for c in model_cache:
        snap = getattr(c, "rollback_state", None)
        if snap is not None:
            c[0], c[1] = snap
            c.rollback_state = None
        elif hasattr(c, "is_trimmable") and c.is_trimmable():
            c.trim(1)


def clear_rollback(model_cache) -> None:
    """Drop draft snapshots after an accepted draft (frees the references)."""
    for c in model_cache:
        if getattr(c, "rollback_state", None) is not None:
            c.rollback_state = None


def _patch_model_class(model: Any) -> None:
    """Replace the model's class with a subclass exposing the MTP API.

    Patching the class (not the instance) is required because Python
    resolves ``model(...)`` through ``type(model).__call__``.
    """
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    from mlx_lm.models.cache import KVCache

    original = model.__class__
    if getattr(original, "_vllm_mlx_mtp_patched", False):
        return

    class _Qwen3_5WithMTP(original):
        _vllm_mlx_mtp_patched = True

        def __call__(
            self,
            inputs,
            cache=None,
            input_embeddings=None,
            return_hidden: bool = False,
            n_confirmed: int = 0,
        ):
            if not return_hidden:
                return super().__call__(
                    inputs, cache=cache, input_embeddings=input_embeddings
                )

            lm = self.language_model
            inner = lm.model
            if input_embeddings is not None:
                h = input_embeddings
            else:
                h = inner.embed_tokens(inputs)

            if cache is None:
                cache = [None] * len(inner.layers)

            fa_mask = create_attention_mask(h, cache[inner.fa_idx])
            ssm_mask = create_ssm_mask(h, cache[inner.ssm_idx])
            split = 0 < n_confirmed < h.shape[1]
            for layer, c in zip(inner.layers, cache):
                if layer.is_linear and split and c is not None:
                    r = _gated_delta_split_forward(
                        layer.linear_attn, layer.input_layernorm(h), c, n_confirmed
                    )
                else:
                    mask = ssm_mask if layer.is_linear else fa_mask
                    r = (
                        layer.linear_attn(layer.input_layernorm(h), mask, c)
                        if layer.is_linear
                        else layer.self_attn(layer.input_layernorm(h), mask, c)
                    )
                hh = h + r
                h = hh + layer.mlp(layer.post_attention_layernorm(hh))

            normed = inner.norm(h)
            if lm.args.tie_word_embeddings:
                out = inner.embed_tokens.as_linear(normed)
            else:
                out = lm.lm_head(normed)
            # `h` is intentionally PRE-norm: the MTP head applies its own
            # pre_fc_norm_hidden and expects unnormalized backbone state.
            return out, h

        def mtp_forward(self, hidden_states, next_token_ids, mtp_cache=None):
            """Draft token t+2 from backbone hidden h_t and sampled token t+1."""
            lm = self.language_model
            out = lm.mtp(
                hidden_states, next_token_ids, lm.model.embed_tokens, mtp_cache
            )
            if lm.args.tie_word_embeddings:
                return lm.model.embed_tokens.as_linear(out)
            return lm.lm_head(out)

        def make_mtp_cache(self):
            lm = self.language_model
            mtp = getattr(lm, "mtp", None)
            if mtp is None:
                return []
            return [KVCache() for _ in mtp.layers]

    model.__class__ = _Qwen3_5WithMTP


def validate_mtp_support(model: Any) -> bool:
    """Check that a model actually has a usable MTP head attached."""
    lm = getattr(model, "language_model", None)
    mtp = getattr(lm, "mtp", None) if lm is not None else None
    if mtp is None or not getattr(mtp, "layers", []):
        return False
    return hasattr(model, "mtp_forward") and hasattr(model, "make_mtp_cache")
