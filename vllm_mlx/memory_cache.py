# SPDX-License-Identifier: Apache-2.0
"""
Memory-aware prefix cache for vllm-mlx.

This module provides a prefix cache implementation that tracks memory usage
and evicts entries based on memory pressure rather than entry count.

Key features:
- Automatic memory limit detection based on available system RAM
- Accurate memory tracking for MLX array caches
- LRU eviction triggered by memory thresholds
- No unnecessary deep copies (MLX arrays are immutable)

Example:
    config = MemoryCacheConfig(max_memory_percent=0.25)
    cache = MemoryAwarePrefixCache(model, config)

    # Fetch returns reference (no copy) - safe because MLX arrays are immutable
    kv_cache, remaining = cache.fetch(tokens)

    # Store tracks memory automatically
    cache.store(tokens, kv_cache)
"""

from __future__ import annotations

import bisect
import logging
import math
import threading as _threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Constants
_BYTES_PER_MB = 1024 * 1024
_DEFAULT_MEMORY_PERCENT = 0.20  # 20% of available RAM
_MIN_MEMORY_BYTES = 100 * _BYTES_PER_MB  # Minimum 100MB
_MAX_ENTRIES_FALLBACK = 50  # Fallback if memory detection fails
# Bump this when the cache on-disk format or KV semantics change.
# Loading a cache with a different version is rejected automatically.
#
# v4 (2026-08-17): content-keyed entry filenames (entry_<sha16>.safetensors)
# instead of positional entry_<i>, so periodic incremental flushes can tell
# "already durable on disk" from "new" without re-serializing everything.
# v3 dirs are simply discarded on load and the cache re-warms.
_CACHE_PERSIST_VERSION = 4

# Defaults for the durable top-K written by flush_to_disk(). Ten entries at
# up to 64 GB covers the Claude Code system-prompt prefixes we actually care
# about surviving a crash without pinning the whole warm cache to disk.
_DEFAULT_PERSIST_MAX_ENTRIES = 10
_DEFAULT_PERSIST_MAX_BYTES = 64 * 1024**3


def _get_available_memory() -> int:
    """
    Get available system memory in bytes.

    Returns:
        Available memory in bytes, or 0 if detection fails.
    """
    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        logger.warning("psutil not installed, using fallback memory limit")
        return 0
    except Exception as e:
        logger.warning(f"Failed to detect available memory: {e}")
        return 0


def _array_memory(arr) -> int:
    """
    Estimate array memory from shape+dtype without triggering lazy eval.

    Accessing .nbytes on a lazy MLX array forces evaluation of the entire
    computation graph, causing a VRAM spike. This function uses shape and
    dtype metadata (which are always available without eval) to compute
    the same value.

    Args:
        arr: An MLX array or similar object.

    Returns:
        Estimated memory in bytes.
    """
    if hasattr(arr, "shape") and hasattr(arr, "dtype"):
        dtype = arr.dtype
        if hasattr(dtype, "size"):
            return math.prod(arr.shape) * dtype.size
    # Fallback for non-MLX arrays or objects without shape/dtype
    if hasattr(arr, "nbytes"):
        return arr.nbytes
    return 0


def estimate_kv_cache_memory(cache: list[Any]) -> int:
    """
    Estimate memory usage of a KV cache in bytes.

    This function inspects MLX arrays in the cache and calculates their
    total memory footprint using shape+dtype metadata to avoid triggering
    lazy evaluation (which would cause a VRAM spike).

    Args:
        cache: List of layer cache objects, each containing keys/values tensors.

    Returns:
        Estimated memory usage in bytes.
    """
    if not cache:
        return 0

    total_bytes = 0

    for layer_cache in cache:
        # Handle different cache object types
        # Check dict first since dicts have .keys() method that would match below
        if isinstance(layer_cache, dict) and "state" in layer_cache:
            # Extracted state dict
            keys, values = layer_cache["state"]
            total_bytes += _array_memory(keys)
            total_bytes += _array_memory(values)
        # Handle QuantizedKVCache: keys/values are tuples of (data, scales, biases)
        elif hasattr(layer_cache, "keys") and isinstance(
            getattr(layer_cache, "keys", None), (list, tuple)
        ):
            for arr in layer_cache.keys:
                total_bytes += _array_memory(arr)
            for arr in layer_cache.values:
                total_bytes += _array_memory(arr)
            continue
        elif hasattr(layer_cache, "state") and not isinstance(layer_cache, dict):
            # Cache with a state property. The common shape is (keys, values),
            # but CacheList and recurrent caches nest arbitrarily; a two-way
            # unpack raised here and the swallowed error accounted the whole
            # entry as 0 bytes, so the byte-based LRU never evicted it
            # (upstream #683 hit Metal buffer exhaustion this way). Walk the
            # structure generically instead.
            def _walk_state(obj) -> int:
                if obj is None:
                    return 0
                if isinstance(obj, (list, tuple)):
                    return sum(_walk_state(item) for item in obj)
                return _array_memory(obj)

            try:
                total_bytes += _walk_state(layer_cache.state)
            except (TypeError, ValueError):
                pass
        elif hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
            # Standard KVCache with keys/values attributes (not dict)
            keys_attr = layer_cache.keys
            values_attr = layer_cache.values
            # Ensure these are arrays, not methods
            if not callable(keys_attr):
                total_bytes += _array_memory(keys_attr)
            if not callable(values_attr):
                total_bytes += _array_memory(values_attr)

    return total_bytes


@dataclass(frozen=True)
class MemoryCacheConfig:
    """
    Configuration for memory-aware prefix cache.

    Attributes:
        max_memory_mb: Maximum memory in MB. If None, auto-detects.
        max_memory_percent: Fraction of available RAM to use (0.0-1.0).
        max_entries: Hard limit on number of entries (safety net).
        enable_memory_tracking: Whether to track per-entry memory.
        kv_quantize: Whether to quantize KV cache layers for reduced memory.
        kv_bits: Number of bits for KV cache quantization.
        kv_group_size: Group size for KV cache quantization.
        kv_min_quantize_tokens: Minimum sequence length for quantization to apply.
        persist_max_entries: How many of the most-recent entries flush_to_disk()
            keeps durable on disk.
        persist_max_bytes: Byte budget the durable set must fit inside.
    """

    max_memory_mb: int | None = None
    max_memory_percent: float = _DEFAULT_MEMORY_PERCENT
    max_entries: int = 1000  # Safety limit
    enable_memory_tracking: bool = True
    kv_quantize: bool = False
    kv_bits: int = 8
    kv_group_size: int = 64
    kv_min_quantize_tokens: int = 256
    persist_max_entries: int = _DEFAULT_PERSIST_MAX_ENTRIES
    persist_max_bytes: int = _DEFAULT_PERSIST_MAX_BYTES

    def __post_init__(self) -> None:
        if not 0.0 < self.max_memory_percent <= 1.0:
            raise ValueError(
                f"max_memory_percent must be in (0, 1], got {self.max_memory_percent}"
            )
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {self.max_entries}")
        if self.kv_min_quantize_tokens < 0:
            raise ValueError(
                f"kv_min_quantize_tokens must be >= 0, got {self.kv_min_quantize_tokens}"
            )
        if self.persist_max_entries < 1:
            raise ValueError(
                f"persist_max_entries must be >= 1, got {self.persist_max_entries}"
            )
        if self.persist_max_bytes <= 0:
            raise ValueError(
                f"persist_max_bytes must be > 0, got {self.persist_max_bytes}"
            )

    def compute_memory_limit(self) -> int:
        """
        Compute the memory limit in bytes.

        Returns:
            Memory limit in bytes.
        """
        if self.max_memory_mb is not None:
            return self.max_memory_mb * _BYTES_PER_MB

        available = _get_available_memory()
        if available > 0:
            limit = int(available * self.max_memory_percent)
            return max(limit, _MIN_MEMORY_BYTES)

        # Fallback: assume 8GB system, use configured percent
        fallback_total = 8 * 1024 * _BYTES_PER_MB
        return int(fallback_total * self.max_memory_percent)


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    tokens_saved: int = 0
    current_memory_bytes: int = 0
    max_memory_bytes: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def memory_utilization(self) -> float:
        if self.max_memory_bytes == 0:
            return 0.0
        return self.current_memory_bytes / self.max_memory_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
            "tokens_saved": self.tokens_saved,
            "current_memory_mb": round(self.current_memory_bytes / _BYTES_PER_MB, 2),
            "max_memory_mb": round(self.max_memory_bytes / _BYTES_PER_MB, 2),
            "memory_utilization": round(self.memory_utilization, 4),
            "entry_count": self.entry_count,
        }


@dataclass
class _CacheEntry:
    """Internal cache entry with memory tracking."""

    tokens: tuple[int, ...]
    cache: list[Any]
    memory_bytes: int

    @classmethod
    def create(cls, tokens: list[int], cache: list[Any]) -> _CacheEntry:
        """Create a cache entry with memory estimation."""
        memory = estimate_kv_cache_memory(cache)
        return cls(
            tokens=tuple(tokens),
            cache=cache,
            memory_bytes=memory,
        )


def _tokens_hash(tokens_key: tuple[int, ...]) -> str:
    """Content-address a token key for its on-disk filename.

    The persisted filename is derived from the tokens themselves so a flush
    can ask "is this exact prefix already durable?" with a single os.path
    existence check — no positional index to keep in sync, and no
    re-serialization of entries that have not changed.
    """
    import array as _array
    import hashlib as _hashlib

    return _hashlib.sha256(_array.array("i", tokens_key).tobytes()).hexdigest()[:16]


def _clear_mlx_buffer_cache() -> None:
    """Return MLX's buffer-cache blocks to Metal.

    Serializing an entry is not allocation-free: save_prompt_cache
    materializes each layer cache's .state, which needs fresh Metal buffers.
    Under pressure — a dying engine at shutdown, or a busy server mid-flush —
    MLX's cache of freed-but-retained blocks can sit near the Metal resource
    limit and every large entry then fails with "[metal::malloc] Resource
    limit exceeded" (observed 2026-08-17: 86/88 entries lost, 59 GB of warm
    prefixes gone — exactly the caches most worth persisting). Returning
    those blocks up front, and once more between a failure and one retry, is
    what lets the saves complete.

    Silently no-ops when MLX is absent (unit tests) or the API is older; the
    retry still runs either way.
    """
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def _select_for_persist(
    snapshot: list[tuple[tuple[int, ...], _CacheEntry]],
    max_entries: int,
    max_bytes: int,
) -> tuple[list[tuple[str, tuple[int, ...], int]], dict[str, _CacheEntry]]:
    """Pick the durable top-K out of an LRU snapshot (newest -> oldest).

    Walks the snapshot newest-first and keeps an entry when fewer than
    ``max_entries`` have been picked AND it fits the remaining byte budget. An
    entry too big for what's left is *skipped*, not a stop signal — smaller
    older prefixes still get their turn (packing).

    Returns ``(selected, pending)`` where ``selected`` carries only scalars
    (hash, token key, memory_bytes) and ``pending`` maps hash -> entry. Split
    that way so the caller can release each entry's MLX arrays as soon as it
    has been written, while still having everything the index needs. This
    function is where the rejected entries' references die: they live only in
    this frame.
    """
    selected: list[tuple[str, tuple[int, ...], int]] = []
    pending: dict[str, _CacheEntry] = {}
    remaining_bytes = max_bytes
    for tokens_key, entry in reversed(snapshot):
        if len(selected) >= max_entries:
            break
        if entry.memory_bytes > remaining_bytes:
            continue
        h = _tokens_hash(tokens_key)
        selected.append((h, tokens_key, entry.memory_bytes))
        pending[h] = entry
        remaining_bytes -= entry.memory_bytes
    return selected, pending


def _write_entry_files(
    cache_dir: str,
    h: str,
    tokens_key: tuple[int, ...],
    entry: _CacheEntry,
    save_prompt_cache: Any,
) -> None:
    """Write one entry's two durable files, atomically and collision-free.

    Both files are built under a unique ``tempfile.mkstemp`` name and
    committed by rename, so (a) a save that dies halfway — killed process,
    full disk, Metal failure — never leaves a partial file at the final path,
    and (b) two writers over the same cache directory cannot land in the same
    temp and produce a file carrying bytes from both.

    The tokens file is renamed ONLY after the safetensors rename succeeds: the
    pair is loadable only if both exist, and a lone tokens file would make an
    entry look half-durable to the incremental check.

    Raises whatever the underlying save raises (after the Metal-pressure
    clear + single retry) so the caller can log and move on to the next entry.
    """
    import array as _array
    import os
    import tempfile

    entry_path = os.path.join(cache_dir, f"entry_{h}.safetensors")
    tokens_path = os.path.join(cache_dir, f"entry_{h}_tokens.bin")

    fd, tmp_entry = tempfile.mkstemp(
        dir=cache_dir, prefix=f".entry_{h}.", suffix=".safetensors"
    )
    os.close(fd)
    tmp_tokens: str | None = None
    try:
        metadata = {"num_tokens": str(len(tokens_key))}
        try:
            save_prompt_cache(tmp_entry, entry.cache, metadata=metadata)
        except Exception as first_err:
            _clear_mlx_buffer_cache()
            logger.info(
                f"[cache_persist] entry {h} failed once "
                f"({first_err}); retrying after mx.clear_cache()"
            )
            save_prompt_cache(tmp_entry, entry.cache, metadata=metadata)

        # Tokens go in a separate binary file (100K+ ints is much smaller as
        # int32 than as JSON).
        fd, tmp_tokens = tempfile.mkstemp(
            dir=cache_dir, prefix=f".entry_{h}_tokens.", suffix=".bin"
        )
        with os.fdopen(fd, "wb") as f:
            _array.array("i", tokens_key).tofile(f)  # 32-bit signed ints

        os.rename(tmp_entry, entry_path)
        tmp_entry = None  # type: ignore[assignment]
        os.rename(tmp_tokens, tokens_path)
        tmp_tokens = None
    finally:
        for leftover in (tmp_entry, tmp_tokens):
            if leftover is None:
                continue
            try:
                os.remove(leftover)
            except OSError:
                pass


def _trim_cache_offset(cache: list[Any], trim_by: int) -> list[Any]:
    """Create shallow copies of KVCache/QuantizedKVCache layers with offset reduced.

    This is used when returning a cached KV state to the scheduler so that
    the last N positions are "freed" and the model will recompute them on the
    next forward pass (preventing duplicate KV entries).

    Supports both KVCache (keys/values are arrays) and QuantizedKVCache
    (keys/values are 3-tuples of arrays).
    """
    try:
        from mlx_lm.models.cache import KVCache
    except ImportError:
        # Environments without mlx_lm (the no-MLX CI test lane) still route
        # mock-based fetches through this trim path. Production always has
        # mlx_lm; the shim only needs the attribute surface the scheduler
        # and tests read.
        class KVCache:  # noqa: N801 - mirrors the mlx_lm class name
            keys = None
            values = None
            offset = 0

    try:
        from mlx_lm.models.cache import QuantizedKVCache
    except ImportError:
        QuantizedKVCache = None  # noqa: N806

    trimmed: list[Any] = []
    for layer_cache in cache:
        if QuantizedKVCache is not None and isinstance(layer_cache, QuantizedKVCache):
            tc = QuantizedKVCache.__new__(QuantizedKVCache)
            tc.keys = layer_cache.keys
            tc.values = layer_cache.values
            tc.offset = max(layer_cache.offset - trim_by, 0)
            tc.group_size = layer_cache.group_size
            tc.bits = layer_cache.bits
            trimmed.append(tc)
        elif (
            hasattr(layer_cache, "offset")
            and hasattr(layer_cache, "keys")
            and not isinstance(layer_cache.keys, (list, tuple))
        ):
            tc = KVCache.__new__(KVCache)
            new_offset = max(layer_cache.offset - trim_by, 0)
            keys = layer_cache.keys
            values = layer_cache.values
            # Slice arrays down to new_offset rather than shrinking only
            # the offset pointer. Sharing the oversized array across
            # requests exposes stale tokens to paths that read
            # cache.state directly — waybarrios/vllm-mlx#384,
            # jackneil/vllm-mlx-patched#29.
            if (
                keys is not None
                and hasattr(keys, "shape")
                and len(keys.shape) >= 3
                and new_offset < keys.shape[-2]
            ):
                tc.keys = keys[..., :new_offset, :]
                tc.values = values[..., :new_offset, :]
            else:
                tc.keys = keys
                tc.values = values
            tc.offset = new_offset
            trimmed.append(tc)
        else:
            trimmed.append(layer_cache)
    return trimmed


def _needs_kv_trim(layer: Any) -> bool:
    """Check if a cache layer has oversized KV arrays (duck-typed, no MLX import)."""
    keys = getattr(layer, "keys", None)
    offset = getattr(layer, "offset", None)
    if keys is None or offset is None:
        return False
    if isinstance(keys, (list, tuple)):
        return False  # QuantizedKVCache — skip
    shape = getattr(keys, "shape", None)
    if shape is None or len(shape) < 3:
        return False
    return 0 < offset < shape[2]


def _trim_to_offset(cache: list[Any]) -> list[Any]:
    """Trim KV arrays to their actual used size (offset) before storage.

    KV arrays are often pre-allocated larger than needed (e.g. 4096 slots
    when only 100 are used).  This slices them down to ``offset`` and
    evaluates the result so the original large buffer can be freed.

    Args:
        cache: List of cache layer objects (KVCache or other types).

    Returns:
        New list with KVCache layers trimmed to their offset.
        Non-KVCache layers are passed through unchanged.
    """
    if not any(_needs_kv_trim(layer) for layer in cache):
        return cache

    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    trimmed = []
    eval_targets = []
    for layer in cache:
        if isinstance(layer, KVCache) and layer.keys is not None:
            offset = layer.offset
            if offset <= 0 or offset >= layer.keys.shape[2]:
                trimmed.append(layer)
                continue
            tc = KVCache()
            tc.keys = layer.keys[:, :, :offset, :]
            tc.values = layer.values[:, :, :offset, :]
            tc.offset = offset
            eval_targets.extend([tc.keys, tc.values])
            trimmed.append(tc)
        else:
            trimmed.append(layer)

    if eval_targets:
        mx.eval(*eval_targets)

    return trimmed


def _quantize_cache(cache: list[Any], bits: int = 8, group_size: int = 64) -> list[Any]:
    """Quantize KVCache layers to reduce memory. Non-KVCache layers are kept as-is."""
    from mlx_lm.models.cache import KVCache

    quantized = []
    for layer in cache:
        if isinstance(layer, KVCache) and layer.keys is not None:
            quantized.append(layer.to_quantized(group_size=group_size, bits=bits))
        else:
            quantized.append(layer)
    return quantized


def _dequantize_cache(cache: list[Any]) -> list[Any]:
    """Dequantize QuantizedKVCache layers back to regular KVCache.

    After dequantize, slice the keys/values down to ``offset`` so readers
    that bypass ``offset`` (e.g. Gemma 4 KV-shared layers reading
    cache.state directly, Qwen3 kickoff on supersequence matches) cannot
    see stale tokens from a previous owner of the buffer. Mirrors the
    plain-KVCache slice in ``_trim_cache_offset`` —
    waybarrios/vllm-mlx#384, jackneil/vllm-mlx-patched#29.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    result = []
    for layer in cache:
        if isinstance(layer, QuantizedKVCache) and layer.keys is not None:
            kv = KVCache()
            kv.keys = mx.dequantize(
                *layer.keys, group_size=layer.group_size, bits=layer.bits
            )
            kv.values = mx.dequantize(
                *layer.values, group_size=layer.group_size, bits=layer.bits
            )
            kv.offset = layer.offset
            if (
                kv.keys is not None
                and hasattr(kv.keys, "shape")
                and len(kv.keys.shape) >= 3
                and kv.offset < kv.keys.shape[-2]
            ):
                kv.keys = kv.keys[..., : kv.offset, :]
                kv.values = kv.values[..., : kv.offset, :]
            result.append(kv)
        else:
            result.append(layer)
    return result


def _compute_model_fingerprint(model: Any) -> str:
    """Compute a fingerprint from model architecture for cache compatibility.

    Used to reject disk-persisted caches created by a different model or a
    different quantisation of the same model. The fingerprint is a short
    hex digest of (num_hidden_layers, hidden_size, vocab_size,
    num_key_value_heads, head_dim, intermediate_size, model_type) —
    lightweight and deterministic.

    Ported from waybarrios/vllm-mlx#365, commit 01261c1.
    """
    import hashlib

    cfg = None
    for cfg_attr in ("config", "args", "model_config"):
        cfg = getattr(model, cfg_attr, None)
        if cfg is not None:
            break
    if cfg is None:
        cfg = model

    parts: list[str] = []
    for key in (
        "num_hidden_layers",
        "hidden_size",
        "vocab_size",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "model_type",
    ):
        val = getattr(cfg, key, None)
        if val is not None:
            parts.append(f"{key}={val}")

    fingerprint = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    logger.debug(f"[model_fingerprint] {fingerprint} ({', '.join(parts)})")
    return fingerprint


class MemoryAwarePrefixCache:
    """
    Prefix cache with memory-based eviction.

    This cache tracks memory usage per entry and evicts based on memory
    pressure rather than entry count. It uses LRU (Least Recently Used)
    ordering for eviction decisions.

    Key design decisions:
    - No deep copies on fetch: MLX arrays are immutable, so sharing is safe
    - Memory tracking per entry: Accurate accounting for eviction
    - Auto-detection of available RAM: Adapts to different systems
    - OrderedDict for O(1) LRU operations

    Thread Safety:
        The entry-table mutators (fetch/store/remove/clear/load_from_disk)
        and flush_to_disk's snapshot take ``self._lock`` (an RLock), which is
        what makes a periodic flush safe while the engine loop keeps serving.
        Persistence additionally holds ``self._persist_lock`` across its file
        I/O so two flushes (or a flush and the shutdown save) can never write
        the same cache directory at once. Nothing else is synchronized — treat
        the rest of the class as single-writer.
    """

    def __init__(
        self,
        model: Any,
        config: MemoryCacheConfig | None = None,
    ) -> None:
        """
        Initialize the memory-aware prefix cache.

        Args:
            model: The MLX model (used for identification).
            config: Cache configuration. Uses defaults if None.
        """
        self._model_id = id(model)
        self._config = config or MemoryCacheConfig()
        self._model_fingerprint = _compute_model_fingerprint(model)

        # Guards the entry table + sorted index. Reentrant because the
        # locked public methods call each other's helpers (store -> _evict_lru).
        # Held only for in-memory bookkeeping — never across file I/O.
        self._lock = _threading.RLock()

        # Serializes persistence for this cache: flush_to_disk holds it from
        # selection through the index rename, and save_to_disk inherits it by
        # delegation. Two writers over one cache directory interleave their
        # entry bytes and prune each other's files out from under a committed
        # index — reproduced by a reviewer as 3-of-4 entry files carrying bytes
        # from both writers. Distinct from _lock: this one IS held across file
        # I/O, which is exactly why it must never be the entry-table lock.
        self._persist_lock = _threading.Lock()

        # OrderedDict maintains insertion order for LRU
        # Key: tuple(tokens), Value: _CacheEntry
        self._entries: OrderedDict[tuple[int, ...], _CacheEntry] = OrderedDict()

        # Sorted index of token keys for efficient prefix/supersequence lookup.
        # Tuple lexicographic ordering means a prefix key P is always < any
        # extension of P, so bisect gives O(log N) range scans instead of O(N).
        self._sorted_keys: list[tuple[int, ...]] = []

        # Memory tracking
        self._max_memory = self._config.compute_memory_limit()
        self._current_memory = 0

        # Statistics
        self._stats = CacheStats(max_memory_bytes=self._max_memory)

        # Track the match type from the last fetch() call
        self._last_match_type: str | None = None

        # In-flight guard for clear() — PR-A Task A.1 + acquire/release
        # wiring for production (follow-up). The set is the canonical
        # representation; _in_flight_count is a derived view kept for
        # backward compatibility with tests that read it directly.
        #
        # acquire(req_id) and release(req_id) are called by the scheduler
        # at request enter / exit. clear() refuses when the set is
        # non-empty so DELETE /v1/cache returns HTTP 409 instead of
        # wiping state mid-decode and crashing the generation loop.
        self._in_flight_lock = _threading.Lock()
        self._in_flight_ids: set[str] = set()

        logger.info(
            f"MemoryAwarePrefixCache initialized: "
            f"max_memory={self._max_memory / _BYTES_PER_MB:.1f}MB, "
            f"max_entries={self._config.max_entries}"
        )

    def fetch(self, tokens: list[int]) -> tuple[list[Any] | None, list[int]]:
        """Locked wrapper around :meth:`_fetch_unlocked`.

        fetch() reorders the LRU OrderedDict (move_to_end), which would make a
        concurrent flush snapshot blow up mid-iteration. See the class-level
        Thread Safety note.
        """
        with self._lock:
            return self._fetch_unlocked(tokens)

    def _fetch_unlocked(self, tokens: list[int]) -> tuple[list[Any] | None, list[int]]:
        """
        Find cached KV state for the given tokens.

        This method searches for exact matches, prefix matches, supersequence
        matches, and longest-common-prefix (LCP) matches.  Uses a sorted key
        index for O(log N) lookup instead of scanning all entries.

        Returns the cached KV state directly (no copy) since MLX arrays
        are immutable and safe to share.

        Args:
            tokens: Input token sequence.

        Returns:
            Tuple of (cache, remaining_tokens):
            - cache: Cached KV state if found, None otherwise
            - remaining_tokens: Tokens that still need processing
        """
        if not tokens:
            self._stats.misses += 1
            self._last_match_type = "miss"
            return None, tokens

        tokens_key = tuple(tokens)

        # --- O(1) exact match ---
        # Never return a cache that covers the ENTIRE key: the scheduler must
        # feed at least one token to kick off generation, and feeding
        # prompt[-1] into a cache that already contains it duplicates the
        # final token (upstream #683 measured the same prompt returning a
        # different answer). When every layer is trimmable, trim one token
        # off the returned copy and hand back the last token as remaining;
        # otherwise fall through — the divergent-suffix paths below skip
        # non-trimmable candidates, so the request cold-prefills correctly.
        if tokens_key in self._entries:
            entry = self._entries[tokens_key]
            exact_trimmable = not any(
                not (hasattr(lc, "offset") and hasattr(lc, "keys"))
                for lc in entry.cache
            )
            if exact_trimmable:
                self._entries.move_to_end(tokens_key)
                self._stats.hits += 1
                self._stats.tokens_saved += len(tokens) - 1
                self._last_match_type = "exact"
                trimmed = _trim_cache_offset(entry.cache, 1)
                cache_out = (
                    _dequantize_cache(trimmed) if self._config.kv_quantize else trimmed
                )
                return cache_out, tokens[-1:]

        # --- O(log N) prefix & supersequence match via sorted index ---
        best_match: _CacheEntry | None = None
        best_length = 0
        best_super: _CacheEntry | None = None

        sorted_keys = self._sorted_keys
        if sorted_keys:
            # Find insertion point for tokens_key in the sorted list.
            # Keys that are prefixes of tokens_key or supersequences will be
            # clustered around this position due to lexicographic ordering.
            idx = bisect.bisect_left(sorted_keys, tokens_key)

            # Scan backwards from idx to find cached keys that are PREFIXES
            # of tokens_key (shorter cached sequences).  A prefix P of T
            # satisfies P <= T lexicographically, so P is at idx-1 or earlier.
            for i in range(idx - 1, -1, -1):
                cached_key = sorted_keys[i]
                cached_len = len(cached_key)
                if cached_len >= len(tokens_key):
                    continue  # Not a prefix (same length or longer)
                # Check if cached_key is a prefix of tokens_key
                if tokens_key[:cached_len] == cached_key:
                    if cached_len > best_length:
                        best_match = self._entries[cached_key]
                        best_length = cached_len
                    # Found best prefix — shorter entries can't be longer
                    break
                # Once we go past the prefix range, stop
                if cached_key[0] != tokens_key[0]:
                    break

            # Scan forward from idx to find cached keys that are SUPERSEQUENCES
            # of tokens_key (longer cached sequences starting with tokens_key).
            for i in range(idx, len(sorted_keys)):
                cached_key = sorted_keys[i]
                cached_len = len(cached_key)
                if cached_len < len(tokens_key):
                    continue
                # Check if tokens_key is a prefix of cached_key
                if cached_key[: len(tokens_key)] == tokens_key:
                    if best_super is None or cached_len > len(best_super.tokens):
                        best_super = self._entries[cached_key]
                else:
                    # Past the supersequence range
                    break

        # --- Supersequence match handling ---
        if best_super is not None:
            n_cached = len(best_super.tokens)
            n_requested = len(tokens)
            excess = n_cached - n_requested

            has_non_trimmable = any(
                not (hasattr(lc, "offset") and hasattr(lc, "keys"))
                for lc in best_super.cache
            )

            if excess > 0 and has_non_trimmable:
                logger.debug(
                    "[cache_fetch] supersequence match skipped: "
                    "non-trimmable cache layers (hybrid model)"
                )
            elif excess > 0:
                # Trim excess + 1 so the returned cache never covers the whole
                # key (see the exact-match note above).
                trimmed_cache = _trim_cache_offset(best_super.cache, excess + 1)
                self._entries.move_to_end(best_super.tokens)
                self._stats.hits += 1
                self._stats.tokens_saved += n_requested - 1
                self._last_match_type = "supersequence"
                trimmed_cache = (
                    _dequantize_cache(trimmed_cache)
                    if self._config.kv_quantize
                    else trimmed_cache
                )
                return trimmed_cache, tokens[-1:]

        # --- Prefix match ---
        if best_match is not None:
            self._entries.move_to_end(best_match.tokens)
            self._stats.hits += 1
            self._stats.tokens_saved += best_length
            remaining = tokens[best_length:]
            self._last_match_type = "prefix"
            cache_out = (
                _dequantize_cache(best_match.cache)
                if self._config.kv_quantize
                else best_match.cache
            )
            return cache_out, remaining

        # --- LCP (Longest Common Prefix) for divergent sequences ---
        # This handles the agentic pattern: same system+context prefix
        # but different final user message.  Use the sorted index to find
        # the nearest neighbor which likely shares the longest prefix.
        best_lcp_entry: _CacheEntry | None = None
        best_lcp_length = 0

        if sorted_keys:
            idx = bisect.bisect_left(sorted_keys, tokens_key)
            # Check neighbors around insertion point (they share the most
            # common prefix due to lexicographic ordering).
            for i in (idx - 1, idx):
                if i < 0 or i >= len(sorted_keys):
                    continue
                cached_key = sorted_keys[i]
                if cached_key == tokens_key:
                    continue  # Skip exact (already handled)
                min_len = min(len(cached_key), len(tokens_key))
                if min_len <= best_lcp_length:
                    continue
                # Compute LCP length
                lcp = 0
                for j in range(min_len):
                    if cached_key[j] != tokens_key[j]:
                        break
                    lcp = j + 1
                if lcp > best_lcp_length:
                    best_lcp_entry = self._entries[cached_key]
                    best_lcp_length = lcp
                    logger.debug(
                        f"[cache_fetch] LCP scan: cached_len={len(cached_key)} "
                        f"req_len={len(tokens_key)} lcp={lcp}"
                    )

        if best_lcp_entry is not None and best_lcp_length > 0:
            excess = len(best_lcp_entry.tokens) - best_lcp_length

            has_non_trimmable = any(
                not (hasattr(lc, "offset") and hasattr(lc, "keys"))
                for lc in best_lcp_entry.cache
            )
            logger.debug(
                f"[cache_fetch] LCP candidate: lcp={best_lcp_length} "
                f"entry_len={len(best_lcp_entry.tokens)} excess={excess} "
                f"non_trimmable={has_non_trimmable} "
                f"cache_layers={len(best_lcp_entry.cache)} "
                f"layer_types={[type(lc).__name__ for lc in best_lcp_entry.cache[:3]]}"
            )

            if not has_non_trimmable:
                trimmed_cache = _trim_cache_offset(best_lcp_entry.cache, excess)
                self._entries.move_to_end(best_lcp_entry.tokens)
                self._stats.hits += 1
                self._stats.tokens_saved += best_lcp_length
                remaining = tokens[best_lcp_length:]
                logger.debug(
                    f"[cache_fetch] LCP hit: shared={best_lcp_length} "
                    f"trimmed={excess} remaining={len(remaining)}"
                )
                self._last_match_type = "lcp"
                trimmed_cache = (
                    _dequantize_cache(trimmed_cache)
                    if self._config.kv_quantize
                    else trimmed_cache
                )
                return trimmed_cache, remaining

        self._stats.misses += 1
        self._last_match_type = "miss"

        return None, tokens

    def store(
        self, tokens: list[int], cache: list[Any], evict_prefixes: bool = True
    ) -> bool:
        """Locked wrapper around :meth:`_store_unlocked`."""
        with self._lock:
            return self._store_unlocked(tokens, cache, evict_prefixes=evict_prefixes)

    def _store_unlocked(
        self, tokens: list[int], cache: list[Any], evict_prefixes: bool = True
    ) -> bool:
        """
        Store KV cache for future reuse.

        This method stores the cache reference directly (no copy) and
        tracks memory usage. If memory limit is exceeded, LRU entries
        are evicted until there's room.

        Args:
            tokens: Token sequence that was processed.
            cache: The computed KV cache to store.
            evict_prefixes: If True, evict existing entries whose token
                sequence is a strict prefix of ``tokens``.  Set to False
                when storing prompt+output entries to preserve prompt-only
                entries created by prompt_cache_save (those are the entries
                that future requests will actually match).

        Returns:
            True if stored successfully, False if rejected.
        """
        if not tokens or not cache:
            return False

        tokens_key = tuple(tokens)

        # If already cached, just update LRU order (skip expensive trim/quantize)
        if tokens_key in self._entries:
            self._entries.move_to_end(tokens_key)
            return True

        # Trim oversized KV arrays to actual used size
        cache = _trim_to_offset(cache)

        # Quantize if enabled and sequence is long enough
        if (
            self._config.kv_quantize
            and len(tokens) >= self._config.kv_min_quantize_tokens
        ):
            cache = _quantize_cache(
                cache, self._config.kv_bits, self._config.kv_group_size
            )

        # Create entry and estimate memory
        entry = _CacheEntry.create(tokens, cache)

        # Check if single entry exceeds limit
        if entry.memory_bytes > self._max_memory:
            logger.warning(
                f"Cache entry too large: {entry.memory_bytes / _BYTES_PER_MB:.1f}MB "
                f"exceeds limit {self._max_memory / _BYTES_PER_MB:.1f}MB"
            )
            return False

        # Prefix-subset eviction: remove entries whose token sequence
        # is a strict prefix of the new entry.  Uses sorted index for
        # O(log N + K) lookup instead of O(N) scan.
        if evict_prefixes and self._sorted_keys:
            to_remove = []
            idx = bisect.bisect_left(self._sorted_keys, tokens_key)
            # Scan backwards — prefixes of tokens_key are immediately before idx
            for i in range(idx - 1, -1, -1):
                key = self._sorted_keys[i]
                klen = len(key)
                if klen >= len(tokens_key):
                    continue
                if tokens_key[:klen] == key:
                    to_remove.append(key)
                elif key[0] != tokens_key[0]:
                    break
            for key in to_remove:
                old = self._entries.pop(key)
                self._current_memory -= old.memory_bytes
                self._stats.evictions += 1
                self._remove_from_sorted(key)
                logger.debug(
                    f"[prefix_evict] removed {len(key)} tokens, "
                    f"freed {old.memory_bytes / _BYTES_PER_MB:.2f}MB, "
                    f"new_entry={len(tokens_key)} tokens"
                )
            if to_remove:
                self._stats.entry_count = len(self._entries)
                self._stats.current_memory_bytes = self._current_memory

        # Evict until we have room
        while (
            self._current_memory + entry.memory_bytes > self._max_memory
            or len(self._entries) >= self._config.max_entries
        ) and self._entries:
            self._evict_lru()

        # Store entry
        self._entries[tokens_key] = entry
        self._current_memory += entry.memory_bytes
        bisect.insort(self._sorted_keys, tokens_key)
        self._stats.entry_count = len(self._entries)
        self._stats.current_memory_bytes = self._current_memory

        logger.debug(
            f"Stored cache: {len(tokens)} tokens, "
            f"{entry.memory_bytes / _BYTES_PER_MB:.2f}MB, "
            f"total={self._current_memory / _BYTES_PER_MB:.1f}MB"
        )

        return True

    def _remove_from_sorted(self, key: tuple[int, ...]) -> None:
        """Remove a key from the sorted index using bisect for O(log N)."""
        idx = bisect.bisect_left(self._sorted_keys, key)
        if idx < len(self._sorted_keys) and self._sorted_keys[idx] == key:
            self._sorted_keys.pop(idx)

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._entries:
            return

        # popitem(last=False) removes oldest entry (FIFO order = LRU)
        tokens_key, entry = self._entries.popitem(last=False)
        self._current_memory -= entry.memory_bytes
        self._remove_from_sorted(tokens_key)
        self._stats.evictions += 1
        self._stats.entry_count = len(self._entries)
        self._stats.current_memory_bytes = self._current_memory

        logger.debug(
            f"[lru_evict] removed {len(tokens_key)} tokens, "
            f"freed {entry.memory_bytes / _BYTES_PER_MB:.2f}MB"
        )

    def remove(self, tokens: list[int]) -> bool:
        """
        Remove a specific cache entry.

        Args:
            tokens: Token sequence to remove.

        Returns:
            True if entry was found and removed.
        """
        with self._lock:
            tokens_key = tuple(tokens)
            entry = self._entries.pop(tokens_key, None)
            if entry is not None:
                self._current_memory -= entry.memory_bytes
                self._remove_from_sorted(tokens_key)
                self._stats.entry_count = len(self._entries)
                self._stats.current_memory_bytes = self._current_memory
                return True
            return False

    def clear(self) -> bool:
        """Clear all cached entries.

        Refuses (returns False) when any entry is held by an in-flight
        request. The adapter caller aggregates refusals so DELETE
        /v1/cache returns HTTP 409 when concurrent load prevents a safe
        clear. Mirrors PagedCacheManager.reset_prefix_cache() in
        paged_cache.py:1149-1156 (UPSTREAM_PIN cache-clear invariant —
        see PR-A post-impl /dc).

        Returns
        -------
        bool
            True if the cache was wiped; False if the clear was refused
            because entries are in use.
        """
        with self._in_flight_lock:
            num_in_use = len(self._in_flight_ids)
            if num_in_use > 0:
                logger.warning(
                    "[prefix-cache-admin] MemoryAwarePrefixCache.clear refused: "
                    "%d entries in use. Drain traffic or wait for idle.",
                    num_in_use,
                )
                return False

            with self._lock:
                self._entries.clear()
                self._sorted_keys.clear()
                self._current_memory = 0
                self._stats = CacheStats(max_memory_bytes=self._max_memory)
            logger.debug("Cache cleared")
            return True

    # -----------------------------------------------------------------
    # In-flight guard API — PR-A Task A.1 + production acquire/release.
    # Scheduler calls acquire(request_id) when a request enters and
    # release(request_id) when it exits (finished or aborted). The
    # underlying set makes both operations idempotent so double-calls
    # and unknown-id releases are safe.
    # -----------------------------------------------------------------
    @property
    def _in_flight_count(self) -> int:
        """Derived view kept for test/backward-compat readers."""
        with self._in_flight_lock:
            return len(self._in_flight_ids)

    def acquire(self, request_id: str) -> None:
        """Record that `request_id` holds entries and blocks clear()."""
        with self._in_flight_lock:
            self._in_flight_ids.add(request_id)

    def release(self, request_id: str) -> None:
        """Release `request_id`'s hold. No-op if unknown (idempotent)."""
        with self._in_flight_lock:
            self._in_flight_ids.discard(request_id)

    def _mark_in_use_for_test(self) -> None:
        """Bump the in-flight set with a synthetic id so `clear()` refuses.

        Test-only affordance. Each call adds a fresh synthetic id so
        multiple invocations stack (matching the pre-acquire counter
        behavior the existing tests rely on).
        """
        import uuid as _uuid

        self.acquire(f"_test_{_uuid.uuid4()}")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._stats.to_dict()

    def reset_stats(self) -> None:
        """Reset statistics while preserving cache contents."""
        self._stats = CacheStats(
            max_memory_bytes=self._max_memory,
            current_memory_bytes=self._current_memory,
            entry_count=len(self._entries),
        )

    @property
    def memory_usage_mb(self) -> float:
        """Current memory usage in MB."""
        return self._current_memory / _BYTES_PER_MB

    @property
    def memory_limit_mb(self) -> float:
        """Memory limit in MB."""
        return self._max_memory / _BYTES_PER_MB

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._entries)

    def __contains__(self, tokens: list[int]) -> bool:
        """Check if tokens are cached."""
        return tuple(tokens) in self._entries

    # -----------------------------------------------------------------
    # Disk persistence — survives server restarts
    # -----------------------------------------------------------------

    def flush_to_disk(
        self,
        cache_dir: str,
        max_entries: int | None = None,
        max_bytes: int | None = None,
    ) -> int:
        """Make the N most-recent cache prefixes durable on disk, incrementally.

        This is the single persistence implementation; :meth:`save_to_disk`
        (the shutdown path) delegates here. It is safe to call repeatedly on a
        live, serving process: entries already on disk are left alone, so a
        steady-state flush with no new prefixes does zero MLX work.

        Directory layout (format v4)::

            cache_dir/
              index.json                     # ordered oldest -> newest
              entry_<sha16>.safetensors      # KV arrays, content-keyed
              entry_<sha16>_tokens.bin       # int32 token key

        Selection walks the LRU newest-first and keeps an entry when fewer
        than ``max_entries`` have been picked AND it fits the remaining byte
        budget. An entry too big for what's left is *skipped*, not a stop
        signal — smaller older prefixes still get their turn (packing).

        Anything on disk outside the selected set is pruned, so the directory
        holds exactly the durable top-K. The one exception: an *empty*
        in-memory cache is a no-op — a fresh boot or a just-cleared cache must
        not wipe what the last run made durable.

        Serialized per instance by ``self._persist_lock``, held from selection
        through the index rename, so a periodic flush and the shutdown save
        can never write the same directory at once.

        Args:
            cache_dir: Directory to persist into (created if absent).
            max_entries: Durable entry count. Defaults to the config's
                ``persist_max_entries``.
            max_bytes: Durable byte budget. Defaults to the config's
                ``persist_max_bytes``.

        Raises:
            ValueError: If ``max_entries`` < 1 or ``max_bytes`` <= 0. This
                method PRUNES everything outside the selected set, so a
                nonsense budget must fail loudly rather than quietly wipe the
                durable cache.

        Returns:
            The number of entries newly written by *this* call (0 when
            everything selected was already durable).
        """
        return self._flush_to_disk(cache_dir, max_entries, max_bytes)[0]

    def _flush_to_disk(
        self,
        cache_dir: str,
        max_entries: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, int]:
        """Shared persistence implementation. See :meth:`flush_to_disk`.

        Returns ``(newly_saved, durable_count)``. :meth:`save_to_disk` needs
        the durable count to answer its historical bool from this call's own
        knowledge, instead of re-reading index.json after the persistence lock
        has been dropped (where a concurrent flush may be mid-rename).
        """
        import json
        import os
        import tempfile
        import time as _time

        if max_entries is None:
            max_entries = self._config.persist_max_entries
        if max_bytes is None:
            max_bytes = self._config.persist_max_bytes
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be > 0, got {max_bytes}")

        t0 = _time.monotonic()

        # Everything below — selection, entry writes, prune, index commit —
        # runs under one lock. Dropping it between the prune and the index
        # rename is what lets a second writer publish an index naming files
        # this one is about to delete.
        with self._persist_lock:
            # Snapshot under _lock so a concurrent store/fetch can't mutate the
            # OrderedDict mid-iteration. _select_for_persist keeps references
            # ONLY to the entries it picked; the snapshot (and with it every
            # rejected entry) is dropped before any file I/O starts.
            with self._lock:
                snapshot = list(self._entries.items())  # oldest -> newest
            had_entries = bool(snapshot)
            selected, pending = _select_for_persist(snapshot, max_entries, max_bytes)
            del snapshot

            if not selected and not had_entries:
                # Nothing in memory at all: a cache that was just cleared
                # (DELETE /v1/cache) or a process that has not warmed up yet.
                # Falling through would prune the durable set and publish an
                # empty index — destroying the prefixes the LAST run made
                # durable, which is the whole point of persisting them.
                logger.debug(
                    "[cache_persist] cache is empty; leaving %s untouched",
                    cache_dir,
                )
                return 0, 0

            os.makedirs(cache_dir, exist_ok=True)

            def _paths(h: str) -> tuple[str, str]:
                return (
                    os.path.join(cache_dir, f"entry_{h}.safetensors"),
                    os.path.join(cache_dir, f"entry_{h}_tokens.bin"),
                )

            # --- incremental: only entries not already durable need saving ---
            need_save = [
                item
                for item in selected
                if not all(os.path.exists(p) for p in _paths(item[0]))
            ]
            need_save_hashes = {item[0] for item in need_save}
            # Entries already on disk are not written again, so stop pinning
            # their MLX arrays against eviction right now.
            for h in list(pending):
                if h not in need_save_hashes:
                    del pending[h]

            saved_hashes: set[str] = set()
            if need_save:
                try:
                    from mlx_lm.models.cache import save_prompt_cache
                except ImportError:
                    logger.warning("[cache_persist] mlx_lm not available, cannot save")
                    return 0, 0

                _clear_mlx_buffer_cache()

                for h, tokens_key, memory_bytes in need_save:
                    entry = pending.pop(h)
                    entry_path, _tokens_path = _paths(h)
                    try:
                        _write_entry_files(
                            cache_dir, h, tokens_key, entry, save_prompt_cache
                        )
                        saved_hashes.add(h)
                        logger.info(
                            f"[cache_persist] saved entry {h}: "
                            f"{len(tokens_key)} tokens, "
                            f"{memory_bytes / _BYTES_PER_MB:.1f}MB KV, "
                            f"file={entry_path}"
                        )
                    except Exception as e:
                        logger.warning(f"[cache_persist] failed to save entry {h}: {e}")
                    finally:
                        # Release THIS entry the moment it is written rather
                        # than holding the whole selection alive for the whole
                        # save: a multi-GB flush would otherwise pin every
                        # selected prefix against eviction for minutes.
                        del entry

            # Durable set = everything selected that is actually on disk now.
            durable = [
                item
                for item in selected
                if item[0] in saved_hashes or item[0] not in need_save_hashes
            ]
            durable_hashes = {item[0] for item in durable}

            # --- prune: disk holds exactly the durable top-K (this also sweeps
            # away positional v3 leftovers, whose "hash" never matches) ---
            pruned = 0
            try:
                for name in os.listdir(cache_dir):
                    if not name.startswith("entry_"):
                        continue
                    if name.endswith(".safetensors"):
                        h = name[len("entry_") : -len(".safetensors")]
                    elif name.endswith("_tokens.bin"):
                        h = name[len("entry_") : -len("_tokens.bin")]
                    else:
                        continue
                    if h in durable_hashes:
                        continue
                    try:
                        os.remove(os.path.join(cache_dir, name))
                        pruned += 1
                    except OSError as e:
                        logger.warning(f"[cache_persist] failed to prune {name}: {e}")
            except OSError as e:
                logger.warning(f"[cache_persist] prune scan failed: {e}")

            # --- index, written oldest -> newest so load_from_disk rebuilds
            # the OrderedDict with recency intact (MRU last) ---
            index = {
                "version": _CACHE_PERSIST_VERSION,
                "model_fingerprint": self._model_fingerprint,
                "num_entries": len(durable),
                "total_memory_bytes": sum(mem for _, _, mem in durable),
                "entries": [
                    {
                        "hash": h,
                        "num_tokens": len(tokens_key),
                        "memory_bytes": memory_bytes,
                    }
                    for h, tokens_key, memory_bytes in reversed(durable)
                ],
            }

            # Atomic: a crash (or a full disk) mid-write must never leave a
            # half-written index.json that poisons the next boot. The temp name
            # is unique (mkstemp), never a fixed index.json.tmp that a second
            # writer would append its own bytes into.
            index_path = os.path.join(cache_dir, "index.json")
            fd, tmp_path = tempfile.mkstemp(
                dir=cache_dir, prefix=".index.", suffix=".json"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(index, f, indent=2)
                    # The index is the SINGLE point of total loss: a panic that
                    # commits the rename ahead of the data blocks leaves a
                    # truncated index, and the next boot then discards every
                    # persisted prefix. Entry files are deliberately NOT
                    # fsynced — losing one to power loss is self-healing, the
                    # next flush simply re-saves it.
                    f.flush()
                    os.fsync(f.fileno())
                os.rename(tmp_path, index_path)
            except BaseException:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                raise

            dt = _time.monotonic() - t0
            logger.info(
                f"[cache_persist] FLUSHED {len(saved_hashes)} new "
                f"({len(durable)} durable, {pruned} files pruned) "
                f"to {cache_dir} in {dt:.1f}s "
                f"({index['total_memory_bytes'] / _BYTES_PER_MB:.0f}MB on disk)"
            )
            return len(saved_hashes), len(durable)

    def save_to_disk(self, cache_dir: str) -> bool:
        """Persist the cache at shutdown. Thin delegate to flush_to_disk().

        Keeps the historical contract: True iff at least one entry is durable
        on disk for this cache after the call, False when there was nothing to
        save (or when mlx_lm is missing and nothing could be written). The
        answer comes from the flush itself — re-reading index.json here would
        read it outside the persistence lock, and would report a *previous*
        run's index as this call's success.
        """
        if not self._entries:
            logger.info("[cache_persist] nothing to save (0 entries)")
            return False

        _newly_saved, durable_count = self._flush_to_disk(cache_dir)
        return durable_count > 0

    def load_from_disk(self, cache_dir: str) -> int:
        """Load cache entries from disk.

        Entries are loaded in index order (oldest -> newest) so the rebuilt
        OrderedDict carries the recency the flush observed: the MRU prefix
        ends up last and is therefore the last thing evicted.

        Returns the number of entries successfully loaded.
        """
        import json
        import os
        import time as _time

        index_path = os.path.join(cache_dir, "index.json")
        if not os.path.exists(index_path):
            logger.info(f"[cache_persist] no index at {index_path}, nothing to load")
            return 0

        t0 = _time.monotonic()

        try:
            from mlx_lm.models.cache import load_prompt_cache
        except ImportError:
            logger.warning("[cache_persist] mlx_lm not available, cannot load")
            return 0

        with open(index_path) as f:
            index = json.load(f)

        version = index.get("version", 1)
        if version != _CACHE_PERSIST_VERSION:
            logger.warning(
                f"[cache_persist] version mismatch: disk={version} "
                f"current={_CACHE_PERSIST_VERSION}, discarding stale cache"
            )
            return 0

        disk_fp = index.get("model_fingerprint", "")
        if disk_fp and disk_fp != self._model_fingerprint:
            logger.warning(
                f"[cache_persist] model fingerprint mismatch: "
                f"disk={disk_fp} current={self._model_fingerprint}, "
                f"discarding incompatible cache"
            )
            return 0

        loaded = 0
        for entry_meta in index.get("entries", []):
            h = entry_meta.get("hash")
            if not h:
                logger.warning("[cache_persist] index entry has no hash, skipping")
                continue
            entry_path = os.path.join(cache_dir, f"entry_{h}.safetensors")
            tokens_path = os.path.join(cache_dir, f"entry_{h}_tokens.bin")

            if not os.path.exists(entry_path) or not os.path.exists(tokens_path):
                logger.warning(f"[cache_persist] missing files for entry {h}, skipping")
                continue

            try:
                # Load tokens from binary
                import array as _array

                arr = _array.array("i")
                with open(tokens_path, "rb") as f:
                    arr.fromfile(f, entry_meta["num_tokens"])
                tokens = list(arr)

                # Load KV cache
                cache = load_prompt_cache(entry_path)

                # Estimate memory
                memory = estimate_kv_cache_memory(cache)

                tokens_key = tuple(tokens)
                entry = _CacheEntry(
                    tokens=tokens_key,
                    cache=cache,
                    memory_bytes=memory,
                )
                # The fit check reads the memory counters under the lock and
                # admits the entry in the same critical section: a concurrent
                # store() moves both, and an unlocked read can admit an entry
                # that pushes the cache past its limit.
                with self._lock:
                    used = self._current_memory
                    fits = used + memory <= self._max_memory
                    if fits:
                        self._entries[tokens_key] = entry
                        self._current_memory += memory
                        bisect.insort(self._sorted_keys, tokens_key)

                if not fits:
                    logger.info(
                        f"[cache_persist] entry {h} would exceed memory limit "
                        f"({(used + memory) / _BYTES_PER_MB:.0f}MB > "
                        f"{self._max_memory / _BYTES_PER_MB:.0f}MB), skipping it"
                    )
                    # Skip, don't stop. The index is ordered oldest -> newest,
                    # so breaking here would throw away the HOTTEST prefixes
                    # because one older entry happened to be huge. Same packing
                    # rule the flush selection uses.
                    continue

                loaded += 1

                logger.info(
                    f"[cache_persist] loaded entry {h}: "
                    f"{len(tokens)} tokens, "
                    f"{memory / _BYTES_PER_MB:.1f}MB KV"
                )

            except Exception as e:
                # A durable entry we cannot read is dead weight: it stays in
                # the index forever, is never loadable, and the flush's
                # existence check treats it as already-saved so it is never
                # rewritten. Delete both files and let the next flush re-save
                # the prefix from memory.
                logger.warning(
                    f"[cache_persist] failed to load entry {h}: {e}; deleting "
                    f"its files so the next flush re-saves it"
                )
                for path in (entry_path, tokens_path):
                    try:
                        os.remove(path)
                    except OSError as rm_err:
                        logger.warning(
                            f"[cache_persist] could not delete {path}: {rm_err}"
                        )

        with self._lock:
            total_memory = self._current_memory
            self._stats.entry_count = len(self._entries)
            self._stats.current_memory_bytes = total_memory

        dt = _time.monotonic() - t0
        logger.info(
            f"[cache_persist] LOADED {loaded} entries from {cache_dir} "
            f"in {dt:.1f}s ({total_memory / _BYTES_PER_MB:.0f}MB total)"
        )
        return loaded
