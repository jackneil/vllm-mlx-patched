# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the mlx_lm 0.31.2+ prefill cache-save hooks.

_install_prefill_cache_hooks replaces the removed Batch-era
_install_chunked_prefill monkey-patch: mlx_lm now chunks prefills natively,
so the only remaining job is capturing cache state (a) at the
prompt->generation transition (prompt-only save — the entry that turns an
identical follow-up prompt into an EXACT prefix-cache hit) and (b) after
each prefill chunk of a single-sequence batch (mid-prefill/boundary save).

These tests exercise the hook installer against pure-Python fakes of the
split batch objects, so they run in the no-MLX CI job.
"""

from __future__ import annotations

import pytest

from vllm_mlx.scheduler import _boundary_segments, _install_prefill_cache_hooks


class FakeGenerationBatch:
    def __init__(self, uids=None):
        self.uids = list(uids or [])
        self.extend_calls: list = []
        self.extracted: list[int] = []

    def extract_cache(self, idx):
        self.extracted.append(idx)
        return f"cache-{self.uids[idx]}"

    def extend(self, batch):
        self.extend_calls.append(batch)
        self.uids.extend(batch.uids)


class FakePromptBatch:
    def __init__(self, uids, tokens=None, prompt_cache=None):
        self.uids = list(uids)
        self.tokens = tokens if tokens is not None else [[] for _ in uids]
        self.prompt_cache = prompt_cache if prompt_cache is not None else ["layer0"]
        self.prompt_calls: list = []

    def prompt(self, tokens):
        self.prompt_calls.append(tokens)
        if not tokens:
            return
        for sti, ti in zip(self.tokens, tokens):
            sti += ti


class FakeBatchGenerator:
    def __init__(self, prompt_batch, generation_batch):
        self._prompt_batch = prompt_batch
        self._generation_batch = generation_batch


# ---------------------------------------------------------------------------
# prompt_cache_save via _generation_batch.extend
# ---------------------------------------------------------------------------


def test_prompt_cache_save_fires_per_uid_on_extend():
    gen = FakeGenerationBatch()
    bg = FakeBatchGenerator(FakePromptBatch(uids=[]), gen)
    saved = []
    _install_prefill_cache_hooks(
        bg, prompt_cache_save=lambda u, c: saved.append((u, c))
    )

    incoming = FakeGenerationBatch(uids=[7, 9])
    bg._generation_batch.extend(incoming)

    assert saved == [(7, "cache-7"), (9, "cache-9")]
    # original extend still ran (merge preserved)
    assert gen.extend_calls == [incoming]
    assert gen.uids == [7, 9]


def test_prompt_cache_save_error_is_swallowed():
    gen = FakeGenerationBatch()
    bg = FakeBatchGenerator(FakePromptBatch(uids=[]), gen)

    def boom(uid, cache):
        raise RuntimeError("store failed")

    _install_prefill_cache_hooks(bg, prompt_cache_save=boom)
    incoming = FakeGenerationBatch(uids=[1])
    bg._generation_batch.extend(incoming)  # must not raise
    assert gen.uids == [1]


def test_no_hooks_when_callbacks_none():
    prompt = FakePromptBatch(uids=[1])
    gen = FakeGenerationBatch()
    orig_prompt, orig_extend = prompt.prompt, gen.extend
    bg = FakeBatchGenerator(prompt, gen)
    _install_prefill_cache_hooks(bg)
    assert bg._prompt_batch.prompt == orig_prompt
    assert bg._generation_batch.extend == orig_extend


# ---------------------------------------------------------------------------
# mid_prefill_save via _prompt_batch.prompt
# ---------------------------------------------------------------------------


def test_mid_prefill_save_fires_after_single_sequence_chunk():
    prompt = FakePromptBatch(uids=[3], tokens=[[10, 11]])
    bg = FakeBatchGenerator(prompt, FakeGenerationBatch())
    saves = []
    _install_prefill_cache_hooks(
        bg, mid_prefill_save=lambda u, n, c: saves.append((u, n, c))
    )

    bg._prompt_batch.prompt([[12, 13, 14]])

    # processed count = tokens accumulated in the batch AFTER the chunk
    assert saves == [(3, 5, prompt.prompt_cache)]
    # underlying prompt() still ran
    assert prompt.prompt_calls == [[[12, 13, 14]]]
    assert prompt.tokens[0] == [10, 11, 12, 13, 14]


def test_mid_prefill_save_skipped_for_multi_sequence_batch():
    prompt = FakePromptBatch(uids=[3, 4])
    bg = FakeBatchGenerator(prompt, FakeGenerationBatch())
    saves = []
    _install_prefill_cache_hooks(
        bg, mid_prefill_save=lambda u, n, c: saves.append((u, n, c))
    )
    bg._prompt_batch.prompt([[1, 2], [3, 4]])
    assert saves == []
    assert prompt.prompt_calls == [[[1, 2], [3, 4]]]


def test_mid_prefill_save_skipped_for_empty_chunk():
    prompt = FakePromptBatch(uids=[3])
    bg = FakeBatchGenerator(prompt, FakeGenerationBatch())
    saves = []
    _install_prefill_cache_hooks(
        bg, mid_prefill_save=lambda u, n, c: saves.append((u, n, c))
    )
    bg._prompt_batch.prompt([])
    bg._prompt_batch.prompt([[]])
    assert saves == []


def test_mid_prefill_save_error_is_swallowed():
    prompt = FakePromptBatch(uids=[3])
    bg = FakeBatchGenerator(prompt, FakeGenerationBatch())

    def boom(uid, n, cache):
        raise RuntimeError("save failed")

    _install_prefill_cache_hooks(bg, mid_prefill_save=boom)
    bg._prompt_batch.prompt([[1, 2]])  # must not raise
    assert prompt.tokens[0] == [1, 2]


def test_both_hooks_compose():
    prompt = FakePromptBatch(uids=[5], tokens=[[]])
    gen = FakeGenerationBatch()
    bg = FakeBatchGenerator(prompt, gen)
    mids, proms = [], []
    _install_prefill_cache_hooks(
        bg,
        mid_prefill_save=lambda u, n, c: mids.append((u, n)),
        prompt_cache_save=lambda u, c: proms.append((u, c)),
    )
    bg._prompt_batch.prompt([[1, 2, 3]])
    bg._generation_batch.extend(FakeGenerationBatch(uids=[5]))
    assert mids == [(5, 3)]
    assert proms == [(5, "cache-5")]


# ---------------------------------------------------------------------------
# _boundary_segments — prefix_boundary segment splitting for insert_segments
# ---------------------------------------------------------------------------


def test_boundary_segments_splits_at_boundary():
    toks = list(range(10))
    assert _boundary_segments(4, 0, toks) == [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9]]


def test_boundary_segments_adjusts_for_cached_tokens():
    toks = list(range(6))  # remaining tokens after a 4-token cache hit
    # absolute boundary 7, 4 already cached -> split at 3 of the remainder
    assert _boundary_segments(7, 4, toks) == [[0, 1, 2], [3, 4, 5]]


@pytest.mark.parametrize(
    "pb,cached",
    [
        (0, 0),  # no boundary
        (4, 4),  # boundary fully covered by cache
        (2, 4),  # boundary behind the cache
        (10, 0),  # boundary at/after the end
        (12, 0),  # boundary past the end
        (None, None),  # unset attributes
    ],
)
def test_boundary_segments_degrades_to_single_segment(pb, cached):
    toks = list(range(10))
    assert _boundary_segments(pb, cached, toks) == [toks]
