# SPDX-License-Identifier: Apache-2.0
"""Unit tests for native MTP speculative decoding (_install_mtp_decode).

The decode loop must satisfy three properties, each covered here against
pure-Python fakes of the mlx_lm split-batch API:

  * accept emits TWO tokens per backbone pass, reject emits ONE;
  * a rejected draft is rolled back out of every cache;
  * nothing is emitted before the backbone has consumed it, and the stock
    ``_next_tokens`` contract survives every round so a second sequence can
    join mid-stream and stock batching resumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import pytest

from vllm_mlx.patches.qwen3_5_mtp import clear_rollback, rollback_draft
from vllm_mlx.scheduler import _install_mtp_decode

VOCAB = 16
HIDDEN = 4


class FakeKVCache:
    """Trimmable cache standing in for an attention layer."""

    def __init__(self):
        self.offset = 0
        self.trims = 0
        self.rollback_state = None

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.trims += n
        self.offset -= n


class FakeArraysCache:
    """Non-trimmable recurrent cache standing in for a GatedDeltaNet layer."""

    def __init__(self):
        self.cache = [mx.zeros((1, 2)), mx.zeros((1, 2))]
        self.rollback_state = None
        self.restores = 0

    def __getitem__(self, i):
        return self.cache[i]

    def __setitem__(self, i, v):
        self.cache[i] = v
        self.restores += 1

    def is_trimmable(self):
        return False


class FakeModel:
    """Deterministic model: `nxt[token]` is the token predicted after `token`."""

    def __init__(self, nxt: dict[int, int], drafts: List[int]):
        self.nxt = nxt
        self.drafts = list(drafts)
        self.calls: List[int] = []  # sequence length of each backbone call
        self.n_confirmed_seen: List[int] = []
        self.mtp_positions: List[list] = []

    def _logits_for(self, toks):
        rows = []
        for t in toks:
            row = mx.full((VOCAB,), -10.0)
            row[self.nxt.get(int(t), 0)] = 10.0
            rows.append(row)
        return mx.stack(rows)[None]

    def __call__(self, inputs, cache=None, return_hidden=False, n_confirmed=0):
        toks = inputs.tolist()[0]
        self.calls.append(len(toks))
        self.n_confirmed_seen.append(n_confirmed)
        logits = self._logits_for(toks)
        if 0 < n_confirmed < len(toks):
            # Mirror the real split forward: stash a rollback snapshot.
            for c in cache:
                if isinstance(c, FakeArraysCache):
                    c.rollback_state = (mx.zeros((1, 2)), mx.zeros((1, 2)))
        for c in cache:
            if isinstance(c, FakeKVCache):
                c.offset += len(toks)
        hidden = mx.zeros((1, len(toks), HIDDEN))
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(self, hidden, next_token_ids, mtp_cache=None):
        self.mtp_positions.append(next_token_ids.tolist()[0])
        d = self.drafts.pop(0) if self.drafts else 0
        row = mx.full((VOCAB,), -10.0)
        row[d] = 10.0
        return row[None][None]

    def make_mtp_cache(self):
        return [FakeKVCache()]


class FakeStateMachine:
    def make_state(self):
        return None

    def match(self, state, token):
        return state, None, None


class FakeGenerationBatch:
    @dataclass
    class Response:
        uid: int
        token: int
        logprobs: Any
        finish_reason: Optional[str]
        current_state: Optional[str]
        match_sequence: Optional[List[int]]
        prompt_cache: Optional[List[Any]]
        all_tokens: Optional[List[int]]

    def __init__(self, uids, first_token, max_tokens=100, n_layers=2):
        self.uids = list(uids)
        self.tokens = [[] for _ in uids]
        self.prompt_cache = [FakeKVCache(), FakeArraysCache()][:n_layers]
        self.samplers = [None] * len(uids)
        self.fallback_sampler = lambda x: mx.argmax(x, axis=-1)
        self.logits_processors = [[] for _ in uids]
        self.max_tokens = [max_tokens] * len(uids)
        self.state_machines = [FakeStateMachine() for _ in uids]
        self._matcher_states = [None] * len(uids)
        self._num_tokens = [0] * len(uids)
        self._next_tokens = mx.array([first_token], mx.uint32)
        self._next_logprobs = [mx.zeros((VOCAB,))]
        self.stock_next_calls = 0
        self.filtered = None

    def extract_cache(self, idx):
        return ["cache"]

    def filter(self, keep):
        self.filtered = list(keep)
        if not keep:
            self.uids = []

    def next(self):
        self.stock_next_calls += 1
        return []


class FakeBatchGenerator:
    def __init__(self, gen):
        self._generation_batch = gen


def _emitted(responses):
    return [r.token for r in responses]


# ---------------------------------------------------------------------------
# accept / reject token accounting
# ---------------------------------------------------------------------------


def test_cold_start_emits_one_token_and_seeds_a_draft():
    # 1 -> 2 -> 3 ...; first draft guesses 3
    gen = FakeGenerationBatch([7], first_token=1)
    model = FakeModel({1: 2, 2: 3, 3: 4}, drafts=[3])
    _install_mtp_decode(FakeBatchGenerator(gen), model)

    r = gen.next()
    assert _emitted(r) == [1]
    assert model.calls == [1]  # single-token pass, no draft to verify yet
    # the produced-but-unfed token is exposed via the stock contract
    assert gen._next_tokens.tolist() == [2]
    assert gen.tokens[0] == [1]  # only the fed token is recorded


def test_accepted_draft_emits_two_tokens_for_one_backbone_pass():
    gen = FakeGenerationBatch([7], first_token=1)
    # backbone: 1->2, 2->3, 3->4. draft after 2 is 3 => matches backbone => accept
    model = FakeModel({1: 2, 2: 3, 3: 4}, drafts=[3, 5])
    _install_mtp_decode(FakeBatchGenerator(gen), model)

    gen.next()  # cold start emits [1]
    r = gen.next()

    assert _emitted(r) == [2, 3]  # confirmed + accepted draft
    assert model.calls == [1, 2]  # second round verified 2 tokens in one pass
    assert model.n_confirmed_seen[-1] == 1
    assert gen.tokens[0] == [1, 2, 3]  # everything emitted is in the cache
    assert gen._next_tokens.tolist() == [4]  # bonus token, produced not yet fed


def test_rejected_draft_emits_one_token_and_rolls_back():
    gen = FakeGenerationBatch([7], first_token=1)
    # draft after 2 is 9, but the backbone says 3 => reject
    model = FakeModel({1: 2, 2: 3, 3: 4}, drafts=[9, 5])
    _install_mtp_decode(FakeBatchGenerator(gen), model)

    gen.next()
    kv = gen.prompt_cache[0]
    arrays = gen.prompt_cache[1]
    r = gen.next()

    assert _emitted(r) == [2]  # only the confirmed token
    assert gen.tokens[0] == [1, 2]  # the rejected draft is NOT recorded
    assert kv.trims == 1  # attention cache trimmed
    assert arrays.restores == 2  # recurrent state restored from the snapshot
    assert arrays.rollback_state is None
    assert gen._next_tokens.tolist() == [3]  # backbone's real next token


def test_accepted_draft_clears_the_rollback_snapshot():
    gen = FakeGenerationBatch([7], first_token=1)
    model = FakeModel({1: 2, 2: 3, 3: 4}, drafts=[3, 5])
    _install_mtp_decode(FakeBatchGenerator(gen), model)
    gen.next()
    gen.next()
    kv, arrays = gen.prompt_cache
    assert arrays.rollback_state is None  # cleared, not left pinning memory
    assert kv.trims == 0  # nothing rolled back on the accept path


# ---------------------------------------------------------------------------
# emitted-implies-cached, and the handoff back to stock batching
# ---------------------------------------------------------------------------


def test_every_emitted_token_is_in_the_cache_across_rounds():
    gen = FakeGenerationBatch([7], first_token=1)
    nxt = {i: i + 1 for i in range(VOCAB - 1)}
    model = FakeModel(nxt, drafts=[2, 4, 99, 7, 9])  # mix of accepts and a reject
    _install_mtp_decode(FakeBatchGenerator(gen), model)

    emitted = []
    for _ in range(5):
        emitted.extend(_emitted(gen.next()))

    # The cache (gen.tokens) must be a prefix-superset of what was emitted:
    # a token is never streamed to a client before the model consumed it.
    assert emitted == gen.tokens[0][: len(emitted)]
    assert len(set(emitted)) == len(emitted)  # no duplicates
    assert emitted == sorted(emitted)  # strictly increasing chain, no gaps


def test_multi_sequence_batch_delegates_to_stock_next():
    gen = FakeGenerationBatch([7, 8], first_token=1)
    model = FakeModel({1: 2}, drafts=[3])
    _install_mtp_decode(FakeBatchGenerator(gen), model)

    gen.next()
    assert gen.stock_next_calls == 1  # MTP declined; stock stepping ran
    assert model.calls == []  # MTP never touched the model


def test_handoff_preserves_next_tokens_when_a_sequence_joins():
    gen = FakeGenerationBatch([7], first_token=1)
    model = FakeModel({1: 2, 2: 3, 3: 4}, drafts=[3, 5])
    _install_mtp_decode(FakeBatchGenerator(gen), model)
    gen.next()
    unfed = gen._next_tokens.tolist()

    # A second sequence joins: MTP must step aside without disturbing the
    # produced-but-unfed token that stock stepping is about to consume.
    gen.uids.append(8)
    gen.next()

    assert gen.stock_next_calls == 1
    assert gen._next_tokens.tolist() == unfed


def test_logits_processors_disable_mtp():
    """A per-request processor (e.g. thinking budget) must see every token."""
    gen = FakeGenerationBatch([7], first_token=1)
    gen.logits_processors = [[lambda toks, lg: lg]]
    model = FakeModel({1: 2}, drafts=[3])
    _install_mtp_decode(FakeBatchGenerator(gen), model)

    gen.next()
    assert gen.stock_next_calls == 1
    assert model.calls == []


def test_stats_track_accepts_and_rejects():
    gen = FakeGenerationBatch([7], first_token=1)
    model = FakeModel({1: 2, 2: 3, 3: 4, 4: 5}, drafts=[3, 99, 6])
    st = _install_mtp_decode(FakeBatchGenerator(gen), model)
    gen.next()  # cold
    gen.next()  # accept (draft 3 == backbone 3)
    gen.next()  # reject (draft 99 != backbone next)
    assert st["accepted"] == 1
    assert st["rejected"] == 1
    assert st["rounds"] == 3


def test_finish_stops_emitting_the_second_token():
    """max_tokens reached on the confirmed token must not leak the draft."""
    gen = FakeGenerationBatch([7], first_token=1, max_tokens=2)
    model = FakeModel({1: 2, 2: 3, 3: 4}, drafts=[3, 5])
    _install_mtp_decode(FakeBatchGenerator(gen), model)

    gen.next()  # emits [1], num_tokens=1
    r = gen.next()  # would emit [2, 3] but hits max_tokens at 2

    assert _emitted(r) == [2]
    assert r[-1].finish_reason == "length"
    assert r[-1].prompt_cache is not None  # cache handed back for reuse
    assert gen.filtered == []  # batch cleared


# ---------------------------------------------------------------------------
# rollback helpers in isolation
# ---------------------------------------------------------------------------


def test_rollback_draft_handles_mixed_cache_types():
    kv, arrays = FakeKVCache(), FakeArraysCache()
    arrays.rollback_state = (mx.ones((1, 2)), mx.ones((1, 2)))
    rollback_draft([kv, arrays])
    assert kv.trims == 1
    assert arrays.rollback_state is None
    assert mx.array_equal(arrays[0], mx.ones((1, 2)))


def test_rollback_draft_without_snapshot_trims_only():
    kv = FakeKVCache()
    rollback_draft([kv])
    assert kv.trims == 1


def test_clear_rollback_drops_snapshots():
    arrays = FakeArraysCache()
    arrays.rollback_state = (mx.ones((1, 2)), mx.ones((1, 2)))
    clear_rollback([arrays])
    assert arrays.rollback_state is None
    assert arrays.restores == 0  # cleared without restoring


@pytest.mark.parametrize("n_layers", [1, 2])
def test_install_returns_stats_dict(n_layers):
    gen = FakeGenerationBatch([7], first_token=1, n_layers=n_layers)
    st = _install_mtp_decode(FakeBatchGenerator(gen), FakeModel({1: 2}, [3]))
    assert st == {
        "y": None,
        "y_lp": None,
        "draft": None,
        "mtp_cache": None,
        "accepted": 0,
        "rejected": 0,
        "rounds": 0,
    }
