"""Qwen3.6-35B-A3B's prompt cache without a GPU: keys, longest prefix, byte budget, eviction order, checkpoints."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen3_5_moe.cuda import prefix_cache as P  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.prefix_cache import BACKOFF, MIN_POINT, PrefixCache  # noqa: E402

OPEN, SYSTEM = 900, 901
MARKS = (OPEN, SYSTEM)
KIB = 1024


def _snap(kib: int) -> dict:
    """A stand-in ``State.snapshot(rows=True)`` of ``kib`` KiB (its ints and a missing tail hold nothing)."""

    return {"pos": 3, "rec": torch.zeros(kib * KIB, dtype=torch.uint8), "conv": torch.zeros(0),
            "mtp": (2, 2, None), "kc": torch.zeros(0), "vc": torch.zeros(0)}


def _text(n: int, seed: int) -> list[int]:
    return np.random.default_rng(seed).integers(0, 800, n).tolist()


def _chat(system: list[int], user: list[int]) -> list[int]:
    return [OPEN, SYSTEM] + system + [OPEN, 1] + user + [OPEN, 2]


def test_the_longest_kept_prefix_the_prompt_extends_wins():
    c = PrefixCache(64 * KIB)
    a = _text(700, 1)
    assert c.add(a[:600], _snap(1)) and c.add(a[:650], _snap(1)) and c.add(_text(600, 2), _snap(1))
    assert c.find(a).tokens.tolist() == a[:650]
    assert c.find(a[:640] + [7]).tokens.tolist() == a[:600]
    assert c.find(a[:650]).tokens.tolist() == a[:600]        # a whole prompt is never a hit: one row must run
    assert c.find(a[:600]) is None and c.find([5, 6]) is None
    assert c.get(a[:650]) is not None and c.get(a[:649]) is None
    assert c.key(a[:600]) == (600, c.key(np.asarray(a[:600], dtype=np.int64))[1])    # the tokens, not their type
    assert len(c) == 3 and c.nbytes == 3 * KIB


def test_a_hash_collision_is_not_a_hit():
    c = PrefixCache(64 * KIB)
    a, b = _text(600, 3), _text(600, 4)
    c.add(a, _snap(1))
    k = c.key(a)
    e = c.entries.pop(k)
    c.entries[c.key(b)] = e                          # b's key now holds a's tokens: compared in full, not a hit
    assert c.find(b + [1]) is None and c.get(b) is None


def test_the_budget_drops_the_least_recently_used_shared_prefix_before_a_system_block():
    c = PrefixCache(3 * KIB)
    t = [_text(600, s) for s in range(10, 16)]
    assert c.add(t[0], _snap(1), pinned=True) and c.add(t[1], _snap(1)) and c.add(t[2], _snap(1))
    c.use(c.find(t[1] + [0]))                        # t1 is now the most recently used
    assert c.add(t[3], _snap(1))                     # over budget: t2 goes (t0 is pinned, t1 was just used)
    assert [e.tokens.tolist() for e in c.entries.values()] == [t[0], t[1], t[3]] and c.nbytes == 3 * KIB
    assert c.add(t[4], _snap(2), pinned=True)        # 2 KiB: two shared prefixes go, the older system block stays
    assert [e.tokens.tolist() for e in c.entries.values()] == [t[0], t[4]] and c.evictions == 3
    assert c.add(t[5], _snap(2), pinned=True)        # only system blocks left: the least recently used goes
    assert [e.tokens.tolist() for e in c.entries.values()] == [t[5]] and c.nbytes == 2 * KIB
    assert not c.add(t[0], _snap(4)) and c.skipped == 1 and len(c) == 1      # larger than the budget: not kept
    assert c.add(t[5], _snap(1)) and c.entries[c.key(t[5])].pinned and c.nbytes == KIB    # a re-add stays pinned
    c.clear()
    assert len(c) == 0 and c.nbytes == 0 and not c.recent


def test_system_blocks_end_at_the_next_message():
    ids = np.asarray(_chat(_text(600, 5), _text(40, 6)), dtype=np.int32)
    assert P.block_end(ids, MARKS) == 602
    assert P.block_end(ids, None) == 0
    assert P.block_end(ids[2:], MARKS) == 0                      # no system message first
    assert P.block_end(np.asarray([OPEN, SYSTEM] + _text(9, 7)), MARKS) == 0   # nothing after it


def test_a_prompt_keeps_its_system_block_once():
    c = PrefixCache(64 * KIB)
    system = _text(700, 8)
    p1 = _chat(system, _text(300, 9))
    assert c.points(p1, 0, MARKS) == [(702, True)]
    assert c.points(p1, 702, MARKS) == [] and c.points(p1, 0, None) == []
    small = _chat(system[:MIN_POINT - 3], _text(300, 9))            # a block under MIN_POINT is not worth it
    assert c.points(small, 0, MARKS) == [] and c.points(_chat(system[:MIN_POINT - 2], [1]), 0, MARKS) == [(512, True)]
    c.add(p1[:702], _snap(1), pinned=True)
    assert c.points(_chat(system, _text(300, 10)), 0, MARKS) == []   # already kept
    assert PrefixCache(0).points(p1, 0, MARKS) == []                # 0 bytes: the prompt cache is off


def test_a_prefix_shared_with_a_recent_prompt_is_kept_away_from_other_points():
    c = PrefixCache(64 * KIB)
    shared = _text(900, 11)
    p1, p2 = shared + [801] + _text(200, 12), shared + [802] + _text(200, 13)
    assert c.points(p2, 0, None) == []                               # nothing seen yet
    c.seen(p1)
    assert c.points(p2, 0, None) == [(900 - BACKOFF, False)]
    assert c.points(p2, 900 - BACKOFF - MIN_POINT + 1, None) == []   # too close to where the prefill resumes
    assert c.points(shared[:600], 0, None) == [(600 - BACKOFF, False)]   # all shared: its last tokens still run
    # a chat prompt: the shared prefix ends just past the system block, which is the point
    system = _text(700, 14)
    c.seen(_chat(system, _text(300, 15)))
    assert c.points(_chat(system, _text(300, 16)), 0, MARKS) == [(702, True)]
    # a system block that shares only its start with recent ones: both points, far enough apart
    c.seen(_chat(system + _text(900, 17), [3]))
    assert c.points(_chat(system + _text(900, 18), [4]), 0, MARKS) == [(702 - BACKOFF, False), (1602, True)]


def test_points_larger_than_the_budget_are_skipped():
    c = PrefixCache(10 * KIB, cost=lambda n: n * 16)                  # 640 tokens fit
    assert c.points(_chat(_text(600, 19), [5]), 0, MARKS) == [(602, True)]
    assert c.points(_chat(_text(700, 19), [5]), 0, MARKS) == [] and c.skipped == 1


def test_snapshot_bytes_counts_every_tensor():
    snap = {"pos": 9, "rec": torch.zeros(4, dtype=torch.float32), "conv": torch.zeros(3, dtype=torch.bfloat16),
            "mtp": (8, 8, torch.zeros(5, dtype=torch.bfloat16)), "kc": torch.zeros(2, 7, dtype=torch.bfloat16),
            "vc": torch.zeros(2, 7, dtype=torch.bfloat16)}
    assert P.snapshot_bytes(snap) == 16 + 6 + 10 + 28 + 28
    assert P.snapshot_bytes({**snap, "mtp": (0, -1, None)}) == 16 + 6 + 28 + 28
