"""The scheduler's prefix cache: longest cached prefix wins, least recently used goes first, bytes stay in budget."""

import torch

from tensorfold.families.qwen3_5.cuda.sched import PrefixCache


def _state(rows: int) -> list[torch.Tensor]:
    return [torch.zeros(rows, 256, dtype=torch.uint8)]          # rows x 256 bytes


def test_longest_prefix_wins():
    c = PrefixCache()
    c.add([1, 2], _state(1), None)
    c.add([1, 2, 3], _state(1), None)
    assert c.best([1, 2, 3, 4])[0] == [1, 2, 3]
    assert c.best([1, 2, 9])[0] == [1, 2]
    assert c.best([1, 2, 3])[0] == [1, 2]                      # a whole prompt is never a hit: one row must run
    assert c.best([7]) is None


def test_bytes_stay_in_budget_least_recently_used_first():
    c = PrefixCache(limit=16, budget=3 * 1024)                 # room for three 1 KiB entries
    for t in range(3):
        c.add([t, 0], _state(4), None)
    c.best([0, 0, 5])                                           # [0, 0] is now the most recently used
    c.add([3, 0], _state(4), None)
    assert [e[0] for e in c.entries] == [[2, 0], [0, 0], [3, 0]]
    assert c.bytes() == 3 * 1024
    c.add([4, 0], _state(8), None)                              # 2 KiB: evicts two
    assert [e[0] for e in c.entries] == [[3, 0], [4, 0]] and c.bytes() == 3 * 1024


def test_an_entry_larger_than_the_budget_is_not_kept():
    c = PrefixCache(budget=1024)
    c.add([1, 0], _state(4), None)
    c.add([2, 0], _state(5), None)
    assert [e[0] for e in c.entries] == [[1, 0]]


def test_entry_limit_still_applies():
    c = PrefixCache(limit=2)
    for t in range(3):
        c.add([t, 0], _state(1), None)
    assert [e[0] for e in c.entries] == [[1, 0], [2, 0]]
