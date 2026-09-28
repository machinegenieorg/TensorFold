"""Qwen3.6-35B-A3B's prompt cache on CUDA: sequence states at shared prompt prefixes, kept across requests.

A prompt that starts with a cached prefix restores that state (GDN recurrent and conv state, the attention and draft
head cache rows below it) and prefills only the rest. Every kernel gives a row the same bits whatever window or chunk
it runs in, so the resumed prompt ends in the state, logits and tokens a fresh prefill gives.

Where a prefill keeps a state (``points``): the end of the prompt's leading system message (pinned: every prompt's
system block), and a prefix it shares with a recent prompt (instructions outside a system message). Entries are keyed
by length and a hash of their tokens (compared in full on a hit) and held within a byte budget: the least recently
used shared prefix goes first, then the least recently used system block.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

MIN_POINT = 512     # the shortest prefix worth keeping (each entry holds the ~61 MiB GDN state besides its rows)
BACKOFF = 8         # a shared prefix ends this many tokens early: its last tokens may merge with what follows them
RECENT = 8          # prompts remembered for finding shared prefixes


def _ids(tokens: Sequence[int] | np.ndarray) -> np.ndarray:
    return np.asarray(tokens, dtype=np.int32)


def snapshot_bytes(snap: dict) -> int:
    """The bytes a snapshot's tensors hold (``State.snapshot``: the GDN state, conv windows, tail, cache rows)."""

    parts = [v for k, v in snap.items() if k != "mtp"] + list(snap.get("mtp", ()))
    return sum(int(v.nbytes) for v in parts if hasattr(v, "nbytes") and not isinstance(v, (int, float)))


def block_end(ids: np.ndarray, marks: tuple[int, int] | None) -> int:
    """Where a prompt's leading system message ends (the next message's ``<|im_start|>``), or 0 without one.

    ``marks``: the ``<|im_start|>`` and ``system`` ids. The system message (tools included) renders as
    ``<|im_start|>system\\n...<|im_end|>\\n``, and special tokens split the text, so its tokens are the same whatever
    follows it."""

    if marks is None or len(ids) < 3 or ids[0] != marks[0] or ids[1] != marks[1]:
        return 0
    at = np.flatnonzero(ids[2:] == marks[0])
    return int(at[0]) + 2 if len(at) else 0


def common(a: np.ndarray, b: np.ndarray) -> int:
    """The length of the longest common prefix."""

    n = min(len(a), len(b))
    diff = np.flatnonzero(a[:n] != b[:n])
    return int(diff[0]) if len(diff) else n


@dataclass
class Entry:
    tokens: np.ndarray        # the prefix, int32
    snap: dict                # ``State.snapshot(rows=True)`` after it
    nbytes: int
    pinned: bool              # a system block: shared prefixes are dropped before it
    hits: int = 0


class PrefixCache:
    """States at prompt prefixes within ``budget`` bytes; ``cost(n)`` estimates an entry of ``n`` tokens."""

    def __init__(self, budget: int, cost: Callable[[int], int] | None = None) -> None:
        if budget < 0:
            raise ValueError(f"a prompt cache of {budget} bytes")
        self.budget = int(budget)
        self.cost = cost
        self.entries: OrderedDict[tuple[int, bytes], Entry] = OrderedDict()    # least recently used first
        self.nbytes = 0
        self.evictions = 0
        self.skipped = 0              # points not kept: larger than the whole budget
        self.recent: deque[np.ndarray] = deque(maxlen=RECENT)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def enabled(self) -> bool:
        return self.budget > 0

    @staticmethod
    def key(tokens: Sequence[int] | np.ndarray) -> tuple[int, bytes]:
        ids = _ids(tokens)
        return len(ids), hashlib.blake2b(ids.tobytes(), digest_size=16).digest()

    @staticmethod
    def _keys(ids: np.ndarray, lengths) -> dict[int, bytes]:
        """The hash of ``ids``' prefix at each of ``lengths``, in one pass over the tokens."""

        h = hashlib.blake2b(digest_size=16)
        out, at = {}, 0
        for n in sorted(set(lengths)):
            h.update(ids[at:n].tobytes())
            at = n
            out[n] = h.copy().digest()
        return out

    def get(self, tokens: Sequence[int] | np.ndarray) -> Entry | None:
        ids = _ids(tokens)
        e = self.entries.get(self.key(ids))
        return e if e is not None and np.array_equal(e.tokens, ids) else None

    def find(self, prompt: Sequence[int] | np.ndarray) -> Entry | None:
        """The longest entry the prompt extends by at least one token (``use`` marks it used)."""

        ids = _ids(prompt)
        lengths = {n for n, _ in self.entries if n < len(ids)}
        if not lengths:
            return None
        digests = self._keys(ids, lengths)
        for n in sorted(lengths, reverse=True):
            e = self.entries.get((n, digests[n]))
            if e is not None and np.array_equal(e.tokens, ids[:n]):
                return e
        return None

    def use(self, e: Entry) -> None:
        self.entries.move_to_end(self.key(e.tokens))
        e.hits += 1

    def add(self, tokens: Sequence[int] | np.ndarray, snap: dict, *, pinned: bool = False) -> bool:
        """Keep ``snap`` for ``tokens``, dropping others to stay in budget; False when it alone does not fit."""

        size = snapshot_bytes(snap)
        if size > self.budget:
            self.skipped += 1
            return False
        ids = _ids(tokens).copy()
        k = self.key(ids)
        old = self.entries.pop(k, None)
        if old is not None:
            self.nbytes -= old.nbytes
            pinned = pinned or old.pinned
        self.entries[k] = Entry(ids, snap, size, pinned)
        self.nbytes += size
        while self.nbytes > self.budget:
            others = [kk for kk in self.entries if kk != k]
            victim = next((kk for kk in others if not self.entries[kk].pinned), others[0])
            self.nbytes -= self.entries.pop(victim).nbytes
            self.evictions += 1
        return True

    def clear(self) -> None:
        self.entries.clear()
        self.recent.clear()
        self.nbytes = 0

    def seen(self, prompt: Sequence[int] | np.ndarray) -> None:
        """A prompt served: later prompts that share a prefix with it keep a state there."""

        self.recent.append(_ids(prompt).copy())

    def points(self, prompt: Sequence[int] | np.ndarray, start: int,
               marks: tuple[int, int] | None) -> list[tuple[int, bool]]:
        """Where a prefill resumed at ``start`` keeps a state: (position, pinned) in order, none already kept.

        The end of the system message (pinned), and the prefix shared with a recent prompt when it ends at least
        ``MIN_POINT`` tokens away from the resumed position and the system block's end. Every point leaves at least
        one prompt token to run, and fits the budget by ``cost``."""

        if not self.enabled:
            return []
        ids = _ids(prompt)
        n = len(ids)
        out: list[tuple[int, bool]] = []
        b = block_end(ids, marks)
        if b >= MIN_POINT and start < b < n and self.get(ids[:b]) is None:
            out.append((b, True))
        m = min(max((common(ids, r) for r in self.recent), default=0) - BACKOFF, n - 1)
        if m - start >= MIN_POINT and (not b or abs(m - b) >= MIN_POINT) and self.get(ids[:m]) is None:
            out.append((m, False))
        fits = [(p, pin) for p, pin in out if self.cost is None or self.cost(p) <= self.budget]
        self.skipped += len(out) - len(fits)
        return sorted(fits)
