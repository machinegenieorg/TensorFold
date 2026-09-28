"""Qwen3.6-35B-A3B's committed sequence state on CUDA: the pool of caches, one sequence's views, a window's rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .forward import Model


class Pool:
    """Committed caches for up to ``seqs`` sequences, allocated once; ``alloc`` hands out a slot as a ``State``."""

    def __init__(self, m: Model, seqs: int, capacity: int, *, mtp_layers: int = 0) -> None:
        c = m.cfg
        dev = m.device
        nl, na = m.n_linear, m.n_attention
        self.capacity = capacity
        self.mtp_layers = mtp_layers
        self.rec = torch.zeros((seqs, 2, nl, c.nv, c.dv, c.dk), dtype=torch.float32, device=dev)
        self.conv = torch.zeros((seqs, nl, c.conv_kernel - 1, c.conv_dim), dtype=torch.bfloat16, device=dev)
        self.kc = torch.zeros((seqs, na + mtp_layers, capacity, c.kv_heads, c.head_dim), dtype=torch.bfloat16,
                              device=dev)
        self.vc = torch.zeros_like(self.kc)
        # with a draft head: each sequence's last hidden row the head has not absorbed yet
        self.tail = torch.zeros((seqs, c.hidden), dtype=torch.bfloat16, device=dev) if mtp_layers else None
        self.free = list(range(seqs - 1, -1, -1))

    def alloc(self) -> "State":
        if not self.free:
            raise RuntimeError("state pool exhausted")
        st = State(self, self.free.pop())
        st.reset()
        return st

    def release(self, st: "State") -> None:
        if st.slot in self.free:
            raise ValueError("state released twice")
        self.free.append(st.slot)

    def clone(self, st: "State") -> "State":
        """A new state holding a copy of ``st``'s committed sequence."""

        other = self.alloc()
        other.copy_(st)
        return other

    def nbytes_per_seq(self) -> int:
        return sum(t[0].numel() * t.element_size() for t in (self.rec, self.conv, self.kc, self.vc))


class State:
    """One sequence's committed caches (views of its pool slot), length ``pos``, GDN parity ``cur``, head state."""

    def __init__(self, pool: Pool, slot: int) -> None:
        self.pool, self.slot = pool, slot
        self.capacity = pool.capacity
        self.rec = pool.rec[slot]             # [2, linear layers, nv, dv, dk] fp32
        self.conv = pool.conv[slot]           # [linear layers, taps - 1, conv_dim] bf16
        self.kc = pool.kc[slot]               # [attention layers (+ MTP), capacity, kv_heads, head_dim] bf16
        self.vc = pool.vc[slot]
        n_att = self.kc.shape[0] - pool.mtp_layers
        self.mtp_kc = self.kc[n_att:]         # the draft head's caches (empty without one)
        self.mtp_vc = self.vc[n_att:]
        self.mtp_tail = pool.tail[slot] if pool.tail is not None else None
        self.cur = 0
        self.pos = 0
        self.mtp_len = 0
        self.mtp_tail_at = -1

    def reset(self) -> None:
        """An empty sequence (cache rows need no clearing: only rows below ``pos`` are ever read)."""

        self.cur = 0
        self.rec[0].zero_()
        self.conv.zero_()
        self.pos = 0
        self.mtp_len = 0
        self.mtp_tail_at = -1

    def copy_(self, src: "State") -> None:
        """Copy ``src``'s committed sequence: its current GDN states, conv windows, and cache rows below its length."""

        if src.kc.shape != self.kc.shape:
            raise ValueError("states from pools of different shapes")
        self.cur = 0
        self.rec[0].copy_(src.rec[src.cur])
        self.conv.copy_(src.conv)
        p = src.pos
        if p:
            self.kc[:, :p].copy_(src.kc[:, :p])
            self.vc[:, :p].copy_(src.vc[:, :p])
        self.pos = p
        self.mtp_len, self.mtp_tail_at = min(src.mtp_len, p), src.mtp_tail_at
        if self.mtp_tail is not None and src.mtp_tail is not None:
            self.mtp_tail.copy_(src.mtp_tail)
        else:
            self.mtp_tail_at = -1

    def snapshot(self, rows: bool = False) -> dict:
        """What the sequence keeps outside its cache rows: with those rows in place, ``restore`` brings it back.

        ``rows``: a copy of the cache rows below ``pos`` too (attention and draft-head keys and values), so the
        snapshot restores into any state of this pool, whatever has been written there since."""

        tail = self.mtp_tail.clone() if self.mtp_tail is not None else None
        snap = {"pos": self.pos, "rec": self.rec[self.cur].clone(), "conv": self.conv.clone(),
                "mtp": (self.mtp_len, self.mtp_tail_at, tail)}
        if rows:
            snap["kc"], snap["vc"] = self.kc[:, :self.pos].clone(), self.vc[:, :self.pos].clone()
        return snap

    def restore(self, snap: dict) -> None:
        if snap["pos"] > self.capacity:
            raise ValueError("snapshot longer than this state's capacity")
        p = int(snap["pos"])
        if "kc" in snap and (snap["kc"].shape[1] != p or snap["kc"].shape[0] != self.kc.shape[0]):
            raise ValueError("a snapshot's cache rows do not match its length or this pool's layers")
        self.cur = 0
        self.rec[0].copy_(snap["rec"])
        self.conv.copy_(snap["conv"])
        if "kc" in snap and p:
            self.kc[:, :p].copy_(snap["kc"])
            self.vc[:, :p].copy_(snap["vc"])
        self.pos = p
        mtp_len, tail_at, tail = snap.get("mtp", (0, -1, None))
        self.mtp_len, self.mtp_tail_at = min(int(mtp_len), self.pos), int(tail_at)
        if self.mtp_tail is not None and tail is not None:
            self.mtp_tail.copy_(tail)
        else:
            self.mtp_tail_at = -1


@dataclass
class Seq:
    """One sequence's rows of a staged window."""

    state: State
    index: int                # its place in the window's table
    row0: int                 # its first row in the window
    rows: int
    pos: int                  # the position of its first row (its committed length when staged)
    full: bool                # a window of at most ``window_rows`` rows: any prefix may be kept
