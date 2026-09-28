"""CUDA graphs for Qwen3.6-35B-A3B's decode windows and MTP steps, keyed by rows, slot, GDN parity and bucket."""

from __future__ import annotations

import gc
from typing import Callable, Sequence

import torch

from .forward import compute, stage
from .mtp import mtp_compute, mtp_stage
from .state import State


class Graphs:
    def __init__(self, e, *, max_rows: int = 8) -> None:
        self.e = e
        self.max_rows = max_rows
        self.main: dict[tuple, torch.cuda.CUDAGraph] = {}
        self.steps: dict[tuple, torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.captures = 0

    def _capture(self, fn: Callable[[], object]) -> torch.cuda.CUDAGraph:
        torch.cuda.synchronize()
        gc.collect()
        g = torch.cuda.CUDAGraph()
        # no garbage collection while capturing: destroying another decoder's graphs would invalidate the capture
        enabled = gc.isenabled()
        gc.disable()
        try:
            with torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
                fn()
        finally:
            if enabled:
                gc.enable()
        torch.cuda.synchronize()
        self.captures += 1
        return g

    @staticmethod
    def _slot(st: State) -> tuple[int, int]:
        return id(st.pool), st.slot

    def _bucket(self, st: State, end: int) -> int:
        """The attention bound for keys below ``end``: Flash Next's bucket (at least 8192, powers of two)."""

        return min(st.capacity, self.e.capacity, max(8192, 1 << (end - 1).bit_length()))

    @torch.no_grad()
    def forward(self, st: State, tokens: Sequence[int]) -> torch.Tensor:
        """A window's logits [R, V] (a view of the buffers), replayed from its graph."""

        e = self.e
        m, b = e.m, e.buf
        R, table = stage(m, b, [(st, tokens)])
        if R > self.max_rows:
            return compute(m, b, R, table, logits="all")
        context = self._bucket(st, st.pos + R)
        key = (R, st.cur, *self._slot(st), context)
        g = self.main.get(key)
        if g is None:
            compute(m, b, R, table, logits="all", context=context)     # eager warm-up: compiles this shape
            g = self._capture(lambda: compute(m, b, R, table, logits="all", context=context))
            self.main[key] = g
        g.replay()
        return b.logits[:R]

    @torch.no_grad()
    def mtp(self, st: State, tokens: Sequence[int], hidden: torch.Tensor, pos0: int, *,
            logits: bool = True) -> torch.Tensor | None:
        """An MTP step (``mtp.mtp_forward``) replayed from its graph."""

        e = self.e
        m, k, b = e.m, e.mtp, e.mbuf
        n = mtp_stage(b, st, tokens, hidden, pos0)
        if n > self.max_rows:
            return mtp_compute(m, k, b, st, n, logits=logits)
        context = self._bucket(st, pos0 + n)
        key = (n, logits, *self._slot(st), context)
        g = self.steps.get(key)
        if g is None:
            mtp_compute(m, k, b, st, n, logits=logits, context=context)
            g = self._capture(lambda: mtp_compute(m, k, b, st, n, logits=logits, context=context))
            self.steps[key] = g
        g.replay()
        return b.logits if logits else None

    @torch.no_grad()
    def warm(self, st: State, rows: int | None = None) -> int:
        """Capture every decode shape of ``st`` at the first context bucket; leaves the sequence dirty (reset it)."""

        e = self.e
        rows = min(rows or self.max_rows, self.max_rows)
        before = self.captures
        st.reset()
        for parity in (0, 1):
            st.cur = parity
            for R in range(1, rows + 1):
                self.forward(st, [0] * R)
        e.buf.pending.clear()
        st.reset()
        if e.mtp is not None:
            for n in range(1, rows + 1):
                for logits in (True, False):
                    self.mtp(st, [0] * n, e.mbuf.hin[:n], 0, logits=logits)
        torch.cuda.synchronize()
        return self.captures - before
