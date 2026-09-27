"""CUDA graphs for Qwen3.6-35B-A3B's decode steps: a forward of about 700 kernel launches becomes one graph launch.

A forward's GPU work (``forward.compute``) reads only static buffers, device-side positions and its sequence's state,
so it can be captured once and replayed with new inputs staged beforehand (token ids and positions, ``forward.stage``).
What a capture fixes, and so what keys it:

- the window's rows R (every launch grid and view);
- the sequence's pool slot (its caches' addresses) and its GDN buffer parity ``State.cur``: a layer reads its
  committed state from one of two buffers and writes the next state into the other, and every commit flips them all.

So a state has two graphs per window size. The MTP head's step (``mtp.mtp_compute``) has no GDN state: one graph per
(rows, with or without logits, slot). Its inputs (next tokens, positions, the rows' hidden states) are staged into
static buffers the same way (``mtp.mtp_stage``).

Replaying a graph runs the same kernels with the same launch parameters as the eager call, so it gives the same bits
(``tests/cuda/test_qwen36moe_mtp.py`` checks logits and tokens against eager decoding). Larger windows (prefill
chunks) run eagerly.
"""

from __future__ import annotations

import gc
from typing import Callable, Sequence

import torch

from .forward import State, compute, stage
from .mtp import mtp_compute, mtp_stage


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
        # no garbage collection while capturing: collecting another decoder's graphs calls cuGraphExecDestroy,
        # which invalidates the capture
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

    @torch.no_grad()
    def forward(self, st: State, tokens: Sequence[int]) -> torch.Tensor:
        """A window's logits [R, V] (a view of the buffers), replayed from its graph."""

        e = self.e
        m, b = e.m, e.buf
        R, table = stage(m, b, [(st, tokens)])
        if R > self.max_rows:
            return compute(m, b, R, table, logits="all")
        key = (R, st.cur, *self._slot(st))
        g = self.main.get(key)
        if g is None:
            compute(m, b, R, table, logits="all")          # eager warm-up: compiles every kernel for this shape
            g = self._capture(lambda: compute(m, b, R, table, logits="all"))
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
        key = (n, logits, *self._slot(st))
        g = self.steps.get(key)
        if g is None:
            mtp_compute(m, k, b, st, n, logits=logits)
            g = self._capture(lambda: mtp_compute(m, k, b, st, n, logits=logits))
            self.steps[key] = g
        g.replay()
        return b.logits if logits else None

    @torch.no_grad()
    def warm(self, st: State, rows: int | None = None) -> int:
        """Capture every decode shape of ``st`` (windows of 1..rows rows at both GDN parities; MTP steps of 1..rows
        rows, with and without logits). Leaves the sequence dirty: reset it before use."""

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
