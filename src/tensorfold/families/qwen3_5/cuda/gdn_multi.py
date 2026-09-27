"""SPIKE: GDN steps for several requests in one launch each (see gdn_multi.cu).

``pre``: the conv + SiLU + norms of every request's rows, reading each request's own conv state.
``tree``: every request's tree walk from its own recurrent state. ``replay``: every (layer, request) commit.
Each request's arithmetic is the single-request kernels', so its bits do not depend on its batch-mates.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    here = Path(__file__).parent
    return load(name="tensorfold_gdn_multi_v1", sources=[str(here / "gdn_multi.cpp"), str(here / "gdn_multi.cu")],
                extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False)


def tree(q, k, v, g, beta, states: torch.Tensor, parents: torch.Tensor, offsets: torch.Tensor,
         chain: torch.Tensor) -> torch.Tensor:
    return _ext().tree_multi(q, k, v, g, beta, states, parents, offsets, chain)


def replay(table: torch.Tensor, rows: torch.Tensor, counts: torch.Tensor, hk: int, hv: int, dv: int) -> torch.Tensor:
    return _ext().replay_multi(table, rows, counts, hk, hv, dv)


@triton.jit
def _gdn_pre_multi(SRC, CW, WIN, A, B, ALOG, DTB, Q, K, V, G, BETA,
                   C: tl.constexpr, KH: tl.constexpr, VH: tl.constexpr, DK: tl.constexpr, NKEEP: tl.constexpr):
    """``glue._gdn_pre`` with the window rows read from one buffer: [every request's conv state | every qkv row]."""

    row = tl.program_id(0)
    head = tl.program_id(1)
    ch = head * DK + tl.arange(0, DK)
    acc = tl.zeros((DK,), dtype=tl.float32)
    for j in tl.static_range(NKEEP + 1):
        src = tl.load(WIN + row * (NKEEP + 1) + j)
        x = tl.load(SRC + src * C + ch, mask=ch < C, other=0.0).to(tl.float32)
        w = tl.load(CW + ch * (NKEEP + 1) + j).to(tl.float32)
        acc = acc + x * w
    c = (acc * tl.sigmoid(acc)).to(tl.bfloat16).to(tl.float32)
    if head < 2 * KH:
        inv = 1.0 / tl.sqrt(tl.sum(c * c, axis=0) / DK + 1e-6)
        is_q = head < KH
        scale = tl.where(is_q, 1.0 / DK, 1.0 / tl.sqrt(DK * 1.0))
        out = (c * inv * scale).to(tl.bfloat16)
        hk = tl.where(is_q, head, head - KH)
        base = (row * KH + hk) * DK + tl.arange(0, DK)
        if is_q:
            tl.store(Q + base, out)
        else:
            tl.store(K + base, out)
    else:
        hv = head - 2 * KH
        tl.store(V + (row * VH + hv) * DK + tl.arange(0, DK), c.to(tl.bfloat16))
        a = tl.load(A + row * VH + hv).to(tl.float32) + tl.load(DTB + hv)
        sp = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
        g = tl.exp(-tl.exp(tl.load(ALOG + hv)) * sp)
        b = tl.load(B + row * VH + hv).to(tl.float32)
        tl.store(G + row * VH + hv, g)
        tl.store(BETA + row * VH + hv, tl.sigmoid(b))


def pre(src: torch.Tensor, conv_w: torch.Tensor, windows: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
        A_log: torch.Tensor, dt_bias: torch.Tensor, *, kh: int, vh: int, dk: int, nkeep: int):
    """src (S, C): conv-state rows then qkv rows; windows (W, nkeep+1) int32 rows of src."""

    W = windows.shape[0]
    C = src.shape[1]
    dev = src.device
    q = torch.empty((W, kh, dk), dtype=torch.bfloat16, device=dev)
    k = torch.empty((W, kh, dk), dtype=torch.bfloat16, device=dev)
    v = torch.empty((W, vh, dk), dtype=torch.bfloat16, device=dev)
    g = torch.empty((W, vh), dtype=torch.float32, device=dev)
    beta = torch.empty((W, vh), dtype=torch.float32, device=dev)
    _gdn_pre_multi[(W, 2 * kh + vh)](src, conv_w, windows, a, b, A_log, dt_bias, q, k, v, g, beta,
                                     C=C, KH=kh, VH=vh, DK=dk, NKEEP=nkeep, num_warps=2)
    return q, k, v, g, beta
