"""SPIKE: ``qmm_fast._qmm_tiled`` with each program covering ``TPP`` adjacent stored 64-column tiles.

At many rows the lane matmul is limited by re-reading the activation tile once per 64 output columns; covering
several tiles per program reads it once for all of them. Storage and per-element arithmetic are unchanged (each
output's k-reduction is the same 64-wide group dots accumulated in group order), so the bits are the same."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .qmm import bucket, group_sums, split_k
from .qmm_fast import BN, _reduce, groups_per_iteration


@triton.jit
def _qmm_wide(X, XS, W, S, B, OUT, PART, M,
              N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
              BLOCK_N: tl.constexpr, TPP: tl.constexpr, GPI: tl.constexpr):
    KG: tl.constexpr = K // 64
    PER: tl.constexpr = KG // SK
    WN: tl.constexpr = BLOCK_N * TPP
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = pid_n * WN + tl.arange(0, WN)
    rk = tl.arange(0, 64)
    rw = tl.arange(0, 8)
    shifts = tl.arange(0, 8) * 4
    m_ok = rm < M
    n_ok = rn < N
    # column c of this program lives in stored tile (pid_n * TPP + c // BLOCK_N), local column c % BLOCK_N
    cols = tl.arange(0, WN)
    tile_of = pid_n * TPP + cols // BLOCK_N
    local = cols % BLOCK_N
    t_ok = tile_of * BLOCK_N < N + BLOCK_N - 1
    acc = tl.zeros((BM, WN), dtype=tl.float32)
    for i in range(PER // GPI):
        for j in tl.static_range(GPI):
            g = pid_s * PER + i * GPI + j
            words = tl.load(W + tile_of[:, None] * (KG * BLOCK_N * 8) + g * (BLOCK_N * 8) + local[:, None] * 8
                            + rw[None, :], mask=t_ok[:, None], other=0)
            x = tl.load(X + rm[:, None] * K + (g * 64 + rk)[None, :], mask=m_ok[:, None], other=0.0)
            q = (words[:, :, None] >> shifts[None, None, :]) & 0xF
            q = tl.reshape(q, (WN, 64)).to(tl.bfloat16)
            p = tl.dot(x, tl.trans(q))
            s = tl.load(S + g * N + rn, mask=n_ok, other=0.0).to(tl.float32)
            b = tl.load(B + g * N + rn, mask=n_ok, other=0.0).to(tl.float32)
            xs = tl.load(XS + rm * KG + g, mask=m_ok, other=0.0)
            acc = acc + p * s[None, :] + xs[:, None] * b[None, :]
    out_mask = m_ok[:, None] & n_ok[None, :]
    if SK == 1:
        tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)


def lane_matmul_wide(x: torch.Tensor, tw: torch.Tensor, ts: torch.Tensor, tb: torch.Tensor, n: int,
                     xs: torch.Tensor | None = None, *, tpp: int = 2, gpi: int = 1, num_warps: int = 8,
                     num_stages: int = 2, bm: int | None = None) -> torch.Tensor:
    m, k = x.shape
    x = x.contiguous()
    bm = bucket(m) if bm is None else bm
    if xs is None:
        xs = group_sums(x)
    sk = split_k(n, k)
    per = (k // 64) // sk
    gpi = groups_per_iteration(per, gpi)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    part = out if sk == 1 else torch.empty((sk, m, n), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, bm), triton.cdiv(n, BN * tpp), sk)
    _qmm_wide[grid](x, xs, tw, ts, tb, out, part, m, N=n, K=k, SK=sk, BM=bm, BLOCK_N=BN, TPP=tpp, GPI=gpi,
                    num_warps=num_warps, num_stages=num_stages)
    if sk > 1:
        total = m * n
        _reduce[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, num_warps=4)
    return out
