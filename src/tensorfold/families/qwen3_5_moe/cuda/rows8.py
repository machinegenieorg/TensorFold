"""A bf16 projection on a prompt chunk's FP8 rows, on the stored weights exactly.

The prompt glue (``prefill_glue``) hands every projection its rows as e4m3 bytes with a scale a row, the bytes of
each 32 inputs in the FP8 GEMM's fragment order. The NVFP4 checkpoint keeps DeltaNet's ``in_proj_a`` and
``in_proj_b`` bf16; rather than an e4m3 copy of them (the 27B's gates), the kernel widens each row's codes to bf16
(exact: every e4m3 value is a bf16 value) against the weights as stored, their inputs put in the rows' order once at
load, so each output is ``a[m] * sum_k code[m, k] * w[n, k]`` in fp32, rounded once to bf16. K runs in order in
one slice and the row tile is fixed, so a row's bits depend only on its own inputs, whatever the chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 32, 64


def fragment_order(k: int, device) -> torch.Tensor:
    """Input index of each stored position of a row (``qmm.quantize_rows``' order within each 32 inputs)."""

    m = torch.arange(k, device=device)
    j = m % 4
    return (m // 32) * 32 + ((m % 32) // 16) * 16 + ((m % 16) // 4) * 2 + (j % 2) + (j // 2) * 8


@dataclass
class Rows8:
    """A bf16 [n, k] weight with its inputs in the FP8 rows' stored order."""

    weight: torch.Tensor

    @classmethod
    def make(cls, weight: torch.Tensor) -> "Rows8":
        n, k = weight.shape
        if k % 32:
            raise ValueError(f"bf16 weight [{n}, {k}]: K must be a multiple of 32")
        return cls(weight[:, fragment_order(k, weight.device)].to(torch.bfloat16).contiguous())

    @property
    def n(self) -> int:
        return int(self.weight.shape[0])

    @property
    def k(self) -> int:
        return int(self.weight.shape[1])

    def nbytes(self) -> int:
        return self.weight.numel() * 2


@triton.jit
def _rows8mm(X8, A, W, OUT, M, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
             BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    n_ok = rn < N
    rm_a = tl.where(m_ok, rm, 0)
    rn_a = tl.where(n_ok, rn, 0)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(X8 + rm_a[:, None] * K + (k0 + rk)[None, :], mask=m_ok[:, None], other=0)
        x = x.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
        w = tl.load(W + rn_a[:, None] * K + (k0 + rk)[None, :], mask=n_ok[:, None], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    a = tl.load(A + rm_a, mask=m_ok, other=0.0)
    tl.store(OUT + rm_a[:, None] * N + rn_a[None, :], (acc * a[:, None]).to(tl.bfloat16),
             mask=m_ok[:, None] & n_ok[None, :])


def matmul(xq: tuple, q: Rows8) -> torch.Tensor:
    """FP8 rows (e4m3 bytes [M, K] in fragment order, ..., row scales [M]) times the weight -> [M, n] bf16."""

    x8, a = xq[0], xq[2]
    m, k = x8.shape
    if k != q.k or not x8.is_contiguous():
        raise ValueError(f"rows8 matmul: rows {tuple(x8.shape)} do not match K={q.k}")
    out = torch.empty((m, q.n), dtype=torch.bfloat16, device=x8.device)
    grid = (triton.cdiv(m, BM), triton.cdiv(q.n, BN))
    _rows8mm[grid](x8, a, q.weight, out, m, N=q.n, K=k, BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3)
    return out
