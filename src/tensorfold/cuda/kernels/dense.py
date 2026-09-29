"""Prompt matmul on unquantized bf16 weights: one fp32 chain over K a row, so a row's bits never depend on the others.

``tl.dot`` issues bf16 m16n8k16 tensor-core products that accumulate into fp32 in increasing K order whatever the
block shape, so the block shapes below differ in speed only: every row gets the bits it gets alone.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# (rows, columns, inputs a step, stages, warps) by row count: small blocks spread a few rows' weight reads over the SMs
BLOCKS = ((16, (16, 64, 64, 4, 4)), (128, (64, 64, 64, 4, 4)), (256, (64, 128, 64, 3, 4)),
          (None, (128, 256, 64, 3, 8)))


@triton.jit
def _matmul(X, W, OUT, M, N, ldx, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            GROUP: tl.constexpr, F32: tl.constexpr):
    pid = tl.program_id(0)
    row_blocks, col_blocks = tl.cdiv(M, BM), tl.cdiv(N, BN)
    band = GROUP * col_blocks                         # GROUP row blocks sweep the columns together (L2 reuse)
    first = pid // band * GROUP
    height = tl.minimum(row_blocks - first, GROUP)
    rows = ((first + pid % band % height) * BM + tl.arange(0, BM)).to(tl.int64)
    cols = ((pid % band // height) * BN + tl.arange(0, BN)).to(tl.int64)
    k = tl.arange(0, BK)
    x = X + rows[:, None] * ldx + k[None, :]
    w = W + cols[:, None] * K + k[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(x, mask=rows[:, None] < M, other=0.0)
        b = tl.load(w, mask=cols[:, None] < N, other=0.0)
        acc = tl.dot(a, tl.trans(b), acc)
        x += BK
        w += BK
    ok = (rows[:, None] < M) & (cols[None, :] < N)
    if F32:
        tl.store(OUT + rows[:, None] * N + cols[None, :], acc, mask=ok)
    else:
        tl.store(OUT + rows[:, None] * N + cols[None, :], acc.to(tl.bfloat16), mask=ok)


def blocks_for(m: int) -> tuple[int, int, int, int, int]:
    """The block shape for ``m`` rows; any of ``BLOCKS`` gives the same bits."""

    return next(shape for top, shape in BLOCKS if top is None or m <= top)


def prefill_matmul(x: torch.Tensor, w: torch.Tensor, *, f32: bool = False, blocks: tuple | None = None,
                   out: torch.Tensor | None = None) -> torch.Tensor:
    """x (M, K) bf16 times w (N, K) bf16 transposed: (M, N) bf16, or unrounded fp32 with ``f32``."""

    if x.dim() != 2 or w.dim() != 2 or x.shape[1] != w.shape[1]:
        raise ValueError(f"prefill_matmul: x (M, K) and w (N, K), not {tuple(x.shape)} and {tuple(w.shape)}")
    m, k = x.shape
    n = w.shape[0]
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16 or k % 64:
        raise ValueError("prefill_matmul takes bf16 inputs and weights, K a multiple of 64")
    if x.stride(1) != 1 or not w.is_contiguous():
        raise ValueError("prefill_matmul reads contiguous input rows and a contiguous weight")
    kind = torch.float32 if f32 else torch.bfloat16
    if out is None:
        out = torch.empty((m, n), dtype=kind, device=x.device)
    elif not out.is_contiguous() or tuple(out.shape) != (m, n) or out.dtype != kind:
        raise ValueError("prefill_matmul: out must be a contiguous (M, N) tensor of the output type")
    bm, bn, bk, stages, warps = blocks or blocks_for(m)
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    _matmul[grid](x, w, out, m, n, x.stride(0), K=k, BM=bm, BN=bn, BK=bk, GROUP=8, F32=f32, num_stages=stages,
                  num_warps=warps)
    return out
