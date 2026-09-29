"""FP8 projections as ModelOpt stores them (e4m3 codes, one fp32 scale a tensor), read weight-only.

A projection's weight is ``fp32(code) * weight_scale``, exactly. An e4m3 code is exact in bf16, so the kernel
takes the codes themselves as the tensor-core operand and applies the scale once, to the fp32 sums:

    y[m, n] = bf16(scale[n] * (sum over K blocks in order of dot(x[m, block], code[n, block])))

Activations stay bf16. K is split into slices fixed by the weight's shape (``bf16.split_k``, never the row
count) and the slices are summed in slice order, so a row's bits depend only on its own input: the chain
Flash Next's ``bf16.matmul`` runs, with a scale after it. ``weight`` holds the checkpoint's bytes, decoded in the
kernel (what the engine loads), or the codes widened to bf16 (``widen``: Flash Next's ``bf16.matmul`` times the
scale, bit for bit, at twice the memory). Prompt chunks (``prefill``) take K in one slice: chunk-invariant bits of
their own, as the 27B's prompt path has. Each form is row-invariant; they agree to rounding, not bit for bit, as
the tensor cores may order a step's products differently for operands decoded in registers. The checkpoint's
``input_scale`` quantizes activations for vLLM's FP8 GEMMs; it is not part of the weights and is not read.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from tensorfold.families.qwen4_exp.cuda import bf16, nvfp4

BN = bf16.BN              # output columns a program
BK = bf16.BK              # inputs a step (the split keeps slices whole steps)
PROMPT_BN = 128           # a prompt chunk's: 4,096 rows through the model's FP8 projections in 51 ms, not 93


@dataclass
class FP8Linear:
    """An [n, k] projection: e4m3 bytes (uint8) or the same codes widened to bf16, and each row's fp32 scale."""

    weight: torch.Tensor      # [n, k] uint8 (e4m3 bytes) or bf16 (the codes, exact)
    scale: torch.Tensor       # [n] fp32: the tensor's weight_scale on each of its rows
    layout: str = "fp8"
    fast: bool = False        # not the 27B's 4-bit g64 path (prompts take bf16 rows)

    @property
    def n(self) -> int:
        return int(self.weight.shape[0])

    @property
    def k(self) -> int:
        return int(self.weight.shape[1])

    def nbytes(self) -> int:
        return self.weight.numel() * self.weight.element_size() + self.scale.numel() * 4

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return matmul(x, self)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """The prompt form: K in one slice and wider tiles, the bits the same for any chunk (not decode's)."""

        return matmul(x, self, sk=1, block_n=PROMPT_BN)

    def widen(self) -> "FP8Linear":
        """The codes as bf16 values (exact), read without decoding: ``bf16.matmul``'s bits, times the scale."""

        if self.weight.dtype == torch.bfloat16:
            return self
        return FP8Linear(nvfp4.e4m3_bits(self.weight).view(torch.bfloat16).contiguous(), self.scale)


def make_fp8(codes: torch.Tensor, weight_scale: torch.Tensor | float) -> FP8Linear:
    """The checkpoint's [n, k] e4m3 weight and its scale (a scalar, or one a row) as stored: nothing is widened."""

    raw = codes.contiguous().view(torch.uint8)
    if raw.dim() != 2:
        raise ValueError(f"an FP8 projection is a matrix, not {tuple(raw.shape)}")
    n = raw.shape[0]
    scale = torch.as_tensor(weight_scale, dtype=torch.float32).to(raw.device).reshape(-1)
    if scale.numel() == 1:
        scale = scale.expand(n)
    elif scale.numel() != n:
        raise ValueError(f"an FP8 weight_scale is one a tensor or one a row, not {scale.numel()} for {n} rows")
    return FP8Linear(raw, scale.contiguous())


def dequantize(q: FP8Linear) -> torch.Tensor:
    """The exact fp32 weight [n, k] (the reference the kernel is checked against)."""

    codes = q.weight if q.weight.dtype == torch.bfloat16 else nvfp4.e4m3_bits(q.weight).view(torch.bfloat16)
    return codes.float() * q.scale[:, None]


@triton.jit
def _fp8mm(X, W, S, OUT, PART, M, x_stride, N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr,
           BM: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, CODES: tl.constexpr):
    """x [M, K] bf16 @ W.T: K in BLOCK_K steps in order, one tensor-core dot each against the codes (decoded
    from e4m3 bytes when ``CODES``), fp32 sums; one slice scales and rounds here, several leave fp32 partials."""

    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m_ok = rm < M
    n_ok = rn < N
    rm_a = tl.where(m_ok, rm, 0)          # masked lanes keep in-range addresses (GB10 faults on them otherwise)
    rn_a = tl.where(n_ok, rn, 0)
    NB: tl.constexpr = K // SK // BLOCK_K
    acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for i in range(NB):
        k0 = (pid_s * NB + i) * BLOCK_K
        x = tl.load(X + rm_a[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
        w = tl.load(W + rn_a[:, None] * K + (k0 + rk)[None, :], mask=n_ok[:, None], other=0)
        if CODES:
            w = w.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)      # exact: every e4m3 value is a bf16 value
        acc = tl.dot(x, tl.trans(w), acc)
    out_mask = m_ok[:, None] & n_ok[None, :]
    if SK == 1:
        s = tl.load(S + rn_a, mask=n_ok, other=0.0)
        tl.store(OUT + rm_a[:, None] * N + rn_a[None, :], (acc * s[None, :]).to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(PART + (pid_s * M + rm_a[:, None]) * N + rn_a[None, :], acc, mask=out_mask)


@triton.jit
def _reduce(PART, S, OUT, total, N: tl.constexpr, SK: tl.constexpr, BLOCK: tl.constexpr):
    """The slices' fp32 partials summed in slice order, times the row's scale, rounded once."""

    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = offs < total
    offs_a = tl.where(ok, offs, 0)
    acc = tl.load(PART + offs_a, mask=ok, other=0.0)
    for s in tl.static_range(1, SK):
        acc = acc + tl.load(PART + s * total + offs_a, mask=ok, other=0.0)
    scale = tl.load(S + offs_a % N, mask=ok, other=0.0)
    tl.store(OUT + offs_a, (acc * scale).to(tl.bfloat16), mask=ok)


def matmul(x: torch.Tensor, q: FP8Linear, *, out: torch.Tensor | None = None, sk: int | None = None,
           num_warps: int = 4, num_stages: int = 3, block_n: int = BN, bk: int = BK,
           target: int = 160) -> torch.Tensor:
    # decode shapes spend their time in launches, not tiles: of 162 (block_n, bk, split target, warps, stages)
    # tried on an RTX PRO 6000 Max-Q, the best took 4% off a decode step's projections and a third more prefill
    """x [M, K] bf16 (rows may be strided) @ the exact weight's transpose -> [M, N] bf16."""

    m, k = x.shape
    if k != q.k or x.stride(1) != 1:
        raise ValueError(f"fp8 matmul: x {tuple(x.shape)} does not match K={q.k}")
    if k % bk:
        raise ValueError(f"fp8 matmul: K {k} is not a multiple of {bk}")
    sk = int(sk) if sk else bf16.split_k(q.n, k, target=target, bk=bk)
    if out is None:
        out = torch.empty((m, q.n), dtype=torch.bfloat16, device=x.device)
    part = torch.empty((sk, m, q.n), dtype=torch.float32, device=x.device) if sk > 1 else out
    bm = 128 if m > 128 else 16
    grid = (triton.cdiv(m, bm), triton.cdiv(q.n, block_n), sk)
    _fp8mm[grid](x, q.weight, q.scale, out, part, m, x.stride(0), N=q.n, K=k, SK=sk, BM=bm, BLOCK_N=block_n,
                 BLOCK_K=bk, CODES=q.weight.dtype == torch.uint8, num_warps=num_warps, num_stages=num_stages)
    if sk > 1:
        total = m * q.n
        _reduce[(triton.cdiv(total, 1024),)](part, q.scale, out, total, N=q.n, SK=sk, BLOCK=1024, num_warps=4)
    return out
