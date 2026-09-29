"""Qwen3's row-local prompt kernels: each program reads one row, so a row's bits never depend on the batch around it."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm(X, R, W, Y, eps, D: tl.constexpr, BLOCK: tl.constexpr, HAS_R: tl.constexpr):
    """x += r in fp32, the residual kept unrounded; y = rmsnorm(x) * w rounded once to bf16 for the projections."""

    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    ok = offs < D
    x = tl.load(X + row * D + offs, mask=ok, other=0.0)
    if HAS_R:
        x = x + tl.load(R + row * D + offs, mask=ok, other=0.0)
        tl.store(X + row * D + offs, x, mask=ok)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
    w = tl.load(W + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(Y + row * D + offs, (x * inv * w).to(tl.bfloat16), mask=ok)


def add_rmsnorm(x: torch.Tensor, r: torch.Tensor | None, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Add ``r`` (fp32 rows) into the fp32 residual ``x`` in place; return rmsnorm(x) * weight as bf16 rows."""

    t, d = x.shape
    if x.dtype != torch.float32 or not x.is_contiguous() or (r is not None and (r.shape != x.shape or
                                                                                r.dtype != torch.float32)):
        raise ValueError("add_rmsnorm takes a contiguous fp32 residual and fp32 updates of its shape")
    y = torch.empty((t, d), dtype=torch.bfloat16, device=x.device)
    _add_rmsnorm[(t,)](x, r if r is not None else x, weight, y, eps, D=d, BLOCK=triton.next_power_of_2(d),
                       HAS_R=r is not None, num_warps=8)
    return y


@triton.jit
def _qkv(QKV, QN, KN, POS, COS, SIN, Q, K, V, eps, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
         HALF: tl.constexpr):
    """Program (row, head): heads 0..H-1 queries and H..H+HK-1 keys get RMSNorm then rotary; the rest are values."""

    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    d = tl.arange(0, D)
    src = QKV + row * ((H + 2 * HK) * D) + head * D
    x = tl.load(src + d).to(tl.float32)
    if head < H + HK:
        partner = tl.where(d < HALF, d + HALF, d - HALF)
        xp = tl.load(src + partner).to(tl.float32)
        if head < H:
            w = tl.load(QN + d).to(tl.float32)
            wp = tl.load(QN + partner).to(tl.float32)
        else:
            w = tl.load(KN + d).to(tl.float32)
            wp = tl.load(KN + partner).to(tl.float32)
        inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
        xn = (x * inv * w).to(tl.bfloat16).to(tl.float32)
        xpn = (xp * inv * wp).to(tl.bfloat16).to(tl.float32)
        at = tl.load(POS + row).to(tl.int64) * HALF + tl.where(d < HALF, d, d - HALF)
        c = tl.load(COS + at)
        s = tl.load(SIN + at)
        out = tl.where(d < HALF, xn * c - xpn * s, xn * c + xpn * s).to(tl.bfloat16)   # rotate-half
        if head < H:
            tl.store(Q + (row * H + head) * D + d, out)
        else:
            tl.store(K + (row * HK + head - H) * D + d, out)
    else:
        tl.store(V + (row * HK + head - H - HK) * D + d, x.to(tl.bfloat16))


def qkv(rows: torch.Tensor, q_norm: torch.Tensor, k_norm: torch.Tensor, pos: torch.Tensor, cos: torch.Tensor,
        sin: torch.Tensor, eps: float, *, heads: int, kv_heads: int, head_dim: int):
    """[q | k | v] projection rows (T, (H + 2 HK) D) -> normed, rotated q (T, H, D), k (T, HK, D) and v (T, HK, D)."""

    t = rows.shape[0]
    if rows.shape[1] != (heads + 2 * kv_heads) * head_dim or not rows.is_contiguous() or pos.shape != (t,):
        raise ValueError("qkv takes contiguous [q | k | v] rows and one position a row")
    q = torch.empty((t, heads, head_dim), dtype=torch.bfloat16, device=rows.device)
    k = torch.empty((t, kv_heads, head_dim), dtype=torch.bfloat16, device=rows.device)
    v = torch.empty((t, kv_heads, head_dim), dtype=torch.bfloat16, device=rows.device)
    _qkv[(t, heads + 2 * kv_heads)](rows, q_norm, k_norm, pos, cos, sin, q, k, v, eps, H=heads, HK=kv_heads,
                                    D=head_dim, HALF=head_dim // 2, num_warps=2)
    return q, k, v


@triton.jit
def _swiglu(GU, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = offs < N
    g = tl.load(GU + row * 2 * N + offs, mask=ok, other=0.0).to(tl.float32)
    u = tl.load(GU + row * 2 * N + N + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(OUT + row * N + offs, (g * tl.sigmoid(g) * u).to(tl.bfloat16), mask=ok)


def swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """[gate | up] rows (T, 2N) -> silu(gate) * up (T, N) bf16."""

    t, two = gate_up.shape
    if two % 2 or not gate_up.is_contiguous():
        raise ValueError("swiglu takes contiguous [gate | up] rows")
    out = torch.empty((t, two // 2), dtype=torch.bfloat16, device=gate_up.device)
    block = 1024
    _swiglu[(t, triton.cdiv(two // 2, block))](gate_up, out, N=two // 2, BLOCK=block, num_warps=4)
    return out


@triton.jit
def _pool(X, R, W, OUT, eps, D: tl.constexpr, BLOCK: tl.constexpr):
    """The last layer's residual sum and the final RMSNorm, all in fp32."""

    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    ok = offs < D
    h = tl.load(X + row * D + offs, mask=ok, other=0.0) + tl.load(R + row * D + offs, mask=ok, other=0.0)
    inv = 1.0 / tl.sqrt(tl.sum(h * h, axis=0) / D + eps)
    tl.store(OUT + row * D + offs, h * inv * tl.load(W + offs, mask=ok, other=0.0).to(tl.float32), mask=ok)


def pool(x: torch.Tensor, r: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Pooled rows' hidden states after the final norm, (n, D) fp32 and not yet L2-normalized."""

    n, d = x.shape
    if r.shape != x.shape or not (x.is_contiguous() and r.is_contiguous()) or {x.dtype, r.dtype} != {torch.float32}:
        raise ValueError("pool takes contiguous fp32 residual and update rows of one shape")
    out = torch.empty((n, d), dtype=torch.float32, device=x.device)
    block = triton.next_power_of_2(d)
    _pool[(n,)](x, r, weight, out, eps, D=d, BLOCK=block, num_warps=8)
    return out
