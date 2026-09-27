"""Qwen3.6-35B-A3B's MoE on CUDA: 256 routed experts (top 8, width 512) and the shared expert as expert 256.

    router    fp32 logits [R, 257] = x . W for the loader's fp32 router table W [257, 2048] (256 router rows, then
              the shared expert's gate row): bf16 x widened to fp32, one fused multiply-add chain per (row, expert)
              over K in order (``tl.dot`` at input_precision "ieee" runs on CUDA cores, k by k), so a logit never
              depends on the other rows, the tile or the launch settings
    select    Flash Next's kernels (``qwen4_exp/cuda/moe.py``): each row's top 8 by fp32 logit, largest first, the
              lower id among equal logits; weights exp(l_k - l_0) / sum over the 8, which is the softmax over the
              256 renormalised over the 8, rounded to bf16 as the reference implementation's weights are; slot 8
              the shared expert with weight bf16(sigmoid(bf16(gate logit))). Then one program groups the window:
              the distinct experts in increasing id order, each with the (row, slot) pairs that picked it in row
              order
    experts   ``qmm.moe_gateup`` / ``qmm.moe_down`` over the 257-expert table at group size 64: each (row,
              expert) pair gets the same bits whatever rows share the window (the K order is fixed; the program
              width and stages chosen for 1-2 row windows change the schedule, not the sums)
    combine   branch = bf16(sum over the 9 slots, in pick order and then the shared expert, of fp32 w_k * y_k),
              one rounding; optionally fused with the residual add and the next RMSNorm (``combine_add_rmsnorm``)
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from ...qwen4_exp.cuda import moe as _fn
from ...qwen4_exp.cuda.moe import MoEBuffers
from . import qmm

EXPERTS = 256             # routed experts; the shared expert is expert EXPERTS of the table
TOP_K = 8
WIDTH = 512               # an expert's intermediate width (routed and shared)
HIDDEN = 2048


def config(experts: int = EXPERTS, top_k: int = TOP_K, width: int = WIDTH, hidden: int = HIDDEN) -> SimpleNamespace:
    """The fields ``MoEBuffers`` reads, under Flash Next's names."""

    return SimpleNamespace(num_experts=experts, num_experts_per_tok=top_k, moe_intermediate_size=width,
                           hidden_size=hidden)


def buffers(rows: int, device: torch.device | str, cfg: SimpleNamespace | None = None) -> MoEBuffers:
    """Static MoE scratch for windows of up to ``rows`` rows: logits [rows, E + 1], picks and weights [rows, k + 1],
    the grouping, act [rows, k + 1, width] bf16 with its 32-sums, y [rows, k + 1, hidden] fp32."""

    return MoEBuffers(rows, cfg or config(), device)


def rows_view(buf: MoEBuffers, rows: int) -> MoEBuffers:
    """``buf`` restricted to its first ``rows`` rows (grouping sized for ``rows``), cached on the buffer."""

    if rows == buf.rows:
        return buf
    if not 0 < rows <= buf.rows:
        raise ValueError(f"rows_view: {rows} rows in a buffer for {buf.rows}")
    subs = buf.__dict__.setdefault("_views", {})
    sub = subs.get(rows)
    if sub is None:
        sub = object.__new__(MoEBuffers)
        sub.rows, sub.slots = rows, buf.slots
        sub.maxu = min(rows * (buf.slots - 1), buf.group.ids.shape[0] - 1) + 1
        sub.logits, sub.pick, sub.wts = buf.logits[:rows], buf.pick[:rows], buf.wts[:rows]
        sub.group = qmm.Group(buf.group.ids[:sub.maxu], buf.group.count,
                              buf.group.members.as_strided((sub.maxu, rows), (rows, 1)))
        sub.act, sub.axs, sub.y = buf.act[:rows], buf.axs[:rows], buf.y[:rows]
        subs[rows] = sub
    return sub


# -- router ------------------------------------------------------------------------------------------------
@triton.jit
def _router(X, W, OUT, M, x_stride, D: tl.constexpr, NE: tl.constexpr, BM: tl.constexpr,
            BLOCK_E: tl.constexpr, BK: tl.constexpr):
    """OUT[m, e] = x[m] . w[e]: fp32(bf16 x) times fp32 w, fp32 fused multiply-adds over K in order."""

    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    re = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    e_ok = re < NE
    acc = tl.zeros((BM, BLOCK_E), dtype=tl.float32)
    for k0 in range(0, D, BK):
        x = tl.load(X + rm[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0).to(tl.float32)
        w = tl.load(W + re[:, None] * D + (k0 + rk)[None, :], mask=e_ok[:, None], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc, input_precision="ieee")
    tl.store(OUT + rm[:, None] * NE + re[None, :], acc, mask=m_ok[:, None] & e_ok[None, :])


# (experts a program, K a step, warps, stages) by row bucket: none changes a logit's bits
ROUTER_CFG = {16: (16, 64, 4, 3), 32: (32, 64, 4, 3), 64: (64, 32, 4, 2), 128: (64, 32, 4, 2)}


def router(x: torch.Tensor, rows: torch.Tensor, out: torch.Tensor | None = None, *, block_e: int | None = None,
           bk: int | None = None, num_warps: int | None = None, num_stages: int | None = None) -> torch.Tensor:
    """x [R, D] bf16 (rows may be strided), rows [E + 1, D] fp32 (router rows, then the shared expert's gate row)
    -> [R, E + 1] fp32 logits."""

    m, d = x.shape
    ne = rows.shape[0]
    if rows.dtype != torch.float32 or rows.shape[1] != d or not rows.is_contiguous():
        raise ValueError(f"router: rows must be a contiguous fp32 [E + 1, {d}], got {rows.dtype} {tuple(rows.shape)}")
    if x.dtype != torch.bfloat16 or x.stride(1) != 1:
        raise ValueError("router: x must be bf16 with contiguous rows")
    if out is None:
        out = torch.empty((m, ne), dtype=torch.float32, device=x.device)
    elif out.shape != (m, ne) or not out.is_contiguous():
        raise ValueError(f"router: out {tuple(out.shape)} must be a contiguous ({m}, {ne})")
    bm = qmm.bucket(m)
    c_be, c_bk, c_w, c_s = ROUTER_CFG[bm]
    be, bk = block_e or c_be, bk or c_bk
    if d % bk:
        raise ValueError(f"router: K={d} is not a multiple of the step {bk}")
    grid = (triton.cdiv(m, bm), triton.cdiv(ne, be))
    _router[grid](x, rows, out, m, x.stride(0), D=d, NE=ne, BM=bm, BLOCK_E=be, BK=bk,
                  num_warps=num_warps or c_w, num_stages=num_stages or c_s)
    return out


def select(logits: torch.Tensor, buf: MoEBuffers, top_k: int = TOP_K, experts: int = EXPERTS) -> None:
    """Each row's experts and weights, then the window's distinct experts and their (row, slot) members
    (Flash Next's ``select``; ``buf`` sized for exactly ``logits.shape[0]`` rows, see ``rows_view``)."""

    _fn.select(logits, buf, top_k, experts)


def moe(x: torch.Tensor, xs: torch.Tensor, router_rows: torch.Tensor, ex: qmm.Experts, buf: MoEBuffers,
        *, top_k: int = TOP_K, experts: int = EXPERTS, **launch) -> MoEBuffers:
    """Route rows x [R, D] bf16 (with their 64-group sums xs) and run their experts. Returns the R-row view of
    ``buf``: y [R, k + 1, D] fp32 (slot k: the shared expert), wts [R, k + 1] (routed weights, then the shared
    gate), pick [R, k + 1]. ``launch``: expert-kernel settings (never change bits)."""

    rows = x.shape[0]
    sub = rows_view(buf, rows)
    router(x, router_rows, sub.logits)
    select(sub.logits, sub, top_k, experts)
    qmm.moe_gateup(x, xs, ex, sub.group, sub.act, sub.axs, **launch)
    qmm.moe_down(sub.act, sub.axs, ex, sub.group, sub.y, **launch)
    return sub


# -- combine -------------------------------------------------------------------------------------------------
@triton.jit
def _combine(Y, WTS, H, NW, HOUT, OUT, XS, eps, D: tl.constexpr, SLOTS: tl.constexpr, FUSED: tl.constexpr):
    """Program r: branch = bf16(sum over slots k in order of fp32 y_k * w_k). FUSED: h = bf16(h + branch) to HOUT,
    OUT = bf16(h * rsqrt(mean(h^2) + eps) * nw), XS = OUT's 64-group sums; else OUT = branch."""

    r = tl.program_id(0)
    d = tl.arange(0, D)
    acc = tl.zeros((D,), dtype=tl.float32)
    for k in tl.static_range(SLOTS):
        wk = tl.load(WTS + r * SLOTS + k)
        yk = tl.load(Y + (r * SLOTS + k) * D + d)
        acc = acc + yk * wk
    branch = acc.to(tl.bfloat16)
    if FUSED:
        x = (tl.load(H + r * D + d).to(tl.float32) + branch.to(tl.float32)).to(tl.bfloat16)
        tl.store(HOUT + r * D + d, x)
        xf = x.to(tl.float32)
        ss = tl.sum(xf * xf, axis=0)
        inv = 1.0 / tl.sqrt(ss / D + eps)
        w = tl.load(NW + d).to(tl.float32)
        y = (xf * inv * w).to(tl.bfloat16)
        tl.store(OUT + r * D + d, y)
        yg = tl.reshape(y.to(tl.float32), (D // 64, 64))
        tl.store(XS + r * (D // 64) + tl.arange(0, D // 64), tl.sum(yg, axis=1))
    else:
        tl.store(OUT + r * D + d, branch)


def _slots(y: torch.Tensor, wts: torch.Tensor) -> tuple[int, int, int]:
    rows, slots, d = y.shape
    if y.dtype != torch.float32 or not y.is_contiguous() or wts.shape != (rows, slots) or not wts.is_contiguous() \
            or wts.dtype != torch.float32:
        raise ValueError("combine: y [R, slots, D] fp32 and wts [R, slots] fp32, both contiguous")
    if d & (d - 1) or d % 64:
        raise ValueError(f"combine: D={d} must be a power of two and a multiple of 64")
    return rows, slots, d


def combine(y: torch.Tensor, wts: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """y [R, k + 1, D] fp32 slots, wts [R, k + 1] -> [R, D] bf16: the slots in order (the routed slots in pick
    order, then the shared expert), summed in fp32 and rounded once."""

    rows, slots, d = _slots(y, wts)
    if out is None:
        out = torch.empty((rows, d), dtype=torch.bfloat16, device=y.device)
    _combine[(rows,)](y, wts, out, wts, out, out, wts, 0.0, D=d, SLOTS=slots, FUSED=False, num_warps=8)
    return out


def combine_add_rmsnorm(y: torch.Tensor, wts: torch.Tensor, h: torch.Tensor, norm: torch.Tensor, eps: float, *,
                        h_out: torch.Tensor | None = None, normed: torch.Tensor | None = None,
                        xs: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The combine, then the residual and the next norm in the same pass: h' = bf16(h + branch) (``h_out``, which
    may be ``h``), normed = bf16(h' * rsqrt(mean(h'^2) + eps) * norm) (``norm``: the fp32 multiplier 1 + w), and
    normed's 64-group sums for the next matmul. Returns (h', normed, xs)."""

    rows, slots, d = _slots(y, wts)
    if h.shape != (rows, d) or h.dtype != torch.bfloat16 or not h.is_contiguous():
        raise ValueError(f"combine_add_rmsnorm: h must be a contiguous bf16 ({rows}, {d})")
    if norm.shape != (d,) or not norm.is_contiguous():
        raise ValueError(f"combine_add_rmsnorm: norm must be a contiguous ({d},)")
    h_out = torch.empty_like(h) if h_out is None else h_out
    normed = torch.empty_like(h) if normed is None else normed
    xs = torch.empty((rows, d // 64), dtype=torch.float32, device=h.device) if xs is None else xs
    _combine[(rows,)](y, wts, h, norm, h_out, normed, xs, float(eps), D=d, SLOTS=slots, FUSED=True, num_warps=8)
    return h_out, normed, xs
