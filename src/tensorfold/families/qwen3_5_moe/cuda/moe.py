"""Qwen3.6-35B-A3B's MoE on CUDA: the fp32 router, Flash Next's top-k, the shared grouped experts, the combine."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from tensorfold.cuda import experts as grouped
from tensorfold.families.qwen4_exp.cuda import moe as fn_moe
from tensorfold.families.qwen4_exp.cuda.moe import MoEBuffers

EXPERTS = 256             # routed experts; the shared expert is expert EXPERTS of the table
TOP_K = 8
WIDTH = 512               # an expert's intermediate width (routed and shared)
HIDDEN = 2048


def config(experts: int = EXPERTS, top_k: int = TOP_K, width: int = WIDTH, hidden: int = HIDDEN) -> SimpleNamespace:
    """The fields ``MoEBuffers`` reads, under Flash Next's names."""

    return SimpleNamespace(num_experts=experts, num_experts_per_tok=top_k, moe_intermediate_size=width,
                           hidden_size=hidden)


def buffers(rows: int, device: torch.device | str, cfg: SimpleNamespace | None = None) -> MoEBuffers:
    """Flash Next's static MoE scratch (decode form) for windows of up to ``rows`` rows."""

    return MoEBuffers(rows, cfg or config(), device)


class Rows:
    """The first ``rows`` rows of a ``MoEBuffers``: logits, pick, wts, act, y (views) and the buffer's plan."""

    __slots__ = ("rows", "logits", "pick", "wts", "act", "y", "plan")

    def __init__(self, buf: MoEBuffers, rows: int) -> None:
        self.rows, self.plan = rows, buf.plan
        self.logits, self.pick, self.wts = buf.logits[:rows], buf.pick[:rows], buf.wts[:rows]
        self.act, self.y = buf.act[:rows], buf.y[:rows]


def rows_view(buf: MoEBuffers, rows: int) -> Rows:
    """``buf`` restricted to its first ``rows`` rows, cached on the buffer."""

    if not 0 < rows <= buf.rows:
        raise ValueError(f"rows_view: {rows} rows in a buffer for {buf.rows}")
    views = buf.__dict__.setdefault("_views", {})
    view = views.get(rows)
    if view is None:
        view = views[rows] = Rows(buf, rows)
    return view


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


# (up to this many rows: rows a program, experts a program, K a step, warps, stages), timed on an RTX 5090
ROUTER_CFG = ((128, (16, 16, 64, 4, 3)), (256, (32, 16, 64, 2, 3)), (1 << 31, (64, 16, 64, 2, 3)))


def router(x: torch.Tensor, rows: torch.Tensor, out: torch.Tensor | None = None, *, block_m: int | None = None,
           block_e: int | None = None, bk: int | None = None, num_warps: int | None = None,
           num_stages: int | None = None) -> torch.Tensor:
    """x [R, D] bf16 (rows may be strided) and the fp32 table [E + 1, D] -> fp32 logits [R, E + 1]."""

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
    c_bm, c_be, c_bk, c_w, c_s = next(cfg for top, cfg in ROUTER_CFG if m <= top)
    bm, be, bk = block_m or c_bm, block_e or c_be, bk or c_bk
    if d % bk:
        raise ValueError(f"router: K={d} is not a multiple of the step {bk}")
    grid = (triton.cdiv(m, bm), triton.cdiv(ne, be))
    _router[grid](x, rows, out, m, x.stride(0), D=d, NE=ne, BM=bm, BLOCK_E=be, BK=bk,
                  num_warps=num_warps or c_w, num_stages=num_stages or c_s)
    return out


def select(logits: torch.Tensor, buf: MoEBuffers, top_k: int = TOP_K, experts: int = EXPERTS) -> None:
    """Each row's experts and weights, then the (row, slot) pairs grouped by expert: Flash Next's ``select``."""

    fn_moe.select(logits, buf, top_k, experts)


def moe(x: torch.Tensor, xs: torch.Tensor | None, router_rows: torch.Tensor, ex: grouped.Experts, buf: MoEBuffers,
        *, top_k: int = TOP_K, experts: int = EXPERTS) -> Rows:
    """Route rows x [R, D] and run their experts; returns the R-row view of ``buf`` (y, wts, pick; slot k shared)."""

    rows = x.shape[0]
    view = rows_view(buf, rows)
    router(x, router_rows, view.logits)
    select(view.logits, buf, top_k, experts)
    act = buf.act.view(-1, ex.width)
    grouped.gate_up(x, ex, buf.plan, act, rows)
    grouped.down(act, ex, buf.plan, buf.y.view(-1, ex.dims), rows)
    return view


# -- combine -------------------------------------------------------------------------------------------------
@triton.jit
def _combine(Y, WTS, H, NW, HOUT, OUT, XS, eps, D: tl.constexpr, SLOTS: tl.constexpr, FUSED: tl.constexpr):
    """Program r: branch = bf16(sum of y_k w_k in slot order); FUSED adds it to h and writes the next RMSNorm, XS."""

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
    """y [R, k + 1, D] fp32 and wts [R, k + 1] -> [R, D] bf16: the slots summed in order in fp32, rounded once."""

    rows, slots, d = _slots(y, wts)
    if out is None:
        out = torch.empty((rows, d), dtype=torch.bfloat16, device=y.device)
    _combine[(rows,)](y, wts, out, wts, out, out, wts, 0.0, D=d, SLOTS=slots, FUSED=False, num_warps=8)
    return out


def combine_add_rmsnorm(y: torch.Tensor, wts: torch.Tensor, h: torch.Tensor, norm: torch.Tensor, eps: float, *,
                        h_out: torch.Tensor | None = None, normed: torch.Tensor | None = None,
                        xs: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The combine, the residual add and the next RMSNorm (fp32 1 + w) with its 64-group sums: (h', normed, xs)."""

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
