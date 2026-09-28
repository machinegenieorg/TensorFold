"""Qwen3.6-35B-A3B's forward on CUDA over a window of rows per sequence, and the commit that keeps a prefix."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from tensorfold.families.qwen4_exp.cuda import attention as attn_mod
from tensorfold.families.qwen4_exp.cuda import gdn as gdn_mod
from tensorfold.families.qwen4_exp.cuda import glue as fn_glue
from tensorfold.families.qwen4_exp.cuda.forward import shift_windows

from . import moe as moe_mod
from . import qmm
from .checkpoint import Config
from .state import Seq, State
from .weights import QW, Weights

GATE = "silu"          # Qwen3.6's gated RMSNorm: w * norm(y) * silu(z)
GS = qmm.GS            # 64: the quantization group; the small kernels write 64-group sums for the next matmul


# -- the model in the kernels' layout ------------------------------------------------------------------------------
@dataclass
class GDNK:
    proj: qmm.Q4              # [qkv | z | b | a] x hidden: the chain's projection row layout
    conv: torch.Tensor        # [conv_dim, 4] bf16
    a_log: torch.Tensor       # [nv] fp32
    dt_bias: torch.Tensor     # [nv] fp32
    norm: torch.Tensor        # [dv] bf16 (not centred)
    out: qmm.Q4               # hidden x nv*dv


@dataclass
class AttnK:
    proj: qmm.Q4              # [q | gate per head, k, v] x hidden
    q_scale: torch.Tensor     # [head_dim] fp32 (1 + w)
    k_scale: torch.Tensor
    o: qmm.Q4                 # hidden x heads*head_dim


@dataclass
class LayerK:
    index: int
    linear: bool
    slot: int                 # index among the GDN layers (linear) or the attention layers
    input_scale: torch.Tensor     # [hidden] fp32 (1 + w)
    post_scale: torch.Tensor
    gdn: GDNK | None
    attn: AttnK | None
    router: torch.Tensor      # [E + 1, hidden] fp32: router rows, then the shared expert's gate row
    experts: qmm.Experts      # E + 1 experts, the shared one last


@dataclass
class Model:
    cfg: Config
    embed: QW                 # MLX layout as loaded (row gather)
    layers: list[LayerK]
    norm: torch.Tensor        # [hidden] fp32 (1 + w): the final norm
    head: qmm.Q4
    inv_freq: torch.Tensor    # [rotary_dim / 2] fp32

    @property
    def device(self) -> torch.device:
        return self.inv_freq.device

    @property
    def n_linear(self) -> int:
        return sum(1 for layer in self.layers if layer.linear)

    @property
    def n_attention(self) -> int:
        return sum(1 for layer in self.layers if not layer.linear)

    @property
    def moe_cfg(self) -> SimpleNamespace:
        c = self.cfg
        return moe_mod.config(c.experts, c.top_k, c.moe_width, c.hidden)

    def shapes(self) -> list[tuple[int, int]]:
        """(N, K) of every dense matrix (for the split-K scratch)."""

        found = {(self.head.n, self.head.k)}
        for layer in self.layers:
            mats = (layer.gdn.proj, layer.gdn.out) if layer.linear else (layer.attn.proj, layer.attn.o)
            found.update((q.n, q.k) for q in mats)
        return sorted(found)

    def nbytes(self) -> int:
        total = self.embed.nbytes() + self.head.nbytes() + self.norm.numel() * 4
        for layer in self.layers:
            total += (layer.experts.up.numel() + layer.experts.down.numel()) * 4
            total += layer.router.numel() * 4
            mats = (layer.gdn.proj, layer.gdn.out) if layer.linear else (layer.attn.proj, layer.attn.o)
            total += sum(q.nbytes() for q in mats)
        return total


def _check_config(c: Config) -> None:
    wrong = []
    if (c.bits, c.group_size) != (4, GS):
        wrong.append(f"{c.bits}-bit groups of {c.group_size} (the kernels read 4-bit groups of {GS})")
    if (c.nk, c.nv, c.dk, c.dv, c.conv_kernel) not in {(16, 32, 128, 128, 4)}:
        wrong.append(f"GDN heads ({c.nk}, {c.nv}) x ({c.dk}, {c.dv}), conv {c.conv_kernel}")
    if not c.attn_gate or c.heads % c.kv_heads or c.heads // c.kv_heads > 16 or c.rotary_dim % 2:
        wrong.append(f"attention {c.heads}/{c.kv_heads} heads, gate {c.attn_gate}, rotary {c.rotary_dim}")
    if not c.norm_topk or c.shared_width != c.moe_width:
        wrong.append(f"MoE renormalise {c.norm_topk}, shared width {c.shared_width} vs {c.moe_width}")
    if c.hidden % GS or c.hidden & (c.hidden - 1):
        wrong.append(f"hidden {c.hidden} (a power of two, a multiple of {GS})")
    if wrong:
        raise ValueError("the Qwen3.6 forward does not support " + "; ".join(wrong))


def prepare(w: Weights, *, release: bool = True) -> Model:
    """The loader's weights regrouped for the kernels layer by layer; ``release`` frees each loader layer once done."""

    c = w.cfg
    _check_config(c)
    layers: list[LayerK] = []
    n_lin = n_att = 0
    for i in range(len(w.layers)):
        lw = w.layers[i]
        gdn = attn = None
        if lw.linear:
            g = lw.gdn
            gdn = GDNK(qmm.make_q4(*g.proj.triple()), g.conv.contiguous(), g.a_log.float().contiguous(),
                       g.dt_bias.float().contiguous(), g.norm.to(torch.bfloat16).contiguous(),
                       qmm.make_q4(*g.out.triple()))
            slot, n_lin = n_lin, n_lin + 1
        else:
            a = lw.attn
            attn = AttnK(qmm.make_q4(*a.proj.triple()), a.q_scale.float().contiguous(),
                         a.k_scale.float().contiguous(), qmm.make_q4(*a.o.triple()))
            slot, n_att = n_att, n_att + 1
        ex = qmm.make_experts(lw.moe.gate.triple(), lw.moe.up.triple(), lw.moe.down.triple())
        if ex.count != c.experts + 1 or lw.moe.router.shape != (c.experts + 1, c.hidden):
            raise ValueError(f"layer {lw.index}: {ex.count} experts and router {tuple(lw.moe.router.shape)}, expected "
                             f"{c.experts} routed + the shared one")
        layers.append(LayerK(lw.index, lw.linear, slot, lw.input_scale.float().contiguous(),
                             lw.post_scale.float().contiguous(), gdn, attn, lw.moe.router.float().contiguous(), ex))
        del lw
        if release:
            w.layers[i] = None
    head = qmm.make_q4(*w.head.triple())
    if release and not c.tie_embeddings:
        w.head = None
    return Model(c, w.embed, layers, w.norm.float().contiguous(), head, w.inv_freq.float().contiguous())


# -- per-window buffers --------------------------------------------------------------------------------------------
class Buffers:
    """Scratch for windows of up to ``rows`` rows over up to ``seqs`` sequences; views [:R] serve smaller windows."""

    def __init__(self, m: Model, rows: int = 512, *, capacity: int, window_rows: int = 32, attn_rows: int = 64,
                 logit_rows: int = 32, seqs: int = 1) -> None:
        c = m.cfg
        dev = m.device
        bf, f32 = torch.bfloat16, torch.float32
        d = c.hidden
        nl = m.n_linear
        conv, pw = gdn_mod.widths(c.nk, c.nv)
        self.rows = rows
        self.window_rows = wr = min(window_rows, rows)
        self.attn_rows = min(attn_rows, rows)
        self.capacity = capacity
        self.seqs = seqs
        self.logit_rows = min(logit_rows, rows)
        pin = torch.cuda.is_available()
        self.ids = torch.zeros((rows,), dtype=torch.int32, device=dev)
        self.pos = torch.zeros((rows,), dtype=torch.int32, device=dev)      # each row's absolute position
        self.ids_host = torch.zeros((rows,), dtype=torch.int32, pin_memory=pin)
        self.pos_host = torch.zeros((rows,), dtype=torch.int32, pin_memory=pin)
        self.staged = torch.cuda.Event() if pin else None
        self.pending: dict[int, Seq] = {}                                   # staged, not yet committed
        # residual stream, the normed rows the next matmul reads, their 64-group sums, the mixer's output
        self.h = torch.empty((rows, d), dtype=bf, device=dev)
        self.normed = torch.empty((rows, d), dtype=bf, device=dev)
        self.xs = torch.empty((rows, d // GS), dtype=f32, device=dev)
        self.branch = torch.empty((rows, d), dtype=bf, device=dev)
        # Gated DeltaNet: projection rows and replay inputs per layer (windows), or one shared buffer (prefill chunks)
        flat = torch.empty((max(nl * wr, rows) * pw,), dtype=bf, device=dev)
        self.proj = flat[:nl * wr * pw].view(nl, wr, pw)
        self.proj_chunk = flat[:rows * pw].view(rows, pw)
        self.rk = torch.empty((nl, wr, c.nk, c.dk), dtype=f32, device=dev)
        self.rv = torch.empty((nl, wr, c.nv, c.dv), dtype=bf, device=dev)
        self.rg = torch.empty((nl, wr, c.nv), dtype=f32, device=dev)
        self.rb = torch.empty((nl, wr, c.nv), dtype=f32, device=dev)
        self.tail = torch.empty((seqs, nl, c.conv_kernel - 1, conv), dtype=bf, device=dev)
        self.none_f32 = torch.empty((0,), dtype=f32, device=dev)
        self.none_bf = torch.empty((0,), dtype=bf, device=dev)
        self.gdn_out = torch.empty((rows, c.nv * c.dv), dtype=bf, device=dev)
        self.gdn_xs = torch.empty((rows, c.nv * c.dv // GS), dtype=f32, device=dev)
        # attention
        self.pa = torch.empty((rows, sum(c.attn_rows)), dtype=bf, device=dev)
        self.q = torch.empty((rows, c.heads, c.head_dim), dtype=bf, device=dev)
        self.attn = attn_mod.AttnScratch(self.attn_rows, c.heads, c.head_dim, capacity, dev, sparse=False)
        self.gated = torch.empty((rows, c.heads * c.head_dim), dtype=bf, device=dev)
        self.gated_xs = torch.empty((rows, c.heads * c.head_dim // GS), dtype=f32, device=dev)
        # MoE, split-K partials, head
        self.moe = moe_mod.buffers(rows, dev, m.moe_cfg)
        self.part = torch.empty((max(1, qmm.split_scratch(rows, m.shapes())),), dtype=f32, device=dev)
        self.logits = torch.empty((max(self.logit_rows, seqs), m.head.n), dtype=bf, device=dev)
        self.last = torch.empty((seqs, d), dtype=bf, device=dev)
        self.last_xs = torch.empty((seqs, d // GS), dtype=f32, device=dev)

    @property
    def hidden(self) -> torch.Tensor:
        """After a forward: every row's hidden state after the final norm (what an MTP head reads), [rows, hidden]."""

        return self.normed

    def nbytes(self) -> int:
        seen, total = set(), 0
        parts = (vars(self), vars(self.moe), vars(self.moe.plan), vars(self.attn))
        for v in [v for part in parts for v in part.values()]:
            if isinstance(v, torch.Tensor) and v.is_cuda:
                base = v.untyped_storage().data_ptr()
                if base not in seen:
                    seen.add(base)
                    total += v.untyped_storage().nbytes()
        return total


class _Rows:
    """The replay inputs ``gdn.chain`` writes and ``gdn.replay`` reads, as views of the window's buffers."""

    __slots__ = ("k", "v", "g", "b")

    def __init__(self, k, v, g, b) -> None:
        self.k, self.v, self.g, self.b = k, v, g, b


# -- small kernels ------------------------------------------------------------------------------------------------
@triton.jit
def _add_norm(H, BR, W, HOUT, Y, XS, eps, RS, D: tl.constexpr, MODE: tl.constexpr, SK: tl.constexpr):
    """Program r: x = h (MODE 0), bf16(h + branch) (1) or bf16(h + SK slices in order) (2); Y = RMSNorm(x) w, XS."""

    r = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, D)
    x = tl.load(H + r * D + d).to(tl.float32)
    if MODE != 0:
        if MODE == 1:
            br = tl.load(BR + r * D + d).to(tl.float32)
        else:
            acc = tl.load(BR + r * D + d)
            for s in tl.static_range(1, SK):
                acc = acc + tl.load(BR + s * RS + r * D + d)
            br = acc.to(tl.bfloat16).to(tl.float32)
        xb = (x + br).to(tl.bfloat16)
        tl.store(HOUT + r * D + d, xb)
        x = xb.to(tl.float32)
    ss = tl.sum(x * x, axis=0)
    inv = 1.0 / tl.sqrt(ss / D + eps)
    w = tl.load(W + d).to(tl.float32)
    y = (x * inv * w).to(tl.bfloat16)
    tl.store(Y + r * D + d, y)
    yg = tl.reshape(y.to(tl.float32), (D // 64, 64))
    tl.store(XS + r * (D // 64) + tl.arange(0, D // 64), tl.sum(yg, axis=1))


def _norm(b: Buffers, R: int, scale: torch.Tensor, eps: float, branch: torch.Tensor | None = None) -> None:
    """b.normed / b.xs from b.h (after adding ``branch`` to b.h in place: bf16 [R, D], or K slices [SK, R, D])."""

    d = b.h.shape[1]
    if branch is None:
        mode, sk, rs, br = 0, 1, 0, b.h
    elif branch.dim() == 3:
        mode, sk, rs, br = 2, branch.shape[0], branch.stride(0), branch
    else:
        mode, sk, rs, br = 1, 1, 0, branch
    _add_norm[(R,)](b.h, br, scale, b.h, b.normed, b.xs, float(eps), rs, D=d, MODE=mode, SK=sk, num_warps=8)


@triton.jit
def _conv_tail(OLD, NEW, OUT, n, NEW_ROW, C: tl.constexpr, T: tl.constexpr, TP: tl.constexpr, BLOCK: tl.constexpr):
    """OUT[j] (j < T) = row n + j of [OLD (T rows); NEW (n rows, first C columns)]: the conv window after n rows."""

    ch = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    j = tl.arange(0, TP)
    src = n + j
    from_old = src < T
    ok = j < T
    old = tl.load(OLD + tl.where(from_old, src, 0)[:, None] * C + ch[None, :], mask=(ok & from_old)[:, None],
                  other=0.0)
    new = tl.load(NEW + tl.where(from_old, 0, src - T)[:, None] * NEW_ROW + ch[None, :],
                  mask=(ok & ~from_old)[:, None], other=0.0)
    tl.store(OUT + j[:, None] * C + ch[None, :], tl.where(from_old[:, None], old, new), mask=ok[:, None])


def _tail(old: torch.Tensor, new: torch.Tensor, out: torch.Tensor, n: int) -> None:
    taps, ch = old.shape
    block = 256
    _conv_tail[(triton.cdiv(ch, block),)](old, new, out, n, new.stride(0), C=ch, T=taps,
                                           TP=triton.next_power_of_2(taps), BLOCK=block, num_warps=4)


# -- blocks --------------------------------------------------------------------------------------------------------
def _gdn_block(m: Model, layer: LayerK, b: Buffers, R: int, table: Sequence[Seq], full: bool) -> torch.Tensor:
    """Gated DeltaNet on b.normed[:R]: projection once, the chain per sequence, out projection (or its K slices)."""

    c = m.cfg
    g = layer.gdn
    li = layer.slot
    p = b.proj[li, :R] if full else b.proj_chunk[:R]
    qmm.matmul(b.normed[:R], g.proj, b.xs[:R], out=p, part=b.part)
    for s in table:
        st = s.state
        lo, hi = s.row0, s.row0 + s.rows
        if full:
            rows = _Rows(b.rk[li, lo:hi], b.rv[li, lo:hi], b.rg[li, lo:hi], b.rb[li, lo:hi])
        else:
            rows = _Rows(b.none_f32, b.none_bf, b.none_f32, b.none_f32)
        gdn_mod.chain(p[lo:hi], st.conv[li], g.conv, st.rec[st.cur, li], g.a_log, g.dt_bias, g.norm, c.eps, s.rows,
                      rows, st.rec[1 - st.cur, li], b.gdn_out[lo:hi], None, gate=GATE)
        if not full:
            _tail(st.conv[li], p[lo:hi], b.tail[s.index, li], s.rows)
    qmm.group_sums(b.gdn_out[:R], out=b.gdn_xs[:R])
    return qmm.matmul(b.gdn_out[:R], g.out, b.gdn_xs[:R], out=b.branch[:R], part=b.part, reduce=False)


def _attn_block(m: Model, layer: LayerK, b: Buffers, R: int, table: Sequence[Seq], full: bool,
                context: int | None) -> torch.Tensor:
    """Gated attention on b.normed[:R]: projection once, per sequence prep and attention in row blocks, o projection."""

    c = m.cfg
    a = layer.attn
    ai = layer.slot
    qmm.matmul(b.normed[:R], a.proj, b.xs[:R], out=b.pa[:R], part=b.part)
    block = b.attn_rows
    scale = c.head_dim ** -0.5
    for s in table:
        st = s.state
        kc, vc = st.kc[ai], st.vc[ai]
        lo, hi = s.row0, s.row0 + s.rows
        fn_glue.attn_prep(b.pa[lo:hi], b.pos[lo:lo + 1], a.q_scale, a.k_scale, None, m.inv_freq, b.q[lo:hi], kc, vc,
                          None, None, c.eps, q_heads=c.heads, kv_heads=c.kv_heads, head_dim=c.head_dim)
        for r0 in range(lo, hi, block):
            n = min(block, hi - r0)
            keys = context if full else s.pos + (r0 - lo) + n          # a prompt chunk's blocks: only their keys
            o = attn_mod.attention(b.q[r0:r0 + n], kc, vc, b.pos[r0:r0 + 1], b.attn, n, scale, context=keys)
            fn_glue.attn_gate(o[:n], b.pa[r0:r0 + n], b.gated[r0:r0 + n], b.gated_xs[r0:r0 + n], q_heads=c.heads,
                              head_dim=c.head_dim, group=GS)
    return qmm.matmul(b.gated[:R], a.o, b.gated_xs[:R], out=b.branch[:R], part=b.part, reduce=False)


def _moe_block(m: Model, layer: LayerK, b: Buffers, R: int, next_scale: torch.Tensor) -> None:
    """h += MoE(normed) (routed slots in pick order, then the shared expert, one rounding), then the next norm."""

    c = m.cfg
    sub = moe_mod.moe(b.normed[:R], b.xs[:R], layer.router, layer.experts, b.moe, top_k=c.top_k, experts=c.experts)
    moe_mod.combine_add_rmsnorm(sub.y, sub.wts, b.h[:R], next_scale, c.eps, h_out=b.h[:R], normed=b.normed[:R],
                                xs=b.xs[:R])


def _check_logits(b: Buffers, R: int, logits: str) -> None:
    if logits not in ("all", "last", "none"):
        raise ValueError(f"logits: 'all', 'last' or 'none', not {logits!r}")
    if logits == "all" and R > b.logit_rows:
        raise ValueError(f"logits of {R} rows, the buffers hold {b.logit_rows} (ask for 'last' or 'none')")


def _head(m: Model, b: Buffers, R: int, table: Sequence[Seq], logits: str) -> torch.Tensor | None:
    if logits == "none":
        return None
    if logits == "all":
        x, xs, n = b.normed[:R], b.xs[:R], R
    elif logits == "last":
        n = len(table)
        if n == 1:
            r = table[0].row0 + table[0].rows - 1
            x, xs = b.normed[r:r + 1], b.xs[r:r + 1]
        else:
            for i, s in enumerate(table):
                r = s.row0 + s.rows - 1
                b.last[i].copy_(b.normed[r])
                b.last_xs[i].copy_(b.xs[r])
            x, xs = b.last[:n], b.last_xs[:n]
    return qmm.matmul(x, m.head, xs, out=b.logits[:n], part=b.part)


# -- forward -------------------------------------------------------------------------------------------------------
def stage(m: Model, b: Buffers, work: Sequence[tuple[State, Sequence[int]]]) -> tuple[int, list[Seq]]:
    """Stage each sequence's tokens and positions into the static device buffers; returns the rows and the table."""

    if not work:
        raise ValueError("a window needs at least one sequence")
    if len(work) > b.seqs:
        raise ValueError(f"{len(work)} sequences in a window, the buffers hold {b.seqs}")
    total = sum(len(t) for _, t in work)
    if total > b.rows:
        raise ValueError(f"window of {total} rows, the buffers hold {b.rows}")
    full = total <= b.window_rows
    table: list[Seq] = []
    ids, pos = [], []
    R = 0
    for i, (st, tokens) in enumerate(work):
        n = len(tokens)
        if n == 0:
            raise ValueError("every sequence of a window needs at least one row")
        if any(s.state is st for s in table):
            raise ValueError("a sequence appears twice in one window")
        if st.pos + n > min(st.capacity, b.capacity):
            raise ValueError(f"context of {st.pos + n} past the cache capacity {min(st.capacity, b.capacity)}")
        table.append(Seq(st, i, R, n, st.pos, full))
        ids.append(np.asarray(tokens, dtype=np.int32))
        pos.append(np.arange(st.pos, st.pos + n, dtype=np.int32))
        R += n
    if b.staged is not None:
        b.staged.synchronize()               # the previous window's copies out of the pinned buffers are done
    b.ids_host[:R].numpy()[:] = np.concatenate(ids)
    b.pos_host[:R].numpy()[:] = np.concatenate(pos)
    b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
    b.pos[:R].copy_(b.pos_host[:R], non_blocking=True)
    if b.staged is not None:
        b.staged.record()
    b.pending = {id(s.state): s for s in table}
    return R, table


def compute(m: Model, b: Buffers, R: int, table: Sequence[Seq], *, logits: str = "all",
            context: int | None = None) -> torch.Tensor | None:
    """The GPU work of a staged forward: logits "all", "last" (per sequence) or "none"; ``context`` bounds the keys."""

    _check_logits(b, R, logits)
    if context is not None and any(s.pos + s.rows > context for s in table):
        raise ValueError(f"context {context} is below a window row's keys")
    c = m.cfg
    full = R <= b.window_rows
    qmm.embed(b.ids[:R], *m.embed.triple(), out=b.h[:R])
    _norm(b, R, m.layers[0].input_scale, c.eps)
    last = len(m.layers) - 1
    for i, layer in enumerate(m.layers):
        if layer.linear:
            branch = _gdn_block(m, layer, b, R, table, full)
        else:
            branch = _attn_block(m, layer, b, R, table, full, context)
        _norm(b, R, layer.post_scale, c.eps, branch)
        _moe_block(m, layer, b, R, m.layers[i + 1].input_scale if i < last else m.norm)
    return _head(m, b, R, table, logits)


@torch.no_grad()
def forward(m: Model, b: Buffers, st: State, tokens: Sequence[int], *, logits: str = "all") -> torch.Tensor | None:
    """One sequence's rows for ``tokens`` from st.pos on (see ``compute``); the state is unchanged until ``commit``."""

    _check_logits(b, len(tokens), logits)
    R, table = stage(m, b, [(st, tokens)])
    return compute(m, b, R, table, logits=logits)


@torch.no_grad()
def forward_many(m: Model, b: Buffers, work: Sequence[tuple[State, Sequence[int]]], *,
                 logits: str = "all") -> tuple[torch.Tensor | None, list[Seq]]:
    """Several sequences' rows in one window (rows in the order given): logits and the table (row offsets)."""

    _check_logits(b, sum(len(t) for _, t in work), logits)
    R, table = stage(m, b, work)
    return compute(m, b, R, table, logits=logits), table


# -- commit --------------------------------------------------------------------------------------------------------
@torch.no_grad()
def commit(m: Model, b: Buffers, st: State, keep: int) -> None:
    """Keep the first ``keep`` rows of ``st``'s rows in the last window these buffers ran (before the next one)."""

    s = b.pending.get(id(st))
    if s is None or s.state is not st:
        raise ValueError("commit: these buffers hold no uncommitted rows of this sequence")
    if st.pos != s.pos:
        raise ValueError("commit: the sequence moved since its window was staged")
    if not 1 <= keep <= s.rows:
        raise ValueError(f"keep must be in 1..{s.rows}, not {keep}")
    if not s.full and keep != s.rows:
        raise ValueError(f"a window of more than {b.window_rows} rows (a prefill chunk) keeps all its rows")
    del b.pending[id(st)]
    cur = st.cur
    lo, hi = s.row0, s.row0 + s.rows
    if st.rec.shape[1]:
        if s.full:
            if keep < s.rows:
                for li in range(st.rec.shape[1]):
                    rows = _Rows(b.rk[li, lo:hi], b.rv[li, lo:hi], b.rg[li, lo:hi], b.rb[li, lo:hi])
                    gdn_mod.replay(st.rec[cur, li], rows, keep, st.rec[1 - cur, li])
            shift_windows(st.conv, b.proj[:, lo:hi], keep, m.cfg.conv_dim)
        else:
            st.conv.copy_(b.tail[s.index])
    st.cur = 1 - cur
    st.pos += keep
