"""Qwen3.6-35B-A3B forward on CUDA for a window of R consecutive rows per sequence (1 = a serial step, 2 to
``window_rows`` = a verify window, up to ``Buffers.rows`` = a prefill chunk), and the commit that keeps a prefix.

The model is plain pre-norm: embed, then per layer

    normed = RMSNorm(h) (1 + w)            fp32, one bf16 rounding (fused into the previous layer's combine)
    h      = h + mixer(normed)             Gated DeltaNet (30 layers) or gated attention (10 layers)
    normed = RMSNorm(h) (1 + w_post)       fused with the residual add (and the out projection's K slices)
    h      = h + MoE(normed)               256 routed experts (top 8) and the shared expert
    normed = RMSNorm(h) (1 + w_next)       the next layer's input norm, or the final norm after the last layer

and the head over the requested rows. Every kernel treats each row on its own (``qmm``, ``moe``, the GDN chain,
attention and the small kernels below), so row r of a window gets the bits of the serial step at its position.

Phase-2 seam: a window is a table of sequences (``Seq``: a state, its rows' offset and count, its first position).
The row-shared kernels (every matmul, the router, the experts, the norms) run once over all R rows; the kernels that
read a sequence's committed state (the GDN chain, attention's cache write and reads) run once per sequence on its
rows. Today's decoders pass one sequence; nothing below assumes it.

The committed state (``State``, from a ``Pool``) is read-only during a forward, except for attention cache rows at
and past the committed length (the window's keys, overwritten by a later window). ``commit`` keeps a sequence's first
``keep`` rows:

- Gated DeltaNet: the forward writes the state after the sequence's last row into the layer's other state buffer; a
  shorter keep replays the kept rows from the committed state into it (``gdn.replay``, the chain's update routine,
  the same bits). The state's buffer parity flips.
- conv windows: rows [keep, keep + 3) of [old window; the rows' q | k | v].
- attention: keys and values were written at their absolute positions; the committed length advances by ``keep``.

A window of at most ``window_rows`` rows keeps what a partial keep needs (each GDN layer's projection rows and the
replay inputs). A longer window is a prefill chunk: it must keep every row, so it saves no replay inputs and writes
each layer's next conv window during the forward (``Buffers.tail``), which bounds the scratch at 512-row chunks.
Attention runs a long window in row blocks of ``attn_rows`` (each row's keys are chunked by absolute position and
merged in order, so the blocking changes no bits), which bounds its fp32 partials at any context; a prefill chunk's
blocks launch only the key chunks their rows read (a window launches every chunk up to the capacity, so a captured
graph stays valid as the sequence grows, and chunks past a row's keys return at once).

No cuBLAS, ``torch.matmul`` or ``F.linear`` anywhere: their algorithms depend on the row count.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from ...qwen4_exp.cuda import attention as attn_mod
from ...qwen4_exp.cuda import gdn as gdn_mod
from ...qwen4_exp.cuda import glue as fn_glue
from ...qwen4_exp.cuda.forward import shift_windows
from . import moe as moe_mod
from . import qmm
from .weights import QW, Config, Weights

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
        return sum(1 for l in self.layers if l.linear)

    @property
    def n_attention(self) -> int:
        return sum(1 for l in self.layers if not l.linear)

    @property
    def moe_cfg(self) -> SimpleNamespace:
        c = self.cfg
        return moe_mod.config(c.experts, c.top_k, c.moe_width, c.hidden)

    def shapes(self) -> list[tuple[int, int]]:
        """(N, K) of every dense matrix (for the split-K scratch)."""

        found = {(self.head.n, self.head.k)}
        for l in self.layers:
            mats = (l.gdn.proj, l.gdn.out) if l.linear else (l.attn.proj, l.attn.o)
            found.update((q.n, q.k) for q in mats)
        return sorted(found)

    def nbytes(self) -> int:
        total = self.embed.nbytes() + self.head.nbytes() + self.norm.numel() * 4
        for l in self.layers:
            total += (l.experts.up.numel() + l.experts.down.numel()) * 4
            total += l.router.numel() * 4
            total += (l.gdn.proj.nbytes() + l.gdn.out.nbytes()) if l.linear else (l.attn.proj.nbytes()
                                                                                   + l.attn.o.nbytes())
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
    """The loader's weights (MLX packing) regrouped for the kernels, layer by layer. ``release`` drops each loader
    layer (and the head) once converted: the MLX arrays and their regrouped copies do not both fit next to each other
    on a 32 GB GPU. The embedding stays in MLX's layout (a row gather)."""

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


# -- committed state -----------------------------------------------------------------------------------------------
class Pool:
    """Committed caches for up to ``seqs`` sequences, allocated once: per sequence the GDN states of every linear layer
    (two buffers: a forward writes the next state into the one it does not read), the conv windows, and the key and
    value caches of every attention layer up to ``capacity`` positions (plus ``mtp_layers`` more for a draft head).
    ``alloc`` hands out a slot as a ``State`` of views."""

    def __init__(self, m: Model, seqs: int, capacity: int, *, mtp_layers: int = 0) -> None:
        c = m.cfg
        dev = m.device
        nl, na = m.n_linear, m.n_attention
        self.capacity = capacity
        self.mtp_layers = mtp_layers
        self.rec = torch.zeros((seqs, 2, nl, c.nv, c.dv, c.dk), dtype=torch.float32, device=dev)
        self.conv = torch.zeros((seqs, nl, c.conv_kernel - 1, c.conv_dim), dtype=torch.bfloat16, device=dev)
        self.kc = torch.zeros((seqs, na + mtp_layers, capacity, c.kv_heads, c.head_dim), dtype=torch.bfloat16,
                              device=dev)
        self.vc = torch.zeros_like(self.kc)
        # with a draft head: each sequence's last hidden row the head has not absorbed yet (``mtp.py``)
        self.tail = torch.zeros((seqs, c.hidden), dtype=torch.bfloat16, device=dev) if mtp_layers else None
        self.free = list(range(seqs - 1, -1, -1))

    def alloc(self) -> "State":
        if not self.free:
            raise RuntimeError("state pool exhausted")
        st = State(self, self.free.pop())
        st.reset()
        return st

    def release(self, st: "State") -> None:
        if st.slot in self.free:
            raise ValueError("state released twice")
        self.free.append(st.slot)

    def clone(self, st: "State") -> "State":
        """A new state holding a copy of ``st``'s committed sequence."""

        other = self.alloc()
        other.copy_(st)
        return other

    def nbytes_per_seq(self) -> int:
        return sum(t[0].numel() * t.element_size() for t in (self.rec, self.conv, self.kc, self.vc))


class State:
    """One sequence's committed caches: views of its pool slot, its length ``pos`` and GDN buffer parity ``cur``.

    With a draft head (``Pool(mtp_layers=1)``) it also carries the head's bookkeeping (``mtp.py``): ``mtp_len``, the
    positions the head's cache has absorbed, and ``mtp_tail``, the hidden row of position ``mtp_tail_at`` (-1: none)
    that the head absorbs once the next token is known."""

    def __init__(self, pool: Pool, slot: int) -> None:
        self.pool, self.slot = pool, slot
        self.capacity = pool.capacity
        self.rec = pool.rec[slot]             # [2, linear layers, nv, dv, dk] fp32
        self.conv = pool.conv[slot]           # [linear layers, taps - 1, conv_dim] bf16
        self.kc = pool.kc[slot]               # [attention layers (+ MTP), capacity, kv_heads, head_dim] bf16
        self.vc = pool.vc[slot]
        n_att = self.kc.shape[0] - pool.mtp_layers
        self.mtp_kc = self.kc[n_att:]         # the draft head's caches (empty without one)
        self.mtp_vc = self.vc[n_att:]
        self.mtp_tail = pool.tail[slot] if pool.tail is not None else None
        self.cur = 0
        self.pos = 0
        self.mtp_len = 0
        self.mtp_tail_at = -1

    def reset(self) -> None:
        """An empty sequence (cache rows need no clearing: only rows below ``pos`` are ever read)."""

        self.cur = 0
        self.rec[0].zero_()
        self.conv.zero_()
        self.pos = 0
        self.mtp_len = 0
        self.mtp_tail_at = -1

    def copy_(self, src: "State") -> None:
        """Copy ``src``'s committed sequence: its current GDN states, conv windows, and cache rows below its length."""

        if src.kc.shape != self.kc.shape:
            raise ValueError("states from pools of different shapes")
        self.cur = 0
        self.rec[0].copy_(src.rec[src.cur])
        self.conv.copy_(src.conv)
        p = src.pos
        if p:
            self.kc[:, :p].copy_(src.kc[:, :p])
            self.vc[:, :p].copy_(src.vc[:, :p])
        self.pos = p
        self.mtp_len, self.mtp_tail_at = min(src.mtp_len, p), src.mtp_tail_at
        if self.mtp_tail is not None and src.mtp_tail is not None:
            self.mtp_tail.copy_(src.mtp_tail)
        else:
            self.mtp_tail_at = -1

    def snapshot(self) -> dict:
        """What the sequence keeps outside its cache rows (GDN states, conv windows, length, the draft head's
        bookkeeping): with the rows below ``pos`` still in place, ``restore`` brings the sequence back (about 64 MB
        for 30 GDN layers)."""

        tail = self.mtp_tail.clone() if self.mtp_tail is not None else None
        return {"pos": self.pos, "rec": self.rec[self.cur].clone(), "conv": self.conv.clone(),
                "mtp": (self.mtp_len, self.mtp_tail_at, tail)}

    def restore(self, snap: dict) -> None:
        if snap["pos"] > self.capacity:
            raise ValueError("snapshot longer than this state's capacity")
        self.cur = 0
        self.rec[0].copy_(snap["rec"])
        self.conv.copy_(snap["conv"])
        self.pos = int(snap["pos"])
        mtp_len, tail_at, tail = snap.get("mtp", (0, -1, None))
        self.mtp_len, self.mtp_tail_at = min(int(mtp_len), self.pos), int(tail_at)
        if self.mtp_tail is not None and tail is not None:
            self.mtp_tail.copy_(tail)
        else:
            self.mtp_tail_at = -1


@dataclass
class Seq:
    """One sequence's rows of a staged window."""

    state: State
    index: int                # its place in the window's table
    row0: int                 # its first row in the window
    rows: int
    pos: int                  # the position of its first row (its committed length when staged)
    full: bool                # a window of at most ``window_rows`` rows: any prefix may be kept


# -- per-window buffers --------------------------------------------------------------------------------------------
class Buffers:
    """Scratch for windows of up to ``rows`` rows over up to ``seqs`` sequences (views [:R] serve smaller windows).

    ``capacity``: the longest sequence attention may read (at least every state's capacity). ``window_rows``: the
    widest window that may keep any prefix. ``attn_rows``: rows per attention block. ``logit_rows``: the most rows
    whose logits one forward returns."""

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
        # residual stream, the normed rows the next matmul reads, their 64-group sums; the mixer's output
        self.h = torch.empty((rows, d), dtype=bf, device=dev)
        self.normed = torch.empty((rows, d), dtype=bf, device=dev)
        self.xs = torch.empty((rows, d // GS), dtype=f32, device=dev)
        self.branch = torch.empty((rows, d), dtype=bf, device=dev)
        # Gated DeltaNet: per-layer projection rows and replay inputs for windows (a partial keep needs them), one
        # shared projection buffer (the same memory) and the next conv windows for prefill chunks
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
    """Program r. MODE 0: x = h. MODE 1: x = bf16(h + branch), a bf16 branch. MODE 2: the branch is a matmul's SK fp32
    K slices RS apart, added in slice order and rounded once (the bits of ``qmm``'s reduce), then x = bf16(h + branch).
    Modes 1 and 2 store x to HOUT. Y = bf16(x * rsqrt(mean(x^2) + eps) * w) (fp32 math, w the fp32 1 + w), XS its
    64-group sums: the arithmetic of ``moe.combine_add_rmsnorm``."""

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
    """OUT[j] (j < T) = row n + j of [OLD (T rows); NEW (n rows, the first C columns)]: the conv window after all
    n rows."""

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
    """Gated DeltaNet on b.normed[:R]: the stacked projection once over all rows, the chain per sequence, the out
    projection once. Returns the out projection (bf16 [R, D] or its K slices [SK, R, D])."""

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
    """Gated attention on b.normed[:R]: the stacked projection once; per sequence the q/k norms, RoPE and the cache
    write at the rows' positions, then attention and the output gate in blocks of ``attn_rows`` rows; the o
    projection once."""

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
    """Host work before a forward: each sequence's tokens and positions (its committed length onward) into the static
    device buffers (pinned, async). Returns the window's rows and its table."""

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
    """The GPU work of a forward on staged rows. ``logits``: "all" rows ([R, V]), "last" (each sequence's last row,
    [sequences, V]) or "none". The returned logits are a view of b.logits; b.hidden[:R] holds the rows' hidden
    states after the final norm. ``context`` (a captured graph's bucket): a bound on every window row's keys, so
    attention launches only the key chunks below it (chunks past a row's keys write nothing either way, so the bits
    are the same); None launches every chunk up to the capacity."""

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
    """One sequence's rows for ``tokens`` at positions st.pos, st.pos + 1, ...: logits (see ``compute``). The
    committed state is unchanged until ``commit``."""

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
