"""Qwen3.6-35B-A3B's MTP head on CUDA (the drafter's weights) through the family's kernels, and its draft head."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from tensorfold.families.qwen4_exp.cuda import attention as attn_mod
from tensorfold.families.qwen4_exp.cuda import glue as fn_glue
from tensorfold.families.qwen4_exp.cuda import weights as fn_weights

from . import moe as moe_mod
from . import qmm
from .forward import GS, AttnK, LayerK, Model, _norm
from .state import State
from .weights import MTPW

DRAFT_VOCAB = Path(__file__).with_name("draft_vocab.txt")


@dataclass
class MTPK:
    """The MTP head in the kernels' layout."""

    norm_e: torch.Tensor      # [hidden] fp32 (1 + w): pre_fc_norm_embedding
    norm_h: torch.Tensor      # [hidden] fp32 (1 + w): pre_fc_norm_hidden
    fc: qmm.Q4                # hidden x 2 hidden: input [normed embedding | normed hidden]
    layer: LayerK             # attention (slot: the head's cache in the pool) + MoE (257-expert table)
    norm: torch.Tensor        # [hidden] fp32 (1 + w)
    head: qmm.Q4              # the draft head: the model's head rows at ``ids`` (or the whole head)
    ids: torch.Tensor | None  # int64 [head rows]: each draft-head row's token id (None: the whole vocabulary)
    ids_host: np.ndarray | None

    def nbytes(self) -> int:
        ex = self.layer.experts
        total = self.fc.nbytes() + self.layer.attn.proj.nbytes() + self.layer.attn.o.nbytes()
        total += (ex.up.numel() + ex.down.numel()) * 4
        return total + self.layer.router.numel() * 4 + self.head.nbytes()


def draft_token_ids(draft_vocab: int | str | Path | Sequence[int] | None) -> np.ndarray | None:
    """The draft head's token ids, sorted: Flash Next's rule ("default" is this family's list), or a sequence of ids."""

    if isinstance(draft_vocab, (list, tuple, range, np.ndarray)):
        return np.unique(np.asarray(list(draft_vocab), dtype=np.int64))
    return fn_weights.draft_token_ids(str(DRAFT_VOCAB) if draft_vocab == "default" else draft_vocab)


def prepare_mtp(mtp: MTPW, m: Model, *, draft_vocab: int | str | Path | Sequence[int] | None = "default") -> MTPK:
    """The drafter's weights regrouped for the kernels, with a draft head over ``draft_vocab`` cut from the model's."""

    c = m.cfg
    lw = mtp.layer
    if lw.linear or lw.attn is None:
        raise ValueError("the MTP head's layer must be an attention layer")
    a = lw.attn
    attn = AttnK(qmm.make_q4(*a.proj.triple()), a.q_scale.float().contiguous(), a.k_scale.float().contiguous(),
                 qmm.make_q4(*a.o.triple()))
    ex = qmm.make_experts(lw.moe.gate.triple(), lw.moe.up.triple(), lw.moe.down.triple())
    if ex.count != c.experts + 1 or lw.moe.router.shape != (c.experts + 1, c.hidden):
        raise ValueError(f"MTP layer: {ex.count} experts and router {tuple(lw.moe.router.shape)}, expected "
                         f"{c.experts} routed + the shared one")
    layer = LayerK(c.layers, False, m.n_attention, lw.input_scale.float().contiguous(),
                   lw.post_scale.float().contiguous(), None, attn, lw.moe.router.float().contiguous(), ex)
    fc = qmm.make_q4(*mtp.fc.triple())
    if (fc.n, fc.k) != (c.hidden, 2 * c.hidden):
        raise ValueError(f"MTP fc is {fc.n} x {fc.k}, expected {c.hidden} x {2 * c.hidden}")
    head, ids_dev, ids = draft_head(m, draft_vocab)
    return MTPK(mtp.norm_e.float().contiguous(), mtp.norm_h.float().contiguous(), fc, layer,
                mtp.norm.float().contiguous(), head, ids_dev, ids)


def draft_head(m: Model, draft_vocab: int | str | Path | Sequence[int] | None = "default"
               ) -> tuple[qmm.Q4, torch.Tensor | None, np.ndarray | None]:
    """(head, device ids, host ids): the model's head rows at the draft vocabulary, or the whole head and no ids."""

    ids = draft_token_ids(draft_vocab)
    if ids is None:
        return m.head, None, None
    ids = ids[(ids >= 0) & (ids < m.head.n)]
    if len(ids) == 0:
        raise ValueError("the draft vocabulary holds no id of this model's vocabulary")
    words, scales, biases = qmm.to_mlx(m.head)
    ids_dev = torch.from_numpy(ids).to(m.device)
    head = qmm.make_q4(words[ids_dev], scales[ids_dev], biases[ids_dev])
    return head, ids_dev, ids


# -- buffers -------------------------------------------------------------------------------------------------------
class MTPBuffers:
    """Static scratch for MTP steps of up to ``rows`` rows (only the last row runs past the cache write)."""

    def __init__(self, m: Model, k: MTPK, rows: int = 64, *, capacity: int) -> None:
        c = m.cfg
        dev = m.device
        bf, f32 = torch.bfloat16, torch.float32
        d = c.hidden
        self.rows = rows
        self.capacity = capacity
        pin = torch.cuda.is_available()
        self.ids = torch.zeros((rows,), dtype=torch.int32, device=dev)
        self.pos = torch.zeros((rows,), dtype=torch.int32, device=dev)
        self.ids_host = torch.zeros((rows,), dtype=torch.int32, pin_memory=pin)
        self.pos_host = torch.zeros((rows,), dtype=torch.int32, pin_memory=pin)
        self.staged = torch.cuda.Event() if pin else None
        self.hin = torch.zeros((rows, d), dtype=bf, device=dev)           # the rows' hidden states (staged)
        self.emb = torch.empty((rows, d), dtype=bf, device=dev)
        self.cat = torch.empty((rows, 2 * d), dtype=bf, device=dev)       # [normed embedding | normed hidden]
        self.cat_xs = torch.empty((rows, 2 * d // GS), dtype=f32, device=dev)
        self.h = torch.empty((rows, d), dtype=bf, device=dev)
        self.normed = torch.empty((rows, d), dtype=bf, device=dev)
        self.xs = torch.empty((rows, d // GS), dtype=f32, device=dev)
        self.pa = torch.empty((rows, sum(c.attn_rows)), dtype=bf, device=dev)
        self.q = torch.empty((rows, c.heads, c.head_dim), dtype=bf, device=dev)
        self.attn = attn_mod.AttnScratch(1, c.heads, c.head_dim, capacity, dev, sparse=False)
        self.gated = torch.empty((1, c.heads * c.head_dim), dtype=bf, device=dev)
        self.gated_xs = torch.empty((1, c.heads * c.head_dim // GS), dtype=f32, device=dev)
        self.branch = torch.empty((1, d), dtype=bf, device=dev)
        self.moe = moe_mod.buffers(1, dev, m.moe_cfg)
        self.out = torch.empty((1, d), dtype=bf, device=dev)              # the last row's output (the next hidden)
        self.out_xs = torch.empty((1, d // GS), dtype=f32, device=dev)
        a = k.layer.attn
        shapes = [(k.fc.n, k.fc.k), (a.proj.n, a.proj.k), (a.o.n, a.o.k), (k.head.n, k.head.k)]
        self.part = torch.empty((max(1, qmm.split_scratch(rows, shapes)),), dtype=f32, device=dev)
        self.logits = torch.empty((1, k.head.n), dtype=bf, device=dev)

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


class _Row:
    """One row of the scratch as the three tensors ``forward._norm`` reads."""

    __slots__ = ("h", "normed", "xs")

    def __init__(self, h: torch.Tensor, normed: torch.Tensor, xs: torch.Tensor) -> None:
        self.h, self.normed, self.xs = h, normed, xs


# -- kernels -------------------------------------------------------------------------------------------------------
@triton.jit
def _norm_into(X, W, Y, XS, eps, y_stride, xs_stride, D: tl.constexpr):
    """Program r: Y[r] = bf16(RMSNorm(x) w) (fp32 math) and its 64-group sums, into one half of the fc input."""

    r = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, D)
    x = tl.load(X + r * D + d).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
    y = (x * inv * tl.load(W + d)).to(tl.bfloat16)
    tl.store(Y + r * y_stride + d, y)
    yg = tl.reshape(y.to(tl.float32), (D // 64, 64))
    tl.store(XS + r * xs_stride + tl.arange(0, D // 64), tl.sum(yg, axis=1))


def _pre_fc(k: MTPK, b: MTPBuffers, n: int, eps: float) -> None:
    d = b.emb.shape[1]
    g = d // GS
    _norm_into[(n,)](b.emb, k.norm_e, b.cat, b.cat_xs, float(eps), b.cat.stride(0), b.cat_xs.stride(0), D=d,
                     num_warps=8)
    _norm_into[(n,)](b.hin, k.norm_h, b.cat[:, d:], b.cat_xs[:, g:], float(eps), b.cat.stride(0),
                     b.cat_xs.stride(0), D=d, num_warps=8)


# -- a step --------------------------------------------------------------------------------------------------------
def mtp_stage(b: MTPBuffers, st: State, tokens: Sequence[int], hidden: torch.Tensor, pos0: int) -> int:
    """Stage a step's next tokens, positions from ``pos0`` and hidden rows into the static buffers; returns rows."""

    n = len(tokens)
    if not 0 < n <= b.rows or hidden.shape[0] != n:
        raise ValueError(f"an MTP step of {n} tokens and {hidden.shape[0]} hidden rows (buffers hold {b.rows})")
    if st.mtp_kc.shape[0] == 0:
        raise ValueError("this state's pool has no draft-head cache (Pool(..., mtp_layers=1))")
    if pos0 < 0 or pos0 + n > min(st.capacity, b.capacity):
        raise ValueError(f"MTP positions {pos0}..{pos0 + n - 1} past the cache capacity")
    if b.staged is not None:
        b.staged.synchronize()
    b.ids_host[:n].numpy()[:] = np.asarray(tokens, dtype=np.int32)
    b.pos_host[:n].numpy()[:] = np.arange(pos0, pos0 + n, dtype=np.int32)
    b.ids[:n].copy_(b.ids_host[:n], non_blocking=True)
    b.pos[:n].copy_(b.pos_host[:n], non_blocking=True)
    if b.staged is not None:
        b.staged.record()
    if hidden.data_ptr() != b.hin.data_ptr():
        if hidden.untyped_storage().data_ptr() == b.hin.untyped_storage().data_ptr():
            hidden = hidden.clone()                   # other rows of the staging buffer itself: no overlapping copy
        b.hin[:n].copy_(hidden)
    return n


def mtp_compute(m: Model, k: MTPK, b: MTPBuffers, st: State, n: int, *, logits: bool = True,
                context: int | None = None) -> torch.Tensor | None:
    """A staged step's GPU work: every row's keys into the head's cache; with ``logits``, the last row's logits."""

    c = m.cfg
    a = k.layer.attn
    kc, vc = st.mtp_kc[0], st.mtp_vc[0]
    qmm.embed(b.ids[:n], *m.embed.triple(), out=b.emb[:n])
    _pre_fc(k, b, n, c.eps)
    qmm.matmul(b.cat[:n], k.fc, b.cat_xs[:n], out=b.h[:n], part=b.part)
    _norm(b, n, k.layer.input_scale, c.eps)
    qmm.matmul(b.normed[:n], a.proj, b.xs[:n], out=b.pa[:n], part=b.part)
    fn_glue.attn_prep(b.pa[:n], b.pos[:1], a.q_scale, a.k_scale, None, m.inv_freq, b.q[:n], kc, vc, None, None,
                      c.eps, q_heads=c.heads, kv_heads=c.kv_heads, head_dim=c.head_dim)
    if not logits:
        return None
    r = n - 1
    o = attn_mod.attention(b.q[r:r + 1], kc, vc, b.pos[r:r + 1], b.attn, 1, c.head_dim ** -0.5, context=context)
    fn_glue.attn_gate(o[:1], b.pa[r:r + 1], b.gated, b.gated_xs, q_heads=c.heads, head_dim=c.head_dim, group=GS)
    branch = qmm.matmul(b.gated, a.o, b.gated_xs, out=b.branch, part=b.part, reduce=False)
    row = _Row(b.h[r:r + 1], b.normed[r:r + 1], b.xs[r:r + 1])
    _norm(row, 1, k.layer.post_scale, c.eps, branch)
    sub = moe_mod.moe(row.normed, row.xs, k.layer.router, k.layer.experts, b.moe, top_k=c.top_k, experts=c.experts)
    moe_mod.combine_add_rmsnorm(sub.y, sub.wts, row.h, k.norm, c.eps, h_out=row.h, normed=b.out, xs=b.out_xs)
    return qmm.matmul(b.out, k.head, b.out_xs, out=b.logits, part=b.part)


@torch.no_grad()
def mtp_forward(m: Model, k: MTPK, b: MTPBuffers, st: State, tokens: Sequence[int], hidden: torch.Tensor,
                pos0: int, *, logits: bool = True) -> torch.Tensor | None:
    """An MTP step over ``tokens`` and ``hidden`` from ``pos0`` (see ``mtp_compute``); the caller keeps mtp_len."""

    n = mtp_stage(b, st, tokens, hidden, pos0)
    return mtp_compute(m, k, b, st, n, logits=logits)
