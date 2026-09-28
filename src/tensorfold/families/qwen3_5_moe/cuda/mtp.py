"""Qwen3.6-35B-A3B's MTP head on CUDA (the drafter repository's weights), through the family's kernels.

Row t of the head reads the model's hidden state at position t (after the final norm, ``Buffers.hidden``) and the
token at position t + 1, and scores the token at position t + 2:

    x = fc([RMSNorm(embed(t_{t+1})) (1 + w_e) | RMSNorm(h_t) (1 + w_h)])      one 2048 x 4096 matmul, bf16 out
    x = x + attention(RMSNorm(x) (1 + w_in))    one gated attention layer over the head's own cache, the model's RoPE
    x = x + MoE(RMSNorm(x) (1 + w_post))        256 routed experts (top 8) and the shared expert
    h = RMSNorm(x) (1 + w_norm);  logits = head(h)

Row t sits at position t of the head's cache (``State.mtp_kc``), which holds the rows the head has absorbed
(``State.mtp_len``). A chained draft feeds the head's own output ``h`` back as the next row's hidden state; its cache
entries sit past ``mtp_len`` and the next absorb overwrites them.

Only the last row of a step needs its output: earlier rows only leave their keys and values in the cache, so a step
runs the attention, MoE and head on its last row alone (and a prefill absorb, which needs no output, stops after the
cache write). The head proposes and never decides a token, so none of this needs row invariance; every kernel is
still deterministic (the same inputs give the same bits), so drafts, and therefore speeds, repeat.

The draft head scores a subset of the vocabulary (``draft_vocab.txt``, the rows of the model's head at those ids): a
full head is 248,320 x 2,048 4-bit weights, about 290 MB a draft, eight times the rest of the MTP head. A token outside
the subset can never be a draft, which costs speed, never correctness. The list holds 76,882 ids: every id below
65,536 (the tokenizer's earliest merges), the 33 added tokens (``<|im_end|>``, ``<think>``, ...), and every id seen at
least 10 times in CPython 3.14's standard library, this repository, and the Python sources and documentation of the
packages in a local Python install (23,237 files, 67 million tokens):

    python tools/draft_vocab.py tokenizer.json out.txt --keep-below 65536 --min-count 10 --size 76856 PATTERNS...

plus ids 248044-248076. On held-out public text it holds 98.2% of the tokens of English prose (the reference test's
passages) and 99.8% of JSON; a draft reads 31% of the full head. (The tokenizer's merges are Qwen3.8's.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from ...qwen4_exp.cuda import attention as attn_mod
from ...qwen4_exp.cuda import glue as fn_glue
from . import moe as moe_mod
from . import qmm
from .forward import GS, AttnK, LayerK, Model, State, _norm
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
    """The token ids the draft head scores, sorted and unique: "default" (``draft_vocab.txt`` beside this module), a
    file of ids, an int N (the ids below N), a sequence of ids, or None (the whole vocabulary)."""

    if draft_vocab is None or (isinstance(draft_vocab, str) and not draft_vocab):
        return None
    if isinstance(draft_vocab, (int, np.integer)):
        return np.arange(int(draft_vocab), dtype=np.int64)
    if isinstance(draft_vocab, (str, Path)):
        source = DRAFT_VOCAB if draft_vocab == "default" else Path(draft_vocab)
        return np.unique(np.loadtxt(source, dtype=np.int64).reshape(-1))
    return np.unique(np.asarray(list(draft_vocab), dtype=np.int64))


def prepare_mtp(mtp: MTPW, m: Model, *, draft_vocab: int | str | Path | Sequence[int] | None = "default") -> MTPK:
    """The drafter's weights (``weights.load_mtp``) regrouped for the kernels, with a draft head over
    ``draft_vocab`` (see ``draft_token_ids``) cut from the model's head."""

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
    """(head, ids on the device, ids on the host): the model's head rows at the draft vocabulary's ids, or the whole
    head (and None, None) without one. ``MTPK``'s last three fields."""

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
    """Program r: Y[r] = bf16(x * rsqrt(mean(x^2) + eps) * w) (fp32 math) and its 64-group sums, into strided rows
    (one half of the fc input)."""

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
    """Host work before a step: the next tokens and positions pos0, pos0 + 1, ... into the static device buffers
    (pinned, async), and the rows' hidden states ``hidden`` [n, hidden] bf16 into ``b.hin``."""

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
    """The GPU work of a staged step (capturable): every row's keys and values into the head's cache; with
    ``logits``, the last row's output (``b.out``) and its draft-head logits [1, head rows] (a view of ``b.logits``).
    ``context``: a bound on the last row's keys (a captured graph's bucket), as in ``forward.compute``."""

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
    """Rows at positions pos0, pos0 + 1, ... reading ``hidden`` [n, hidden] and the next ``tokens``: see
    ``mtp_compute``. The caller keeps ``st.mtp_len``."""

    n = mtp_stage(b, st, tokens, hidden, pos0)
    return mtp_compute(m, k, b, st, n, logits=logits)
