"""The 27B's prefill: bits never depend on chunking but differ from decode's, so only prompt-end states resume."""

from __future__ import annotations

from typing import Sequence

import torch
import triton
import triton.language as tl

from tensorfold.cuda import moe
from tensorfold.cuda.kernels import dense as dense_kernel
from tensorfold.cuda.kernels import gdn as deltanet
from tensorfold.cuda.kernels import qmm as shared
from tensorfold.cuda.kernels.prefill_attention import attention

from . import glue
from . import prefill_bf16, prefill_glue
from .decode import clone_state
from .forward import State
from .qmm_fast import matmul, matmul_partial, tile
from .weights import QLinear, Weights

CHUNK = 4096
TAP_LAYERS = (5, 19, 33, 47, 61)


def _mm(x, w: QLinear, f32: bool = False) -> torch.Tensor:
    """``x``: e4m3 inputs with group sums and row scales from ``prefill_glue``, or bf16 rows from ``prefill_bf16``."""

    if not isinstance(w, QLinear):
        return w.prefill(x)                               # an EXL3 pack's projection
    if isinstance(x, tuple):
        return shared.prefill_matmul8(x, tile(w), f32=f32)
    if w.layout == "dense":
        # a plain (unquantized) checkpoint's prompt rows: tensor-core GEMM, not the generic per-4-column
        # reference kernel affine.matmul falls back to (correct there, but far too slow for a 4,096-row chunk)
        return dense_kernel.prefill_matmul(x, w.weight, f32=f32)
    packed = tile(w)                                      # an affine format past the FP8 four-bit path
    return matmul_partial(x, packed) if f32 else matmul(x, packed)


def _row_mm(x, w: QLinear, tp: bool, f32: bool = False) -> torch.Tensor:
    if not tp:
        return _mm(x, w, f32=f32)
    from .distributed import gather_rank_partials

    return gather_rank_partials(_mm(x, w))                 # bf16 partials: half the bytes of fp32 over the link
                                                            # (unchanged by f32: TP halves link bytes, not precision)


def _precise_mm(x: torch.Tensor, w: QLinear) -> torch.Tensor:
    """Every projection on the precise scoring path (GDN in/out, attention q/k/v/o, MLP gate/up/down): a genuine
    fp32 row input against the checkpoint's bf16 weight, fp32 accumulate, a fixed block shape whatever the row
    count (``dense_kernel.precise_matmul`` — exact row/batch invariance, round 4). Single GPU only (the precise
    scoring path never runs tensor-parallel)."""

    return dense_kernel.precise_matmul(x, w.weight)


def is_dense_checkpoint(w: Weights) -> bool:
    """A plain (unquantized) bf16/fp16/fp32 checkpoint, read at its stored precision throughout — as opposed to an
    MLX affine, EXL3 or NVFP4 one (which keep their own tested bf16-rounding conventions unchanged here)."""

    return isinstance(w.head, QLinear) and w.head.layout == "dense"


@triton.jit
def _add_rmsnorm_f32_kernel(X, R, W, H, Y, eps, D: tl.constexpr, BLOCK: tl.constexpr, HAS_R: tl.constexpr,
                            Y_BF16: tl.constexpr):
    """One program a row: reads only that row (and ``W``), so a row's bits never depend on the row count — the
    same guarantee ``glue._add_rmsnorm`` gives its bf16 ``H``, kept here for an fp32 one."""

    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ok = offs < D
    x = tl.load(X + row * D + offs, mask=ok, other=0.0).to(tl.float32)
    if HAS_R:
        x = x + tl.load(R + row * D + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(H + row * D + offs, x, mask=ok)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
    y = x * inv * tl.load(W + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(Y + row * D + offs, y.to(tl.bfloat16) if Y_BF16 else y, mask=ok)


def _add_rmsnorm_f32(x: torch.Tensor, residual: torch.Tensor | None, weight: torch.Tensor, eps: float):
    """The residual stream kept in fp32 throughout, pushing a dense checkpoint's prefill toward the fp32
    reference instead of rounding the running sum at every layer: ``x`` (and the returned ``h``) fp32;
    ``residual`` (a layer's own output projection) fp32 or bf16, either way added as fp32. The normed value
    returned is fp32 too (round 4: every projection now takes an fp32 row input, weights as the checkpoint
    stores them — see ``precise_matmul``), not rounded to bf16 first. Row-invariant: a fixed-size Triton program
    a row, like every other kernel in this family."""

    rows, d = x.shape
    h = torch.empty_like(x)
    y = torch.empty((rows, d), dtype=torch.float32, device=x.device)
    _add_rmsnorm_f32_kernel[(rows,)](x, residual if residual is not None else x, weight, h, y, eps, D=d,
                                     BLOCK=triton.next_power_of_2(d), HAS_R=residual is not None, Y_BF16=False,
                                     num_warps=8)
    return h, y


@triton.jit
def _attn_prep_precise(QG, KV, QN, KN, POS, INV, QOUT, KOUT, eps,
                       H: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr,
                       MROPE: tl.constexpr, ROWS: tl.constexpr, HSEC: tl.constexpr, WSEC: tl.constexpr):
    """``glue._attn_prep`` (never modified — shared with decode), but the per-head RMSNorm stays fp32 into the
    rotation instead of rounding to bf16 mid-computation (``xn``/``xpn``); only the final rotated value, which
    the attention kernel takes as bf16 regardless of checkpoint precision, rounds down."""

    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, D)
    is_q = head < H
    if is_q:
        x = tl.load(QG + (row * H + head) * 2 * D + d).to(tl.float32)
        w = tl.load(QN + d).to(tl.float32)
    else:
        x = tl.load(KV + (row * HKV + head - H) * D + d).to(tl.float32)
        w = tl.load(KN + d).to(tl.float32)
    xn = x * (1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)) * w
    if MROPE:
        i = tl.where(d < HALF, d, tl.where(d < 2 * HALF, d - HALF, 0))
        axis = tl.where((i % 3 == 1) & (i < 3 * HSEC), 1, tl.where((i % 3 == 2) & (i < 3 * WSEC), 2, 0))
        pos = tl.load(POS + axis * ROWS + row).to(tl.float32)
    else:
        pos = tl.load(POS + row).to(tl.float32)
        i = tl.where(d < HALF, d, tl.where(d < 2 * HALF, d - HALF, 0))
    ang = pos * tl.load(INV + i)
    cos = tl.cos(ang)
    sin = tl.sin(ang)
    partner = tl.where(d < HALF, d + HALF, tl.where(d < 2 * HALF, d - HALF, d))
    if is_q:
        xp = tl.load(QG + (row * H + head) * 2 * D + partner).to(tl.float32)
    else:
        xp = tl.load(KV + (row * HKV + head - H) * D + partner).to(tl.float32)
    wp = tl.load((QN if is_q else KN) + partner).to(tl.float32)
    xpn = xp * (1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)) * wp
    rot = tl.where(d < HALF, xn * cos - xpn * sin, tl.where(d < 2 * HALF, xn * cos + xpn * sin, xn))
    out = rot.to(tl.bfloat16)
    if is_q:
        tl.store(QOUT + (row * H + head) * D + d, out)
    else:
        tl.store(KOUT + (row * HKV + head - H) * D + d, out)


def attn_prep_precise(qg: torch.Tensor, k: torch.Tensor, q_norm: torch.Tensor, k_norm: torch.Tensor,
                      pos: torch.Tensor, inv_freq: torch.Tensor, eps: float, *, heads: int, kv_heads: int,
                      head_dim: int, mrope_section: tuple[int, int, int] = (11, 11, 10)):
    """``glue.attn_prep``'s precise variant: same shapes and contract, no intermediate bf16 rounding."""

    W = qg.shape[0]
    multi = pos.ndim == 2
    if tuple(pos.shape) != ((3, W) if multi else (W,)):
        raise ValueError("attention positions must be one or three coordinates per row")
    if len(mrope_section) != 3 or any(not isinstance(v, int) or v < 0 for v in mrope_section):
        raise ValueError("mrope_section requires three nonnegative integers")
    pos = pos.contiguous()
    qo = torch.empty((W, heads, head_dim), dtype=torch.bfloat16, device=qg.device)
    ko = torch.empty((W, kv_heads, head_dim), dtype=torch.bfloat16, device=qg.device)
    _attn_prep_precise[(W, heads + kv_heads)](qg, k, q_norm, k_norm, pos, inv_freq, qo, ko, eps, H=heads,
                                              HKV=kv_heads, D=head_dim, HALF=inv_freq.numel(), MROPE=multi,
                                              ROWS=W if multi else 0, HSEC=mrope_section[1], WSEC=mrope_section[2],
                                              num_warps=2)
    return qo, ko


# The three activations between a layer's projections (glue.gated_norm/swiglu/gate_mul) all round their output to
# bf16 for the FP8-fast and ordinary bf16 paths' next matmul; the precise path's next matmul takes an fp32 row
# input (precise_matmul), so these keep the same value in fp32 instead — no group sums (XS): only the FP8 path
# ever reads them.


@triton.jit
def _gated_norm_precise(Yr, Z, W, OUT, eps, VH: tl.constexpr, DV: tl.constexpr):
    row = tl.program_id(0)
    h = tl.program_id(1)
    offs = (row * VH + h) * DV + tl.arange(0, DV)
    y = tl.load(Yr + offs).to(tl.float32)
    z = tl.load(Z + offs).to(tl.float32)
    w = tl.load(W + tl.arange(0, DV)).to(tl.float32)
    yn = y * (1.0 / tl.sqrt(tl.sum(y * y, axis=0) / DV + eps)) * w
    tl.store(OUT + offs, z * tl.sigmoid(z) * yn)


def gated_norm_precise(y: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    W, vh, dv = y.shape
    out = torch.empty((W, vh * dv), dtype=torch.float32, device=y.device)
    _gated_norm_precise[(W, vh)](y, z, w, out, eps, VH=vh, DV=dv, num_warps=1)
    return out


@triton.jit
def _swiglu_precise(GATE, UP, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    ok = offs < N
    g = tl.load(GATE + row * N + offs, mask=ok, other=0.0).to(tl.float32)
    u = tl.load(UP + row * N + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(OUT + row * N + offs, g * tl.sigmoid(g) * u, mask=ok)


def swiglu_precise(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    W, n = gate.shape
    out = torch.empty((W, n), dtype=torch.float32, device=gate.device)
    block = 1024
    _swiglu_precise[(W, triton.cdiv(n, block))](gate, up, out, N=n, BLOCK=block, num_warps=4)
    return out


@triton.jit
def _gate_mul_precise(O, QG, OUT, H: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    o = tl.load(O + (row * H + h) * D + d).to(tl.float32)
    g = tl.load(QG + (row * H + h) * 2 * D + D + d).to(tl.float32)
    tl.store(OUT + (row * H + h) * D + d, o * tl.sigmoid(g))


def gate_mul_precise(o: torch.Tensor, qg: torch.Tensor, *, heads: int, head_dim: int) -> torch.Tensor:
    W = o.shape[0]
    out = torch.empty((W, heads * head_dim), dtype=torch.float32, device=o.device)
    _gate_mul_precise[(W, heads)](o, qg, out, H=heads, D=head_dim, num_warps=1)
    return out


def _norm_f32(x: torch.Tensor, residual: torch.Tensor | None, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Like ``_add_rmsnorm_f32``, but for the final norm: returns the fp32 normed value itself (never rounded to
    bf16), for an fp32 lm_head matmul instead of a bf16-rounded one. Same row-invariance."""

    rows, d = x.shape
    h = torch.empty_like(x)
    y = torch.empty((rows, d), dtype=torch.float32, device=x.device)
    _add_rmsnorm_f32_kernel[(rows,)](x, residual if residual is not None else x, weight, h, y, eps, D=d,
                                     BLOCK=triton.next_power_of_2(d), HAS_R=residual is not None, Y_BF16=False,
                                     num_warps=8)
    return y


def _grow(st: State, i: int, need: int) -> tuple[torch.Tensor, torch.Tensor]:
    kbuf, vbuf = st.kv[i]
    if kbuf.shape[0] < need:
        cap = max(need, 2 * kbuf.shape[0], 1024)
        if st.limit:
            cap = max(need, min(cap, st.limit))
        grown_k, grown_v = kbuf.new_empty((cap, *kbuf.shape[1:])), vbuf.new_empty((cap, *vbuf.shape[1:]))
        grown_k[:st.pos] = kbuf[:st.pos]
        grown_v[:st.pos] = vbuf[:st.pos]
        st.kv[i] = (grown_k, grown_v)
    return st.kv[i]


@torch.no_grad()
def prefill_chunk(w: Weights, tokens: torch.Tensor, st: State, *, tp: bool = False, capture_taps: bool = False,
                  last: bool = True, every: bool = False, cut: int = 0, vision=None, precise: bool = False):
    """Commit ``tokens`` at [st.pos, st.pos + W) into ``st`` without writing through its entries (``every``: all rows' final normed states; ``cut``: also the state after the first ``cut`` rows, the GDN chains run as two launches with one launch's bits). ``precise``: a dense checkpoint's residual stream stays fp32 throughout instead of rounding to bf16 every layer (the readout scoring contract; ordinary generation leaves this off, unchanged)."""

    c = w.config
    pg = prefill_glue if w.fast_prefill else prefill_bf16         # FP8 inputs only where every projection is 4-bit g64
    dense = precise and is_dense_checkpoint(w)
    add_norm = _add_rmsnorm_f32 if dense else pg.add_rmsnorm
    mm = _precise_mm if dense else _mm                 # every projection's row input, fp32 on the precise path
    out_mm = (lambda x, w_lin: _precise_mm(x, w_lin)) if dense else (lambda x, w_lin: _row_mm(x, w_lin, tp))
    attn_prep = attn_prep_precise if dense else glue.attn_prep
    W = int(tokens.shape[0])
    if not 0 <= cut < W:
        raise ValueError(f"cut {cut} is not inside a chunk of {W} rows")
    p0 = st.pos
    keep = c.conv_kernel - 1
    dev = tokens.device
    pos = (torch.arange(p0, p0 + W, device=dev, dtype=torch.int32) if vision is None
           else vision.positions[:, p0:p0 + W].contiguous())
    windows = (torch.arange(W, device=dev, dtype=torch.int32)[:, None]
               + torch.arange(keep + 1, device=dev, dtype=torch.int32)[None, :])
    x = glue.embedding(tokens.to(torch.int32), w.embed)
    if dense:
        x = x.float()
    if vision is not None:
        from tensorfold.vision.qwen_cuda import replace_rows

        x = replace_rows(x, vision, p0, p0 + W)
    pending: torch.Tensor | None = None
    taps: list[torch.Tensor] = []
    part = clone_state(st) if cut else None
    if part is not None:
        part.kv = []            # the chunk's final buffers, set below: these would outlive a grow that replaces them
    for i, layer in enumerate(w.layers):
        x, h = add_norm(x, pending, layer.input_norm, c.eps)
        if layer.linear:
            gdn = layer.gdn
            qkv = mm(h, gdn.qkv)
            if gdn.zba is not None:
                zba = mm(h, gdn.zba)
                vd = c.v_heads * c.dv
                z = zba[:, :vd].contiguous().reshape(W, c.v_heads, c.dv)
                b = zba[:, vd:vd + c.v_heads].contiguous()
                a = zba[:, vd + c.v_heads:].contiguous()
            else:
                z = mm(h, gdn.z).reshape(W, c.v_heads, c.dv)
                b = mm(h, gdn.b)
                a = mm(h, gdn.a)
            # the conv history: fp32 on the precise path (a chunk boundary must not round it — a one-shot call's
            # own causal window never rounds these same rows, so rounding only at a chunk edge would make the
            # precise path chunk-count dependent, exactly the ~1e-4 gap a first attempt at this measured);
            # bf16 (State's own allocation) otherwise, unchanged.
            conv_prev = st.conv[i].float() if dense else st.conv[i]
            q, k, v, g, beta = glue.gdn_pre(qkv, conv_prev, gdn.conv, windows, a, b, gdn.A_log, gdn.dt_bias,
                                            kh=c.k_heads, vh=c.v_heads, dk=c.dk)
            final = torch.empty_like(st.rec[i])
            if part is None:
                yr = deltanet.chain(q, k, v, g, beta, st.rec[i], final)
            else:
                part.rec[i] = torch.empty_like(st.rec[i])
                yr = torch.cat([deltanet.chain(q[:cut], k[:cut], v[:cut], g[:cut], beta[:cut], st.rec[i], part.rec[i]),
                                deltanet.chain(q[cut:], k[cut:], v[cut:], g[cut:], beta[cut:], part.rec[i], final)])
                part.conv[i] = torch.cat([conv_prev, qkv[max(0, cut - keep):cut]])[-keep:].contiguous()
            gated = gated_norm_precise(yr, z, gdn.norm, c.eps) if dense else pg.gated_norm(yr, z, gdn.norm, c.eps)
            r = out_mm(gated, gdn.out)
            st.conv[i] = torch.cat([conv_prev, qkv[-keep:]])[-keep:].contiguous()
            st.rec[i] = final
        else:
            attn = layer.attn
            qg = mm(h, attn.q)
            if attn.kv is not None:
                kv = mm(h, attn.kv)
                kd = c.kv_heads * c.head_dim
                key = kv[:, :kd].contiguous()
                value = kv[:, kd:].contiguous().reshape(W, c.kv_heads, c.head_dim)
            else:
                key = mm(h, attn.k)
                value = mm(h, attn.v).reshape(W, c.kv_heads, c.head_dim)
            q, key = attn_prep(qg, key, attn.q_norm, attn.k_norm, pos, w.inv_freq, c.eps, heads=c.heads,
                               kv_heads=c.kv_heads, head_dim=c.head_dim, mrope_section=c.mrope_section)
            kbuf, vbuf = _grow(st, i, p0 + W)
            kbuf[p0:p0 + W] = key.view(W, c.kv_heads, c.head_dim)
            vbuf[p0:p0 + W] = value
            out = attention(q.view(W, c.heads, c.head_dim), kbuf, vbuf, p0, scale=c.head_dim ** -0.5)
            gated = (gate_mul_precise(out, qg, heads=c.heads, head_dim=c.head_dim) if dense
                    else pg.gate_mul(out, qg, heads=c.heads, head_dim=c.head_dim))
            r = out_mm(gated, attn.o)
        if layer.moe is not None:                          # routed experts read bf16 rows (their prefill form)
            x, h, _ = glue.add_rmsnorm(x, r, layer.post_norm, c.eps)
            pending = moe.run(h, layer.moe, prefill=True)
        else:
            x, h = add_norm(x, r, layer.post_norm, c.eps)
            act = (swiglu_precise(mm(h, layer.gate), mm(h, layer.up)) if dense
                  else pg.swiglu(_mm(h, layer.gate), _mm(h, layer.up)))
            pending = out_mm(act, layer.down)
        if capture_taps and i in TAP_LAYERS:
            taps.append((x.float() + pending.float()).to(torch.bfloat16))
    st.pos = p0 + W
    normed = None
    final_norm = _norm_f32 if dense else (lambda *a: glue.add_rmsnorm(*a)[1])
    if every:
        normed = final_norm(x, pending, w.norm, c.eps)
    elif last:
        normed = final_norm(x[-1:].contiguous(), pending[-1:].contiguous(), w.norm, c.eps)
    taps_out = torch.cat(taps, dim=-1) if capture_taps else None
    if part is None:
        return normed, taps_out
    part.pos, part.kv = p0 + cut, st.kv.copy()      # the chunk's buffers: their rows below part.pos stay as committed
    return normed, taps_out, part


def _multi_conv_windows(lengths: Sequence[int], keep: int, device) -> torch.Tensor:
    """``windows`` for several fresh (cold) texts packed back to back: each text's own sliding causal window over
    its own zero-initialized conv state, then its own rows, never another text's.

    Generalizes the single-stream ``torch.arange(W)[:, None] + torch.arange(keep + 1)[None, :]`` (``base`` 0) the
    same way ``forward.multi_tree_forward`` generalizes it for tree windows: a text-relative row ``r``'s window
    entry ``j`` is ``raw = r - keep + j``; ``raw < 0`` reads that text's own (zeroed) conv-state block (index
    ``raw + keep``, resolved per row by ``stream_ids`` in ``glue.gdn_pre``'s kernel), ``raw >= 0`` reads this
    text's own qkv row at ``raw`` (global index ``raw + keep + base``).
    """

    parts = []
    base = 0
    for length in lengths:
        r = torch.arange(length, device=device, dtype=torch.int64)[:, None]
        j = torch.arange(keep + 1, device=device, dtype=torch.int64)[None, :]
        raw = r - keep + j
        parts.append(torch.where(raw < 0, raw + keep, raw + keep + base).to(torch.int32))
        base += length
    return torch.cat(parts, dim=0)


@torch.no_grad()
def multi_prefill_logits(w: Weights, texts: Sequence[Sequence[int]]) -> torch.Tensor:
    """Several independent cold prefills in one forward: token-id lists packed back to back, the row-invariant
    projections (and MLP) shared across all of them in one bigger matmul a layer, each text kept to its own GDN
    chain and its own causal attention window (``glue.gdn_pre``'s existing ``stream_ids`` mode and
    ``attention_texts``, both already used this way for other batched shapes).

    The GDN chain itself still runs once a text (no shared kernel batches independent chunked-delta-rule states
    yet): this batches the projections and MLP, which dominate a dense 2,560-wide checkpoint's prefill cost, not
    the whole forward.

    Returns each text's last-position full-vocab logits, ``(len(texts), vocab)`` fp32. No prefix reuse, no KV
    growth: built for the readout scoring contract, where every request is a complete fresh prompt and
    ``max_tokens`` is 1.
    """

    from tensorfold.cuda.kernels.prefill_attention import attention_texts, text_blocks

    c = w.config
    if w.fast_prefill:
        raise ValueError("multi_prefill_logits takes the bf16 prompt-glue path only (fast_prefill checkpoints "
                         "already have FP8-glued, single-stream, prefill kernels: batch those the usual way)")
    lengths = [len(t) for t in texts]
    if not texts or min(lengths) < 1:
        raise ValueError("multi_prefill_logits takes one or more nonempty texts")
    dev = w.norm.device
    keep = c.conv_kernel - 1
    n, total = len(texts), sum(lengths)
    ids = torch.tensor([int(tok) for text in texts for tok in text], dtype=torch.int32, device=dev)
    pos = torch.cat([torch.arange(length, dtype=torch.int32, device=dev) for length in lengths])
    stream_ids = torch.cat([torch.full((length,), t, dtype=torch.int32, device=dev)
                            for t, length in enumerate(lengths)])
    windows = _multi_conv_windows(lengths, keep, dev)
    blocks = text_blocks(lengths, dev)
    last = torch.tensor(lengths, dtype=torch.int64, device=dev).cumsum(0) - 1
    x = glue.embedding(ids, w.embed).float()   # the residual stream stays fp32 throughout (the scoring contract
                                                # only, per _add_rmsnorm_f32's own docstring: this function is
                                                # never used for ordinary generation)
    pending: torch.Tensor | None = None
    for layer in w.layers:
        x, h = _add_rmsnorm_f32(x, pending, layer.input_norm, c.eps)
        if layer.linear:
            gdn = layer.gdn
            qkv = _precise_mm(h, gdn.qkv)
            if gdn.zba is not None:
                zba = _precise_mm(h, gdn.zba)
                vd = c.v_heads * c.dv
                z = zba[:, :vd].contiguous().reshape(total, c.v_heads, c.dv)
                b = zba[:, vd:vd + c.v_heads].contiguous()
                a = zba[:, vd + c.v_heads:].contiguous()
            else:
                z = _precise_mm(h, gdn.z).reshape(total, c.v_heads, c.dv)
                b = _precise_mm(h, gdn.b)
                a = _precise_mm(h, gdn.a)
            conv_state = torch.zeros((n * keep, qkv.shape[1]), dtype=torch.bfloat16, device=dev)
            q, k, v, g, beta = glue.gdn_pre(qkv, conv_state, gdn.conv, windows, a, b, gdn.A_log, gdn.dt_bias,
                                            kh=c.k_heads, vh=c.v_heads, dk=c.dk, stream_ids=stream_ids, nkeep=keep)
            zero_state = torch.zeros((c.v_heads, c.dv, c.dk), dtype=torch.float32, device=dev)
            yr_parts, offset = [], 0
            for length in lengths:                              # the chunked delta rule itself: one chain a text
                sl = slice(offset, offset + length)
                yr_parts.append(deltanet.chain(q[sl], k[sl], v[sl], g[sl], beta[sl], zero_state,
                                               torch.empty_like(zero_state)))
                offset += length
            gated = gated_norm_precise(torch.cat(yr_parts), z, gdn.norm, c.eps)
            r = _precise_mm(gated, gdn.out)
        else:
            attn = layer.attn
            qg = _precise_mm(h, attn.q)
            if attn.kv is not None:
                kv = _precise_mm(h, attn.kv)
                kd = c.kv_heads * c.head_dim
                key = kv[:, :kd].contiguous()
                value = kv[:, kd:].contiguous().reshape(total, c.kv_heads, c.head_dim)
            else:
                key = _precise_mm(h, attn.k)
                value = _precise_mm(h, attn.v).reshape(total, c.kv_heads, c.head_dim)
            q, key = attn_prep_precise(qg, key, attn.q_norm, attn.k_norm, pos, w.inv_freq, c.eps, heads=c.heads,
                                       kv_heads=c.kv_heads, head_dim=c.head_dim, mrope_section=c.mrope_section)
            # attention_texts (like attn_prep_precise's own output) takes bf16 q/k/v regardless of checkpoint
            # precision; value comes straight from the precise projection (fp32) with no kv-cache assignment to
            # narrow it the way the single-stream path's kbuf/vbuf does, so round it explicitly here.
            out = attention_texts(q.contiguous(), key.contiguous(), value.to(torch.bfloat16).contiguous(), blocks,
                                  scale=c.head_dim ** -0.5)
            gated = gate_mul_precise(out, qg, heads=c.heads, head_dim=c.head_dim)
            r = _precise_mm(gated, attn.o)
        x, h = _add_rmsnorm_f32(x, r, layer.post_norm, c.eps)
        act = swiglu_precise(_precise_mm(h, layer.gate), _precise_mm(h, layer.up))
        pending = _precise_mm(act, layer.down)
    x_last = x.index_select(0, last).contiguous()
    pending_last = pending.index_select(0, last).contiguous()
    normed = _norm_f32(x_last, pending_last, w.norm, c.eps)
    return head_logits(w, normed)


def chunks(start: int, end: int, size: int = CHUNK) -> list[tuple[int, int]]:
    """Even chunks of at most ``size`` rows (a short one costs a whole weight pass); any bounds give the same bits."""

    n = -(-(end - start) // size)
    return [(start + (end - start) * j // n, start + (end - start) * (j + 1) // n) for j in range(n)] if n else []


@torch.no_grad()
def head_logits(w: Weights, normed: torch.Tensor) -> torch.Tensor:
    """The vocab-wide logits from the final normed hidden state, fp32.

    ``normed`` fp32 (``precise=True`` on a dense checkpoint, via ``_norm_f32``): a plain fp32 matmul against the
    head weight (bf16 as the checkpoint stores it), instead of rounding the more precise ``normed`` down to bf16
    for the tensor-core kernel on top of its own fp32 accumulation. ``normed`` bf16 (everything else, unchanged):
    the existing dense tensor-core kernel or the generic row-invariant one, whichever this checkpoint's head takes.
    """

    if normed.dtype == torch.float32:
        return torch.nn.functional.linear(normed, w.head.weight.float())
    if isinstance(w.head, QLinear) and w.head.layout == "dense":
        return dense_kernel.prefill_matmul(normed, w.head.weight, f32=True)
    return _mm(normed, w.head, f32=True)


def prefill_state(w: Weights, prompt: Sequence[int], st: State, *, tp: bool = False, draft=None,
                  size: int = CHUNK, keep_at: int | None = None, vision=None, precise: bool = False):
    """Commit prompt[st.pos:] into ``st``, tapping the drafter's window; ``keep_at``: ``(normed, (state, snapshot))``, the state after prompt[:keep_at] from a cut chunk."""

    dev = w.norm.device
    base, n = st.pos, len(prompt)
    if keep_at is not None and not base <= keep_at <= n:
        raise ValueError(f"keep_at {keep_at} is outside the prefilled range [{base}, {n}]")
    if vision is not None:                         # a later prefill step goes on with the rope its first step set
        if base and getattr(st, "rope_delta", None) is not vision.rope_delta:
            raise ValueError("image prompts require a fresh prefill state")
        st.rope_delta = vision.rope_delta
    ids = torch.tensor(list(prompt[base:]), dtype=torch.int32, device=dev)
    normed, kept = None, None
    end = n if keep_at is None else keep_at        # the drafter's window then also covers the kept point
    tap_from = base
    if draft is not None and end - draft.window > base:
        tap_from = end - draft.window
        draft.skip(tap_from - base)
    spans = chunks(base, n, size)
    for j, (a, b) in enumerate(spans):
        if keep_at == a:
            kept = (clone_state(st), draft.snapshot() if draft is not None else None)
            kept[0].kv = []                        # the final buffers, set below, as for a cut
        cut = keep_at - a if keep_at is not None and a < keep_at < b else 0
        want = draft is not None and b > tap_from
        normed, taps, *part = prefill_chunk(w, ids[a - base:b - base], st, tp=tp, capture_taps=want,
                                            last=j == len(spans) - 1, cut=cut, vision=vision, precise=precise)
        snap = None
        if want:
            rows = taps[max(0, tap_from - a):]
            if cut:                                # the drafter at the point, then the rest of the chunk
                split = keep_at - max(a, tap_from)
                if split:
                    draft.add_taps(rows[:split])
                snap, rows = draft.snapshot(), rows[split:]
            draft.add_taps(rows)
        if part:
            kept = (part[0], snap)
    if keep_at is None:
        return normed
    if keep_at == n:
        kept = (clone_state(st), draft.snapshot() if draft is not None else None)
    kept[0].kv = st.kv.copy()                      # grown buffers copy the committed rows: never hold the old ones
    return normed, kept
