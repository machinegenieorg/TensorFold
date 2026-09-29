"""The 27B's prefill: bits never depend on chunking but differ from decode's, so only prompt-end states resume."""

from __future__ import annotations

from typing import Sequence

import torch

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


def _row_mm(x, w: QLinear, tp: bool) -> torch.Tensor:
    if not tp:
        return _mm(x, w)
    from .distributed import gather_rank_partials

    return gather_rank_partials(_mm(x, w))                 # bf16 partials: half the bytes of fp32 over the link


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
                  last: bool = True, every: bool = False, cut: int = 0, vision=None):
    """Commit ``tokens`` at [st.pos, st.pos + W) into ``st`` without writing through its entries (``every``: all rows' final normed states; ``cut``: also the state after the first ``cut`` rows, the GDN chains run as two launches with one launch's bits)."""

    c = w.config
    pg = prefill_glue if w.fast_prefill else prefill_bf16         # FP8 inputs only where every projection is 4-bit g64
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
    if vision is not None:
        from tensorfold.vision.qwen_cuda import replace_rows

        x = replace_rows(x, vision, p0, p0 + W)
    pending: torch.Tensor | None = None
    taps: list[torch.Tensor] = []
    part = clone_state(st) if cut else None
    if part is not None:
        part.kv = []            # the chunk's final buffers, set below: these would outlive a grow that replaces them
    for i, layer in enumerate(w.layers):
        x, h = pg.add_rmsnorm(x, pending, layer.input_norm, c.eps)
        if layer.linear:
            gdn = layer.gdn
            qkv = _mm(h, gdn.qkv)
            if gdn.zba is not None:
                zba = _mm(h, gdn.zba)
                vd = c.v_heads * c.dv
                z = zba[:, :vd].contiguous().reshape(W, c.v_heads, c.dv)
                b = zba[:, vd:vd + c.v_heads].contiguous()
                a = zba[:, vd + c.v_heads:].contiguous()
            else:
                z = _mm(h, gdn.z).reshape(W, c.v_heads, c.dv)
                b = _mm(h, gdn.b)
                a = _mm(h, gdn.a)
            q, k, v, g, beta = glue.gdn_pre(qkv, st.conv[i], gdn.conv, windows, a, b, gdn.A_log, gdn.dt_bias,
                                            kh=c.k_heads, vh=c.v_heads, dk=c.dk)
            final = torch.empty_like(st.rec[i])
            if part is None:
                yr = deltanet.chain(q, k, v, g, beta, st.rec[i], final)
            else:
                part.rec[i] = torch.empty_like(st.rec[i])
                yr = torch.cat([deltanet.chain(q[:cut], k[:cut], v[:cut], g[:cut], beta[:cut], st.rec[i], part.rec[i]),
                                deltanet.chain(q[cut:], k[cut:], v[cut:], g[cut:], beta[cut:], part.rec[i], final)])
                part.conv[i] = torch.cat([st.conv[i], qkv[max(0, cut - keep):cut]])[-keep:].contiguous()
            r = _row_mm(pg.gated_norm(yr, z, gdn.norm, c.eps), gdn.out, tp)
            st.conv[i] = torch.cat([st.conv[i], qkv[-keep:]])[-keep:].contiguous()
            st.rec[i] = final
        else:
            attn = layer.attn
            qg = _mm(h, attn.q)
            if attn.kv is not None:
                kv = _mm(h, attn.kv)
                kd = c.kv_heads * c.head_dim
                key = kv[:, :kd].contiguous()
                value = kv[:, kd:].contiguous().reshape(W, c.kv_heads, c.head_dim)
            else:
                key = _mm(h, attn.k)
                value = _mm(h, attn.v).reshape(W, c.kv_heads, c.head_dim)
            q, key = glue.attn_prep(qg, key, attn.q_norm, attn.k_norm, pos, w.inv_freq, c.eps, heads=c.heads,
                                    kv_heads=c.kv_heads, head_dim=c.head_dim, mrope_section=c.mrope_section)
            kbuf, vbuf = _grow(st, i, p0 + W)
            kbuf[p0:p0 + W] = key.view(W, c.kv_heads, c.head_dim)
            vbuf[p0:p0 + W] = value
            out = attention(q.view(W, c.heads, c.head_dim), kbuf, vbuf, p0, scale=c.head_dim ** -0.5)
            r = _row_mm(pg.gate_mul(out, qg, heads=c.heads, head_dim=c.head_dim), attn.o, tp)
        if layer.moe is not None:                          # routed experts read bf16 rows (their prefill form)
            x, h, _ = glue.add_rmsnorm(x, r, layer.post_norm, c.eps)
            pending = moe.run(h, layer.moe, prefill=True)
        else:
            x, h = pg.add_rmsnorm(x, r, layer.post_norm, c.eps)
            pending = _row_mm(pg.swiglu(_mm(h, layer.gate), _mm(h, layer.up)), layer.down, tp)
        if capture_taps and i in TAP_LAYERS:
            taps.append((x.float() + pending.float()).to(torch.bfloat16))
    st.pos = p0 + W
    normed = None
    if every:
        _, normed, _ = glue.add_rmsnorm(x, pending, w.norm, c.eps)
    elif last:
        _, normed, _ = glue.add_rmsnorm(x[-1:].contiguous(), pending[-1:].contiguous(), w.norm, c.eps)
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
    x = glue.embedding(ids, w.embed)
    pending: torch.Tensor | None = None
    for layer in w.layers:
        x, h = prefill_bf16.add_rmsnorm(x, pending, layer.input_norm, c.eps)
        if layer.linear:
            gdn = layer.gdn
            qkv = _mm(h, gdn.qkv)
            if gdn.zba is not None:
                zba = _mm(h, gdn.zba)
                vd = c.v_heads * c.dv
                z = zba[:, :vd].contiguous().reshape(total, c.v_heads, c.dv)
                b = zba[:, vd:vd + c.v_heads].contiguous()
                a = zba[:, vd + c.v_heads:].contiguous()
            else:
                z = _mm(h, gdn.z).reshape(total, c.v_heads, c.dv)
                b = _mm(h, gdn.b)
                a = _mm(h, gdn.a)
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
            r = _mm(prefill_bf16.gated_norm(torch.cat(yr_parts), z, gdn.norm, c.eps), gdn.out)
        else:
            attn = layer.attn
            qg = _mm(h, attn.q)
            if attn.kv is not None:
                kv = _mm(h, attn.kv)
                kd = c.kv_heads * c.head_dim
                key = kv[:, :kd].contiguous()
                value = kv[:, kd:].contiguous().reshape(total, c.kv_heads, c.head_dim)
            else:
                key = _mm(h, attn.k)
                value = _mm(h, attn.v).reshape(total, c.kv_heads, c.head_dim)
            q, key = glue.attn_prep(qg, key, attn.q_norm, attn.k_norm, pos, w.inv_freq, c.eps, heads=c.heads,
                                    kv_heads=c.kv_heads, head_dim=c.head_dim, mrope_section=c.mrope_section)
            out = attention_texts(q.contiguous(), key.contiguous(), value.contiguous(), blocks,
                                  scale=c.head_dim ** -0.5)
            r = _mm(prefill_bf16.gate_mul(out, qg, heads=c.heads, head_dim=c.head_dim), attn.o)
        x, h = prefill_bf16.add_rmsnorm(x, r, layer.post_norm, c.eps)
        pending = _mm(prefill_bf16.swiglu(_mm(h, layer.gate), _mm(h, layer.up)), layer.down)
    x_last = x.index_select(0, last).contiguous()
    pending_last = pending.index_select(0, last).contiguous()
    _, normed, _ = glue.add_rmsnorm(x_last, pending_last, w.norm, c.eps)
    return dense_kernel.prefill_matmul(normed, w.head.weight, f32=True)


def chunks(start: int, end: int, size: int = CHUNK) -> list[tuple[int, int]]:
    """Even chunks of at most ``size`` rows (a short one costs a whole weight pass); any bounds give the same bits."""

    n = -(-(end - start) // size)
    return [(start + (end - start) * j // n, start + (end - start) * (j + 1) // n) for j in range(n)] if n else []


@torch.no_grad()
def prefill_state(w: Weights, prompt: Sequence[int], st: State, *, tp: bool = False, draft=None,
                  size: int = CHUNK, keep_at: int | None = None, vision=None):
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
                                            last=j == len(spans) - 1, cut=cut, vision=vision)
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
