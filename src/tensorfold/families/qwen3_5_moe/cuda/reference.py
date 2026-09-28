"""A plain fp32 PyTorch forward of Qwen3.6-35B-A3B and its MTP head, to check the CUDA forward against."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .checkpoint import Config
from .weights import MTPW, QW, AttnW, GDNW, LayerW, MoEW, Weights, dequantize

BF, F32 = torch.bfloat16, torch.float32
_DEQ_ROWS = 16384            # rows dequantized at once: bounds dequantize's int64 unpacking to rows x K x 8 bytes
_HEAD_ROWS = 32768           # vocabulary rows of the head per slice
_QUERY_ROWS = 1024           # attention query rows per score block


class State:
    """One sequence's fp32 caches (KV, conv tails, recurrent states; bf16-rounded with ``bf16``) and position."""

    def __init__(self, cfg: Config, linear: list[bool], device: torch.device | str, *, bf16: bool = True):
        self.bf16 = bf16
        self.pos = 0
        self.kv: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        self.conv: list[torch.Tensor | None] = []
        self.rec: list[torch.Tensor | None] = []
        for lin in linear:
            if lin:
                self.conv.append(torch.zeros((cfg.conv_kernel - 1, cfg.conv_dim), dtype=F32, device=device))
                self.rec.append(torch.zeros((cfg.nv, cfg.dv, cfg.dk), dtype=F32, device=device))
                self.kv.append(None)
            else:
                empty = torch.zeros((0, cfg.kv_heads, cfg.head_dim), dtype=F32, device=device)
                self.conv.append(None)
                self.rec.append(None)
                self.kv.append((empty, empty.clone()))


def new_state(w: Weights, *, bf16: bool = True) -> State:
    """An empty sequence for the main model."""

    return State(w.cfg, [layer.linear for layer in w.layers], w.device, bf16=bf16)


def new_mtp_state(mtp: MTPW, w: Weights, *, bf16: bool = True) -> State:
    """An empty sequence for the MTP head (its one attention layer's KV cache)."""

    return State(mtp.cfg, [False], w.device, bf16=bf16)


# -- pieces ----------------------------------------------------------------------------------------------------
def _r(x: torch.Tensor, st: State) -> torch.Tensor:
    return x.to(BF).to(F32) if st.bf16 else x


def deq(q: QW) -> torch.Tensor:
    """A quantized matrix (or expert) as fp32 values, dequantized a slice of rows at a time."""

    if q.n <= _DEQ_ROWS:
        return dequantize(q)
    return torch.cat([dequantize(q.rows(s, min(q.n, s + _DEQ_ROWS))) for s in range(0, q.n, _DEQ_ROWS)], dim=-2)


def _expert(q: QW, e: int) -> QW:
    return QW(q.words[e], q.scales[e], q.biases[e], q.bits, q.group)


def _lin(x: torch.Tensor, q: QW, st: State) -> torch.Tensor:
    return _r(x @ deq(q).T, st)


def _rms(x: torch.Tensor, scale: torch.Tensor | None, eps: float) -> torch.Tensor:
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y * scale if scale is not None else y


def _l2(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def _rope(x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """x (T, H, D) fp32: rotate-half over the first 2 * len(inv_freq) dims of each head, the rest unchanged."""

    half = inv_freq.numel()
    ang = pos.to(F32)[:, None] * inv_freq[None, :]             # (T, half)
    cos, sin = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:2 * half]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., 2 * half:]], dim=-1)


def embed(w: Weights, tokens: torch.Tensor, st: State) -> torch.Tensor:
    """(T, hidden) fp32: the embedding rows of ``tokens``."""

    ids = tokens.to(w.device).long()
    e = w.embed
    return _r(dequantize(QW(e.words[ids], e.scales[ids], e.biases[ids], e.bits, e.group)), st)


def _gdn(g: GDNW, x: torch.Tensor, st: State, i: int, c: Config) -> torch.Tensor:
    T = x.shape[0]
    qkv, z, b, a = _lin(x, g.proj, st).split(list(c.gdn_rows), dim=-1)
    taps = c.conv_kernel
    inp = torch.cat([st.conv[i], qkv], 0)                      # (taps - 1 + T, conv_dim)
    st.conv[i] = inp[-(taps - 1):].clone()
    cw = g.conv.to(F32)                                        # (conv_dim, taps)
    conv = inp[0:T] * cw[:, 0]
    for j in range(1, taps):
        conv = conv + inp[j:j + T] * cw[:, j]
    conv = _r(F.silu(conv), st)
    kd = c.nk * c.dk
    rep = c.nv // c.nk
    q = (_l2(conv[:, :kd].reshape(T, c.nk, c.dk)) * c.dk ** -0.5).repeat_interleave(rep, 1)   # (T, nv, dk)
    k = _l2(conv[:, kd:2 * kd].reshape(T, c.nk, c.dk)).repeat_interleave(rep, 1)
    v = conv[:, 2 * kd:].reshape(T, c.nv, c.dv)
    beta = torch.sigmoid(b)                                    # (T, nv)
    decay = torch.exp(-torch.exp(g.a_log) * F.softplus(a + g.dt_bias))
    S = st.rec[i]                                              # (nv, dv, dk)
    y = torch.empty((T, c.nv, c.dv), dtype=F32, device=x.device)
    for t in range(T):
        S = S * decay[t][:, None, None]
        kt = k[t][:, None, :]
        delta = (v[t] - (S * kt).sum(-1)) * beta[t][:, None]
        S = S + delta[:, :, None] * kt
        y[t] = (S * q[t][:, None, :]).sum(-1)
    st.rec[i] = S
    y = _r(y, st)
    out = _r(_rms(y, g.norm.to(F32), c.eps) * F.silu(z.reshape(T, c.nv, c.dv)), st)
    return _lin(out.reshape(T, -1), g.out, st)


def _attention(at: AttnW, x: torch.Tensor, st: State, i: int, c: Config, inv_freq: torch.Tensor) -> torch.Tensor:
    T, D = x.shape[0], c.head_dim
    qg, k, v = _lin(x, at.proj, st).split(list(c.attn_rows), dim=-1)
    qg = qg.reshape(T, c.heads, 2 * D)
    q, gate = qg[..., :D], qg[..., D:].reshape(T, -1)
    k = k.reshape(T, c.kv_heads, D)
    v = v.reshape(T, c.kv_heads, D)
    pos = torch.arange(st.pos, st.pos + T, device=x.device)
    q = _r(_rope(_rms(q, at.q_scale, c.eps), pos, inv_freq), st)
    k = _r(_rope(_rms(k, at.k_scale, c.eps), pos, inv_freq), st)
    K0, V0 = st.kv[i]
    K, V = torch.cat([K0, k]), torch.cat([V0, v])
    st.kv[i] = (K, V)
    L = K.shape[0]
    rep = c.heads // c.kv_heads
    Kf, Vf = K.repeat_interleave(rep, 1), V.repeat_interleave(rep, 1)     # (L, heads, D)
    keys = torch.arange(L, device=x.device)
    o = torch.empty((T, c.heads, D), dtype=F32, device=x.device)
    for q0 in range(0, T, _QUERY_ROWS):
        q1 = min(T, q0 + _QUERY_ROWS)
        s = torch.einsum("thd,lhd->htl", q[q0:q1], Kf) * D ** -0.5
        last = st.pos + torch.arange(q0, q1, device=x.device)
        s = s.masked_fill(keys[None, None, :] > last[None, :, None], float("-inf"))
        o[q0:q1] = torch.einsum("htl,lhd->thd", torch.softmax(s, -1), Vf)
    o = _r(o, st).reshape(T, -1)
    o = _r(o * torch.sigmoid(gate), st)
    return _lin(o, at.o, st)


def route(m: MoEW, x: torch.Tensor, c: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """(ids, weights), each (T, top_k): fp32 softmax over the routed experts, top_k (ties to lower ids), renormed."""

    probs = torch.softmax(x @ m.router[:c.experts].T, dim=-1)
    vals, ids = torch.sort(probs, dim=-1, descending=True, stable=True)
    top, ids = vals[:, :c.top_k], ids[:, :c.top_k]
    if c.norm_topk:
        top = top / top.sum(-1, keepdim=True)
    return ids, top


def _swiglu(m: MoEW, e: int, x: torch.Tensor, st: State) -> torch.Tensor:
    h = _r(F.silu(x @ deq(_expert(m.gate, e)).T) * (x @ deq(_expert(m.up, e)).T), st)
    return _r(h @ deq(_expert(m.down, e)).T, st)


def _moe(m: MoEW, x: torch.Tensor, st: State, c: Config) -> torch.Tensor:
    """(T, hidden) fp32, unrounded: the caller adds it to the residual."""

    T = x.shape[0]
    ids, top = route(m, x, c)
    slots = torch.zeros((T, c.top_k, c.hidden), dtype=F32, device=x.device)
    for e in torch.unique(ids).tolist():
        rows, slot = (ids == e).nonzero(as_tuple=True)
        slots[rows, slot] = _swiglu(m, e, x[rows], st) * top[rows, slot, None]
    out = slots[:, 0]
    for s in range(1, c.top_k):
        out = out + slots[:, s]
    shared = c.experts                                          # the table's last expert
    return out + torch.sigmoid(x @ m.router[shared])[:, None] * _swiglu(m, shared, x, st)


def _layer(layer: LayerW, x: torch.Tensor, st: State, i: int, c: Config, inv_freq: torch.Tensor) -> torch.Tensor:
    h = _r(_rms(x, layer.input_scale, c.eps), st)
    r = _gdn(layer.gdn, h, st, i, c) if layer.linear else _attention(layer.attn, h, st, i, c, inv_freq)
    x = _r(x + r, st)
    h = _r(_rms(x, layer.post_scale, c.eps), st)
    return _r(x + _moe(layer.moe, h, st, c), st)


# -- the model -------------------------------------------------------------------------------------------------
@torch.no_grad()
def hidden(w: Weights, tokens: torch.Tensor, st: State, *, normed: bool = True) -> torch.Tensor:
    """(T, hidden) fp32 after the final norm (or before it) for ``tokens`` continuing the sequence in ``st``."""

    x = embed(w, tokens, st)
    for i, layer in enumerate(w.layers):
        x = _layer(layer, x, st, i, w.cfg, w.inv_freq)
    st.pos += int(tokens.shape[0])
    return final_norm(w, x, st) if normed else x


def final_norm(w: Weights, x: torch.Tensor, st: State) -> torch.Tensor:
    return _r(_rms(x, w.norm, w.cfg.eps), st)


@torch.no_grad()
def logits(w: Weights, h: torch.Tensor) -> torch.Tensor:
    """(T, vocab) fp32 from hidden states after the final norm (the head a slice at a time)."""

    hd = w.head
    return torch.cat([h @ deq(hd.rows(s, min(hd.n, s + _HEAD_ROWS))).T for s in range(0, hd.n, _HEAD_ROWS)], dim=1)


@torch.no_grad()
def head_top1(w: Weights, h: torch.Tensor, targets: torch.Tensor | None = None
              ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(argmax, -log p(target) or None) over vocabulary slices, never holding (T, vocab) logits."""

    hd = w.head
    T = h.shape[0]
    best = torch.full((T,), float("-inf"), dtype=F32, device=h.device)
    arg = torch.zeros((T,), dtype=torch.int64, device=h.device)
    lse = torch.full((T,), float("-inf"), dtype=F32, device=h.device)
    tgt = torch.zeros((T,), dtype=F32, device=h.device)
    targets = None if targets is None else targets.to(h.device).long()
    for s in range(0, hd.n, _HEAD_ROWS):
        e = min(hd.n, s + _HEAD_ROWS)
        z = h @ deq(hd.rows(s, e)).T
        m, a = z.max(dim=1)
        better = m > best
        best = torch.where(better, m, best)
        arg = torch.where(better, a + s, arg)
        lse = torch.logaddexp(lse, torch.logsumexp(z, dim=1))
        if targets is not None:
            inside = (targets >= s) & (targets < e)
            got = z.gather(1, (targets - s).clamp(0, e - s - 1)[:, None])[:, 0]
            tgt = torch.where(inside, got, tgt)
    return arg, (lse - tgt if targets is not None else None)


@torch.no_grad()
def forward(w: Weights, tokens: torch.Tensor, st: State) -> torch.Tensor:
    """(T, vocab) fp32 logits for ``tokens`` continuing the sequence in ``st``."""

    return logits(w, hidden(w, tokens, st))


@torch.no_grad()
def greedy(w: Weights, prompt: list[int], max_new: int, eos: tuple[int, ...] = (), *, bf16: bool = True) -> list[int]:
    """Greedy continuation from the caches, up to ``max_new`` tokens or an eos id (included)."""

    st = new_state(w, bf16=bf16)
    h = hidden(w, torch.tensor(prompt), st)[-1:]
    out: list[int] = []
    while len(out) < max_new:
        token = int(head_top1(w, h)[0][0])
        out.append(token)
        if token in eos:
            break
        h = hidden(w, torch.tensor([token]), st)
    return out


# -- the MTP head ----------------------------------------------------------------------------------------------
@torch.no_grad()
def mtp_hidden(mtp: MTPW, w: Weights, tokens: torch.Tensor, h: torch.Tensor, st: State) -> torch.Tensor:
    """(T, hidden) fp32 after the MTP head's norm: row t reads ``tokens[t]`` and the target's hidden row ``h[t]``."""

    c = mtp.cfg
    e = _r(_rms(embed(w, tokens, st), mtp.norm_e, c.eps), st)
    hh = _r(_rms(h.to(F32), mtp.norm_h, c.eps), st)
    x = _lin(torch.cat([e, hh], dim=-1), mtp.fc, st)
    x = _layer(mtp.layer, x, st, 0, c, w.inv_freq)
    st.pos += int(tokens.shape[0])
    return _r(_rms(x, mtp.norm, c.eps), st)
