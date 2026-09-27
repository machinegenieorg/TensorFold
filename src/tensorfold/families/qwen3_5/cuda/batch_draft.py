"""SPIKE: one DFlash2 draft pass for several requests (Qwen3.8 dense, one GPU, fused 4-bit path).

``DFlash2.propose_tree`` reads the drafter's ~1 GB of weights once per request per round and ``add_taps``
copies each request's whole 2,047-row key/value context every round. Here every request's block shares one
pass: the projections run once over all rows, attention is one call over per-request slots, and a slot is
written in place (a request's context is the ``ctx`` rows before its write pointer ``ptr``).

Drafts only change acceptance, never output: the verifier checks every proposed token.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .dflash2 import DFlash2, _best_first, _norm
from .glue import embed, swiglu
from .qmm import group_sums
from .qmm_fast import matmul


class _Prof:
    """CUDA-event sections of one propose/add_taps call (``DPROF.on``), summed into ``totals``."""

    def __init__(self):
        self.on, self.marks, self.totals = False, [], {}

    def mark(self, name):
        if self.on:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.marks.append((name, e))

    def flush(self):
        if not self.marks:
            return
        torch.cuda.synchronize()
        for (name, a), (_, b) in zip(self.marks, self.marks[1:]):
            self.totals[name] = self.totals.get(name, 0.0) + a.elapsed_time(b) / 1000
        self.marks = []


DPROF = _Prof()


@triton.jit
def _dconv_seg_kernel(X, DYN, BASE, RES, OUT, SEG, D: tl.constexpr, G: tl.constexpr, GS: tl.constexpr,
                      BRANCH: tl.constexpr, HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    """``dflash2._dconv_kernel`` over stacked blocks of ``SEG`` rows: a block's first row has no previous row."""

    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = c < D
    x = tl.load(X + row * D + c, mask=ok, other=0.0).to(tl.float32)
    prev = tl.load(X + (row - 1) * D + c, mask=ok & (row % SEG > 0), other=0.0).to(tl.float32)
    grp = c // GS
    d0 = tl.load(DYN + ((row * 2 + BRANCH) * 2) * G + grp, mask=ok, other=0.0).to(tl.float32)
    d1 = tl.load(DYN + ((row * 2 + BRANCH) * 2 + 1) * G + grp, mask=ok, other=0.0).to(tl.float32)
    b0 = tl.load(BASE + (BRANCH * 2) * D + c, mask=ok, other=0.0).to(tl.float32)
    b1 = tl.load(BASE + (BRANCH * 2 + 1) * D + c, mask=ok, other=0.0).to(tl.float32)
    k0 = (b0 + d0).to(tl.bfloat16).to(tl.float32)
    k1 = (b1 + d1).to(tl.bfloat16).to(tl.float32)
    y = (x * k0 + prev * k1).to(tl.bfloat16)
    if HAS_RES:
        r = tl.load(RES + row * D + c, mask=ok, other=0.0).to(tl.float32)
        y = (r + y.to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + row * D + c, y, mask=ok)


def _dconv(x, dyn, base, branch, group_size, seg, residual=None):
    rows, d = x.shape
    x, dyn = x.contiguous(), dyn.contiguous()
    out = torch.empty_like(x)
    _dconv_seg_kernel[(rows, triton.cdiv(d, 1024))](x, dyn, base, residual if residual is not None else x, out, seg,
                                                    D=d, G=d // group_size, GS=group_size, BRANCH=branch,
                                                    HAS_RES=residual is not None, BLOCK=1024, num_warps=4)
    return out


class BatchDraft:
    """Per-request DFlash2 context in fixed slots, drafted and extended for many requests at once."""

    def __init__(self, d: DFlash2, slots: int, slack: int = 512):
        if not d.fast or d.world != 1:
            raise ValueError("the batched drafter needs the fused single-rank 4-bit path")
        self.d, self.slots = d, slots
        self.cap = -(-(d.window + slack) // 64) * 64
        shape = (slots, d.kv_local, self.cap, d.head_dim)
        # zeros, not empty: a masked key still multiplies its value by 0, and 0 * NaN garbage is NaN
        self.kc = [torch.zeros(shape, dtype=torch.bfloat16, device=d.device) for _ in range(d.layers)]
        self.vc = [torch.zeros(shape, dtype=torch.bfloat16, device=d.device) for _ in range(d.layers)]
        self.ptr = [0] * slots      # next write row
        self.ctx = [0] * slots      # context rows before ptr
        self.end = [0] * slots      # absolute position of the next context row (rotary)
        self.free = list(range(slots))

    # ---- slots ----
    def acquire(self) -> int:
        return self.free.pop(0)

    def release(self, slot: int) -> None:
        self.ptr[slot] = self.ctx[slot] = self.end[slot] = 0
        self.free.append(slot)

    def load(self, slot: int, snap) -> None:
        """Copy a ``DFlash2.snapshot()`` (e.g. after a single-request prefill) into a slot."""

        kc, vc, n, end = snap
        for layer in range(self.d.layers):
            if n:
                self.kc[layer][slot, :, :n] = kc[layer][:, -n:]
                self.vc[layer][slot, :, :n] = vc[layer][:, -n:]
        self.ptr[slot], self.ctx[slot], self.end[slot] = n, n, end

    def snapshot(self, slot: int):
        """A slot's context in ``DFlash2.snapshot()`` form (copies), for ``load``."""

        n, p = self.ctx[slot], self.ptr[slot]
        kc = [buf[slot, :, p - n:p].clone() for buf in self.kc]
        vc = [buf[slot, :, p - n:p].clone() for buf in self.vc]
        return kc, vc, n, self.end[slot]

    def _room(self, slot: int, rows: int) -> None:
        """Make ``rows`` free rows after the pointer: move the context to the front of the slot."""

        if self.ptr[slot] + rows <= self.cap:
            return
        n, p = self.ctx[slot], self.ptr[slot]
        for buf in self.kc + self.vc:
            buf[slot, :, :n] = buf[slot, :, p - n:p].clone()
        self.ptr[slot] = n

    # ---- context ----
    @torch.no_grad()
    def add_taps(self, slots: list[int], taps, counts: list[int] | None = None) -> None:
        """Append each slot's committed target taps (rows, 5 * hidden) to its context: a list of one tensor
        per slot, or one tensor of every slot's rows back to back with ``counts``."""

        d = self.d
        if counts is None:
            counts = [t.shape[0] for t in taps]
            x = torch.cat(taps) if len(taps) > 1 else taps[0]
        else:
            x = taps
        for s, n in zip(slots, counts):
            self._room(s, n + d.block)
        DPROF.mark("t_fc")
        projected = _norm(d._lin(x, "fc.weight"), d.weights["hidden_norm.weight"], d.eps)
        dev = d.device
        pos = torch.cat([torch.arange(self.end[s], self.end[s] + n, device=dev) for s, n in zip(slots, counts)])
        phase = pos.float()[:, None] * d.inv_freq[None, :]
        cos, sin = phase.cos().contiguous(), phase.sin().contiguous()
        bi = torch.tensor([s for s, n in zip(slots, counts) for _ in range(n)], device=dev)
        ti = torch.cat([torch.arange(self.ptr[s], self.ptr[s] + n, device=dev) for s, n in zip(slots, counts)])
        DPROF.mark("t_kv")
        for layer in range(d.layers):
            _, k, v = d._prep(d._lin(projected, f"layers.{layer}.self_attn.kv.weight"), layer, cos, sin, 0)
            self.kc[layer][bi, :, ti] = k.transpose(0, 1)
            self.vc[layer][bi, :, ti] = v.transpose(0, 1)
        DPROF.mark("t_end")
        DPROF.flush()
        for s, n in zip(slots, counts):
            self.ptr[s] += n
            self.ctx[s] = min(d.window, self.ctx[s] + n)
            self.end[s] += n

    # ---- proposals ----
    @torch.no_grad()
    def propose(self, slots: list[int], pending: list[int], context_lengths: list[int], max_nodes: int,
                samplings: list) -> list[tuple[list[int], list[int]]]:
        """One draft tree per slot, all of the same block length."""

        d = self.d
        dev = d.device
        b = len(slots)
        DPROF.mark("d_setup")
        length = min(d.block, max_nodes + 1)
        for s in slots:
            self._room(s, length)
        tokens = torch.tensor([t for p in pending for t in [p] + [d.mask_id] * (length - 1)], dtype=torch.int32, device=dev)
        x = embed(tokens, d.target.embed.weight, d.target.embed.scales, d.target.embed.biases, d.hidden)
        pos = torch.cat([torch.arange(self.end[s], self.end[s] + length, device=dev) for s in slots])
        phase = pos.float()[:, None] * d.inv_freq[None, :]
        cos, sin = phase.cos().contiguous(), phase.sin().contiguous()
        # where the block's keys go (scratch rows after the pointer; the next add_taps overwrites them)
        bi = torch.tensor([s for s in slots for _ in range(length)], device=dev)
        ti = torch.cat([torch.arange(self.ptr[s], self.ptr[s] + length, device=dev) for s in slots])
        # mask (b, 1, L, cap): context row at age a (1 = newest) is visible to block row q iff a + q <= window
        ptr = torch.tensor([self.ptr[s] for s in slots], device=dev)[:, None, None]
        ctx = torch.tensor([self.ctx[s] for s in slots], device=dev)[:, None, None]
        kidx = torch.arange(self.cap, device=dev)[None, None, :]
        qidx = torch.arange(length, device=dev)[None, :, None]
        age = ptr - kidx
        block_ok = (kidx >= ptr) & (kidx < ptr + length)
        if d.is_causal:
            block_ok = block_ok & (kidx - ptr <= qidx)
        mask = ((age >= 1) & (age <= ctx) & (age + qidx <= d.window)) | block_ok
        kv, G = d.kv_local, d.heads_local // d.kv_local
        gmask = mask[:, None, None].expand(b, 1, G, length, self.cap).reshape(b, 1, G * length, self.cap)
        # consecutive slots are a view; otherwise gather them (a copy)
        run = slots == list(range(slots[0], slots[0] + b))
        sl = slice(slots[0], slots[0] + b) if run else torch.tensor(slots, device=dev)
        seg = length
        w = d.weights
        for i in range(d.layers):
            DPROF.mark("d_proj")
            base = f"layers.{i}."
            normed = F.rms_norm(x, (d.hidden,), w[base + "input_layernorm.weight"], d.eps)
            dyn = d._lin(normed, base + "attention_conv.kernel_projection.weight")
            conv = w[base + "attention_conv.base_kernel"]
            q, k, v = d._prep(d._lin(_dconv(normed, dyn, conv, 0, d.group_size, seg), base + "self_attn.qkv.weight"),
                              i, cos, sin, d.heads_local)
            self.kc[i][bi, :, ti] = k.transpose(0, 1)
            self.vc[i][bi, :, ti] = v.transpose(0, 1)
            DPROF.mark("d_attn")
            # grouped attention without expanding the keys: the G query heads of a key head share one matmul
            qg = q.view(kv, G, b, length, d.head_dim).permute(2, 0, 1, 3, 4).reshape(b, kv, G * length, d.head_dim)
            keys, values = self.kc[i][sl], self.vc[i][sl]
            scores = torch.matmul(qg, keys.transpose(-1, -2)).float() * d.head_dim ** -0.5
            p = torch.softmax(scores.masked_fill_(~gmask, float("-inf")), dim=-1).to(torch.bfloat16)
            out = torch.matmul(p, values).view(b, kv, G, length, d.head_dim)
            out = out.permute(0, 3, 1, 2, 4).reshape(b * length, d.heads_local * d.head_dim)
            x = _dconv(self._chunked(out, base + "self_attn.o_proj.weight"), dyn, conv, 1, d.group_size, seg, x)
            DPROF.mark("d_mlp")
            normed = F.rms_norm(x, (d.hidden,), w[base + "post_attention_layernorm.weight"], d.eps)
            dyn = d._lin(normed, base + "mlp_conv.kernel_projection.weight")
            conv = w[base + "mlp_conv.base_kernel"]
            h = _dconv(normed, dyn, conv, 0, d.group_size, seg)
            act, act_xs = self._mlp(h, base)
            mlp = self._chunked(act, base + "mlp.down_proj.weight", act_xs)
            x = _dconv(mlp, dyn, conv, 1, d.group_size, seg, x)
        DPROF.mark("d_head")
        h = x.view(b, length, d.hidden)[:, 1:].reshape(b * (length - 1), d.hidden)
        h = F.rms_norm(h, (d.hidden,), w["norm.weight"], d.eps)
        projected = d._lin(h, "candidate_selector.hidden_projection.weight").float()
        logits = torch.cat([matmul(h[r:r + 128].contiguous(), d.sub_head) for r in range(0, h.shape[0], 128)])
        values, local_ids = torch.topk(logits.float(), k=16, dim=-1, sorted=False)
        ids = d.head_ids[local_ids]
        DPROF.mark("d_copy")
        ids, unary, hproj = (t.cpu().numpy() for t in (ids, values, projected))
        DPROF.mark("d_tree")
        ids = ids.astype("int64").reshape(b, length - 1, 16)
        unary = unary.astype("float64").reshape(b, length - 1, 16)
        hproj = hproj.astype("float64").reshape(b, length - 1, -1)
        out = []
        for j in range(b):
            out.append(_best_first(ids[j], unary[j], hproj[j], w["candidate_selector.predecessor_codebook"],
                                   w["candidate_selector.successor_codebook"], int(pending[j]), min(127, max_nodes),
                                   samplings[j], context_lengths[j]))
        DPROF.mark("d_end")
        DPROF.flush()
        return out

    def _mlp(self, h: torch.Tensor, base: str):
        d = self.d
        outs = []
        for r in range(0, h.shape[0], 128):
            hc = h[r:r + 128].contiguous()
            xs = group_sums(hc)
            outs.append(swiglu(matmul(hc, d.q4[base + "mlp.gate_proj.weight"], xs),
                               matmul(hc, d.q4[base + "mlp.up_proj.weight"], xs)))
        if len(outs) == 1:
            return outs[0]
        return torch.cat([a for a, _ in outs]), torch.cat([s for _, s in outs])

    def _chunked(self, x: torch.Tensor, name: str, xs: torch.Tensor | None = None) -> torch.Tensor:
        d = self.d
        if x.shape[0] <= 128:
            return matmul(x.contiguous(), d.q4[name], xs)
        # group sums are per row (and per group along K), so a row slice of xs is that row's sums
        return torch.cat([matmul(x[r:r + 128].contiguous(), d.q4[name],
                                 None if xs is None else xs[r:r + 128].contiguous())
                          for r in range(0, x.shape[0], 128)])
