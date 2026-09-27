"""SPIKE: several requests in one exact verify forward (Qwen3.8 dense, one GPU).

Every weight matmul, norm and elementwise kernel in ``tree_forward`` computes each row on its own (the lane
contract), so rows from different requests can share one launch without changing a row's bits. Only the
sequence-local steps differ per request: the GDN convolution and recurrence (``st.conv``, ``st.rec``), full
attention over the request's own committed keys (``st.kv``) and the commit. This module runs those per request
and everything else once for all rows, so the weights are read once per round for the whole batch.

Contract under test: every request's tokens equal its own serial decode, whatever else shares the forward.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import torch

from tensorfold.engine.exact_sampling import Sampling

from . import gdn_tree, glue
from .decode import CopyIndex, _tokens, clone_state, prefill
from .forward import AttentionRecord, GDNRecord, State, _conv_windows, _mm, _paths, commit
from .sampling import sample_rows
from .weights import Weights

TAP_LAYERS = (5, 19, 33, 47, 61)


@dataclass
class Item:
    tokens: list[int]
    parents: list[int]
    st: State


@torch.no_grad()
def batch_forward(w: Weights, items: Sequence[Item], *, capture_taps: bool = False):
    """``tree_forward`` for several independent windows at once. Returns one (logits, record, taps) per item."""

    c = w.config
    dev = w.norm.device
    spans, rows = [], 0
    for it in items:
        spans.append((rows, rows + len(it.tokens)))
        rows += len(it.tokens)
    if rows > 128:
        raise ValueError("the lane matmul takes at most 128 rows a launch")
    meta = []
    for it in items:
        depths, chain = _paths(it.parents)
        meta.append(dict(
            chain=chain,
            parent_buf=torch.tensor(it.parents, device=dev, dtype=torch.int32),
            windows=_conv_windows(it.parents, c.conv_kernel - 1).to(dev),
            pos=[it.st.pos + d for d in depths]))
    ids = torch.tensor([t for it in items for t in it.tokens], device=dev, dtype=torch.int32)
    pos = torch.tensor([p for m in meta for p in m["pos"]], device=dev, dtype=torch.int32)
    x = glue.embed(ids, w.embed.weight, w.embed.scales, w.embed.biases, c.hidden)
    pending = None
    records: list[list] = [[] for _ in items]
    taps: list[torch.Tensor] = []
    for i, layer in enumerate(w.layers):
        x, h, xs = glue.add_rmsnorm(x, pending, layer.input_norm, c.eps)
        if layer.linear:
            gdn = layer.gdn
            qkv = _mm(h, gdn.qkv, xs)
            if gdn.zba is not None:
                zba = _mm(h, gdn.zba, xs)
                vd = c.v_heads * c.dv
                z = zba[:, :vd].contiguous().reshape(rows, c.v_heads, c.dv)
                b = zba[:, vd:vd + c.v_heads].contiguous()
                a = zba[:, vd + c.v_heads:].contiguous()
            else:
                z = _mm(h, gdn.z, xs).reshape(rows, c.v_heads, c.dv)
                b = _mm(h, gdn.b, xs)
                a = _mm(h, gdn.a, xs)
            ys = []
            for j, (it, m, (r0, r1)) in enumerate(zip(items, meta, spans)):
                q, k, v, g, beta = glue.gdn_pre(qkv[r0:r1], it.st.conv[i], gdn.conv, m["windows"], a[r0:r1], b[r0:r1],
                                                gdn.A_log, gdn.dt_bias, kh=c.k_heads, vh=c.v_heads, dk=c.dk)
                ys.append(gdn_tree.tree(q, k, v, g, beta, it.st.rec[i], m["parent_buf"], chain=m["chain"]))
                records[j].append(GDNRecord(q, k, v, g, beta, qkv[r0:r1]))
            yr = ys[0] if len(ys) == 1 else torch.cat(ys, dim=0)
            out, out_xs = glue.gated_norm(yr, z, gdn.norm, c.eps)
            r = _mm(out, gdn.out, out_xs)
        else:
            from .attention import attention

            attn = layer.attn
            qg = _mm(h, attn.q, xs)
            if attn.kv is not None:
                kv = _mm(h, attn.kv, xs)
                kd = c.kv_heads * c.head_dim
                key = kv[:, :kd].contiguous()
                value = kv[:, kd:].contiguous().reshape(rows, c.kv_heads, c.head_dim)
            else:
                key = _mm(h, attn.k, xs)
                value = _mm(h, attn.v, xs).reshape(rows, c.kv_heads, c.head_dim)
            q, key = glue.attn_prep(qg, key, attn.q_norm, attn.k_norm, pos, w.inv_freq, c.eps,
                                    heads=c.heads, kv_heads=c.kv_heads, head_dim=c.head_dim)
            outs = []
            for j, (it, m, (r0, r1)) in enumerate(zip(items, meta, spans)):
                old_k, old_v = it.st.kv[i]
                kj, vj = key[r0:r1].contiguous(), value[r0:r1].contiguous()
                outs.append(attention(q[r0:r1].contiguous(), kj, vj, old_k[:it.st.pos], old_v[:it.st.pos],
                                      m["parent_buf"], scale=c.head_dim ** -0.5))
                records[j].append(AttentionRecord(kj, vj))
            o = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
            gated, out_xs = glue.gate_mul(o, qg, heads=c.heads, head_dim=c.head_dim)
            r = _mm(gated, attn.o, out_xs)
        x, h, xs = glue.add_rmsnorm(x, r, layer.post_norm, c.eps)
        gate = _mm(h, layer.gate, xs)
        up = _mm(h, layer.up, xs)
        act, act_xs = glue.swiglu(gate, up)
        pending = _mm(act, layer.down, act_xs)
        if capture_taps and i in TAP_LAYERS:
            taps.append((x.float() + pending.float()).to(torch.bfloat16))
    _, h, xs = glue.add_rmsnorm(x, pending, w.norm, c.eps)
    logits = _mm(h, w.head, xs)
    tap = torch.cat(taps, dim=-1) if capture_taps else None
    return [(logits[r0:r1], records[j], tap[r0:r1] if tap is not None else None) for j, (r0, r1) in enumerate(spans)]


@dataclass
class Job:
    prompt: list[int]
    count: int
    sampling: Sampling | None = None
    st: State | None = None
    snap: object = None
    out: list[int] = field(default_factory=list)
    context: list[int] = field(default_factory=list)
    copies: CopyIndex | None = None
    done: bool = False
    rounds: int = 0
    accepted: int = 0


@torch.no_grad()
def batch_decode(w: Weights, jobs: list[Job], draft=None, *, row_budget: int = 128, max_rows: int = 12,
                 allow_copy: bool = True, stop_eos: bool = True) -> dict:
    """Prefill every job, then decode them together: each round every active job proposes a window
    (copies from its context, else a DFlash2 tree), all windows share one forward, each job commits its path."""

    t0 = time.perf_counter()
    for j in jobs:
        if draft is not None:
            draft.restore(([None] * draft.layers, [None] * draft.layers, 0, 0))
        j.st, pending = prefill(w, j.prompt, j.sampling, draft)
        j.snap = draft.snapshot() if draft is not None else None
        j.out = [pending]
        j.context = list(j.prompt) + [pending]
        j.copies = CopyIndex() if allow_copy else None
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    stats = dict(rounds=0, rows=0, verify_s=0.0, draft_s=0.0, commit_s=0.0)
    eos = set(w.config.eos)
    while True:
        active = [j for j in jobs if not j.done]
        if not active:
            break
        per = max(1, min(max_rows, row_budget // len(active)))
        stage = time.perf_counter()
        items = []
        for j in active:
            copied = j.copies.propose(j.context, per - 1) if j.copies is not None else []
            if copied:
                guesses, parents = copied, list(range(-1, len(copied) - 1))
            elif draft is not None and per > 1:
                draft.restore(j.snap)
                guesses, parents = draft.propose_tree(j.out[-1], len(j.context), per - 1, j.sampling)
            else:
                guesses, parents = [], []
            tokens = [j.out[-1]] + guesses
            items.append(Item(tokens, [-1] + [0 if p < 0 else p + 1 for p in parents], j.st))
        torch.cuda.synchronize()
        stats["draft_s"] += time.perf_counter() - stage
        stage = time.perf_counter()
        results = batch_forward(w, items, capture_taps=draft is not None)
        torch.cuda.synchronize()
        stats["verify_s"] += time.perf_counter() - stage
        stage = time.perf_counter()
        for j, it, (logits, record, taps) in zip(active, items, results):
            depths, _ = _paths(it.parents)
            sampled = sample_rows(logits, [j.st.pos + d + 1 for d in depths], j.sampling)
            children: dict[tuple[int, int], int] = {}
            for row in range(1, len(it.tokens)):
                children.setdefault((it.parents[row], it.tokens[row]), row)
            path, terminal = [0], sampled[0]
            while len(j.out) + len(path) < j.count:
                if stop_eos and terminal in eos:
                    break
                child = children.get((path[-1], terminal))
                if child is None:
                    break
                path.append(child)
                terminal = sampled[child]
            commit(j.st, record, path)
            if draft is not None:
                draft.restore(j.snap)
                draft.add_taps(taps[path])
                j.snap = draft.snapshot()
            new = [it.tokens[row] for row in path[1:]] + [terminal]
            j.out.extend(new)
            j.context.extend(new)
            j.rounds += 1
            j.accepted += len(path) - 1
            stats["rows"] += len(it.tokens)
            if len(j.out) >= j.count or (stop_eos and j.out[-1] in eos):
                j.done = True
        torch.cuda.synchronize()
        stats["commit_s"] += time.perf_counter() - stage
        stats["rounds"] += 1
    decode_s = time.perf_counter() - t1
    gen = sum(len(j.out) - 1 for j in jobs)
    return dict(prefill_s=prefill_s, decode_s=decode_s, generated=gen, agg_tok_s=gen / decode_s if decode_s else 0.0,
                **stats)
