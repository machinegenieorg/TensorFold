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
from .forward import AttentionRecord, GDNRecord, State, _conv_windows, _mm, _paths, commit, tree_forward
from .sampling import sample_rows
from .weights import Weights

TAP_LAYERS = (5, 19, 33, 47, 61)


class Prof:
    """Optional CUDA-event section timer: ``mark(name)`` starts a section that runs until the next mark."""

    def __init__(self):
        self.on, self.marks, self.totals = False, [], {}

    def mark(self, name: str) -> None:
        if self.on:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.marks.append((name, e))

    def flush(self) -> None:
        if not self.marks:
            return
        torch.cuda.synchronize()
        for (name, a), (_, b) in zip(self.marks, self.marks[1:]):
            self.totals[name] = self.totals.get(name, 0.0) + a.elapsed_time(b) / 1000
        self.marks = []


PROF = Prof()


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
    PROF.mark("setup")
    ids = torch.tensor([t for it in items for t in it.tokens], device=dev, dtype=torch.int32)
    pos = torch.tensor([p for m in meta for p in m["pos"]], device=dev, dtype=torch.int32)
    x = glue.embed(ids, w.embed.weight, w.embed.scales, w.embed.biases, c.hidden)
    PROF.mark("dense")
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
            PROF.mark("gdn_loop")
            for j, (it, m, (r0, r1)) in enumerate(zip(items, meta, spans)):
                q, k, v, g, beta = glue.gdn_pre(qkv[r0:r1], it.st.conv[i], gdn.conv, m["windows"], a[r0:r1], b[r0:r1],
                                                gdn.A_log, gdn.dt_bias, kh=c.k_heads, vh=c.v_heads, dk=c.dk)
                ys.append(gdn_tree.tree(q, k, v, g, beta, it.st.rec[i], m["parent_buf"], chain=m["chain"]))
                records[j].append(GDNRecord(q, k, v, g, beta, qkv[r0:r1]))
            PROF.mark("dense")
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
            PROF.mark("attn_loop")
            for j, (it, m, (r0, r1)) in enumerate(zip(items, meta, spans)):
                old_k, old_v = it.st.kv[i]
                kj, vj = key[r0:r1].contiguous(), value[r0:r1].contiguous()
                outs.append(attention(q[r0:r1].contiguous(), kj, vj, old_k[:it.st.pos], old_v[:it.st.pos],
                                      m["parent_buf"], scale=c.head_dim ** -0.5))
                records[j].append(AttentionRecord(kj, vj))
            PROF.mark("dense")
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
    PROF.mark("head")
    _, h, xs = glue.add_rmsnorm(x, pending, w.norm, c.eps)
    logits = _mm(h, w.head, xs)
    PROF.mark("end")
    tap = torch.cat(taps, dim=-1) if capture_taps else None
    return [(logits[r0:r1], records[j], tap[r0:r1] if tap is not None else None) for j, (r0, r1) in enumerate(spans)]


def _windows(parents: Sequence[int], keep: int) -> list[list[int]]:
    """``forward._conv_windows`` as lists."""

    windows: list[list[int]] = []
    for row, parent in enumerate(parents):
        tail = list(range(keep)) if parent < 0 else windows[parent][1:]
        windows.append(tail + [keep + row])
    return windows


@dataclass
class Forward:
    """One multi-request verify forward: full-row outputs, and what a commit needs."""

    logits: torch.Tensor
    spans: list[tuple[int, int]]
    taps: torch.Tensor | None
    gdn: dict            # layer -> (k, v, g, beta, src): src = [every request's conv state | qkv rows]
    att: dict            # layer -> (key, value)
    states: list          # host (layers, items) int64 recurrent-state pointers
    keep: int
    finals: list          # per item: index into a GDN layer's finals, or -1


@torch.no_grad()
def batch_forward_multi(w: Weights, items: Sequence[Item], *, capture_taps: bool = False,
                        multi_attention: bool = True, logit_rows: Sequence[int] | None = None,
                        committed: Sequence[int] = ()) -> Forward:
    """``batch_forward`` with every request's GDN pre and tree in one launch each (gdn_multi), and every
    request's tree attention in one launch per kernel (attention_multi)."""

    from . import gdn_multi
    from .attention import attention
    from .attention_multi import Plan

    c = w.config
    dev = w.norm.device
    n = len(items)
    keep = c.conv_kernel - 1
    spans, rows = [], 0
    for it in items:
        spans.append((rows, rows + len(it.tokens)))
        rows += len(it.tokens)
    if rows > 128:
        raise ValueError("the lane matmul takes at most 128 rows a launch")
    ns = n * keep
    parents, wins, pos, chains, offsets = [], [], [], [], [0]
    for j, it in enumerate(items):
        depths, chain = _paths(it.parents)
        r0 = spans[j][0]
        parents += it.parents
        for win in _windows(it.parents, keep):
            wins += [j * keep + s if s < keep else ns + r0 + s - keep for s in win]
        pos += [it.st.pos + d for d in depths]
        chains.append(int(chain))
        offsets.append(spans[j][1])
    # ``committed``: items (chains) whose every row will be committed; their last GDN states come from the tree walk
    fin = [-1] * n
    for f_i, j in enumerate(j for j in committed if chains[j]):
        fin[j] = f_i
    n_final = sum(1 for x in fin if x >= 0)
    ints = torch.tensor(parents + wins + pos + offsets + chains + fin, dtype=torch.int32).to(dev)
    parents_d = ints[:rows]
    wins_d = ints[rows:rows + rows * (keep + 1)].view(rows, keep + 1)
    at = rows * (keep + 2)
    pos_d = ints[at:at + rows]
    offsets_d = ints[at + rows:at + rows + n + 1]
    chain_d = ints[at + rows + n + 1:at + rows + 2 * n + 1]
    fin_d = ints[at + rows + 2 * n + 1:]
    states = [[it.st.rec[i].data_ptr() if layer.linear else 0 for it in items] for i, layer in enumerate(w.layers)]
    states_d = torch.tensor(states, dtype=torch.int64).to(dev)
    att_layers = [i for i, layer in enumerate(w.layers) if not layer.linear]
    plan = None
    if multi_attention:
        plan = Plan(parents_d, offsets_d, [it.st.pos for it in items], [len(it.tokens) for it in items],
                    [[(it.st.kv[i][0], it.st.kv[i][1]) for it in items] for i in att_layers], c.heads, c.head_dim)
    att_index = {i: li for li, i in enumerate(att_layers)}
    PROF.mark("dense")
    ids = torch.tensor([t for it in items for t in it.tokens], dtype=torch.int32).to(dev)
    x = glue.embed(ids, w.embed.weight, w.embed.scales, w.embed.biases, c.hidden)
    pending = None
    gdn_out, att_out, taps = {}, {}, []
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
            PROF.mark("gdn_loop")
            src = torch.cat([it.st.conv[i] for it in items] + [qkv])
            q, k, v, g, beta = gdn_multi.pre(src, gdn.conv, wins_d, a, b, gdn.A_log, gdn.dt_bias,
                                             kh=c.k_heads, vh=c.v_heads, dk=c.dk, nkeep=keep)
            yr, finals = gdn_multi.tree(q, k, v, g, beta, states_d[i], parents_d, offsets_d, chain_d, fin_d, n_final)
            gdn_out[i] = (k, v, g, beta, src, finals)
            PROF.mark("dense")
            out, out_xs = glue.gated_norm(yr, z, gdn.norm, c.eps)
            r = _mm(out, gdn.out, out_xs)
        else:
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
            q, key = glue.attn_prep(qg, key, attn.q_norm, attn.k_norm, pos_d, w.inv_freq, c.eps,
                                    heads=c.heads, kv_heads=c.kv_heads, head_dim=c.head_dim)
            PROF.mark("attn_loop")
            if plan is not None:
                o = plan(att_index[i], q, key, value, scale=c.head_dim ** -0.5)
            else:
                outs = []
                for j, (it, (r0, r1)) in enumerate(zip(items, spans)):
                    old_k, old_v = it.st.kv[i]
                    outs.append(attention(q[r0:r1], key[r0:r1], value[r0:r1], old_k[:it.st.pos],
                                          old_v[:it.st.pos], parents_d[r0:r1], scale=c.head_dim ** -0.5))
                o = outs[0] if n == 1 else torch.cat(outs, dim=0)
            att_out[i] = (key, value)
            PROF.mark("dense")
            gated, out_xs = glue.gate_mul(o, qg, heads=c.heads, head_dim=c.head_dim)
            r = _mm(gated, attn.o, out_xs)
        x, h, xs = glue.add_rmsnorm(x, r, layer.post_norm, c.eps)
        gate = _mm(h, layer.gate, xs)
        up = _mm(h, layer.up, xs)
        act, act_xs = glue.swiglu(gate, up)
        pending = _mm(act, layer.down, act_xs)
        if capture_taps and i in TAP_LAYERS:
            taps.append((x.float() + pending.float()).to(torch.bfloat16))
    PROF.mark("head")
    _, h, xs = glue.add_rmsnorm(x, pending, w.norm, c.eps)
    if logit_rows is not None:
        # only the rows whose next token is needed (rows are independent, so the same bits)
        pick = torch.tensor(list(logit_rows), dtype=torch.int64).to(dev)
        h, xs = h.index_select(0, pick), xs.index_select(0, pick)
    logits = _mm(h, w.head, xs) if h.shape[0] else h.new_empty((0, 0))
    PROF.mark("end")
    return Forward(logits, spans, torch.cat(taps, dim=-1) if capture_taps else None, gdn_out, att_out, states, keep,
                   fin)


@torch.no_grad()
def commit_many(w: Weights, items: Sequence[Item], f: Forward, paths: Sequence[Sequence[int]]):
    """``commit`` for every request at once: one GDN replay launch for every (layer, request), one gather per
    GDN layer for the conv rows, one multi-tensor copy for every new key/value row. Returns the accepted rows
    (global indices, on the device) for the drafter taps."""

    from . import gdn_multi

    c = w.config
    dev = w.norm.device
    n, keep = len(items), f.keep
    ns = n * keep
    counts = [len(p) for p in paths]
    take, conv_idx, rows = [], [], []
    for j, ((r0, _), p) in enumerate(zip(f.spans, paths)):
        take += [r0 + r for r in p]
        seq = [j * keep + t for t in range(keep)] + [ns + r0 + r for r in p]
        conv_idx += seq[-keep:]
        rows += list(p) + [0] * (128 - len(p))
    ints = torch.tensor(rows + counts + conv_idx + take, dtype=torch.int32).to(dev)
    rows_d = ints[:n * 128].view(n, 128)
    counts_d = ints[n * 128:n * 128 + n]
    conv_d = ints[n * 128 + n:n * 128 + n + ns]
    take_d = ints[n * 128 + n + ns:]
    gl = sorted(f.gdn)
    for j, it in enumerate(items):
        # a committed chain's states were already updated in place by the tree walk
        if f.finals[j] >= 0 and list(paths[j]) != list(range(len(it.tokens))):
            raise ValueError("a committed chain must commit every row")
    replay_jobs = [j for j in range(n) if f.finals[j] < 0]
    if gl and replay_jobs:
        # pair (request j, GDN layer li) -> column index(j) * len(gl) + li; each request's rows start at its span
        cols = []
        for j in replay_jobs:
            r0 = f.spans[j][0]
            for i in gl:
                k, v, g, beta = f.gdn[i][:4]
                cols.append((k.data_ptr() + r0 * k.stride(0) * k.element_size(),
                             v.data_ptr() + r0 * v.stride(0) * v.element_size(),
                             g.data_ptr() + r0 * g.stride(0) * g.element_size(),
                             beta.data_ptr() + r0 * beta.stride(0) * beta.element_size(),
                             f.states[i][j], j))
        table = torch.tensor(cols, dtype=torch.int64).t().contiguous().to(dev)
        gdn_multi.replay(table, rows_d, counts_d, c.k_heads, c.v_heads, c.dv)
    for i in gl:
        conv = f.gdn[i][4].index_select(0, conv_d).split(keep)
        for j, it in enumerate(items):
            it.st.conv[i] = conv[j]
    al = sorted(f.att)
    if al:
        keys = torch.stack([f.att[i][0] for i in al]).index_select(1, take_d)
        values = torch.stack([f.att[i][1] for i in al]).index_select(1, take_d)
        dst, src = [], []
        o = 0
        for j, it in enumerate(items):
            m, need = counts[j], it.st.pos + counts[j]
            for li, i in enumerate(al):
                kbuf, vbuf = it.st.kv[i]
                if kbuf.shape[0] < need:
                    cap = max(need, 2 * kbuf.shape[0], 1024)
                    grown_k = kbuf.new_empty((cap, *kbuf.shape[1:]))
                    grown_v = vbuf.new_empty((cap, *vbuf.shape[1:]))
                    grown_k[:it.st.pos] = kbuf[:it.st.pos]
                    grown_v[:it.st.pos] = vbuf[:it.st.pos]
                    it.st.kv[i] = kbuf, vbuf = grown_k, grown_v
                dst += [kbuf[it.st.pos:need], vbuf[it.st.pos:need]]
                src += [keys[li, o:o + m], values[li, o:o + m]]
            o += m
        torch._foreach_copy_(dst, src)
    for j, it in enumerate(items):
        it.st.pos += counts[j]
    return take_d


def private_clone(st: State) -> State:
    """A clone whose attention buffers are its own: clones share KV buffers and a commit writes rows past
    ``pos`` in place, so jobs resumed from one shared prefix would overwrite each other's rows."""

    o = clone_state(st)
    o.kv = [None if kv is None else (kv[0][:st.pos].clone(), kv[1][:st.pos].clone()) for kv in st.kv]
    # own compact copies: commits update recurrent states in place, and a view would pin a whole round's block
    o.rec = [None if r is None else r.clone() for r in st.rec]
    o.conv = [None if x is None else x.clone() for x in st.conv]
    return o


def reserve_kv(st: State, rows: int) -> None:
    """Attention buffers of exactly ``rows`` rows (e.g. prompt + output): no doubling growth, no copies later."""

    for i, kv in enumerate(st.kv):
        if kv is None or kv[0].shape[0] >= rows:
            continue
        k, v = kv
        grown_k, grown_v = k.new_empty((rows, *k.shape[1:])), v.new_empty((rows, *v.shape[1:]))
        grown_k[:st.pos] = k[:st.pos]
        grown_v[:st.pos] = v[:st.pos]
        st.kv[i] = (grown_k, grown_v)


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
    slot: int = -1


@torch.no_grad()
def batch_decode(w: Weights, jobs: list[Job], draft=None, *, row_budget: int = 128, max_rows: int = 12,
                 allow_copy: bool = True, stop_eos: bool = True, share_prefix: bool = True,
                 batch_draft: bool = True, multi: bool = True, multi_attention: bool = True) -> dict:
    """Prefill every job, then decode them together: each round every active job proposes a window
    (copies from its context, else a DFlash2 tree), all windows share one forward, each job commits its path."""

    t0 = time.perf_counter()
    empty = ([None] * draft.layers, [None] * draft.layers, 0, 0) if draft is not None else None
    # Shared prefix (e.g. one system prompt): commit it once and resume every job from it. ``prefill`` with a state
    # processes only the rest, and rows never depend on their chain-mates, so this equals a fresh prefill.
    base, base_snap, shared = None, empty, 0
    if share_prefix and len(jobs) > 1:
        first = jobs[0].prompt
        shared = min(len(j.prompt) for j in jobs) - 1
        for j in jobs[1:]:
            shared = min(shared, next((i for i, (a, b) in enumerate(zip(first, j.prompt)) if a != b), shared))
        if shared >= 64:
            if draft is not None:
                draft.restore(empty)
            base = State(w)
            for start in range(0, shared, 128):
                chunk = first[start:min(start + 128, shared)]
                logits, record, *tapped = tree_forward(w, _tokens(chunk, w.norm.device),
                                                       list(range(-1, len(chunk) - 1)), base,
                                                       capture_taps=draft is not None)
                if draft is not None:
                    draft.add_taps(tapped[0])
                commit(base, record, list(range(len(chunk))))
            base_snap = draft.snapshot() if draft is not None else None
        else:
            shared = 0
    bd = None
    if draft is not None and batch_draft:
        from .batch_draft import BatchDraft
        bd = BatchDraft(draft, len(jobs))
    for n, j in enumerate(jobs):
        if draft is not None:
            draft.restore(base_snap)
        j.st, pending = prefill(w, j.prompt, j.sampling, draft, state=private_clone(base) if base is not None else None)
        j.snap = draft.snapshot() if draft is not None else None
        if bd is not None:
            j.slot = n
            bd.load(n, j.snap)
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
        proposals: dict[int, tuple[list[int], list[int]]] = {}
        need = []
        for n, j in enumerate(active):
            copied = j.copies.propose(j.context, per - 1) if j.copies is not None else []
            if copied:
                proposals[n] = copied, list(range(-1, len(copied) - 1))
            elif draft is not None and per > 1:
                need.append(n)
            else:
                proposals[n] = [], []
        if need and bd is not None:
            trees = bd.propose([active[n].slot for n in need], [active[n].out[-1] for n in need],
                               [len(active[n].context) for n in need], per - 1, [active[n].sampling for n in need])
            proposals.update(zip(need, trees))
        for n in need if bd is None else []:
            j = active[n]
            draft.restore(j.snap)
            proposals[n] = draft.propose_tree(j.out[-1], len(j.context), per - 1, j.sampling)
        items = []
        for n, j in enumerate(active):
            guesses, parents = proposals[n]
            tokens = [j.out[-1]] + guesses
            items.append(Item(tokens, [-1] + [0 if p < 0 else p + 1 for p in parents], j.st))
        torch.cuda.synchronize()
        stats["draft_s"] += time.perf_counter() - stage
        stage = time.perf_counter()
        if multi:
            f = batch_forward_multi(w, items, capture_taps=draft is not None, multi_attention=multi_attention)
            torch.cuda.synchronize()
            stats["verify_s"] += time.perf_counter() - stage
            PROF.flush()
            stage = time.perf_counter()
            greedy = [j.sampling is None or j.sampling.temperature <= 0 for j in active]
            picks = f.logits.argmax(dim=-1).cpu().tolist() if any(greedy) else None
            paths = []
            for j, it, g, (r0, r1) in zip(active, items, greedy, f.spans):
                depths, _ = _paths(it.parents)
                sampled = picks[r0:r1] if g else sample_rows(f.logits[r0:r1], [j.st.pos + d + 1 for d in depths],
                                                             j.sampling)
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
                paths.append(path)
                new = [it.tokens[row] for row in path[1:]] + [terminal]
                j.out.extend(new)
                j.context.extend(new)
                j.rounds += 1
                j.accepted += len(path) - 1
                stats["rows"] += len(it.tokens)
                if len(j.out) >= j.count or (stop_eos and j.out[-1] in eos):
                    j.done = True
            take = commit_many(w, items, f, paths)
            if draft is not None:
                taps = f.taps.index_select(0, take)
                if bd is not None:
                    bd.add_taps([j.slot for j in active], taps, [len(p) for p in paths])
                else:
                    o = 0
                    for j, p in zip(active, paths):
                        draft.restore(j.snap)
                        draft.add_taps(taps[o:o + len(p)])
                        j.snap = draft.snapshot()
                        o += len(p)
            torch.cuda.synchronize()
            stats["commit_s"] += time.perf_counter() - stage
            stats["rounds"] += 1
            continue
        results = batch_forward(w, items, capture_taps=draft is not None)
        torch.cuda.synchronize()
        stats["verify_s"] += time.perf_counter() - stage
        PROF.flush()
        stage = time.perf_counter()
        # greedy rows: one argmax and one host copy for every job (argmax is per row, so the same tokens)
        greedy = [j.sampling is None or j.sampling.temperature <= 0 for j in active]
        picks = None
        if any(greedy):
            picks = torch.cat([r[0].argmax(dim=-1) for r, g in zip(results, greedy) if g]).cpu().tolist()
        at = 0
        tap_slots, tap_rows = [], []
        for j, it, (logits, record, taps), g in zip(active, items, results, greedy):
            depths, _ = _paths(it.parents)
            if g:
                sampled = picks[at:at + len(it.tokens)]
                at += len(it.tokens)
            else:
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
            if bd is not None:
                tap_slots.append(j.slot)
                tap_rows.append(taps[path])
            elif draft is not None:
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
        if tap_slots:
            bd.add_taps(tap_slots, tap_rows)
        torch.cuda.synchronize()
        stats["commit_s"] += time.perf_counter() - stage
        stats["rounds"] += 1
    decode_s = time.perf_counter() - t1
    gen = sum(len(j.out) - 1 for j in jobs)
    return dict(prefill_s=prefill_s, shared_prefix=shared, decode_s=decode_s, generated=gen, agg_tok_s=gen / decode_s if decode_s else 0.0,
                **stats)
