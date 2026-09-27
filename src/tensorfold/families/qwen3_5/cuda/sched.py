"""SPIKE: continuous batching for the Qwen3.8 dense CUDA engine.

Requests wait in a queue; up to ``concurrency`` are live. Every round is one multi-request verify forward of at
most ``row_budget`` rows: each decoding request's draft window, plus prompt chunks of requests still prefilling
(a prompt chunk is a chain window whose rows are all committed). A prompt resumes from the longest cached
prefix; the prefix it shares with other requests is cached the first time a request's prefill reaches it.

Contract: every request's tokens equal its own serial decode. Rows never depend on their batch-mates or chunk
boundaries, and a resumed prefix equals a fresh prefill.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import torch

from tensorfold.engine.exact_sampling import Sampling

from .batch import PROF, Item, batch_forward_multi, commit_many, private_clone, reserve_kv
from .decode import CopyIndex
from .forward import State, _paths
from .sampling import sample_rows
from .weights import Weights


@dataclass
class Request:
    prompt: list[int]
    count: int
    sampling: Sampling | None = None
    # live state
    st: State | None = None
    slot: int = -1
    pos: int = 0                  # prompt tokens committed
    cache_at: int = 0             # cache the state when prefill reaches this position (0: never)
    out: list[int] = field(default_factory=list)
    context: list[int] = field(default_factory=list)
    copies: CopyIndex | None = None
    done: bool = False
    rounds: int = 0
    accepted: int = 0
    t_admit: float = 0.0
    t_first: float = 0.0
    t_done: float = 0.0


def _lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class PrefixCache:
    """Committed states (and drafter contexts) for prompt prefixes, longest match wins."""

    def __init__(self, limit: int = 16):
        self.entries: list[tuple[list[int], State, object]] = []
        self.limit = limit

    def best(self, prompt: list[int]):
        hit = None
        for e in self.entries:
            tokens = e[0]
            if len(tokens) < len(prompt) and prompt[:len(tokens)] == tokens and (hit is None or len(tokens) > len(hit[0])):
                hit = e
        if hit is not None:                      # least recently used goes first
            self.entries.remove(hit)
            self.entries.append(hit)
        return hit

    def add(self, tokens: list[int], st: State, snap) -> None:
        self.entries.append((tokens, st, snap))
        if len(self.entries) > self.limit:
            self.entries.pop(0)


@torch.no_grad()
def run(w: Weights, requests: list[Request], draft=None, *, concurrency: int = 8, row_budget: int = 128,
        max_rows: int = 12, allow_copy: bool = True, stop_eos: bool = True, min_prefix: int = 64,
        prefill_reserve: int = 32,
        cache: PrefixCache | None = None) -> dict:
    """Serve every request, ``concurrency`` at a time. Returns aggregate and per-request timings."""

    from .batch_draft import BatchDraft

    dev = w.norm.device
    bd = BatchDraft(draft, concurrency) if draft is not None else None
    cache = cache if cache is not None else PrefixCache()
    waiting = deque(requests)
    live: list[Request] = []
    eos = set(w.config.eos)
    stats = dict(rounds=0, rows=0, decode_rows=0, prefill_rows=0, verify_s=0.0, draft_s=0.0, commit_s=0.0,
                 cached_tokens=0)
    t0 = time.perf_counter()

    def admit(r: Request) -> None:
        r.t_admit = time.perf_counter() - t0
        r.slot = bd.acquire() if bd is not None else -1
        hit = cache.best(r.prompt)
        if hit is not None:
            tokens, st, snap = hit
            r.st, r.pos = private_clone(st), len(tokens)
            if bd is not None:
                bd.load(r.slot, snap)
            stats["cached_tokens"] += len(tokens)
        else:
            r.st, r.pos = State(w), 0
            if bd is not None:
                bd.load(r.slot, ([None] * draft.layers, [None] * draft.layers, 0, 0))
        reserve_kv(r.st, len(r.prompt) + r.count + 1)
        # the prefix this prompt shares with a request not yet served: cache it on the way past
        others = list(waiting)[:4] + [x for x in live if x is not r][:4]
        share = max([_lcp(r.prompt, o.prompt) for o in others] + [0])
        share = min(share, len(r.prompt) - 1)
        r.cache_at = share if share >= min_prefix and share > r.pos else 0

    while waiting or live:
        while waiting and len(live) < concurrency:
            r = waiting.popleft()
            admit(r)
            live.append(r)
        decoding = [r for r in live if r.out]
        prefilling = [r for r in live if not r.out]
        stage = time.perf_counter()
        # decode windows share the budget, less a reserve for prompt chunks while any request is prefilling
        budget = row_budget - (min(prefill_reserve, row_budget // 2) if prefilling else 0)
        per = max(1, min(max_rows, budget // max(1, len(decoding)))) if decoding else 0
        items, kinds = [], []
        # decode windows first
        need = []
        proposals = {}
        for n, r in enumerate(decoding):
            copied = r.copies.propose(r.context, per - 1) if r.copies is not None else []
            if copied:
                proposals[n] = copied, list(range(-1, len(copied) - 1))
            elif bd is not None and per > 1:
                need.append(n)
            else:
                proposals[n] = [], []
        if need:
            trees = bd.propose([decoding[n].slot for n in need], [decoding[n].out[-1] for n in need],
                               [len(decoding[n].context) for n in need], per - 1, [decoding[n].sampling for n in need])
            proposals.update(zip(need, trees))
        left = row_budget
        for n, r in enumerate(decoding):
            guesses, parents = proposals[n]
            tokens = [r.out[-1]] + guesses
            items.append(Item(tokens, [-1] + [0 if p < 0 else p + 1 for p in parents], r.st))
            kinds.append(("decode", r))
            left -= len(tokens)
        # prompt chunks with what is left, oldest first. A request resumes from a cached prefix as soon as one
        # covers more than it has committed; one that shares the prefix another is about to cache waits for it.
        leaders = []
        for r in prefilling:
            hit = cache.best(r.prompt)
            if hit is not None and len(hit[0]) > r.pos:
                tokens, st, snap = hit
                r.st, r.pos = private_clone(st), len(tokens)
                reserve_kv(r.st, len(r.prompt) + r.count + 1)
                if bd is not None:
                    bd.load(r.slot, snap)
                stats["cached_tokens"] += len(tokens)
            if any(r.pos < o.cache_at and r.prompt[:o.cache_at] == o.prompt[:o.cache_at] for o in leaders):
                continue
            if r.cache_at > r.pos:
                leaders.append(r)
            if left <= 0:
                continue
            stop = r.cache_at if r.cache_at > r.pos else len(r.prompt)
            k = min(left, stop - r.pos, 128)
            chunk = r.prompt[r.pos:r.pos + k]
            items.append(Item(chunk, list(range(-1, k - 1)), r.st))
            kinds.append(("prefill", r))
            left -= k
        torch.cuda.synchronize()
        stats["draft_s"] += time.perf_counter() - stage
        stage = time.perf_counter()
        # logits only where a next token is needed: every decode row, and a prompt's last row
        nd = sum(len(it.tokens) for (kind, _), it in zip(kinds, items) if kind == "decode")
        want, at, last_row = list(range(nd)), 0, {}
        for (kind, r), it in zip(kinds, items):
            if kind == "prefill" and r.pos + len(it.tokens) == len(r.prompt):
                last_row[id(r)] = len(want)
                want.append(at + len(it.tokens) - 1)
            at += len(it.tokens)
        f = batch_forward_multi(w, items, capture_taps=bd is not None, logit_rows=want,
                                committed=[x for x, (kind, _) in enumerate(kinds) if kind == "prefill"])
        greedy_rows = any(r.sampling is None or r.sampling.temperature <= 0 for _, r in kinds)
        picks = f.logits.argmax(dim=-1).cpu().tolist() if greedy_rows and len(want) else None
        stats["verify_s"] += time.perf_counter() - stage
        PROF.flush()
        stage = time.perf_counter()
        paths = []
        now = time.perf_counter() - t0
        for (kind, r), it, (r0, r1) in zip(kinds, items, f.spans):
            greedy = r.sampling is None or r.sampling.temperature <= 0
            if kind == "prefill":
                paths.append(list(range(len(it.tokens))))
                stats["prefill_rows"] += len(it.tokens)
                continue
            depths, _ = _paths(it.parents)
            sampled = picks[r0:r1] if greedy else sample_rows(f.logits[r0:r1], [r.st.pos + d + 1 for d in depths],
                                                               r.sampling)
            children: dict[tuple[int, int], int] = {}
            for row in range(1, len(it.tokens)):
                children.setdefault((it.parents[row], it.tokens[row]), row)
            path, terminal = [0], sampled[0]
            while len(r.out) + len(path) < r.count:
                if stop_eos and terminal in eos:
                    break
                child = children.get((path[-1], terminal))
                if child is None:
                    break
                path.append(child)
                terminal = sampled[child]
            paths.append(path)
            new = [it.tokens[row] for row in path[1:]] + [terminal]
            r.out.extend(new)
            r.context.extend(new)
            r.rounds += 1
            r.accepted += len(path) - 1
            stats["decode_rows"] += len(it.tokens)
            if len(r.out) >= r.count or (stop_eos and r.out[-1] in eos):
                r.done, r.t_done = True, now
        take = commit_many(w, items, f, paths)
        if bd is not None:
            bd.add_taps([r.slot for _, r in kinds], f.taps.index_select(0, take), [len(p) for p in paths])
        # prompt chunks: advance; cache a shared prefix; the last chunk's last row gives the first token
        for (kind, r), it, (r0, r1) in zip(kinds, items, f.spans):
            if kind != "prefill":
                continue
            r.pos += len(it.tokens)
            if r.cache_at and r.pos == r.cache_at:
                hit = cache.best(r.prompt[:r.pos + 1])
                if hit is None or len(hit[0]) < r.pos:
                    cache.add(r.prompt[:r.pos], private_clone(r.st), bd.snapshot(r.slot) if bd is not None else None)
            if r.pos == len(r.prompt):
                greedy = r.sampling is None or r.sampling.temperature <= 0
                lr = last_row[id(r)]
                first = picks[lr] if greedy else sample_rows(f.logits[lr:lr + 1], [len(r.prompt)], r.sampling)[0]
                r.out, r.context = [first], list(r.prompt) + [first]
                r.copies = CopyIndex() if allow_copy else None
                r.t_first = now
                if r.count <= 1 or (stop_eos and first in eos):
                    r.done, r.t_done = True, now
        for r in [r for r in live if r.done]:
            live.remove(r)
            if bd is not None:
                bd.release(r.slot)
            r.st = None
        torch.cuda.synchronize()
        stats["commit_s"] += time.perf_counter() - stage
        stats["rounds"] += 1
        stats["rows"] += sum(len(it.tokens) for it in items)
    wall = time.perf_counter() - t0
    gen = sum(len(r.out) - 1 for r in requests)
    return dict(wall_s=wall, generated=gen, agg_tok_s=gen / wall if wall else 0.0,
                accept_per_round=sum(r.accepted for r in requests) / max(1, sum(r.rounds for r in requests)),
                ttft_s=[round(r.t_first, 2) for r in requests], done_s=[round(r.t_done, 2) for r in requests],
                **stats)
