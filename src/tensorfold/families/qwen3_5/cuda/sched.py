"""SPIKE: continuous batching for the Qwen3.8 dense CUDA engine.

Requests wait in a queue; up to ``concurrency`` are live. Every round is one multi-request verify forward of at
most ``row_budget`` rows: each decoding request's draft window, plus prompt chunks of requests still prefilling
(a prompt chunk is a chain window whose rows are all committed). A prompt resumes from the longest cached
prefix; the prefix it shares with other requests is cached the first time a request's prefill reaches it.

Contract: every request's tokens equal its own serial decode. Rows never depend on their batch-mates or chunk
boundaries, and a resumed prefix equals a fresh prefill.

``Scheduler`` runs online (``submit`` from any thread, ``step`` from one): the server's concurrent engine drives it
from a background thread. ``run`` serves a fixed list of requests (benchmarks).
"""

from __future__ import annotations

import threading
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
    t_submit: float = 0.0
    t_admit: float = 0.0
    t_first: float = 0.0
    t_done: float = 0.0
    serial: bool = False          # one token a round, no drafts and no copies (the serial reference)
    cancel: bool = False          # set from another thread: drop the request at the next round
    emit: object = None           # emit(new_ids) after each round; emit(None) when the request is finished
    constraint: object = None     # grammar: masks the rows' logits and follows the accepted tokens
    error: str = ""
    cached: int = 0               # prompt tokens resumed from a cached state
    checkpoint: int = 0           # end of the prompt's leading system-and-tools block (cache it for the next call)
    cache_points: list = field(default_factory=list)


def _lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _nbytes(x) -> int:
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, (list, tuple)):
        return sum(_nbytes(y) for y in x)
    if isinstance(x, State):
        return _nbytes(x.kv) + _nbytes(x.rec) + _nbytes(x.conv)
    return 0


class PrefixCache:
    """Committed states (and drafter contexts) for prompt prefixes, longest match wins. Holds at most ``limit``
    entries and ``budget`` bytes (None: no byte limit), dropping the least recently used first."""

    def __init__(self, limit: int = 16, budget: int | None = None):
        self.entries: list[tuple[list[int], State, object]] = []
        self.limit, self.budget = limit, budget
        self.sizes: dict[int, int] = {}

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

    def bytes(self) -> int:
        return sum(self.sizes.values())

    def add(self, tokens: list[int], st: State, snap) -> None:
        size = _nbytes(st) + _nbytes(snap)
        if self.budget is not None and size > self.budget:
            return                                # larger than the whole cache: keep what is there
        e = (tokens, st, snap)
        self.entries.append(e)
        self.sizes[id(e)] = size
        while len(self.entries) > self.limit or (self.budget is not None and self.bytes() > self.budget):
            self.sizes.pop(id(self.entries.pop(0)), None)


class Scheduler:
    """Continuous batching: ``submit`` requests, call ``step`` until ``idle``. One thread calls ``step``."""

    def __init__(self, w: Weights, draft=None, *, concurrency: int = 16, row_budget: int = 128, max_rows: int = 6,
                 allow_copy: bool = True, stop_eos: bool = True, min_prefix: int = 64, prefill_reserve: int = 32,
                 cache: PrefixCache | None = None, kv_budget_gib: float | None = None, turn_entries: int = 4,
                 turn_min: int = 8192, turn_budget: int | None = None, block_budget: int | None = None, log=None):
        from .batch_draft import BatchDraft

        self.log = log

        self.w, self.draft = w, draft
        self.bd = BatchDraft(draft, concurrency) if draft is not None else None
        self.cache = cache if cache is not None else PrefixCache()
        self.concurrency, self.row_budget, self.max_rows = concurrency, row_budget, max_rows
        self.allow_copy, self.stop_eos, self.min_prefix, self.prefill_reserve = allow_copy, stop_eos, min_prefix, prefill_reserve
        # states after a long prompt and after its reply: an agent loop sends the growing conversation again
        # each call, and resumes from these. Kept apart so they cannot evict the shared-prefix entries.
        self.turns = PrefixCache(turn_entries, turn_budget)
        # the system-and-tools blocks the server marks (``checkpoint``): every new agent conversation starts with
        # one, and bursts of other requests or the agent's own turns must not push it out
        self.blocks = PrefixCache(2, block_budget)
        self.turn_min = turn_min
        self.last_step = time.perf_counter()
        self.failed_steps = 0
        self.waiting: deque[Request] = deque()
        self.live: list[Request] = []
        self.recent: deque[list[int]] = deque(maxlen=8)      # recent prompts: a shared prefix with them is cached too
        # long prompts (agent calls) are rarer than bursts of short ones: keep their own history, so a new agent call
        # still finds the system-and-tools prefix it shares with the last one
        self.recent_long: deque[list[int]] = deque(maxlen=8)
        self.lock = threading.Lock()
        self.eos = set(w.config.eos)
        c = w.config
        # admission budget: attention rows (every full-attention layer's K and V) and recurrent state per request
        n_att = sum(1 for layer in w.layers if not layer.linear)
        n_gdn = len(w.layers) - n_att
        self.kv_row_bytes = n_att * 2 * c.kv_heads * c.head_dim * 2
        self.state_bytes = n_gdn * (c.v_heads * c.dv * c.dk * 4 + (c.conv_kernel - 1) * (2 * c.k_heads * c.dk + c.v_heads * c.dv) * 2)
        self.kv_budget = None if kv_budget_gib is None else kv_budget_gib * 2**30
        self.stats = dict(rounds=0, rows=0, decode_rows=0, prefill_rows=0, verify_s=0.0, draft_s=0.0, commit_s=0.0,
                          cached_tokens=0)
        self.t0 = time.perf_counter()

    # ---- queue ----
    def submit(self, r: Request) -> None:
        r.t_submit = time.perf_counter() - self.t0
        with self.lock:
            self.waiting.append(r)

    def idle(self) -> bool:
        with self.lock:
            return not self.waiting and not self.live

    def _need(self, r: Request) -> int:
        return (len(r.prompt) + r.count + 1) * self.kv_row_bytes + self.state_bytes

    def _finish(self, r: Request) -> None:
        r.done = True
        r.t_done = time.perf_counter() - self.t0
        if r in self.live:
            self.live.remove(r)
            if self.bd is not None and r.slot >= 0:
                self.bd.release(r.slot)
        r.st = None
        if self.log is not None:
            first = f"{r.t_first - r.t_admit:.1f}s" if r.t_first else "-"
            end = ", cancelled" if r.cancel else (f", error: {r.error[:120]}" if r.error else "")
            self.log(f"done: prompt {len(r.prompt)} tokens (resumed {r.cached}), {len(r.out)} out, queued "
                     f"{max(0.0, r.t_admit - r.t_submit):.1f}s, first token {first}, total {r.t_done - r.t_submit:.1f}s{end}")
        if r.emit is not None:
            r.emit(None)

    def fail_all(self, message: str) -> None:
        """An error inside a round: every live and waiting request ends with it."""

        with self.lock:
            dead = list(self.live) + list(self.waiting)
            self.waiting.clear()
        for r in dead:
            r.error = message
            self._finish(r)

    def _admit(self, r: Request) -> None:
        w, bd = self.w, self.bd
        r.t_admit = time.perf_counter() - self.t0
        r.slot = bd.acquire() if bd is not None else -1
        hit = self._best(r.prompt)
        if hit is not None:
            tokens, st, snap = hit
            r.st, r.pos = private_clone(st), len(tokens)
            if bd is not None:
                bd.load(r.slot, snap)
            self.stats["cached_tokens"] += len(tokens)
            r.cached = len(tokens)
        else:
            r.st, r.pos = State(w), 0
            if bd is not None:
                bd.load(r.slot, ([None] * self.draft.layers, [None] * self.draft.layers, 0, 0))
        reserve_kv(r.st, len(r.prompt) + r.count + 1)
        # the prefix this prompt shares with a request queued, running or just served: cache it on the way past
        others = ([o.prompt for o in list(self.waiting)[:4]] + [x.prompt for x in self.live if x is not r][:4]
                  + list(self.recent) + list(self.recent_long))
        share = max([_lcp(r.prompt, o) for o in others] + [0])
        share = min(share, len(r.prompt) - 1)
        points = [p for p in (share, r.checkpoint) if self.min_prefix <= p < len(r.prompt) and p > r.pos]
        r.cache_points = sorted(set(points))
        r.cache_at = r.cache_points[0] if r.cache_points else 0
        self.recent.append(r.prompt)
        if len(r.prompt) >= self.turn_min:
            self.recent_long.append(r.prompt)
        if self.log is not None:
            self.log(f"admit: prompt {len(r.prompt)} tokens, resumed {r.pos}, will cache at {r.cache_points or '-'}, "
                     f"live {len(self.live) + 1}, waiting {len(self.waiting)}")

    def _best(self, prompt: list[int]):
        hits = [h for h in (self.cache.best(prompt), self.turns.best(prompt), self.blocks.best(prompt)) if h is not None]
        return max(hits, key=lambda h: len(h[0])) if hits else None

    @torch.no_grad()
    def step(self) -> bool:
        """One round. False when there was nothing to do."""

        w, bd, cache, stats, eos, stop_eos = self.w, self.bd, self.cache, self.stats, self.eos, self.stop_eos
        with self.lock:
            for r in [r for r in self.waiting if r.cancel]:
                self.waiting.remove(r)
                self._finish(r)
            for r in [r for r in self.live if r.cancel]:
                self._finish(r)
            used = sum(self._need(r) for r in self.live)
            while self.waiting and len(self.live) < self.concurrency:
                r = self.waiting[0]
                if self.kv_budget is not None and self.live and used + self._need(r) > self.kv_budget:
                    break
                self.waiting.popleft()
                self._admit(r)
                self.live.append(r)
                used += self._need(r)
        if not self.live:
            return False
        decoding = [r for r in self.live if r.out]
        prefilling = [r for r in self.live if not r.out]
        stage = time.perf_counter()
        # decode windows share the budget, less a reserve for prompt chunks while any request is prefilling
        budget = self.row_budget - (min(self.prefill_reserve, self.row_budget // 2) if prefilling else 0)
        per = max(1, min(self.max_rows, budget // max(1, len(decoding)))) if decoding else 0
        items, kinds = [], []
        need = []
        proposals = {}
        for n, r in enumerate(decoding):
            width = 1 if r.serial else per
            copied = r.copies.propose(r.context, width - 1) if r.copies is not None and width > 1 else []
            if copied:
                proposals[n] = copied, list(range(-1, len(copied) - 1))
            elif bd is not None and width > 1:
                need.append(n)
            else:
                proposals[n] = [], []
        if need:
            trees = bd.propose([decoding[n].slot for n in need], [decoding[n].out[-1] for n in need],
                               [len(decoding[n].context) for n in need], per - 1, [decoding[n].sampling for n in need])
            proposals.update(zip(need, trees))
        left = self.row_budget
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
            hit = self._best(r.prompt)
            if hit is not None and len(hit[0]) > r.pos:
                tokens, st, snap = hit
                r.st, r.pos, r.cached = private_clone(st), len(tokens), len(tokens)
                reserve_kv(r.st, len(r.prompt) + r.count + 1)
                later = [p for p in r.cache_points if p > r.pos]
                r.cache_at = later[0] if later else 0
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
        if not items:
            return True
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
        # grammar: each constrained row may only produce tokens its request's grammar allows after that row's path
        for (kind, r), it, (r0, r1) in zip(kinds, items, f.spans):
            if r.constraint is None:
                continue
            try:
                if kind == "decode":
                    r.constraint.mask_tree(it.tokens, it.parents, f.logits[r0:r1])
                elif id(r) in last_row:
                    lr = last_row[id(r)]
                    r.constraint.mask_first(f.logits[lr:lr + 1])
            except Exception as exc:              # this request's grammar failed: end it alone
                r.error, r.cancel, r.constraint = f"grammar: {type(exc).__name__}: {exc}", True, None
        greedy_rows = any(r.sampling is None or r.sampling.temperature <= 0 for _, r in kinds)
        picks = f.logits.argmax(dim=-1).cpu().tolist() if greedy_rows and len(want) else None
        stats["verify_s"] += time.perf_counter() - stage
        PROF.flush()
        stage = time.perf_counter()
        paths, emits = [], []
        now = time.perf_counter() - self.t0
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
            if r.constraint is not None:
                try:
                    r.constraint.advance(new)
                except Exception as exc:
                    r.error, r.cancel, r.constraint = f"grammar: {type(exc).__name__}: {exc}", True, None
            r.out.extend(new)
            r.context.extend(new)
            r.rounds += 1
            r.accepted += len(path) - 1
            stats["decode_rows"] += len(it.tokens)
            emits.append((r, new))
            if len(r.out) >= r.count or (stop_eos and r.out[-1] in eos) or \
                    (r.constraint is not None and r.constraint.finished()):
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
                hit = self._best(r.prompt[:r.pos + 1])
                if hit is None or len(hit[0]) < r.pos:
                    (self.blocks if r.pos == r.checkpoint else cache).add(
                        r.prompt[:r.pos], private_clone(r.st), bd.snapshot(r.slot) if bd is not None else None)
                later = [p for p in r.cache_points if p > r.pos]
                r.cache_at = later[0] if later else 0
            if r.pos == len(r.prompt) and len(r.prompt) >= self.turn_min:
                self.turns.add(list(r.prompt), private_clone(r.st), bd.snapshot(r.slot) if bd is not None else None)
            if r.pos == len(r.prompt):
                greedy = r.sampling is None or r.sampling.temperature <= 0
                lr = last_row[id(r)]
                first = picks[lr] if greedy else sample_rows(f.logits[lr:lr + 1], [len(r.prompt)], r.sampling)[0]
                if r.constraint is not None:
                    try:
                        r.constraint.advance([first])
                    except Exception as exc:
                        r.error, r.cancel, r.constraint = f"grammar: {type(exc).__name__}: {exc}", True, None
                r.out, r.context = [first], list(r.prompt) + [first]
                r.copies = CopyIndex() if self.allow_copy and not r.serial else None
                r.t_first = now
                emits.append((r, [first]))
                if r.count <= 1 or (stop_eos and first in eos) or \
                        (r.constraint is not None and r.constraint.finished()):
                    r.done, r.t_done = True, now
        torch.cuda.synchronize()
        for r, new in emits:
            if r.emit is not None:
                r.emit(new)
        with self.lock:
            for r in [r for r in self.live if r.done]:
                if not r.cancel and not r.error and r.st is not None and len(r.prompt) >= self.turn_min and len(r.out) > 1:
                    committed = list(r.prompt) + r.out[:-1]          # the last token is not in the state yet
                    if r.st.pos == len(committed):
                        self.turns.add(committed, private_clone(r.st), bd.snapshot(r.slot) if bd is not None else None)
                self._finish(r)
        stats["commit_s"] += time.perf_counter() - stage
        stats["rounds"] += 1
        stats["rows"] += sum(len(it.tokens) for it in items)
        self.last_step = time.perf_counter()
        return True

    def health(self) -> dict:
        with self.lock:
            live, waiting = len(self.live), len(self.waiting)
        return {"live": live, "waiting": waiting, "last_step_age_s": round(time.perf_counter() - self.last_step, 1),
                "failed_steps": self.failed_steps, "rounds": self.stats["rounds"],
                "shared_prefixes": len(self.cache.entries), "turn_states": len(self.turns.entries),
                "system_blocks": len(self.blocks.entries),
                "cache_gib": round((self.cache.bytes() + self.turns.bytes() + self.blocks.bytes()) / 2**30, 2)}


@torch.no_grad()
def run(w: Weights, requests: list[Request], draft=None, *, concurrency: int = 8, row_budget: int = 128,
        max_rows: int = 12, allow_copy: bool = True, stop_eos: bool = True, min_prefix: int = 64,
        prefill_reserve: int = 32, cache: PrefixCache | None = None) -> dict:
    """Serve every request, ``concurrency`` at a time. Returns aggregate and per-request timings."""

    sched = Scheduler(w, draft, concurrency=concurrency, row_budget=row_budget, max_rows=max_rows,
                      allow_copy=allow_copy, stop_eos=stop_eos, min_prefix=min_prefix,
                      prefill_reserve=prefill_reserve, cache=cache)
    for r in requests:
        sched.submit(r)
    while not sched.idle():
        sched.step()
    wall = time.perf_counter() - sched.t0
    gen = sum(len(r.out) - 1 for r in requests)
    return dict(wall_s=wall, generated=gen, agg_tok_s=gen / wall if wall else 0.0,
                accept_per_round=sum(r.accepted for r in requests) / max(1, sum(r.rounds for r in requests)),
                ttft_s=[round(r.t_first, 2) for r in requests], done_s=[round(r.t_done, 2) for r in requests],
                **sched.stats)
