"""Qwen3.6-35B-A3B decoding on CUDA: prefill, serial decoding and MTP-drafted decoding that emits serial tokens.

A ``constraint`` (``tensorfold.cuda.grammar.Constraint``, a request's JSON schema) masks the logits at every token
choice, serial or drafted, drops the drafts it cannot follow, and follows the chosen tokens; without one nothing here
changes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

from tensorfold.cuda.sampling import sample_rows
from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows

from . import qmm
from .forward import Buffers, Model, commit, forward
from .mtp import MTPK, MTPBuffers, mtp_forward
from .state import Pool, State

PREFILL_ROWS = 512        # rows a prefill chunk runs at once
MTP_ROWS = 64             # rows an MTP step absorbs at once (prefill absorbs in steps of this many)
GRAPH_ROWS = 8            # windows (and MTP steps) up to this many rows replay CUDA graphs, when enabled
# the drafting recipe, chosen on an RTX 5090 (tools/bench_q36.py drafting): retune on GB10
DEPTH = 6                 # most MTP drafts a round (windows of up to DEPTH + 1 rows: window_rows, graph_rows above)
CONFIDENCE = 0.5          # a chain ends before a later draft the head gives less than this


def sample_draft(k: MTPK, logits: torch.Tensor, position: int, sampling: Sampling | None) -> tuple[int, float]:
    """The head's keyed draft at ``position`` and its probability at temperature 1 (the chain's confidence)."""

    row = logits[:1].float()
    lse = torch.logsumexp(row, dim=-1, keepdim=True)
    ids = k.ids_host
    if sampling is None or sampling.temperature <= 0:
        top, col = row.max(dim=-1, keepdim=True)            # the first maximum: argmax's choice
        got = torch.cat([top, lse, col.float()], dim=1).cpu().numpy()[0]
        c = int(got[2])
        return (int(ids[c]) if ids is not None else c), float(np.exp(float(got[0]) - float(got[1])))
    n = min(row.shape[1], int(sampling.top_k) + MARGIN) if sampling.top_k else row.shape[1]
    vals, idx = torch.topk(row, n, dim=-1, sorted=False)
    got = torch.cat([vals, lse, idx.float()], dim=1).cpu().numpy()[0]
    cols = got[n + 1:].astype(np.int64)
    toks = ids[cols] if ids is not None else cols
    tok = choose_rows(got[None, :n].astype(np.float32), toks[None, :], [position], sampling)[0]
    hit = np.nonzero(toks == tok)[0]
    return int(tok), (float(np.exp(float(got[hit[0]]) - float(got[n]))) if len(hit) else 0.0)


class Decoder:
    """A model, window buffers, a pool of ``states`` sequences (``st`` the default one), the MTP head and graphs."""

    def __init__(self, m: Model, *, capacity: int, rows: int = PREFILL_ROWS, window_rows: int = 32,
                 attn_rows: int = 64, logit_rows: int = 32, states: int = 1, mtp: MTPK | None = None,
                 mtp_rows: int = MTP_ROWS, graphs: bool = False, graph_rows: int = GRAPH_ROWS) -> None:
        self.m = m
        self.capacity = capacity
        self.buf = Buffers(m, rows, capacity=capacity, window_rows=window_rows, attn_rows=attn_rows,
                           logit_rows=logit_rows)
        self.mtp = mtp
        self.mbuf = MTPBuffers(m, mtp, mtp_rows, capacity=capacity) if mtp is not None else None
        self.pool = Pool(m, states, capacity, mtp_layers=1 if mtp is not None else 0)
        self.st = self.pool.alloc()
        self.graphs = None
        if graphs:
            from .graphs import Graphs

            self.graphs = Graphs(self, max_rows=min(graph_rows, self.buf.logit_rows, self.buf.window_rows))

    @property
    def eos(self) -> tuple[int, ...]:
        return tuple(self.m.cfg.eos)

    def forward(self, tokens: Sequence[int], st: State | None = None, *, logits: str = "all") -> torch.Tensor | None:
        """A window of ``tokens`` after ``st``'s sequence: logits [R, V], from a CUDA graph when enabled and small."""

        st = st or self.st
        if self.graphs is not None and logits == "all":
            return self.graphs.forward(st, tokens)
        return forward(self.m, self.buf, st, tokens, logits=logits)

    def commit(self, keep: int, st: State | None = None) -> None:
        commit(self.m, self.buf, st or self.st, keep)

    def mtp_step(self, st: State, tokens: Sequence[int], hidden: torch.Tensor, pos0: int, *,
                 logits: bool = True) -> torch.Tensor | None:
        """An MTP step (``mtp.mtp_forward``), from a CUDA graph when enabled and small."""

        if self.mtp is None:
            raise ValueError("this decoder has no MTP head")
        if self.graphs is not None:
            return self.graphs.mtp(st, tokens, hidden, pos0, logits=logits)
        return mtp_forward(self.m, self.mtp, self.mbuf, st, tokens, hidden, pos0, logits=logits)

    def warm(self, rows: int | None = None, states: Sequence[State] | None = None) -> int:
        """Capture every decode graph of ``states`` (default ``st``) up front and reset them; returns the count."""

        if self.graphs is None:
            return 0
        got = 0
        for st in states or [self.st]:
            got += self.graphs.warm(st, rows)
            st.reset()
        return got


# -- the MTP head's bookkeeping ----------------------------------------------------------------------------------------
def _zero_mtp(st: State, lo: int, hi: int) -> None:
    """Cache rows the head could not absorb (their hidden rows are gone): zero keys and values, a fixed stand-in."""

    if hi > lo:
        st.mtp_kc[0, lo:hi].zero_()
        st.mtp_vc[0, lo:hi].zero_()


def _absorb(e: Decoder, st: State, hidden: torch.Tensor, next_tokens: Sequence[int], *,
            logits: bool) -> torch.Tensor | None:
    """The head absorbs positions from st.mtp_len on in steps; with ``logits``, the last row's draft logits."""

    n = len(next_tokens)
    out = None
    for i in range(0, n, e.mbuf.rows):
        j = min(n, i + e.mbuf.rows)
        out = e.mtp_step(st, list(next_tokens[i:j]), hidden[i:j], st.mtp_len, logits=logits and j == n)
        st.mtp_len += j - i
    return out


def _keep_tail(st: State, hidden_row: torch.Tensor) -> None:
    """The hidden row of the last committed position waits for its next token (the next reply or prompt token)."""

    st.mtp_tail.copy_(hidden_row)
    st.mtp_tail_at = st.pos - 1


def _tail_ready(st: State) -> bool:
    """Whether the head holds every position but the last, whose row waits in ``mtp_tail`` (gaps zero-filled)."""

    p = st.pos
    if p == 0 or st.mtp_tail is None or st.mtp_tail_at != p - 1:
        return False
    _zero_mtp(st, st.mtp_len, p - 1)
    st.mtp_len = p - 1
    return True


def _mtp_catch_up(e: Decoder, st: State, token: int | None) -> None:
    """Before rows at st.pos: the head absorbs the waiting tail with ``token``, or zero-fills what it cannot."""

    if token is not None and _tail_ready(st):
        _absorb(e, st, st.mtp_tail[None], [token], logits=False)
    else:
        _zero_mtp(st, st.mtp_len, st.pos)
        st.mtp_len = st.pos


def draft(e: Decoder, st: State, hidden: torch.Tensor, next_tokens: Sequence[int], position: int, count: int,
          sampling: Sampling | None, confidence: float = 0.0) -> list[int]:
    """Absorb the kept rows, then chain up to ``count`` drafts from ``position``, stopping on low ``confidence``."""

    k = e.mtp
    logits = _absorb(e, st, hidden, next_tokens, logits=True)
    drafts: list[int] = []
    for j in range(count):
        d, p = sample_draft(k, logits, position + j, sampling)
        low = confidence > 0 and p < confidence
        if low and j > 0:
            break
        drafts.append(d)
        if low:
            break
        if j + 1 < count:
            logits = e.mtp_step(st, [d], e.mbuf.out, st.mtp_len + j)
    return drafts


# -- prefill ---------------------------------------------------------------------------------------------------------
@torch.no_grad()
def run_prompt(e: Decoder, prompt: Sequence[int], *, st: State | None = None, chunk: int | None = None,
               resume: dict | None = None, mtp: bool = True) -> torch.Tensor:
    """Commit ``prompt`` in chunks (after ``resume``, a snapshot it extends); returns the last row's logits [1, V]."""

    if not prompt:
        raise ValueError("a prompt needs at least one token")
    st = st or e.st
    chunk = min(chunk or e.buf.rows, e.buf.rows)
    if resume is None:
        st.reset()
    else:
        st.restore(resume)
        if not 0 < st.pos < len(prompt):
            raise ValueError("a resumed prompt must extend the snapshot's tokens")
    use_mtp = mtp and e.mtp is not None
    if use_mtp:
        _mtp_catch_up(e, st, prompt[st.pos] if st.pos else None)
    logits = None
    for start in range(st.pos, len(prompt), chunk):
        part = list(prompt[start:start + chunk])
        final = start + len(part) >= len(prompt)
        logits = forward(e.m, e.buf, st, part, logits="last" if final else "none")
        commit(e.m, e.buf, st, len(part))
        if use_mtp:
            nxt = list(prompt[start + 1:start + len(part) + 1])
            if nxt:
                _absorb(e, st, e.buf.hidden[:len(nxt)], nxt, logits=False)
            if final:
                _keep_tail(st, e.buf.hidden[len(part) - 1])
    return logits[:1].clone()


@torch.no_grad()
def prefill(e: Decoder, prompt: Sequence[int], sampling: Sampling | None = None, *, st: State | None = None,
            chunk: int | None = None, resume: dict | None = None, mtp: bool = True, constraint=None) -> int:
    """Commit the prompt and sample the first reply token (at position len(prompt)), which ``constraint`` follows."""

    logits = run_prompt(e, prompt, st=st, chunk=chunk, resume=resume, mtp=mtp)
    if constraint is None:
        return sample_rows(logits, [len(prompt)], sampling)[0]
    first = sample_rows(constraint.mask(logits), [len(prompt)], sampling)[0]
    constraint.advance([first])
    return first


# -- decoding ----------------------------------------------------------------------------------------------------------
@dataclass
class DecodeResult:
    tokens: list[int]                                   # the pending token, then each round's
    seconds: float
    rounds: int
    widths: list[int] = field(default_factory=list)     # rows each round verified (1: serial)
    drafted: int = 0                                    # drafts verified
    accepted: int = 0                                   # drafts kept
    keeps: list[int] = field(default_factory=list)      # tokens each round kept
    committed: list[int] = field(default_factory=list)  # the tokens now in the caches (all but the pending one)

    @property
    def tokens_per_second(self) -> float:
        return (len(self.tokens) - 1) / self.seconds if self.seconds else 0.0

    @property
    def acceptance(self) -> float:
        """Drafts kept over drafts verified."""

        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_round(self) -> float:
        return sum(self.keeps) / len(self.keeps) if self.keeps else (1.0 if self.rounds else 0.0)


@torch.no_grad()
def serial_decode(e: Decoder, pending: int, count: int, sampling: Sampling | None = None, *,
                  st: State | None = None, stop_eos: bool = False,
                  on_tokens: Callable[[list[int]], bool] | None = None, constraint=None) -> DecodeResult:
    """The serial reference: up to ``count`` tokens from ``pending``, one row a step; the last is not committed.

    ``constraint`` has followed ``pending``; it masks each step, and its reply ends at its stop token."""

    st = st or e.st
    stop_eos = stop_eos or constraint is not None
    out = [pending]
    pos0 = st.pos
    torch.cuda.synchronize()
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in e.eos):
        logits = e.forward([out[-1]], st)
        if constraint is not None:
            logits = constraint.mask(logits[:1])
        tok = sample_rows(logits[:1], [st.pos + 1], sampling)[0]
        if constraint is not None:
            constraint.advance([tok])
        commit(e.m, e.buf, st, 1)
        out.append(tok)
        if on_tokens is not None and on_tokens([tok]):
            break
    torch.cuda.synchronize()
    n = len(out) - 1
    return DecodeResult(out, time.perf_counter() - start, n, [1] * n, keeps=[1] * n, committed=out[:st.pos - pos0])


@torch.no_grad()
def mtp_decode(e: Decoder, pending: int, count: int, sampling: Sampling | None = None, *, st: State | None = None,
               depth: int = DEPTH, confidence: float = CONFIDENCE, stop_eos: bool = False,
               on_tokens: Callable[[list[int]], bool] | None = None, constraint=None) -> DecodeResult:
    """Up to ``count`` tokens from ``pending``, ``serial_decode``'s tokens, each round verifying MTP drafts.

    ``constraint`` has followed ``pending``: it drops the drafts it cannot follow, masks every row of a window as
    the serial step at that path, and follows the kept tokens."""

    if e.mtp is None:
        raise ValueError("drafted decoding needs the MTP head (Decoder(..., mtp=prepare_mtp(...)))")
    st = st or e.st
    stop_eos = stop_eos or constraint is not None
    m, b = e.m, e.buf
    out = [pending]
    rounds = drafted = accepted = 0
    keeps: list[int] = []
    widths: list[int] = []
    pos0 = st.pos
    kept: list[int] = []                           # the last round's kept tokens (the rows after their positions)
    torch.cuda.synchronize()
    start = time.perf_counter()
    drafts: list[int] = []
    if len(out) < count and not (stop_eos and pending in e.eos) and _tail_ready(st):
        drafts = draft(e, st, st.mtp_tail[None], [pending], st.pos + 1, min(depth, count - 2), sampling, confidence)
    while len(out) < count and not (stop_eos and out[-1] in e.eos):
        if constraint is not None:
            drafts = constraint.admissible(drafts)
        tokens = [out[-1]] + drafts
        R = len(tokens)
        logits = e.forward(tokens, st)
        if constraint is not None:
            logits = constraint.mask(logits[:R], drafts)
        sampled = sample_rows(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in e.eos):
                break
            keep += 1
        commit(m, b, st, keep)
        kept = sampled[:keep]
        if constraint is not None:
            constraint.advance(kept)
        rounds += 1
        drafted += len(drafts)
        accepted += keep - 1
        keeps.append(keep)
        widths.append(R)
        new = kept[:max(0, count - len(out))]
        out.extend(kept)
        if on_tokens is not None and new and on_tokens(new):
            break
        if len(out) >= count or (stop_eos and out[-1] in e.eos):
            break
        if st.mtp_len < st.pos - keep:             # a round without drafts: the head had no tail to start from
            _zero_mtp(st, st.mtp_len, st.pos - keep)
            st.mtp_len = st.pos - keep
        drafts = draft(e, st, b.hidden[:keep], kept, st.pos + 1, min(depth, count - len(out) - 1), sampling,
                       confidence)
        kept = []
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    if kept:                          # the last round's rows: all but the last into the head, the last waits
        keep = len(kept)
        if st.mtp_len < st.pos - keep:
            _zero_mtp(st, st.mtp_len, st.pos - keep)
            st.mtp_len = st.pos - keep
        if keep > 1:
            _absorb(e, st, b.hidden[:keep - 1], kept[:keep - 1], logits=False)
        _keep_tail(st, b.hidden[keep - 1])
    return DecodeResult(out[:count], seconds, rounds, widths, drafted, accepted, keeps, out[:st.pos - pos0])


@torch.no_grad()
def generate_result(e: Decoder, prompt: Sequence[int], max_tokens: int, sampling: Sampling | None = None, *,
                    stop_eos: bool = True, st: State | None = None, draft: bool = True,
                    on_tokens: Callable[[list[int]], bool] | None = None, depth: int = DEPTH,
                    confidence: float = CONFIDENCE, constraint=None) -> DecodeResult:
    """``generate`` with the decode loop's counts (rounds, drafts, acceptance, seconds after the prefill)."""

    if max_tokens < 1:
        return DecodeResult([], 0.0, 0)
    stop_eos = stop_eos or constraint is not None
    drafting = draft and e.mtp is not None and depth > 0
    first = prefill(e, prompt, sampling, st=st, mtp=drafting, constraint=constraint)
    if (on_tokens is not None and on_tokens([first])) or (stop_eos and first in e.eos) or max_tokens == 1:
        return DecodeResult([first], 0.0, 0)
    if drafting:
        return mtp_decode(e, first, max_tokens, sampling, st=st, depth=depth, confidence=confidence,
                          stop_eos=stop_eos, on_tokens=on_tokens, constraint=constraint)
    return serial_decode(e, first, max_tokens, sampling, st=st, stop_eos=stop_eos, on_tokens=on_tokens,
                         constraint=constraint)


@torch.no_grad()
def generate(e: Decoder, prompt: Sequence[int], max_tokens: int, sampling: Sampling | None = None, *,
             stop_eos: bool = True, st: State | None = None, draft: bool = True,
             on_tokens: Callable[[list[int]], bool] | None = None, depth: int = DEPTH,
             confidence: float = CONFIDENCE, constraint=None) -> list[int]:
    """Prefill, then up to ``max_tokens`` reply tokens, drafted with the head or (``draft=False``) serially."""

    return generate_result(e, prompt, max_tokens, sampling, stop_eos=stop_eos, st=st, draft=draft,
                           on_tokens=on_tokens, depth=depth, confidence=confidence, constraint=constraint).tokens


# -- teacher-forced scoring ------------------------------------------------------------------------------------------
@torch.no_grad()
def score(e: Decoder, ids: Sequence[int], *, st: State | None = None, chunk: int | None = None
          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher forcing over ``ids``: (argmax at each position, NLL of each next token) from the bf16 logits."""

    st = st or e.st
    b = e.buf
    chunk = min(chunk or b.rows, b.rows)
    st.reset()
    target = torch.tensor(list(ids[1:]) + [0], dtype=torch.int64, device=e.m.device)
    args, nlls = [], []
    for start in range(0, len(ids), chunk):
        part = list(ids[start:start + chunk])
        forward(e.m, b, st, part, logits="none")
        for r in range(0, len(part), b.logit_rows):
            n = min(b.logit_rows, len(part) - r)
            lg = qmm.matmul(b.hidden[r:r + n], e.m.head, b.xs[r:r + n], out=b.logits[:n], part=b.part).float()
            args.append(lg.argmax(dim=-1))
            t = target[start + r:start + r + n]
            nlls.append(torch.logsumexp(lg, dim=-1) - lg.gather(1, t[:, None])[:, 0])
        commit(e.m, b, st, len(part))
    return torch.cat(args), torch.cat(nlls)[:-1]
