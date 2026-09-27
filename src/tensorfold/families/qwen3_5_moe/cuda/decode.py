"""Qwen3.6-35B-A3B decode on CUDA: prefill, serial decoding (the reference), and MTP-drafted decoding that emits
serial decoding's tokens.

Every emitted token is the keyed sample (``tensorfold.engine.exact_sampling``: seeded Gumbel over top-k/top-p, ties by
token id) of this engine's logits at its position, or the argmax (the lowest id among equal logits) when greedy. A
serial step is a one-row window of ``forward`` followed by a one-row ``commit``.

A drafted round verifies the pending token and up to ``depth`` MTP drafts as one window through the same kernels and
the same sampler (row r of a window has the bits of the serial step at its position), keeps the rows up to the first
draft that differs from what the sampler picks there (``forward.commit``), then the MTP head absorbs the kept rows and
chains the next drafts. Drafts are sampled with the same keyed rule at their positions, so a sampled draft shares its
position's noise with the verify. With ``confidence`` > 0 a chain always keeps its first draft (every round verifies
at least two rows) and ends before a later draft the head gives less than ``confidence``, or right after a first draft
under it. Drafts change speed only, never the output.

A prompt runs in chunks of up to ``Decoder.buf.rows`` rows (512 by default); with the MTP head the head absorbs each
position whose next token is known, and keeps the last position's hidden row (``State.mtp_tail``) until the first
reply token. Rows never depend on their chunk, so any chunking, and a resumed prompt (``State.snapshot``), ends in the
state and logits of one fresh prefill.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows

from . import qmm
from .forward import Buffers, Model, Pool, State, commit, forward
from .mtp import MTPK, MTPBuffers, mtp_forward

CONTEXT = 4096            # prompt plus reply tokens the caches hold, by default
PREFILL_ROWS = 512        # rows a prefill chunk runs at once
MTP_ROWS = 64             # rows an MTP step absorbs at once (prefill absorbs in steps of this many)
GRAPH_ROWS = 8            # windows (and MTP steps) up to this many rows replay CUDA graphs, when enabled
# the drafting recipe, measured on an RTX 5090 (tests/cuda/test_qwen36moe_mtp.py; retune on GB10, where a verify row
# costs more): chat chains end by confidence well before the depth, JSON chains run deep (acceptance near 1)
DEPTH = 6                 # most MTP drafts a round (windows of up to DEPTH + 1 rows: keep window_rows and graph_rows above)
CONFIDENCE = 0.5          # a chain ends before a later draft the head gives less than this


def sample_rows(logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None) -> list[int]:
    """Each row from its own logits and absolute position: the argmax (greedy), or CUDA picks the top-k plus a margin
    and the host's keyed rule draws, ties by token id. A serial row and the same row in a verify window go through
    this one function."""

    if logits.ndim != 2 or len(positions) != logits.shape[0]:
        raise ValueError("expected logits [rows, vocab] and one position per row")
    if sampling is None or sampling.temperature <= 0:
        return [int(x) for x in logits.argmax(dim=-1).cpu().tolist()]
    width = logits.shape[1]
    count = min(width, int(sampling.top_k) + MARGIN) if sampling.top_k else width
    if count < width:
        values, ids = torch.topk(logits.float(), count, dim=-1, sorted=False)
        values_np = values.cpu().numpy()
        ids_np = ids.cpu().numpy().astype(np.int64, copy=False)
    else:
        values_np = logits.float().cpu().numpy()
        ids_np = np.broadcast_to(np.arange(width, dtype=np.int64), values_np.shape)
    return choose_rows(values_np, ids_np, positions, sampling)


def sample_draft(k: MTPK, logits: torch.Tensor, position: int, sampling: Sampling | None) -> tuple[int, float]:
    """The MTP head's draft at ``position`` (keyed like every sample, over the draft head's tokens) and its probability
    at temperature 1 under the head's distribution over those tokens (the confidence that ends a chain; speed only).
    One device-to-host copy."""

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
    """A model, one set of window buffers, and a pool of sequence states; ``st`` is the sequence the helpers below
    run unless given another. ``states``: how many sequences the pool holds (tests and A/B runs take clones).

    ``mtp``: the MTP head (``mtp.prepare_mtp``); the pool then holds its cache and the drafted loop can run.
    ``graphs``: windows and MTP steps of up to ``graph_rows`` rows replay CUDA graphs (``graphs.py``), the same bits
    as eager; ``warm`` captures them up front."""

    def __init__(self, m: Model, *, capacity: int = CONTEXT, rows: int = PREFILL_ROWS, window_rows: int = 32,
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
        """A window of ``tokens`` after ``st``'s committed sequence (logits [R, V], a view of the buffers); a CUDA graph
        replay when enabled and the window is small."""

        st = st or self.st
        if self.graphs is not None and logits == "all":
            return self.graphs.forward(st, tokens)
        return forward(self.m, self.buf, st, tokens, logits=logits)

    def commit(self, keep: int, st: State | None = None) -> None:
        commit(self.m, self.buf, st or self.st, keep)

    def mtp_step(self, st: State, tokens: Sequence[int], hidden: torch.Tensor, pos0: int, *,
                 logits: bool = True) -> torch.Tensor | None:
        """An MTP step (``mtp.mtp_forward``), a CUDA graph replay when enabled and the step is small."""

        if self.mtp is None:
            raise ValueError("this decoder has no MTP head")
        if self.graphs is not None:
            return self.graphs.mtp(st, tokens, hidden, pos0, logits=logits)
        return mtp_forward(self.m, self.mtp, self.mbuf, st, tokens, hidden, pos0, logits=logits)

    def warm(self, rows: int | None = None, states: Sequence[State] | None = None) -> int:
        """Capture every decode graph for ``states`` (default: ``st``) up front, so no capture lands inside a timed
        run: windows of 1..rows rows at both GDN parities, MTP steps of 1..rows rows. Resets those states (it runs
        windows on them). Returns the number of graphs captured."""

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
    """The head absorbs positions st.mtp_len, st.mtp_len + 1, ... (their hidden rows and next tokens) in steps of up
    to ``e.mbuf.rows`` rows; with ``logits``, the draft-head logits of the last row."""

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
    """Whether the head holds every position but the last committed one, whose hidden row waits in ``mtp_tail``
    (missing earlier rows are zero-filled). Then the next absorb starts at that position."""

    p = st.pos
    if p == 0 or st.mtp_tail is None or st.mtp_tail_at != p - 1:
        return False
    _zero_mtp(st, st.mtp_len, p - 1)
    st.mtp_len = p - 1
    return True


def _mtp_catch_up(e: Decoder, st: State, token: int | None) -> None:
    """Before new rows at position st.pos: the head absorbs the waiting tail with ``token`` (the token at st.pos), or
    zero-fills what it cannot absorb, so it holds every position below st.pos."""

    if token is not None and _tail_ready(st):
        _absorb(e, st, st.mtp_tail[None], [token], logits=False)
    else:
        _zero_mtp(st, st.mtp_len, st.pos)
        st.mtp_len = st.pos


def draft(e: Decoder, st: State, hidden: torch.Tensor, next_tokens: Sequence[int], position: int, count: int,
          sampling: Sampling | None, confidence: float = 0.0) -> list[int]:
    """Absorb rows (hidden states [n, hidden], next tokens) at st.mtp_len onward, then chain up to ``count`` drafts
    for positions position, position + 1, ... (the chain's cache entries sit past st.mtp_len). With ``confidence``
    > 0 the first draft is always kept; the chain ends before a later draft under it, and right after a first
    draft under it."""

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
    """Commit ``prompt`` into ``st`` in chunks of ``chunk`` rows; returns the last prompt row's logits [1, V] (a copy).
    ``resume``: a snapshot of this state's sequence (``State.snapshot``, its cache rows still in place) that the prompt
    extends: only the tokens after it run. ``mtp`` (with an MTP head): the head absorbs every prompt position whose
    next token is known and keeps the last one's hidden row for the first reply token."""

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
            chunk: int | None = None, resume: dict | None = None, mtp: bool = True) -> int:
    """Commit the prompt and sample the first reply token (at position len(prompt))."""

    logits = run_prompt(e, prompt, st=st, chunk=chunk, resume=resume, mtp=mtp)
    return sample_rows(logits, [len(prompt)], sampling)[0]


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
                  on_tokens: Callable[[list[int]], bool] | None = None) -> DecodeResult:
    """Up to ``count`` tokens, starting with ``pending`` (the token ``prefill`` sampled): each step runs one row (the
    last token) and commits it, then samples the next at its position. ``on_tokens(new)`` hears each step's token and
    returns True to stop early. The last token is sampled, not committed (it is the next step's input). The MTP head,
    if any, is not run: a drafted continuation of this state re-syncs it (one round without drafts)."""

    st = st or e.st
    out = [pending]
    pos0 = st.pos
    torch.cuda.synchronize()
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in e.eos):
        logits = e.forward([out[-1]], st)
        tok = sample_rows(logits[:1], [st.pos + 1], sampling)[0]
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
               on_tokens: Callable[[list[int]], bool] | None = None) -> DecodeResult:
    """Up to ``count`` tokens starting with ``pending``, the tokens ``serial_decode`` emits: each round verifies the
    pending token and its MTP drafts in one window, keeps up to the first mismatch, and drafts again. Starts from the
    state ``prefill`` leaves (the head holds every prompt position but the last, whose hidden row waits in the tail);
    from any other state the first round runs without drafts while the head catches up. A round never drafts past
    ``count``, so the caches end as serial decoding leaves them (every token but the last committed), and the head
    holds every committed position but the last, whose row waits in the tail (a continuation resumes drafting).
    ``on_tokens(new)`` hears each round's kept tokens (after ``pending``); it returns True to stop early."""

    if e.mtp is None:
        raise ValueError("drafted decoding needs the MTP head (Decoder(..., mtp=prepare_mtp(...)))")
    st = st or e.st
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
        tokens = [out[-1]] + drafts
        R = len(tokens)
        logits = e.forward(tokens, st)
        sampled = sample_rows(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in e.eos):
                break
            keep += 1
        commit(m, b, st, keep)
        kept = sampled[:keep]
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
    if kept:                                       # the last round's rows: all but the last into the head, the last waits
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
                    confidence: float = CONFIDENCE) -> DecodeResult:
    """``generate`` with the decode loop's counts (rounds, drafts, acceptance, seconds after the prefill)."""

    if max_tokens < 1:
        return DecodeResult([], 0.0, 0)
    drafting = draft and e.mtp is not None and depth > 0
    first = prefill(e, prompt, sampling, st=st, mtp=drafting)
    if (on_tokens is not None and on_tokens([first])) or (stop_eos and first in e.eos) or max_tokens == 1:
        return DecodeResult([first], 0.0, 0)
    if drafting:
        return mtp_decode(e, first, max_tokens, sampling, st=st, depth=depth, confidence=confidence,
                          stop_eos=stop_eos, on_tokens=on_tokens)
    return serial_decode(e, first, max_tokens, sampling, st=st, stop_eos=stop_eos, on_tokens=on_tokens)


@torch.no_grad()
def generate(e: Decoder, prompt: Sequence[int], max_tokens: int, sampling: Sampling | None = None, *,
             stop_eos: bool = True, st: State | None = None, draft: bool = True,
             on_tokens: Callable[[list[int]], bool] | None = None, depth: int = DEPTH,
             confidence: float = CONFIDENCE) -> list[int]:
    """Prefill, then decode up to ``max_tokens`` reply tokens (ending with an eos id when ``stop_eos`` and one comes).
    ``draft`` (the default) drafts with the MTP head when the decoder has one; ``draft=False`` (or no head) decodes one
    token a step, the serial reference. Both return the same tokens. ``on_tokens(new)`` hears the reply's tokens as
    they are decided (the first alone) and returns True to stop early."""

    return generate_result(e, prompt, max_tokens, sampling, stop_eos=stop_eos, st=st, draft=draft,
                           on_tokens=on_tokens, depth=depth, confidence=confidence).tokens


# -- teacher-forced scoring ------------------------------------------------------------------------------------------
@torch.no_grad()
def score(e: Decoder, ids: Sequence[int], *, st: State | None = None, chunk: int | None = None
          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher forcing over ``ids`` from an empty sequence: (argmax at each position (T,) int64, the NLL of each next
    token (T - 1,) fp32), from this engine's bf16 logits (the head over every row, ``logit_rows`` at a time)."""

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
