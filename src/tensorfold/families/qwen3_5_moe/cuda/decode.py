"""Qwen3.6-35B-A3B decode on CUDA: prefill and serial decoding, the reference behaviour drafted decoding must match.

Every emitted token is the keyed sample (``tensorfold.cuda.sampling.sample_rows``: seeded Gumbel over top-k/top-p,
ties by token id) of this engine's logits at its position, or the argmax (the lowest id among equal logits) when
greedy. A serial step is a one-row window of ``forward`` followed by a one-row ``commit``; a drafted round verifies
several rows through the same kernels and sampler, so it keeps a draft exactly when it equals what this loop samples
there.

A prompt runs in chunks of up to ``Decoder.buf.rows`` rows (512 by default). Rows never depend on their chunk, so any
chunking, and a resumed prompt (``State.snapshot``), ends in the state and logits of one fresh prefill.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from tensorfold.cuda.sampling import sample_rows  # noqa: F401  (the CUDA families' keyed sampler, re-exported)
from tensorfold.engine.exact_sampling import Sampling

from . import qmm
from .forward import Buffers, Model, Pool, State, commit, forward

CONTEXT = 4096            # prompt plus reply tokens the caches hold, by default
PREFILL_ROWS = 512        # rows a prefill chunk runs at once


class Decoder:
    """A model, one set of window buffers, and a pool of sequence states; ``st`` is the sequence the helpers below
    run unless given another. ``states``: how many sequences the pool holds (tests and A/B runs take clones)."""

    def __init__(self, m: Model, *, capacity: int = CONTEXT, rows: int = PREFILL_ROWS, window_rows: int = 32,
                 attn_rows: int = 64, logit_rows: int = 32, states: int = 1) -> None:
        self.m = m
        self.capacity = capacity
        self.buf = Buffers(m, rows, capacity=capacity, window_rows=window_rows, attn_rows=attn_rows,
                           logit_rows=logit_rows)
        self.pool = Pool(m, states, capacity)
        self.st = self.pool.alloc()

    @property
    def eos(self) -> tuple[int, ...]:
        return tuple(self.m.cfg.eos)

    def forward(self, tokens: Sequence[int], st: State | None = None, *, logits: str = "all") -> torch.Tensor | None:
        """A window of ``tokens`` after ``st``'s committed sequence (logits [R, V], a view of the buffers)."""

        return forward(self.m, self.buf, st or self.st, tokens, logits=logits)

    def commit(self, keep: int, st: State | None = None) -> None:
        commit(self.m, self.buf, st or self.st, keep)


# -- prefill ---------------------------------------------------------------------------------------------------------
@torch.no_grad()
def run_prompt(e: Decoder, prompt: Sequence[int], *, st: State | None = None, chunk: int | None = None,
               resume: dict | None = None) -> torch.Tensor:
    """Commit ``prompt`` into ``st`` in chunks of ``chunk`` rows; returns the last prompt row's logits [1, V] (a copy).
    ``resume``: a snapshot of this state's sequence (``State.snapshot``, its cache rows still in place) that the prompt
    extends: only the tokens after it run."""

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
    logits = None
    for start in range(st.pos, len(prompt), chunk):
        part = list(prompt[start:start + chunk])
        final = start + len(part) >= len(prompt)
        logits = forward(e.m, e.buf, st, part, logits="last" if final else "none")
        commit(e.m, e.buf, st, len(part))
    return logits[:1].clone()


@torch.no_grad()
def prefill(e: Decoder, prompt: Sequence[int], sampling: Sampling | None = None, *, st: State | None = None,
            chunk: int | None = None, resume: dict | None = None) -> int:
    """Commit the prompt and sample the first reply token (at position len(prompt))."""

    logits = run_prompt(e, prompt, st=st, chunk=chunk, resume=resume)
    return sample_rows(logits, [len(prompt)], sampling)[0]


# -- serial decoding -------------------------------------------------------------------------------------------------
@dataclass
class DecodeResult:
    tokens: list[int]                                   # the pending token, then each step's
    seconds: float
    rounds: int
    widths: list[int] = field(default_factory=list)     # rows each round verified (1: serial)

    @property
    def tokens_per_second(self) -> float:
        return (len(self.tokens) - 1) / self.seconds if self.seconds else 0.0


@torch.no_grad()
def serial_decode(e: Decoder, pending: int, count: int, sampling: Sampling | None = None, *,
                  st: State | None = None, stop_eos: bool = False,
                  on_tokens: Callable[[list[int]], bool] | None = None) -> DecodeResult:
    """Up to ``count`` tokens, starting with ``pending`` (the token ``prefill`` sampled): each step runs one row (the
    last token) and commits it, then samples the next at its position. ``on_tokens(new)`` hears each step's token and
    returns True to stop early. The last token is sampled, not committed (it is the next step's input)."""

    st = st or e.st
    out = [pending]
    torch.cuda.synchronize()
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in e.eos):
        logits = forward(e.m, e.buf, st, [out[-1]])
        tok = sample_rows(logits[:1], [st.pos + 1], sampling)[0]
        commit(e.m, e.buf, st, 1)
        out.append(tok)
        if on_tokens is not None and on_tokens([tok]):
            break
    torch.cuda.synchronize()
    return DecodeResult(out, time.perf_counter() - start, len(out) - 1, [1] * (len(out) - 1))


@torch.no_grad()
def generate(e: Decoder, prompt: Sequence[int], max_tokens: int, sampling: Sampling | None = None, *,
             stop_eos: bool = True, st: State | None = None) -> list[int]:
    """The serial reference: prefill, then one token a step; returns up to ``max_tokens`` reply tokens (ending with an
    eos id when ``stop_eos`` and one comes)."""

    if max_tokens < 1:
        return []
    first = prefill(e, prompt, sampling, st=st)
    return serial_decode(e, first, max_tokens, sampling, st=st, stop_eos=stop_eos).tokens


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
