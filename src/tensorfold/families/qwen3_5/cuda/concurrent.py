"""Qwen3.8-27B on one GPU, serving many requests at once (``tensorfold serve --concurrency N``).

The server calls ``generate`` from one thread per request, as with ``Qwen27Engine``; here the calls do not take
turns. Each call queues its request with a ``Scheduler`` (``sched.py``), which one background thread steps:
every round is one verify forward over the draft windows of the requests decoding and the prompt chunks of
the requests prefilling. A request's tokens are the same as its serial decode on this engine, whatever else is
running with it.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Callable


class ConcurrentEngine:
    """Many requests at once on one GPU; ``generate`` blocks its caller and streams through ``on_tokens``."""

    concurrent = True

    def __init__(self, model_dir: Path, draft_dir: Path | None, *, concurrency: int = 16, max_rows: int = 6,
                 row_budget: int = 128, prefill_reserve: int = 32, kv_budget_gib: float | None = None,
                 cache_entries: int = 4, allow_copy: bool = True):
        import torch

        from .sched import PrefixCache, Scheduler
        from .weights import load

        self.torch = torch
        self.model_dir = Path(model_dir)
        self.grammars = None
        torch.cuda.set_device(0)
        self.w = load(model_dir, tiled=True)
        self.draft = None
        if draft_dir is not None:
            from .dflash2 import DFlash2

            self.draft = DFlash2(draft_dir, self.w)
        torch.cuda.empty_cache()
        self.eos = tuple(self.w.config.eos)
        self.sched = Scheduler(self.w, self.draft, concurrency=concurrency, row_budget=row_budget, max_rows=max_rows,
                               allow_copy=allow_copy, prefill_reserve=prefill_reserve,
                               cache=PrefixCache(cache_entries), kv_budget_gib=kv_budget_gib)
        self.wake = threading.Condition()
        self.thread = threading.Thread(target=self._loop, name="tensorfold-scheduler", daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        self.torch.cuda.set_device(0)
        while True:
            with self.wake:
                while self.sched.idle():
                    self.wake.wait()
            try:
                self.sched.step()
            except Exception as exc:                  # end every request with the error, keep serving
                traceback.print_exc()
                self.sched.fail_all(f"{type(exc).__name__}: {exc}")

    def make_constraint(self, body: dict, *, after_think: bool = False):
        """The grammar a request asks for (``response_format`` json_schema / json_object, vLLM's ``guided_json``
        or ``structured_outputs.json``), or None."""

        from .grammar import Grammars, schema_spec

        spec = schema_spec(body)
        if spec is None:
            return None
        if self.grammars is None:
            self.grammars = Grammars(self.model_dir, self.w.config.vocab, self.eos)
        return self.grammars.constraint(spec, after_think=after_think)

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens: Callable[[list[int]], bool | None],
                 draft: bool = True, constraint=None) -> dict:
        """``draft=False``: one token a round, no drafts and no copies (the serial reference). ``constraint``: a
        grammar the reply must follow (see ``grammar.py``)."""

        from .sched import Request

        inbox: queue.SimpleQueue = queue.SimpleQueue()
        r = Request(list(prompt), int(max_tokens), sampling, serial=not draft, emit=inbox.put, constraint=constraint)
        t0 = time.perf_counter()
        with self.wake:
            self.sched.submit(r)
            self.wake.notify()
        stopped = False
        while True:
            new = inbox.get()
            if new is None:
                break
            if not stopped and on_tokens(new):
                stopped, r.cancel = True, True
        if r.error:
            raise RuntimeError(r.error)
        waited = max(0.0, r.t_admit - (t0 - self.sched.t0))
        return {"queued_s": round(waited, 3), "first_token_s": round(max(0.0, r.t_first - r.t_admit), 3),
                "decode_s": round(max(0.0, r.t_done - r.t_first), 3), "rounds": r.rounds, "drafts": draft,
                "accepted": r.accepted}
