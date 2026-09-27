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
                 cache_entries: int = 4, cache_gib: float | None = 8.0, allow_copy: bool = True):
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
        # cached states hold their attention rows: ~64 KiB a token, so a 24k-token agent prompt is ~1.5 GiB. A
        # quarter of the budget for shared prefixes, a quarter for agents' system blocks, half for their turns.
        budget = None if cache_gib is None else int(cache_gib * 2**30)
        shared, blocks, turns = (None, None, None) if budget is None else (budget // 4, budget // 4, budget // 2)
        self.sched = Scheduler(self.w, self.draft, concurrency=concurrency, row_budget=row_budget, max_rows=max_rows,
                               allow_copy=allow_copy, prefill_reserve=prefill_reserve,
                               cache=PrefixCache(cache_entries, shared), kv_budget_gib=kv_budget_gib, turn_budget=turns,
                               block_budget=blocks,
                               log=lambda m: print(f"[tensorfold] {m}", flush=True))
        self.wake = threading.Condition()
        self.thread = threading.Thread(target=self._loop, name="tensorfold-scheduler", daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        self.torch.cuda.set_device(0)
        released = True
        while True:
            with self.wake:
                while self.sched.idle():
                    # 5 s with nothing to do: hand the memory the last burst used back to the machine (the
                    # allocator keeps freed blocks otherwise; cached states stay, within --cache-gib)
                    if not released and not self.wake.wait(timeout=5.0) and self.sched.idle():
                        self.torch.cuda.empty_cache()
                        released = True
                    elif released:
                        self.wake.wait()
            released = False
            try:
                self.sched.step()
                self.sched.failed_steps = 0
            except Exception as exc:                  # end every request with the error, keep serving
                traceback.print_exc()
                self.sched.failed_steps += 1
                self.sched.fail_all(f"{type(exc).__name__}: {exc}")

    def health(self) -> tuple[bool, dict]:
        """Unhealthy when the scheduler thread is gone, rounds keep failing, or work is waiting and no round has
        finished for 5 minutes (a round is seconds; a stuck engine never finishes one)."""

        h = self.sched.health()
        h["thread_alive"] = self.thread.is_alive()
        stuck = (h["live"] or h["waiting"]) and h["last_step_age_s"] > 300
        ok = h["thread_alive"] and h["failed_steps"] < 3 and not stuck
        return bool(ok), h

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
                 draft: bool = True, constraint=None, checkpoint: int = 0) -> dict:
        """``draft=False``: one token a round, no drafts and no copies (the serial reference). ``constraint``: a
        grammar the reply must follow (see ``grammar.py``)."""

        from .sched import Request

        inbox: queue.SimpleQueue = queue.SimpleQueue()
        r = Request(list(prompt), int(max_tokens), sampling, serial=not draft, emit=inbox.put, constraint=constraint,
                    checkpoint=int(checkpoint))
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
        return {"cached": r.cached, "queued_s": round(waited, 3), "first_token_s": round(max(0.0, r.t_first - r.t_admit), 3),
                "decode_s": round(max(0.0, r.t_done - r.t_first), 3), "rounds": r.rounds, "drafts": draft,
                "accepted": r.accepted}
