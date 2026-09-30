"""Requests submit from any thread; one worker thread runs the rounds, and a slow client only fills its own queue.

Waiting requests start in ``priority`` order (an integer, lower first, as with vLLM's priority scheduling), and in
arrival order within one priority. ``reserve`` streams are kept for foreground requests (priority 0 or below): background
work (priority above 0) never holds more than ``max_streams - reserve`` of them, so a foreground request finds a stream
without waiting for a background one to finish. Streams already running are never paused, and the order requests start
in never changes a request's tokens."""

from __future__ import annotations

import itertools
import queue
import threading
from typing import Any, Callable

from .streams import Stream


class Scheduler:
    def __init__(self, decoder: Any, *, max_streams: int = 4, reserve: int = 0) -> None:
        if not 0 <= reserve < max_streams:
            raise ValueError(f"reserve takes 0 to {max_streams - 1} of {max_streams} streams, not {reserve}")
        self.decoder = decoder
        self.max_streams = max_streams
        self.reserve = reserve
        self.waiting: queue.PriorityQueue = queue.PriorityQueue()   # (priority, arrival, stream, box)
        self.arrivals = itertools.count()
        self.boxes: dict[int, queue.Queue] = {}
        self.background: set[int] = set()            # live streams whose request has priority above 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, prompt: list[int], count: int, sampling: Any, draft: bool,
               emit: Callable[[list[int]], bool | None], stop_eos: bool = True, *, vision: Any = None,
               constraint: Any = None, priority: int = 0) -> dict:
        """Decode one request; ``emit`` runs on the calling thread and returns True to stop. Returns its stats."""

        box: queue.Queue = queue.Queue()
        stream = Stream(list(prompt), max(1, count), sampling, draft=draft, stop_eos=stop_eos, vision=vision,
                        constraint=constraint)
        cancel = [False]
        stream.emit = lambda new: (box.put(("tokens", new)), cancel[0])[1]
        self.waiting.put((int(priority), next(self.arrivals), stream, box))
        while True:
            kind, value = box.get()
            if kind == "tokens":
                if not cancel[0] and emit(value):
                    cancel[0] = True                 # the client left: the stream ends after its next round
            elif kind == "error":
                raise value
            else:
                return value

    def _admit(self, first=None) -> list[Stream]:
        done = []
        while self.decoder.live() < self.max_streams:
            if first is not None:
                item, first = first, None
            else:
                try:
                    item = self.waiting.get_nowait()
                except queue.Empty:
                    break
            priority, _, stream, box = item
            if priority > 0 and len(self.background) >= self.max_streams - self.reserve:
                self.waiting.put(item)               # the head is background work and its share is full: wait
                break
            self.boxes[id(stream)] = box
            if priority > 0:
                self.background.add(id(stream))
            try:
                self.decoder.admit(stream)
            except Exception as exc:                 # noqa: BLE001  (this request fails, the others go on)
                self.background.discard(id(stream))
                self.boxes.pop(id(stream)).put(("error", exc))
                continue
            if stream.done:
                done.append(stream)
        return done

    def _reply(self, s: Stream, kind: str, value: Any) -> None:
        self.background.discard(id(s))
        box = self.boxes.pop(id(s), None)            # None: the stream's request has had its reply
        if box is not None:
            box.put((kind, value))

    def _loop(self) -> None:
        while True:
            done = self._admit(None if self.decoder.live() else self.waiting.get())   # idle: wait for a request
            try:
                done += self.decoder.round()
            except Exception as exc:                 # noqa: BLE001  (the live requests fail)
                for s in self.decoder.drop():
                    self._reply(s, "error", exc)
            self.decoder.finish(done)
            for s in done:
                self._reply(s, *(("error", s.error) if s.error is not None else ("done", s.stats())))
