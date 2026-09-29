"""OpenAI ``/v1/embeddings`` for CUDA engines that embed: request checks, a priority queue of forward steps, replies.

An engine that embeds exposes ``embed(texts) -> (n, dimensions) float32 rows`` (pooled, not yet L2-normalized),
``dimensions``, ``vocab``, ``context_window`` (the longest text) and ``batch_tokens`` (the tokens one step takes).
Its rows must not depend on the batch, so the queue packs waiting texts from any requests into each step.
"""

from __future__ import annotations

import base64
import itertools
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from tensorfold.server.cancellation import RequestCancelled
from tensorfold.server.errors import RequestError

MAX_INPUTS = 2048            # texts one request may carry, as OpenAI's endpoint allows
MIN_DIMENSIONS = 32          # Qwen3-Embedding's smallest trained output (MRL); a family may set its own


@dataclass(slots=True)
class EmbeddingRequest:
    texts: list[list[int]]           # token ids as the model reads them, truncated where asked
    dimensions: int                  # leading dimensions kept, then L2-normalized again
    base64: bool
    priority: int                    # lower is served first, as vLLM's priority scheduling
    model: str | None

    @property
    def tokens(self) -> int:
        return sum(len(t) for t in self.texts)


def _integer(body: dict[str, Any], name: str) -> int | None:
    value = body.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError(f"{name} must be an integer")
    return value


def _texts(value: Any) -> tuple[list[str] | None, list[list[int]] | None]:
    """``input``: a string, strings, a token array or token arrays."""

    if isinstance(value, str):
        return [value], None
    if not isinstance(value, list) or not value:
        raise RequestError("input must be a string, a list of strings, a token array or a list of token arrays, "
                           "and not empty")
    if len(value) > MAX_INPUTS and not all(isinstance(v, int) for v in value):
        raise RequestError(f"input takes at most {MAX_INPUTS} texts a request")
    if all(isinstance(v, str) for v in value):
        return list(value), None
    if all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        return None, [list(value)]
    if all(isinstance(v, list) for v in value):
        arrays = []
        for v in value:
            if not v or not all(isinstance(t, int) and not isinstance(t, bool) for t in v):
                raise RequestError("each token array in input must be a nonempty list of integers")
            arrays.append(list(v))
        return None, arrays
    raise RequestError("input mixes strings and token arrays")


def truncate(tok: Any, text: str | None, ids: list[int] | None, limit: int | None, side: str) -> list[int]:
    """One input's tokens, cut to ``limit`` if it is longer.

    Text keeps its start by default as Hugging Face truncation does: the words are cut and the tokenizer's
    special tokens (Qwen3-Embedding's closing ``<|endoftext|>``) are added after, so the pooled token stays the
    one the model was trained to pool. ``truncation_side: "left"`` keeps the last ``limit`` tokens instead.
    Token arrays are read as given and keep their first (or, "left", last) ``limit`` tokens.
    """

    if ids is not None:
        if limit is None or len(ids) <= limit:
            return ids
        return ids[-limit:] if side == "left" else ids[:limit]
    if limit is None or side == "left":
        full = tok.encode(text, add_special_tokens=True).ids
        return full if limit is None or len(full) <= limit else full[-limit:]
    words = tok.encode(text, add_special_tokens=False)
    processor = tok.post_processor
    extra = processor.num_special_tokens_to_add(False) if processor is not None else 0
    if len(words.ids) + extra > limit:
        words.truncate(max(0, limit - extra))
    return tok.post_process(words).ids


def parse(body: Any, *, tok: Any, context: int, dimensions: int, vocab: int,
          min_dimensions: int = MIN_DIMENSIONS) -> EmbeddingRequest:
    """Every check a request gets before it waits for the GPU; a ``RequestError`` is an HTTP 400."""

    if not isinstance(body, dict):
        raise RequestError("the request body must be a JSON object")
    if "input" not in body:
        raise RequestError("input is required")
    strings, arrays = _texts(body["input"])
    fmt = body.get("encoding_format")
    if fmt not in (None, "float", "base64"):
        raise RequestError('encoding_format must be "float" or "base64"')
    dims = _integer(body, "dimensions")
    if dims is not None and not min_dimensions <= dims <= dimensions:
        raise RequestError(f"dimensions must be from {min_dimensions} to {dimensions}")
    limit = _integer(body, "truncate_prompt_tokens")
    if limit is not None:
        if limit == -1:
            limit = context
        elif limit < 1:
            raise RequestError("truncate_prompt_tokens must be a positive token count, or -1 for the server's limit")
        elif limit > context:
            raise RequestError(f"truncate_prompt_tokens ({limit}) exceeds the server's {context}-token limit a text")
    side = body.get("truncation_side")
    if side not in (None, "left", "right"):
        raise RequestError('truncation_side must be "left" or "right"')
    priority = _integer(body, "priority") or 0
    model = body.get("model")
    if model is not None and not isinstance(model, str):
        raise RequestError("model must be a string")
    if strings is not None:
        texts = [truncate(tok, s, None, limit, side or "right") for s in strings]
    else:
        for ids in arrays:
            if any(not 0 <= t < vocab for t in ids):
                raise RequestError(f"token ids must be integers from 0 to {vocab - 1}")
        texts = [truncate(tok, None, ids, limit, side or "right") for ids in arrays]
    for i, ids in enumerate(texts):
        if not ids:
            raise RequestError(f"input {i} has no tokens")
        if len(ids) > context:
            raise RequestError(f"input {i} has {len(ids)} tokens, more than the server's {context}-token limit a "
                               "text; shorten it or send truncate_prompt_tokens")
    return EmbeddingRequest(texts, dims or dimensions, fmt == "base64", priority, model)


def normalize(row: np.ndarray, dimensions: int) -> np.ndarray:
    """The leading ``dimensions`` of one pooled row, L2-normalized in float64 and returned as float32 (MRL)."""

    kept = np.asarray(row[:dimensions], dtype=np.float64)
    norm = float(np.sqrt(np.dot(kept, kept)))
    return (kept / max(norm, 1e-12)).astype(np.float32)


def reply(request: EmbeddingRequest, rows: list[np.ndarray], model: str) -> dict[str, Any]:
    """The OpenAI response body; float vectors round-trip to the same float32 bits."""

    data = []
    for i, row in enumerate(rows):
        vector = normalize(row, request.dimensions)
        value = (base64.b64encode(vector.astype("<f4").tobytes()).decode("ascii") if request.base64
                 else vector.tolist())
        data.append({"object": "embedding", "index": i, "embedding": value})
    tokens = request.tokens
    return {"id": f"embd-{uuid.uuid4().hex}", "object": "list", "created": int(time.time()), "model": model,
            "data": data, "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}


@dataclass(eq=False)
class _Job:
    priority: int
    seq: int
    texts: list[list[int]]
    rows: list[Any] = field(default_factory=list)
    taken: int = 0                   # texts handed to steps so far
    filled: int = 0
    cancelled: bool = False
    error: BaseException | None = None
    done: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self):
        self.rows = [None] * len(self.texts)


class EmbedQueue:
    """One worker thread runs the engine; each step takes waiting texts by (priority, arrival) up to a token budget.

    A lower ``priority`` number goes first, as in vLLM's priority scheduling: a waiting query gets the next step to
    itself (with any other waiting texts of its priority), so it waits for the step in flight, not for the whole
    bulk batch. Requests of equal priority are served in arrival order and share steps. A text longer than the
    budget runs alone.
    """

    def __init__(self, engine: Any, budget: int, *, poll: float = 0.05):
        self.engine, self.budget, self.poll = engine, max(1, int(budget)), poll
        self.cond = threading.Condition()
        self.jobs: list[_Job] = []
        self.seq = itertools.count()
        self.steps = 0
        self.thread = threading.Thread(target=self._run, name="tensorfold-embed", daemon=True)
        self.thread.start()

    def submit(self, texts: list[list[int]], priority: int = 0,
               cancelled: Callable[[], bool] | None = None) -> list[Any]:
        job = _Job(int(priority), next(self.seq), texts)
        with self.cond:
            self.jobs.append(job)
            self.cond.notify()
        while not job.done.wait(self.poll):
            if cancelled is not None and cancelled():
                with self.cond:
                    job.cancelled = True
                raise RequestCancelled("the client left before its embeddings were ready")
        if job.error is not None:
            raise job.error
        return job.rows

    def _take(self) -> list[tuple[_Job, int]]:
        """The next step: the most urgent priority's waiting texts in arrival order while they fit the budget (always
        one). Texts of a later priority never fill it, so an urgent step is only as long as its own texts."""

        self.jobs = [j for j in self.jobs if not j.cancelled and j.taken < len(j.texts)]
        step, used = [], 0
        waiting = sorted(self.jobs, key=lambda j: (j.priority, j.seq))
        for job in waiting:
            if job.priority != waiting[0].priority:
                break
            while job.taken < len(job.texts):
                n = len(job.texts[job.taken])
                if step and used + n > self.budget:
                    return step
                step.append((job, job.taken))
                job.taken += 1
                used += n
        return step

    def _run(self) -> None:
        while True:
            with self.cond:
                step = self._take()
                while not step:
                    self.cond.wait()
                    step = self._take()
            try:
                rows = self.engine.embed([job.texts[i] for job, i in step])
            except Exception as exc:  # noqa: BLE001 - the step's requests fail; the others go on
                with self.cond:
                    for job in {job for job, _ in step}:
                        job.error, job.cancelled = exc, True
                        job.done.set()
                continue
            self.steps += 1
            for (job, i), row in zip(step, rows):
                job.rows[i] = row
                job.filled += 1
                if job.filled == len(job.texts) and job.error is None:
                    job.done.set()


_QUEUE_LOCK = threading.Lock()


def queue_for(app: Any) -> EmbedQueue:
    """The app's queue, started on the first embeddings request."""

    with _QUEUE_LOCK:
        found = getattr(app, "_embed_queue", None)
        if found is None:
            engine = app.engine
            found = EmbedQueue(engine, getattr(engine, "batch_tokens", 8192))
            app._embed_queue = found
        return found


def serve(app: Any, body: Any, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
    """One ``/v1/embeddings`` request: parsed and refused before it queues, then embedded with the others."""

    engine = app.engine
    if not callable(getattr(engine, "embed", None)):
        raise RequestError(f"{app.served} generates text and does not serve embeddings")
    request = parse(body, tok=app.tok, context=int(engine.context_window), dimensions=int(engine.dimensions),
                    vocab=int(engine.vocab), min_dimensions=int(getattr(engine, "min_dimensions", MIN_DIMENSIONS)))
    rows = queue_for(app).submit(request.texts, request.priority, cancelled)
    ids = app.model_ids
    return reply(request, rows, request.model if request.model in ids else app.served)
