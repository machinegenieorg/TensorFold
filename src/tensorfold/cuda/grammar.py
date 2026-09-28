"""Structured output for the CUDA engines: a request's JSON schema as a token mask that keeps drafted replies exact.

A request asks with OpenAI's ``response_format`` (``{"type": "json_schema", "json_schema": {"schema": ...}}`` or
``{"type": "json_object"}``), or with vLLM's ``guided_json`` / ``structured_outputs: {"json": ...}``.
``request_spec`` reads that from the body without xgrammar; ``Grammars`` compiles it (``pip install
'tensorfold[grammar]'``), and ``Constraint`` is one reply's grammar state, which an engine's decode loop drives:

- ``mask(logits, drafts)`` at the token choice: row 0 is the next token after the committed ones, row r > 0 the
  token after ``drafts[:r]``; each row's disallowed tokens become -inf before sampling, so a serial step and a
  drafted window's row with the same path choose from the same masked logits (the same token, greedy or keyed).
- ``admissible(drafts)``: the drafts the grammar can follow. A draft it rejects cannot match its row's choice (that
  row is masked), and a stop token ends the reply, so the drafts from either on are dropped before verification.
- ``advance(tokens)``: the chosen (committed) tokens only. ``mask`` and ``admissible`` roll the matcher back to the
  committed state, so a partial keep needs nothing restored.
- ``draft_ok`` / ``mask_draft``: for the drafter only (speed, never the reply): a draft the grammar rejects is
  redrawn from the draft head's logits masked the same way, so the chain proposes tokens a verify row can keep.

The grammar decides when a reply ends: the mask allows a stop token only once the JSON value is complete. With
thinking on, the grammar applies from the token after ``</think>``.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tensorfold.server.errors import RequestError

EXTRA = "tensorfold[grammar]"             # the optional dependency that brings xgrammar
CACHE_BYTES = 256 << 20                   # compiled grammars kept for repeated schemas
OBJECT = '{"type": "object"}'             # json_object: any JSON object (OpenAI's contract), not an array


@dataclass(frozen=True)
class Spec:
    """A request's structured output: ``kind`` "json" (any JSON object) or "json_schema" (``schema`` as JSON text)."""

    kind: str
    schema: str = ""
    field: str = "response_format"


def _schema_text(value: Any, where: str) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RequestError(f"{where} is not valid JSON: {exc.msg}") from None
    if isinstance(value, bool):
        value = {} if value else None
    if not isinstance(value, dict):
        raise RequestError(f"{where} must be a JSON schema object")
    return json.dumps(value, ensure_ascii=False)


def request_spec(body: dict[str, Any]) -> Spec | None:
    """The body's structured-output request, None for plain text; RequestError (HTTP 400) when it is malformed."""

    rf = body.get("response_format")
    if rf is not None:
        if not isinstance(rf, dict):
            raise RequestError("response_format must be an object such as {\"type\": \"json_object\"}")
        kind = rf.get("type")
        if kind == "text":
            rf = None
        elif kind == "json_object":
            return Spec("json")
        elif kind == "json_schema":
            js = rf.get("json_schema")
            if not isinstance(js, dict) or js.get("schema") is None:
                raise RequestError("response_format json_schema needs json_schema.schema (a JSON schema object)")
            return Spec("json_schema", _schema_text(js["schema"], "response_format json_schema.schema"))
        else:
            raise RequestError(f"response_format type must be text, json_object or json_schema, not {kind!r}")
    if body.get("guided_json") is not None:
        return Spec("json_schema", _schema_text(body["guided_json"], "guided_json"), "guided_json")
    for name in ("guided_regex", "guided_choice", "guided_grammar"):
        if body.get(name) is not None:
            raise RequestError(f"{name} is not supported: constrain replies with a JSON schema (response_format)")
    so = body.get("structured_outputs")
    if so is not None:
        if not isinstance(so, dict):
            raise RequestError("structured_outputs must be an object such as {\"json\": {...}}")
        other = sorted(k for k, v in so.items() if v is not None and k != "json")
        if other:
            raise RequestError(f"structured_outputs {', '.join(other)} is not supported: use structured_outputs.json")
        if so.get("json") is not None:
            return Spec("json_schema", _schema_text(so["json"], "structured_outputs.json"), "structured_outputs")
    return None


def _message(exc: Exception) -> str:
    """xgrammar's error without its timestamp and source location."""

    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return re.sub(r"^\[[^\]]*\]\s*\S+:\d+:\s*", "", text)


class Grammars:
    """One tokenizer's grammar compiler (xgrammar), with compiled grammars cached by schema."""

    def __init__(self, info, *, think_end: int | None = None) -> None:
        import xgrammar as xgr

        self.xgr = xgr
        self.info = info
        self.vocab_size = int(info.vocab_size)
        self.think_end = think_end
        self.compiler = xgr.GrammarCompiler(info, max_threads=8, cache_limit_bytes=CACHE_BYTES)
        self.lock = threading.Lock()              # requests compile on their own HTTP threads

    @classmethod
    def for_model(cls, model_dir: str | Path, vocab_size: int, stop_ids: Sequence[int]) -> "Grammars":
        """From the checkpoint's ``tokenizer.json``; ``vocab_size`` is the logits' width, ``stop_ids`` its eos ids."""

        try:
            import xgrammar as xgr
            from transformers import PreTrainedTokenizerFast
        except ImportError:
            raise RequestError(f"structured output needs xgrammar on the server: pip install '{EXTRA}'") from None
        tok = PreTrainedTokenizerFast(tokenizer_file=str(Path(model_dir) / "tokenizer.json"))
        info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=int(vocab_size), stop_token_ids=list(stop_ids))
        think_end = tok.convert_tokens_to_ids("</think>")
        return cls(info, think_end=think_end if isinstance(think_end, int) and think_end >= 0 else None)

    def compile(self, spec: Spec):
        """The compiled grammar, or RequestError naming what the schema gets wrong."""

        try:
            with self.lock:
                return self.compiler.compile_json_schema(OBJECT if spec.kind == "json" else spec.schema)
        except (RuntimeError, ValueError, TypeError) as exc:
            raise RequestError(f"{spec.field}: the JSON schema cannot be enforced: {_message(exc)}") from None

    def constraint(self, compiled, *, after_think: bool = False) -> "Constraint":
        return Constraint(self.xgr, compiled, self.vocab_size,
                          think_end=self.think_end if after_think else None)


_MODELS: dict[tuple[str, int, tuple[int, ...]], Grammars] = {}
_BUILD = threading.Lock()


def for_model(model_dir: str | Path, vocab_size: int, stop_ids: Sequence[int]) -> Grammars:
    """``Grammars.for_model``, built once per checkpoint (the first structured request pays about a second)."""

    key = (str(Path(model_dir).resolve()), int(vocab_size), tuple(int(t) for t in stop_ids))
    with _BUILD:
        found = _MODELS.get(key)
        if found is None:
            found = _MODELS[key] = Grammars.for_model(model_dir, vocab_size, stop_ids)
        return found


def vocab_size(model_dir: str | Path) -> int | None:
    """The checkpoint's logits width from ``config.json`` (``text_config`` first), or None."""

    path = Path(model_dir) / "config.json"
    if not path.is_file():
        return None
    config = json.loads(path.read_text())
    for part in (config.get("text_config") or {}, config):
        if isinstance(part.get("vocab_size"), int):
            return int(part["vocab_size"])
    return None


class Constraint:
    """One reply's grammar at its committed tokens: masks verify rows, filters drafts, follows chosen tokens."""

    def __init__(self, xgr, compiled, vocab_size: int, *, think_end: int | None = None) -> None:
        self.xgr = xgr
        self.m = xgr.GrammarMatcher(compiled)
        self.vocab = int(vocab_size)
        self.words = (self.vocab + 31) // 32
        self.think_end = think_end
        self.active = think_end is None               # with thinking on: from the token after </think>
        self._shifts: dict[Any, Any] = {}
        self._columns: dict[Any, Any] = {}

    @property
    def finished(self) -> bool:
        """The grammar has taken its stop token: the reply is complete."""

        return self.active and self.m.is_terminated()

    def _walk(self, drafts: Sequence[int], bitmask=None, *, last: bool = False) -> tuple[int, list[int]]:
        """Follow ``drafts`` from the committed state and roll back: (drafts it admits, rows whose mask it filled).

        ``last``: fill only the row after every draft, as the bitmask's row 0."""

        m = self.m
        active = self.active
        accepted = 0
        admitted = len(drafts)
        filled: list[int] = []
        try:
            for r in range(len(drafts) + 1):
                if r:
                    t = int(drafts[r - 1])
                    if active:
                        if not m.accept_token(t):
                            admitted = r - 1
                            break
                        accepted += 1
                        if m.is_terminated():           # a stop token: the reply ends with it
                            admitted = r - 1
                            break
                    elif t == self.think_end:
                        active = True
                if bitmask is not None and active and not m.is_terminated() and (not last or r == len(drafts)):
                    m.fill_next_token_bitmask(bitmask, 0 if last else r)
                    filled.append(r)
        finally:
            if accepted:
                m.rollback(accepted)
        return admitted, filled

    def admissible(self, drafts: Sequence[int]) -> list[int]:
        """The drafts before the first one the grammar rejects or that ends the reply (neither can be kept)."""

        if not drafts or (not self.active and self.think_end not in drafts):
            return list(drafts)
        return list(drafts[:self._walk(drafts)[0]])

    def mask(self, logits, drafts: Sequence[int] = ()):
        """``logits`` [1 + len(drafts), V] with each row's disallowed tokens at -inf (a copy; unmasked rows kept)."""

        import torch

        rows = logits.shape[0]
        if rows != len(drafts) + 1:
            raise ValueError("mask takes one row per path: the committed tokens, then each draft")
        if not self.active and self.think_end not in drafts:
            return logits
        bitmask = torch.full((rows, self.words), -1, dtype=torch.int32, pin_memory=logits.is_cuda)
        _, filled = self._walk(drafts, bitmask)
        if not filled:
            return logits
        dev = logits.device
        shifts = self._shifts.get(dev)
        if shifts is None:
            shifts = self._shifts[dev] = torch.arange(32, dtype=torch.int32, device=dev)
        bits = bitmask.to(dev, non_blocking=True)
        allowed = ((bits.unsqueeze(-1) >> shifts) & 1).view(rows, -1).bool()
        width = logits.shape[1]
        if allowed.shape[1] < width:                  # logits past the tokenizer's vocabulary: never allowed
            allowed = torch.nn.functional.pad(allowed, (0, width - allowed.shape[1]), value=False)
        return logits.masked_fill(~allowed[:, :width], float("-inf"))

    def draft_ok(self, drafts: Sequence[int], token: int) -> bool:
        """Whether the grammar follows ``drafts`` and then ``token`` without the reply ending there."""

        path = list(drafts) + [int(token)]
        if not self.active and self.think_end not in path:
            return True
        return self._walk(path)[0] == len(path)

    def mask_draft(self, logits, drafts: Sequence[int], ids=None):
        """A draft head's first row [1, n] after ``drafts`` with disallowed tokens at -inf (column c is ``ids[c]``),
        or None when the head has no column the grammar allows."""

        import torch

        if not self.active and self.think_end not in drafts:
            return logits
        bitmask = torch.full((1, self.words), -1, dtype=torch.int32, pin_memory=logits.is_cuda)
        if not self._walk(drafts, bitmask, last=True)[1]:
            return logits
        dev, width = logits.device, logits.shape[1]
        key = (dev, width, None if ids is None else ids.data_ptr())
        cols = self._columns.get(key)
        if cols is None:                              # each column's word and bit, and whether it is a token at all
            t = torch.arange(width, device=dev, dtype=torch.int64) if ids is None else ids.to(dev, torch.int64)
            real = t < self.vocab
            t = torch.where(real, t, 0)
            cols = self._columns[key] = (t >> 5, (t & 31).to(torch.int32), real)
        word, bit, real = cols
        bits = bitmask.to(dev, non_blocking=True)[0]
        allowed = (((bits.index_select(0, word) >> bit) & 1).bool() & real)[None]
        if not bool(allowed.any()):
            return None
        return logits[:1].masked_fill(~allowed, float("-inf"))

    def advance(self, tokens: Sequence[int]) -> None:
        """Follow chosen tokens (each chosen under this grammar's mask); after the stop token, nothing follows."""

        for t in tokens:
            t = int(t)
            if not self.active:
                self.active = t == self.think_end
                continue
            if self.m.is_terminated():
                return
            if not self.m.accept_token(t):
                raise RuntimeError(f"the grammar rejected chosen token {t}")


__all__ = ["Constraint", "EXTRA", "Grammars", "Spec", "for_model", "request_spec", "vocab_size"]
