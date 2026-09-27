"""JSON-schema (and plain JSON) constrained replies that stay exact under drafting and batching.

A constraint masks each verify row's logits to the tokens its grammar allows after that row's path, before any
token is chosen: a draft tree is walked depth first, the matcher accepting a node's token on the way down and
rolling it back on the way up. The same rows, masked the same way, choose the same tokens in a drafted window,
in a batch or one at a time, so a constrained reply is byte-identical to its constrained serial decode. A row
whose token the grammar rejects can never be on an accepted path (its parent cannot choose it), so it is left
unmasked. With thinking on, the grammar applies from the token after ``</think>``.

Needs ``xgrammar`` (``pip install xgrammar``).
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


def schema_spec(body: dict[str, Any]) -> tuple[str, Any] | None:
    """The structured-output request in an OpenAI (or vLLM) body: ("json_schema", schema), ("json", None), or None."""

    rf = body.get("response_format")
    if isinstance(rf, dict):
        if rf.get("type") == "json_schema":
            js = rf.get("json_schema") or {}
            schema = js.get("schema", js.get("json_schema"))
            if schema is not None:
                return "json_schema", schema
        if rf.get("type") == "json_object":
            return "json", None
    if body.get("guided_json") is not None:
        return "json_schema", body["guided_json"]
    so = body.get("structured_outputs")
    if isinstance(so, dict) and so.get("json") is not None:
        return "json_schema", so["json"]
    return None


class Grammars:
    """Compiles and caches grammars for one tokenizer."""

    def __init__(self, model_dir: Path, vocab_size: int, eos: tuple[int, ...], cache: int = 64):
        import xgrammar as xgr
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(model_dir))
        info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab_size, stop_token_ids=list(eos))
        self.xgr = xgr
        self.compiler = xgr.GrammarCompiler(info, max_threads=8)
        self.vocab_size = vocab_size
        self.think_end = tok.convert_tokens_to_ids("</think>")
        self.compiled: OrderedDict[str, Any] = OrderedDict()
        self.limit = cache
        self.lock = threading.Lock()

    def _compile(self, kind: str, schema: Any):
        key = kind + ":" + (schema if isinstance(schema, str) else json.dumps(schema, sort_keys=True))
        with self.lock:
            hit = self.compiled.get(key)
            if hit is not None:
                self.compiled.move_to_end(key)
                return hit
        if kind == "json":
            grammar = self.compiler.compile_builtin_json_grammar()
        else:
            grammar = self.compiler.compile_json_schema(schema if isinstance(schema, str) else json.dumps(schema))
        with self.lock:
            self.compiled[key] = grammar
            while len(self.compiled) > self.limit:
                self.compiled.popitem(last=False)
        return grammar

    def constraint(self, spec: tuple[str, Any], *, after_think: bool = False) -> "Constraint":
        return Constraint(self.xgr, self.xgr.GrammarMatcher(self._compile(*spec)), self.vocab_size,
                          active=not after_think, think_end=self.think_end)


class Constraint:
    """One reply's grammar state at its committed position (the scheduler's ``constraint`` hooks)."""

    def __init__(self, xgr, matcher, vocab_size: int, *, active: bool, think_end: int):
        self.xgr, self.m, self.vocab = xgr, matcher, vocab_size
        self.active, self.think_end = active, think_end
        self.words = (vocab_size + 31) // 32

    def _apply(self, logits: torch.Tensor, bitmask: torch.Tensor, rows: list[int]) -> None:
        if not rows:
            return
        bits = bitmask[rows].to(logits.device, non_blocking=True)
        shifts = torch.arange(32, device=logits.device, dtype=torch.int32)
        allowed = ((bits.unsqueeze(-1) >> shifts) & 1).reshape(len(rows), -1)[:, :logits.shape[1]].bool()
        idx = torch.tensor(rows, device=logits.device)
        logits[idx] = logits[idx].masked_fill(~allowed, float("-inf"))

    def mask_first(self, logits: torch.Tensor) -> None:
        """The first reply token (the prompt's last row)."""

        if not self.active or self.m.is_terminated():
            return
        bitmask = self.xgr.allocate_token_bitmask(1, self.vocab)
        self.m.fill_next_token_bitmask(bitmask, 0)
        self._apply(logits, bitmask, [0])

    def mask_tree(self, tokens: list[int], parents: list[int], logits: torch.Tensor) -> None:
        """Row 0 is the last committed token (already followed); row r > 0 follows its parent."""

        rows = len(tokens)
        children: list[list[int]] = [[] for _ in range(rows)]
        for r in range(1, rows):
            children[parents[r]].append(r)
        bitmask = self.xgr.allocate_token_bitmask(rows, self.vocab)
        masked: list[int] = []

        def visit(r: int, active: bool) -> None:
            if active and self.m.is_terminated():
                return                            # after the stop token: the path ends here, the row is unused
            if active:
                self.m.fill_next_token_bitmask(bitmask, r)
                masked.append(r)
            for c in children[r]:
                if active:
                    if not self.m.accept_token(tokens[c]):
                        continue                  # unreachable: its parent's row cannot choose this token
                    visit(c, True)
                    self.m.rollback(1)
                else:
                    visit(c, tokens[c] == self.think_end)

        visit(0, self.active)
        self._apply(logits, bitmask, masked)

    def advance(self, tokens: list[int]) -> None:
        """Follow the accepted tokens (each was chosen under this grammar's mask)."""

        for t in tokens:
            if self.active:
                if self.m.is_terminated():
                    break
                if not self.m.accept_token(t):
                    raise RuntimeError(f"grammar rejected accepted token {t}")
            elif t == self.think_end:
                self.active = True

    def finished(self) -> bool:
        return self.active and self.m.is_terminated()
