"""Qwen3.6-35B-A3B's JSON-schema replies on CUDA: drafted equals serial, concurrent equals solo, and the replies validate.

Needs ``TENSORFOLD_QWEN36_MODEL=<Vontra/Qwen3.6-35B-A3B-MLX-4bit-MTP dir>``, xgrammar (``pip install
'tensorfold[grammar]'``) and jsonschema; skipped otherwise. About 24 GB of GPU memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
pytest.importorskip("xgrammar")
jsonschema = pytest.importorskip("jsonschema")

from tensorfold.cuda import grammar  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

MODEL = os.environ.get("TENSORFOLD_QWEN36_MODEL", "")
CONTEXT = 8192
SAMPLINGS = {"greedy": None, "seed7": Sampling(7, 1.0, 20, 0.95)}

# public, made-up schemas and prompts
PERSON = {"type": "object", "additionalProperties": False, "required": ["name", "born", "fields"],
          "properties": {"name": {"type": "string"}, "born": {"type": "integer", "minimum": 1000, "maximum": 2100},
                         "fields": {"type": "array", "maxItems": 3,
                                    "items": {"type": "string", "enum": ["mathematics", "computing", "physics",
                                                                         "poetry", "music"]}}}}
WEATHER = {"type": "object", "additionalProperties": False, "required": ["city", "celsius", "sky", "wind_kph"],
           "properties": {"city": {"type": "string"}, "celsius": {"type": "number"},
                          "sky": {"type": "string", "enum": ["sunny", "cloudy", "rain", "snow"]},
                          "wind_kph": {"type": "integer", "minimum": 0, "maximum": 200}}}
EVENT = {"type": "object", "additionalProperties": False, "required": ["title", "date", "attendees"],
         "properties": {"title": {"type": "string"},
                        "date": {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"},
                        "attendees": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
                        "online": {"type": "boolean"}}}
CASES = {
    "person": (PERSON, "Describe Ada Lovelace as a short JSON record."),
    "weather": (WEATHER, "Make up a plausible weather report for Lisbon in spring, as JSON."),
    "event": (EVENT, "Invent a small team meeting next month and describe it as JSON."),
}


def _sha(ids: list[int]) -> str:
    return hashlib.sha256(",".join(str(int(t)) for t in ids).encode()).hexdigest()


@pytest.fixture(scope="module")
def engine():
    if not (MODEL and Path(MODEL).is_dir()):
        pytest.skip("needs TENSORFOLD_QWEN36_MODEL")
    from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine

    return Qwen36Engine(Path(MODEL), context=CONTEXT, context_explicit=True)


@pytest.fixture(scope="module")
def chat(engine):
    from tokenizers import Tokenizer

    from tensorfold.cuda.server import ChatTemplate

    tok = Tokenizer.from_file(str(Path(MODEL) / "tokenizer.json"))
    template = ChatTemplate(Path(MODEL))
    grammars = grammar.for_model(MODEL, grammar.vocab_size(MODEL), engine.eos)
    compiled = {name: grammars.compile(grammar.Spec("json_schema", json.dumps(schema)))
                for name, (schema, _) in CASES.items()}

    def prompt(text: str, thinking: bool = False) -> list[int]:
        rendered = template.render([{"role": "user", "content": text}], tools=None, enable_thinking=thinking)
        return tok.encode(rendered, add_special_tokens=False).ids

    def fresh(name: str, thinking: bool = False):
        return grammars.constraint(compiled[name], after_think=thinking)

    return type("Chat", (), {"prompt": staticmethod(prompt), "fresh": staticmethod(fresh), "tok": tok})


def _run(engine, prompt, count, sampling, draft, constraint=None):
    out: list[int] = []
    stats = engine.generate(list(prompt), count, sampling, lambda new: out.extend(new) and False, draft=draft,
                            constraint=constraint)
    return out, stats


def _answer(chat, ids: list[int], engine) -> str:
    text = chat.tok.decode([t for t in ids if t not in engine.eos], skip_special_tokens=False)
    return text.split("</think>")[-1]


@pytest.mark.parametrize("sampling", list(SAMPLINGS), ids=list(SAMPLINGS))
@pytest.mark.parametrize("name", list(CASES))
def test_constrained_drafted_replies_equal_serial_and_validate(engine, chat, name, sampling):
    schema, text = CASES[name]
    prompt = chat.prompt(text)
    serial, s_stats = _run(engine, prompt, 320, SAMPLINGS[sampling], False, chat.fresh(name))
    drafted, d_stats = _run(engine, prompt, 320, SAMPLINGS[sampling], True, chat.fresh(name))
    assert _sha(drafted) == _sha(serial), (name, sampling)
    assert serial[-1] in engine.eos, "the JSON value completed within the reply limit"
    jsonschema.validate(json.loads(_answer(chat, serial, engine)), schema)
    assert d_stats["rounds"] < s_stats["rounds"]                     # drafting kept tokens under the grammar


@pytest.mark.parametrize("sampling", list(SAMPLINGS), ids=list(SAMPLINGS))
def test_unconstrained_drafted_replies_still_equal_serial(engine, chat, sampling):
    prompt = chat.prompt(CASES["weather"][1])
    serial, _ = _run(engine, prompt, 160, SAMPLINGS[sampling], False)
    drafted, _ = _run(engine, prompt, 160, SAMPLINGS[sampling], True)
    assert _sha(drafted) == _sha(serial)


def test_with_thinking_the_schema_holds_after_think_end(engine, chat):
    prompt = chat.prompt("Think briefly, then answer. " + CASES["weather"][1], thinking=True)
    sampling = SAMPLINGS["seed7"]
    serial, _ = _run(engine, prompt, 2000, sampling, False, chat.fresh("weather", thinking=True))
    drafted, _ = _run(engine, prompt, 2000, sampling, True, chat.fresh("weather", thinking=True))
    assert _sha(drafted) == _sha(serial)
    think_end = chat.tok.token_to_id("</think>")
    if think_end in serial and serial[-1] in engine.eos:          # the model finished thinking within the limit
        jsonschema.validate(json.loads(_answer(chat, serial, engine)), WEATHER)


def test_concurrent_constrained_replies_equal_their_solo_runs(engine, chat):
    """Constrained and plain, drafted and serial requests share rounds (a --parallel 4 decoder on the engine's weights
    and graphs); each reply equals its solo run through the same decoder and its serial reply on the one-stream
    engine."""

    from tensorfold.cuda.scheduler import Scheduler
    from tensorfold.families.qwen3_5_moe.cuda.engine import KEEP_MANY
    from tensorfold.families.qwen3_5_moe.cuda.multi import MultiDecoder

    multi = MultiDecoder(engine.w, engine.head, depth=engine.depth, confidence=engine.confidence, context=4096,
                         keep=KEEP_MANY, points=engine.points, graphs=engine.graphs)
    scheduler = Scheduler(multi, max_streams=4)
    requests = [(name, sampling, True) for name in CASES for sampling in SAMPLINGS]
    requests += [(None, "greedy", True), (None, "seed7", True), ("weather", "seed7", False)]

    def submit(request):
        name, sampling, draft = request
        text = CASES[name or "event"][1]
        out: list[int] = []
        scheduler.submit(chat.prompt(text), 240, SAMPLINGS[sampling], draft, lambda new: out.extend(new) and False,
                         constraint=chat.fresh(name) if name else None)
        return out

    together: list = [None] * len(requests)

    def worker(i):
        together[i] = submit(requests[i])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(requests))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    solos = [submit(request) for request in requests]
    assert multi.resident is None                      # the engine's graphs are free again for its own requests
    for request, got, solo in zip(requests, together, solos):
        name, sampling, _ = request
        assert _sha(got) == _sha(solo), request
        serial, _ = _run(engine, chat.prompt(CASES[name or "event"][1]), 240, SAMPLINGS[sampling], False,
                         chat.fresh(name) if name else None)
        assert _sha(solo) == _sha(serial), request
        if name is not None and got[-1] in engine.eos:
            jsonschema.validate(json.loads(_answer(chat, got, engine)), CASES[name][0])
