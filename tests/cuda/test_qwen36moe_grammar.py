"""Qwen3.6-35B-A3B's JSON-schema replies on CUDA: drafted replies equal serial ones under the grammar, and parse."""

from __future__ import annotations

import gc
import hashlib
import http.client
import json
import random
import re
import threading
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
xgr = pytest.importorskip("xgrammar")

from tensorfold.cuda import grammar as G  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families import qwen3_5_moe as family  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda import decode as D  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.decode import Decoder, mtp_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.forward import prepare  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.mtp import prepare_mtp  # noqa: E402

from test_qwen36moe_forward import V, _cached, _first_difference, _tokens  # noqa: E402
from test_qwen36moe_mtp import _random, _same_state  # noqa: E402

SAMPLINGS = [None, Sampling(seed=7, top_k=20, top_p=0.95)]
EOS = V - 1                        # the random-weight model's eos


def label_schema(events: int = 4, labels: int = 8, facts: int = 40) -> dict:
    """The label A/B's shape: events [{e, labels [{i, k, m, t, a, o, d}]}] (synthetic enums and bounds)."""

    date = {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"}
    row = {"type": "object", "additionalProperties": False, "required": ["i", "k", "a"],
           "properties": {"i": {"type": "integer", "minimum": 0, "maximum": facts - 1},
                          "k": {"type": "string", "enum": ["plan", "decision", "request", "claim", "preference"]},
                          "m": {"type": "string", "enum": ["certain", "likely", "possible"]},
                          "t": {"type": "string", "enum": ["purchase", "booking", "refund", "transfer"]},
                          "a": {"type": "string", "enum": ["self", "other", "unknown"]},
                          "o": date, "d": date}}
    event = {"type": "object", "additionalProperties": False, "required": ["e", "labels"],
             "properties": {"e": {"type": "integer", "minimum": 0, "maximum": events - 1},
                            "labels": {"type": "array", "maxItems": labels, "items": row}}}
    return {"type": "object", "additionalProperties": False, "required": ["events"],
            "properties": {"events": {"type": "array", "maxItems": events, "items": event}}}


def valid(value, schema: dict) -> bool:
    """The JSON-schema subset ``label_schema`` uses."""

    kind = schema.get("type")
    if kind == "object":
        props = schema.get("properties", {})
        return (isinstance(value, dict) and all(k in value for k in schema.get("required", []))
                and (schema.get("additionalProperties", True) or set(value) <= set(props))
                and all(valid(v, props[k]) for k, v in value.items() if k in props))
    if kind == "array":
        return (isinstance(value, list) and len(value) <= schema.get("maxItems", len(value))
                and all(valid(v, schema["items"]) for v in value))
    if kind == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)
                and schema.get("minimum", value) <= value <= schema.get("maximum", value))
    if kind == "string":
        return (isinstance(value, str) and value in schema.get("enum", [value])
                and re.search(schema.get("pattern", ""), value) is not None)
    return False


def _sha(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(str(t) for t in tokens).encode()).hexdigest()


# -- random weights, a toy vocabulary -----------------------------------------------------------------------------------
def _toy_vocabulary() -> list[str]:
    """V tokens: ASCII characters (0 unused), JSON pieces, the schema's words and numbers, filler; eos (V - 1) last."""

    vocab = [""] + [chr(t) for t in range(1, 128)]
    words = ["events", "e", "labels", "i", "k", "m", "t", "a", "o", "d", "plan", "decision", "request", "claim",
             "preference", "certain", "likely", "possible", "purchase", "booking", "refund", "transfer", "self",
             "other", "unknown"]
    pieces = ['{"', '"}', '":', '",', '":"', '"},', '":[', '"]', '}]', '}]}', '[{', '},{', '", "', '": ', ', "',
              "\n  ", "    ", "2026", "-09", "-28", "2025-"]
    pieces += [f'"{w}"' for w in words] + [f'{w}"' for w in words] + [f'"{w}":' for w in words]
    pieces += [w for w in words if len(w) > 1] + [str(n) for n in range(10, 400)]
    seen = set(vocab)
    for p in pieces:
        if p not in seen:
            vocab.append(p)
            seen.add(p)
    rng = random.Random(0)
    letters = "abcdefghijklmnopqrstuvwxyz0123456789 _-"
    while len(vocab) < V - 1:
        p = "".join(rng.choice(letters) for _ in range(rng.randint(2, 4)))
        if p not in seen:
            vocab.append(p)
            seen.add(p)
    return vocab + [""]


@pytest.fixture(scope="module")
def toy():
    info = xgr.TokenizerInfo(_toy_vocabulary(), xgr.VocabType.RAW, vocab_size=V, stop_token_ids=[EOS])
    grammars = G.Grammars(info)
    small = {"type": "object", "additionalProperties": False, "required": ["e"],
             "properties": {"e": {"type": "integer", "minimum": 0, "maximum": 9},
                            "k": {"type": "string", "enum": ["plan", "claim"]}}}
    specs = {"labels": G.Spec("json_schema", json.dumps(label_schema(2, 3, 10))),
             "small": G.Spec("json_schema", json.dumps(small)), "json": G.Spec("json")}
    compiled = {name: grammars.compile(spec) for name, spec in specs.items()}
    return SimpleNamespace(grammars=grammars, compiled=compiled,
                           fresh=lambda name: grammars.constraint(compiled[name]))


@pytest.fixture(scope="module")
def rnd():
    w, mtpw = _random()
    m = prepare(w, release=False)
    return SimpleNamespace(w=w, mtpw=mtpw, m=m, k=prepare_mtp(mtpw, m, draft_vocab=None))


def _decoder(r, *, graphs: bool = False, states: int = 2) -> Decoder:
    return Decoder(r.m, capacity=1024, rows=512, window_rows=32, attn_rows=48, mtp_rows=16, states=states,
                   mtp=r.k, graphs=graphs)


def _serial(e: Decoder, prompt, count, sampling, constraint):
    first = prefill(e, prompt, sampling, mtp=False, constraint=constraint)
    if first == EOS:
        return D.DecodeResult([first], 0.0, 0)
    return serial_decode(e, first, count, sampling, constraint=constraint)


def _follows(toy, name: str, tokens: list[int]) -> bool:
    """Every token is one the grammar allows after the ones before it (the stop token last, if any)."""

    m = xgr.GrammarMatcher(toy.compiled[name])
    return all(m.accept_token(t) for t in tokens)


@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_drafted_constrained_decoding_emits_serial_tokens_eager_and_graphs(rnd, toy, sampling):
    eager, graphs = _decoder(rnd), _decoder(rnd, graphs=True)
    ended = 0
    for name, seed in (("labels", 31), ("small", 32), ("json", 33), ("small", 34)):
        prompt = _tokens(60, seed)
        ref = _serial(eager, prompt, 80, sampling, toy.fresh(name))
        plain = _serial(eager, prompt, 80, sampling, None)
        assert _follows(toy, name, ref.tokens) and ref.tokens != plain.tokens[:len(ref.tokens)]
        assert _serial(graphs, prompt, 80, sampling, toy.fresh(name)).tokens == ref.tokens
        ended += ref.tokens[-1] == EOS
        for depth, conf in ((1, 0.0), (3, 0.0), (6, 0.5)):
            for e in (eager, graphs):
                c = toy.fresh(name)
                first = prefill(e, prompt, sampling, constraint=c)
                got = mtp_decode(e, first, 80, sampling, depth=depth, confidence=conf, constraint=c)
                assert got.tokens == ref.tokens, (name, depth, conf, _first_difference(got.tokens, ref.tokens))
                assert got.committed == ref.tokens[:-1] and max(got.widths) <= depth + 1
                assert c.finished == (ref.tokens[-1] == EOS)
    assert ended >= 1                                   # a reply the grammar ended, besides max_tokens ones


@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_an_oracle_drafter_under_the_grammar_keeps_long_prefixes_and_ends_in_serial_state(rnd, toy, sampling,
                                                                                         monkeypatch):
    """Right drafts, rejected ones and a drafted stop token: serial tokens, serial caches, partial keeps."""

    e = _decoder(rnd, graphs=True)
    prompt = _tokens(50, 41)
    ref = _serial(e, prompt, 60, sampling, toy.fresh("small"))
    if ref.tokens[-1] != EOS:
        prompt, ref = next((p, r) for p, r in ((q, _serial(e, q, 60, sampling, toy.fresh("small")))
                                               for q in (_tokens(50, s) for s in range(42, 90)))
                           if r.tokens[-1] == EOS)
    ser = e.pool.clone(e.st)
    real = D.draft
    calls, bad = [], {"rejected": 0, "wrong": 0}

    def oracle(e_, st, hidden, next_tokens, position, count, sampling_, confidence=0.0):
        real(e_, st, hidden, next_tokens, position, count, sampling_, confidence)    # the head's bookkeeping
        base = position - len(prompt)
        out = [ref.tokens[base + j] if base + j < len(ref.tokens) else EOS for j in range(count)]
        calls.append(len(calls))
        if len(calls) % 3 == 0 and out:                 # a draft the grammar rejects ("x"), or a wrong digit it takes
            at = len(calls) % len(out)
            out[at] = ord("x") if len(calls) % 2 else (ord("7") if out[at] != ord("7") else ord("8"))
        return out

    seen = []
    admissible = G.Constraint.admissible

    def spy(self, drafts):
        kept = admissible(self, drafts)
        seen.append((list(drafts), kept))
        return kept

    monkeypatch.setattr(D, "draft", oracle)
    monkeypatch.setattr(G.Constraint, "admissible", spy)
    for depth in (2, 3, 4, 5, 6):
        calls.clear()
        c = toy.fresh("small")
        first = prefill(e, prompt, sampling, constraint=c)
        heard: list[int] = []
        got = mtp_decode(e, first, 60, sampling, depth=depth, confidence=0.0, on_tokens=heard.extend, constraint=c)
        assert got.tokens == ref.tokens and [first] + heard == ref.tokens, depth
        assert max(got.keeps) >= 3 and got.accepted >= got.rounds, (depth, got.keeps)
        assert got.committed == ref.tokens[:-1] and c.finished
        assert _same_state(e.st, ser), depth
    assert any(len(k) < len(d) and d[len(k)] == ord("x") for d, k in seen)           # rejected drafts dropped
    assert any(len(k) < len(d) and d[len(k)] == EOS for d, k in seen), (len(ref.tokens), seen)   # a drafted stop
    assert any(len(k) == len(d) for d, k in seen)


@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_the_engine_serves_constrained_requests_as_its_serial_switch(rnd, toy, sampling):
    eng = Qwen36Engine(None, model=rnd.m, mtp=rnd.k, context=1024, mtp_drafts=4, confidence=0.0)
    try:
        def ask(prompt, name, **kw):
            got: list[int] = []
            stats = eng.generate(prompt, 60, sampling, got.extend, constraint=toy.fresh(name) if name else None,
                                 **kw)
            return got, stats

        prompt = _tokens(80, 51)
        plain, _ = ask(prompt, None)
        for name in ("labels", "small", "json"):
            drafted, stats = ask(prompt, name)
            serial, ss = ask(prompt, name, draft=False)
            assert _sha(drafted) == _sha(serial), (name, _first_difference(drafted, serial))
            assert stats["drafts"] is True and ss["drafts"] is False and _follows(toy, name, drafted)
            follow = prompt + drafted[:-1] + _tokens(5, 52)                 # resumed after the kept reply
            got, fs = ask(follow, name)
            assert fs["cached"] > len(prompt) and got == ask(follow, name, draft=False)[0]
        again, _ = ask(prompt, None)                                        # no grammar left behind
        assert again == plain
    finally:
        eng.e = None
        gc.collect()
        torch.cuda.empty_cache()


# -- the real checkpoint behind the server --------------------------------------------------------------------------
EVENTS = """Events (synthetic):
0. Priya Nandakumar emailed the Harbourview hotel on 2026-09-02 to book two nights from 2026-10-14.
1. Tomas Rivera asked the bike shop for a refund of the helmet he bought on 2026-08-30.
2. The team agreed that Lena Fischer will present the quarterly results on 2026-11-05.

Facts:
0. Priya booked a hotel stay starting 2026-10-14.
1. Priya prefers rooms facing the harbour.
2. Tomas requested a refund for a helmet.
3. Tomas bought the helmet on 2026-08-30.
4. Lena Fischer will present the quarterly results.
5. The presentation is due on 2026-11-05."""

LABEL_PROMPT = (EVENTS + "\n\nLabel every fact. Reply with compact JSON only: "
                '{"events":[{"e":<event number>,"labels":[{"i":<fact number>,"k":<kind>,"m":<modality>,'
                '"t":<transaction type>,"a":<authority>,"o":<occurs_at YYYY-MM-DD>,"d":<due_at YYYY-MM-DD>}]}]}. '
                "Kinds: plan, decision, request, claim, preference. Modality: certain, likely, possible. "
                "Transaction types: purchase, booking, refund, transfer. Authority: self, other, unknown. "
                'Omit "m", "t", "o" and "d" when they do not apply.')
OTHER_PROMPTS = ["List two synthetic events about a library book club as JSON with the same fields: one event, "
                 "two labels.",
                 "Label this single fact: 0. Ana plans to buy a tram pass next Monday."]
REAL_SAMPLINGS = {"greedy": {"temperature": 0},
                  "sampled": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "seed": 20260928}}


class _Spy:
    """Records each request's prompt, sampling, draft switch, constraint and tokens."""

    def __init__(self, eng) -> None:
        self.calls: list[SimpleNamespace] = []
        self.inner = eng.generate

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None):
        got: list[int] = []

        def tap(new):
            got.extend(new)
            return on_tokens(new)

        stats = self.inner(prompt, max_tokens, sampling, tap, draft=draft, constraint=constraint)
        self.calls.append(SimpleNamespace(prompt=list(prompt), sampling=sampling, draft=draft,
                                          constraint=constraint, tokens=got, stats=stats))
        return stats


@pytest.fixture(scope="module")
def served():
    snap = _cached(family.MODELS[0])
    from http.server import ThreadingHTTPServer

    from tensorfold import families
    from tensorfold.cli import _drafter, _generation_config
    from tensorfold.cuda.server import App, make_handler

    gc.collect()
    torch.cuda.empty_cache()
    drafter = _drafter(families.families()["qwen3_5_moe"], "auto")
    eng = family.cuda_engine(snap, drafter=drafter, tp=1, rank=0, master="", master_port=29551, no_drafts=False)
    spy = _Spy(eng)
    eng.generate = spy.generate
    app = App(eng, snap, "Qwen3.6-35B-A3B-4bit", default_thinking=False, sampling=_generation_config(snap),
              max_tokens=4096)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    ns = SimpleNamespace(snap=snap, eng=eng, app=app, spy=spy, port=server.server_address[1])
    del eng, app
    yield ns
    server.shutdown()
    server.server_close()
    ns.eng.generate = spy.inner
    ns.eng.e = None
    ns.eng = ns.app = ns.spy = None
    gc.collect()
    torch.cuda.empty_cache()


def _post(port: int, body: dict) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return resp.status, data


def _body(text: str, sampling: dict, **extra) -> dict:
    return {"model": "Qwen3.6-35B-A3B-4bit", "messages": [{"role": "user", "content": text}], "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False}, **sampling, **extra}


def test_real_label_schema_replies_parse_validate_and_drafted_equals_serial(served):
    eng, spy, app = served.eng, served.spy, served.app
    if not eng.depth:
        pytest.skip(f"{family.DRAFTER} is not in the Hugging Face cache: the engine serves without drafts")
    schema = label_schema()
    rf = {"type": "json_schema", "json_schema": {"name": "labels", "strict": True, "schema": schema}}
    for sname, sampling in REAL_SAMPLINGS.items():
        for text in [LABEL_PROMPT] + OTHER_PROMPTS:
            status, drafted = _post(served.port, _body(text, sampling, response_format=rf))
            c_dr = spy.calls[-1]
            status_s, serial = _post(served.port, _body(text, sampling, response_format=rf, draft=False))
            c_se = spy.calls[-1]
            assert status == status_s == 200, (drafted, serial)
            ref = D.generate(eng.e, c_dr.prompt, 512, c_dr.sampling, stop_eos=True, st=eng.serial, draft=False,
                             constraint=app._grammars().constraint(app._grammars().compile(G.request_spec(
                                 {"response_format": rf}))))
            assert _sha(c_dr.tokens) == _sha(c_se.tokens) == _sha(ref), (sname, text[:30],
                                                                         _first_difference(c_dr.tokens, ref))
            assert c_dr.stats["drafts"] is True and c_se.stats["drafts"] is False and c_dr.stats["drafted"] > 0
            assert isinstance(c_dr.constraint, G.Constraint) and c_dr.constraint.finished
            content = drafted["choices"][0]["message"]["content"]
            assert content == serial["choices"][0]["message"]["content"]
            assert drafted["choices"][0]["finish_reason"] == "stop" and ref[-1] in eng.eos
            value = json.loads(content)
            assert valid(value, schema), content
            if text == LABEL_PROMPT:
                assert value["events"] and any(ev["labels"] for ev in value["events"]), content


def test_real_json_object_and_unconstrained_replies(served):
    eng, spy = served.eng, served.spy
    sampling = REAL_SAMPLINGS["greedy"]
    status, plain = _post(served.port, _body(LABEL_PROMPT, sampling))
    first_plain = spy.calls[-1].tokens
    assert status == 200 and spy.calls[-1].constraint is None
    for text in (LABEL_PROMPT, "Give a JSON object describing a synthetic bicycle with three fields."):
        status, drafted = _post(served.port, _body(text, sampling, response_format={"type": "json_object"}))
        c_dr = spy.calls[-1]
        status_s, serial = _post(served.port, _body(text, sampling, response_format={"type": "json_object"},
                                                    draft=False))
        assert status == status_s == 200 and _sha(c_dr.tokens) == _sha(spy.calls[-1].tokens)
        assert isinstance(json.loads(drafted["choices"][0]["message"]["content"]), dict)
    status, again = _post(served.port, _body(LABEL_PROMPT, sampling))            # nothing left behind
    assert status == 200 and spy.calls[-1].tokens == first_plain and again["choices"] == plain["choices"]


def test_real_bad_schema_is_a_400_and_the_server_keeps_serving(served):
    n = len(served.spy.calls)
    for rf in ({"type": "json_schema", "json_schema": {"name": "v", "schema": {"type": "nonsense"}}},
               {"type": "json_schema", "json_schema": {"name": "v"}}, {"type": "xml"}):
        for stream in (False, True):
            status, data = _post(served.port, _body("Hi", REAL_SAMPLINGS["greedy"], response_format=rf,
                                                    stream=stream))
            assert status == 400 and data["error"]["type"] == "invalid_request_error", data
    assert len(served.spy.calls) == n
    status, data = _post(served.port, _body("Say hi.", REAL_SAMPLINGS["greedy"], max_tokens=8))
    assert status == 200 and data["choices"][0]["message"]["content"]
