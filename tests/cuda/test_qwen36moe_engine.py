"""Qwen3.6-35B-A3B's CUDA engine (``qwen3_5_moe/cuda/engine.py``): what ``tensorfold serve`` runs.

Small random weights (the MTP test's model: the real per-layer shapes, eight layers, 32 experts, a random MTP head), for
three engines: serial (no head), drafted as served (up to 6 drafts, the 50% chain stop) and a deep chain (4 drafts, no
stop), all with CUDA graphs: the engine streams exactly serial decoding's tokens, greedy and sampled; an eos id ends
the reply and ``on_tokens`` returning True stops it; a prompt that extends the last prompt or reply resumes from the
kept state and decodes what a fresh prefill decodes; ``draft=False`` decodes one token a round from a fresh prefill
and leaves the kept states alone; the context limit; the memory plan covers what the engine allocates, and refuses,
before loading anything, when memory is short.

The real checkpoint and drafter (skipped when not in the Hugging Face cache): the engine built as ``tensorfold serve``
builds it with no flags, behind the upstream server in-process. OpenAI chat requests with thinking off: the served
(drafted) tokens equal the engine's serial decoding of the same prompt by SHA-256 for three chat and two JSON prompts,
greedy and sampled, as do ``"draft": false`` replies; streamed and non-streamed replies agree; a follow-up resumed from
the kept reply equals a fresh prefill; drafted against serial speed is reported (relative only).
"""

from __future__ import annotations

import gc
import hashlib
import http.client
import json
import threading
import time
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families import qwen3_5_moe as family  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda import decode  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.engine import (  # noqa: E402
    GiB, Qwen36Engine, drafter_bytes, memory_plan, state_bytes, weight_bytes)
from tensorfold.families.qwen3_5_moe.cuda.forward import prepare  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.mtp import prepare_mtp  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.weights import Config  # noqa: E402

from test_qwen36moe_forward import _cached, _first_difference, _tokens  # noqa: E402
from test_qwen36moe_mtp import _random  # noqa: E402

SAMPLINGS = [None, Sampling(seed=7, top_k=20, top_p=0.95)]
ENGINES = {"serial": {}, "drafted": {"mtp_drafts": None}, "deep": {"mtp_drafts": 4, "confidence": 0.0}}


@pytest.fixture(scope="module")
def rnd():
    w, mtpw = _random()
    m = prepare(w)
    return SimpleNamespace(m=m, k=prepare_mtp(mtpw, m, draft_vocab=None))


@pytest.fixture(scope="module", params=list(ENGINES))
def engine(request, rnd):
    kw = ENGINES[request.param]
    eng = Qwen36Engine(None, model=rnd.m, mtp=None if request.param == "serial" else rnd.k, capacity=1024, **kw)
    assert eng.depth == {"serial": 0, "drafted": decode.DEPTH, "deep": 4}[request.param]
    assert (eng.e.mtp is not None) == (eng.depth > 0) and eng.e.graphs is not None
    yield eng
    eng.e = None
    gc.collect()
    torch.cuda.empty_cache()


def _ask(eng, prompt, n, sampling, **kw):
    got: list[int] = []
    stats = eng.generate(prompt, n, sampling, lambda new: got.extend(new), **kw)
    return got, stats


def _reference(eng, prompt, n, sampling):
    """Serial decoding straight from ``decode``: a fresh prefill in the serial state, then one token a step."""

    return decode.generate(eng.e, prompt, n, sampling, stop_eos=True, st=eng.serial, draft=False)


def _sha(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(str(t) for t in tokens).encode()).hexdigest()[:16]


# -- random weights -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_generate_streams_serial_decoding_and_stops(engine, rnd, sampling):
    prompt = _tokens(40, 21)
    ref = _reference(engine, prompt, 30, sampling)
    assert len(ref) == 30 and len(set(ref)) > 5
    got, stats = _ask(engine, prompt, 30, sampling)
    assert got == ref, _first_difference(got, ref)
    assert (stats["prompt_tokens"], stats["cached"], stats["completion_tokens"]) == (40, 0, 30)
    assert stats["decode_tps"] > 0 and stats["prefill_s"] > 0
    if engine.depth:
        assert stats["drafts"] is True and stats["drafted"] >= stats["rounds"] - 1 and stats["rounds"] <= 29
    else:
        assert stats["drafts"] is False and stats["rounds"] == 29
    # on_tokens returning True stops the decode: after the round that brings the third token, and after the first
    seen: list[int] = []
    stats = engine.generate(prompt, 30, sampling, lambda new: (seen.extend(new), len(seen) >= 3)[1])
    assert 3 <= len(seen) <= 3 + engine.depth and seen == ref[:len(seen)] and stats["completion_tokens"] == len(seen)
    seen = []
    stats = engine.generate(prompt, 30, sampling, lambda new: (seen.extend(new), True)[1])
    assert seen == ref[:1] and stats["completion_tokens"] == 1
    # an eos id ends the reply, and is its last token (in the middle of a drafted round too)
    for at in (6, 13):
        j = ref.index(ref[at])
        kept = rnd.m.cfg.eos
        rnd.m.cfg.eos = (ref[at],) + tuple(kept)
        try:
            assert ref[at] in engine.eos
            got, stats = _ask(engine, prompt, 30, sampling)
            assert got == ref[:j + 1] and stats["completion_tokens"] == j + 1, at
        finally:
            rnd.m.cfg.eos = kept


@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_prefix_reuse_equals_fresh_prefills_and_the_serial_switch_leaves_it(engine, sampling):
    first = _tokens(600, 31)                            # two prefill chunks (512 + 88 rows)
    reply, stats = _ask(engine, first, 16, sampling)
    assert stats["cached"] == 0 and len(reply) == 16
    for extend, tail in (("reply", 5), ("prompt", 40)):  # a 5-row window and a 40-row chunk after the kept state
        if extend == "prompt":
            _ask(engine, first, 16, sampling)            # the first request's states again
        prompt = first + (reply if extend == "reply" else []) + _tokens(tail, 32)
        warm, ws = _ask(engine, prompt, 16, sampling)
        want = len(first) + len(reply) - 1 if extend == "reply" else len(first)
        assert ws["cached"] == want and ws["prompt_tokens"] == len(prompt), (extend, ws)
        ref = _reference(engine, prompt, 16, sampling)
        assert warm == ref, (extend, _first_difference(warm, ref))
        serial, ss = _ask(engine, prompt, 16, sampling, draft=False)    # a fresh prefill in the other state
        assert serial == warm and ss["cached"] == 0 and ss["drafts"] is False and ss["rounds"] == 15
        again, ag = _ask(engine, prompt + [9], 16, sampling)            # the kept states survived it
        assert ag["cached"] >= len(prompt) and again == _reference(engine, prompt + [9], 16, sampling)
        _ask(engine, _tokens(7, 33), 4, sampling)                      # an unrelated prompt: nothing to resume
        cold, cs = _ask(engine, prompt, 16, sampling)
        assert cs["cached"] == 0 and cold == warm, extend
    # a reply stopped early keeps the tokens it committed: a follow-up resumes after them
    seen: list[int] = []
    engine.generate(first, 16, sampling, lambda new: (seen.extend(new), len(seen) >= 3)[1])
    follow = first + seen + _tokens(3, 34)
    got, fs = _ask(engine, follow, 12, sampling)
    assert fs["cached"] == len(first) + len(seen) - 1 and got == _reference(engine, follow, 12, sampling)


def test_the_context_limit(engine):
    with pytest.raises(ValueError, match="no room"):
        engine.generate(_tokens(1024 - engine.depth, 40), 5, None, None)
    with pytest.raises(ValueError, match="at least one token"):
        engine.generate([], 5, None, None)
    got, stats = _ask(engine, _tokens(1000, 41), 100, None)        # the reply fits the room that is left
    room = 24 - engine.depth
    assert len(got) == room and stats["completion_tokens"] == room
    assert got == _reference(engine, _tokens(1000, 41), room, None)


def test_memory_plan_covers_what_the_engine_allocates(engine, rnd):
    m, cfg = rnd.m, rnd.m.cfg
    mtp = 1 if engine.depth else 0
    plan = memory_plan(cfg, engine.capacity, rows=engine.e.buf.rows, mtp_layers=mtp,
                       drafter=(0, 0, engine.e.mtp.head.n) if mtp else None, mtp_rows=engine.e.mbuf.rows if mtp else 64)
    one, snap = state_bytes(cfg, engine.capacity, mtp)
    assert one == engine.e.pool.nbytes_per_seq() + mtp * cfg.hidden * 2 and plan.states == 2 * one
    s = engine.e.st.snapshot()
    tail = s["mtp"][2]
    assert snap == s["rec"].numel() * 4 + s["conv"].numel() * 2 + (tail.numel() * 2 if tail is not None else 0)
    actual = engine.e.buf.nbytes() + (engine.e.mbuf.nbytes() if mtp else 0)
    assert actual <= plan.buffers <= 1.3 * actual + (8 << 20), (plan.buffers, actual)
    weights, _ = weight_bytes(cfg)
    assert m.nbytes() <= weights <= 1.03 * m.nbytes(), (weights, m.nbytes())


def test_bad_settings_are_refused(rnd):
    with pytest.raises(ValueError, match="0 to 15"):
        Qwen36Engine(None, model=rnd.m, mtp=rnd.k, mtp_drafts=16, capacity=256)
    with pytest.raises(ValueError, match="come from the drafter"):
        Qwen36Engine(None, model=rnd.m, mtp_drafts=3, capacity=256)
    with pytest.raises(ValueError, match="one GPU"):
        Qwen36Engine(None, model=rnd.m, tp=2)
    eng = Qwen36Engine(None, model=rnd.m, mtp=rnd.k, mtp_drafts=0, capacity=256, warm=False)   # 0: serial
    assert eng.depth == 0 and eng.e.mtp is None
    eng = Qwen36Engine(None, model=rnd.m, mtp=rnd.k, no_drafts=True, capacity=256, warm=False)
    assert eng.depth == 0 and eng.e.mtp is None


def test_the_memory_check_refuses_before_loading(tmp_path):
    from test_qwen36moe_package import _config, _drafter_config, _write

    folder = _write(tmp_path / "main", _config())       # the real model's and drafter's configs, no weights
    drafter = _write(tmp_path / "mtp", _drafter_config(), generation=None)
    cfg = Config.read(folder)
    weights, peak = weight_bytes(cfg)
    assert 18.0 * GiB < weights < 18.6 * GiB and peak < 0.6 * GiB, (weights / GiB, peak / GiB)
    head, head_peak, rows = drafter_bytes(drafter, cfg)
    assert rows == 76882 and 530 < head / 2 ** 20 < 550 and head_peak < 0.8 * GiB, (head / 2 ** 20, head_peak)
    plan = memory_plan(cfg, 32768, rows=512, mtp_layers=1, drafter=(head, head_peak, rows))
    print(f"\nplan at 32,768 tokens with the MTP head: {plan.describe()}; total {plan.total / GiB:.2f} GiB")
    before = torch.cuda.memory_allocated()
    with pytest.raises(ValueError, match=r"needs about \d+\.\d GiB .* 16\.0 GiB is free \(given\)"):
        Qwen36Engine(folder, drafter=str(drafter), free_memory=16 * GiB)
    with pytest.raises(ValueError, match="shorter --context"):
        Qwen36Engine(folder, drafter=str(drafter), capacity=262144, free_memory=int(plan.total))
    with pytest.raises(ValueError, match="shorter --context"):   # the head is counted: the plan without it is short
        Qwen36Engine(folder, drafter=str(drafter), free_memory=int(plan.total) - head // 2)
    assert torch.cuda.memory_allocated() == before
    with pytest.raises(FileNotFoundError):              # with room, it goes on to load (and finds no weights)
        Qwen36Engine(folder, drafter=str(drafter), free_memory=plan.total + GiB, warm=False)
    with pytest.raises(ValueError, match="one GPU"):
        Qwen36Engine(folder, tp=2)


# -- the real checkpoint behind the server --------------------------------------------------------------------------
class _Spy:
    """Wraps the engine's ``generate`` to record each request's prompt, sampling and streamed tokens."""

    def __init__(self, eng) -> None:
        self.calls: list[SimpleNamespace] = []
        self.inner = eng.generate

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        got: list[int] = []

        def tap(new):
            got.extend(new)
            return on_tokens(new)

        stats = self.inner(prompt, max_tokens, sampling, tap, draft=draft)
        self.calls.append(SimpleNamespace(prompt=list(prompt), sampling=sampling, draft=draft, tokens=got,
                                          stats=stats))
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
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    # what `tensorfold serve mlx-community/Qwen3.6-35B-A3B-4bit` passes with no flags
    drafter = _drafter(families.families()["qwen3_5_moe"], "auto")
    eng = family.cuda_engine(snap, drafter=drafter, tp=1, rank=0, master="", master_port=29551, no_drafts=False)
    built_s = time.time() - t0
    spy = _Spy(eng)
    eng.generate = spy.generate
    app = App(eng, snap, "Qwen3.6-35B-A3B-4bit", default_thinking=False, sampling=_generation_config(snap),
              max_tokens=4096)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"\nengine built in {built_s:.0f} s (drafter {'present' if drafter else 'absent'}, depth {eng.depth}), "
          f"capacity {eng.capacity}; plan {eng.plan.describe()}")
    ns = SimpleNamespace(snap=snap, eng=eng, app=app, spy=spy, port=server.server_address[1], before=before)
    del eng, app
    yield ns
    server.shutdown()
    server.server_close()
    del server, thread
    ns.eng.generate = spy.inner
    ns.eng.e = None                 # the next test module loads the checkpoint again: free this one first
    ns.eng = ns.app = ns.spy = None
    del spy
    gc.collect()
    torch.cuda.empty_cache()


def _post(port: int, body: dict) -> dict:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    assert resp.status == 200, data
    return json.loads(data)


def _post_stream(port: int, body: dict) -> dict:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    conn.request("POST", "/v1/chat/completions", json.dumps({**body, "stream": True,
                                                              "stream_options": {"include_usage": True}}),
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    assert resp.status == 200
    content, reasoning, finish, usage = [], [], None, None
    for raw in resp.read().decode().split("\n\n"):
        line = raw.strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[len("data: "):])
        choice = chunk["choices"][0]
        content.append(choice["delta"].get("content") or "")
        reasoning.append(choice["delta"].get("reasoning_content") or "")
        finish = choice["finish_reason"] or finish
        usage = chunk.get("usage") or usage
    conn.close()
    return {"content": "".join(content), "reasoning": "".join(reasoning), "finish": finish, "usage": usage}


QUESTION = "Name three things a lighthouse keeper did every night, one short sentence each."


@pytest.mark.parametrize("sampled", [False, True])
def test_real_server_streams_the_engines_serial_decoding(served, sampled):
    eng, app, spy = served.eng, served.app, served.spy
    body = {"model": "Qwen3.6-35B-A3B-4bit", "messages": [{"role": "user", "content": QUESTION}], "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update({"temperature": 1.0, "top_k": 20, "top_p": 0.95, "seed": 1234} if sampled else {"temperature": 0})
    plain = _post(served.port, body)
    c_plain = spy.calls[-1]
    streamed = _post_stream(served.port, body)
    c_stream = spy.calls[-1]
    serial = _post(served.port, {**body, "draft": False})
    c_serial = spy.calls[-1]
    prompt = app.tok.encode(app.template.render(body["messages"], tools=None, enable_thinking=False),
                            add_special_tokens=False).ids
    assert c_plain.prompt == prompt and c_stream.prompt == prompt and c_serial.prompt == prompt
    assert c_plain.sampling == (Sampling(1234, 1.0, 20, 0.95) if sampled else None)
    assert (c_plain.draft, c_stream.draft, c_serial.draft) == (True, True, False)
    ref = decode.generate(eng.e, prompt, 64, c_plain.sampling, stop_eos=True, st=eng.serial, draft=False)
    for name, call in (("non-streamed", c_plain), ("streamed", c_stream), ('"draft": false', c_serial)):
        assert call.tokens == ref, (name, _first_difference(call.tokens, ref))
    text = app.tok.decode([t for t in ref if t not in eng.eos], skip_special_tokens=False)
    message = plain["choices"][0]["message"]
    print(f"\n{'sampled' if sampled else 'greedy'}: {len(ref)} tokens, served == serial decoding; stats "
          f"{plain['tensorfold']}\n{text!r}")
    assert message["content"] == text and serial["choices"][0]["message"]["content"] == text
    assert streamed["content"] == text and streamed["reasoning"] == "" and "reasoning_content" not in message
    finish = "stop" if ref[-1] in eng.eos else "length"
    assert plain["choices"][0]["finish_reason"] == streamed["finish"] == finish
    assert plain["usage"]["completion_tokens"] == streamed["usage"]["completion_tokens"] == len(ref)
    assert plain["usage"]["prompt_tokens"] == len(prompt)
    assert plain["tensorfold"]["drafts"] is bool(eng.depth) and serial["tensorfold"]["drafts"] is False
    assert serial["tensorfold"]["cached"] == 0
    assert "</think>" not in text and "<think>" not in text


@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_real_follow_up_resumed_from_the_kept_reply_equals_a_fresh_prefill(served, sampling):
    eng, app, spy = served.eng, served.app, served.spy
    prompt = app.tok.encode(app.template.render([{"role": "user", "content": QUESTION}], tools=None,
                                                enable_thinking=False), add_special_tokens=False).ids
    reply, _ = _ask(spy, prompt, 40, sampling)
    more = app.tok.encode("\n<|im_start|>user\nAnd one more, please.<|im_end|>\n<|im_start|>assistant\n"
                          "<think>\n\n</think>\n\n", add_special_tokens=False).ids
    follow = prompt + reply + more
    got, stats = _ask(spy, follow, 40, sampling)
    assert stats["cached"] == len(prompt) + len(reply) - 1 and stats["drafts"] is bool(eng.depth), stats
    ref = decode.generate(eng.e, follow, 40, sampling, stop_eos=True, st=eng.serial, draft=False)
    print(f"\nfollow-up of {len(follow)} tokens resumed after {stats['cached']}: prefill {stats['prefill_s']} s, "
          f"{len(got)} tokens at {stats.get('decode_tps')} tok/s; equal to a fresh prefill: {got == ref}")
    assert got == ref, _first_difference(got, ref)


# greedy, and the checkpoint's generation_config.json (temperature 1.0, top-k 20, top-p 0.95: the server's default)
REAL_SAMPLINGS = {"greedy": {"temperature": 0},
                  "sampled": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "seed": 20260928}}


def test_real_served_drafted_replies_equal_serial_by_sha256(served):
    """Three chat and two JSON-extraction prompts (the MTP test's), greedy and sampled, 256 tokens: the served reply
    (drafted, as `tensorfold serve` runs it) and the served `"draft": false` reply against the engine's serial decoding
    of the same prompt, by SHA-256 of the token ids. Drafted against serial speed is reported, not gated."""

    from test_qwen36moe_mtp import CHAT, EXTRACT

    eng, spy = served.eng, served.spy
    if not eng.depth:
        pytest.skip(f"{family.DRAFTER} is not in the Hugging Face cache: the engine serves without drafts")
    names = ["chat 1", "chat 2", "chat 3", "json 1", "json 2"]
    print()
    totals: dict[str, list] = {}
    for sname, sampling in REAL_SAMPLINGS.items():
        for name, text in zip(names, CHAT + EXTRACT):
            body = {"model": "Qwen3.6-35B-A3B-4bit", "messages": [{"role": "user", "content": text}],
                    "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}, **sampling}
            drafted = _post(served.port, body)
            c_dr = spy.calls[-1]
            serial = _post(served.port, {**body, "draft": False})
            c_se = spy.calls[-1]
            ref = decode.generate(eng.e, c_dr.prompt, 256, c_dr.sampling, stop_eos=True, st=eng.serial, draft=False)
            sd, ss = c_dr.stats, c_se.stats
            same = _sha(c_dr.tokens) == _sha(c_se.tokens) == _sha(ref)
            ratio = sd["decode_tps"] / ss["decode_tps"] if ss.get("decode_tps") else 0.0
            print(f"{sname:7s} {name}: {len(ref):3d} tokens sha256 {_sha(ref)} served drafted {_sha(c_dr.tokens)} "
                  f"served draft:false {_sha(c_se.tokens)} {'EQUAL' if same else 'DIFFERENT'}; drafted "
                  f"{sd.get('decode_tps')} tok/s vs serial {ss.get('decode_tps')} = {ratio:.2f}x; acceptance "
                  f"{sd.get('acceptance')}, {sd.get('tokens_per_round')} tokens a round")
            assert c_dr.tokens == ref and c_se.tokens == ref, (sname, name, _first_difference(c_dr.tokens, ref))
            assert sd["drafts"] is True and ss["drafts"] is False and ss["cached"] == 0
            assert drafted["choices"][0]["message"]["content"] == serial["choices"][0]["message"]["content"]
            t = totals.setdefault(sname, [0, 0.0, 0.0, 0, 0, 0])
            n = len(ref) - 1
            t[0] += n
            t[1] += sd.get("decode_s", 0.0)
            t[2] += ss.get("decode_s", 0.0)
            t[3] += sd.get("drafted", 0)
            t[4] += sd.get("accepted", 0)
            t[5] += sd.get("rounds", 0)
    for sname, (n, ds, ss, drafted_n, accepted, rounds) in totals.items():
        print(f"{sname:7s} all: {n} tokens; served drafted {n / ds:.1f} tok/s, served serial {n / ss:.1f} tok/s = "
              f"{ss / ds:.2f}x; acceptance {accepted / max(1, drafted_n):.3f}; {n / max(1, rounds):.2f} tokens a round")


def test_real_memory_plan_covers_the_engine(served):
    eng = served.eng
    plan = eng.plan
    peak = torch.cuda.max_memory_allocated() - served.before
    now = torch.cuda.memory_allocated() - served.before
    free, total = torch.cuda.mem_get_info()
    head = eng.e.mtp.nbytes() if eng.e.mtp is not None else 0
    mbuf = eng.e.mbuf.nbytes() if eng.e.mbuf is not None else 0
    tails = 2 * eng.e.m.cfg.hidden * 2 if eng.depth else 0
    print(f"\nmodel {eng.e.m.nbytes() / GiB:.2f} + MTP head {head / GiB:.2f} GiB (plan {plan.weights / GiB:.2f}), "
          f"states {2 * eng.e.pool.nbytes_per_seq() / GiB:.2f} GiB (plan {plan.states / GiB:.2f}), buffers "
          f"{(eng.e.buf.nbytes() + mbuf) / GiB:.2f} GiB (plan {plan.buffers / GiB:.2f}); allocated {now / GiB:.2f} "
          f"GiB, peak {peak / GiB:.2f} GiB, plan without the reserve {(plan.total - plan.reserve) / GiB:.2f} GiB; "
          f"{(total - free) / GiB:.2f} of {total / GiB:.2f} GiB in use on the device")
    assert eng.eos == (248046, 248044)                     # generation_config's, <|im_end|> first
    assert eng.e.m.nbytes() + head <= plan.weights and eng.e.buf.nbytes() + mbuf <= plan.buffers
    assert plan.states == 2 * eng.e.pool.nbytes_per_seq() + tails
    assert peak <= plan.total - plan.reserve, (peak / GiB, plan.describe())
