"""Qwen3.6-35B-A3B's CUDA engine: random-weight engines against serial decoding, and the real checkpoint served."""

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
from tensorfold.families.qwen3_5_moe.cuda import engine as E  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.engine import GiB, Qwen36Engine, cache_bytes  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.forward import prepare  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.mtp import prepare_mtp  # noqa: E402

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
    eng = Qwen36Engine(None, model=rnd.m, mtp=None if request.param == "serial" else rnd.k, context=1024, **kw)
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
    """Serial decoding straight from ``decode``: a fresh prefill in the serial state, one token a step."""

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
    """--context 1024: a 1024-token window over 1024 + depth + 1 slots; the last reply that fits is serial's."""

    slots = 1025 + engine.depth
    assert (engine.context_window, engine.max_len) == (1024, slots)
    plan = engine.capacity_plan
    assert (plan["context_window"], plan["cache_slots"], plan["explicit_context"]) == (1024, slots, True)
    assert plan["largest_window"] >= 1024 and not engine.concurrent
    with pytest.raises(ValueError, match="no room"):
        engine.generate(_tokens(1024, 40), 5, None, None)
    with pytest.raises(ValueError, match="at least one token"):
        engine.generate([], 5, None, None)
    got, stats = _ask(engine, _tokens(1000, 41), 100, None)        # the reply fits the room that is left
    assert len(got) == 24 and stats["completion_tokens"] == 24
    assert got == _reference(engine, _tokens(1000, 41), 24, None)


def test_the_cache_estimate_covers_what_the_engine_allocates(engine, rnd):
    cfg = rnd.m.cfg
    mtp = 1 if engine.depth else 0
    one, snap = E.state_bytes(cfg, engine.max_len, mtp)
    assert one == engine.e.pool.nbytes_per_seq() + mtp * cfg.hidden * 2        # the pool keeps the tails apart
    s = engine.e.st.snapshot()
    tail = s["mtp"][2]
    assert snap == s["rec"].numel() * 4 + s["conv"].numel() * 2 + (tail.numel() * 2 if tail is not None else 0)
    actual = engine.e.buf.nbytes()
    planned = E.buffer_bytes(cfg, engine.max_len, engine.e.buf.rows)
    assert actual <= planned <= 1.3 * actual + (8 << 20), (planned, actual)
    head_rows = engine.e.mtp.head.n if mtp else 0
    need = cache_bytes(cfg, engine.max_len, rows=engine.e.buf.rows, mtp_layers=mtp, head_rows=head_rows)
    extra = 0
    if mtp:
        mbuf = engine.e.mbuf.nbytes()
        planned_m = E.mtp_buffer_bytes(cfg, engine.max_len, engine.e.mbuf.rows, head_rows)
        assert mbuf <= planned_m <= 1.3 * mbuf + (4 << 20), (planned_m, mbuf)
        extra = planned_m + E.draft_head_bytes(cfg, head_rows)
        assert engine.e.mtp.head.nbytes() <= E.draft_head_bytes(cfg, head_rows)
    assert need == E.STATES * one + E.SNAPSHOTS * snap + planned + E.WORKSPACE + extra
    assert engine.capacity_plan["cache_workspace_bytes_estimate"] == need


def test_bad_settings_are_refused(rnd):
    with pytest.raises(ValueError, match="0 to 15"):
        Qwen36Engine(None, model=rnd.m, mtp=rnd.k, mtp_drafts=16, context=256)
    with pytest.raises(ValueError, match="come from the drafter"):
        Qwen36Engine(None, model=rnd.m, mtp_drafts=3, context=256)
    with pytest.raises(ValueError, match="one GPU"):
        Qwen36Engine(None, model=rnd.m, tp=2)
    eng = Qwen36Engine(None, model=rnd.m, mtp=rnd.k, mtp_drafts=0, context=256, warm=False)   # 0: serial
    assert eng.depth == 0 and eng.e.mtp is None and eng.max_len == 257
    eng = Qwen36Engine(None, model=rnd.m, mtp=rnd.k, no_drafts=True, context=256, warm=False)
    assert eng.depth == 0 and eng.e.mtp is None


def test_the_engine_refuses_a_context_that_cannot_fit_before_loading_anything(tmp_path):
    from tensorfold.cuda import capacity as cap
    from tests.test_cuda_qwen36moe_admission import fake_checkpoint, fake_drafter

    folder, cfg = fake_checkpoint(tmp_path / "main")
    weights = cap.estimate_weights(folder, E.weight_transform(cfg.hidden))
    room = weights.resident + max(weights.staging, cache_bytes(cfg, 8193, rows=512))
    before = torch.cuda.memory_allocated()
    with pytest.raises(ValueError, match="cannot fit"):
        Qwen36Engine(folder, context=32768, context_explicit=False, free_memory=16 * GiB)
    with pytest.raises(ValueError, match=r"cannot fit requested context 262144.*largest fitting"):
        Qwen36Engine(folder, context=262144, context_explicit=True, free_memory=room)
    drafter = fake_drafter(tmp_path / "mtp")
    with pytest.raises(ValueError, match="cannot fit requested context 8192"):
        Qwen36Engine(folder, drafter=str(drafter), context=8192, context_explicit=True, free_memory=room)
    assert torch.cuda.memory_allocated() == before
    with pytest.raises(Exception) as err:                           # with room, it goes on to load (no weight bytes)
        Qwen36Engine(folder, context=4096, free_memory=10 * room, warm=False)
    assert "cannot fit" not in str(err.value)
    with pytest.raises(ValueError, match="one GPU"):
        Qwen36Engine(folder, tp=2)


@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_an_exactly_repeated_prompt_replies_as_before(engine, sampling):
    """A repeat prefills afresh (the engine resumes only a longer prompt) and replies alike; the kept states hold."""

    prompt = _tokens(70, 35)
    first, _ = _ask(engine, prompt, 12, sampling)
    again, stats = _ask(engine, prompt, 12, sampling)
    assert again == first == _reference(engine, prompt, 12, sampling)
    assert stats["cached"] == 0
    follow, stats = _ask(engine, prompt + [5], 12, sampling)
    assert stats["cached"] >= len(prompt) and follow == _reference(engine, prompt + [5], 12, sampling)


def test_fifteen_drafts_verify_sixteen_row_windows_from_graphs(rnd, monkeypatch):
    """--mtp-drafts 15 without the chain's stop: 16-row windows replay CUDA graphs and emit serial decoding's tokens."""

    eng = Qwen36Engine(None, model=rnd.m, mtp=rnd.k, context=1024, mtp_drafts=family.MAX_DRAFTS, confidence=0.0)
    try:
        assert eng.depth == 15 and eng.e.graphs.max_rows == 16
        assert eng.e.buf.window_rows >= 16 and eng.e.buf.logit_rows >= 16
        replayed: list[int] = []
        inner = eng.e.graphs.forward
        monkeypatch.setattr(eng.e.graphs, "forward", lambda st, tokens: (replayed.append(len(tokens)),
                                                                         inner(st, tokens))[1])
        for sampling in SAMPLINGS:
            prompt = _tokens(40, 36)
            got, stats = _ask(eng, prompt, 40, sampling)
            ref = _reference(eng, prompt, 40, sampling)
            assert got == ref, _first_difference(got, ref)
            assert stats["drafted"] >= 15
        assert 16 in replayed
    finally:
        eng.e = None
        gc.collect()
        torch.cuda.empty_cache()


def test_parallel_requests_are_served_one_at_a_time(rnd):
    eng = Qwen36Engine(None, model=rnd.m, context=256, streams=2, warm=False)
    assert eng.concurrent is False


# -- the real checkpoint behind the server --------------------------------------------------------------------------
class _Spy:
    """Wraps the engine's ``generate`` to record each request's prompt, sampling, tokens, stats and times."""

    def __init__(self, eng) -> None:
        self.calls: list[SimpleNamespace] = []
        self.inner = eng.generate

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        got: list[int] = []

        def tap(new):
            got.extend(new)
            return on_tokens(new)

        start = time.perf_counter()
        stats = self.inner(prompt, max_tokens, sampling, tap, draft=draft)
        self.calls.append(SimpleNamespace(prompt=list(prompt), sampling=sampling, draft=draft, tokens=got,
                                          stats=stats, start=start, end=time.perf_counter()))
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
    # what `tensorfold serve mlx-community/Qwen3.6-35B-A3B-4bit` passes with no flags
    drafter = _drafter(families.families()["qwen3_5_moe"], "auto")
    eng = family.cuda_engine(snap, drafter=drafter, tp=1, rank=0, master="", master_port=29551, no_drafts=False)
    spy = _Spy(eng)
    eng.generate = spy.generate
    app = App(eng, snap, "Qwen3.6-35B-A3B-4bit", default_thinking=False, sampling=_generation_config(snap),
              max_tokens=4096)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
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
    assert got == ref, _first_difference(got, ref)


# greedy, and the checkpoint's generation_config.json (temperature 1.0, top-k 20, top-p 0.95: the server's default)
REAL_SAMPLINGS = {"greedy": {"temperature": 0},
                  "sampled": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "seed": 20260928}}


def test_real_served_drafted_replies_equal_serial_by_sha256(served):
    """Three chat and two JSON prompts, greedy and sampled: served drafted and draft:false replies equal serial."""

    from test_qwen36moe_mtp import CHAT, EXTRACT

    eng, spy = served.eng, served.spy
    if not eng.depth:
        pytest.skip(f"{family.DRAFTER} is not in the Hugging Face cache: the engine serves without drafts")
    for sname, sampling in REAL_SAMPLINGS.items():
        for text in CHAT + EXTRACT:
            body = {"model": "Qwen3.6-35B-A3B-4bit", "messages": [{"role": "user", "content": text}],
                    "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}, **sampling}
            drafted = _post(served.port, body)
            c_dr = spy.calls[-1]
            serial = _post(served.port, {**body, "draft": False})
            c_se = spy.calls[-1]
            ref = decode.generate(eng.e, c_dr.prompt, 256, c_dr.sampling, stop_eos=True, st=eng.serial, draft=False)
            assert _sha(c_dr.tokens) == _sha(c_se.tokens) == _sha(ref), (sname, _first_difference(c_dr.tokens, ref))
            assert c_dr.stats["drafts"] is True and c_se.stats["drafts"] is False and c_se.stats["cached"] == 0
            assert drafted["choices"][0]["message"]["content"] == serial["choices"][0]["message"]["content"]
            assert drafted["tensorfold"]["token_sha"] == serial["tensorfold"]["token_sha"]


def test_real_concurrent_requests_take_turns_and_each_equals_its_solo_reply(served):
    """Two requests at once (as with --parallel 2) run one at a time, each exactly as it runs alone."""

    spy = served.spy
    bodies = [{"model": "Qwen3.6-35B-A3B-4bit", "messages": [{"role": "user", "content": QUESTION}], "max_tokens": 48,
               "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0},
              {"model": "Qwen3.6-35B-A3B-4bit", "max_tokens": 48, "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": "Describe a harbour at dawn."}], **REAL_SAMPLINGS["sampled"]}]
    solo = {}
    for body in bodies:
        _post(served.port, body)
        solo[tuple(spy.calls[-1].prompt)] = spy.calls[-1].tokens
    gate = threading.Barrier(len(bodies))
    n = len(spy.calls)

    def send(body):
        gate.wait()
        _post(served.port, body)

    threads = [threading.Thread(target=send, args=(body,)) for body in bodies]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    first, second = sorted(spy.calls[n:], key=lambda c: c.start)
    assert first.end <= second.start
    for call in (first, second):
        assert call.tokens == solo[tuple(call.prompt)], _first_difference(call.tokens, solo[tuple(call.prompt)])


def test_real_request_past_the_window_is_refused_before_streaming(served):
    eng, app = served.eng, served.app
    assert app.effective_context_window == eng.context_window == 32768 and eng.max_len == 32769 + eng.depth
    body = {"model": "Qwen3.6-35B-A3B-4bit", "messages": [{"role": "user", "content": QUESTION}], "max_tokens": 40000,
            "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0}
    for stream in (False, True):
        conn = http.client.HTTPConnection("127.0.0.1", served.port, timeout=60)
        conn.request("POST", "/v1/chat/completions", json.dumps({**body, "stream": stream}),
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
        assert resp.status == 400 and "32768-token safe cache capacity" in data["error"]["message"], data


def test_real_memory_estimate_covers_the_engine(served):
    eng = served.eng
    plan = eng.capacity_plan
    peak = torch.cuda.max_memory_allocated() - served.before
    states = 2 * eng.e.pool.nbytes_per_seq()
    k = eng.e.mtp
    head = k.nbytes() - k.head.nbytes() if k is not None else 0          # the draft head counts with the caches
    mbuf = eng.e.mbuf.nbytes() + k.head.nbytes() if k is not None else 0
    mtp = 1 if eng.depth else 0
    assert eng.eos == (248046, 248044)                     # generation_config's, <|im_end|> first
    weights = eng.e.m.nbytes() + head
    assert weights <= plan["weight_bytes_estimate"] <= 1.03 * weights
    assert states + eng.e.buf.nbytes() + mbuf <= plan["cache_workspace_bytes_estimate"]
    assert states + mtp * 2 * eng.e.m.cfg.hidden * 2 == E.STATES * E.state_bytes(eng.e.m.cfg, eng.max_len, mtp)[0]
    assert peak <= plan["total_bytes_estimate"], (peak / GiB, plan)
