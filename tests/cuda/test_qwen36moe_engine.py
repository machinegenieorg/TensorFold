"""Qwen3.6-35B-A3B's CUDA engine (``qwen3_5_moe/cuda/engine.py``): what ``tensorfold serve`` runs.

Small random weights (``test_qwen36moe_forward``'s model: the real per-layer shapes, eight layers, 32 experts): the
engine streams exactly serial decoding's tokens, greedy and sampled; an eos id ends the reply and ``on_tokens``
returning True stops it; a prompt that extends the last prompt or reply resumes from the kept state and decodes what a
fresh prefill decodes; ``draft=False`` decodes from a fresh prefill and leaves the kept states alone; the prompt/reply
window the server checks against; the admission's cache estimate covers what the engine allocates, and on a fake
checkpoint (headers only) it refuses an explicit context that does not fit, before loading anything, and shrinks the
default one.

The real checkpoint (skipped when it is not in the Hugging Face cache): the engine built as ``tensorfold serve`` builds
it with no flags, behind the upstream server in-process. OpenAI chat requests with thinking off, greedy and sampled:
the served tokens equal the engine's serial decoding of the same prompt, streamed and non-streamed replies agree, and
``"draft": false`` gives the same tokens; a follow-up prompt resumed from the kept reply equals a fresh prefill; a
request whose prompt plus reply passes the window is refused before streaming.
"""

from __future__ import annotations

import gc
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
from tensorfold.families.qwen3_5_moe.cuda.engine import GiB, Qwen36Engine, admission, cache_bytes  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.forward import prepare  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.weights import Config  # noqa: E402

from test_qwen36moe_forward import _cached, _first_difference, _tokens, _weights  # noqa: E402

SAMPLINGS = [None, Sampling(seed=7, top_k=20, top_p=0.95)]


@pytest.fixture(scope="module")
def model():
    return prepare(_weights())


@pytest.fixture(scope="module")
def engine(model):
    return Qwen36Engine(None, model=model, context=1024)


def _ask(eng, prompt, n, sampling, **kw):
    got: list[int] = []
    stats = eng.generate(prompt, n, sampling, lambda new: got.extend(new), **kw)
    return got, stats


def _reference(eng, prompt, n, sampling):
    """Serial decoding straight from ``decode``: a fresh prefill in the serial state, then one token a step."""

    return decode.generate(eng.e, prompt, n, sampling, stop_eos=True, st=eng.serial)


# -- random weights -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_generate_streams_serial_decoding_and_stops(engine, model, sampling):
    prompt = _tokens(40, 21)
    ref = _reference(engine, prompt, 30, sampling)
    assert len(ref) == 30 and len(set(ref)) > 5
    got, stats = _ask(engine, prompt, 30, sampling)
    assert got == ref, _first_difference(got, ref)
    assert (stats["prompt_tokens"], stats["cached"], stats["completion_tokens"], stats["rounds"]) == (40, 0, 30, 29)
    assert stats["drafts"] is False and stats["decode_tps"] > 0 and stats["prefill_s"] > 0
    # on_tokens returning True stops the decode: after the third token, and after the first
    seen: list[int] = []
    stats = engine.generate(prompt, 30, sampling, lambda new: (seen.extend(new), len(seen) >= 3)[1])
    assert seen == ref[:3] and stats["completion_tokens"] == 3
    seen = []
    stats = engine.generate(prompt, 30, sampling, lambda new: (seen.extend(new), True)[1])
    assert seen == ref[:1] and stats["completion_tokens"] == 1
    # an eos id ends the reply, and is its last token
    j = ref.index(ref[6])
    kept = model.cfg.eos
    model.cfg.eos = (ref[6],) + tuple(kept)
    try:
        assert ref[6] in engine.eos
        got, stats = _ask(engine, prompt, 30, sampling)
        assert got == ref[:j + 1] and stats["completion_tokens"] == j + 1
    finally:
        model.cfg.eos = kept


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
        assert serial == warm and ss["cached"] == 0 and ss["drafts"] is False
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
    assert fs["cached"] == len(first) + 2 and got == _reference(engine, follow, 12, sampling)


def test_the_context_limit(engine):
    """--context 1024: a 1024-token prompt/reply window over 1025 cache slots (a reply's last token is sampled, not
    committed); the receipt the server reads says so."""

    assert (engine.context_window, engine.max_len) == (1024, 1025)
    plan = engine.capacity_plan
    assert (plan["context_window"], plan["cache_slots"], plan["explicit_context"]) == (1024, 1025, True)
    assert plan["largest_window"] >= 1024 and not engine.concurrent
    with pytest.raises(ValueError, match="no room"):
        engine.generate(_tokens(1024, 40), 5, None, None)
    with pytest.raises(ValueError, match="at least one token"):
        engine.generate([], 5, None, None)
    got, stats = _ask(engine, _tokens(1000, 41), 100, None)        # the reply fits the room that is left
    assert len(got) == 24 and stats["completion_tokens"] == 24


def test_the_cache_estimate_covers_what_the_engine_allocates(engine, model):
    cfg = model.cfg
    one, snap = E.state_bytes(cfg, engine.max_len)
    assert one == engine.e.pool.nbytes_per_seq()
    s = engine.e.st.snapshot()
    assert snap == s["rec"].numel() * 4 + s["conv"].numel() * 2
    actual = engine.e.buf.nbytes()
    planned = E.buffer_bytes(cfg, engine.max_len, engine.e.buf.rows)
    assert actual <= planned <= 1.3 * actual + (8 << 20), (planned, actual)
    need = cache_bytes(cfg, engine.max_len, rows=engine.e.buf.rows)
    assert need == E.STATES * one + E.SNAPSHOTS * snap + planned + E.WORKSPACE
    assert engine.capacity_plan["cache_workspace_bytes_estimate"] == need


def _fake_checkpoint(folder, vision: bool = True):
    """The real config and a header-only safetensors file naming every tensor the loader reads (and a vision one)."""

    import json
    import struct

    from test_qwen36moe_package import _config, _write

    from tensorfold.families.qwen3_5_moe.cuda.weights import layout

    folder = _write(folder, _config())
    cfg = Config.read(folder)
    size = {"U32": 4, "BF16": 2, "F32": 4}
    header, at = {}, 0
    spec = dict(layout(cfg, "language_model."))
    if vision:
        spec["vision_tower.blocks.0.attn.qkv.weight"] = ("BF16", (3456, 1152))
    for name, (dtype, shape) in spec.items():
        n = size[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [at, at + n]}
        at += n
    raw = json.dumps(header).encode()
    (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    return folder, cfg


def test_the_admission_refuses_before_loading_and_shrinks_the_default(tmp_path):
    from tensorfold.cuda import capacity as cap

    folder, cfg = _fake_checkpoint(tmp_path / "main")
    weights = cap.estimate_weights(folder, E.weight_transform(cfg.hidden))
    print(f"\nweights {weights.resident / GiB:.2f} GiB, staging {weights.staging / GiB:.2f} GiB, cache at 32,768 "
          f"tokens {cache_bytes(cfg, 32769, rows=512) / GiB:.2f} GiB")
    assert 18.0 * GiB < weights.resident < 18.6 * GiB and weights.staging < 2 * GiB
    before = torch.cuda.memory_allocated()
    with pytest.raises(ValueError, match="cannot fit"):             # not even the weights
        Qwen36Engine(folder, context=32768, context_explicit=False, free_memory=16 * GiB)
    room = weights.resident + max(weights.staging, cache_bytes(cfg, 8193, rows=512))
    with pytest.raises(ValueError, match=r"cannot fit requested context 262144.*largest fitting"):
        Qwen36Engine(folder, context=262144, context_explicit=True, free_memory=room)
    with pytest.raises(ValueError, match="exceeds the checkpoint's 262144-token native window"):
        Qwen36Engine(folder, context=262145, context_explicit=True, free_memory=room)
    assert torch.cuda.memory_allocated() == before
    got = admission(folder, cfg, 32768, False, rows=512, reserve=1, free_memory=room)      # the default shrinks
    assert 8192 <= got["context_window"] < 8300 and got["cache_slots"] == got["context_window"] + 1, got
    assert got["weight_bytes_estimate"] == weights.resident and got["largest_window"] == got["context_window"]
    got = admission(folder, cfg, 0, True, rows=512, reserve=1, free_memory=10 * room)        # 0: the model's window
    assert got["context_window"] == 262144
    with pytest.raises(Exception) as err:                           # with room, it goes on to load (no weight bytes)
        Qwen36Engine(folder, context=4096, free_memory=10 * room, warm=False)
    assert "cannot fit" not in str(err.value)
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
    plan = eng.capacity_plan
    print(f"\nengine built in {built_s:.0f} s (drafter {'present' if drafter else 'absent'}), window "
          f"{eng.context_window}, cache {eng.max_len}; estimate {plan['total_bytes_estimate'] / GiB:.2f} GiB "
          f"within {plan['budget_bytes'] / GiB:.2f} GiB; server window {app.effective_context_window}")
    yield SimpleNamespace(snap=snap, eng=eng, app=app, spy=spy, port=server.server_address[1], before=before)
    server.shutdown()
    server.server_close()
    eng.generate = spy.inner
    del eng, app, spy
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
    ref = decode.generate(eng.e, prompt, 64, c_plain.sampling, stop_eos=True, st=eng.serial)
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
    assert plain["tensorfold"]["drafts"] is False and serial["tensorfold"]["cached"] == 0
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
    assert stats["cached"] == len(prompt) + len(reply) - 1, stats
    ref = decode.generate(eng.e, follow, 40, sampling, stop_eos=True, st=eng.serial)
    print(f"\nfollow-up of {len(follow)} tokens resumed after {stats['cached']}: prefill {stats['prefill_s']} s, "
          f"{len(got)} tokens at {stats.get('decode_tps')} tok/s; equal to a fresh prefill: {got == ref}")
    assert got == ref, _first_difference(got, ref)


def test_real_request_past_the_window_is_refused_before_streaming(served):
    eng, app = served.eng, served.app
    assert app.effective_context_window == eng.context_window == 32768 and eng.max_len == 32769
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
    now = torch.cuda.memory_allocated() - served.before
    free, total = torch.cuda.mem_get_info()
    states = 2 * eng.e.pool.nbytes_per_seq()
    print(f"\nmodel {eng.e.m.nbytes() / GiB:.2f} GiB (estimate {plan['weight_bytes_estimate'] / GiB:.2f}), states "
          f"{states / GiB:.2f} GiB and buffers {eng.e.buf.nbytes() / GiB:.2f} GiB (cache estimate "
          f"{plan['cache_workspace_bytes_estimate'] / GiB:.2f}); allocated {now / GiB:.2f} GiB, peak "
          f"{peak / GiB:.2f} GiB, estimate {plan['total_bytes_estimate'] / GiB:.2f} GiB; "
          f"{(total - free) / GiB:.2f} of {total / GiB:.2f} GiB in use on the device")
    assert eng.eos == (248046, 248044)                     # generation_config's, <|im_end|> first
    assert eng.e.m.nbytes() <= plan["weight_bytes_estimate"] <= 1.03 * eng.e.m.nbytes()
    assert states + eng.e.buf.nbytes() <= plan["cache_workspace_bytes_estimate"]
    assert states == E.STATES * E.state_bytes(eng.e.m.cfg, eng.max_len)[0]
    assert peak <= plan["total_bytes_estimate"], (peak / GiB, plan)
