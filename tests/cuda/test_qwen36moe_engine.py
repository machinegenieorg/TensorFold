"""Qwen3.6-35B-A3B's CUDA engine (``qwen3_5_moe/cuda/engine.py``): what ``tensorfold serve`` runs.

Small random weights (the MTP test's model: the real per-layer shapes, eight layers, 32 experts, a random MTP head), for
three engines: serial (no head), drafted as served (up to 6 drafts, the 50% chain stop) and a deep chain (4 drafts, no
stop), all with CUDA graphs: the engine streams exactly serial decoding's tokens, greedy and sampled; an eos id ends
the reply and ``on_tokens`` returning True stops it; a prompt that extends the last prompt or reply resumes from the
kept state and decodes what a fresh prefill decodes; ``draft=False`` decodes one token a round from a fresh prefill
and leaves the kept states alone; the prompt/reply window the server checks against (the cache slots less the
speculative positions); the admission's cache estimate covers what the engine allocates, and on a fake checkpoint and
drafter (headers only) it counts the head, refuses an explicit context that does not fit before loading anything, and
shrinks the default one.

The real checkpoint and drafter (skipped when not in the Hugging Face cache): the engine built as ``tensorfold serve``
builds it with no flags, behind the upstream server in-process. OpenAI chat requests with thinking off: the served
(drafted) tokens equal the engine's serial decoding of the same prompt by SHA-256 for three chat and two JSON prompts,
greedy and sampled, as do ``"draft": false`` replies; streamed and non-streamed replies agree; a follow-up resumed from
the kept reply equals a fresh prefill; a request past the window is refused before streaming; drafted against serial
speed is reported (relative only).
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
from tensorfold.families.qwen3_5_moe.cuda import engine as E  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.engine import GiB, Qwen36Engine, admission, cache_bytes  # noqa: E402
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
    """--context 1024: a 1024-token prompt/reply window over 1024 + depth + 1 cache slots (a reply's last token is
    sampled, not committed; a drafted round's window reaches depth rows past it); the receipt the server reads says
    so, and the last reply that fits equals serial decoding."""

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


def _headers(folder, spec: dict) -> None:
    """A header-only safetensors file naming ``spec``'s tensors (what the admission reads)."""

    import json
    import struct

    size = {"U32": 4, "BF16": 2, "F32": 4}
    header, at = {}, 0
    for name, (dtype, shape) in spec.items():
        n = size[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [at, at + n]}
        at += n
    raw = json.dumps(header).encode()
    (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)


def _fake_checkpoint(folder, vision: bool = True):
    """The real config and header-only weights naming every tensor the loader reads (and a vision one)."""

    from tests.test_cuda_qwen36moe_package import _config, _write

    from tensorfold.families.qwen3_5_moe.cuda.weights import layout

    folder = _write(folder, _config())
    cfg = Config.read(folder)
    spec = dict(layout(cfg, "language_model."))
    if vision:
        spec["vision_tower.blocks.0.attn.qkv.weight"] = ("BF16", (3456, 1152))
    _headers(folder, spec)
    return folder, cfg


def _fake_drafter(folder):
    """The drafter's real config and header-only weights naming every tensor ``load_mtp`` reads."""

    from tests.test_cuda_qwen36moe_package import _drafter_config, _write

    from tensorfold.families.qwen3_5_moe.cuda.weights import mtp_layout

    folder = _write(folder, _drafter_config(), generation=None)
    _headers(folder, dict(mtp_layout(Config.read(folder))))
    return folder


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


def test_the_admission_counts_the_mtp_head(tmp_path):
    """With the drafter: its tensors with the model's (its router in fp32), the head's cache, tail, step buffers and
    the 76,882-row draft head in the cache estimate, and depth + 1 speculative slots; a budget that holds the model
    alone at a window refuses it with the head."""

    from tensorfold.cuda import capacity as cap

    from tensorfold.families.qwen3_5_moe.cuda.mtp import draft_token_ids

    folder, cfg = _fake_checkpoint(tmp_path / "main")
    drafter = _fake_drafter(tmp_path / "mtp")
    rows = int(((draft_token_ids("default") >= 0) & (draft_token_ids("default") < cfg.vocab)).sum())
    assert rows == 76882
    plain = cap.estimate_weights(folder, E.weight_transform(cfg.hidden))
    head = cap.estimate_weights(folder, E.weight_transform(cfg.hidden), files=sorted(drafter.glob("*.safetensors")))
    draft = E.draft_head_bytes(cfg, rows)
    more = cache_bytes(cfg, 32775, rows=512, mtp_layers=1, head_rows=rows) - cache_bytes(cfg, 32775, rows=512) - draft
    print(f"\ndrafter {head.resident / 2 ** 20:.0f} MiB, draft head {draft / 2 ** 20:.0f} MiB, head caches and "
          f"buffers at 32,768 tokens {more / 2 ** 20:.0f} MiB")
    assert 440 < head.resident / 2 ** 20 < 480 and 80 < draft / 2 ** 20 < 95
    got = admission(folder, cfg, 8192, True, rows=512, reserve=7, drafter=drafter, head_rows=rows,
                    free_memory=100 * GiB)
    assert got["cache_slots"] == 8192 + 7 and got["weight_bytes_estimate"] == plain.resident + head.resident
    assert got["cache_workspace_bytes_estimate"] == cache_bytes(cfg, 8199, rows=512, mtp_layers=1, head_rows=rows)
    alone = plain.resident + max(plain.staging, cache_bytes(cfg, 8193, rows=512))
    with pytest.raises(ValueError, match="cannot fit requested context 8192"):
        admission(folder, cfg, 8192, True, rows=512, reserve=7, drafter=drafter, head_rows=rows, free_memory=alone)
    with pytest.raises(ValueError, match="cannot fit requested context 8192"):
        Qwen36Engine(folder, drafter=str(drafter), context=8192, context_explicit=True, free_memory=alone)


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
    print(f"\nengine built in {built_s:.0f} s (drafter {'present' if drafter else 'absent'}, depth {eng.depth}), "
          f"window {eng.context_window}, cache {eng.max_len}; estimate {plan['total_bytes_estimate'] / GiB:.2f} GiB "
          f"within {plan['budget_bytes'] / GiB:.2f} GiB; server window {app.effective_context_window}")
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
            assert drafted["tensorfold"]["token_sha"] == serial["tensorfold"]["token_sha"]
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
    now = torch.cuda.memory_allocated() - served.before
    free, total = torch.cuda.mem_get_info()
    states = 2 * eng.e.pool.nbytes_per_seq()
    k = eng.e.mtp
    head = k.nbytes() - k.head.nbytes() if k is not None else 0          # the draft head counts with the caches
    mbuf = eng.e.mbuf.nbytes() + k.head.nbytes() if k is not None else 0
    mtp = 1 if eng.depth else 0
    print(f"\nmodel {eng.e.m.nbytes() / GiB:.2f} + MTP head {head / GiB:.2f} GiB (estimate "
          f"{plan['weight_bytes_estimate'] / GiB:.2f}), states {states / GiB:.2f} GiB and buffers "
          f"{(eng.e.buf.nbytes() + mbuf) / GiB:.2f} GiB (cache estimate "
          f"{plan['cache_workspace_bytes_estimate'] / GiB:.2f}); allocated {now / GiB:.2f} GiB, peak "
          f"{peak / GiB:.2f} GiB, estimate {plan['total_bytes_estimate'] / GiB:.2f} GiB; "
          f"{(total - free) / GiB:.2f} of {total / GiB:.2f} GiB in use on the device")
    assert eng.eos == (248046, 248044)                     # generation_config's, <|im_end|> first
    weights = eng.e.m.nbytes() + head
    assert weights <= plan["weight_bytes_estimate"] <= 1.03 * weights
    assert states + eng.e.buf.nbytes() + mbuf <= plan["cache_workspace_bytes_estimate"]
    assert states + mtp * 2 * eng.e.m.cfg.hidden * 2 == E.STATES * E.state_bytes(eng.e.m.cfg, eng.max_len, mtp)[0]
    assert peak <= plan["total_bytes_estimate"], (peak / GiB, plan)
