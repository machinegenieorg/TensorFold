"""Qwen3.6-35B-A3B MTP drafting and CUDA graphs: drafted decoding emits serial tokens, graphs give eager bits."""

from __future__ import annotations

import gc
import hashlib
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families import qwen3_5_moe as family  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda import decode as D  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda import weights as W  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.decode import (  # noqa: E402
    Decoder, generate, generate_result, mtp_decode, prefill, run_prompt, serial_decode)
from tensorfold.families.qwen3_5_moe.cuda.forward import prepare  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.state import State  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.mtp import draft_token_ids, mtp_forward, prepare_mtp  # noqa: E402
from test_qwen36moe_forward import V, _cached, _config, _Rand, _tokens  # noqa: E402  (the forward test's model)

DEV = "cuda"
SAMPLINGS = [None, Sampling(seed=7, top_k=20, top_p=0.95)]


def _random(seed: int = 0) -> tuple[W.Weights, W.MTPW]:
    c = _config()
    r = _Rand(seed)
    layers = [r.layer(i, c) for i in range(c.layers)]
    half = c.rotary_dim // 2
    inv = (torch.tensor(c.rope_theta, dtype=torch.float64) ** (-torch.arange(half, dtype=torch.float64) / half))
    w = W.Weights(c, r.qw(V, c.hidden, gain=8.0), layers, r.scale(c.hidden), r.qw(V, c.hidden, gain=4.0),
                  inv.float().to(DEV))
    attn_cfg = W.Config(**{**vars(c), "layer_types": ["attention"]})
    lw = r.layer(0, attn_cfg)
    mtp = W.MTPW(c, r.scale(c.hidden), r.scale(c.hidden), r.qw(c.hidden, 2 * c.hidden), lw, r.scale(c.hidden))
    return w, mtp


@pytest.fixture(scope="module")
def rnd():
    w, mtpw = _random()
    m = prepare(w, release=False)
    k = prepare_mtp(mtpw, m, draft_vocab=None)
    return SimpleNamespace(w=w, mtpw=mtpw, m=m, k=k)


def _decoder(r, *, graphs: bool = False, k=None, states: int = 2, **kw) -> Decoder:
    kw = {"capacity": 1024, "rows": 512, "window_rows": 32, "attn_rows": 48, "mtp_rows": 16, **kw}
    return Decoder(r.m, states=states, mtp=r.k if k is None else k, graphs=graphs, **kw)


def _same_state(a: State, b: State) -> bool:
    p = a.pos
    return (p == b.pos and torch.equal(a.rec[a.cur], b.rec[b.cur]) and torch.equal(a.conv, b.conv)
            and torch.equal(a.kc[:-1, :p], b.kc[:-1, :p]) and torch.equal(a.vc[:-1, :p], b.vc[:-1, :p]))


def _serial(e: Decoder, prompt, count, sampling):
    first = prefill(e, prompt, sampling, mtp=False)
    return serial_decode(e, first, count, sampling)


# -- the head -------------------------------------------------------------------------------------------------------
def test_the_mtp_head_follows_the_reference(rnd):
    """The head's rows track the fp32 reference; chunked absorbs leave the same cache and last-row bits."""

    from tensorfold.families.qwen3_5_moe.cuda import reference as R

    e = _decoder(rnd)
    prompt = _tokens(40, 21)
    run_prompt(e, prompt, mtp=False)
    hidden = e.buf.hidden[:len(prompt)].clone()
    st = e.st
    st.mtp_len = 0
    outs = []
    for t in range(len(prompt) - 1):
        mtp_forward(e.m, e.mtp, e.mbuf, st, [prompt[t + 1]], hidden[t:t + 1], t)
        outs.append(e.mbuf.out[0].float().clone())
    ours = torch.stack(outs)
    ref = R.mtp_hidden(rnd.mtpw, rnd.w, torch.tensor(prompt[1:]), hidden[:-1].float(),
                       R.new_mtp_state(rnd.mtpw, rnd.w, bf16=True))
    assert float((ours - ref).norm() / ref.norm()) < 2e-2
    # the same positions absorbed in steps of 16 then 7 rows, and all at once: the same cache and last output
    kc_serial = st.mtp_kc[0, :len(prompt) - 1].clone()
    last_serial = e.mbuf.logits.clone()
    for steps in ((16, 16, 7), (39,)):
        at = 0
        for n in steps:
            lg = mtp_forward(e.m, e.mtp, e.mbuf if n <= e.mbuf.rows else _big(e, rnd), st, prompt[at + 1:at + 1 + n],
                             hidden[at:at + n], at)
            at += n
        assert torch.equal(st.mtp_kc[0, :at], kc_serial), steps
        assert torch.equal(lg, last_serial), steps


def _big(e: Decoder, rnd):
    from tensorfold.families.qwen3_5_moe.cuda.mtp import MTPBuffers

    return MTPBuffers(rnd.m, e.mtp, 64, capacity=e.capacity)


def test_graphs_across_context_buckets_give_eager_bits(rnd):
    """Crossing 8,192 keys moves to the next bucket's graphs; drafted and serial decoding keep eager's tokens."""

    prompt = _tokens(8180, 71)
    for sampling in SAMPLINGS:
        eager = _decoder(rnd, capacity=16384)
        ref = _serial(eager, prompt, 30, sampling)
        del eager
        graphs = _decoder(rnd, capacity=16384, graphs=True)
        assert _serial(graphs, prompt, 30, sampling).tokens == ref.tokens
        prefill(graphs, prompt, sampling)
        got = mtp_decode(graphs, ref.tokens[0], 30, sampling, depth=4, confidence=0.0)
        assert got.tokens == ref.tokens
        buckets = {key[-1] for key in graphs.graphs.main}
        assert buckets == {8192, 16384}, buckets
        del graphs
        gc.collect()
        torch.cuda.empty_cache()


# -- drafted == serial ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_drafted_decoding_emits_serial_tokens_eager_and_graphs(rnd, sampling):
    prompt = _tokens(60, 31)
    eager = _decoder(rnd)
    ref = _serial(eager, prompt, 40, sampling)
    graphs = _decoder(rnd, graphs=True)
    assert _serial(graphs, prompt, 40, sampling).tokens == ref.tokens
    seen = {}
    for depth in (1, 2, 4):
        for conf in (0.0, 0.3):
            for e in (eager, graphs):
                first = prefill(e, prompt, sampling)
                got = mtp_decode(e, first, 40, sampling, depth=depth, confidence=conf)
                assert got.tokens == ref.tokens, (depth, conf, e.graphs is not None)
                assert got.committed == ref.tokens[:-1]
                assert max(got.widths) <= depth + 1 and got.rounds == len(got.widths) == len(got.keeps)
                seen.setdefault((depth, conf), []).append((got.keeps, got.widths, got.drafted))
    for key, runs in seen.items():                  # drafts are deterministic: eager and graphs draft alike
        assert runs[0] == runs[1], key
    assert generate(eager, prompt, 40, sampling, stop_eos=False) == ref.tokens
    assert generate(graphs, prompt, 40, sampling, stop_eos=False, draft=False) == ref.tokens


@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_an_oracle_drafter_keeps_long_prefixes_and_ends_in_serial_state(rnd, sampling, monkeypatch):
    """Mostly right drafts keep long prefixes; the tokens and the committed caches equal serial decoding's."""

    prompt = _tokens(50, 41)
    e = _decoder(rnd, graphs=True, states=2)
    ref = _serial(e, prompt, 48, sampling)
    ser = e.pool.clone(e.st)
    real = D.draft
    calls = []

    def oracle(e_, st, hidden, next_tokens, position, count, sampling_, confidence=0.0):
        real(e_, st, hidden, next_tokens, position, count, sampling_, confidence)    # the head's bookkeeping
        base = position - len(prompt)
        out = [ref.tokens[base + j] if base + j < len(ref.tokens) else 0 for j in range(count)]
        calls.append(len(calls))
        if len(calls) % 3 == 0 and out:
            bad = len(calls) % len(out)
            out[bad] = (out[bad] + 1) % V
        return out

    monkeypatch.setattr(D, "draft", oracle)
    for depth in (3, 5):
        calls.clear()
        first = prefill(e, prompt, sampling)
        heard: list[int] = []
        got = mtp_decode(e, first, 48, sampling, depth=depth, confidence=0.0, on_tokens=heard.extend)
        assert got.tokens == ref.tokens and [first] + heard == ref.tokens, depth
        assert max(got.keeps) == depth + 1 and got.accepted > got.rounds, (depth, got.keeps)
        assert sum(got.keeps) >= len(ref.tokens) - 1
        assert got.committed == ref.tokens[:-1]                 # the caches end as serial decoding leaves them
        assert _same_state(e.st, ser), depth


@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_confidence_stopped_chains_and_a_draft_vocabulary_change_speed_only(rnd, sampling):
    prompt = _tokens(30, 51)
    subset = prepare_mtp(rnd.mtpw, rnd.m, draft_vocab=list(range(1, V, 3)))
    assert subset.head.n == len(range(1, V, 3)) and int(subset.ids[0]) == 1
    for k in (rnd.k, subset):
        e = _decoder(rnd, k=k, graphs=True)
        ref = _serial(e, prompt, 32, sampling)
        drafted, rounds = {}, {}
        for conf in (0.0, 0.002, 0.3, 0.9):
            first = prefill(e, prompt, sampling)
            got = mtp_decode(e, first, 32, sampling, depth=4, confidence=conf)
            assert got.tokens == ref.tokens, (conf, k is subset)
            assert min(got.widths[:-1]) >= 2, (conf, got.widths)        # every round but a last one-token round
            assert got.committed == ref.tokens[:-1]                                     # drafts one at least
            drafted[conf], rounds[conf] = got.drafted, got.rounds
        assert drafted[0.0] > drafted[0.9] and drafted[0.9] < 1.5 * rounds[0.9], (drafted, rounds)


@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_resumed_prompts_and_a_lagging_head_still_emit_serial_tokens(rnd, sampling):
    """A resumed follow-up, a serially decoded state and a prompt run without the head all draft serial tokens."""

    e = _decoder(rnd, graphs=True, states=3)
    p1 = _tokens(40, 61)
    first = prefill(e, p1, sampling)
    r1 = mtp_decode(e, first, 20, sampling, depth=3, confidence=0.0)
    assert e.st.mtp_tail_at == e.st.pos - 1 and e.st.mtp_len == e.st.pos - 1
    snap = e.st.snapshot()
    p2 = p1 + r1.committed + _tokens(10, 62)
    other = e.pool.alloc()
    got = D.generate_result(e, p2, 24, sampling, stop_eos=False, st=other)          # fresh, in another state
    e.pool.release(other)
    first = prefill(e, p2, sampling, resume=snap)
    resumed = mtp_decode(e, first, 24, sampling, depth=3, confidence=0.0)
    ref = _serial(e, p2, 24, sampling)
    assert got.tokens == resumed.tokens == ref.tokens
    assert resumed.keeps == got.keeps                                    # the head resumed in the same state
    # serially decoded, then drafted: the first round catches the head up without drafts
    first = prefill(e, p1, sampling)
    s = serial_decode(e, first, 10, sampling)
    tail = s.tokens[-1]
    cont = mtp_decode(e, tail, 20, sampling, depth=3, confidence=0.0)
    full = _serial(e, p1, 29, sampling)
    assert s.tokens[:-1] + cont.tokens == full.tokens
    assert cont.widths[0] == 1 and max(cont.widths) > 1
    # a prompt committed without the head: the same
    first = prefill(e, p1, sampling, mtp=False)
    assert mtp_decode(e, first, 20, sampling, depth=2).tokens == full.tokens[:20]


def test_graph_replays_give_eager_bits(rnd):
    e = _decoder(rnd, states=3)
    g = _decoder(rnd, graphs=True, states=3)
    assert g.warm(5) == 2 * 5 + 2 * 5
    prompt = _tokens(70, 71)
    for d in (e, g):
        run_prompt(d, prompt)
    nxt = _tokens(6, 72)
    for R in range(1, 6):
        a = e.forward(nxt[:R]).clone()
        b = g.forward(nxt[:R]).clone()
        assert torch.equal(a, b), R
        e.commit(1)
        g.commit(1)
    assert _same_state(e.st, g.st)
    hidden = e.buf.hidden[:4].clone()
    for n in range(1, 5):
        pos0 = e.st.mtp_len
        a = e.mtp_step(e.st, nxt[:n], hidden[:n], pos0).clone()
        b = g.mtp_step(g.st, nxt[:n], hidden[:n], pos0).clone()
        assert torch.equal(a, b) and torch.equal(e.mbuf.out, g.mbuf.out), n
        assert torch.equal(e.st.mtp_kc[0, :pos0 + n], g.st.mtp_kc[0, :pos0 + n]), n


def test_bad_calls_are_refused(rnd):
    e = Decoder(rnd.m, capacity=256, rows=64, window_rows=16, attn_rows=32)
    with pytest.raises(ValueError):
        mtp_decode(e, 1, 4)                                     # no head
    assert generate(e, _tokens(20, 81), 6, stop_eos=False) == generate(e, _tokens(20, 81), 6, stop_eos=False,
                                                                        draft=False)
    assert draft_token_ids(None) is None and list(draft_token_ids(3)) == [0, 1, 2]
    assert list(draft_token_ids([5, 2, 5])) == [2, 5]


# -- the real checkpoint --------------------------------------------------------------------------------------------
CHAT = [
    ("Write a detailed guide, around 400 words, on how to plan a week-long walking holiday in the Scottish Highlands: "
     "route choice, kit, weather, food and safety."),
    ("Explain to a curious twelve-year-old how a refrigerator keeps food cold, step by step, with an everyday analogy "
     "for each step. Use at least five paragraphs."),
    ("Compare three ways of learning a new language as an adult (classes, apps, and living abroad), giving the "
     "strengths and weaknesses of each and a recommendation at the end."),
]

_ORDERS = """Order log, warehouse 7, Tuesday.
09:02 Order A-1041 from Harbour Books: 12 copies of "Tide Tables 2027" at 8.50 each, ship by courier, paid.
09:15 Order A-1042 from Greenway Cafe: 3 boxes of oat milk (12 cartons per box) at 18.00 per box, collect, unpaid.
09:31 Order A-1043 from Linden Primary School: 40 exercise books at 0.90 each and 40 pencils at 0.25 each, post,
paid by invoice.
10:04 Order A-1044 from M. Okafor: one folding bicycle, model FB-20, at 349.00, courier, paid.
10:22 Order A-1045 from Harbour Books: 5 copies of "Coastal Birds" at 12.00 each, courier, unpaid.
11:47 Order A-1046 from Riverside Dental: 2 packs of nitrile gloves (size M) at 9.75 each, post, paid.
12:10 Order A-1047 from Greenway Cafe: 6 bags of coffee beans (1 kg) at 21.00 each, collect, paid.
13:35 Order A-1048 from P. Lindqvist: a desk lamp, model DL-4, at 42.50, post, unpaid."""

_PEOPLE = """Staff notes for the spring volunteer rota.
Amira Haddad can help on Saturdays, speaks Arabic and French, and has a first-aid certificate that expires in 2028.
Tom Reilly is available weekday evenings, drives a van, and prefers outdoor tasks.
Keiko Tanaka joins on Sundays, speaks Japanese, and has run the bake sale for three years.
Luis Ortega can do any weekday morning, is a qualified electrician, and cannot lift heavy loads.
Grace Mensah is free on alternate Saturdays, speaks Twi, and manages the charity's social media.
Pavel Novak helps on Wednesdays, has a forklift licence, and speaks Czech and German."""

EXTRACT = [
    "Extract every order from the log below into a JSON array. Each element must have the keys order_id, customer, "
    "items (a list of objects with name, quantity and unit_price), delivery, and paid (true or false). Reply with "
    "the JSON only.\n\n" + _ORDERS,
    "From the notes below, produce a JSON object with a key volunteers holding a list of objects with the keys "
    "name, availability, languages (a list, empty if none mentioned), skills (a list) and restrictions (a list). "
    "Reply with the JSON only.\n\n" + _PEOPLE,
]


def _sha(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(str(t) for t in tokens).encode()).hexdigest()[:16]


@pytest.fixture(scope="module")
def real():
    snap = _cached(family.MODELS[0])
    drafter = _cached(family.DRAFTER)
    from tokenizers import Tokenizer

    from tensorfold.cuda.server import ChatTemplate

    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    template = ChatTemplate(snap)
    w = W.load(snap, "cuda")
    mtpw = W.load_mtp(drafter, w.cfg, "cuda")
    m = prepare(w)
    del w
    gc.collect()
    torch.cuda.empty_cache()
    import dataclasses

    from tensorfold.families.qwen3_5_moe.cuda.mtp import draft_head

    full = prepare_mtp(mtpw, m, draft_vocab=None)
    head, ids, ids_host = draft_head(m, "default")
    heads = {"full": full, "draft vocabulary": dataclasses.replace(full, head=head, ids=ids, ids_host=ids_host)}
    gc.collect()
    torch.cuda.empty_cache()
    e = Decoder(m, capacity=4096, rows=512, mtp=heads["draft vocabulary"], graphs=True, states=1)
    e.warm(8)
    prompts = []
    for text in CHAT + EXTRACT:
        prompts.append(tok.encode(template.render([{"role": "user", "content": text}], tools=None,
                                                  enable_thinking=False), add_special_tokens=False).ids)
    yield SimpleNamespace(tok=tok, m=m, e=e, heads=heads, mtpw=mtpw, prompts=prompts,
                          names=["chat 1", "chat 2", "chat 3", "json 1", "json 2"])
    del e, m, heads, mtpw
    gc.collect()
    torch.cuda.empty_cache()


# greedy, and the checkpoint's generation_config.json (temperature 1.0, top-k 20, top-p 0.95: the server's default)
REAL_SAMPLINGS = {"greedy": None, "sampled": Sampling(seed=20260928, temperature=1.0, top_k=20, top_p=0.95)}


def _run(e: Decoder, ids, sampling, *, draft: bool, depth: int = D.DEPTH, confidence: float = D.CONFIDENCE,
         graphs: bool = True) -> D.DecodeResult:
    saved = e.graphs
    if not graphs:
        e.graphs = None
    try:
        return generate_result(e, ids, 256, sampling, stop_eos=True, draft=draft, depth=depth, confidence=confidence)
    finally:
        e.graphs = saved


def test_real_mtp_head_follows_the_reference_head(real):
    """On a passage and a reply, the CUDA head picks the fp32 reference head's top-1, and the model's as often."""

    from tensorfold.families.qwen3_5_moe.cuda import qmm
    from tensorfold.families.qwen3_5_moe.cuda import reference as R
    from tensorfold.families.qwen3_5_moe.cuda.mtp import MTPBuffers
    from test_qwen36moe_reference import PASSAGES

    e, m, k = real.e, real.m, real.heads["full"]
    reply = generate(e, real.prompts[0], 300, None, draft=False)
    texts = {"lighthouse": real.tok.encode(PASSAGES["lighthouse"], add_special_tokens=False).ids[:480],
             "chat reply": (real.prompts[0] + reply)[:480]}
    wlike = SimpleNamespace(embed=m.embed, device=m.device, inv_freq=m.inv_freq)
    mb = MTPBuffers(m, k, 16, capacity=e.capacity)
    b = e.buf

    def top1(h: torch.Tensor) -> torch.Tensor:
        out = []
        for r in range(0, h.shape[0], b.logit_rows):
            x = h[r:r + b.logit_rows].to(torch.bfloat16).contiguous()
            out.append(qmm.matmul(x, m.head, out=b.logits[:x.shape[0]], part=b.part).argmax(-1))
        return torch.cat(out)

    for ids in texts.values():
        T = len(ids)
        run_prompt(e, ids, mtp=False)
        hidden = b.hidden[:T].clone()
        model = top1(hidden)
        st = e.st
        ours_h, ours = [], []
        for t in range(T - 1):
            lg = mtp_forward(m, k, mb, st, [ids[t + 1]], hidden[t:t + 1], t)
            ours.append(int(lg.argmax()))
            ours_h.append(mb.out[0].float().clone())
        ours_h = torch.stack(ours_h)
        ref_h = R.mtp_hidden(real.mtpw, wlike, torch.tensor(ids[1:]), hidden[:-1].float(),
                             R.new_mtp_state(real.mtpw, wlike))
        ref = top1(ref_h).tolist()
        want = model[1:].tolist()
        n = T - 2
        assert float((ours_h - ref_h).norm() / ref_h.norm()) < 2e-2
        a_ours = sum(ours[t] == want[t] for t in range(n)) / n
        a_ref = sum(ref[t] == want[t] for t in range(n)) / n
        same = sum(ours[t] == ref[t] for t in range(n)) / n
        assert same > 0.9 and a_ours > a_ref - 0.03


def test_real_drafted_replies_equal_serial_by_sha256(real):
    """Three chat and two JSON prompts, greedy and sampled: drafted, serial and eager serial replies are equal."""

    e = real.e
    for sname, sampling in REAL_SAMPLINGS.items():
        for name, ids in zip(real.names, real.prompts):
            ser = _run(e, ids, sampling, draft=False)
            eager = _run(e, ids, sampling, draft=False, graphs=False)
            dr = _run(e, ids, sampling, draft=True)
            assert _sha(dr.tokens) == _sha(ser.tokens) == _sha(eager.tokens), (sname, name)
