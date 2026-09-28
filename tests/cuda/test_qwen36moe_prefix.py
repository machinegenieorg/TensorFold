"""Qwen3.6-35B-A3B's prompt cache on CUDA: resumed prompts give fresh bits, budgets hold, and the real checkpoint."""

from __future__ import annotations

import gc
import hashlib
import time
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families import qwen3_5_moe as family  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda import decode  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.decode import Decoder, run_prompt  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.engine import GiB, Qwen36Engine, checkpoint_bytes  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.forward import prepare  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.mtp import prepare_mtp  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.prefix_cache import BACKOFF, MIN_POINT, block_end, common  # noqa: E402

from test_qwen36moe_forward import V, _cached, _first_difference  # noqa: E402
from test_qwen36moe_mtp import _random  # noqa: E402

SAMPLINGS = [None, Sampling(seed=7, top_k=20, top_p=0.95)]
OPEN, SYSTEM = V - 3, V - 2            # stand-ins for <|im_start|> and "system" (V - 1 is the random model's eos)
MARKS = (OPEN, SYSTEM)
ENGINES = {"serial": {}, "drafted": {"mtp_drafts": 4, "confidence": 0.0}}


def _text(n: int, seed: int) -> list[int]:
    return torch.randint(0, V - 16, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _chat(system: list[int], user: list[int]) -> list[int]:
    """A system message, a user message and the assistant's opener, in the chat template's shape."""

    return [OPEN, SYSTEM] + system + [OPEN, 7] + user + [OPEN, 8]


def _sha(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(str(t) for t in tokens).encode()).hexdigest()


@pytest.fixture(scope="module")
def rnd():
    w, mtpw = _random()
    m = prepare(w)
    return SimpleNamespace(m=m, k=prepare_mtp(mtpw, m, draft_vocab=None))


@pytest.fixture(scope="module", params=list(ENGINES))
def engine(request, rnd):
    eng = Qwen36Engine(None, model=rnd.m, mtp=None if request.param == "serial" else rnd.k, context=2048,
                       marks=MARKS, **ENGINES[request.param])
    assert eng.prefixes.budget == 4 * GiB and (eng.depth > 0) == (request.param == "drafted")
    yield eng
    eng.e = None
    gc.collect()
    torch.cuda.empty_cache()


def _clean(eng) -> None:
    eng.prefixes.clear()
    eng.cache = []


def _ask(eng, prompt, n, sampling, **kw):
    got: list[int] = []
    stats = eng.generate(prompt, n, sampling, lambda new: got.extend(new), **kw)
    return got, stats


def _reference(eng, prompt, n, sampling):
    """Serial decoding from a fresh prefill (512-row chunks) in the serial state."""

    return decode.generate(eng.e, prompt, n, sampling, stop_eos=True, st=eng.serial, draft=False)


# -- random weights -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_a_system_block_is_kept_and_resumed_replies_equal_fresh_ones(engine, sampling):
    """Blocks ending on a chunk boundary and inside chunks of 7, 64 and 512 rows: resumed replies are fresh ones."""

    try:
        for size, chunk in ((510, 512), (700, 512), (775, 64), (1028, 512), (560, 7)):
            _clean(engine)
            engine.chunk = chunk
            b = size + 2
            system = _text(size, size)
            a, c = _chat(system, _text(90, 1)), _chat(system, _text(130, 2))
            got_a, sa = _ask(engine, a, 24, sampling)
            assert (sa["cached"], sa["resumed"], sa["checkpoints"]) == (0, None, [b]), (size, sa)
            got_c, sc = _ask(engine, c, 24, sampling)
            assert (sc["cached"], sc["resumed"], sc["checkpoints"]) == (b, "system", []), (size, sc)
            for prompt, got in ((a, got_a), (c, got_c)):
                ref = _reference(engine, prompt, 24, sampling)
                assert got == ref, (size, chunk, _first_difference(got, ref))
            # the block outlives an unrelated prompt and the serial switch; the kept reply is resumed first
            _ask(engine, _text(600, 3), 4, sampling)
            serial, ss = _ask(engine, c, 24, sampling, draft=False)
            assert ss["cached"] == 0 and serial == got_c
            again, sg = _ask(engine, c, 24, sampling)
            assert (sg["cached"], sg["resumed"]) == (b, "system") and again == got_c
            follow = c + got_c + _text(9, 4)
            more, sf = _ask(engine, follow, 12, sampling)
            assert sf["resumed"] == "kept" and sf["cached"] >= len(c), sf
            assert more == _reference(engine, follow, 12, sampling)
            assert len(engine.prefixes) == 1 and engine.prefixes.nbytes <= engine.prefixes.budget
    finally:
        engine.chunk = None


@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_a_prefix_shared_outside_a_system_message_is_kept_by_the_second_prompt(engine, sampling):
    _clean(engine)
    shared = _text(800, 11)
    p = [shared + _text(60, 12 + i) for i in range(3)]
    at = common(*[torch.tensor(x).numpy() for x in p[:2]]) - BACKOFF
    assert at >= 800 - BACKOFF
    _, s0 = _ask(engine, p[0], 16, sampling)
    _, s1 = _ask(engine, p[1], 16, sampling)
    got, s2 = _ask(engine, p[2], 16, sampling)
    assert (s0["checkpoints"], s1["checkpoints"], s2["checkpoints"]) == ([], [at], []), (s0, s1, s2)
    assert (s2["cached"], s2["resumed"]) == (at, "shared")
    assert got == _reference(engine, p[2], 16, sampling)


def _same(a, b) -> bool:
    """The model's committed state and the draft head's absorbed rows and waiting tail."""

    p, n_att = a.pos, a.kc.shape[0] - a.pool.mtp_layers
    same = (p == b.pos and torch.equal(a.rec[a.cur], b.rec[b.cur]) and torch.equal(a.conv, b.conv)
            and torch.equal(a.kc[:n_att, :p], b.kc[:n_att, :p]) and torch.equal(a.vc[:n_att, :p], b.vc[:n_att, :p]))
    L = a.mtp_len
    return same and (L, a.mtp_tail_at) == (b.mtp_len, b.mtp_tail_at) and torch.equal(a.mtp_tail, b.mtp_tail) \
        and torch.equal(a.mtp_kc[:, :L], b.mtp_kc[:, :L]) and torch.equal(a.mtp_vc[:, :L], b.mtp_vc[:, :L])


def test_restored_snapshots_with_their_rows_give_fresh_bits_at_every_length_and_chunk(rnd):
    """A snapshot with its cache rows, restored after another prompt overwrote the state: the fresh logits and
    state, draft head included, for prefixes ending at, before and after chunk boundaries."""

    e = Decoder(rnd.m, capacity=1024, rows=512, window_rows=32, attn_rows=48, states=2, mtp=rnd.k, mtp_rows=16)
    prompt = _text(900, 21)
    fresh = e.pool.alloc()
    want = run_prompt(e, prompt, st=fresh, chunk=512)
    for n, chunk in ((1, 512), (63, 64), (64, 64), (65, 512), (511, 512), (512, 128), (513, 512), (700, 7)):
        run_prompt(e, prompt[:n], st=e.st, chunk=chunk)
        snap = e.st.snapshot(rows=True)
        assert snap["kc"].shape[1] == n
        run_prompt(e, _text(800, 22), st=e.st)                 # another prompt rewrites the state and its rows
        got = run_prompt(e, prompt, st=e.st, chunk=chunk, resume=snap)
        assert torch.equal(got, want), (n, chunk)
        assert _same(e.st, fresh), (n, chunk)
    bad = dict(snap, kc=snap["kc"][:, :5])
    with pytest.raises(ValueError, match="do not match"):
        e.st.restore(bad)


def test_a_small_budget_keeps_the_most_recently_used_system_blocks(rnd):
    cfg = rnd.m.cfg
    one = checkpoint_bytes(cfg, 602)
    eng = Qwen36Engine(None, model=rnd.m, context=2048, marks=MARKS, prompt_cache_gib=2.5 * one / GiB, warm=False)
    try:
        blocks = [_text(600, 30 + i) for i in range(4)]

        def ask(i: int, seed: int):
            prompt = _chat(blocks[i], _text(40, seed))
            got, stats = _ask(eng, prompt, 6, None)
            assert got == _reference(eng, prompt, 6, None)
            assert eng.prefixes.nbytes <= eng.prefixes.budget
            return stats

        for i in range(3):
            assert ask(i, 40 + i)["checkpoints"] == [602]
        kept = lambda: [blocks.index(e.tokens[2:].tolist()) for e in eng.prefixes.entries.values()]  # noqa: E731
        assert kept() == [1, 2] and eng.prefixes.evictions == 1
        assert all(e.nbytes == one for e in eng.prefixes.entries.values())      # the estimate is the allocation
        assert ask(1, 50)["resumed"] == "system" and kept() == [2, 1]           # block 1 is now the most recent
        assert ask(3, 51)["checkpoints"] == [602] and kept() == [1, 3]          # so block 2 goes
        s = ask(2, 52)
        assert (s["cached"], s["checkpoints"]) == (0, [602]) and kept() == [3, 2]
    finally:
        eng.e = None
        gc.collect()
        torch.cuda.empty_cache()


def test_a_budget_too_small_is_refused_or_skipped_and_zero_turns_the_cache_off(rnd):
    cfg = rnd.m.cfg
    for gib in (-1.0, 0.5 * checkpoint_bytes(cfg, MIN_POINT) / GiB):
        with pytest.raises(ValueError, match="holds no prompt prefix"):
            Qwen36Engine(None, model=rnd.m, context=2048, marks=MARKS, prompt_cache_gib=gib, warm=False)
    prompt = _chat(_text(1000, 60), _text(50, 61))
    engines = []
    try:
        for gib in (checkpoint_bytes(cfg, 700) / GiB, 0.0):       # a 512-token prefix fits, this 1002-token one not
            eng = Qwen36Engine(None, model=rnd.m, context=2048, marks=MARKS, prompt_cache_gib=gib, warm=False)
            engines.append(eng)
            got, stats = _ask(eng, prompt, 12, None)
            assert stats["checkpoints"] == [] and len(eng.prefixes) == 0 and got == _reference(eng, prompt, 12, None)
        small, off = engines
        assert small.prefixes.skipped == 1 and not off.prefixes.enabled
        plans = [x.capacity_plan["cache_workspace_bytes_estimate"] for x in engines]
        assert plans[0] - plans[1] == small.prefixes.budget          # the budget is admitted with the caches
    finally:
        for eng in engines:
            eng.e = None
        gc.collect()
        torch.cuda.empty_cache()


# -- the real checkpoint --------------------------------------------------------------------------------------------
_PARTIES = ["shipper", "consignee", "carrier", "freight forwarder", "customs broker", "insurer", "warehouse operator",
            "haulier", "notify party", "bank"]
_ATTRS = [("name", "string", "the legal or trading name exactly as written, without titles or honorifics",
           "Harbourline Freight Ltd"),
          ("address", "string", "the full postal address on one line, parts separated by commas, country last",
           "14 Quay Road, Portmere, PM3 8QT, United Kingdom"),
          ("reference", "string", "the reference number this party gives the shipment, as printed",
           "HF-2031-77A"),
          ("contact", "string", "the contact person's name, or null when only a department is given",
           "Dana Okafor"),
          ("date", "date", "the date this party signed or stamped the document, as YYYY-MM-DD", "2026-03-14")]
_RULES = [
    "Copy values exactly as they appear; do not correct spelling, expand abbreviations or translate.",
    "When a field is absent from the document, output null; never guess or infer a value from context.",
    "When a field appears more than once with different values, use the value in the signed section.",
    "Dates written with month names or in day-first order are converted to YYYY-MM-DD.",
    "Weights are given in kilograms as numbers; convert tonnes and pounds, rounding to one decimal place.",
    "Monetary amounts are numbers without currency symbols; the currency goes in its own field as ISO 4217.",
    "Container numbers are four letters followed by seven digits; drop spaces and dashes inside them.",
    "A party named only by role, such as 'the buyer', is recorded with that role as its name.",
    "Handwritten corrections that are initialled replace the printed value; unsigned ones are ignored.",
    "Output one JSON object and nothing else: no commentary, no markdown fences, no trailing text.",
    "Keys appear in the order they are defined below, including the ones whose value is null.",
    "Free text fields keep their line breaks as single spaces and have leading and trailing spaces removed.",
]
_GOODS = [("goods_description", "string", "the description of the goods as declared, in the document's words"),
          ("hs_code", "string", "the six to ten digit tariff code, digits only"),
          ("packages", "integer", "the number of packages, cartons or pallets"),
          ("gross_weight_kg", "number", "the gross weight in kilograms"),
          ("net_weight_kg", "number", "the net weight in kilograms, or null when not given"),
          ("volume_m3", "number", "the volume in cubic metres"),
          ("container_numbers", "array", "every container number, in the order listed"),
          ("seal_numbers", "array", "every seal number, in the order listed"),
          ("port_of_loading", "string", "the port or place where the goods were loaded"),
          ("port_of_discharge", "string", "the port or place where the goods are unloaded"),
          ("incoterm", "string", "the three-letter Incoterm and its named place, such as FOB Portmere"),
          ("declared_value", "number", "the declared value of the goods"),
          ("currency", "string", "the ISO 4217 code of the declared value"),
          ("dangerous_goods", "boolean", "true when any UN number or hazard class is given"),
          ("un_numbers", "array", "every UN number, as four digits each")]


def _system_prompt() -> str:
    """A synthetic ~3k-token extraction system prompt: shipping documents to JSON (no real data)."""

    out = ["You are a careful data extraction assistant for a freight company. You read one shipping document at a "
           "time (bills of lading, commercial invoices, packing lists, delivery notes and customs declarations) and "
           "return its facts as a single JSON object that follows the schema below. Accuracy matters more than "
           "completeness: an empty field is better than a wrong one.", "", "General rules:"]
    out += [f"{i}. {r}" for i, r in enumerate(_RULES, 1)]
    out += ["", "Fields about the parties. Each party has the same five fields:"]
    k = 0
    for party in _PARTIES:
        key = party.replace(" ", "_")
        for attr, kind, desc, example in _ATTRS:
            k += 1
            out.append(f"{k}. `{key}_{attr}` ({kind}): {desc}, for the {party}. Example: \"{example}\".")
    out += ["", "Fields about the goods and the journey:"]
    for key, kind, desc in _GOODS:
        k += 1
        out.append(f"{k}. `{key}` ({kind}): {desc}.")
    out += ["", "Before answering, check that every key is present once, that dates and numbers follow the rules, "
            "and that nothing outside the JSON object is written."]
    return "\n".join(out)


_DOCS = [
    "BILL OF LADING No. PMR-4471\nShipper: Northgate Ceramics, 2 Kiln Lane, Stavely, ST4 1AB, United Kingdom "
    "(contact: Priya Nand)\nConsignee: Casa Azul Interiors S.L., Calle del Puerto 19, 46024 Valencia, Spain\n"
    "Carrier: Harbourline Freight Ltd, ref HF-2031-77A, signed 14 March 2026\nPort of loading: Portmere. Port of "
    "discharge: Valencia. Incoterm: FOB Portmere.\n36 pallets of glazed floor tiles, HS 690721, gross weight 18.4 t, "
    "net 17 900 kg, 41.2 m3. Container MSKU 204 7719, seal 88120-4. Declared value EUR 52,300.",
    "COMMERCIAL INVOICE 2026/118\nSeller (shipper): Lindqvist Marine Parts AB, Hamngatan 7, 413 01 Goteborg, "
    "Sweden, ref LMP-118, 3 April 2026\nBuyer: the buyer (see order 5521), address on file\nForwarder: Baltic "
    "Bridge Logistics, contact Tomas Berg\n120 cartons of stainless steel deck fittings, HS 7326909890, gross "
    "2,640 lb, 6.8 m3, value USD 18,975.50, Incoterm CIF Rotterdam. Loaded at Goteborg, discharged at Rotterdam. "
    "Containers: TGHU 8801234, TGHU 8801235. Seals 5512 and 5513.",
    "DELIVERY NOTE DN-0099\nFrom warehouse operator: Midvale Storage, Unit 4 Canal Park, Midvale, MV2 9LL, UK, "
    "stamped 2026-05-02\nTo: Greenleaf Garden Centres, 88 High Street, Ashby, AB1 2CD, UK (Mr Owen Price)\n"
    "Haulier: Fastway Transport, ref FW-77\n12 pallets of garden fertiliser, UN 2067, class 5.1, gross 9.6 tonnes, "
    "declared value GBP 7,200, delivered duty paid. Correction: packages 14 (initialled OP).",
]


def _packing(doc: int) -> str:
    """A packing list for document ``doc``: ~400 more tokens of line items, as production prompts carry."""

    items = [("carton", "glazed tiles 30 x 30 cm", 24.5), ("crate", "deck cleats, 316 steel", 18.2),
             ("pallet", "fertiliser sacks, 25 kg", 1012.0), ("drum", "sealant, non-hazardous", 61.7)]
    lines = [f"PACKING LIST (document {doc + 1})"]
    for i in range(14):
        kind, what, kg = items[(i + doc) % len(items)]
        lines.append(f"Line {i + 1}: {kind}s {3 * i + 1}-{3 * i + 3}, {what}, {kg + i * (doc + 1):.1f} kg each, "
                     f"marks {chr(65 + doc)}{i:02d}/{2026 - i}")
    return "\n".join(lines)


@pytest.fixture(scope="module")
def real():
    snap = _cached(family.MODELS[0])
    from tensorfold import families
    from tensorfold.cli import _drafter, _generation_config
    from tensorfold.cuda.server import App

    gc.collect()
    torch.cuda.empty_cache()
    drafter = _drafter(families.families()["qwen3_5_moe"], "auto")
    eng = family.cuda_engine(snap, drafter=drafter, tp=1, rank=0, master="", master_port=29551, no_drafts=False)
    # a fresh engine on the same weights: its own states, buffers and graphs, no prompt cache
    fresh = Qwen36Engine(None, model=eng.e.m, mtp=eng.e.mtp, context=eng.context_window, context_explicit=True,
                         mtp_drafts=eng.depth, confidence=eng.confidence, prompt_cache_gib=0, marks=eng.marks)
    app = App(eng, snap, "Qwen3.6-35B-A3B-4bit", default_thinking=False, sampling=_generation_config(snap),
              max_tokens=4096)

    def render(system: str, doc: str) -> list[int]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": doc}]
        return app.tok.encode(app.template.render(messages, tools=None, enable_thinking=False),
                              add_special_tokens=False).ids

    ns = SimpleNamespace(eng=eng, fresh=fresh, app=app, render=render)
    del eng, fresh
    yield ns
    ns.eng.e = ns.fresh.e = None
    ns.eng = ns.fresh = ns.app = ns.render = None
    gc.collect()
    torch.cuda.empty_cache()


def _timed(eng, prompt, n, sampling, **kw):
    """(tokens, stats, seconds to the first token)."""

    got: list[int] = []
    first: list[float] = []

    def on(new):
        if not first:
            first.append(time.perf_counter())
        got.extend(new)

    t0 = time.perf_counter()
    stats = eng.generate(prompt, n, sampling, on, **kw)
    return got, stats, first[0] - t0


REAL_SAMPLINGS = {"greedy": None, "sampled": Sampling(seed=20260928, temperature=1.0, top_k=20, top_p=0.95)}


def test_real_shared_system_prompt_resumes_fast_and_replies_equal_a_fresh_engine(real, record_property):
    """Documents under one ~3k-token system prompt: the first keeps the block, the next resume from it in a
    fraction of a fresh prefill's time, and every reply equals a fresh engine's and serial decoding's by SHA-256."""

    eng, fresh = real.eng, real.fresh
    system = _system_prompt()
    prompts = [real.render(system, doc + "\n\n" + _packing(i)) for i, doc in enumerate(_DOCS)]
    b = block_end(torch.tensor(prompts[0]).numpy(), eng.marks)
    assert eng.marks == (248045, 8678) and 2700 <= b <= 3300, b
    assert all(p[:b] == prompts[0][:b] and p[b] == 248045 for p in prompts) and b < min(map(len, prompts)) - 100
    # warm both engines at these lengths with another system prompt (kernels compile per shape, not per engine)
    other = real.render(system.replace("freight company", "shipping agency").upper(), _DOCS[0] + _packing(3))
    for x in (eng, fresh):
        _timed(x, other, 8, None)
    _clean(eng)
    one = checkpoint_bytes(eng.e.m.cfg, b, 1 if eng.depth else 0)
    timings = {}
    for name, sampling in REAL_SAMPLINGS.items():
        _clean(eng)
        a, sa, ta = _timed(eng, prompts[0], 160, sampling)
        assert sa["checkpoints"] == [b] and sa["cached"] == 0, sa
        entry = eng.prefixes.get(prompts[0][:b])
        assert entry is not None and entry.pinned and entry.nbytes == one
        for i, prompt in enumerate(prompts):
            got, s, t = _timed(eng, prompt, 160, sampling) if i else (a, sa, ta)
            ref, sf, tf = _timed(fresh, prompt, 160, sampling)
            ser, ss, _ = _timed(eng, prompt, 160, sampling, draft=False)
            assert _sha(got) == _sha(ref) == _sha(ser), (name, i, _first_difference(got, ref),
                                                         _first_difference(got, ser))
            assert sf["cached"] == 0 and ss["cached"] == 0 and len(got) > 20
            if i:
                assert (s["cached"], s["resumed"], s["checkpoints"]) == (b, "system", []), s
                assert t < 0.5 * tf, (name, i, t, tf)
            timings[f"{name}_{i}"] = {"prompt": len(prompt), "cached": s["cached"], "first_token_s": round(t, 4),
                                      "fresh_first_token_s": round(tf, 4), "sha": _sha(got)[:16],
                                      "drafts": s["drafts"]}
    # the resumed prompt's logits equal a fresh prefill's bit for bit (restored into the fresh engine's state)
    entry = eng.prefixes.get(prompts[0][:b])
    for chunk in (None, 64):
        resumed = run_prompt(fresh.e, prompts[1], st=fresh.e.st, chunk=chunk, resume=entry.snap,
                             mtp=bool(fresh.depth))
        want = run_prompt(fresh.e, prompts[1], st=fresh.serial, mtp=False)
        assert torch.equal(resumed, want), chunk
    record_property("system_block_tokens", b)
    record_property("entry_bytes", one)
    record_property("timings", timings)
