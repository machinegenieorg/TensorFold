"""Qwen3.6-35B-A3B's forward on CUDA: windows, commits and prefill chunks give serial bits; the real checkpoint."""

from __future__ import annotations

import gc
import os
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families import qwen3_5_moe as family  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda import weights as W  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.decode import (  # noqa: E402
    Decoder, generate, prefill, run_prompt, sample_rows, score, serial_decode)
from tensorfold.families.qwen3_5_moe.cuda.forward import (  # noqa: E402
    Buffers, commit, forward, forward_many, prepare)
from tensorfold.families.qwen3_5_moe.cuda.state import Pool, State  # noqa: E402

DEV = "cuda"
BF = torch.bfloat16
D, E, V = 2048, 32, 6144
TYPES = ["linear", "linear", "linear", "attention"] * 2


def _config() -> W.Config:
    return W.Config(hidden=D, layers=len(TYPES), layer_types=list(TYPES), vocab=V, eps=1e-6, heads=16, kv_heads=2,
                    head_dim=256, attn_gate=True, rope_theta=1e7, partial_rotary=0.25, rotary_dim=64,
                    mrope_section=(11, 11, 10), nk=16, nv=32, dk=128, dv=128, conv_kernel=4, experts=E, top_k=8,
                    moe_width=512, shared_width=512, norm_topk=True, tie_embeddings=False, mtp_layers=0,
                    max_position=262144, eos=(V - 1,), bits=4, group_size=64)


class _Rand:
    def __init__(self, seed: int) -> None:
        self.g = torch.Generator(device=DEV).manual_seed(seed)

    def qw(self, n: int, k: int, lead: tuple[int, ...] = (), gain: float = 1.0) -> W.QW:
        """MLX 4-bit g64 weights, values near uniform with variance gain^2 / k."""

        words = torch.randint(-2 ** 31, 2 ** 31, (*lead, n, k // 8), generator=self.g, device=DEV,
                              dtype=torch.int64).to(torch.int32)
        step = gain * (12.0 / k) ** 0.5 / 15
        scales = (step * (0.5 + torch.rand((*lead, n, k // 64), generator=self.g, device=DEV))).to(BF)
        noise = torch.randn((*lead, n, k // 64), generator=self.g, device=DEV)
        biases = (-7.5 * scales.float() + 0.1 * step * noise).to(BF)
        return W.QW(words, scales, biases, 4, 64)

    def scale(self, n: int) -> torch.Tensor:
        return 1.0 + 0.1 * torch.randn(n, generator=self.g, device=DEV)

    def layer(self, i: int, c: W.Config) -> W.LayerW:
        e = c.experts + 1
        moe = W.MoEW(3.0 * torch.randn((e, D), generator=self.g, device=DEV) / D ** 0.5,
                     self.qw(c.moe_width, D, (e,)), self.qw(c.moe_width, D, (e,)), self.qw(D, c.moe_width, (e,)))
        gdn = attn = None
        if c.layer_types[i] == "linear":
            conv = (0.5 * torch.randn((c.conv_dim, c.conv_kernel), generator=self.g, device=DEV)).to(BF)
            a_log = torch.log(1.0 + 15.0 * torch.rand(c.nv, generator=self.g, device=DEV))
            gdn = W.GDNW(self.qw(sum(c.gdn_rows), D), conv, a_log, torch.randn(c.nv, generator=self.g, device=DEV),
                         self.scale(c.dv).to(BF), self.qw(D, c.nv * c.dv))
        else:
            attn = W.AttnW(self.qw(sum(c.attn_rows), D), self.scale(c.head_dim), self.scale(c.head_dim),
                           self.qw(D, c.heads * c.head_dim))
        return W.LayerW(i, gdn is not None, self.scale(D), self.scale(D), gdn, attn, moe)


def _weights(seed: int = 0) -> W.Weights:
    c = _config()
    r = _Rand(seed)
    layers = [r.layer(i, c) for i in range(c.layers)]
    half = c.rotary_dim // 2
    inv = (torch.tensor(c.rope_theta, dtype=torch.float64) ** (-torch.arange(half, dtype=torch.float64) / half))
    return W.Weights(c, r.qw(V, D, gain=8.0), layers, r.scale(D), r.qw(V, D, gain=4.0), inv.float().to(DEV))


@pytest.fixture(scope="module")
def model():
    return prepare(_weights())


def _tokens(n: int, seed: int) -> list[int]:
    return torch.randint(0, V, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _decoder(m, states: int = 2, **kw) -> Decoder:
    kw = {"capacity": 1024, "rows": 512, "window_rows": 32, "attn_rows": 48, **kw}
    return Decoder(m, states=states, **kw)


def _same_state(a: State, b: State) -> bool:
    p = a.pos
    return (p == b.pos and torch.equal(a.rec[a.cur], b.rec[b.cur]) and torch.equal(a.conv, b.conv)
            and torch.equal(a.kc[:, :p], b.kc[:, :p]) and torch.equal(a.vc[:, :p], b.vc[:, :p]))


# -- random weights -------------------------------------------------------------------------------------------------
def test_windows_give_serial_logits_and_a_kept_prefix_continues_like_serial(model):
    e = _decoder(model, states=3)
    run_prompt(e, _tokens(505, 1))                     # the windows' keys cross the first 512-key attention chunk
    base = e.st
    nxt = _tokens(17, 2)
    ser = e.pool.clone(base)
    logits, snaps = [], [ser.snapshot()]
    for t in nxt:
        logits.append(e.forward([t], ser)[0].clone())
        e.commit(1, ser)
        snaps.append(ser.snapshot())
    assert all(torch.isfinite(lg.float()).all() for lg in logits)
    assert not torch.equal(logits[0], logits[1]) and len({int(lg.argmax()) for lg in logits}) > 3
    win = e.pool.alloc()
    for R in (2, 3, 8, 16):
        for keep in range(1, R + 1):
            win.copy_(base)
            lg = e.forward(nxt[:R], win)
            for r in range(R):
                assert torch.equal(lg[r], logits[r]), (R, r)
            e.commit(keep, win)
            s = snaps[keep]
            assert win.pos == s["pos"] and torch.equal(win.rec[win.cur], s["rec"]) and torch.equal(win.conv, s["conv"])
            assert torch.equal(win.kc[:, :win.pos], ser.kc[:, :win.pos]), (R, keep)
            assert torch.equal(win.vc[:, :win.pos], ser.vc[:, :win.pos]), (R, keep)
            assert torch.equal(e.forward([nxt[keep]], win)[0], logits[keep]), (R, keep)
            e.commit(1, win)


def test_prefill_chunking_changes_no_bits(model):
    """Chunks of 1 and 7 rows run as windows, 64 to 512 as prefill chunks: the same state and last logits."""

    e = _decoder(model)
    prompt = _tokens(600, 3)
    want = run_prompt(e, prompt, chunk=512)
    ref = e.pool.clone(e.st)
    for chunk in (1, 7, 64, 128):
        got = run_prompt(e, prompt, chunk=chunk)
        assert torch.equal(got, want), chunk
        assert _same_state(e.st, ref), chunk


def test_a_resumed_prompt_ends_where_a_fresh_one_does(model):
    e = _decoder(model, states=3)
    prompt = _tokens(600, 4)
    fresh = run_prompt(e, prompt)
    ref = e.pool.clone(e.st)
    run_prompt(e, prompt[:257], chunk=64)
    snap = e.st.snapshot()
    e.forward(_tokens(40, 5), logits="none")           # another continuation moves the state and writes cache rows
    e.commit(40)
    assert e.st.pos == 297
    assert torch.equal(run_prompt(e, prompt, resume=snap), fresh)
    assert _same_state(e.st, ref)
    other = e.pool.clone(ref)                           # a copy continues like its source
    tail = _tokens(5, 6)
    a = e.forward(tail, other).clone()
    e.commit(5, other)
    b = e.forward(tail, ref).clone()
    e.commit(5, ref)
    assert torch.equal(a, b) and _same_state(other, ref)


def test_two_sequences_in_one_window_get_their_own_bits(model):
    """A window of two sequences: row-shared kernels over both, per-sequence GDN chains and attention."""

    m = model
    b = Buffers(m, 512, capacity=1024, window_rows=32, attn_rows=48, seqs=2)
    pool = Pool(m, 4, 1024)
    pa, pb = _tokens(300, 7), _tokens(211, 8)
    a1, b1, a2, b2 = (pool.alloc() for _ in range(4))
    la = forward(m, b, a1, pa, logits="last").clone()
    commit(m, b, a1, len(pa))
    lb = forward(m, b, b1, pb, logits="last").clone()
    commit(m, b, b1, len(pb))
    both, table = forward_many(m, b, [(a2, pa), (b2, pb)], logits="last")         # a 511-row prefill chunk
    assert [(s.row0, s.rows, s.full) for s in table] == [(0, 300, False), (300, 211, False)]
    assert torch.equal(both[0], la[0]) and torch.equal(both[1], lb[0])
    commit(m, b, b2, len(pb))
    commit(m, b, a2, len(pa))
    assert _same_state(a1, a2) and _same_state(b1, b2)
    ta, tb = _tokens(3, 9), _tokens(5, 10)
    wa = forward(m, b, a1, ta).clone()
    commit(m, b, a1, 2)
    wb = forward(m, b, b1, tb).clone()
    commit(m, b, b1, 4)
    both, table = forward_many(m, b, [(a2, ta), (b2, tb)])                         # an 8-row window
    assert table[1].row0 == 3 and table[1].full
    assert torch.equal(both[:3], wa) and torch.equal(both[3:8], wb)
    commit(m, b, a2, 2)
    commit(m, b, b2, 4)
    assert _same_state(a1, a2) and _same_state(b1, b2)


@pytest.mark.parametrize("sampling", [None, Sampling(seed=7, top_k=20, top_p=0.95)])
def test_serial_decode_repeats_and_a_window_samples_the_same_tokens(model, sampling):
    e = _decoder(model)
    prompt = _tokens(100, 11)
    first = prefill(e, prompt, sampling)
    snap = e.st.snapshot()
    toks = serial_decode(e, first, 24, sampling).tokens
    assert len(toks) == 24
    e.st.restore(snap)
    assert serial_decode(e, first, 24, sampling).tokens == toks
    e.st.restore(snap)
    lg = e.forward(toks[:16])
    assert sample_rows(lg, [len(prompt) + 1 + r for r in range(16)], sampling) == toks[1:17]
    assert generate(e, prompt, 24, sampling, stop_eos=False) == toks


def test_bad_windows_are_refused(model):
    e = _decoder(model, capacity=256, rows=64)
    with pytest.raises(ValueError):
        e.forward(_tokens(65, 12))                      # more rows than the buffers hold
    run_prompt(e, _tokens(250, 13), chunk=64)
    with pytest.raises(ValueError):
        e.forward(_tokens(7, 14))                       # past the capacity
    e.st.reset()
    with pytest.raises(ValueError):
        e.forward(_tokens(40, 15))                      # logits of more rows than the buffers hold
    e.forward(_tokens(40, 15), logits="none")
    with pytest.raises(ValueError):
        e.commit(39)                                    # a prefill-size window keeps all its rows
    e.commit(40)
    assert e.st.pos == 40
    with pytest.raises(ValueError):
        e.commit(1)                                     # nothing left to commit


# -- the real checkpoint --------------------------------------------------------------------------------------------
def _cached(repo: str):
    import json

    from tensorfold import hub

    try:
        found = hub.cached(repo)
    except Exception:  # noqa: BLE001 - no huggingface_hub or no cache
        found = None
    index = None if found is None else found / "model.safetensors.index.json"
    if index is None or not index.is_file():
        pytest.skip(f"{repo} is not in the Hugging Face cache")
    if not all((found / s).is_file() for s in set(json.loads(index.read_text())["weight_map"].values())):
        pytest.skip(f"{repo}'s shards are not all in the Hugging Face cache")
    return found


def _reference_results(w: W.Weights, tok, template, passages: dict, prompts: list) -> dict:
    """The fp32 reference with bf16 roundings: per passage (ids, argmax, NLL), per chat prompt (ids, greedy reply)."""

    from tensorfold.families.qwen3_5_moe.cuda import reference as R

    out = {"passages": {}, "replies": []}
    for name, text in passages.items():
        ids = tok.encode(text, add_special_tokens=False).ids
        h = R.hidden(w, torch.tensor(ids), R.new_state(w, bf16=True))
        arg, nll = R.head_top1(w, h, torch.tensor(ids[1:] + [0]))
        out["passages"][name] = (ids, arg.cpu(), nll[:-1].cpu())
        del h
    for prompt, _ in prompts:
        ids = tok.encode(template.render([{"role": "user", "content": prompt}], tools=None, enable_thinking=False),
                         add_special_tokens=False).ids
        out["replies"].append((ids, R.greedy(w, ids, 64, w.cfg.eos, bf16=True)))
    return out


@pytest.fixture(scope="module")
def real():
    snap = _cached(family.MODELS[0])
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        pytest.skip("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 runs the reference's fp32 matmuls in TF32")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from tokenizers import Tokenizer

    from tensorfold.cuda.server import ChatTemplate
    from test_qwen36moe_reference import PASSAGES, PROMPTS

    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    template = ChatTemplate(snap)
    w = W.load(snap, "cuda")
    # the reference first: its MLX weights and the regrouped ones do not both fit a 32 GB GPU
    ref = _reference_results(w, tok, template, PASSAGES, PROMPTS)
    torch.cuda.empty_cache()
    m = prepare(w)
    del w
    gc.collect()
    torch.cuda.empty_cache()
    e = Decoder(m, capacity=4096, rows=512)
    yield SimpleNamespace(snap=snap, tok=tok, template=template, m=m, e=e, ref=ref, passages=PASSAGES,
                          prompts=PROMPTS)
    del e, m
    gc.collect()
    torch.cuda.empty_cache()


def _first_difference(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def test_real_forward_agrees_with_the_fp32_reference(real):
    e, tok = real.e, real.tok
    for name, (ids, arg_r, nll_r) in real.ref["passages"].items():
        arg, nll = score(e, ids)
        n = len(ids) - 1
        assert abs(float(nll.mean()) - float(nll_r.mean())) < 0.05, name
        assert int((arg[:-1].cpu() == arg_r[:-1]).sum()) / n > 0.93, name
    for (_, must), (ids, _) in zip(real.prompts, real.ref["replies"]):
        reply = tok.decode(generate(e, ids, 64, None, stop_eos=True), skip_special_tokens=True).strip()
        assert all(word in reply.lower() for word in must), reply


def test_real_window_of_sixteen_rows_equals_sixteen_serial_steps(real):
    e, tok = real.e, real.tok
    msg = "Write a short paragraph about why lighthouses were built, and how a sailor tells one from another."
    ids = tok.encode(real.template.render([{"role": "user", "content": msg}], tools=None, enable_thinking=False),
                     add_special_tokens=False).ids
    first = prefill(e, ids)
    snap = e.st.snapshot()
    res = serial_decode(e, first, 17)
    assert len(res.tokens) == 17 and len(set(res.tokens)) > 5
    e.st.restore(snap)
    serial = []
    for t in res.tokens[:16]:
        serial.append(e.forward([t])[0].clone())
        e.commit(1)
    e.st.restore(snap)
    window = e.forward(res.tokens[:16])
    assert all(torch.equal(window[r], serial[r]) for r in range(16))
    assert [int(x) for x in window.argmax(-1).tolist()] == res.tokens[1:17]
