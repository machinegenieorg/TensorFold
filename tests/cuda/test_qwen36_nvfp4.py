"""Qwen3.6 MoE's NVFP4 route: nvidia/Qwen3.6-35B-A3B-NVFP4 read as it ships.

The FP8 projections' kernel kernel-first (exact decode, row invariance), the loader against the checkpoint's own
arrays, the tiny model against the fp32 reference, the draft head, and the engine: capacity admitted before any
load, drafted equal to ``"draft": false``, ``--parallel`` equal to solo. ``test_qwen36_moe.py`` runs the family's
window, graph, resume and stream tests on this route's tiny model too.
"""

import threading

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from qwen36_nvfp4_tiny import LM, tensors, write  # noqa: E402  (pytest puts tests/cuda on sys.path)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda import fp8, modelopt  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import bf16, nvfp4  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.nvfp4_moe import MoE4  # noqa: E402


def _codes(n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(n, k, generator=g, device="cuda") * 40).clamp(-448, 448).to(torch.float8_e4m3fn)


def test_every_e4m3_code_decodes_to_its_value():
    b = torch.arange(256, dtype=torch.uint8, device="cuda")
    finite = ~torch.isnan(b.view(torch.float8_e4m3fn).float())
    codes = b[finite]
    q = fp8.FP8Linear(codes[:, None].repeat(1, 64).contiguous(), torch.ones(codes.numel(), device="cuda"))
    x = torch.zeros((1, 64), dtype=torch.bfloat16, device="cuda")
    x[0, 0] = 1
    assert torch.equal(fp8.matmul(x, q)[0].float(), codes.view(torch.float8_e4m3fn).float())


@pytest.mark.parametrize("n, k", [(8192, 2048), (512, 2048), (2048, 4096), (384, 256), (100, 128), (1, 256)])
def test_fp8_rows_keep_their_bits_in_any_window(n, k):
    """Each row's bits alone equal its bits among up to 1,200 rows (decode windows and prompt chunks), for the
    stored bytes and for the widened codes; the widened codes give Flash Next's ``bf16.matmul`` times the scale."""

    q = fp8.make_fp8(_codes(n, k), torch.tensor(0.000913))
    wide = q.widen()
    x = torch.randn((1200, k), device="cuda").bfloat16()
    for lin in (q, wide):
        full = fp8.matmul(x, lin)
        for m in (1, 3, 16, 17, 64, 128, 129, 1000):
            for a in (0, 7):
                assert torch.equal(fp8.matmul(x[a:a + m], lin), full[a:a + m]), (m, a)
    assert torch.equal(fp8.matmul(x, wide),
                       (bf16.matmul(x, bf16.make_b16(wide.weight), f32=True) * q.scale).to(torch.bfloat16))
    ref = x.float() @ fp8.dequantize(q).T
    assert ((fp8.matmul(x, q).float() - ref).abs().max() / ref.abs().max()).item() < 1e-2
    assert q.nbytes() == n * k + 4 * n and wide.nbytes() == 2 * n * k + 4 * n


def test_an_fp8_scale_is_one_a_tensor_or_one_a_row():
    codes = _codes(128, 64)
    assert torch.equal(fp8.make_fp8(codes, torch.tensor(0.5)).scale, torch.full((128,), 0.5, device="cuda"))
    rows = torch.rand(128, device="cuda")
    assert torch.equal(fp8.make_fp8(codes, rows[:, None]).scale, rows)
    with pytest.raises(ValueError, match="one a tensor or one a row"):
        fp8.make_fp8(codes, torch.ones(64))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return write(tmp_path_factory.mktemp("q36nvfp4"))


def test_the_checkpoint_loads_as_it_ships(checkpoint):
    """Every stored array reaches the device as stored, each in the format its layer declares; the norms the model
    applies as 1 + w become that scale in fp32; the vision tower and the activation scales are not read."""

    from tensorfold.families.qwen3_5.cuda.weights import Plain
    from tensorfold.families.qwen3_5_moe.cuda.weights import load, load_mtp

    raw = tensors()
    w = load(checkpoint)
    m = load_mtp(checkpoint, w)
    assert w.quant == "modelopt" and not w.fast_prefill and isinstance(w.embed, Plain)
    gdn, attn = w.layers[0].gdn, w.layers[1].attn
    a, s = LM + "layers.0.linear_attn.", LM + "layers.1.self_attn."
    for lin, name in ((gdn.qkv, a + "in_proj_qkv"), (gdn.z, a + "in_proj_z"), (gdn.out, a + "out_proj"),
                      (attn.q, s + "q_proj"), (attn.o, s + "o_proj")):
        assert isinstance(lin, fp8.FP8Linear) and lin.weight.dtype == torch.uint8
        assert torch.equal(lin.weight.cpu(), raw[name + ".weight"].view(torch.uint8))
        assert torch.equal(lin.scale.cpu(), raw[name + ".weight_scale"].expand(lin.n))
    assert isinstance(gdn.b, modelopt.Dense) and torch.equal(gdn.b.b.weight.cpu(), raw[a + "in_proj_b.weight"])
    assert torch.equal(gdn.norm.cpu(), raw[a + "norm.weight"])                       # absolute, as stored
    assert gdn.A_log.dtype == torch.float32 and gdn.conv.shape == (384, 4)
    for got, name in ((w.layers[0].input_norm, LM + "layers.0.input_layernorm.weight"),
                      (attn.q_norm, s + "q_norm.weight"), (w.norm, LM + "norm.weight"),
                      (m.norm_e, "mtp.pre_fc_norm_embedding.weight"),
                      (m.attn.k_norm, "mtp.layers.0.self_attn.k_norm.weight")):
        assert got.dtype == torch.float32 and torch.equal(got.cpu(), 1.0 + raw[name].float())
    head = w.head
    assert isinstance(head, modelopt.FP4Linear) and head.fp.packed
    fp4 = ("weight", "weight_scale", "weight_scale_2")
    assert torch.equal(nvfp4.dequantize_fp4(head.fp).cpu(), nvfp4.dequantize(*(raw[f"lm_head.{t}"] for t in fp4)))
    ex = w.layers[0].moe.experts
    assert isinstance(ex, MoE4) and ex.gate_up.packed and ex.shared.gu.packed and ex.count == 17
    p = LM + "layers.0.mlp."
    for e in (0, 5, 15):
        one = ex._expert(e)
        want = torch.cat([nvfp4.dequantize(*(raw[f"{p}experts.{e}.{proj}.{t}"] for t in fp4))
                          for proj in ("gate_proj", "up_proj")])
        assert torch.equal(nvfp4.dequantize_fp4(one.gu).cpu(), want)
    shared = torch.cat([nvfp4.dequantize(*(raw[f"{p}shared_expert.{proj}.{t}"] for t in fp4))
                        for proj in ("gate_proj", "up_proj")])
    assert torch.equal(nvfp4.dequantize_fp4(ex.shared.gu).cpu(), shared)
    assert torch.equal(w.layers[0].moe.router.cpu(),
                       torch.cat([raw[p + "gate.weight"], raw[p + "shared_expert_gate.weight"]]))
    assert isinstance(m.fc_e, modelopt.Dense) and torch.equal(m.fc_h.b.weight.cpu(), raw["mtp.fc.weight"][:, 256:])
    assert isinstance(m.moe.experts, MoE4) and not m.moe.experts.gate_up.packed


def test_the_shared_experts_buffers_stay_bounded(checkpoint):
    """Prompt chunks of every length and rounds of every width share one pool of shared-expert buffers: each row gets
    the bits Flash Next's per-count buffers give it, and the pool stops growing once each power of two has come."""

    from tensorfold.cuda import moe
    from tensorfold.cuda.moe import Routed
    from tensorfold.families.qwen3_5_moe.cuda.weights import load

    w = load(checkpoint)
    m = w.layers[0].moe
    assert isinstance(m.experts, modelopt.PooledMoE4)
    plain = Routed(m.router, MoE4(m.experts.gate_up, m.experts.down_proj, m.experts.shared), m.top_k)
    x = torch.randn((600, 256), device="cuda").bfloat16()
    for prefill in (False, True):
        for rows in (1, 3, 16, 17, 100, 129, 600):
            assert torch.equal(moe.run(x[:rows], m, prefill=prefill), moe.run(x[:rows], plain, prefill=prefill))
    for rows in range(1, 601):
        for layer in w.layers:
            moe.run(x[:rows], layer.moe, prefill=rows > 16)
    torch.cuda.synchronize()
    keys, held = len(modelopt._POOL), torch.cuda.memory_allocated()
    for rows in range(600, 0, -7):
        for layer in w.layers:
            moe.run(x[:rows], layer.moe, prefill=rows > 16)
    torch.cuda.synchronize()
    assert len(modelopt._POOL) == keys <= 2 * 7 and torch.cuda.memory_allocated() == held


def test_a_tensor_the_loader_does_not_read_is_refused(tmp_path):
    from safetensors.torch import save_file

    from tensorfold.families.qwen3_5_moe.cuda.weights import load

    folder = write(tmp_path)
    save_file({LM + "layers.0.mlp.stray.weight": torch.zeros(2, 2)}, str(folder / "model-extra.safetensors"))
    with pytest.raises(ValueError, match="unused checkpoint tensors"):
        load(folder)


@pytest.mark.parametrize("top_k", [16, 4])
def test_the_route_scores_like_the_fp32_reference(tmp_path, top_k):
    """The verify windows' and the prompt path's next-token scores against the fp32 forward of the exactly dequantized
    weights, bf16 activations the only difference. With every expert selected no routing decision can flip, so every
    position's score agrees to rounding (a near tie between two tokens can still swap the top one); with the top 4
    of 16 a near tie can also pick another expert for one token."""

    from tensorfold.families.qwen3_5_moe.cuda.reference import forward, route

    folder = write(tmp_path, top_k=top_k)
    ids = torch.randint(1, 256, (2, 200), generator=torch.Generator().manual_seed(3))
    ref = forward(folder, ids, full=True)
    got = route(folder, ids, window=16)
    assert ref["logits"].std(-1).mean() > 1.0              # a model whose next token is not a coin toss
    for kind in ("", "prefill_"):
        agree = (got[kind + "top"] == ref["top"]).float().mean()
        gap = (got[kind + "logp"] - ref["logp"]).abs()
        if top_k == 16:
            assert agree > 0.95 and gap.mean() < 0.02 and gap.max() < 0.2, (kind, agree, gap.mean(), gap.max())
        else:
            assert agree > 0.95 and gap.median() < 0.03, (kind, agree, gap.median())


def test_the_draft_head_is_the_heads_own_rows(checkpoint):
    import numpy as np

    from tensorfold.families.qwen3_5_moe.cuda.weights import load

    w = load(checkpoint)
    ids = np.array(sorted(set(range(3, 256, 3)) | {255}))             # 85 rows: padded to 128, cut back to 85
    draft = modelopt.draft_head(checkpoint, ids)
    assert draft.n == len(ids) and draft.fp.n == 128
    assert torch.equal(nvfp4.dequantize_fp4(draft.fp)[:len(ids)], nvfp4.dequantize_fp4(w.head.fp)[torch.as_tensor(ids)])
    x = torch.randn((3, 256), device="cuda").bfloat16()
    got, full = draft(x).float(), w.head(x)[:, torch.as_tensor(ids, device="cuda")].float()
    assert got.shape == (3, len(ids)) and (got - full).abs().max() <= 0.02 * full.abs().max()


def test_capacity_is_admitted_before_any_load(checkpoint, monkeypatch, allocator):
    from tensorfold.cuda import capacity
    from tensorfold.families.qwen3_5_moe.cuda import engine as engine_mod

    def never(*args, **kwargs):
        raise AssertionError("weights loaded before admission refused")

    monkeypatch.setattr(modelopt, "load", never)
    with pytest.raises(ValueError, match="native window"):
        engine_mod.Qwen36Engine(checkpoint, context=10**6, context_explicit=True)
    monkeypatch.setattr(capacity, "available_bytes", lambda torch: 64 * 1024**2)
    with pytest.raises(ValueError, match="cannot fit"):
        engine_mod.Qwen36Engine(checkpoint, context=4096, context_explicit=True, streams=4)


def test_the_estimate_covers_what_loads(checkpoint):
    """The admission's weight estimate (before any load) against the device bytes the weights, MTP layer and draft
    head then take."""

    from tensorfold.cuda import capacity
    from tensorfold.families.qwen3_5_moe.cuda.weights import load, load_mtp

    ids = list(range(256))
    estimate = capacity.estimate_weights(checkpoint, modelopt.weight_bytes(len(ids), True)).resident
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    w = load(checkpoint)
    m = load_mtp(checkpoint, w)
    head = modelopt.draft_head(checkpoint, ids)
    torch.cuda.synchronize()
    used = torch.cuda.memory_allocated() - before
    assert used <= estimate <= 1.5 * used, (used, estimate)
    del w, m, head


@pytest.fixture
def allocator():
    """The engines turn expandable segments on for --parallel: put the process's setting back after each test."""

    import os

    yield
    default = "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    torch.cuda.memory._set_allocator_settings(f"expandable_segments:{default}")


PROMPTS = [[5, 6, 7, 8] * 3, list(range(20, 60)), [9, 10, 11, 12, 13, 14, 15, 16, 17], [3, 4, 5] * 40]
SAMPLINGS = [None, Sampling(1234, 1.0, 20, 0.95), Sampling(99, 0.8, 0, 1.0), None]


def _run(engine, prompt, sampling, draft, count=32):
    got: list[int] = []
    stats = engine.generate(prompt, count, sampling, lambda new: got.extend(new) or False, draft=draft)
    return got, stats


def test_the_engine_drafts_serial_tokens_and_resumes_to_fresh_bits(checkpoint, allocator):
    from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine

    engine = Qwen36Engine(checkpoint, depth=3, context=2048, context_explicit=True)
    assert engine.head is not None and isinstance(engine.head.head, modelopt.FP4Linear)
    for prompt, sampling in zip(PROMPTS, SAMPLINGS):
        drafted, stats = _run(engine, prompt, sampling, True)
        serial, _ = _run(engine, prompt, sampling, False)
        assert drafted == serial and stats["drafts"], prompt
    longer = PROMPTS[1] + _run(engine, PROMPTS[1], SAMPLINGS[1], True)[0][:-1] + [42, 43]
    resumed, stats = _run(engine, longer, SAMPLINGS[1], True)
    assert stats["cached"] == len(PROMPTS[1]) and resumed == _run(engine, longer, SAMPLINGS[1], False)[0]


def test_parallel_replies_equal_solo(checkpoint, allocator):
    from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine

    solo = Qwen36Engine(checkpoint, depth=3, context=2048, context_explicit=True)
    want = {(i, d): _run(solo, p, s, d)[0] for i, (p, s) in enumerate(zip(PROMPTS, SAMPLINGS)) for d in (True, False)}
    del solo
    many = Qwen36Engine(checkpoint, depth=3, context=2048, context_explicit=True, streams=4)
    got: dict = {}

    def go(i, d):
        got[(i, d)] = _run(many, PROMPTS[i], SAMPLINGS[i], d)[0]

    threads = [threading.Thread(target=go, args=key) for key in want]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=600)
    assert got == want
