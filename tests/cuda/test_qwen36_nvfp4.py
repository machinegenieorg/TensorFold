"""Qwen3.6 MoE's NVFP4 route: nvidia/Qwen3.6-35B-A3B-NVFP4 read as it ships, on ``tensorfold.cuda.nvfp4``.

The bf16 gates' prompt kernel (``rows8``: exact on FP8 rows, row invariance), the loader against the checkpoint's
own arrays, the tiny model against the fp32 reference, the draft head, and the engine: capacity admitted before any
load, drafted equal to ``"draft": false``, ``--parallel`` equal to solo.
``test_qwen36_moe.py`` runs the family's window, graph, resume and stream tests on this route's tiny model too.
"""

import threading

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from qwen36_nvfp4_tiny import LM, tensors, write  # noqa: E402  (pytest puts tests/cuda on sys.path)

from tensorfold.cuda.nvfp4 import experts_split as nvs  # noqa: E402
from tensorfold.cuda.kernels import qmm  # noqa: E402
from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8Linear, fragment_index  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda import fp8, modelopt, rows8  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import bf16, nvfp4  # noqa: E402

FP4 = ("weight", "weight_scale", "weight_scale_2")


def _reference(raw: dict, name: str) -> torch.Tensor:
    """The exact fp32 weight of an NVFP4 projection in the checkpoint, code x (e4m3 x scale)."""

    return nvfp4.dequantize(*(raw[f"{name}.{t}"] for t in FP4))


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
    """``fp8.py`` (``TF_NVFP4_PROMPT_ROWS=bf16``): each row's bits alone equal its bits among up to 1,200 rows (decode
    windows and prompt chunks), for the stored bytes and for the widened codes; the widened codes give Flash Next's
    ``bf16.matmul`` times the scale; the prompt form (K in one slice) keeps a row's bits in any chunk."""

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
    prompt = q.prefill(x)                                    # the prompt form: chunk-invariant bits of its own
    for m in (1, 17, 129, 1000):
        assert torch.equal(q.prefill(x[3:3 + m]), prompt[3:3 + m]), m
    ref = x.float() @ fp8.dequantize(q).T
    assert ((fp8.matmul(x, q).float() - ref).abs().max() / ref.abs().max()).item() < 1e-2
    assert q.nbytes() == n * k + 4 * n and wide.nbytes() == 2 * n * k + 4 * n


def _fp8_rows(lin: Fp8Linear) -> torch.Tensor:
    """An Fp8Linear's stored bytes back to [n, K] (its fragment order undone)."""

    kk, nn = fragment_index(lin.k, lin.npad, lin.w8.device)
    rows = torch.empty((lin.npad, lin.k), dtype=torch.uint8, device=lin.w8.device)
    rows.t()[kk, nn] = lin.w8.view(kk.shape)
    return rows[:lin.n]


def _natural(x8: torch.Tensor) -> torch.Tensor:
    """FP8 rows' bytes as fp32 values in input order (``quantize_rows`` stores each 32 in fragment order)."""

    src = rows8.fragment_order(x8.shape[1], x8.device)
    out = torch.empty(x8.shape, dtype=torch.float32, device=x8.device)
    out[:, src] = x8.view(torch.float8_e4m3fn).float()
    return out


@pytest.mark.parametrize("n, k", [(32, 2048), (64, 2048), (100, 256), (1, 128)])
def test_the_gates_take_fp8_rows_on_their_stored_weights(n, k):
    """``rows8``: a row's output is its row scale times its e4m3 codes against the bf16 weights as stored (fp32
    sums, one rounding to bf16), and its bits alone equal its bits among up to 1,200 rows."""

    g = torch.Generator(device="cuda").manual_seed(n + k)
    w = (torch.randn((n, k), generator=g, device="cuda") * 0.05).bfloat16()
    x = torch.randn((1200, k), generator=g, device="cuda").bfloat16()
    xq = qmm.quantize_rows(x)
    q = rows8.Rows8.make(w)
    full = rows8.matmul(xq, q)
    ref = (_natural(xq[0]).double() @ w.double().T) * xq[2].double()[:, None]
    err = (full.double() - ref).abs()
    assert (err <= ref.abs() * 2 ** -8 + 1e-5 * ref.abs().max()).all()
    for m in (1, 3, 64, 65, 1000):
        for a in (0, 7):
            part = tuple(t[a:a + m] for t in xq)
            assert torch.equal(rows8.matmul(part, q), full[a:a + m]), (m, a)
    assert q.nbytes() == 2 * n * k


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
    assert w.quant == "nvfp4" and w.fast_prefill and isinstance(w.embed, Plain)     # prompts take FP8 rows
    gdn, attn = w.layers[0].gdn, w.layers[1].attn
    a, s = LM + "layers.0.linear_attn.", LM + "layers.1.self_attn."
    for lin, name in ((gdn.qkv, a + "in_proj_qkv"), (gdn.z, a + "in_proj_z"), (gdn.out, a + "out_proj"),
                      (attn.q, s + "q_proj"), (attn.o, s + "o_proj")):
        assert isinstance(lin, Fp8Linear) and lin.scale == raw[name + ".weight_scale"].item()
        assert torch.equal(_fp8_rows(lin).cpu(), raw[name + ".weight"].view(torch.uint8))
    for lin, name in ((gdn.b, a + "in_proj_b"), (gdn.a, a + "in_proj_a")):
        assert isinstance(lin, modelopt.Dense) and torch.equal(lin.b.weight.cpu(), raw[name + ".weight"])
        order = rows8.fragment_order(lin.k, "cpu")
        assert torch.equal(lin.rows8.weight.cpu(), raw[name + ".weight"][:, order])      # prompt rows' order
    assert torch.equal(gdn.norm.cpu(), raw[a + "norm.weight"])                       # absolute, as stored
    assert gdn.A_log.dtype == torch.float32 and gdn.conv.shape == (384, 4)
    for got, name in ((w.layers[0].input_norm, LM + "layers.0.input_layernorm.weight"),
                      (attn.q_norm, s + "q_norm.weight"), (w.norm, LM + "norm.weight"),
                      (m.norm_e, "mtp.pre_fc_norm_embedding.weight"),
                      (m.attn.k_norm, "mtp.layers.0.self_attn.k_norm.weight")):
        assert got.dtype == torch.float32 and torch.equal(got.cpu(), 1.0 + raw[name].float())
    head = w.head
    assert isinstance(head, Fp4Linear) and head.n == 256 and head.scale == raw["lm_head.weight_scale_2"].item()
    x = torch.randn((3, 256), device="cuda").bfloat16()
    ref = x.float() @ _reference(raw, "lm_head").cuda().T
    assert ((head(x).float() - ref).abs().max() <= 0.01 * ref.abs().max()).item()
    ex = w.layers[0].moe.experts
    assert isinstance(ex, nvs.Experts) and ex.count == 17                    # 16 routed, the shared one last
    p = LM + "layers.0.mlp."
    for j, proj in enumerate(("gate", "up")):
        words, scales = nvs.unpack(ex.up[:, :, :, j].cpu())
        for e in (0, 5, 15, 16):
            name = f"{p}experts.{e}.{proj}_proj" if e < 16 else f"{p}shared_expert.{proj}_proj"
            assert torch.equal(words[e], raw[name + ".weight"])
            assert torch.equal(scales[e], raw[name + ".weight_scale"].view(torch.uint8))
            assert ex.s_up[e, j].item() == raw[name + ".weight_scale_2"].item() / 2
    assert torch.equal(w.layers[0].moe.router.cpu(),
                       torch.cat([raw[p + "gate.weight"], raw[p + "shared_expert_gate.weight"]]))
    assert isinstance(m.fc_e, modelopt.Dense) and torch.equal(m.fc_h.b.weight.cpu(), raw["mtp.fc.weight"][:, 256:])
    assert isinstance(m.attn.q, modelopt.Dense) and m.attn.q.rows8 is None     # decode rows only: no prompt copy
    assert isinstance(m.moe.experts, nvs.Experts) and m.moe.experts.count == 17     # NVFP4 at load: they only draft
    words, scales = nvs.unpack(m.moe.experts.down[:, :, :, 0])
    want = raw["mtp.layers.0.mlp.experts.down_proj"][3].float().cuda()
    fit = nvfp4.dequantize(words[3], scales[3].view(torch.float8_e4m3fn), 2 * m.moe.experts.s_down[3, 0].item())
    assert ((fit - want).abs().max() <= 0.2 * want.abs().max()).item()


def test_bf16_prompt_rows_load_the_row_major_fp8_path(checkpoint, monkeypatch):
    """``TF_NVFP4_PROMPT_ROWS=bf16``: the FP8 projections as ``fp8.py`` holds them (the stored bytes, a scale a row),
    the gates without a prompt copy, prompts on bf16 rows (quant "modelopt"); anything else is refused."""

    from tensorfold.families.qwen3_5_moe.cuda.weights import load

    raw = tensors()
    monkeypatch.setenv("TF_NVFP4_PROMPT_ROWS", "bf16")
    w = load(checkpoint)
    assert w.quant == "modelopt" and not w.fast_prefill
    gdn = w.layers[0].gdn
    a = LM + "layers.0.linear_attn."
    assert isinstance(gdn.qkv, fp8.FP8Linear) and gdn.b.rows8 is None
    assert torch.equal(gdn.qkv.weight.cpu(), raw[a + "in_proj_qkv.weight"].view(torch.uint8))
    assert torch.equal(gdn.qkv.scale.cpu(), raw[a + "in_proj_qkv.weight_scale"].expand(gdn.qkv.n))
    monkeypatch.setenv("TF_NVFP4_PROMPT_ROWS", "fp4")
    with pytest.raises(ValueError, match="fp8 or bf16"):
        load(checkpoint)


def test_a_tensor_the_loader_does_not_read_is_refused(tmp_path):
    from safetensors.torch import save_file

    from tensorfold.families.qwen3_5_moe.cuda.weights import load

    folder = write(tmp_path)
    save_file({LM + "layers.0.mlp.stray.weight": torch.zeros(2, 2)}, str(folder / "model-extra.safetensors"))
    with pytest.raises(ValueError, match="unused checkpoint tensors"):
        load(folder)


@pytest.mark.parametrize("rows", ["fp8", "bf16"])
@pytest.mark.parametrize("top_k", [16, 4])
def test_the_route_scores_like_the_fp32_reference(tmp_path, top_k, rows, monkeypatch):
    """The verify windows' and the prompt path's next-token scores against the fp32 forward of the exactly dequantized
    weights. Verify windows differ by bf16 activations alone: with every expert selected no routing decision can
    flip, so every position's score agrees to rounding (a near tie between two tokens can still swap the top one);
    with the top 4 of 16 a near tie can also pick another expert for one token. Prompt chunks take their rows to the
    projections as FP8 with a scale a row, which this tiny random model feels more than the release does; with
    ``TF_NVFP4_PROMPT_ROWS=bf16`` they stay bf16 and score as tightly as the verify windows."""

    from tensorfold.families.qwen3_5_moe.cuda.reference import forward, route

    monkeypatch.setenv("TF_NVFP4_PROMPT_ROWS", rows)
    folder = write(tmp_path, top_k=top_k)
    ids = torch.randint(1, 256, (2, 200), generator=torch.Generator().manual_seed(3))
    ref = forward(folder, ids, full=True)
    got = route(folder, ids, window=16)
    assert ref["logits"].std(-1).mean() > 1.0              # a model whose next token is not a coin toss
    bounds = {("", 16): (0.95, 0.02), ("", 4): (0.95, 0.03),                      # (top tokens equal, gap)
              ("prefill_", 16): (0.85, 0.12), ("prefill_", 4): (0.75, 0.12)}
    for kind in ("", "prefill_"):
        agree = (got[kind + "top"] == ref["top"]).float().mean()
        gap = (got[kind + "logp"] - ref["logp"]).abs()
        least, most = bounds[("" if rows == "bf16" else kind, top_k)]
        assert agree > least and (gap.mean() if top_k == 16 else gap.median()) < most, (kind, agree, gap.mean())


def test_the_draft_head_is_the_heads_own_rows(checkpoint):
    import numpy as np

    from tensorfold.families.qwen3_5_moe.cuda.weights import load

    w = load(checkpoint)
    ids = np.array(sorted(set(range(3, 256, 3)) | {255}))             # 85 rows
    draft = modelopt.draft_head(checkpoint, ids)
    assert isinstance(draft, Fp4Linear) and draft.n == len(ids) and draft.scale == w.head.scale
    x = torch.randn((3, 256), device="cuda").bfloat16()
    got, full = draft(x).float(), w.head(x)[:, torch.as_tensor(ids, device="cuda")].float()
    assert got.shape == (3, len(ids)) and (got - full).abs().max() <= 0.01 * full.abs().max()


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


@pytest.mark.parametrize("rows", ["fp8", "bf16"])
def test_the_estimate_covers_what_loads(checkpoint, rows, monkeypatch):
    """The admission's weight estimate (before any load) against the device bytes the weights, MTP layer and draft
    head then take, for either prompt row form."""

    monkeypatch.setenv("TF_NVFP4_PROMPT_ROWS", rows)

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
    tensors = 30 * 2 + 20                                  # the allocator rounds each block up to 512 bytes
    assert used <= estimate + 512 * tensors and estimate <= 1.5 * used, (used, estimate)
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


@pytest.mark.parametrize("rows", ["fp8", "bf16"])
def test_the_engine_drafts_serial_tokens_and_resumes_to_fresh_bits(checkpoint, allocator, rows, monkeypatch):
    from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine

    monkeypatch.setenv("TF_NVFP4_PROMPT_ROWS", rows)

    engine = Qwen36Engine(checkpoint, depth=3, context=2048, context_explicit=True)
    assert engine.head is not None and isinstance(engine.head.head, Fp4Linear)
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
