"""Qwen3 embeddings on CUDA: row-invariant kernels, packed texts equal texts alone, and the checkpoint's vectors.

The checkpoint tests run with ``TF_QWEN3_EMBED_MODEL`` (the bf16 ``Qwen/Qwen3-Embedding-8B`` folder), the 4-bit
ones with ``TF_QWEN3_EMBED_Q4`` (its ``python -m tensorfold.families.qwen3.convert`` output), and the Hugging Face
fp32 comparison also needs ``TF_QWEN3_EMBED_HF=1`` (transformers, and about 32 GB of free GPU memory).
"""

import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorfold.cuda.kernels import dense, qmm
from tensorfold.cuda.kernels.prefill_attention import attention, attention_texts, text_blocks
from tensorfold.families.qwen3.convert import quantize
from tensorfold.families.qwen3.cuda import forward
from tensorfold.families.qwen3.cuda.weights import Config, Layer, Linear, Weights, rotary
from tensorfold.families.qwen3_5.cuda.weights import QLinear

DEV = "cuda"
ROOT = Path(__file__).resolve().parents[2]
MODEL = os.environ.get("TF_QWEN3_EMBED_MODEL", "")
Q4 = os.environ.get("TF_QWEN3_EMBED_Q4", "")


def test_dense_prefill_matches_fp32_and_keeps_each_rows_bits():
    torch.manual_seed(0)
    for n, k in ((384, 512), (200, 256), (1024, 1024)):
        w = (torch.randn(n, k, device=DEV) * 0.05).to(torch.bfloat16)
        x = torch.randn(300, k, device=DEV).to(torch.bfloat16)
        ref = x.float() @ w.float().t()
        got = dense.prefill_matmul(x, w, f32=True)
        assert ((got - ref).abs().max() / ref.abs().max()) < 1e-5
        full = dense.prefill_matmul(x, w)
        for _, blocks in dense.BLOCKS:
            assert torch.equal(dense.prefill_matmul(x, w, blocks=blocks), full)
        for r in (0, 15, 16, 63, 64, 130, 299):
            for m in (1, 2, 17, 65, 129):
                lo = max(0, min(r - m // 2, 300 - m))
                assert torch.equal(dense.prefill_matmul(x[lo:lo + m], w)[r - lo], full[r])


def test_text_attention_equals_each_text_alone_in_any_packing():
    torch.manual_seed(1)
    h, hk, d = 8, 2, 128
    lengths = [1, 63, 64, 65, 127, 128, 129, 300, 7]
    texts = [(torch.randn(n, h, d, device=DEV).to(torch.bfloat16), torch.randn(n, hk, d, device=DEV).to(torch.bfloat16),
              torch.randn(n, hk, d, device=DEV).to(torch.bfloat16)) for n in lengths]
    alone = [attention_texts(q, k, v, text_blocks([q.shape[0]], DEV), scale=d ** -0.5) for q, k, v in texts]
    for (q, k, v), got in zip(texts, alone):                  # the 27B's prompt attention, up to summation order
        assert torch.allclose(got.float(), attention(q, k, v, 0, scale=d ** -0.5).float(), atol=2e-2, rtol=2e-2)
    for order in (list(range(len(lengths))), [8, 2, 0, 4, 6, 1, 7, 5, 3]):
        packed = [torch.cat([texts[i][j] for i in order]).contiguous() for j in range(3)]
        out = attention_texts(*packed, text_blocks([lengths[i] for i in order], DEV), scale=d ** -0.5)
        at = 0
        for i in order:
            assert torch.equal(out[at:at + lengths[i]], alone[i])
            at += lengths[i]


CONFIG = Config(hidden=256, intermediate=512, layers=3, heads=4, kv_heads=2, head_dim=64, vocab=500, eps=1e-6,
                rope_theta=1e6, max_positions=4096)


def dequantized(words, scales, biases, group=64):
    """``code * scale + bias`` rounded once to bf16, as the 4-bit prompt kernel's fused multiply-add does."""

    codes = ((words.to(torch.int64) & 0xFFFFFFFF)[..., None] >> torch.arange(0, 32, 4, device=words.device)) & 15
    codes = codes.reshape(words.shape[0], -1, group).double()
    value = codes * scales.double()[..., None] + biases.double()[..., None]
    return value.reshape(words.shape[0], -1).to(torch.bfloat16)


def tiny(quantized: bool = False, seed: int = 2):
    """Random Qwen3 weights; ``quantized``: every projection 4-bit, returned with its dequantized bf16 twin."""

    g = torch.Generator(device="cpu").manual_seed(seed)
    c = CONFIG

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(DEV)

    def proj(n, k):
        w = rand(n, k)
        if not quantized:
            return Linear(w.to(torch.bfloat16)), Linear(w.to(torch.bfloat16))
        words, s, b = quantize(w, 4, 64)
        return Linear(qmm.pack(words, s, b, 64)), Linear(dequantized(words, s, b))

    layers, twins = [], []
    for _ in range(c.layers):
        norms = [(1 + rand(n, scale=0.1)).to(torch.bfloat16) for n in (c.hidden, c.hidden, c.head_dim, c.head_dim)]
        parts = [proj((c.heads + 2 * c.kv_heads) * c.head_dim, c.hidden), proj(c.hidden, c.heads * c.head_dim),
                 proj(2 * c.intermediate, c.hidden), proj(c.hidden, c.intermediate)]
        for out, pick in ((layers, 0), (twins, 1)):
            out.append(Layer(input_norm=norms[0], post_norm=norms[1], qkv=parts[0][pick], q_norm=norms[2],
                             k_norm=norms[3], o=parts[1][pick], gate_up=parts[2][pick], down=parts[3][pick]))
    embed = QLinear(rand(c.vocab, c.hidden, scale=1.0).to(torch.bfloat16), None, None, layout="dense", bits=0, gs=0)
    norm = (1 + rand(c.hidden, scale=0.1)).to(torch.bfloat16)
    cos, sin = rotary(c, c.max_positions, DEV)
    made = [Weights(c, embed, ls, norm, cos, sin) for ls in (layers, twins)]
    return made if quantized else made[0]


def reference(w: Weights, tokens: list[int]) -> torch.Tensor:
    """A plain fp32 Qwen3 forward over one text (weights as the engine holds them), last token, final norm."""

    c = w.config

    def weight(linear):
        return (dequantized(*qmm.unpack(linear.weight)) if isinstance(linear.weight, qmm.Q4) else linear.weight).float()

    def rms(x, g):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + c.eps) * g.float()

    n = len(tokens)
    x = w.embed.weight[torch.tensor(tokens, device=DEV)].float()
    cos, sin = (torch.cat([t[:n], t[:n]], -1) for t in (w.cos, w.sin))
    half = c.head_dim // 2
    rot = lambda t: t * cos[:, None] + torch.cat([-t[..., half:], t[..., :half]], -1) * sin[:, None]
    for layer in w.layers:
        qkv = rms(x, layer.input_norm) @ weight(layer.qkv).t()
        q, k, v = qkv.split([c.heads * c.head_dim, c.kv_heads * c.head_dim, c.kv_heads * c.head_dim], -1)
        q = rot(rms(q.view(n, c.heads, c.head_dim), layer.q_norm))
        k = rot(rms(k.view(n, c.kv_heads, c.head_dim), layer.k_norm)).repeat_interleave(c.heads // c.kv_heads, 1)
        v = v.view(n, c.kv_heads, c.head_dim).repeat_interleave(c.heads // c.kv_heads, 1)
        att = torch.nn.functional.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
                                                               is_causal=True)
        x = x + att.transpose(0, 1).reshape(n, -1) @ weight(layer.o).t()
        gate, up = (rms(x, layer.post_norm) @ weight(layer.gate_up).t()).chunk(2, -1)
        x = x + (torch.nn.functional.silu(gate) * up) @ weight(layer.down).t()
    return rms(x[-1], w.norm)


def texts_of(lengths, seed=3, vocab=CONFIG.vocab):
    rnd = random.Random(seed)
    return [[rnd.randrange(vocab) for _ in range(n)] for n in lengths]


def cosine(a, b):
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    return float(a @ b / a.norm() / b.norm())


def assert_batch_invariant(embed, texts, sizes=(2, 3, 8, 16, 64), seed=4):
    """Every text's row, alone and in shuffled batches of each size, has the same bytes."""

    alone = [embed([t])[0].tobytes() for t in texts]
    rnd = random.Random(seed)
    for size in sizes:
        order = list(range(len(texts)))
        rnd.shuffle(order)
        for at in range(0, len(order), size):
            group = order[at:at + size]
            for i, got in zip(group, embed([texts[i] for i in group])):
                assert got.tobytes() == alone[i], f"text {i} (length {len(texts[i])}) in a batch of {len(group)}"
    return alone


def test_tiny_model_matches_fp32_and_every_text_keeps_its_bits():
    torch.backends.cuda.matmul.allow_tf32 = False
    w = tiny()
    texts = texts_of([1, 2, 5, 63, 64, 65, 130, 300, 17, 9, 1000, 3] * 6, seed=5)
    embed = lambda batch: forward.embed(w, batch).cpu().numpy()
    alone = assert_batch_invariant(embed, texts)
    for t in texts[:12]:
        got = embed([t])[0]
        assert cosine(got, reference(w, t).cpu()) > 0.9999


def test_tiny_4bit_model_is_its_dequantized_bf16_forward():
    q4, twin = tiny(quantized=True)
    texts = texts_of([1, 7, 64, 65, 200, 33] * 4, seed=6)
    embed = lambda batch: forward.embed(q4, batch).cpu().numpy()
    assert_batch_invariant(embed, texts, sizes=(2, 5, 24))
    ours, bf16 = embed(texts), forward.embed(twin, texts).cpu().numpy()
    assert all(cosine(a, b) > 0.99999 for a, b in zip(ours, bf16))
    assert cosine(ours[0], reference(q4, texts[0]).cpu()) > 0.9999


def test_long_text_alone_equals_its_rows_beside_others():
    w = tiny()
    long, short = texts_of([3000, 5], seed=8)
    both = forward.embed(w, [short, long, short]).cpu().numpy()
    assert both[1].tobytes() == forward.embed(w, [long]).cpu().numpy()[0].tobytes()
    assert both[0].tobytes() == both[2].tobytes()


# -- the checkpoint -----------------------------------------------------------------------------------------------


def public_texts() -> list[str]:
    """Paragraphs of this repository's own documentation (MIT): public text of many lengths."""

    paragraphs = []
    for path in [ROOT / "README.md", ROOT / "RUNBOOK.md", *sorted((ROOT / "docs").rglob("*.md"))]:
        paragraphs += [p.strip() for p in re.split(r"\n\s*\n", path.read_text()) if len(p.strip()) > 40]
    joined = ["\n\n".join(paragraphs[i:i + 12]) for i in range(0, 120, 12)]
    return paragraphs[:40] + joined + ["What does TensorFold serve?", "x"]


@pytest.fixture(scope="module")
def engine():
    if not MODEL:
        pytest.skip("TF_QWEN3_EMBED_MODEL names the bf16 Qwen3-Embedding checkpoint")
    from tensorfold.families.qwen3.cuda.engine import Qwen3EmbedEngine

    found = Qwen3EmbedEngine(Path(MODEL), context=8192, context_explicit=True)
    yield found
    del found.w
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def tokens():
    if not MODEL:
        pytest.skip("TF_QWEN3_EMBED_MODEL names the bf16 Qwen3-Embedding checkpoint")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(MODEL) / "tokenizer.json"))
    return [tok.encode(t).ids for t in public_texts()]


def test_checkpoint_vectors_do_not_depend_on_the_batch(engine, tokens):
    assert_batch_invariant(engine.embed, tokens, sizes=(2, 5, 16, 32, 64))
    assert max(map(len, tokens)) > 1000 and min(map(len, tokens)) < 5


def test_checkpoint_endpoint_truncates_to_the_start(engine, tmp_path):
    from tensorfold.cuda import embeddings, server
    from tokenizers import Tokenizer

    app = server.App.__new__(server.App)
    app.engine, app.served, app.tok = engine, "qwen3-embed-8b", Tokenizer.from_file(str(Path(MODEL) / "tokenizer.json"))
    text = "\n\n".join(public_texts()[:30])
    reply = embeddings.serve(app, {"input": text, "truncate_prompt_tokens": 100})
    words = app.tok.encode(text, add_special_tokens=False).ids[:99]
    expected = embeddings.normalize(engine.embed([words + [app.tok.token_to_id("<|endoftext|>")]])[0], 4096)
    assert np.array(reply["data"][0]["embedding"], dtype=np.float32).tobytes() == expected.tobytes()
    assert reply["usage"]["prompt_tokens"] == 100


@pytest.mark.skipif(os.environ.get("TF_QWEN3_EMBED_HF") != "1", reason="TF_QWEN3_EMBED_HF=1 runs the fp32 reference")
def test_checkpoint_matches_hugging_face_fp32(engine, tokens):
    from transformers import AutoModel

    torch.backends.cuda.matmul.allow_tf32 = False
    picked = [tokens[i] for i in (0, 3, 10, 41, 44, len(tokens) - 2)]
    ours = [engine.embed([t])[0] for t in picked]
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    try:
        with torch.no_grad():
            for t, got in zip(picked, ours):
                ref = model(input_ids=torch.tensor([t], device=DEV)).last_hidden_state[0, -1]
                assert cosine(got, ref.cpu()) >= 0.9999, len(t)
    finally:
        del model
        torch.cuda.empty_cache()


@pytest.mark.skipif(not (Q4 and MODEL), reason="TF_QWEN3_EMBED_Q4 and TF_QWEN3_EMBED_MODEL name both checkpoints")
def test_4bit_checkpoint_keeps_its_bits_and_stays_near_bf16(engine, tokens):
    from tensorfold.families.qwen3.cuda.engine import Qwen3EmbedEngine

    bf16 = [engine.embed([t])[0] for t in tokens[:24]]
    q4 = Qwen3EmbedEngine(Path(Q4), context=8192, context_explicit=True)
    try:
        assert_batch_invariant(q4.embed, tokens, sizes=(3, 16, 64))
        cos = [cosine(q4.embed([t])[0], b) for t, b in zip(tokens[:24], bf16)]
        assert min(cos) > float(os.environ.get("TF_QWEN3_EMBED_Q4_MIN_COS", "0.97")), cos
        receipt = json.loads((Path(Q4) / "tensorfold_convert.json").read_text())
        assert receipt["bits"] == 4 and receipt["group_size"] in (32, 64)
    finally:
        del q4.w
        torch.cuda.empty_cache()
