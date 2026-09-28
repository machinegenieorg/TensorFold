"""Qwen3.6-35B-A3B's fp32 reference on tiny random weights (CPU): cached decode equals the full forward, and ties."""

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen3_5_moe.cuda import reference as R  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda import weights as W  # noqa: E402

BF, F32 = torch.bfloat16, torch.float32


def _config() -> W.Config:
    return W.Config(
        hidden=128, layers=4, layer_types=["linear", "attention", "linear", "attention"], vocab=320, eps=1e-6,
        heads=4, kv_heads=2, head_dim=64, attn_gate=True, rope_theta=1e7, partial_rotary=0.25, rotary_dim=16,
        mrope_section=(3, 3, 2), nk=2, nv=4, dk=32, dv=32, conv_kernel=4, experts=8, top_k=2, moe_width=64,
        shared_width=64, norm_topk=True, tie_embeddings=False, mtp_layers=1, max_position=4096, eos=(2,), bits=4,
        group_size=64)


def _qw(gen: torch.Generator, n: int, k: int, lead: tuple[int, ...] = (), gain: float = 1.0) -> W.QW:
    """Random 4-bit g64 weights whose values are near uniform with variance gain^2 / k."""

    words = torch.randint(-2 ** 31, 2 ** 31, (*lead, n, k // 8), generator=gen, dtype=torch.int64).to(torch.int32)
    step = gain * (12.0 / k) ** 0.5 / 15
    scales = (step * (0.5 + torch.rand((*lead, n, k // 64), generator=gen))).to(BF)
    biases = (-7.5 * scales.float() + 0.1 * step * torch.randn((*lead, n, k // 64), generator=gen)).to(BF)
    return W.QW(words, scales, biases, 4, 64)


def _scale(gen: torch.Generator, n: int) -> torch.Tensor:
    return 1.0 + 0.1 * torch.randn(n, generator=gen)


def _moe(gen: torch.Generator, c: W.Config) -> W.MoEW:
    e, d, m = c.experts + 1, c.hidden, c.moe_width
    router = 3.0 * torch.randn((e, d), generator=gen) / d ** 0.5
    return W.MoEW(router, _qw(gen, m, d, (e,)), _qw(gen, m, d, (e,)), _qw(gen, d, m, (e,)))


def _attn(gen: torch.Generator, c: W.Config) -> W.AttnW:
    return W.AttnW(_qw(gen, sum(c.attn_rows), c.hidden), _scale(gen, c.head_dim), _scale(gen, c.head_dim),
                   _qw(gen, c.hidden, c.heads * c.head_dim))


def _gdn(gen: torch.Generator, c: W.Config) -> W.GDNW:
    conv = (0.5 * torch.randn((c.conv_dim, c.conv_kernel), generator=gen)).to(BF)
    return W.GDNW(_qw(gen, sum(c.gdn_rows), c.hidden), conv, torch.log(1.0 + 15.0 * torch.rand(c.nv, generator=gen)),
                  torch.randn(c.nv, generator=gen), _scale(gen, c.dv).to(BF), _qw(gen, c.hidden, c.nv * c.dv))


def _tiny(seed: int = 0) -> tuple[W.Weights, W.MTPW]:
    gen = torch.Generator().manual_seed(seed)
    c = _config()
    layers = [W.LayerW(i, kind == "linear", _scale(gen, c.hidden), _scale(gen, c.hidden),
                       _gdn(gen, c) if kind == "linear" else None, None if kind == "linear" else _attn(gen, c),
                       _moe(gen, c)) for i, kind in enumerate(c.layer_types)]
    half = c.rotary_dim // 2
    inv = (torch.tensor(c.rope_theta, dtype=torch.float64) ** (-torch.arange(half, dtype=torch.float64) / half)).to(F32)
    w = W.Weights(c, _qw(gen, c.vocab, c.hidden, gain=8.0), layers, _scale(gen, c.hidden),
                  _qw(gen, c.vocab, c.hidden, gain=4.0), inv)
    mtp_layer = W.LayerW(0, False, _scale(gen, c.hidden), _scale(gen, c.hidden), None, _attn(gen, c), _moe(gen, c))
    mtp = W.MTPW(c, _scale(gen, c.hidden), _scale(gen, c.hidden), _qw(gen, c.hidden, 2 * c.hidden), mtp_layer,
                 _scale(gen, c.hidden))
    return w, mtp


@pytest.fixture
def small_blocks(monkeypatch):
    """Slice the head, the dequantization and the attention queries finely, so the tiny model crosses slices."""

    monkeypatch.setattr(R, "_HEAD_ROWS", 100)
    monkeypatch.setattr(R, "_DEQ_ROWS", 96)
    monkeypatch.setattr(R, "_QUERY_ROWS", 3)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max())


@pytest.mark.parametrize("bf16, tol", [(False, 2e-5), (True, 5e-3)])
@pytest.mark.parametrize("schedule", [[3] + [1] * 10, [4, 1, 1, 5, 2]])
def test_cached_serial_decode_equals_the_full_forward(small_blocks, bf16, tol, schedule):
    w, _ = _tiny()
    tokens = torch.randint(0, w.cfg.vocab, (sum(schedule),), generator=torch.Generator().manual_seed(1))
    full_st = R.new_state(w, bf16=bf16)
    full = R.forward(w, tokens, full_st)
    st, parts, at = R.new_state(w, bf16=bf16), [], 0
    for n in schedule:
        parts.append(R.forward(w, tokens[at:at + n], st))
        at += n
    serial = torch.cat(parts)
    assert serial.shape == full.shape == (len(tokens), w.cfg.vocab)
    assert st.pos == full_st.pos == len(tokens)
    assert _rel(serial, full) < tol
    if not bf16:                                               # the caches end where one full pass leaves them
        for i, layer in enumerate(w.layers):
            if layer.linear:
                assert _rel(st.rec[i], full_st.rec[i]) < tol and _rel(st.conv[i], full_st.conv[i]) < tol
            else:
                assert _rel(st.kv[i][0], full_st.kv[i][0]) < tol and _rel(st.kv[i][1], full_st.kv[i][1]) < tol


def test_the_mtp_head_decodes_serially_as_it_runs_in_full(small_blocks):
    w, mtp = _tiny()
    tokens = torch.randint(0, w.cfg.vocab, (10,), generator=torch.Generator().manual_seed(2))
    h = R.hidden(w, tokens, R.new_state(w, bf16=False))
    full = R.mtp_hidden(mtp, w, tokens[1:], h[:-1], R.new_mtp_state(mtp, w, bf16=False))
    st = R.new_mtp_state(mtp, w, bf16=False)
    serial = torch.cat([R.mtp_hidden(mtp, w, tokens[1 + t:2 + t], h[t:t + 1], st) for t in range(len(tokens) - 1)])
    assert st.pos == len(tokens) - 1
    assert _rel(R.logits(w, serial), R.logits(w, full)) < 1e-5
    # the MTP head reads its inputs: another hidden state or another token changes its output
    assert _rel(R.mtp_hidden(mtp, w, tokens[1:], h[:-1].flip(0), R.new_mtp_state(mtp, w, bf16=False)), full) > 1e-2
    assert _rel(R.mtp_hidden(mtp, w, tokens[1:].flip(0), h[:-1], R.new_mtp_state(mtp, w, bf16=False)), full) > 1e-2


def test_sliced_head_scores_equal_full_logits(small_blocks):
    w, _ = _tiny()
    tokens = torch.randint(0, w.cfg.vocab, (9,), generator=torch.Generator().manual_seed(3))
    h = R.hidden(w, tokens, R.new_state(w))
    full = R.logits(w, h)
    targets = torch.randint(0, w.cfg.vocab, (9,), generator=torch.Generator().manual_seed(4))
    arg, nll = R.head_top1(w, h, targets)
    assert torch.equal(arg, full.argmax(1))
    want = -torch.log_softmax(full, 1).gather(1, targets[:, None])[:, 0]
    assert torch.allclose(nll, want, rtol=1e-5, atol=1e-5)
    assert R.head_top1(w, h)[1] is None


def test_router_ties_go_to_the_lower_id():
    w, _ = _tiny()
    c, m = w.cfg, w.layers[0].moe
    router = torch.zeros_like(m.router)
    router[6, 0] = router[1, 0] = 5.0                          # experts 1 and 6 tie for first
    router[[2, 5], 0] = 2.0                                    # 2 and 5 tie below them
    x = torch.zeros((3, c.hidden))
    x[:, 0] = 1.0
    ids, top = R.route(W.MoEW(router, m.gate, m.up, m.down), x, c)
    assert ids.tolist() == [[1, 6]] * 3
    assert torch.allclose(top, torch.full_like(top, 0.5))
    ids, _ = R.route(W.MoEW(router, m.gate, m.up, m.down), x, W.Config(**{**c.__dict__, "top_k": 4}))
    assert ids.tolist() == [[1, 6, 2, 5]] * 3
