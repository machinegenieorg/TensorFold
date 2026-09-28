"""Qwen3.6-35B-A3B's 4-bit matmuls: row-invariant at any window and tile, pinned K splits, float64 references."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import experts as grouped  # noqa: E402
from tensorfold.cuda.kernels import qmm as shared  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda import qmm  # noqa: E402

DEV = "cuda"
ROWS = [1, 2, 3, 16, 17, 64, 128, 200]          # 200: past one 128-row tile (prefill chunks)

# every matrix of the model: stacked as the loader stores them, and each part alone
SHAPES = [(12352, 2048), (9216, 2048), (2048, 4096), (248320, 2048),
          (8192, 2048), (4096, 2048), (512, 2048), (32, 2048)]
PINNED = {(12352, 2048): 1, (9216, 2048): 2, (2048, 4096): 8, (248320, 2048): 1,
          (8192, 2048): 2, (4096, 2048): 4, (512, 2048): 4, (32, 2048): 4}


def mlx_weights(n: int, k: int, seed: int, lead: tuple = ()):
    g = torch.Generator(device=DEV).manual_seed(seed)
    words = torch.randint(-(2**31), 2**31 - 1, (*lead, n, k // 8), generator=g, device=DEV,
                          dtype=torch.int64).to(torch.int32)
    scales = (torch.rand((*lead, n, k // 64), generator=g, device=DEV) * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((*lead, n, k // 64), generator=g, device=DEV) * 0.02).to(torch.bfloat16)
    return words, scales, biases


def inputs(m: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn((m, k), generator=g, device=DEV).to(torch.bfloat16)


def test_every_model_shape_has_a_pinned_k_split():
    hidden, vocab = 2048, 248320
    gdn = 2 * 16 * 128 + 32 * 128 + 32 * 128 + 32 + 32           # [qkv | z | b | a]
    attn = 16 * 256 * 2 + 2 * 256 + 2 * 256                         # [q|gate | k | v]
    model = {(gdn, hidden), (attn, hidden), (hidden, 32 * 128), (hidden, 16 * 256), (vocab, hidden)}
    assert model <= set(qmm.SPLITS)
    for shape, sk in PINNED.items():
        assert qmm.SPLITS[shape] == sk == qmm.split_for(*shape) == shared.split_k(*shape, qmm.GS), shape
    assert set(PINNED) == set(qmm.SPLITS)


@pytest.mark.parametrize("n,k", SHAPES)
def test_rows_do_not_depend_on_the_window(n, k):
    q = qmm.make_q4(*mlx_weights(n, k, n + k))
    top = max(ROWS)
    x = inputs(top, k, 3 * n + k)
    alone = torch.cat([qmm.matmul(x[r:r + 1], q) for r in range(top)])
    alone32 = torch.cat([qmm.matmul(x[r:r + 1], q, f32=True) for r in range(top)])
    for m in ROWS:
        assert torch.equal(qmm.matmul(x[:m], q), alone[:m]), (n, k, m)
        assert torch.equal(qmm.matmul(x[:m], q, f32=True), alone32[:m]), (n, k, m)
    # later rows of a window, and any order
    assert torch.equal(qmm.matmul(x[5:22], q), alone[5:22])
    perm = torch.randperm(top, generator=torch.Generator().manual_seed(n)).to(DEV)
    assert torch.equal(qmm.matmul(x[perm], q), alone[perm])
    # the fp32 sums round to the bf16 output, and precomputed group sums give the same bits
    assert torch.equal(alone32.to(torch.bfloat16), alone)
    assert torch.equal(qmm.matmul(x[:17], q, qmm.group_sums(x[:17])), alone[:17])
    # strided rows (a view into a wider buffer)
    wide = torch.zeros((17, k + 128), dtype=torch.bfloat16, device=DEV)
    wide[:, 64:64 + k] = x[:17]
    assert torch.equal(qmm.matmul(wide[:, 64:64 + k], q), alone[:17])


@pytest.mark.parametrize("n,k", [s for s in SHAPES if qmm.split_for(*s) > 1])
def test_unreduced_slices_add_up_to_the_output(n, k):
    q = qmm.make_q4(*mlx_weights(n, k, 5 * n + k))
    x = inputs(17, k, n)
    parts = qmm.matmul(x, q, reduce=False)
    sk = qmm.split_for(n, k)
    assert parts.shape == (sk, 17, n)
    acc = parts[0].clone()
    for s in range(1, sk):
        acc = acc + parts[s]
    assert torch.equal(acc, qmm.matmul(x, q, f32=True))
    assert torch.equal(acc.to(torch.bfloat16), qmm.matmul(x, q))


@pytest.mark.parametrize("n,k", SHAPES)
def test_row_tiles_keep_the_bits(n, k):
    """The shared kernel's 16-, 32- and 64-row tiles change the schedule, never a row's sums."""

    q = qmm.make_q4(*mlx_weights(n, k, 7 * n + k))
    sk = qmm.split_for(n, k)
    for m in (5, 40):
        x = inputs(m, k, m + n)
        xs = qmm.group_sums(x)
        outs = []
        for bm in (16, 32, 64):
            for f32 in (False, True):
                out = torch.empty((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=DEV)
                shared._ext().qmm(x, xs, q.weight, q.scales, q.biases, out, None, q.n, sk, q.gs, bm, f32, True)
                outs.append(out)
        assert all(torch.equal(o, outs[i % 2]) for i, o in enumerate(outs)), (n, k, m)
        assert torch.equal(outs[0], qmm.matmul(x, q)) and torch.equal(outs[1], qmm.matmul(x, q, f32=True))


def _reference(x: torch.Tensor, w: tuple, chunk: int = 32768) -> torch.Tensor:
    out = []
    xd = x.double()
    for c in range(0, w[0].shape[0], chunk):
        deq = qmm.dequantize(w[0][c:c + chunk], w[1][c:c + chunk], w[2][c:c + chunk]).double()
        out.append(xd @ deq.T)
    return torch.cat(out, dim=1)


@pytest.mark.parametrize("n,k", SHAPES)
def test_matches_a_float64_reference(n, k):
    w = mlx_weights(n, k, 11 * n + k)
    q = qmm.make_q4(*w)
    x = inputs(17, k, 13 * n)
    ref = _reference(x, w)
    y32 = qmm.matmul(x, q, f32=True).double()
    y = qmm.matmul(x, q).double()
    scale = ref.abs().max().item()
    err32 = (y32 - ref).abs().max().item()
    err = (y - ref).abs().max().item()
    assert err32 <= scale * 1e-5, err32
    assert err <= scale * 2 ** -8, err
    back = qmm.to_mlx(q)
    assert all(torch.equal(a, b) for a, b in zip(back, w))


def test_stacked_rows_keep_their_values():
    parts = [mlx_weights(8192, 2048, 1), mlx_weights(4096, 2048, 2), mlx_weights(32, 2048, 3),
             mlx_weights(32, 2048, 4)]
    q = qmm.stack_q4(parts)
    assert (q.n, q.k) == (12352, 2048)
    w, s, b = qmm.to_mlx(q)
    assert torch.equal(w, torch.cat([p[0] for p in parts])) and torch.equal(s, torch.cat([p[1] for p in parts]))
    assert torch.equal(b, torch.cat([p[2] for p in parts]))
    # unpacked a few rows at a time (the head's is chunked): the same arrays, a part-filled last chunk included
    assert all(torch.equal(x, y) for x, y in zip(qmm.to_mlx(q, rows=128), (w, s, b)))
    x = inputs(3, 2048, 9)
    ref = _reference(x, (w, s, b))
    assert (qmm.matmul(x, q).double() - ref).abs().max().item() <= ref.abs().max().item() * 2 ** -8


def test_embedding_gathers_exact_rows():
    vocab, d = 4096, 2048
    w = mlx_weights(vocab, d, 21)
    ids = torch.tensor([0, 7, 4095, 7, 1234, 2, 3000, 17, 99], dtype=torch.int32, device=DEV)
    got = qmm.embed(ids, *w)
    want = qmm.dequantize(w[0][ids.long()], w[1][ids.long()], w[2][ids.long()]).to(torch.bfloat16)
    assert torch.equal(got, want)             # q * s is exact in fp32, so one rounding either way
    for r in range(ids.shape[0]):
        assert torch.equal(qmm.embed(ids[r:r + 1], *w)[0], got[r])
    out = torch.empty((ids.shape[0], d), dtype=torch.bfloat16, device=DEV)
    assert qmm.embed(ids.long(), *w, out=out) is out and torch.equal(out, got)


def test_group_sums_rows_alone_and_against_float64():
    x = inputs(33, 4096, 5)
    xs = qmm.group_sums(x)
    assert xs.shape == (33, 64)
    for r in (0, 16, 32):
        assert torch.equal(qmm.group_sums(x[r:r + 1])[0], xs[r])
    ref = x.double().view(33, 64, 64).sum(-1)
    assert (xs.double() - ref).abs().max().item() < 1e-4


def test_expert_tables_pack_for_the_grouped_kernels():
    """Nine experts, the last appended as the loader appends the shared one: the packed tables unpack to MLX's."""

    gate, up = mlx_weights(512, 2048, 31, (8,)), mlx_weights(512, 2048, 32, (8,))
    down = mlx_weights(2048, 512, 33, (8,))
    extra = mlx_weights(512, 2048, 34), mlx_weights(512, 2048, 35), mlx_weights(2048, 512, 36)
    ex = qmm.make_experts(gate, up, down, extra)
    assert (ex.count, ex.width, ex.dims, ex.gs, ex.swiglu) == (9, 512, 2048, 64, True)
    for m, src in ((0, gate), (1, up)):
        got = grouped.unpack(ex.up[:, :, :, m].contiguous(), 64)
        for a, b, e in zip(got, src, extra[m]):
            assert torch.equal(a[:8], b) and torch.equal(a[8], e)
    got = grouped.unpack(ex.down[:, :, :, 0].contiguous(), 64)
    for a, b, e in zip(got, down, extra[2]):
        assert torch.equal(a[:8], b) and torch.equal(a[8], e)
    tables = [tuple(torch.cat([a, b[None]]) for a, b in zip(t, x)) for t, x in zip((gate, up, down), extra)]
    same = qmm.make_experts(*tables)
    assert torch.equal(same.up, ex.up) and torch.equal(same.down, ex.down)


def test_wrong_formats_are_refused():
    w, s, b = mlx_weights(64, 2048, 1)
    with pytest.raises(ValueError):
        qmm.make_q4(w, s[:, :16].contiguous(), b[:, :16].contiguous())       # group-128 scales
    wide_s = s.repeat_interleave(2, dim=1)
    with pytest.raises(ValueError):
        qmm.make_q4(w, wide_s, wide_s)                                     # group-32 scales
    q = qmm.make_q4(w, s, b)
    with pytest.raises(TypeError):
        qmm.matmul(inputs(1, 2048, 1), q, sk=2)                             # the K split is the shape's
    with pytest.raises(TypeError):
        qmm.matmul(inputs(1, 2048, 1), torch.zeros((64, 2048), device=DEV))
