"""Qwen3.6-35B-A3B attention on Flash Next's CUDA kernels: dense at any context (no sparse attention, no indexer),
a row's bits never depend on the window, and the prep, attention and gate match an fp32 reference. Flash Next's
own outputs keep their bits (SHA-256 of fixed random inputs, recorded on the base commit for each GPU).

Recording Flash Next's hashes on a new GPU or after a Triton upgrade: on the base commit run
``python tests/cuda/test_qwen36moe_attention.py`` and add its output to ``FN_HASHES`` under the device's
compute capability."""

import hashlib

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen4_exp.cuda import attention as att, glue  # noqa: E402

DEV = "cuda"
BF, F32 = torch.bfloat16, torch.float32
EPS = 1e-6


def _sha(*tensors: torch.Tensor) -> str:
    h = hashlib.sha256()
    for t in tensors:
        t = t.detach().contiguous()
        h.update(f"{t.dtype}{tuple(t.shape)}".encode())
        h.update(t.view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def _inv_freq(rotary_dim: int = 64, theta: float = 1e7) -> torch.Tensor:
    """As the loaders build it: theta^(-i / (rotary_dim / 2)) in float64, then fp32."""

    half = rotary_dim // 2
    inv = torch.tensor(theta, dtype=torch.float64) ** (-torch.arange(0, half, dtype=torch.float64) / half)
    return inv.to(F32).to(DEV)


def _pos(p0: int) -> torch.Tensor:
    return torch.tensor([p0], dtype=torch.int32, device=DEV)


# -- Flash Next keeps its bits --------------------------------------------------------------------------------
FN_H, FN_HK, FN_D, FN_NI, FN_DI = 24, 2, 256, 4, 128
FN_CASES = {                     # name: (capacity, P0, rows)
    "dense": (2048, 1000, 7),    # capacity within the indexer's budget: sparse attention off
    "mixed": (4096, 2046, 8),    # sparse attention on; rows 0-4 dense, rows 5-7 sparse (more than 512 blocks)
    "sparse": (4096, 2600, 6),   # every row sparse
}


def fn_outputs(case: str) -> dict[str, str]:
    """Flash Next's attention path on fixed random inputs: prep (q, caches, indexer q and key), block selection
    (when sparse attention is on), attention and gate, each stage's outputs hashed."""

    cap, p0, rows = FN_CASES[case]
    g = torch.Generator(device=DEV).manual_seed(1000 + p0)
    pw = FN_H * 2 * FN_D + 2 * FN_HK * FN_D + (FN_NI + 1) * FN_DI
    p = torch.randn((rows, pw), generator=g, device=DEV).to(BF)
    kc = torch.randn((cap, FN_HK, FN_D), generator=g, device=DEV).to(BF)
    vc = torch.randn((cap, FN_HK, FN_D), generator=g, device=DEV).to(BF)
    ikc = torch.randn((cap, FN_DI), generator=g, device=DEV).to(BF)
    qs, ks = ((1 + 0.05 * torch.randn((FN_D,), generator=g, device=DEV)).float() for _ in range(2))
    iqs, iks = ((1 + 0.05 * torch.randn((FN_DI,), generator=g, device=DEV)).float() for _ in range(2))
    inv, pos = _inv_freq(), _pos(p0)
    q = torch.zeros((rows, FN_H, FN_D), dtype=BF, device=DEV)
    iq = torch.zeros((rows, FN_NI, FN_DI), dtype=BF, device=DEV)
    glue.attn_prep(p, pos, qs, ks, iqs, inv, q, kc, vc, iq, ikc, EPS, q_heads=FN_H, kv_heads=FN_HK,
                   head_dim=FN_D, index_heads=FN_NI, index_dim=FN_DI)
    out = {"prep": _sha(q, kc, vc, iq, ikc)}
    scratch = att.AttnScratch(rows, FN_H, FN_D, cap, DEV)
    if scratch.qsa:
        pooled = torch.zeros((cap // 4, FN_DI), dtype=BF, device=DEV)
        for start in range(0, p0, 64):   # the blocks before the window, pooled as earlier forwards would have
            att._pool[(64 // 4 + 2,)](ikc, pooled, _pos(start), iks, inv, EPS, min(64, p0 - start), DI=FN_DI,
                                      HALF=32, RATIO=4, num_warps=1)
        att.qsa_select(iq, ikc, pooled, pos, iks, inv, EPS, scratch, rows)
        out["select"] = _sha(pooled, scratch.ids, scratch.nk, scratch.sparse, scratch.scores)
    o = att.attention(q, kc, vc, pos, scratch, rows, FN_D ** -0.5)
    out["attn"] = _sha(o[:rows])
    gated = torch.empty((rows, FN_H * FN_D), dtype=BF, device=DEV)
    xs = torch.empty((rows, FN_H * FN_D // 32), dtype=F32, device=DEV)
    glue.attn_gate(o, p, gated, xs, q_heads=FN_H, head_dim=FN_D)
    out["gate"] = _sha(gated, xs)
    return out


# recorded on the base commit (bb4b4a3, TensorFold 0.3.4.1) by this file's __main__, per compute capability
FN_HASHES: dict[tuple[int, int], dict[str, dict[str, str]]] = {
    (12, 0): {                   # RTX 5090, Triton 3.7.1 (NVIDIA PyTorch 26.07)
        "dense": {
            "prep": "42cf3a56d37c416a7e375b22d4d5cced1e05ff45a2dd5429c63924639a17ebec",
            "attn": "b72122e02bad1049778f7feff0973c283f14f3a112e37a4aebf3f7fc78f8696e",
            "gate": "1e90cbc40fe51d34c25a0db0a26bc68c61573f8473acc3f181529ee35c71e73d",
        },
        "mixed": {
            "prep": "c7446018042986d880a37e41a70278a3a80e37637c09aaac54954ed3ae3acc1e",
            "select": "f553d759760c1e8db7c668608dd099d4ac122d6f2aa72c4e5a947e8b5650585e",
            "attn": "9c8b4399e8edd0970debdf84237bfefedc119a7267265a96e0b9dd587f198999",
            "gate": "295f08dde8410a8e44180981a0901b835533666d0e3895ad7aa9e8b854cd406f",
        },
        "sparse": {
            "prep": "dc2cc6b02a80bf60e6487429c796a6b0e3e291864cac9c5fb302e4c42acc1d88",
            "select": "f0633a885d8bc7fc5d9c819ea8e7d2d778baea519d1be1c8d10498527f982e31",
            "attn": "129d0448274ca9538bb99e817d55911ab5027211dabdb88751c4e5d8b681066d",
            "gate": "947e3c71a83d5809f205635f9d9d5b6d62b358bcc136086e623d7c581d42cecf",
        },
    },
}


@pytest.mark.parametrize("case", sorted(FN_CASES))
def test_flash_next_attention_keeps_its_bits(case):
    arch = torch.cuda.get_device_capability()
    if arch not in FN_HASHES:
        pytest.skip(f"no Flash Next hashes recorded for sm_{arch[0]}{arch[1]}: run this file on the base commit")
    assert fn_outputs(case) == FN_HASHES[arch][case]


# -- Qwen3.6: 16 query heads, 2 KV heads, head_dim 256, [q_h | gate_h] per head, rotary on 64 of 256 dims -----
H, HK, D = 16, 2, 256
PW = H * 2 * D + 2 * HK * D          # [q|gate pairs | k | v]: no indexer columns


class _Inputs:
    """A window's projection rows, the cache before it (normed, rotated keys; values) and the norm weights as
    the checkpoint stores them (zero-centred, bf16) with the fp32 scales 1 + w the loader passes."""

    def __init__(self, rows: int, cap: int, seed: int) -> None:
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.p = torch.randn((rows, PW), generator=g, device=DEV).to(BF)
        self.kc = torch.randn((cap, HK, D), generator=g, device=DEV).to(BF)
        self.vc = torch.randn((cap, HK, D), generator=g, device=DEV).to(BF)
        self.wq = (0.1 * torch.randn((D,), generator=g, device=DEV)).to(BF)
        self.wk = (0.1 * torch.randn((D,), generator=g, device=DEV)).to(BF)
        self.qs, self.ks = 1.0 + self.wq.float(), 1.0 + self.wk.float()
        self.cap = cap


def _run(x: _Inputs, rows: slice, p0: int, kc: torch.Tensor, vc: torch.Tensor, cap: int | None = None,
         group: int = 64):
    """Prep (writes the rows' keys and values to kc/vc at P0...), dense attention and gate for x.p[rows]."""

    p = x.p[rows]
    r = p.shape[0]
    pos = _pos(p0)
    q = torch.empty((r, H, D), dtype=BF, device=DEV)
    glue.attn_prep(p, pos, x.qs, x.ks, None, _inv_freq(), q, kc, vc, None, None, EPS, q_heads=H, kv_heads=HK,
                   head_dim=D)
    scratch = att.AttnScratch(r, H, D, cap or x.cap, DEV, sparse=False)
    assert not scratch.qsa and scratch.scores is None
    o = att.attention(q, kc, vc, pos, scratch, r, D ** -0.5).clone()
    gated = torch.empty((r, H * D), dtype=BF, device=DEV)
    xs = torch.empty((r, H * D // group), dtype=F32, device=DEV)
    glue.attn_gate(o, p, gated, xs, q_heads=H, head_dim=D, group=group)
    return q, o, gated, xs


def _serial(x: _Inputs, p0: int, n: int, cap: int | None = None):
    """Rows 0..n-1 one at a time, each at its own position, each writing its key before the next row."""

    kc, vc = x.kc.clone(), x.vc.clone()
    steps = [_run(x, slice(r, r + 1), p0 + r, kc, vc, cap) for r in range(n)]
    return steps, kc, vc


def _assert_window_matches(x: _Inputs, p0: int, m: int, steps, kc_s, vc_s, cap: int | None = None) -> None:
    kc, vc = x.kc.clone(), x.vc.clone()
    p_before = x.p.clone()
    q, o, gated, xs = _run(x, slice(0, m), p0, kc, vc, cap)
    for r in range(m):
        sq, so, sg, sx = steps[r]
        assert torch.equal(q[r], sq[0]), (p0, m, r, "q")
        assert torch.equal(o[r], so[0]), (p0, m, r, "attention")
        assert torch.equal(gated[r], sg[0]) and torch.equal(xs[r], sx[0]), (p0, m, r, "gate")
    assert torch.equal(kc[:p0 + m], kc_s[:p0 + m]) and torch.equal(vc[:p0 + m], vc_s[:p0 + m]), (p0, m)
    assert torch.equal(kc[p0 + m:], x.kc[p0 + m:]) and torch.equal(vc[p0 + m:], x.vc[p0 + m:]), (p0, m)
    assert torch.equal(x.p, p_before)


@pytest.mark.parametrize("p0", [0, 450, 3990])
def test_rows_alone_match_rows_in_windows(p0):
    """Windows of 2, 3, 16, 17, 64 and 128 rows starting at P0 (450: across the 512-key chunk boundary; 3990:
    across 4096, a context where Flash Next's attention would be sparse): each row equals the row alone."""

    x = _Inputs(128, 4608, seed=2000 + p0)
    steps, kc_s, vc_s = _serial(x, p0, 128)
    for m in (2, 3, 16, 17, 64, 128):
        _assert_window_matches(x, p0, m, steps, kc_s, vc_s)


def _reference(x: _Inputs, p0: int, rows: int, keep: int | None = None):
    """fp32 on the CPU, as the model defines it: RMSNorm x * rsqrt(mean(x^2) + eps) * (1 + w); rotate-half RoPE on
    the first 64 dims at theta 1e7; SDPA over the cached keys and the window's causal keys (the last ``keep`` keys
    only, when given); out = attention * sigmoid(gate)."""

    F = torch.nn.functional
    pr = x.p[:rows].float().cpu()
    qg = pr[:, :H * 2 * D].reshape(rows, H, 2, D)
    qr, gate = qg[:, :, 0], qg[:, :, 1]
    kr = pr[:, H * 2 * D:H * 2 * D + HK * D].reshape(rows, HK, D)
    vr = pr[:, H * 2 * D + HK * D:].reshape(rows, HK, D)

    def norm(t, w):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + w.float().cpu())

    pos = torch.arange(p0, p0 + rows, dtype=torch.float64)
    inv = torch.tensor(1e7, dtype=torch.float64) ** (-torch.arange(0, 32, dtype=torch.float64) / 32)
    ang = pos[:, None] * inv[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1).float()[:, None, :]
    sin = torch.cat([ang.sin(), ang.sin()], -1).float()[:, None, :]

    def rope(t):
        rot = t[..., :64]
        half = torch.cat([-rot[..., 32:], rot[..., :32]], -1)
        return torch.cat([rot * cos + half * sin, t[..., 64:]], -1)

    q, k = rope(norm(qr, x.wq)), rope(norm(kr, x.wk))
    keys = torch.cat([x.kc[:p0].float().cpu(), k])              # [p0 + rows, HK, D]
    vals = torch.cat([x.vc[:p0].float().cpu(), vr])
    n = p0 + rows
    mask = torch.ones((rows, n), dtype=torch.bool).tril(diagonal=p0)
    if keep is not None:
        ends = torch.arange(p0 + 1, n + 1)[:, None]
        mask &= torch.arange(n)[None, :] >= ends - keep
    Q = q.transpose(0, 1)[None]                                  # [1, H, rows, D]
    K = keys.transpose(0, 1).repeat_interleave(H // HK, 0)[None]  # query head h reads KV head h // 8
    V = vals.transpose(0, 1).repeat_interleave(H // HK, 0)[None]
    o = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, scale=D ** -0.5)[0].transpose(0, 1)
    return q, k, vr, o, (o * torch.sigmoid(gate)).reshape(rows, H * D)


def _err(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    return (got.float().cpu() - ref).abs().max().item(), ref.abs().max().item()


@pytest.mark.parametrize("p0", [3, 700, 4090])
def test_prep_attention_and_gate_match_fp32(p0):
    """q/k RMSNorm (1 + w, fp32, one bf16 rounding), partial rotary, dense attention and the sigmoid gate against
    an fp32 SDPA reference; values are copied to the cache bit for bit."""

    rows = 5
    x = _Inputs(rows, 4608, seed=3000 + p0)
    kc, vc = x.kc.clone(), x.vc.clone()
    q, o, gated, xs = _run(x, slice(0, rows), p0, kc, vc)
    rq, rk, rv, ro, rg = _reference(x, p0, rows)
    assert torch.equal(vc[p0:p0 + rows], x.p[:, H * 2 * D + HK * D:].reshape(rows, HK, D))
    errs = {"q": _err(q, rq), "k": _err(kc[p0:p0 + rows], rk), "attention": _err(o, ro), "gated": _err(gated, rg)}
    print(f"\nfp32 max abs error at P0={p0}: " + ", ".join(f"{k} {e:.3g} (max |ref| {m:.3g})"
                                                         for k, (e, m) in errs.items()))
    for k, (e, m) in errs.items():
        assert e <= m * 2 ** -6, (k, e, m)
    sums = gated.float().reshape(rows, -1, 64).sum(-1)
    assert torch.allclose(xs, sums, rtol=1e-5, atol=1e-4)


def test_rows_at_the_chunk_boundary_are_dense_and_exact():
    """Rows whose last key is 511, 512 and 513 (one full 512-key chunk, then one and two keys into the next):
    alone equals in the window, and each reads every one of its keys (the fp32 reference over all of them)."""

    p0, rows = 509, 8
    x = _Inputs(rows, 1024, seed=4000)
    steps, kc_s, vc_s = _serial(x, p0, rows)
    for m in (3, 5, 8):
        _assert_window_matches(x, p0, m, steps, kc_s, vc_s)
    _, _, _, ro, _ = _reference(x, p0, rows)
    for pos in (511, 512, 513):
        e, m = _err(steps[pos - p0][1][0], ro[pos - p0])
        assert e <= m * 2 ** -6, (pos, e, m)


def test_a_4k_context_stays_dense_and_exact():
    """P0 = 4087, 17 rows (across 4096) in a 4608- and an 8192-key cache: no sparse attention (Flash Next's scratch
    at this capacity would turn it on), the same bits alone, in the window and at either capacity, and the fp32
    reference over every key (a reference that drops the oldest keys is far off)."""

    p0, rows = 4087, 17
    assert att.AttnScratch(1, H, D, 4608, DEV).qsa                  # the sparse default at this capacity
    x = _Inputs(rows, 8192, seed=5000)
    steps, kc_s, vc_s = _serial(x, p0, rows, cap=4608)
    for m in (2, 16, 17):
        _assert_window_matches(x, p0, m, steps, kc_s, vc_s, cap=4608)
        _assert_window_matches(x, p0, m, steps, kc_s, vc_s, cap=8192)
    o = torch.stack([s[1][0] for s in steps])
    _, _, _, ro, _ = _reference(x, p0, rows)
    _, _, _, rtrunc, _ = _reference(x, p0, rows, keep=2048)
    e, m = _err(o, ro)
    far, _ = _err(o, rtrunc)
    assert e <= m * 2 ** -6, (e, m)
    assert far > 8 * e, (far, e)


if __name__ == "__main__":
    arch = torch.cuda.get_device_capability()
    print(f"    {arch}: {{")
    for name in sorted(FN_CASES):
        print(f"        {name!r}: {fn_outputs(name)!r},")
    print("    },")
