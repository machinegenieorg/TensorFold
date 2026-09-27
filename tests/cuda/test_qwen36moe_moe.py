"""Qwen3.6-35B-A3B MoE kernels on synthetic weights (256 experts, top 8, width 512, the shared expert as expert
256): router logits, selection and grouping, and every (row, slot) output give a row the same bits alone, in any
window and in any row order; exact ties go to the lower expert id; the combine adds the slots in pick order and
then the shared expert; results agree with a float64 reference.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen3_5_moe.cuda import moe, qmm  # noqa: E402

DEV = "cuda"
E, K_TOP, W, D = moe.EXPERTS, moe.TOP_K, moe.WIDTH, moe.HIDDEN
SLOTS = K_TOP + 1
ROWS = [1, 2, 3, 16, 17, 64, 128]


def mlx_weights(n: int, k: int, seed: int, lead: tuple = ()):
    g = torch.Generator(device=DEV).manual_seed(seed)
    words = torch.randint(-(2**31), 2**31 - 1, (*lead, n, k // 8), generator=g, device=DEV,
                          dtype=torch.int64).to(torch.int32)
    scales = (torch.rand((*lead, n, k // 64), generator=g, device=DEV) * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((*lead, n, k // 64), generator=g, device=DEV) * 0.02).to(torch.bfloat16)
    return words, scales, biases


def inputs(m: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn((m, D), generator=g, device=DEV) * scale).to(torch.bfloat16)


def router_table(seed: int) -> torch.Tensor:
    """fp32 [257, 2048] as the loader builds it: an 8-bit group-64 dequantization, fp32(s) * q + fp32(b)."""

    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randint(0, 256, (E + 1, D), generator=g, device=DEV).float()
    s = (torch.rand((E + 1, D // 64), generator=g, device=DEV) * 2e-4 + 1e-5).to(torch.bfloat16).float()
    b = (-128 * s + torch.randn((E + 1, D // 64), generator=g, device=DEV) * 1e-4).to(torch.bfloat16).float()
    return (s.repeat_interleave(64, 1) * q + b.repeat_interleave(64, 1)).contiguous()


_EXPERTS = {}


def experts():
    """The 257-expert table (the loader's layout: the shared expert is expert 256), built once per session."""

    if not _EXPERTS:
        gate = mlx_weights(W, D, 11, (E + 1,))
        up = mlx_weights(W, D, 12, (E + 1,))
        down = mlx_weights(D, W, 13, (E + 1,))
        _EXPERTS.update(ex=qmm.make_experts(gate, up, down), raw=(gate, up, down))
    return _EXPERTS["ex"], _EXPERTS["raw"]


def run(x: torch.Tensor, rows_table: torch.Tensor, buf=None, **launch):
    ex, _ = experts()
    buf = buf or moe.buffers(x.shape[0], DEV)
    sub = moe.moe(x, qmm.group_sums(x), rows_table, ex, buf, **launch)
    return sub


# -- router ----------------------------------------------------------------------------------------------------
def test_router_rows_do_not_depend_on_the_window():
    table = router_table(1)
    top = 200
    x = inputs(top, 2)
    alone = torch.cat([moe.router(x[r:r + 1], table) for r in range(top)])
    for m in ROWS + [top]:
        assert torch.equal(moe.router(x[:m], table), alone[:m]), m
    perm = torch.randperm(top, generator=torch.Generator().manual_seed(4)).to(DEV)
    assert torch.equal(moe.router(x[perm], table), alone[perm])
    for be, bk, warps, stages in ((16, 32, 4, 2), (32, 64, 8, 2), (64, 32, 4, 3), (16, 64, 2, 2)):
        got = moe.router(x[:40], table, block_e=be, bk=bk, num_warps=warps, num_stages=stages)
        assert torch.equal(got, alone[:40]), (be, bk, warps, stages)
    ref = x.double() @ table.double().T
    err = (alone.double() - ref).abs().max().item()
    print(f"\nrouter: max|logit| {ref.abs().max().item():.3f}, max err vs float64 {err:.3e}")
    assert err <= ref.abs().max().item() * 1e-5
    # the logit is one fp32 fused multiply-add chain over k = 0, 1, ..., D - 1 (exact emulation)
    for r, e in ((0, 0), (7, 128), (199, E)):
        assert _fma_chain(x[r].float().cpu().numpy(), table[e].cpu().numpy()) == alone[r, e].item(), (r, e)


def _round_f32(f):
    """A rational rounded to the nearest fp32 (ties to even)."""

    from fractions import Fraction

    import numpy as np

    c = np.float32(float(f))
    best = None
    for cand in (np.nextafter(c, np.float32(-np.inf)), c, np.nextafter(c, np.float32(np.inf))):
        dist = abs(Fraction(float(cand)) - f)
        even = (int(np.array(cand, dtype=np.float32).view(np.int32)) & 1) == 0
        if best is None or dist < best[0] or (dist == best[0] and even):
            best = (dist, cand)
    return best[1]


def _fma_chain(x, w) -> float:
    from fractions import Fraction

    import numpy as np

    acc = np.float32(0.0)
    for k in range(x.shape[0]):
        acc = _round_f32(Fraction(float(x[k])) * Fraction(float(w[k])) + Fraction(float(acc)))
    return float(acc)


def test_selection_ties_go_to_the_lower_id():
    buf = moe.buffers(4, DEV)
    logits = torch.randn((4, E + 1), generator=torch.Generator(device=DEV).manual_seed(3), device=DEV) * 0.1
    logits[0, [5, 3, 200]] = 4.0                                     # three equal at the top
    logits[1, [17, 42, 99, 150, 201, 255, 7]] = torch.tensor([9.0, 8, 7, 6, 5, 4, 3], device=DEV)
    logits[1, [230, 10, 250]] = 2.0                                  # tie at the cut: 10 in, 230 and 250 out
    logits[2, :E] = 1.5                                              # all equal: experts 0..7, weights 1/8
    logits[3, [255, 254, 253, 252, 251, 250, 249, 248]] = 3.0       # equal, all at the top end
    moe.select(logits, buf)
    pick = buf.pick.tolist()
    assert pick[0][:3] == [3, 5, 200]
    assert pick[1][:K_TOP] == [17, 42, 99, 150, 201, 255, 7, 10]
    assert pick[2][:K_TOP] == list(range(8))
    assert pick[3][:K_TOP] == list(range(248, 256))
    assert all(p[K_TOP] == E for p in pick)
    assert torch.equal(buf.wts[2, :K_TOP], torch.full((K_TOP,), 0.125, device=DEV))
    assert buf.wts[0, 0] == buf.wts[0, 1] == buf.wts[0, 2]
    # the weights: softmax over the 256, top 8, renormalised; rounded to bf16
    for r in range(4):
        top = logits[r, pick[r][:K_TOP]].double()
        want = torch.softmax(logits[r, :E].double(), 0)[pick[r][:K_TOP]]
        want = (want / want.sum()).float().to(torch.bfloat16).float()
        assert torch.allclose(buf.wts[r, :K_TOP], want, atol=0, rtol=2 ** -7), r
        assert bool((top[:-1] >= top[1:]).all())                     # largest first
        sg = torch.sigmoid(logits[r, E].to(torch.bfloat16).float()).to(torch.bfloat16).float()
        assert buf.wts[r, K_TOP] == sg


def test_router_ties_from_equal_rows_go_to_the_lower_id():
    """Equal router rows give bit-equal logits on every row; at the top-8 cut the lower ids win."""

    x = inputs(6, 8)
    table = router_table(9) * 0.01
    xf = x[0].double()
    unit = (xf / xf.dot(xf)).float()
    for i, e in enumerate((17, 42, 99, 150, 201, 255)):
        table[e] = unit * (10 + i)
    for e in (3, 100, 200):
        table[e] = unit * 5
    ex_logits = moe.router(x, table)
    assert torch.equal(ex_logits[:, 3], ex_logits[:, 100]) and torch.equal(ex_logits[:, 3], ex_logits[:, 200])
    for m in (1, 6):
        buf = moe.buffers(m, DEV)
        moe.select(moe.router(x[:m], table), buf)
        assert buf.pick[0, :K_TOP].tolist() == [255, 201, 150, 99, 42, 17, 3, 100]


def test_grouping_is_ascending_with_members_in_row_order():
    table = router_table(5)
    for m in (1, 2, 17, 64):
        buf = moe.buffers(m, DEV)
        moe.select(moe.router(inputs(m, 10 + m), table), buf)
        pick = buf.pick.tolist()
        count = int(buf.group.count.item())
        ids = buf.group.ids[:count].tolist()
        assert ids == sorted({e for p in pick for e in p})
        assert E in ids
        members = buf.group.members.tolist()
        for u, e in enumerate(ids):
            want = [r * 32 + s for r in range(m) for s in range(SLOTS) if pick[r][s] == e]
            assert members[u][:len(want)] == want and all(v == -1 for v in members[u][len(want):]), (m, e)
        for u in range(count, buf.maxu):
            assert all(v == -1 for v in members[u])


# -- experts ---------------------------------------------------------------------------------------------------
def test_moe_rows_do_not_depend_on_the_window():
    table = router_table(6)
    top = max(ROWS)
    x = inputs(top, 20)
    big = moe.buffers(top, DEV)
    singles = []
    for r in range(top):
        sub = run(x[r:r + 1], table, big)
        singles.append((sub.pick[0].clone(), sub.wts[0].clone(), sub.y[0].clone(), sub.act[0].clone()))
    for m in ROWS:
        sub = run(x[:m], table, big)                        # a view of the 128-row buffer
        own = run(x[:m], table)                             # a buffer of exactly m rows
        for got in (sub, own):
            for r in range(m):
                p, w, y, a = singles[r]
                assert torch.equal(got.pick[r], p) and torch.equal(got.wts[r], w), (m, r)
                assert torch.equal(got.act[r], a) and torch.equal(got.y[r], y), (m, r)
    perm = torch.randperm(64, generator=torch.Generator().manual_seed(1)).to(DEV)
    sub = run(x[perm], table, moe.buffers(64, DEV))
    for i, r in enumerate(perm.tolist()):
        assert torch.equal(sub.y[i], singles[r][2]) and torch.equal(sub.pick[i], singles[r][0]), (i, r)
    # windows with a repeated row: the same (row, expert) pair twice, same bits
    dup = torch.cat([x[:3], x[1:2], x[1:2]])
    sub = run(dup, table)
    for i, r in enumerate((0, 1, 2, 1, 1)):
        assert torch.equal(sub.y[i], singles[r][2])


def test_moe_launch_settings_keep_the_bits():
    """1-2 row windows use 32-wide programs and other stages (design risk 4): the K order is the same, so the
    bits are too, for every setting."""

    table = router_table(7)
    ex, _ = experts()
    for m in (1, 2, 5):
        x = inputs(m, 30 + m)
        ref = run(x, table)
        y0, a0, s0 = ref.y.clone(), ref.act.clone(), ref.axs.clone()
        xs = qmm.group_sums(x)
        for block_n, gpi, warps, stages in ((32, 1, 4, 2), (32, 4, 8, 3), (64, 2, 8, 4), (64, 8, 4, 2)):
            ref.y.zero_()
            ref.act.zero_()
            qmm.moe_gateup(x, xs, ex, ref.group, ref.act, ref.axs, gpi=gpi, num_warps=warps, num_stages=stages,
                           block_n=block_n)
            qmm.moe_down(ref.act, ref.axs, ex, ref.group, ref.y, gpi=gpi, num_warps=warps, num_stages=stages,
                         block_n=block_n)
            assert torch.equal(ref.act, a0) and torch.equal(ref.axs, s0), (m, block_n, gpi, warps, stages)
            assert torch.equal(ref.y, y0), (m, block_n, gpi, warps, stages)


def test_moe_matches_a_float64_reference():
    table = router_table(8)
    _, (gate, up, down) = experts()
    x = inputs(3, 40)
    sub = run(x, table)
    xd = x.double()
    logits = xd @ table.double().T
    worst = 0.0
    for r in range(3):
        order = sorted(range(E), key=lambda e: (-logits[r, e].item(), e))[:K_TOP]
        assert sub.pick[r, :K_TOP].tolist() == order
        for s, e in enumerate(order + [E]):
            gw, uw, dw = (qmm.dequantize(t[0][e], t[1][e], t[2][e]).double() for t in (gate, up, down))
            act = torch.nn.functional.silu(xd[r] @ gw.T) * (xd[r] @ uw.T)
            y = act @ dw.T
            err = (sub.y[r, s].double() - y).abs().max().item()
            worst = max(worst, err / y.abs().max().item())
            assert err <= y.abs().max().item() * 2 ** -5 + 1e-3, (r, s, err)
        ref_act_sums = sub.act[r].float().view(SLOTS, W // 32, 32).sum(-1)
        assert (sub.axs[r] - ref_act_sums).abs().max().item() < 1e-3
    print(f"\nexperts: worst (row, slot) max err vs float64 {worst:.2e} of that slot's max")


# -- combine ---------------------------------------------------------------------------------------------------
def _slots_and_pow2_weights(rows: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    y = torch.randn((rows, SLOTS, D), generator=g, device=DEV) * torch.logspace(0, 3, SLOTS, device=DEV)[None, :, None]
    wts = 2.0 ** -torch.randint(0, 6, (rows, SLOTS), generator=g, device=DEV).float()
    return y.contiguous(), wts.contiguous()


def test_combine_adds_slots_in_pick_order_then_the_shared_expert():
    """Power-of-two weights make every product exact, so only the order of the fp32 additions decides the bits."""

    rows = 17
    y, wts = _slots_and_pow2_weights(rows, 1)
    got = moe.combine(y, wts)
    acc = torch.zeros((rows, D), device=DEV)
    for k in range(SLOTS):
        acc = acc + y[:, k] * wts[:, k:k + 1]
    assert torch.equal(got, acc.to(torch.bfloat16))
    rev = torch.zeros((rows, D), device=DEV)
    for k in reversed(range(SLOTS)):
        rev = rev + y[:, k] * wts[:, k:k + 1]
    assert not torch.equal(rev, acc)                     # the check can see the order
    for r in (0, 9, 16):
        assert torch.equal(moe.combine(y[r:r + 1].contiguous(), wts[r:r + 1].contiguous())[0], got[r])
    # on a real MoE window: the kernel's output is the combination of its slots
    table = router_table(2)
    sub = run(inputs(5, 3), table)
    ref = torch.zeros((5, D), device=DEV, dtype=torch.float64)
    for k in range(SLOTS):
        ref += sub.y[:, k].double() * sub.wts[:, k:k + 1].double()
    assert (moe.combine(sub.y, sub.wts).double() - ref).abs().max().item() <= ref.abs().max().item() * 2 ** -8


def test_combine_add_rmsnorm_fuses_the_residual_and_the_next_norm():
    rows, eps = 17, 1e-6
    y, wts = _slots_and_pow2_weights(rows, 2)
    g = torch.Generator(device=DEV).manual_seed(3)
    h = (torch.randn((rows, D), generator=g, device=DEV) * 20).to(torch.bfloat16)
    norm = 1 + 0.1 * torch.randn((D,), generator=g, device=DEV)
    branch = moe.combine(y, wts)
    h2, normed, xs = moe.combine_add_rmsnorm(y, wts, h, norm, eps)
    assert torch.equal(h2, (h.float() + branch.float()).to(torch.bfloat16))
    hd = h2.double()
    ref = hd * torch.rsqrt(hd.pow(2).mean(-1, keepdim=True) + eps) * norm.double()
    assert (normed.double() - ref).abs().max().item() <= ref.abs().max().item() * 2 ** -8
    assert (xs.double() - normed.double().view(rows, D // 64, 64).sum(-1)).abs().max().item() < 1e-3
    for r in (0, 8, 16):
        one = moe.combine_add_rmsnorm(y[r:r + 1].contiguous(), wts[r:r + 1].contiguous(), h[r:r + 1].contiguous(),
                                      norm, eps)
        assert torch.equal(one[0][0], h2[r]) and torch.equal(one[1][0], normed[r]) and torch.equal(one[2][0], xs[r])
    # in place on the residual
    hh = h.clone()
    out = moe.combine_add_rmsnorm(y, wts, hh, norm, eps, h_out=hh)
    assert out[0] is hh and torch.equal(hh, h2) and torch.equal(out[1], normed)
