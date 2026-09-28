"""Qwen3.6-35B-A3B's Gated DeltaNet on Flash Next's chain: serial bits in windows and replays, fp32, FN's own bits."""

import hashlib

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen4_exp.cuda import gdn  # noqa: E402

DEV = "cuda"
NK, NV, DK, DV = 16, 32, 128, 128
EPS = 1e-6


def _inputs(nk: int, nv: int, rows: int, seed: int):
    """Projection rows, conv state and weights, fp32 state, drawn on the CPU; odd heads decay slowly, even fast."""

    g = torch.Generator().manual_seed(seed)
    conv, pw = gdn.widths(nk, nv)

    def rnd(shape, scale):
        return torch.randn(shape, generator=g) * scale

    slow = 2.0 * (torch.arange(nv) % 2)
    p = rnd((rows, pw), 0.5).to(torch.bfloat16)
    cs = rnd((3, conv), 0.5).to(torch.bfloat16)
    cw = rnd((conv, 4), 0.3).to(torch.bfloat16)
    state = rnd((nv, DV, DK), 0.05)
    a_log = rnd((nv,), 0.5) - slow
    dt = rnd((nv,), 0.5) - slow
    nw = (1 + rnd((DV,), 0.1)).to(torch.bfloat16)
    return [x.to(DEV) for x in (p, cs, cw, state, a_log, dt, nw)]


def _outs(rows: int, nv: int, sums: bool = False):
    """The chain's outputs: bf16 [rows, nv * 128] and, with ``sums``, their 32-group sums (else None)."""

    out = torch.empty((rows, nv * DV), dtype=torch.bfloat16, device=DEV)
    xs = torch.empty((rows, nv * DV // 32), dtype=torch.float32, device=DEV) if sums else None
    return out, xs


def _bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()


# SHA-256 of Flash Next's chain outputs, replay inputs and replays per compute capability: flashnext_digests() on v0.3.5
CASES = ((16, 48, 1), (16, 48, 6), (16, 48, 17), (8, 24, 5))
FLASHNEXT = {
    (12, 0): {  # RTX 5090 (sm_120), tensorfold-dev:26.07: torch 2.13.0a0 nv26.07, CUDA 13.3
        (16, 48, 1): "f7d64b9ad137763ecceed0920ab72d6faf34c5c6e4dfe3eee9d9e385b216736f",
        (16, 48, 6): "6aed0f8cc3db5474d7b0cc5797ede37cc63099600dc4a2b9d1f81ee36698a6d4",
        (16, 48, 17): "78d1a4274dca019060ec64392418be539b7d8b691f02d114852b9028ede86673",
        (8, 24, 5): "f2efd765f0ea895b4e06459bbfd242e38e453bc370c8a36fd3620e61ff2bebdf",
    },
    # (12, 1), GB10: the entries come from the GB10 run of flashnext_digests() on v0.3.5
}


def _flashnext_digest(nk: int, nv: int, rows: int, **gate) -> str:
    p, cs, cw, state, a_log, dt, nw = _inputs(nk, nv, rows, 7000 + 100 * nk + rows)
    win = gdn.GDNScratch(rows, DEV, nk, nv)
    out, xs = _outs(rows, nv, sums=True)
    out_state = torch.empty_like(state)
    gdn.chain(p, cs, cw, state, a_log, dt, nw, EPS, rows, win, out_state, out, xs, **gate)
    h = hashlib.sha256()
    for t in (out, xs, win.k, win.v, win.g, win.b):
        h.update(_bytes(t[:rows]))
    h.update(_bytes(out_state))
    for keep in sorted({1, (rows + 1) // 2, rows}):
        rep = torch.empty_like(state)
        gdn.replay(state, win, keep, rep)
        h.update(_bytes(rep))
    return h.hexdigest()


def flashnext_digests() -> dict:
    return {case: _flashnext_digest(*case) for case in CASES}


@pytest.mark.parametrize("case", CASES)
def test_flashnext_keeps_its_bits(case):
    """(16, 48) and a tensor-parallel rank's (8, 24), sigmoid gate, group sums: the bits from before."""

    cap = torch.cuda.get_device_capability()
    want = FLASHNEXT.get(cap, {}).get(case)
    if want is None:
        pytest.skip(f"no Flash Next digests recorded for sm_{cap[0]}{cap[1]}: record them from v0.3.5 (FLASHNEXT)")
    assert _flashnext_digest(*case) == want, case
    assert _flashnext_digest(*case, gate="sigmoid") == want, case


@pytest.mark.parametrize("rows", [1, 2, 3, 16, 17, 64, 128])
def test_window_rows_and_replayed_prefixes_match_serial_steps(rows):
    p, cs, cw, state, a_log, dt, nw = _inputs(NK, NV, rows, 300 + rows)
    conv = gdn.widths(NK, NV)[0]
    assert conv == 8192
    win = gdn.GDNScratch(128, DEV, NK, NV)
    out, _ = _outs(128, NV)
    out_state = torch.empty_like(state)
    gdn.chain(p, cs, cw, state, a_log, dt, nw, EPS, rows, win, out_state, out[:rows], None, gate="silu")
    one = gdn.GDNScratch(1, DEV, NK, NV)
    one_out, _ = _outs(1, NV)
    st, cv, rep = state, cs, torch.empty_like(state)
    for r in range(rows):
        nxt = torch.empty_like(st)
        gdn.chain(p[r:r + 1].contiguous(), cv, cw, st, a_log, dt, nw, EPS, 1, one, nxt, one_out, None, gate="silu")
        assert torch.equal(one_out[0], out[r]), (rows, r)
        assert torch.equal(one.k[0], win.k[r]) and torch.equal(one.v[0], win.v[r]), (rows, r)
        assert torch.equal(one.g[0], win.g[r]) and torch.equal(one.b[0], win.b[r]), (rows, r)
        cv = torch.cat([cv, p[r:r + 1, :conv]])[1:].contiguous()
        st = nxt
        gdn.replay(state, win, r + 1, rep)
        assert torch.equal(rep, st), (rows, r + 1)
    assert torch.equal(out_state, st)


def test_a_kept_prefix_then_the_rest_gives_the_whole_windows_bits():
    """Keep 5 rows of 17 (replay), then chain the other 12 from there: the rows and the state of one window."""

    rows, keep = 17, 5
    p, cs, cw, state, a_log, dt, nw = _inputs(NK, NV, rows, 41)
    conv = gdn.widths(NK, NV)[0]
    win = gdn.GDNScratch(rows, DEV, NK, NV)
    out, _ = _outs(rows, NV)
    out_state = torch.empty_like(state)
    gdn.chain(p, cs, cw, state, a_log, dt, nw, EPS, rows, win, out_state, out, None, gate="silu")
    kept = torch.empty_like(state)
    gdn.replay(state, win, keep, kept)
    cv = torch.cat([cs, p[:keep, :conv]])[keep:].contiguous()
    rest = gdn.GDNScratch(rows - keep, DEV, NK, NV)
    rest_out, _ = _outs(rows - keep, NV)
    rest_state = torch.empty_like(state)
    gdn.chain(p[keep:].contiguous(), cv, cw, kept, a_log, dt, nw, EPS, rows - keep, rest, rest_state, rest_out, None,
              gate="silu")
    assert torch.equal(rest_out, out[keep:])
    assert torch.equal(rest_state, out_state)


def test_group_sums_are_optional_and_never_change_a_row():
    rows = 9
    p, cs, cw, state, a_log, dt, nw = _inputs(NK, NV, rows, 43)
    runs = []
    for sums in (True, False):
        sc = gdn.GDNScratch(rows, DEV, NK, NV)
        out, xs = _outs(rows, NV, sums)
        st = torch.empty_like(state)
        gdn.chain(p, cs, cw, state, a_log, dt, nw, EPS, rows, sc, st, out, xs, gate="silu")
        runs.append((sc, out, xs, st))
    (with_sc, with_out, xs, st1), (without, without_out, none, st2) = runs
    assert none is None and xs.shape == (rows, NV * DV // 32)
    assert torch.equal(with_out, without_out) and torch.equal(st1, st2)
    for a, b in ((with_sc.k, without.k), (with_sc.v, without.v), (with_sc.g, without.g), (with_sc.b, without.b)):
        assert torch.equal(a, b)
    sums = with_out.float().reshape(rows, -1, 32).sum(-1)
    assert torch.allclose(xs, sums, rtol=1e-5, atol=1e-4)


def _fp32_reference(p, cs, cw, state, a_log, dt, nw, rows: int, nk: int, nv: int, gate: str):
    """The recurrence in fp32 without intermediate rounding, gated by silu(z) or sigmoid(z)."""

    F = torch.nn.functional
    conv = gdn.widths(nk, nv)[0]
    x = torch.cat([cs, p[:, :conv]]).float()
    c = F.silu(sum(x[j:j + rows] * cw.float()[:, j] for j in range(4)))
    q = c[:, :nk * DK].reshape(rows, nk, DK)
    k = c[:, nk * DK:2 * nk * DK].reshape(rows, nk, DK)
    v = c[:, 2 * nk * DK:].reshape(rows, nv, DV)
    q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6) * DK ** -0.5
    k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
    z = p[:, conv:conv + nv * DV].float().reshape(rows, nv, DV)
    b = p[:, conv + nv * DV:conv + nv * DV + nv].float()
    a = p[:, conv + nv * DV + nv:].float()
    g = -torch.exp(a_log) * F.softplus(a + dt)
    beta = torch.sigmoid(b)
    S = state.clone()
    outs = []
    for t in range(rows):
        qt, kt = q[t].repeat_interleave(nv // nk, 0), k[t].repeat_interleave(nv // nk, 0)
        S = S * torch.exp(g[t])[:, None, None]
        kv = (S * kt[:, None, :]).sum(-1)
        S = S + kt[:, None, :] * ((v[t] - kv) * beta[t][:, None])[:, :, None]
        y = (S * qt[:, None, :]).sum(-1)
        yn = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + EPS) * nw.float()
        gz = F.silu(z[t]) if gate == "silu" else torch.sigmoid(z[t])
        outs.append((yn * gz).reshape(-1))
    return torch.stack(outs), S


def test_matches_an_fp32_recurrence():
    rows = 64
    p, cs, cw, state, a_log, dt, nw = _inputs(NK, NV, rows, 47)
    win = gdn.GDNScratch(rows, DEV, NK, NV)
    out, _ = _outs(rows, NV)
    out_state = torch.empty_like(state)
    gdn.chain(p, cs, cw, state, a_log, dt, nw, EPS, rows, win, out_state, out, None, gate="silu")
    ref, S = _fp32_reference(p, cs, cw, state, a_log, dt, nw, rows, NK, NV, "silu")
    err = (out.float() - ref).abs().max().item()
    top = ref.abs().max().item()
    serr = (out_state - S).abs().max().item()
    stop = S.abs().max().item()
    assert err < 0.03 * top, (err, top)
    assert serr < max(1e-2, 0.01 * stop), (serr, stop)
    # the gate is SiLU, not Flash Next's sigmoid
    sig, _ = _fp32_reference(p, cs, cw, state, a_log, dt, nw, rows, NK, NV, "sigmoid")
    assert (out.float() - sig).abs().max().item() > 10 * err


def test_unknown_gates_and_head_counts_are_refused():
    rows = 2
    p, cs, cw, state, a_log, dt, nw = _inputs(NK, NV, rows, 53)
    sc = gdn.GDNScratch(rows, DEV, NK, NV)
    out, _ = _outs(rows, NV)
    with pytest.raises(ValueError):
        gdn.chain(p, cs, cw, state, a_log, dt, nw, EPS, rows, sc, torch.empty_like(state), out, None, gate="relu")
    nv = 40
    p, cs, cw, state, a_log, dt, nw = _inputs(NK, nv, rows, 59)
    sc = gdn.GDNScratch(rows, DEV, NK, nv)
    out, _ = _outs(rows, nv)
    with pytest.raises(RuntimeError):
        gdn.chain(p, cs, cw, state, a_log, dt, nw, EPS, rows, sc, torch.empty_like(state), out, None, gate="silu")
