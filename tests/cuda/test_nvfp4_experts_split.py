"""NVFP4 grouped experts with K in slices (tensorfold.cuda.nvfp4.experts_split): the fp32 reference's values, and each
(row, slot) pair's bits alone and among up to 1,024 rows (plans past 1,024 pairs), in the decode and the prompt form."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import moe as moe_mod  # noqa: E402
from tensorfold.cuda.nvfp4 import experts_split as fx  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import nvfp4  # noqa: E402

E, TOP = 33, 4                     # 32 routed experts and the shared one


def _stack(e, n, k, g):
    words = torch.randint(0, 256, (e, n, k // 2), generator=g, device="cuda", dtype=torch.uint8)
    scales = (torch.rand(e, n, k // 16, generator=g, device="cuda") * 0.9 + 0.005).to(torch.float8_e4m3fn)
    scales.view(torch.uint8)[..., 0] = torch.randint(1, 8, (e, n), generator=g, device="cuda", dtype=torch.uint8)
    s2 = torch.rand(e, generator=g, device="cuda") * 1e-2 + 1e-3              # the first block subnormal (e = 0)
    return words, scales.view(torch.uint8), s2


def _experts(ni, d, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    stacks = _stack(E, ni, d, g), _stack(E, ni, d, g), _stack(E, d, ni, g)
    return stacks, fx.make(*stacks)


class _Cfg:
    def __init__(self, ni, d):
        self.num_experts_per_tok, self.num_experts, self.moe_intermediate_size, self.hidden_size = TOP, E - 1, ni, d


def _step(ex, router, x, prefill):
    rows = x.shape[0]
    buf = moe_mod.MoEBuffers(1 << max(4, (rows - 1).bit_length()), _Cfg(ex.width, ex.dims), "cuda", prefill=prefill)
    moe_mod.moe(x, router, ex, buf, TOP, E - 1)
    return buf


def test_the_table_keeps_the_checkpoints_bytes():
    (gate, up, down), ex = _experts(64, 256)
    for got, (w, s, _) in ((ex.up[:, :, :, 0], gate), (ex.up[:, :, :, 1], up), (ex.down[:, :, :, 0], down)):
        words, scales = fx.unpack(got)
        assert torch.equal(words, w) and torch.equal(scales, s)
    assert torch.equal(ex.s_up, torch.stack([gate[2], up[2]], 1) * 0.5) and ex.count == E
    assert ex.nbytes() == sum(t.numel() for stack in (gate, up, down) for t in stack[:2]) + E * 3 * 4
    assert [fx.split(k) for k in (64, 256, 512, 2048, 4096)] == [1, 1, 2, 4, 4]


@pytest.mark.parametrize("ni, d", [(64, 256), (512, 2048)])
@pytest.mark.parametrize("prefill", [False, True])
def test_pairs_get_the_references_values(ni, d, prefill):
    """Each pair's SwiGLU of gate and up and its down projection against the fp32 product of the dequantized weights
    (bf16 rounding of the stored values the only difference)."""

    stacks, ex = _experts(ni, d)
    g = torch.Generator(device="cuda").manual_seed(1)
    router = (torch.randn((E, d), generator=g, device="cuda") * 0.05).bfloat16()
    x = torch.randn((7, d), generator=g, device="cuda").bfloat16()
    buf = _step(ex, router, x, prefill)

    def weight(stack, e):
        return nvfp4.dequantize(stack[0][e], stack[1][e].view(torch.float8_e4m3fn), float(stack[2][e]))

    for r in range(7):
        for slot in range(TOP + 1):
            e = int(buf.pick[r, slot])
            gv = (x[r].float() @ weight(stacks[0], e).T).bfloat16().float()
            uv = (x[r].float() @ weight(stacks[1], e).T).bfloat16().float()
            act = ((gv / (1 + torch.exp(-gv))).bfloat16().float() * uv).bfloat16().float()
            got = buf.act[r, slot].float()
            assert ((got - act).abs().max() <= 0.02 * act.abs().max()).item(), (r, slot)
            y = got @ weight(stacks[2], e).T
            assert ((buf.y[r, slot].float() - y).abs().max() <= (0.01 if prefill else 1e-4) * y.abs().max()).item()


@pytest.mark.parametrize("prefill", [False, True])
def test_a_pairs_bits_never_depend_on_the_other_rows(prefill):
    """Every row alone and among up to 1,024 rows (5,120 pairs: the plan's wide form), gate/up and down."""

    _, ex = _experts(512, 2048, seed=2)
    g = torch.Generator(device="cuda").manual_seed(3)
    router = (torch.randn((E, 2048), generator=g, device="cuda") * 0.05).bfloat16()
    x = torch.randn((1024, 2048), generator=g, device="cuda").bfloat16()
    full = _step(ex, router, x, prefill)
    act, y = full.act[:1024].clone(), full.y[:1024].clone()
    for a, b in ((0, 1), (5, 6), (1023, 1024), (3, 7), (16, 33), (100, 228), (0, 128), (200, 1000)):
        part = _step(ex, router, x[a:b], prefill)
        assert torch.equal(part.act[:b - a], act[a:b]) and torch.equal(part.y[:b - a], y[a:b]), (a, b)

