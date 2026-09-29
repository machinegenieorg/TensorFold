"""Grouped MoE experts on NVIDIA's NVFP4 weights, the MLX experts kernel's form: a (row, slot) pair's bits never depend
on the other rows of the call.

A table keeps the checkpoint's bytes: its E2M1 nibbles in ``experts.pack``'s fragment order (the MLX and the NVFP4
layouts pack nibbles alike) and, in place of each 64-input group's scales and biases, the group's e4m3 block scales
laid out a word a lane. The kernel decodes a lane's B fragment to exact bf16 values (2 x code x block scale has at
most six significant bits), runs one fp32 mma chain over K in slices fixed by K (``split``; a prompt's in one), adds
the slices in order, and multiplies by the expert's per-tensor scale over two. The shared expert is expert
``count - 1`` of the table, so a MoE step is the router, the plan, one gate/up launch and one down launch, as on MLX
weights. ``Dense`` is one projection as a table of one expert (the vocabulary heads), called without a plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from . import experts as grouped

GS = 64                  # inputs a group: four NVFP4 blocks of 16
BLOCK = 288              # int32 a (32 columns x 64 inputs) block: 256 of nibbles, 32 of block scales


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_nvfp4_experts_v1", sources=[str(here / "nvfp4_experts.cpp"),
                                                               str(here / "nvfp4_experts.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def scale_words(scales: torch.Tensor) -> torch.Tensor:
    """[E, N, K/16] e4m3 bytes -> [E, N/32, K/64, 32] int32: lane gq * 4 + t's word holds, byte j, the scale of column
    8 j + gq (of the 32) for block t of the group (the inputs lane t's fragments carry)."""

    e, n, kb = scales.shape
    b = scales.contiguous().view(torch.uint8).reshape(e, n // 32, 4, 8, kb // 4, 4)     # (cb, j, gq), (g, t)
    return b.permute(0, 1, 4, 3, 5, 2).contiguous().view(torch.int32).reshape(e, n // 32, kb // 4, 32)


def pack(words: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """One projection's stacked arrays ([E, N, K/2] uint8 nibbles, [E, N, K/16] e4m3 block scales) -> the table
    [E, N/32, K/64, 288] int32."""

    e, n, k2 = words.shape
    if n % 32 or (2 * k2) % GS or scales.shape != (e, n, 2 * k2 // 16):
        raise ValueError(f"nvfp4 experts: words {tuple(words.shape)} and scales {tuple(scales.shape)} do not pack")
    zeros = torch.zeros((e, n, 2 * k2 // GS), dtype=torch.bfloat16, device=words.device)
    blocks = grouped.pack(words.contiguous().view(torch.int32), zeros, zeros, GS)       # nibbles in fragment order
    blocks[..., 256:] = scale_words(scales)
    return blocks


def unpack(blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``pack``'s inverse: [E, N/32, K/64, 288] -> ([E, N, K/2] uint8, [E, N, K/16] e4m3 bytes as uint8)."""

    e, nb, kg, _ = blocks.shape
    fake = blocks.clone()
    fake[..., 256:] = 0
    words, _, _ = grouped.unpack(fake, GS)
    s = blocks[..., 256:].contiguous().view(torch.uint8).reshape(e, nb, kg, 8, 4, 4)     # (g), (gq, t, j)
    scales = s.permute(0, 1, 5, 3, 2, 4).reshape(e, nb * 32, kg * 4)
    return words.contiguous().view(torch.uint8), scales.contiguous()


@dataclass
class Experts:
    """One layer's experts, the shared one last: gate and up (SwiGLU) and down, each with an fp32 per-tensor scale."""

    up: torch.Tensor          # [E, NI/32, D/64, 2, 288] int32: gate, then up, a group at a time
    down: torch.Tensor        # [E, D/32, NI/64, 1, 288]
    s_up: torch.Tensor        # [E, 2] fp32: gate's and up's weight_scale_2, halved (the kernel's values are 2 x code)
    s_down: torch.Tensor      # [E, 1] fp32
    width: int                # NI
    dims: int                 # D
    kernel = "nvfp4-grouped"

    @property
    def count(self) -> int:
        return self.up.shape[0]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.up, self.down, self.s_up, self.s_down))


def make(gate: tuple, up: tuple, down: tuple) -> Experts:
    """Stacked checkpoint arrays, each ([E, N, K/2] uint8, [E, N, K/16] e4m3, [E] fp32 weight_scale_2)."""

    (gw, gs, g2), (uw, us, u2), (dw, ds, d2) = gate, up, down
    u = torch.stack([pack(gw, gs), pack(uw, us)], dim=3)
    dn = pack(dw, ds).unsqueeze(3)
    halve = lambda s: (torch.as_tensor(s, dtype=torch.float32, device=gw.device).reshape(-1) * 0.5)   # noqa: E731
    return Experts(u, dn, torch.stack([halve(g2), halve(u2)], 1).contiguous(), halve(d2)[:, None].contiguous(),
                   int(gw.shape[1]), int(dw.shape[1]))


def split(k: int) -> int:
    """K slices of a projection with K inputs: a function of K alone (never the rows), so every call has its bits."""

    groups = k // GS
    sk = 4 if k >= 2048 else 2 if k >= 512 else 1
    while groups % sk:
        sk //= 2
    return sk


def _shape(m: int, pairs: int, prompt: bool) -> tuple[int, int, int]:
    """(units a block, register stages, row tiles a warp) for a call of ``pairs``: none changes a bit (measured on an
    RTX PRO 6000 Max-Q with Qwen3.6's experts)."""

    if prompt:
        return PROMPT
    if m == 2:
        return (1, 4, 1) if pairs <= 16 else (1, 2, 1) if pairs <= 160 else (2, 2, 1)
    return (2, 2, 1) if pairs <= 80 else (4, 2, 1)


PROMPT = (2, 1, 4)        # a prompt chunk's 4,096 rows: gate/up 1.07 ms and down 0.69 ms a layer, the MLX prompt form's


def _run(m: int, epi: int, x: torch.Tensor, slots: int, w: torch.Tensor, s2: torch.Tensor, n: int, plan,
         out: torch.Tensor, rows: int) -> None:
    """A decode plan's call takes K in ``split`` slices; a prompt plan's (items of 64 pairs) one slice: the prompt
    form, chunk-invariant bits of its own, as the MLX experts' prompt form has."""

    kg, nb = x.shape[1] // GS, n // 32
    pairs = rows * plan.slots
    upb, d, rt = _shape(m, pairs, plan.prefill)
    rg = plan.tile // (16 * rt)
    units = grouped.max_items(pairs, plan.experts, plan.tile) * nb * rg
    _ext().run(m, epi, 1 if plan.prefill else split(x.shape[1]), upb, d, rt, x, slots, w, s2, kg, nb, rg,
               plan.items, plan.counts, plan.members, out, n, units, 0)


def gate_up(x: torch.Tensor, ex: Experts, plan, out: torch.Tensor, rows: int) -> None:
    """x [R, D] bf16 -> out [R * slots, NI] bf16, each pair's bf16(bf16(silu(gate)) * up)."""

    _run(2, 2, x, plan.slots, ex.up, ex.s_up, ex.width, plan, out, rows)


def down(act: torch.Tensor, ex: Experts, plan, out: torch.Tensor, rows: int) -> None:
    """act [R * slots, NI] bf16 -> out [R * slots, D]: fp32, or bf16 when ``out`` is (a prompt's buffers)."""

    _run(1, 3 if out.dtype == torch.bfloat16 else 0, act, 0, ex.down, ex.s_down, ex.dims, plan, out, rows)


def moe(x: torch.Tensor, router_rows: torch.Tensor, ex: Experts, buf, top_k: int, experts: int) -> None:
    """``tensorfold.cuda.moe.moe`` on an NVFP4 table: route x [R, D], then its experts into buf.y [R, k + 1, D]."""

    from . import moe as moe_mod

    rows = x.shape[0]
    moe_mod.router(x, router_rows, buf.logits[:rows])
    moe_mod.select(buf.logits[:rows], buf, top_k, experts)
    gate_up(x, ex, buf.plan, buf.act.view(-1, ex.width), rows)
    down(buf.act.view(-1, ex.width), ex, buf.plan, buf.y.view(-1, ex.dims), rows)


@dataclass
class Dense:
    """One NVFP4 projection [n, k] as a table of one expert (the vocabulary heads): every row one call of the kernel's
    dense form, with no plan."""

    w: torch.Tensor           # [1, N/32, K/64, 1, 288] int32
    s2: torch.Tensor          # [1, 1] fp32: weight_scale_2, halved
    n: int                    # rows of the table (a multiple of 32)
    k: int

    def nbytes(self) -> int:
        return self.w.numel() * 4 + 4


def make_dense(words: torch.Tensor, scales: torch.Tensor, scale2) -> Dense:
    """[n, k/2] uint8 nibbles, [n, k/16] e4m3 block scales and the fp32 per-tensor scale; zero rows pad n to 32."""

    n, k2 = words.shape
    pad = -n % 32
    scales = scales.contiguous().view(torch.uint8)
    if pad:
        words = torch.cat([words, words.new_zeros((pad, k2))])
        scales = torch.cat([scales, scales.new_zeros((pad, scales.shape[1]))])
    s2 = torch.as_tensor(scale2, dtype=torch.float32).reshape(1, 1).to(words.device) * 0.5
    return Dense(pack(words[None], scales[None]).unsqueeze(3), s2, n + pad, 2 * k2)


DENSE_SLICES = 1          # a head's K in one slice: 248,320 rows by 2,048 in 210 us for one row, 1.1 ms for 128
_plan: dict = {}


def dense(x: torch.Tensor, d: Dense, *, sk: int = DENSE_SLICES, shape: tuple[int, int] | None = None) -> torch.Tensor:
    """x [R, K] bf16 @ the table's transpose -> [R, n] bf16: each row the bits it gets alone (K slices fixed, the row
    tiles a warp takes changing no bit)."""

    rows = x.shape[0]
    rt = 1 if rows <= 16 else 4
    upb, stages = shape or ((2, 2) if rows <= 16 else (1, 1))
    empty = _plan.get(x.device)
    if empty is None:
        empty = _plan[x.device] = torch.zeros(3, dtype=torch.int32, device=x.device)
    out = torch.empty((rows, d.n), dtype=torch.bfloat16, device=x.device)
    nb = d.n // 32
    _ext().run(1, 3, sk, upb, stages, rt, x, 0, d.w, d.s2, d.k // GS, nb, 1, empty, empty, empty, out, d.n,
               nb * -(-rows // (16 * rt)), rows)
    return out
