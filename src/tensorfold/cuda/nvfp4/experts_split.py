"""Grouped NVFP4 experts with K split in slices: a proposed alternative to ``experts`` for decode's few rows a call.

The same checkpoint bytes and plan as ``experts``, another arithmetic, as exact and as row-invariant. A table keeps
the E2M1 nibbles in ``tensorfold.cuda.experts.pack``'s fragment order (the MLX and the NVFP4 layouts pack nibbles
alike) and each 64-input group's e4m3 block scales a word a lane. Lane t of a quad carries one NVFP4 block's inputs,
so it decodes its fragment to exact bf16 values, 2 x code x block scale (at most six significant bits), and each
(pair, column) is one fp32 mma chain over K: in slices fixed by K (``split``: 4 at 2,048 inputs, 2 at 512), added in
order in shared memory, times the expert's per-tensor scale over two. A prompt plan's call takes K in one slice,
a CTA an item (64 pairs) against 128 columns, each 64-input group of its rows and weights staged in shared memory
once for its eight warps (the MLX experts' prompt kernel on this format): chunk-invariant bits of its own.

On an RTX PRO 6000 Max-Q with Qwen3.6's experts (257 of 512 x 2,048), a layer's gate/up and down against
``experts``: one row 12 and 6 us (54 and 11), four rows 32 and 17 (55 and 17), 16 rows 93 and 47 (97 and 46),
128 rows 243 and 121 (241 and 118), a 4,096-row prompt chunk 1.13 and 0.73 ms (2.06 and 1.19; the MLX experts'
1.04 and 0.67). On a GB10 (shared with a serving engine) the chunk's 40 layers take 382 ms (the MLX experts' 391).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from tensorfold.cuda import experts as grouped

GS = 64                  # inputs a group: four NVFP4 blocks of 16
BLOCK = 288              # int32 a (32 columns x 64 inputs) block: 256 of nibbles, 32 of block scales


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_nvfp4_split_v2", sources=[str(here / "experts_split.cpp"),
                                                             str(here / "experts_split.cu")],
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
    """One layer's experts (a shared one, if any, last): gate and up (SwiGLU) and down, each with an fp32 scale."""

    up: torch.Tensor          # [E, NI/32, D/64, 2, 288] int32: gate, then up, a group at a time
    down: torch.Tensor        # [E, D/32, NI/64, 1, 288]
    s_up: torch.Tensor        # [E, 2] fp32: gate's and up's weight_scale_2, halved (the kernel's values are 2 x code)
    s_down: torch.Tensor      # [E, 1] fp32
    width: int                # NI
    dims: int                 # D

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


def _shape(m: int, pairs: int) -> tuple[int, int, int]:
    """(units a block, register stages, row tiles a warp) for a decode call of ``pairs``: none changes a bit
    (measured on an RTX PRO 6000 Max-Q with Qwen3.6's experts)."""

    if m == 2:
        return (1, 4, 1) if pairs <= 16 else (1, 2, 1) if pairs <= 160 else (2, 2, 1)
    return (2, 2, 1) if pairs <= 80 else (4, 2, 1)


def _run(m: int, epi: int, x: torch.Tensor, slots: int, w: torch.Tensor, s2: torch.Tensor, n: int, plan,
         out: torch.Tensor, rows: int) -> None:
    """A decode plan's call takes K in ``split`` slices, a warp a unit; a prompt plan's (items of 64 pairs) one slice,
    staged in shared memory a CTA an item: the prompt form, chunk-invariant bits of its own, as the MLX experts'
    prompt form has."""

    kg, nb = x.shape[1] // GS, n // 32
    pairs = rows * plan.slots
    if plan.prefill:
        _ext().prefill(m, epi, x, slots, w, s2, kg, nb, plan.items, plan.counts, plan.members, out, n,
                       grouped.max_items(pairs, plan.experts, plan.tile))
        return
    upb, d, rt = _shape(m, pairs)
    rg = plan.tile // (16 * rt)
    units = grouped.max_items(pairs, plan.experts, plan.tile) * nb * rg
    _ext().run(m, epi, split(x.shape[1]), upb, d, rt, x, slots, w, s2, kg, nb, rg,
               plan.items, plan.counts, plan.members, out, n, units)


def gate_up(x: torch.Tensor, ex: Experts, plan, out: torch.Tensor, rows: int) -> None:
    """x [R, D] bf16 -> out [R * slots, NI] bf16, each pair's bf16(bf16(silu(gate)) * up)."""

    _run(2, 2, x, plan.slots, ex.up, ex.s_up, ex.width, plan, out, rows)


def down(act: torch.Tensor, ex: Experts, plan, out: torch.Tensor, rows: int) -> None:
    """act [R * slots, NI] bf16 -> out [R * slots, D]: fp32, or bf16 when ``out`` is (a prompt's buffers)."""

    _run(1, 3 if out.dtype == torch.bfloat16 else 0, act, 0, ex.down, ex.s_down, ex.dims, plan, out, rows)

