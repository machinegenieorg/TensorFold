"""Qwen3.6-35B-A3B's 4-bit matmuls on CUDA: MLX affine weights in groups of 64 on the shared lane kernels.

Dense projections run ``tensorfold.cuda.kernels.qmm`` (the lane matmul Flash Next, the 27B and Nemotron use) at
group size 64. For weight group g (64 inputs, one scale s and one bias b per output column):

    P[m, n, g] = x[m, g-block] . q[n, g-block]     tensor cores, bf16 x integer-valued bf16 -> fp32
    y[m, n]    = sum over g, in order, of  s[n, g] * P[m, n, g] + b[n, g] * xs[m, g]

where xs[m, g] is the fp32 sum of the group's 64 bf16 inputs. The K groups are split into SK slices that are a
constant of the weight's shape (``SPLITS``, pinned here so a retune of the shared rule cannot move these bits), and
the slices are added in slice order. A row's bits do not depend on the row tile, the other rows or their order
(``tests/cuda/test_qwen36moe_qmm.py``).

Experts (``make_experts``) are packed for ``tensorfold.cuda.experts``, the grouped kernels Flash Next, GLM and
Nemotron share; ``moe.py`` runs them.

The loader hands over MLX's arrays as stored (``weights.QW.triple()``: int32 words [N, K/8], bf16 scales and biases
[N, K/64]); ``make_q4`` / ``make_experts`` regroup them once for the kernels. The embedding stays in MLX's row
layout for the gather.

Format interface: ``matmul`` dispatches on the weight's type. ``Q4`` is the only format today; an NVFP4 matrix type
would add its own branch and kernels here without touching the callers.
"""

from __future__ import annotations

import torch
import triton

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.kernels import qmm as shared
from tensorfold.cuda.kernels.qmm import Q4, bucket  # noqa: F401  (re-exported)
from tensorfold.cuda.experts import Experts  # noqa: F401  (re-exported)

GS = 64                   # inputs per quantization group

# The K split of every Qwen3.6 matrix, pinned: ``shared.split_k``'s choice at group 64 for these shapes (about 192
# programs a matrix at one row, no split for the wide ones). The K slices are part of the arithmetic (a new value
# changes every row's bits alike, so serial and windows stay equal, but stored hashes move); retime on GB10 before
# any hashes are stored.
SPLITS: dict[tuple[int, int], int] = {
    (12352, 2048): 1,        # Gated DeltaNet [in_proj_qkv | in_proj_z | in_proj_b | in_proj_a]
    (9216, 2048): 2,         # attention [q_proj (query | gate per head) | k_proj | v_proj]
    (2048, 4096): 8,         # out_proj, o_proj; the drafter's fc
    (248320, 2048): 1,       # lm_head
    (8192, 2048): 2,         # in_proj_qkv, q_proj, unstacked
    (4096, 2048): 4,         # in_proj_z, unstacked
    (512, 2048): 4,          # k_proj, v_proj, unstacked
    (32, 2048): 4,           # in_proj_a, in_proj_b, unstacked
}


def split_for(n: int, k: int) -> int:
    """The K slices of an (n, k) matrix: its pinned ``SPLITS`` entry, else the shared rule (shape only)."""

    got = SPLITS.get((n, k))
    return got if got else shared.split_k(n, k, GS)


def split_scratch(rows: int, shapes: list[tuple[int, int]] | None = None) -> int:
    """fp32 elements of the split-K scratch (``part``) that every listed matrix needs at ``rows`` rows."""

    need = 0
    for n, k in (shapes if shapes is not None else list(SPLITS)):
        sk = split_for(n, k)
        if sk > 1:
            need = max(need, sk * rows * n)
    return need


# -- matrices ------------------------------------------------------------------------------------------------
def _i32(w: torch.Tensor) -> torch.Tensor:
    return w.view(torch.int32) if w.dtype != torch.int32 else w


def _check(words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> None:
    *lead, n, k8 = words.shape
    want = (*lead, n, k8 * 8 // GS)
    if tuple(scales.shape) != want or tuple(biases.shape) != want:
        raise ValueError(f"4-bit group-{GS} weights: words {tuple(words.shape)} want scales and biases {want}, got "
                         f"{tuple(scales.shape)} / {tuple(biases.shape)}")
    if words.dtype not in (torch.int32, torch.uint32) or scales.dtype != torch.bfloat16 \
            or biases.dtype != torch.bfloat16:
        raise ValueError(f"4-bit weights: words int32/uint32, scales and biases bf16; got {words.dtype}, "
                         f"{scales.dtype}, {biases.dtype}")


def make_q4(words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> Q4:
    """MLX's arrays (words [N, K/8] uint32/int32, scales and biases [N, K/64] bf16) packed for ``matmul``."""

    _check(words, scales, biases)
    return shared.pack(_i32(words), scales, biases, GS)


def stack_q4(parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Q4:
    """Rows of several MLX (words, scales, biases) with the same K, stacked in order, then packed."""

    for p in parts:
        _check(*p)
    return make_q4(torch.cat([_i32(p[0]) for p in parts]), torch.cat([p[1] for p in parts]),
                   torch.cat([p[2] for p in parts]))


def to_mlx(q: Q4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The stored MLX layout again: (N, K/8) int32 words, (N, K/64) scales and biases."""

    return shared.unpack(q)


def dequantize(words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """Reference: MLX (..., N, K/8) words -> (..., N, K) fp32 values s * q + b."""

    _check(words, scales, biases)
    k8 = words.shape[-1]
    w = _i32(words).to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, device=words.device, dtype=torch.int64) * 4
    q = ((w[..., None] >> shifts) & 0xF).reshape(*words.shape[:-1], k8 * 8).to(torch.float32)
    return q * scales.float().repeat_interleave(GS, dim=-1) + biases.float().repeat_interleave(GS, dim=-1)


def dequantize_q4(q: Q4) -> torch.Tensor:
    return dequantize(*to_mlx(q))


def group_sums(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """(M, K) bf16 (rows may be strided) -> (M, K/64) fp32 sums of each 64-input group (one program per row)."""

    m, k = x.shape
    kg = k // GS
    if out is None:
        out = torch.empty((m, kg), dtype=torch.float32, device=x.device)
    elif out.shape != (m, kg) or not out.is_contiguous() or out.dtype != torch.float32:
        raise ValueError(f"group_sums: out {tuple(out.shape)} must be a contiguous fp32 ({m}, {kg})")
    shared._group_sums[(m, triton.cdiv(kg, 16))](x, out, x.stride(0), KG=kg, GS=GS, GB=16, num_warps=2)
    return out


# -- the matmul (format dispatch) ------------------------------------------------------------------------------
def matmul(x: torch.Tensor, w: Q4, xs: torch.Tensor | None = None, *, out: torch.Tensor | None = None,
           f32: bool = False, part: torch.Tensor | None = None, reduce: bool = True) -> torch.Tensor:
    """x (M, K) bf16 (rows may be strided, any M) @ w.T -> (M, N) bf16, or unrounded fp32 sums with ``f32``.
    ``xs``: x's 64-group sums (computed when None). ``part``: the split-K scratch (``split_scratch``).
    ``reduce=False`` with a split K returns the unreduced fp32 slices [SK, M, N], to be added in slice order."""

    if isinstance(w, Q4):
        if w.gs != GS:
            raise ValueError(f"matmul: a group-{w.gs} matrix in the group-{GS} family")
        if xs is None:
            xs = group_sums(x)
        return shared.matmul(x, w, xs, sk=split_for(w.n, w.k), f32=f32, out=out, part=part, reduce=reduce)
    raise TypeError(f"matmul: no kernel for {type(w).__name__} weights")


# -- embedding -------------------------------------------------------------------------------------------------
def embed(ids: torch.Tensor, words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor,
          out: torch.Tensor | None = None) -> torch.Tensor:
    """ids (R,) int32/int64 -> (R, D) bf16 rows of the 4-bit embedding (MLX layout as stored, group 64):
    bf16(fp32 q * s + b), the dense Qwen family's gather kernel into a caller's buffer."""

    from tensorfold.families.qwen3_5.cuda import glue as dense

    _check(words, scales, biases)
    rows = ids.shape[0]
    d = words.shape[-1] * 8
    if out is None:
        out = torch.empty((rows, d), dtype=torch.bfloat16, device=ids.device)
    elif out.shape != (rows, d) or not out.is_contiguous() or out.dtype != torch.bfloat16:
        raise ValueError(f"embed: out {tuple(out.shape)} must be a contiguous bf16 ({rows}, {d})")
    dense._embed[(rows, d // GS)](ids, _i32(words), scales, biases, out, D=d, num_warps=1)
    return out


# -- experts ---------------------------------------------------------------------------------------------------
def make_experts(gate: tuple, up: tuple, down: tuple, shared_expert: tuple | None = None) -> Experts:
    """gate / up: MLX (words [E, NI, D/8], scales [E, NI, D/64], biases); down: ([E, D, NI/8], [E, D, NI/64], ...),
    packed for ``tensorfold.cuda.experts`` (SwiGLU). The loader's table already holds the shared expert as expert
    E - 1 (``shared_expert`` None); ``shared_expert`` (gate, up, down) appends one."""

    parts = [gate, up, down]
    for t in parts + (list(shared_expert) if shared_expert is not None else []):
        _check(*t)
    if shared_expert is not None:
        parts = [tuple(torch.cat([a, b[None]]) for a, b in zip(t, s)) for t, s in zip(parts, shared_expert)]
    gate, up, down = ((_i32(t[0]), t[1], t[2]) for t in parts)
    return grouped.make([gate, up], down, GS)
