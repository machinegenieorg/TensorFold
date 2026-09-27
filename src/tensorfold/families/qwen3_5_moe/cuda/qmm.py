"""Qwen3.6-35B-A3B's 4-bit matmuls on CUDA: MLX affine weights in groups of 64, in Triton.

These are Flash Next's kernels (``qwen4_exp/cuda/qmm.py``) run at group size 64. For weight group g (64 inputs,
one scale s and one bias b per output column):

    P[m, n, g] = x[m, g-block] . q[n, g-block]     tensor cores, bf16 x integer-valued bf16 -> fp32
    y[m, n]    = sum over g, in order, of  s[n, g] * P[m, n, g] + b[n, g] * xs[m, g]

where xs[m, g] is the fp32 sum of the group's 64 bf16 inputs. The K groups are split into SK slices that are a
constant of the weight's shape (``SHAPES``, frozen: never the row count), and the slices are added in slice order.
A row's bits do not depend on the row tile, the column tile, the other rows or their order, and the launch
settings (program width, groups per step, warps, stages) change the schedule, never the sums
(``tests/cuda/test_qwen36moe_qmm.py``).

The loader hands over MLX's arrays as stored (``weights.QW.triple()``: int32 words [N, K/8], bf16 scales and
biases [N, K/64]); ``make_q4`` / ``make_experts`` regroup them once for the kernels (words to [N/64][K/64][64][8],
scales and biases group-major). The embedding stays in MLX's row layout for the gather.

Format interface: ``matmul`` dispatches on the weight's type. ``Q4`` is the only format today; an NVFP4 matrix
type would add its own branch and kernels here without touching the callers.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...qwen4_exp.cuda import qmm as _fn
from ...qwen4_exp.cuda.qmm import BN, Q4, Experts, Group, bucket, to_mlx  # noqa: F401  (re-exported)

GS = 64                   # inputs per quantization group

# The K split of every Qwen3.6 matrix, frozen, with the <= 16-row launch settings:
# (N, K): (K slices, groups per unrolled step, warps, stages, program width). The K slices are part of the
# arithmetic (a new value changes every row's bits alike, so serial and windows stay equal, but stored hashes
# move); the other four change no bits and may be retuned freely. Chosen on shape grounds (about 128 to 256
# programs a matrix at one row, no split for the wide matrices); retime on GB10 before any hashes are stored.
SHAPES: dict[tuple[int, int], tuple[int, int, int, int, int]] = {
    (12352, 2048): (1, 2, 4, 3, 64),         # Gated DeltaNet [in_proj_qkv | in_proj_z | in_proj_b | in_proj_a]
    (9216, 2048): (1, 2, 4, 3, 64),          # attention [q_proj (query | gate per head) | k_proj | v_proj]
    (2048, 4096): (4, 2, 4, 3, 64),          # out_proj, o_proj; the drafter's fc
    (248320, 2048): (1, 4, 4, 2, 64),        # lm_head
    (8192, 2048): (1, 2, 4, 3, 64),          # in_proj_qkv, q_proj, unstacked
    (4096, 2048): (2, 2, 4, 3, 64),          # in_proj_z, unstacked
    (512, 2048): (4, 2, 4, 3, 32),           # k_proj, v_proj, unstacked
    (32, 2048): (4, 2, 4, 3, 32),            # in_proj_a, in_proj_b, unstacked
}

# The rule for a shape missing from SHAPES (still a function of the shape only). Kept here, not taken from
# Flash Next's default, so a Flash Next retune cannot move these bits.
SPLIT_TARGET = 160


def split_for(n: int, k: int) -> int:
    """The K slices of an (n, k) matrix: its frozen SHAPES entry, else the rule (``qwen4_exp.qmm.split_k``)."""

    got = SHAPES.get((n, k))
    return got[0] if got else _fn.split_k(n, k, SPLIT_TARGET, GS)


def split_scratch(rows: int, shapes: list[tuple[int, int]] | None = None) -> int:
    """fp32 elements of the split-K scratch (``part``) that every listed matrix needs at ``rows`` rows."""

    need = 0
    for n, k in (shapes if shapes is not None else list(SHAPES)):
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
    """MLX's arrays (words [N, K/8] uint32/int32, scales and biases [N, K/64] bf16) regrouped for ``matmul``."""

    _check(words, scales, biases)
    return _fn.make_q4(_i32(words), scales, biases, gs=GS)


def stack_q4(parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Q4:
    """Rows of several MLX (words, scales, biases) with the same K, stacked in order, then regrouped."""

    for p in parts:
        _check(*p)
    return _fn.stack_q4([(_i32(w), s, b) for w, s, b in parts], gs=GS)


def dequantize(words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """Reference: MLX (..., N, K/8) words -> (..., N, K) fp32 values s * q + b."""

    return _fn.dequantize(_i32(words), scales, biases, gs=GS)


def dequantize_q4(q: Q4) -> torch.Tensor:
    return _fn.dequantize_q4(q)


def group_sums(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """(M, K) bf16 (rows may be strided) -> (M, K/64) fp32 sums of each 64-input group (one program per row)."""

    return _fn.group_sums(x, GS, out)


# -- the matmul (format dispatch) ------------------------------------------------------------------------------
def matmul(x: torch.Tensor, w: Q4, xs: torch.Tensor | None = None, *, out: torch.Tensor | None = None,
           f32: bool = False, part: torch.Tensor | None = None, reduce: bool = True, **launch) -> torch.Tensor:
    """x (M, K) bf16 (rows may be strided, any M) @ w.T -> (M, N) bf16, or unrounded fp32 sums with ``f32``.
    ``xs``: x's 64-group sums (computed when None). ``part``: the split-K scratch (``split_scratch``).
    ``reduce=False`` with a split K returns the unreduced fp32 slices [SK, M, N], to be added in slice order.
    ``launch``: gpi / num_warps / num_stages / block_n overrides, which never change bits."""

    if isinstance(w, Q4):
        if w.gs != GS:
            raise ValueError(f"matmul: a group-{w.gs} matrix in the group-{GS} family")
        bad = set(launch) - {"gpi", "num_warps", "num_stages", "block_n"}
        if bad:
            raise TypeError(f"matmul: unknown settings {sorted(bad)} (the K split is fixed by the shape)")
        return _fn.matmul(x, w, xs, out=out, f32=f32, sk=split_for(w.n, w.k), part=part, reduce=reduce,
                          shapes=SHAPES, **launch)
    raise TypeError(f"matmul: no kernel for {type(w).__name__} weights")


# -- embedding -------------------------------------------------------------------------------------------------
@triton.jit
def _embed(IDS, W, S, B, OUT, D: tl.constexpr):
    """Row r, group g: the 64 values of token IDS[r]'s row, bf16(fp32 q * s + b) (MLX layout, untiled)."""

    row = tl.program_id(0)
    g = tl.program_id(1)
    tok = tl.load(IDS + row).to(tl.int64)
    words = tl.load(W + tok * (D // 8) + g * 8 + tl.arange(0, 8))
    q = tl.reshape((words[:, None] >> (tl.arange(0, 8) * 4)[None, :]) & 0xF, (64,)).to(tl.float32)
    s = tl.load(S + tok * (D // 64) + g).to(tl.float32)
    b = tl.load(B + tok * (D // 64) + g).to(tl.float32)
    tl.store(OUT + row * D + g * 64 + tl.arange(0, 64), (q * s + b).to(tl.bfloat16))


def embed(ids: torch.Tensor, words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor,
          out: torch.Tensor | None = None) -> torch.Tensor:
    """ids (R,) int32/int64 -> (R, D) bf16 rows of the 4-bit embedding (MLX layout as stored, group 64)."""

    _check(words, scales, biases)
    rows = ids.shape[0]
    d = words.shape[-1] * 8
    if out is None:
        out = torch.empty((rows, d), dtype=torch.bfloat16, device=ids.device)
    elif out.shape != (rows, d) or not out.is_contiguous() or out.dtype != torch.bfloat16:
        raise ValueError(f"embed: out {tuple(out.shape)} must be a contiguous bf16 ({rows}, {d})")
    _embed[(rows, d // GS)](ids, _i32(words), scales, biases, out, D=d, num_warps=1)
    return out


# -- experts ---------------------------------------------------------------------------------------------------
def make_experts(gate: tuple, up: tuple, down: tuple, shared: tuple | None = None) -> Experts:
    """gate / up: MLX (words [E, NI, D/8], scales [E, NI, D/64], biases); down: ([E, D, NI/8], [E, D, NI/64], ...).
    The loader's table already holds the shared expert as expert E - 1 (``shared`` None); ``shared`` appends one."""

    for t in [gate, up, down] + (list(shared) if shared is not None else []):
        _check(*t)

    def i32(t):
        return (_i32(t[0]), t[1], t[2])

    one = tuple(i32(t) for t in shared) if shared is not None else None
    return _fn.make_experts(i32(gate), i32(up), i32(down), one, gs=GS)


def moe_gateup(x: torch.Tensor, xs: torch.Tensor, ex: Experts, group: Group, act: torch.Tensor,
               axs: torch.Tensor, **launch) -> None:
    """ACT[row, slot] = bf16(bf16(silu(bf16 gate)) * bf16 up) for every (row, slot) of ``group``, and AXS[row,
    slot] the fp32 sums of each 32 of those values (the down projection adds two per group of 64)."""

    if ex.group_size != GS:
        raise ValueError(f"moe_gateup: group-{ex.group_size} experts in the group-{GS} family")
    rows = x.shape[0]
    if not x.is_contiguous() or x.shape[1] != ex.dims or not xs.is_contiguous() or xs.shape != (rows, ex.dims // GS):
        raise ValueError(f"moe_gateup: x must be a contiguous ({rows}, {ex.dims}) with contiguous 64-group sums")
    _fn.moe_gateup(x, xs, ex, group, act, axs, **launch)


def moe_down(act: torch.Tensor, axs: torch.Tensor, ex: Experts, group: Group, y: torch.Tensor, **launch) -> None:
    """Y[row, slot] (fp32) = down_e @ ACT[row, slot] for every (row, slot) of ``group``."""

    if ex.group_size != GS:
        raise ValueError(f"moe_down: group-{ex.group_size} experts in the group-{GS} family")
    _fn.moe_down(act, axs, ex, group, y, **launch)
