"""Gated DeltaNet chain on CUDA (``gdn.cu``) for Flash Next and Qwen3.6-35B-A3B: one fused kernel per layer
and window.

``chain`` runs R consecutive rows from the committed state: conv + SiLU, q/k L2 norms, gates, the
delta rule and the gated RMSNorm (``gate="sigmoid"`` for Flash Next, ``"silu"`` for Qwen3.6), writing the
rows' outputs (and their 32-group sums for an affine-quantized out projection, unless the scratch was made
without them), the state after the last row, and what a replay needs (normalized k, v, g, beta).
``replay`` rebuilds the state after the first ``keep`` rows of a window with the same update routine,
so keeping a prefix of a window gives the bits of serial steps.

(key, value) heads: Flash Next's layer (16, 48) or one of two tensor-parallel ranks (8, 24); Qwen3.6-35B-A3B's
layer (16, 32).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

NK, NV, DK, DV = 16, 48, 128, 128  # Flash Next's layer
CONV = 2 * NK * DK + NV * DV
PW = CONV + NV * DV + 2 * NV
HEADS = ((16, 48), (8, 24), (16, 32))
GATES = {"sigmoid": 0, "silu": 1}
# no FMA contraction: ``replay`` recomputes the chain's state update with the chain's bits
CUDA_FLAGS = ("-O3", "--fmad=false")


def widths(nk: int, nv: int) -> tuple[int, int]:
    """(conv channels, projection row width) for nk key heads and nv value heads."""

    conv = 2 * nk * DK + nv * DV
    return conv, conv + nv * DV + 2 * nv


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    here = Path(__file__).parent
    return load(name="tensorfold_qwen4_exp_gdn", sources=[str(here / "gdn.cpp"), str(here / "gdn.cu")],
                extra_cuda_cflags=list(CUDA_FLAGS), verbose=False)


class GDNScratch:
    """Static per-window buffers (outputs and replay inputs) for up to ``rows`` rows. With
    ``group_sums=False`` the chain skips the outputs' 32-group sums and ``xs`` is None."""

    def __init__(self, rows: int, device, nk: int = NK, nv: int = NV, group_sums: bool = True) -> None:
        self.out = torch.empty((rows, nv * DV), dtype=torch.bfloat16, device=device)
        self.xs = torch.empty((rows, nv * DV // 32), dtype=torch.float32, device=device) if group_sums else None
        self.k = torch.empty((rows, nk, DK), dtype=torch.float32, device=device)
        self.v = torch.empty((rows, nv, DV), dtype=torch.bfloat16, device=device)
        self.g = torch.empty((rows, nv), dtype=torch.float32, device=device)
        self.b = torch.empty((rows, nv), dtype=torch.float32, device=device)


def chain(p: torch.Tensor, conv_state: torch.Tensor, conv_w: torch.Tensor, state_in: torch.Tensor,
          a_log: torch.Tensor, dt_bias: torch.Tensor, norm_w: torch.Tensor, eps: float, rows: int,
          scratch: GDNScratch, state_out: torch.Tensor, gate: str = "sigmoid") -> None:
    if gate not in GATES:
        raise ValueError(f"gdn chain: gate must be one of {sorted(GATES)}, not {gate!r}")
    xs = scratch.xs if scratch.xs is not None else p.new_empty(0, dtype=torch.float32)
    _ext().chain(p, conv_state, conv_w, state_in, a_log, dt_bias, norm_w, float(eps), int(rows), scratch.out,
                 xs, state_out, scratch.k, scratch.v, scratch.g, scratch.b, GATES[gate])


def replay(state_in: torch.Tensor, scratch: GDNScratch, rows: int, state_out: torch.Tensor) -> None:
    _ext().replay(state_in, scratch.k, scratch.v, scratch.g, scratch.b, int(rows), state_out)
