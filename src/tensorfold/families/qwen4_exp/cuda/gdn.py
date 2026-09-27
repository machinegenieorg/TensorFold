"""Run a row-exact DeltaNet chain and retain normalized k, v, g and beta so replaying any accepted prefix reconstructs serial state with the same update routine.

(key, value) heads: Flash Next's layer (16, 48) or a tensor-parallel rank's (8, 24), gated by sigmoid(z);
Qwen3.6-35B-A3B's layer (16, 32), gated by silu(z) and without the 32-group sums (its out projection groups 64
inputs).
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
_NONE: dict = {}                   # a zero-size fp32 tensor a device: the kernel's "no group sums"


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
    """A sequence's replay inputs for its last window of up to ``rows`` rows."""

    def __init__(self, rows: int, device, nk: int = NK, nv: int = NV) -> None:
        self.k = torch.empty((rows, nk, DK), dtype=torch.float32, device=device)
        self.v = torch.empty((rows, nv, DV), dtype=torch.bfloat16, device=device)
        self.g = torch.empty((rows, nv), dtype=torch.float32, device=device)
        self.b = torch.empty((rows, nv), dtype=torch.float32, device=device)


def chain(p: torch.Tensor, conv_state: torch.Tensor, conv_w: torch.Tensor, state_in: torch.Tensor,
          a_log: torch.Tensor, dt_bias: torch.Tensor, norm_w: torch.Tensor, eps: float, rows: int,
          scratch: GDNScratch, state_out: torch.Tensor, out: torch.Tensor, xs: torch.Tensor | None, *,
          gate: str = "sigmoid") -> None:
    """``out`` [rows, NV*DV] bf16 and ``xs`` (its 32-group sums, or None to skip them) may be row views of a larger
    buffer. ``gate``: "sigmoid" (Flash Next) or "silu" (Qwen3.6) on z in the gated RMSNorm."""

    if gate not in GATES:
        raise ValueError(f"gdn chain: gate must be one of {sorted(GATES)}, not {gate!r}")
    if xs is None:
        xs = _NONE.get(p.device)
        if xs is None:
            xs = _NONE[p.device] = torch.empty((0,), dtype=torch.float32, device=p.device)
    _ext().chain(p, conv_state, conv_w, state_in, a_log, dt_bias, norm_w, float(eps), int(rows), out, xs,
                 state_out, scratch.k, scratch.v, scratch.g, scratch.b, GATES[gate])


def replay(state_in: torch.Tensor, scratch: GDNScratch, rows: int, state_out: torch.Tensor) -> None:
    _ext().replay(state_in, scratch.k, scratch.v, scratch.g, scratch.b, int(rows), state_out)
