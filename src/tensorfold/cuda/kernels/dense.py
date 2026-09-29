"""Prompt matmul on unquantized bf16 weights: one fp32 chain over K a row, so a row's bits never depend on the others."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_dense_prefill_v1", sources=[str(here / "dense_prefill.cpp"),
                                                             str(here / "dense_prefill.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def tile_for(m: int) -> int:
    """The block shape for ``m`` rows: small blocks spread a few rows' weight reads over every SM; shapes never change bits."""

    return 3 if m <= 16 else 4 if m <= 64 else 2 if m <= 128 else 0


def prefill_matmul(x: torch.Tensor, w: torch.Tensor, *, f32: bool = False, tile: int | None = None,
                   out: torch.Tensor | None = None) -> torch.Tensor:
    """x (M, K) bf16 times w (N, K) bf16 transposed: (M, N) bf16, or unrounded fp32 with ``f32``."""

    if x.dtype != torch.bfloat16 or x.dim() != 2 or w.dim() != 2 or x.shape[1] != w.shape[1]:
        raise ValueError(f"prefill_matmul: x must be (M, {w.shape[1]}) bf16")
    if x.stride(1) != 1 or (x.shape[0] > 1 and x.stride(0) % 8) or x.data_ptr() % 16:
        x = x.clone(memory_format=torch.contiguous_format)     # cp.async reads rows in 16-byte pieces
    if out is None:
        out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    _ext().dense_prefill(x, w, out, f32, tile_for(x.shape[0]) if tile is None else int(tile))
    return out
