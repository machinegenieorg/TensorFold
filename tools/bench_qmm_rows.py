#!/usr/bin/env python3
"""SPIKE: lane matmul time by row count and launch config over the Qwen3.8-27B dense projection shapes.

Every (bm, gpi, warps, stages) choice gives the same bits (the per-row math is fixed); this finds the fastest
at each row count. Prints ms per full set of 64 layers' projections and the achieved TFLOP/s."""

from __future__ import annotations

import itertools
import json

import torch

from tensorfold.families.qwen3_5.cuda.qmm import bucket
from tensorfold.families.qwen3_5.cuda.qmm_fast import lane_matmul_tiled, tile
from tensorfold.families.qwen3_5.cuda.weights import QLinear

# (n, k, count per forward): 48 GDN layers + 16 attention layers + 64 MLPs
SHAPES = [(10240, 5120, 48), (6240, 5120, 48), (5120, 6144, 48),       # GDN qkv, zba, out
          (12288, 5120, 16), (2048, 5120, 16), (5120, 6144, 16),       # attention q|gate, kv, o
          (17408, 5120, 128), (5120, 17408, 64)]                       # MLP gate + up, down


def weight(n, k):
    return tile(QLinear(torch.randint(-2**31, 2**31 - 1, (n, k // 8), dtype=torch.int32, device="cuda"),
                        (torch.rand((n, k // 64), device="cuda") * 0.01).to(torch.bfloat16),
                        (torch.randn((n, k // 64), device="cuda") * 0.01).to(torch.bfloat16)))


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def main():
    ws = {(n, k): weight(n, k) for n, k, _ in SHAPES}
    out = []
    for m in (64, 96, 128):
        best = None
        cands = [(bm, gpi, nw, ns) for bm in sorted({bucket(m), 64})
                 for gpi, nw, ns in itertools.product((1, 2), (4, 8), (2, 3, 4))]
        default = None
        for bm, gpi, nw, ns in cands:
            total = 0.0
            try:
                for n, k, cnt in SHAPES:
                    q = ws[(n, k)]
                    x = torch.randn((m, k), device="cuda").to(torch.bfloat16)
                    total += cnt * timed(lambda: lane_matmul_tiled(x, q.weight, q.scales, q.biases, q.n, None,
                                                                   gpi=gpi, num_warps=nw, num_stages=ns, bm=bm), reps=5)
            except Exception as exc:          # a config Triton cannot build (e.g. shared memory)
                continue
            flops = sum(2 * m * n * k * cnt for n, k, cnt in SHAPES)
            row = dict(m=m, bm=bm, gpi=gpi, warps=nw, stages=ns, ms=round(total, 2), tflops=round(flops / total / 1e9, 1))
            if best is None or total < best["ms"]:
                best = row
            if (bm, gpi, nw, ns) == (bucket(m), *{16: (4, 4, 2), 32: (2, 4, 2), 64: (1, 4, 2), 128: (1, 4, 3)}[bucket(m)]):
                default = row
        out.append({"m": m, "default": default, "best": best})
        print(json.dumps(out[-1]), flush=True)


if __name__ == "__main__":
    main()
