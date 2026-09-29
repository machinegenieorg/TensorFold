"""Prefill attention in 64-key tiles by absolute position, so chunking never changes bits; not decode's arithmetic."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BM = 64
BN = 64


@triton.jit
def _tile(q, k, v, m, l, o, valid, SCALE: tl.constexpr):
    s = tl.dot(q, tl.trans(k)).to(tl.float32) * SCALE
    s = tl.where(valid, s, float("-inf"))
    tile_m = tl.max(s, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid & active[:, None], tl.exp(s - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    l = l * alpha + tl.sum(p, 1)
    return next_m, l, o


@triton.jit
def _attend(Q, K, V, OUT, p0, W, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, BM: tl.constexpr,
            BN: tl.constexpr, SCALE: tl.constexpr):
    block = tl.program_id(0)
    head = tl.program_id(1)
    hk = head // (H // HK)
    rows = block * BM + tl.arange(0, BM)
    ok = rows < W
    pos = p0 + rows
    d = tl.arange(0, D)
    q = tl.load(Q + (rows[:, None] * H + head) * D + d[None, :], mask=ok[:, None], other=0.0)
    m = tl.full((BM,), float("-inf"), tl.float32)
    l = tl.zeros((BM,), tl.float32)
    o = tl.zeros((BM, D), tl.float32)
    first_pos = p0 + block * BM
    last_pos = p0 + tl.minimum(block * BM + BM, W) - 1
    full = (first_pos + 1) // BN                  # tiles every row of the block sees whole
    for t in range(0, full):
        keys = t * BN + tl.arange(0, BN)
        k = tl.load(K + (keys[:, None] * HK + hk) * D + d[None, :])
        v = tl.load(V + (keys[:, None] * HK + hk) * D + d[None, :])
        m, l, o = _tile(q, k, v, m, l, o, keys[None, :] <= pos[:, None], SCALE)
    for t in range(full, last_pos // BN + 1):
        keys = t * BN + tl.arange(0, BN)
        seen = keys <= last_pos
        k = tl.load(K + (keys[:, None] * HK + hk) * D + d[None, :], mask=seen[:, None], other=0.0)
        v = tl.load(V + (keys[:, None] * HK + hk) * D + d[None, :], mask=seen[:, None], other=0.0)
        m, l, o = _tile(q, k, v, m, l, o, keys[None, :] <= pos[:, None], SCALE)
    out = o / l[:, None]
    tl.store(OUT + (rows[:, None] * H + head) * D + d[None, :], out.to(tl.bfloat16), mask=ok[:, None])


def attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, p0: int, *, scale: float) -> torch.Tensor:
    """q (W, H, D) bf16 at positions [p0, p0 + W); the caches must already hold every key through p0 + W - 1."""

    w, h, d = q.shape
    hk = k_cache.shape[1]
    if k_cache.shape[0] < p0 + w or v_cache.shape != k_cache.shape or h % hk or d not in (64, 128, 256):
        raise ValueError("prefill attention: caches must hold the chunk's keys; heads a multiple of kv heads")
    if not (q.is_contiguous() and k_cache.is_contiguous() and v_cache.is_contiguous()):
        raise ValueError("prefill attention takes contiguous tensors")
    out = torch.empty_like(q)
    _attend[(triton.cdiv(w, BM), h)](q, k_cache, v_cache, out, p0, w, H=h, HK=hk, D=d, BM=BM, BN=BN, SCALE=scale,
                                     num_warps=8, num_stages=1 if d > 128 else 2)
    return out


@triton.jit
def _attend_texts(Q, K, V, OUT, BLOCKS, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, BM: tl.constexpr,
                  BN: tl.constexpr, SCALE: tl.constexpr):
    """``_attend`` for one 64-row block of one text: (first row, length, block's first position) from ``BLOCKS``."""

    block = tl.program_id(0)
    head = tl.program_id(1)
    start = tl.load(BLOCKS + 3 * block).to(tl.int64)
    length = tl.load(BLOCKS + 3 * block + 1)
    first = tl.load(BLOCKS + 3 * block + 2)
    hk = head // (H // HK)
    pos = first + tl.arange(0, BM)                # positions within the text
    ok = pos < length
    d = tl.arange(0, D)
    q = tl.load(Q + ((start + pos)[:, None] * H + head) * D + d[None, :], mask=ok[:, None], other=0.0)
    m = tl.full((BM,), float("-inf"), tl.float32)
    l = tl.zeros((BM,), tl.float32)
    o = tl.zeros((BM, D), tl.float32)
    last_pos = tl.minimum(first + BM, length) - 1
    full = (first + 1) // BN                      # tiles every row of the block sees whole
    for t in range(0, full):
        keys = t * BN + tl.arange(0, BN)
        k = tl.load(K + ((start + keys)[:, None] * HK + hk) * D + d[None, :])
        v = tl.load(V + ((start + keys)[:, None] * HK + hk) * D + d[None, :])
        m, l, o = _tile(q, k, v, m, l, o, keys[None, :] <= pos[:, None], SCALE)
    for t in range(full, last_pos // BN + 1):
        keys = t * BN + tl.arange(0, BN)
        seen = keys <= last_pos
        k = tl.load(K + ((start + keys)[:, None] * HK + hk) * D + d[None, :], mask=seen[:, None], other=0.0)
        v = tl.load(V + ((start + keys)[:, None] * HK + hk) * D + d[None, :], mask=seen[:, None], other=0.0)
        m, l, o = _tile(q, k, v, m, l, o, keys[None, :] <= pos[:, None], SCALE)
    out = o / l[:, None]
    tl.store(OUT + ((start + pos)[:, None] * H + head) * D + d[None, :], out.to(tl.bfloat16), mask=ok[:, None])


def text_blocks(lengths, device) -> torch.Tensor:
    """(blocks, 3) int32: each text's first row, its length and a block's first position, in ``BM``-row blocks."""

    rows, start = [], 0
    for n in lengths:
        if n < 1:
            raise ValueError("every text needs at least one token")
        rows.extend((start, n, first) for first in range(0, n, BM))
        start += n
    return torch.tensor(rows, dtype=torch.int32).to(device, non_blocking=True)


def attention_texts(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, blocks: torch.Tensor, *,
                    scale: float) -> torch.Tensor:
    """Causal attention within each text of a packed batch: q (T, H, D), k and v (T, HK, D) bf16, texts back to back.

    A row reads only its own text's keys, in the tiles ``attention`` gives that text alone from position 0, so its
    bits never depend on the other texts, their lengths or their order.
    """

    t, h, d = q.shape
    hk = k.shape[1]
    if k.shape != (t, hk, d) or v.shape != k.shape or h % hk or d not in (64, 128, 256):
        raise ValueError("text attention: keys and values one row a query row; heads a multiple of kv heads")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and blocks.is_contiguous()):
        raise ValueError("text attention takes contiguous tensors")
    out = torch.empty_like(q)
    _attend_texts[(blocks.shape[0], h)](q, k, v, out, blocks, H=h, HK=hk, D=d, BM=BM, BN=BN, SCALE=scale,
                                        num_warps=8, num_stages=1 if d > 128 else 2)
    return out
