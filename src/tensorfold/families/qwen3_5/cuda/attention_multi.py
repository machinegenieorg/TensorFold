"""SPIKE: ``attention.attention`` for several requests in one launch per kernel.

Each request keeps its own committed cache (pointer table), cache length P, window rows and parents. Its chunks,
tiles and merge order are exactly the single-request call's: request i sees chunks 0..full_i-1 over its cache and
tail chunks full_i..nch_i-1 over [cache | path], merged in chunk order. Programs past a request's own extent exit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .attention import CHUNK, MAX_NODES, QUERY_TILE, _tile


@triton.jit
def _paths_multi(PARENTS, OFF, PATHS, DEPTHS):
    local = tl.program_id(0)
    item = tl.program_id(1)
    r0 = tl.load(OFF + item)
    w = tl.load(OFF + item + 1) - r0
    if local < w:
        cur = local
        depth = 0
        while (cur >= 0) & (depth < w):
            depth += 1
            cur = tl.load(PARENTS + r0 + cur)
        tl.store(DEPTHS + r0 + local, depth)
        cur = local
        slot = depth - 1
        while slot >= 0:
            tl.store(PATHS + (r0 + local) * 128 + slot, cur)
            cur = tl.load(PARENTS + r0 + cur)
            slot -= 1


@triton.jit
def _shared_multi(Q, KCT, VCT, PT, OFF, PBASE, PO, PM, PL, MAXFULL, CH: tl.constexpr,
                  H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, SCALE: tl.constexpr):
    row_block = tl.program_id(0)
    hk = tl.program_id(1)
    item = tl.program_id(2) // MAXFULL
    chunk = tl.program_id(2) % MAXFULL
    P = tl.load(PT + item)
    r0 = tl.load(OFF + item)
    W = tl.load(OFF + item + 1) - r0
    if (chunk < P // CH) & (row_block * 16 < W * G):
        KC = tl.load(KCT + item).to(tl.pointer_type(tl.bfloat16))
        VC = tl.load(VCT + item).to(tl.pointer_type(tl.bfloat16))
        pbase = tl.load(PBASE + item)
        rr = row_block * 16 + tl.arange(0, 16)
        node = rr // G
        head = hk * G + rr % G
        d = tl.arange(0, D)
        key = chunk * CH + tl.arange(0, 64)
        q = tl.load(Q + ((r0 + node[:, None]) * H + head[:, None]) * D + d[None, :],
                    mask=(rr[:, None] < W * G), other=0).to(tl.bfloat16)
        m = tl.full((16,), float("-inf"), tl.float32)
        l = tl.zeros((16,), tl.float32)
        o = tl.zeros((16, D), tl.float32)
        for t in range(CH // 64):
            ki = key + t * 64
            kk = tl.load(KC + (ki[:, None] * HK + hk) * D + d[None, :]).to(tl.bfloat16)
            vv = tl.load(VC + (ki[:, None] * HK + hk) * D + d[None, :]).to(tl.bfloat16)
            m, l, o = _tile(q, kk, vv, m, l, o, ki < P, SCALE)
        base = ((pbase + chunk * W + node) * H + head)
        tl.store(PO + base[:, None] * D + d[None, :], o, mask=rr[:, None] < W * G)
        tl.store(PM + base, m, mask=rr < W * G)
        tl.store(PL + base, l, mask=rr < W * G)


@triton.jit
def _tail_multi(Q, KN, VN, KCT, VCT, PT, OFF, PBASE, PATHS, DEPTHS, PO, PM, PL, MAXTAIL,
                H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, CH: tl.constexpr,
                SCALE: tl.constexpr):
    local = tl.program_id(0)
    hk = tl.program_id(1)
    item = tl.program_id(2) // MAXTAIL
    extra = tl.program_id(2) % MAXTAIL
    P = tl.load(PT + item)
    r0 = tl.load(OFF + item)
    W = tl.load(OFF + item + 1) - r0
    full = P // CH
    nch = (P + W + CH - 1) // CH
    chunk = full + extra
    if (local < W) & (chunk < nch):
        KC = tl.load(KCT + item).to(tl.pointer_type(tl.bfloat16))
        VC = tl.load(VCT + item).to(tl.pointer_type(tl.bfloat16))
        pbase = tl.load(PBASE + item)
        node = r0 + local
        gg = tl.arange(0, 16)
        d = tl.arange(0, D)
        q = tl.load(Q + (node * H + hk * G + gg[:, None]) * D + d[None, :],
                    mask=gg[:, None] < G, other=0).to(tl.bfloat16)
        depth = tl.load(DEPTHS + node)
        m = tl.full((16,), float("-inf"), tl.float32)
        l = tl.zeros((16,), tl.float32)
        o = tl.zeros((16, D), tl.float32)
        key = chunk * CH + tl.arange(0, 64)
        for t in range(CH // 64):
            logical = key + t * 64
            committed = logical < P
            path_slot = logical - P
            on_path = (path_slot >= 0) & (path_slot < depth)
            path_node = tl.load(PATHS + node * 128 + path_slot, mask=on_path, other=0)
            kc = tl.load(KC + (logical[:, None] * HK + hk) * D + d[None, :], mask=committed[:, None], other=0)
            vc = tl.load(VC + (logical[:, None] * HK + hk) * D + d[None, :], mask=committed[:, None], other=0)
            kn = tl.load(KN + ((r0 + path_node[:, None]) * HK + hk) * D + d[None, :], mask=on_path[:, None], other=0)
            vn = tl.load(VN + ((r0 + path_node[:, None]) * HK + hk) * D + d[None, :], mask=on_path[:, None], other=0)
            kk = tl.where(committed[:, None], kc, kn).to(tl.bfloat16)
            vv = tl.where(committed[:, None], vc, vn).to(tl.bfloat16)
            m, l, o = _tile(q, kk, vv, m, l, o, committed | on_path, SCALE)
        base = ((pbase + chunk * W + local) * H + hk * G + gg)
        tl.store(PO + base[:, None] * D + d[None, :], o, mask=gg[:, None] < G)
        tl.store(PM + base, m, mask=gg < G)
        tl.store(PL + base, l, mask=gg < G)


@triton.jit
def _merge_multi(PO, PM, PL, PT, OFF, PBASE, OUT, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                 G: tl.constexpr, CH: tl.constexpr):
    local = tl.program_id(0)
    hk = tl.program_id(1)
    item = tl.program_id(2)
    P = tl.load(PT + item)
    r0 = tl.load(OFF + item)
    W = tl.load(OFF + item + 1) - r0
    if local < W:
        nch = (P + W + CH - 1) // CH
        pbase = tl.load(PBASE + item)
        gg = tl.arange(0, 16)
        d = tl.arange(0, D)
        head = hk * G + gg
        m = tl.full((16,), float("-inf"), tl.float32)
        l = tl.zeros((16,), tl.float32)
        o = tl.zeros((16, D), tl.float32)
        for chunk in range(nch):
            base = ((pbase + chunk * W + local) * H + head)
            cm = tl.load(PM + base, mask=gg < G, other=float("-inf"))
            cl = tl.load(PL + base, mask=gg < G, other=0.0)
            co = tl.load(PO + base[:, None] * D + d[None, :], mask=gg[:, None] < G, other=0.0)
            active = cl > 0.0
            next_m = tl.where(active, tl.maximum(m, cm), m)
            a = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
            b = tl.where(active, tl.exp(cm - next_m), 0.0)
            o = o * a[:, None] + co * b[:, None]
            l = l * a + cl * b
            m = next_m
        result = o / l[:, None]
        tl.store(OUT + ((r0 + local) * H + head[:, None]) * D + d[None, :], result.to(tl.bfloat16),
                 mask=gg[:, None] < G)


class Plan:
    """What every attention layer of one multi-request forward shares: extents, partial-buffer bases, paths."""

    def __init__(self, parents: torch.Tensor, offsets: torch.Tensor, lengths: list[int], widths: list[int],
                 caches: list[list], h: int, d: int, chunk_size: int = CHUNK):
        """``caches``: per attention layer, one (K, V) buffer pair per request."""

        dev = parents.device
        ch = self.ch = chunk_size
        self.n, self.h, self.d = len(widths), h, d
        if max(widths) > MAX_NODES:
            raise ValueError("at most 128 window rows a request")
        nchs = [(p + w + ch - 1) // ch for p, w in zip(lengths, widths)]
        fulls = [p // ch for p in lengths]
        pbase, total = [], 0
        for nc, w in zip(nchs, widths):
            pbase.append(total)
            total += nc * w
        self.total, self.maxw = total, max(widths)
        self.maxfull = max(fulls)
        self.maxtail = max(nc - f for nc, f in zip(nchs, fulls))
        n, L = self.n, len(caches)
        flat = [k.data_ptr() for layer in caches for k, _ in layer] + [v.data_ptr() for layer in caches for _, v in layer]
        t = torch.tensor(lengths + pbase + flat, dtype=torch.int64).to(dev)
        self.pt, self.pb = t[:n], t[n:2 * n]
        self.kct = t[2 * n:2 * n + L * n].view(L, n)
        self.vct = t[2 * n + L * n:].view(L, n)
        R = parents.shape[0]
        self.offsets = offsets
        self.paths = torch.empty((R, MAX_NODES), dtype=torch.int32, device=dev)
        self.depths = torch.empty((R,), dtype=torch.int32, device=dev)
        _paths_multi[(self.maxw, n)](parents, offsets, self.paths, self.depths, num_warps=1)

    def __call__(self, layer: int, q: torch.Tensor, k_nodes: torch.Tensor, v_nodes: torch.Tensor, *,
                 scale: float) -> torch.Tensor:
        """Layer ``layer`` (index into ``caches``): q (R, H, D), k/v_nodes (R, HK, D), every request back to back."""

        h, d = q.shape[1], q.shape[2]
        hk = k_nodes.shape[1]
        g = h // hk
        if g > QUERY_TILE:
            raise ValueError("unsupported attention shape")
        po = torch.empty((self.total, h, d), dtype=torch.float32, device=q.device)
        pm = torch.empty((self.total, h), dtype=torch.float32, device=q.device)
        pl = torch.empty_like(pm)
        kct, vct = self.kct[layer], self.vct[layer]
        if self.maxfull:
            _shared_multi[(triton.cdiv(self.maxw * g, QUERY_TILE), hk, self.n * self.maxfull)](
                q, kct, vct, self.pt, self.offsets, self.pb, po, pm, pl, self.maxfull,
                CH=self.ch, H=h, HK=hk, D=d, G=g, SCALE=scale, num_warps=4, num_stages=1)
        _tail_multi[(self.maxw, hk, self.n * self.maxtail)](
            q, k_nodes, v_nodes, kct, vct, self.pt, self.offsets, self.pb, self.paths, self.depths, po, pm, pl,
            self.maxtail, H=h, HK=hk, D=d, G=g, CH=self.ch, SCALE=scale, num_warps=4, num_stages=1)
        out = torch.empty_like(q)
        _merge_multi[(self.maxw, hk, self.n)](po, pm, pl, self.pt, self.offsets, self.pb, out,
                                              H=h, HK=hk, D=d, G=g, CH=self.ch, num_warps=4)
        return out
