"""The embedding forward: texts packed back to back, every kernel row-local and attention within each text.

A text's vector therefore has the same bits alone or in any batch, at any position, beside texts of any length:
the projections add each output's K in one fixed chain whatever the row count (``dense`` and ``qmm`` prompt
kernels), norms, rotary and SwiGLU read one row, and ``attention_texts`` tiles a text's keys from its own start.
The residual stream stays in fp32 (the output and down projections add into it unrounded); the projections read
bf16 rows.
"""

from __future__ import annotations

from typing import Sequence

import torch

from tensorfold.cuda.kernels.prefill_attention import attention_texts, text_blocks
from tensorfold.families.qwen3_5.cuda import glue as dense_glue

from . import glue
from .weights import Layer, Weights

MLP_ROWS = 4096        # rows a [gate | up] matmul takes at once: bounds its workspace, never a row's bits


def _mlp(layer: Layer, h: torch.Tensor) -> torch.Tensor:
    """(T, hidden) bf16 rows -> the MLP's fp32 output, [gate | up] in ``MLP_ROWS``-row pieces."""

    out = torch.empty(h.shape, dtype=torch.float32, device=h.device)
    for a in range(0, h.shape[0], MLP_ROWS):
        b = min(a + MLP_ROWS, h.shape[0])
        layer.down(glue.swiglu(layer.gate_up(h[a:b])), out=out[a:b], f32=True)
    return out


def pack(texts: Sequence[Sequence[int]], device, *, vocab: int, positions: int):
    """Token ids and positions back to back, the attention block table, and each text's last row."""

    lengths = [len(t) for t in texts]
    if not lengths or min(lengths) < 1:
        raise ValueError("embed takes one or more texts of at least one token")
    if max(lengths) > positions:
        raise ValueError(f"a text of {max(lengths)} tokens exceeds the engine's {positions}-token window")
    ids = torch.tensor([int(tok) for text in texts for tok in text], dtype=torch.int64)
    if int(ids.min()) < 0 or int(ids.max()) >= vocab:
        raise ValueError(f"token ids must lie in [0, {vocab})")
    ids = ids.to(torch.int32)
    pos = torch.cat([torch.arange(n, dtype=torch.int32) for n in lengths])
    last = torch.tensor(lengths, dtype=torch.int64).cumsum(0) - 1
    return ids.to(device), pos.to(device), text_blocks(lengths, device), last.to(device)


@torch.no_grad()
def embed(w: Weights, texts: Sequence[Sequence[int]]) -> torch.Tensor:
    """Each text's last-token hidden state after the final norm: (len(texts), hidden) fp32, not yet L2-normalized."""

    c = w.config
    ids, pos, blocks, last = pack(texts, w.norm.device, vocab=c.vocab, positions=w.cos.shape[0])
    x = dense_glue.embedding(ids, w.embed).float()
    pending = None
    for i, layer in enumerate(w.layers):
        h = glue.add_rmsnorm(x, pending, layer.input_norm, c.eps)
        pending = None
        q, k, v = glue.qkv(layer.qkv(h), layer.q_norm, layer.k_norm, pos, w.cos, w.sin, c.eps, heads=c.heads,
                           kv_heads=c.kv_heads, head_dim=c.head_dim)
        del h
        o = attention_texts(q, k, v, blocks, scale=c.head_dim ** -0.5).view(-1, c.heads * c.head_dim)
        del q, k, v
        if i == len(w.layers) - 1:          # the last layer goes on for the pooled rows only
            o, x = o.index_select(0, last), x.index_select(0, last)
        h = glue.add_rmsnorm(x, layer.o(o, f32=True), layer.post_norm, c.eps)
        del o
        pending = _mlp(layer, h)
        del h
    return glue.pool(x, pending, w.norm, c.eps)
