"""GPTQ for the 4-bit converter: each projection's rounding error is spread over its not-yet-rounded inputs.

The layers are quantized in order, on public calibration text run through the engine's own prompt kernels: a
projection's Hessian comes from the inputs the already-quantized projections before it produce, and the rounded
weights keep MLX's affine format (codes 0-15, a bf16 scale and bias a group of 64 inputs, groups in input order).
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

BLOCK = 128            # columns whose updates are applied lazily together (a multiple of the group)
DAMP = 0.01            # of the Hessian's mean diagonal, added to it


def dequantized(codes: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """``code * scale + bias`` rounded once to bf16, as the 4-bit prompt kernel's fused multiply-add rounds it."""

    return (codes.double() * scale.double() + bias.double()).to(torch.bfloat16).float()


def group_params(w: torch.Tensor, bits: int, search: bool):
    """Per-row scale and bias (bf16 values, fp32 tensors) for a group's current weights (n, group)."""

    from .convert import SHRINK, _codes_for

    levels = 2 ** bits - 1
    lo, hi = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
    best = None
    for shrink in (SHRINK if search else (1.0,)):
        scale, bias, codes = _codes_for(w, lo * shrink, hi * shrink, levels)
        err = (dequantized(codes, scale, bias) - w).square().sum(-1, keepdim=True)
        if best is None:
            best = (scale, bias, err)
        else:
            better = err < best[2]
            best = tuple(torch.where(better, a, b) for a, b in zip((scale, bias, err), best))
    return best[0], best[1]


@torch.no_grad()
def gptq(weight: torch.Tensor, hessian: torch.Tensor, *, bits: int = 4, group: int = 64, search: bool = True,
         damp: float = DAMP, act_order: bool = False):
    """(n, k) weight and its (k, k) input Hessian -> codes (n, k) int64, scales and biases (n, k / group) fp32.

    ``act_order`` rounds the inputs with the largest Hessian diagonal first; the groups then keep their input
    order in the checkpoint, with scales and biases fixed from the weights before any update (static groups).
    """

    n, k = weight.shape
    w = weight.float().clone()
    h = hessian.float().clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    levels = 2 ** bits - 1
    scales = torch.zeros((n, k // group), device=w.device)
    biases = torch.zeros((n, k // group), device=w.device)
    order = torch.arange(k, device=w.device)
    if act_order:
        for g in range(k // group):
            s, t = group_params(w[:, g * group:(g + 1) * group], bits, search)
            scales[:, g], biases[:, g] = s[:, 0], t[:, 0]
        order = torch.argsort(torch.diag(h), descending=True)
        w, h = w[:, order], h[order][:, order]
    h += damp * torch.mean(torch.diag(h)) * torch.eye(k, device=h.device)
    inverse = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    del h
    codes = torch.zeros((n, k), dtype=torch.int64, device=w.device)
    groups = (order // group).tolist()
    for a in range(0, k, BLOCK):
        b = min(a + BLOCK, k)
        block = w[:, a:b].clone()
        errors = torch.zeros_like(block)
        inv = inverse[a:b, a:b]
        for i in range(b - a):
            col = a + i
            if not act_order and col % group == 0:
                s, t = group_params(block[:, i:i + group], bits, search)
                scales[:, col // group], biases[:, col // group] = s[:, 0], t[:, 0]
            s, t = scales[:, groups[col]], biases[:, groups[col]]
            code = torch.round((block[:, i] - t) / s).clamp_(0, levels)
            codes[:, col] = code.to(torch.int64)
            err = (block[:, i] - dequantized(code, s, t)) / inv[i, i]
            block[:, i:] -= err[:, None] * inv[i, i:][None, :]
            errors[:, i] = err
        w[:, b:] -= errors @ inverse[a:b, b:]
    if act_order:
        codes[:, order] = codes.clone()
    return codes, scales, biases


def pack_codes(codes: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """(n, k) codes -> MLX's (n, k * bits / 32) words, lowest bits first, as int32."""

    per = 32 // bits
    n, k = codes.shape
    shifts = torch.arange(per, device=codes.device, dtype=torch.int64) * bits
    words = (codes.reshape(n, k // per, per) << shifts).sum(-1)
    return torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)


class Hessian:
    """2/N sum of x x^T over calibration rows, accumulated in fp32 a chunk at a time."""

    def __init__(self, k: int, device):
        self.h = torch.zeros((k, k), dtype=torch.float32, device=device)
        self.rows = 0

    def add(self, x: torch.Tensor, chunk: int = 8192) -> None:
        for a in range(0, x.shape[0], chunk):
            part = x[a:a + chunk].float()
            self.h.addmm_(part.t(), part)
        self.rows += x.shape[0]

    def value(self) -> torch.Tensor:
        return self.h * (2.0 / max(1, self.rows))


@torch.no_grad()
def quantize_layers(w, texts: Sequence[Sequence[int]], *, bits: int = 4, group: int = 64, search: bool = True,
                    act_order: bool = False, log: Callable[[str], None] = print) -> dict:
    """Quantize every projection of the loaded bf16 ``Weights`` in place (bf16 dequantized weights for the next
    layers' inputs) and return {(layer, name): (words, scales, biases)} for the checkpoint."""

    from tensorfold.cuda.kernels.prefill_attention import attention_texts
    from tensorfold.families.qwen3_5.cuda import glue as dense_glue

    from .cuda import glue
    from .cuda.forward import MLP_ROWS, pack

    torch.backends.cuda.matmul.allow_tf32 = False
    c = w.config
    dev = w.norm.device
    ids, pos, blocks, _ = pack(texts, dev, vocab=c.vocab, positions=w.cos.shape[0])
    x = dense_glue.embedding(ids, w.embed).float()
    pending = None
    out = {}
    report = []

    def run(layer_index, name, linear, inputs: list[torch.Tensor]):
        hess = Hessian(linear.k, dev)
        for part in inputs:
            hess.add(part)
        codes, scales, biases = gptq(linear.weight, hess.value(), bits=bits, group=group, search=search,
                                     act_order=act_order)
        rounded = dequantized(codes, scales.repeat_interleave(group, 1), biases.repeat_interleave(group, 1))
        before = linear.weight.float()
        report.append(float((rounded - before).norm() / before.norm()))
        linear.weight.copy_(rounded.to(torch.bfloat16))
        out[(layer_index, name)] = (pack_codes(codes, bits).cpu(), scales.to(torch.bfloat16).cpu(),
                                    biases.to(torch.bfloat16).cpu())
        del hess, codes, rounded, before

    for i, layer in enumerate(w.layers):
        h = glue.add_rmsnorm(x, pending, layer.input_norm, c.eps)
        run(i, "qkv", layer.qkv, [h])
        q, k, v = glue.qkv(layer.qkv(h), layer.q_norm, layer.k_norm, pos, w.cos, w.sin, c.eps, heads=c.heads,
                           kv_heads=c.kv_heads, head_dim=c.head_dim)
        del h
        o = attention_texts(q, k, v, blocks, scale=c.head_dim ** -0.5).view(-1, c.heads * c.head_dim)
        del q, k, v
        run(i, "o", layer.o, [o])
        h = glue.add_rmsnorm(x, layer.o(o, f32=True), layer.post_norm, c.eps)
        del o
        run(i, "gate_up", layer.gate_up, [h])
        acts = [glue.swiglu(layer.gate_up(h[a:a + MLP_ROWS])) for a in range(0, h.shape[0], MLP_ROWS)]
        run(i, "down", layer.down, acts)
        pending = torch.cat([layer.down(a, f32=True) for a in acts])
        del h, acts
        torch.cuda.empty_cache()
        log(f"[tensorfold] layer {i}: GPTQ relative weight change {sum(report[-4:]) / 4:.4f}")
    return out
