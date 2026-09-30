"""Qwen3.6 MoE in fp32 straight from checkpoint files, a layer at a time: the quality reference for the CUDA routes.

``forward`` reads the bf16 release (``Qwen/Qwen3.6-35B-A3B``) or the NVFP4 one with every weight dequantized exactly to
fp32 (NVFP4 as code x e4m3 x scale_2, FP8 as code x scale), keeps every activation in fp32 and runs a batch of
passages together layer by layer, so one layer's weights are on the GPU at a time. It is the model's arithmetic as the
Hugging Face definition writes it (zero-centred RMSNorms, DeltaNet's recurrence token by token, gated attention with
partial rotary, top-8 softmax routing renormalized, a sigmoid-gated shared expert), never TensorFold's kernels.

``python -m tensorfold.families.qwen3_5_moe.cuda.reference`` scores passages: ``reference DIR OUT`` with this
forward, ``route DIR OUT`` with the CUDA engine's own forwards (the verify path in 128-row windows, which serial
decoding equals, and the prompt path), ``compare OUT...`` tabulates next-token NLL and top-1 agreement. The
passages are public text from the Python standard library (``pydoc_data`` topics and module source).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from tensorfold.families.qwen3_5.cuda.weights import Config


class Source:
    """A checkpoint's tensors, each weight as the exact fp32 matrix on the GPU."""

    def __init__(self, model_dir: str | Path, device: str = "cuda") -> None:
        from .modelopt import _Reader

        self.rd, self.device = _Reader(Path(model_dir)), device

    def vector(self, name: str) -> torch.Tensor:
        return self.rd.get(name).to(self.device).float()

    def weight(self, name: str) -> torch.Tensor:
        from tensorfold.families.qwen4_exp.cuda import nvfp4

        w = self.rd.get(name + ".weight").to(self.device)
        if self.rd.has(name + ".weight_scale_2"):
            return nvfp4.dequantize(w, self.rd.get(name + ".weight_scale").to(self.device),
                                    self.rd.get(name + ".weight_scale_2"))
        if w.dtype == torch.float8_e4m3fn:
            scale = self.rd.get(name + ".weight_scale").to(self.device).float()
            return nvfp4._bits_to_f32(nvfp4.e4m3_bits(w)) * (scale if scale.dim() == 0 else scale.reshape(-1, 1))
        return w.float()

    def expert(self, prefix: str, e: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expert ``e``'s gate [ni, d], up [ni, d] and down [d, ni]."""

        if self.rd.has(f"{prefix}experts.gate_up_proj"):          # the bf16 release's stacks
            if getattr(self, "_stack", (None,))[0] != prefix:
                self._stack = (prefix, self.rd.get(f"{prefix}experts.gate_up_proj"),
                               self.rd.get(f"{prefix}experts.down_proj"))
            _, gu, dn = self._stack
            gu = gu[e].to(self.device).float()
            ni = gu.shape[0] // 2
            return gu[:ni], gu[ni:], dn[e].to(self.device).float()
        return tuple(self.weight(f"{prefix}experts.{e}.{p}_proj") for p in ("gate", "up", "down"))


def _rms(x: torch.Tensor, w: torch.Tensor | None, eps: float, centred: bool = True) -> torch.Tensor:
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if w is None else y * ((1.0 + w) if centred else w)


def _deltanet(src: Source, p: str, h: torch.Tensor, c: Config) -> torch.Tensor:
    P, T, _ = h.shape
    qkv = h @ src.weight(p + "in_proj_qkv").T
    z = (h @ src.weight(p + "in_proj_z").T).view(P, T, c.v_heads, c.dv)
    b = h @ src.weight(p + "in_proj_b").T
    a = h @ src.weight(p + "in_proj_a").T
    kern = src.vector(p + "conv1d.weight").reshape(qkv.shape[-1], 1, c.conv_kernel)
    conv = F.conv1d(F.pad(qkv.transpose(1, 2), (c.conv_kernel - 1, 0)), kern, groups=qkv.shape[-1])
    act = F.silu(conv).transpose(1, 2)
    kd = c.k_heads * c.dk
    q = act[..., :kd].reshape(P, T, c.k_heads, c.dk)
    k = act[..., kd:2 * kd].reshape(P, T, c.k_heads, c.dk)
    v = act[..., 2 * kd:].reshape(P, T, c.v_heads, c.dv)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * c.dk ** -0.5
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    rep = c.v_heads // c.k_heads
    q, k = q.repeat_interleave(rep, 2), k.repeat_interleave(rep, 2)
    beta = torch.sigmoid(b)
    g = -torch.exp(src.vector(p + "A_log")) * F.softplus(a + src.vector(p + "dt_bias"))
    state = torch.zeros((P, c.v_heads, c.dk, c.dv), device=h.device)
    out = torch.empty((P, T, c.v_heads, c.dv), device=h.device)
    for t in range(T):
        state = state * torch.exp(g[:, t])[..., None, None]
        kt = k[:, t][..., None]
        delta = (v[:, t] - (state * kt).sum(-2)) * beta[:, t][..., None]
        state = state + kt * delta[..., None, :]
        out[:, t] = (state * q[:, t][..., None]).sum(-2)
    y = _rms(out, src.vector(p + "norm.weight"), c.eps, centred=False) * F.silu(z)
    return y.reshape(P, T, -1) @ src.weight(p + "out_proj").T


def _attention(src: Source, p: str, h: torch.Tensor, c: Config) -> torch.Tensor:
    P, T, _ = h.shape
    qg = (h @ src.weight(p + "q_proj").T).view(P, T, c.heads, 2 * c.head_dim)
    q, gate = qg[..., :c.head_dim], qg[..., c.head_dim:]
    k = (h @ src.weight(p + "k_proj").T).view(P, T, c.kv_heads, c.head_dim)
    v = (h @ src.weight(p + "v_proj").T).view(P, T, c.kv_heads, c.head_dim)
    q, k = _rms(q, src.vector(p + "q_norm.weight"), c.eps), _rms(k, src.vector(p + "k_norm.weight"), c.eps)
    half = c.rope_dims // 2
    inv = c.rope_theta ** (-torch.arange(0, half, dtype=torch.float64, device=h.device) / half)
    ang = (torch.arange(T, dtype=torch.float64, device=h.device)[:, None] * inv[None]).float()
    cos, sin = torch.cos(ang)[None, :, None], torch.sin(ang)[None, :, None]

    def rope(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :half], x[..., half:2 * half]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., 2 * half:]], -1)

    q, k = rope(q).transpose(1, 2), rope(k).transpose(1, 2)
    rep = c.heads // c.kv_heads
    k, v = k.repeat_interleave(rep, 1), v.transpose(1, 2).repeat_interleave(rep, 1)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=c.head_dim ** -0.5)
    o = o.transpose(1, 2) * torch.sigmoid(gate)
    return o.reshape(P, T, -1) @ src.weight(p + "o_proj").T


def _moe(src: Source, p: str, h: torch.Tensor, c: Config) -> torch.Tensor:
    x = h.reshape(-1, h.shape[-1])
    probs = torch.softmax(x @ src.weight(p + "gate").T, -1)
    top, pick = torch.topk(probs, c.top_k, -1)
    top = top / top.sum(-1, keepdim=True)
    out = torch.zeros_like(x)
    for e in pick.unique().tolist():
        rows, slots = torch.nonzero(pick == e, as_tuple=True)
        gate, up, down = src.expert(p, e)
        y = (F.silu(x[rows] @ gate.T) * (x[rows] @ up.T)) @ down.T
        out.index_add_(0, rows, top[rows, slots][:, None] * y)
    sp = p + "shared_expert."
    shared = (F.silu(x @ src.weight(sp + "gate_proj").T) * (x @ src.weight(sp + "up_proj").T)) \
        @ src.weight(sp + "down_proj").T
    out += torch.sigmoid(x @ src.weight(p + "shared_expert_gate").T) * shared
    return out.view_as(h)


@torch.no_grad()
def forward(model_dir: str | Path, ids: torch.Tensor, device: str = "cuda", *,
            full: bool = False) -> dict[str, torch.Tensor]:
    """Passages ``ids`` [P, T]: each position's log-probability of the next token (``logp`` [P, T - 1]), its argmax
    (``top`` [P, T - 1]) and its logsumexp; ``full`` adds every position's logits (a small vocabulary's)."""

    c = Config.read(model_dir)
    src = Source(model_dir, device)
    ids = ids.to(device)
    x = src.rd.get("model.embed_tokens.weight").to(device)[ids].float()
    for i in range(c.layers):
        p = f"model.layers.{i}."
        h = _rms(x, src.vector(p + "input_layernorm.weight"), c.eps)
        x = x + (_deltanet(src, p + "linear_attn.", h, c) if c.is_linear(i) else
                 _attention(src, p + "self_attn.", h, c))
        h = _rms(x, src.vector(p + "post_attention_layernorm.weight"), c.eps)
        x = x + _moe(src, p + "mlp.", h, c)
        torch.cuda.empty_cache()
    h = _rms(x, src.vector("model.norm.weight"), c.eps)[:, :-1].reshape(-1, c.hidden)
    target = ids[:, 1:].reshape(-1)
    best = torch.full(target.shape, -math.inf, device=device)
    top = torch.zeros_like(target)
    lse = torch.full(target.shape, -math.inf, device=device)
    hit = torch.zeros(target.shape, device=device)
    head = src.weight("lm_head")                      # [vocab, d] fp32 (2 GB)
    kept = []
    for a in range(0, head.shape[0], 32768):
        logits = h @ head[a:a + 32768].T
        if full:
            kept.append(logits.cpu())
        value, col = logits.max(-1)
        take = value > best
        best, top = torch.where(take, value, best), torch.where(take, col + a, top)
        lse = torch.logaddexp(lse, torch.logsumexp(logits, -1))
        inside = (target >= a) & (target < a + logits.shape[1])
        hit[inside] = logits[inside, target[inside] - a]
    src.rd.close()
    P, T = ids.shape
    got = {"logp": (hit - lse).view(P, T - 1).cpu(), "top": top.view(P, T - 1).cpu(), "lse": lse.view(P, T - 1).cpu()}
    if full:
        got["logits"] = torch.cat(kept, 1).view(P, T - 1, -1)
    return got


@torch.no_grad()
def route(model_dir: str | Path, ids: torch.Tensor, window: int = 128) -> dict[str, torch.Tensor]:
    """The CUDA engine's own scores of the same passages: ``logp``/``top`` from verify windows of ``window`` rows
    (the bits serial decoding gives each row) and ``prefill_logp``/``prefill_top`` from the prompt path."""

    from tensorfold.families.qwen3_5.cuda.forward import State, _mm, commit, tree_forward
    from tensorfold.families.qwen3_5.cuda.prefill import chunks, prefill_chunk

    from .weights import load

    w = load(model_dir)
    out: dict[str, list[torch.Tensor]] = {"logp": [], "top": [], "prefill_logp": [], "prefill_top": []}

    def score(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lf = logits.float()
        return (lf.gather(1, target[:, None])[:, 0] - torch.logsumexp(lf, -1)).cpu(), lf.argmax(-1).cpu()

    for row in ids.tolist():
        dev = torch.tensor(row, dtype=torch.int32, device="cuda")
        target = dev[1:].long()
        st, parts = State(w), []
        for a in range(0, len(row) - 1, window):
            b = min(a + window, len(row) - 1)
            logits, record = tree_forward(w, dev[a:b], list(range(-1, b - a - 1)), st)
            commit(st, record, list(range(b - a)))
            parts.append(logits)
        logp, top = score(torch.cat(parts), target)
        out["logp"].append(logp)
        out["top"].append(top)
        st, parts = State(w), []
        for a, b in chunks(0, len(row) - 1):
            normed, _ = prefill_chunk(w, dev[a:b], st, every=True)
            parts.append(_mm(normed, w.head))
        logp, top = score(torch.cat(parts), target)
        out["prefill_logp"].append(logp)
        out["prefill_top"].append(top)
    return {k: torch.stack(v) for k, v in out.items()}


def passages(tokenizer: str | Path, count: int = 8, tokens: int = 1024) -> torch.Tensor:
    """``count`` public passages of ``tokens`` tokens each: ``pydoc_data`` topics (the Python documentation's prose)
    and standard-library source, the same ids for every checkpoint of this tokenizer."""

    import inspect
    import json as json_module
    import textwrap

    from pydoc_data.topics import topics
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer))
    texts = ["\n\n".join(topics[k] for k in names) for names in (
        ("assignment", "augassign"), ("class", "function", "customization"), ("exceptions", "try", "raise"),
        ("for", "while", "if", "with", "execmodel"), ("import", "lambda", "naming"), ("typesseq", "typesmapping"))]
    texts += [inspect.getsource(textwrap), inspect.getsource(json_module.decoder)]
    rows = []
    for text in texts[:count]:
        ids = tok.encode(text).ids
        if len(ids) < tokens:
            raise ValueError(f"a passage has {len(ids)} tokens, fewer than {tokens}")
        rows.append(ids[:tokens])
    return torch.tensor(rows, dtype=torch.int64)


def compare(files: list[str]) -> str:
    """A table of next-token NLL and top-1 agreement (the first file's argmax is the reference)."""

    runs = [(Path(f).stem, torch.load(f)) for f in files]
    ref_name, ref = runs[0]
    lines = [f"| scores | NLL (nats/token) | top-1 = text | top-1 = {ref_name} |", "| --- | ---: | ---: | ---: |"]
    for name, run in runs:
        for kind in ("", "prefill_"):
            if kind + "logp" not in run:
                continue
            logp, top = run[kind + "logp"], run[kind + "top"]
            text, same = (top == run["ids"][:, 1:]).float().mean(), (top == ref["top"]).float().mean()
            lines.append(f"| {name}{' (prompt path)' if kind else ''} | {-logp.mean().item():.4f} | "
                         f"{text.item():.2%} | {same.item():.2%} |")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("reference", "route", "compare"):
        print(__doc__)
        return 2
    if argv[0] == "compare":
        print(compare(argv[1:]))
        return 0
    model_dir, out, tokenizer = Path(argv[1]), argv[2], Path(argv[3] if len(argv) > 3 else argv[1]) / "tokenizer.json"
    ids = passages(tokenizer)
    scores = forward(model_dir, ids) if argv[0] == "reference" else route(model_dir, ids)
    torch.save({**scores, "ids": ids, "model": str(model_dir)}, out)
    print(json.dumps({"out": out, "nll": -scores["logp"].mean().item()}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
