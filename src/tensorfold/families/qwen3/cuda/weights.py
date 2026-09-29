"""Load a Qwen3 embedding checkpoint for prompt rows: bf16 as shipped, or MLX affine 4-bit (groups of 32 or 64) packed
once for ``qmm``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from tensorfold.cuda.kernels import dense, qmm


@dataclass
class Linear:
    """A projection for prompt rows: an unquantized bf16 (N, K) weight, or a packed 4-bit ``qmm.Q4``."""

    weight: torch.Tensor | qmm.Q4

    @property
    def n(self) -> int:
        return self.weight.n if isinstance(self.weight, qmm.Q4) else int(self.weight.shape[0])

    @property
    def k(self) -> int:
        return self.weight.k if isinstance(self.weight, qmm.Q4) else int(self.weight.shape[1])

    def nbytes(self) -> int:
        if isinstance(self.weight, qmm.Q4):
            return self.weight.nbytes()
        return self.weight.numel() * self.weight.element_size()

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None, *, f32: bool = False) -> torch.Tensor:
        """x (M, K) bf16 -> (M, N) bf16, or fp32 unrounded; either kernel adds each output's K in one fixed chain."""

        if isinstance(self.weight, qmm.Q4):
            return qmm.prefill_matmul(x, self.weight, f32=f32, tile=q4_tile(x.shape[0]), out=out)
        return dense.prefill_matmul(x, self.weight, f32=f32, out=out)


def q4_tile(m: int) -> int:
    """``qmm_prefill``'s block shape for ``m`` rows (64 x 64 spreads a few rows' weights over the SMs); same bits."""

    return 3 if m <= 128 else 0


@dataclass
class Config:
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    eps: float
    rope_theta: float
    max_positions: int

    @classmethod
    def read(cls, model_dir: str | Path) -> "Config":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        t = raw.get("text_config") or raw
        return cls(hidden=int(t["hidden_size"]), intermediate=int(t["intermediate_size"]),
                   layers=int(t["num_hidden_layers"]), heads=int(t["num_attention_heads"]),
                   kv_heads=int(t["num_key_value_heads"]),
                   head_dim=int(t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]),
                   vocab=int(t["vocab_size"]), eps=float(t.get("rms_norm_eps", 1e-6)),
                   rope_theta=float(t.get("rope_theta") or (t.get("rope_parameters") or {}).get("rope_theta", 1e6)),
                   max_positions=int(t.get("max_position_embeddings") or 0))


@dataclass
class Layer:
    input_norm: torch.Tensor      # (hidden,) bf16
    post_norm: torch.Tensor
    qkv: Linear                   # [q | k | v] rows: (heads + 2 kv_heads) * head_dim outputs
    q_norm: torch.Tensor          # (head_dim,) bf16
    k_norm: torch.Tensor
    o: Linear
    gate_up: Linear               # [gate | up] rows: 2 * intermediate outputs
    down: Linear


@dataclass
class Weights:
    config: Config
    embed: object                 # a ``qwen3_5.cuda.weights.QLinear``: bf16 rows, or MLX affine rows
    layers: list[Layer]
    norm: torch.Tensor
    cos: torch.Tensor | None = None   # (positions, head_dim / 2) fp32, Hugging Face's rotary angles
    sin: torch.Tensor | None = None
    quant: str = "bf16"               # "bf16", or "mlx" for affine 4-bit projections

    def nbytes(self) -> int:
        e = self.embed
        total = sum(t.numel() * t.element_size() for t in (e.weight, e.scales, e.biases) if t is not None)
        for layer in self.layers:
            total += sum(m.nbytes() for m in (layer.qkv, layer.o, layer.gate_up, layer.down))
        return total


def rotary(config: Config, positions: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of Hugging Face's fp32 angles (``inv_freq @ position``) for positions [0, positions)."""

    half = config.head_dim // 2
    inv = 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2, dtype=torch.int64, device=device)
                                       .float() / config.head_dim))
    angles = torch.arange(positions, device=device, dtype=torch.float32)[:, None] * inv[None, :half]
    return angles.cos().contiguous(), angles.sin().contiguous()


def weight_transform(model_dir: str | Path):
    """Startup estimate: bytes each checkpoint tensor holds once loaded (the head, if any, is never read)."""

    from tensorfold.cuda.geometry import linear_weights

    def transform(name: str, info: dict) -> tuple[int, int]:
        if name.startswith("lm_head.") or name.startswith("model.lm_head."):
            return 0, 0
        return linear_weights(name, info)

    return transform


def load(model_dir: str | Path, device: str = "cuda", *, positions: int = 0) -> Weights:
    """Read every projection as stored, stacking [q | k | v] and [gate | up] (stacking never changes a row's bits)."""

    from tensorfold.families.qwen3_5.cuda.weights import QLinear, _Tensors
    from tensorfold.quantization import resolve_affine, validate_shapes

    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    raw = json.loads((model_dir / "config.json").read_text())
    t = _Tensors(model_dir, device, skip=lambda name: name.startswith(("lm_head.", "model.lm_head.")))
    prefix = "model." if "model.embed_tokens.weight" in t else ""

    def get(name: str) -> torch.Tensor:
        return t.pop(prefix + name)

    def stored(name: str) -> QLinear:
        w = get(name + ".weight")
        spec = resolve_affine(raw, prefix + name)
        if spec is None:
            if w.ndim != 2 or w.dtype != torch.bfloat16:
                raise ValueError(f"{name}: unquantized weights must be 2-D bf16, not {w.dtype}")
            return QLinear(w.contiguous(), None, None, layout="dense", bits=0, gs=0)
        if w.dtype not in (torch.int32, torch.uint32):
            raise ValueError(f"{name} declares affine quantization but its words are not 32-bit integers")
        scales, biases = get(name + ".scales"), get(name + ".biases")
        if scales.dtype != torch.bfloat16 or biases.dtype != torch.bfloat16:
            raise ValueError(f"{name}: affine scales and biases must be bf16")
        validate_shapes(w.shape, scales.shape, biases.shape, spec)
        return QLinear(w.view(torch.int32).contiguous(), scales.contiguous(), biases.contiguous(),
                       gs=spec.group_size, bits=spec.bits)

    def linear(*names: str) -> Linear:
        parts = [stored(n) for n in names]
        if len({q.layout for q in parts}) != 1 or len({(q.bits, q.gs, q.k) for q in parts}) != 1:
            raise ValueError(f"{', '.join(names)} must share one storage format to be stacked")
        if parts[0].layout == "dense":
            w = parts[0].weight if len(parts) == 1 else torch.cat([q.weight for q in parts]).contiguous()
            return Linear(w)
        if parts[0].bits != 4 or parts[0].gs not in (32, 64):
            raise ValueError(f"{names[0]}: the prompt kernels read 4-bit weights in groups of 32 or 64")
        cat = (lambda ts: ts[0]) if len(parts) == 1 else (lambda ts: torch.cat(ts).contiguous())
        words, scales, biases = (cat([getattr(q, f) for q in parts]) for f in ("weight", "scales", "biases"))
        group = parts[0].gs
        del parts
        return Linear(qmm.pack(words, scales, biases, group))

    layers = []
    for i in range(cfg.layers):
        p = f"layers.{i}."
        a = p + "self_attn."
        layers.append(Layer(
            input_norm=get(p + "input_layernorm.weight").contiguous(),
            post_norm=get(p + "post_attention_layernorm.weight").contiguous(),
            qkv=linear(a + "q_proj", a + "k_proj", a + "v_proj"),
            q_norm=get(a + "q_norm.weight").contiguous(), k_norm=get(a + "k_norm.weight").contiguous(),
            o=linear(a + "o_proj"), gate_up=linear(p + "mlp.gate_proj", p + "mlp.up_proj"),
            down=linear(p + "mlp.down_proj")))
        torch.cuda.empty_cache()
    embed = stored("embed_tokens")
    if embed.layout != "dense" and (embed.bits, embed.gs) not in ((4, 64), (8, 64)):
        raise ValueError("the token table must be unquantized or affine 4- or 8-bit in groups of 64")
    w = Weights(config=cfg, embed=embed, layers=layers, norm=get("norm.weight").contiguous(),
                quant="mlx" if isinstance(layers[0].qkv.weight, qmm.Q4) else "bf16")
    left = list(t)
    t.close()
    if left:
        raise ValueError(f"unused checkpoint tensors: {left[:5]} ...")
    for layer in layers:
        if layer.qkv.n != (cfg.heads + 2 * cfg.kv_heads) * cfg.head_dim or layer.qkv.k != cfg.hidden:
            raise ValueError("q/k/v projections do not match config.json's heads and widths")
        if layer.gate_up.n != 2 * cfg.intermediate or layer.down.k != cfg.intermediate:
            raise ValueError("MLP projections do not match config.json's intermediate width")
        if tuple(layer.q_norm.shape) != (cfg.head_dim,) or tuple(layer.input_norm.shape) != (cfg.hidden,):
            raise ValueError("norm weights do not match config.json's widths")
    w.cos, w.sin = rotary(cfg, max(1, positions), device)
    torch.cuda.empty_cache()
    return w
