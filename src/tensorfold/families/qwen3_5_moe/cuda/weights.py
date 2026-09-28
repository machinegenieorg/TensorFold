"""Qwen3.6-35B-A3B's weights from the MLX 4-bit checkpoint and the MTP drafter, in the layout the kernels take."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .. import DRAFTER, DRAFTER_MODEL_TYPE
from .checkpoint import DTYPES, Config, Reader, Spec, checkpoint_tensors, layout, layout_problems, mtp_layout

# each centred-norm group's mean weight w in the original checkpoints (MLX stores 1 + w: its group means less one)
NORM_W = {"input_layernorm": -0.06, "post_attention_layernorm": 0.31, "q_norm": 0.49, "k_norm": 0.49,
          "model.norm": 1.63}
MTP_NORM_W = {"input_layernorm": -0.1, "post_attention_layernorm": 0.87, "q_norm": 0.77, "k_norm": 0.74, "norm": 1.93,
              "pre_fc_norm_embedding": -0.73, "pre_fc_norm_hidden": -0.51}


@dataclass
class QW:
    """A quantized matrix in MLX's layout as stored (int32 words, bf16 scales and biases); leading dims for experts."""

    words: torch.Tensor       # [..., N, K*bits/32] int32
    scales: torch.Tensor      # [..., N, K/group] bf16
    biases: torch.Tensor
    bits: int = 4
    group: int = 64

    @property
    def n(self) -> int:
        return int(self.words.shape[-2])

    @property
    def k(self) -> int:
        return int(self.words.shape[-1]) * (32 // self.bits)

    def rows(self, lo: int, hi: int) -> "QW":
        """Output rows [lo, hi) as views (contiguous when the matrix is)."""

        return QW(self.words[..., lo:hi, :], self.scales[..., lo:hi, :], self.biases[..., lo:hi, :], self.bits,
                  self.group)

    def split(self, sizes: tuple[int, ...]) -> list["QW"]:
        if sum(sizes) != self.n:
            raise ValueError(f"row split {sizes} does not cover {self.n} rows")
        starts = [sum(sizes[:i]) for i in range(len(sizes))]
        return [self.rows(s, s + n) for s, n in zip(starts, sizes)]

    def triple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.words, self.scales, self.biases

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.triple())


@dataclass
class GDNW:
    proj: QW                  # [qkv | z | b | a] x hidden (Config.gdn_rows)
    conv: torch.Tensor        # [conv_dim, taps] bf16 (depthwise, no bias)
    a_log: torch.Tensor       # [nv] fp32
    dt_bias: torch.Tensor     # [nv] fp32
    norm: torch.Tensor        # [dv] bf16: the gated RMSNorm's weight, not centred, used as stored
    out: QW                   # [hidden, nv*dv]


@dataclass
class AttnW:
    proj: QW                  # [q|gate per head | k | v] x hidden (Config.attn_rows)
    q_scale: torch.Tensor     # [head_dim] fp32 (1 + w)
    k_scale: torch.Tensor
    o: QW                     # [hidden, heads*head_dim]


@dataclass
class MoEW:
    router: torch.Tensor      # [E + 1, hidden] fp32: router rows, then the shared expert's gate row
    gate: QW                  # [E + 1, width, hidden]: the shared expert is expert E
    up: QW                    # [E + 1, width, hidden]
    down: QW                  # [E + 1, hidden, width]


@dataclass
class LayerW:
    index: int
    linear: bool
    input_scale: torch.Tensor     # [hidden] fp32 (1 + w)
    post_scale: torch.Tensor      # [hidden] fp32 (1 + w)
    gdn: GDNW | None
    attn: AttnW | None
    moe: MoEW


def _nbytes(x: Any) -> int:
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, QW):
        return x.nbytes()
    if hasattr(x, "__dataclass_fields__") and not isinstance(x, Config):
        return sum(_nbytes(getattr(x, f)) for f in x.__dataclass_fields__)
    if isinstance(x, (list, tuple)):
        return sum(_nbytes(y) for y in x)
    return 0


@dataclass
class Weights:
    cfg: Config
    embed: QW                 # [vocab, hidden] (row lookup)
    layers: list[LayerW]
    norm: torch.Tensor        # [hidden] fp32 (1 + w): the final norm
    head: QW                  # [vocab, hidden]
    inv_freq: torch.Tensor    # [rotary_dim / 2] fp32
    around_one: bool = True   # the checkpoint stored 1 + w (MLX)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def device(self) -> torch.device:
        return self.inv_freq.device

    def nbytes(self) -> int:
        return _nbytes([self.embed, self.layers, self.norm, self.head])


@dataclass
class MTPW:
    """The MTP drafter: fc over [normed embedding | normed hidden], one attention layer and MoE, then ``norm``."""

    cfg: Config
    norm_e: torch.Tensor      # [hidden] fp32 (1 + w)
    norm_h: torch.Tensor      # [hidden] fp32 (1 + w)
    fc: QW                    # [hidden, 2*hidden]: input [normed embedding | normed hidden]
    layer: LayerW             # attention + MoE (257-expert table, fp32 router)
    norm: torch.Tensor        # [hidden] fp32 (1 + w)
    around_one: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def nbytes(self) -> int:
        return _nbytes([self.norm_e, self.norm_h, self.fc, self.layer, self.norm])


def norm_means(rd: Reader, reference: dict[str, float]) -> dict[str, float]:
    """Each centred-norm group's mean stored weight, over every norm of the group the reader's spec names."""

    means = {}
    for group in reference:
        names = [n for n in rd.spec if n == group + ".weight" or n.endswith("." + group + ".weight")]
        if names:
            means[group] = sum(float(rd.get(n).float().mean()) for n in names) / len(names)
    return means


def norms_around_one(means: dict[str, float], reference: dict[str, float]) -> bool:
    """Whether the centred norms store 1 + w (each group's mean nearer w + 1 than w); a mix of the two is refused."""

    plus = {group: mean > reference[group] + 0.5 for group, mean in means.items()}
    if len(set(plus.values())) != 1:
        found = ", ".join(f"{g} {'1 + w' if p else 'w'} (mean {means[g]:.3f})" for g, p in sorted(plus.items()))
        raise ValueError(f"the centred norm weights mix w and 1 + w: {found}")
    return next(iter(plus.values()))


def dequantize(q: QW) -> torch.Tensor:
    """MLX affine values [..., N, K] fp32: fp32(scale) * q + fp32(bias), elementwise (no reduction)."""

    per = 32 // q.bits
    words = q.words.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(per, dtype=torch.int64, device=words.device) * q.bits
    vals = ((words[..., None] >> shifts) & ((1 << q.bits) - 1)).reshape(*words.shape[:-1], words.shape[-1] * per)
    scales = q.scales.to(torch.float32).repeat_interleave(q.group, dim=-1)
    biases = q.biases.to(torch.float32).repeat_interleave(q.group, dim=-1)
    return vals.to(torch.float32) * scales + biases


def stack_rows(parts: list[QW]) -> QW:
    """Matrices with the same K and quantization, stacked by rows in order."""

    if len({(p.k, p.bits, p.group) for p in parts}) != 1:
        raise ValueError(f"cannot stack {[(p.n, p.k, p.bits, p.group) for p in parts]}: K and quantization differ")
    return QW(torch.cat([p.words for p in parts], dim=-2), torch.cat([p.scales for p in parts], dim=-2),
              torch.cat([p.biases for p in parts], dim=-2), parts[0].bits, parts[0].group)


class _Builder:
    """Layers from a reader: shared by the main checkpoint and the drafter."""

    def __init__(self, rd: Reader, cfg: Config, around_one: bool) -> None:
        self.rd, self.cfg, self.around_one = rd, cfg, around_one
        self.device = rd.device

    def qw(self, name: str) -> QW:
        bits, group = self.cfg.quant_of(name)
        return QW(self.rd.get(name + ".weight"), self.rd.get(name + ".scales"), self.rd.get(name + ".biases"),
                  bits, group)

    def scale(self, name: str) -> torch.Tensor:
        """A centred norm's multiplier 1 + w, fp32."""

        w = self.rd.get(name).to(torch.float32)
        return (w if self.around_one else w + 1.0).contiguous()

    def table(self, routed: str, shared: str) -> QW:
        """The routed experts [E, N, W] with the shared expert [N, W] appended as expert E, read in place."""

        if self.cfg.quant_of(routed) != self.cfg.quant_of(shared):
            raise ValueError(f"{routed} and {shared} are quantized differently: they cannot share one table")
        parts = []
        for suffix in ("weight", "scales", "biases"):
            dtype, (e, n, k) = self.rd.spec[f"{routed}.{suffix}"]
            if self.rd.spec[f"{shared}.{suffix}"] != (dtype, (n, k)):
                raise ValueError(f"{shared}.{suffix} is {self.rd.spec[f'{shared}.{suffix}']}: not one expert of "
                                 f"{routed} ({dtype} {[n, k]})")
            out = torch.empty((e + 1, n, k), dtype=DTYPES[dtype], device=self.device)
            self.rd.get(f"{routed}.{suffix}", out=out[:e])
            self.rd.get(f"{shared}.{suffix}", out=out[e])
            parts.append(out)
        return QW(*parts, *self.cfg.quant_of(routed))

    def moe(self, m: str) -> MoEW:
        e, d = self.cfg.experts, self.cfg.hidden
        router = torch.empty((e + 1, d), dtype=torch.float32, device=self.device)
        router[:e] = dequantize(self.qw(m + ".gate"))
        router[e:] = dequantize(self.qw(m + ".shared_expert_gate"))
        return MoEW(router, self.table(m + ".switch_mlp.gate_proj", m + ".shared_expert.gate_proj"),
                    self.table(m + ".switch_mlp.up_proj", m + ".shared_expert.up_proj"),
                    self.table(m + ".switch_mlp.down_proj", m + ".shared_expert.down_proj"))

    def gdn(self, a: str) -> GDNW:
        cfg = self.cfg
        proj = stack_rows([self.qw(a + ".in_proj_qkv"), self.qw(a + ".in_proj_z"), self.qw(a + ".in_proj_b"),
                           self.qw(a + ".in_proj_a")])
        conv = self.rd.get(a + ".conv1d.weight").reshape(cfg.conv_dim, cfg.conv_kernel).contiguous()
        return GDNW(proj, conv, self.rd.get(a + ".A_log").to(torch.float32),
                    self.rd.get(a + ".dt_bias").to(torch.float32), self.rd.get(a + ".norm.weight"),
                    self.qw(a + ".out_proj"))

    def attention(self, a: str) -> AttnW:
        proj = stack_rows([self.qw(a + ".q_proj"), self.qw(a + ".k_proj"), self.qw(a + ".v_proj")])
        return AttnW(proj, self.scale(a + ".q_norm.weight"), self.scale(a + ".k_norm.weight"), self.qw(a + ".o_proj"))

    def layer(self, index: int, base: str, kind: str) -> LayerW:
        linear = kind == "linear"
        return LayerW(index, linear, self.scale(base + ".input_layernorm.weight"),
                      self.scale(base + ".post_attention_layernorm.weight"),
                      self.gdn(base + ".linear_attn") if linear else None,
                      None if linear else self.attention(base + ".self_attn"), self.moe(base + ".mlp"))


def _prefix(found: Spec) -> str:
    return "language_model." if "language_model.model.embed_tokens.weight" in found else ""


def _settle(rd: Reader) -> None:
    rd.release()
    if rd.device.type == "cuda":
        torch.cuda.empty_cache()


def load(model_dir: str | Path, device: str | torch.device = "cuda") -> Weights:
    """The main checkpoint on ``device``; refuses one whose tensors differ from ``layout`` (the vision tower aside)."""

    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    found = checkpoint_tensors(model_dir)
    prefix = _prefix(found)
    spec = layout(cfg, prefix)
    problems = layout_problems(found, spec)
    if problems:
        raise ValueError(f"{model_dir} does not have the layout the loader reads ({len(problems)} differences): "
                         + "; ".join(problems[:6]))
    t0 = time.time()
    rd = Reader(model_dir, device, spec)
    build = _Builder(rd, cfg, norms_around_one(norm_means(rd, NORM_W), NORM_W))
    embed = build.qw(prefix + "model.embed_tokens")
    layers = []
    for i, kind in enumerate(cfg.layer_types):
        layers.append(build.layer(i, f"{prefix}model.layers.{i}", kind))
        _settle(rd)
    norm = build.scale(prefix + "model.norm.weight")
    head = embed if cfg.tie_embeddings else build.qw(prefix + "lm_head")
    half = cfg.rotary_dim // 2
    inv = torch.tensor(cfg.rope_theta, dtype=torch.float64) ** (-torch.arange(half, dtype=torch.float64) / half)
    if rd.read != set(spec):
        raise RuntimeError(f"loader left {sorted(set(spec) - rd.read)[:4]} unread")
    _settle(rd)
    skipped = sorted(name for name in found if name not in spec)
    return Weights(cfg, embed, layers, norm, head, inv.to(torch.float32).to(rd.device), build.around_one,
                   meta={"prefix": prefix, "tensors_read": len(rd.read), "skipped": len(skipped),
                         "load_seconds": time.time() - t0})


def load_mtp(drafter_dir: str | Path, target: Config, device: str | torch.device = "cuda") -> MTPW:
    """The MTP drafter on ``device``: its config must match the target's shapes and its tensors ``mtp_layout``."""

    drafter_dir = Path(drafter_dir)
    kind = json.loads((drafter_dir / "config.json").read_text()).get("model_type")
    if kind != DRAFTER_MODEL_TYPE:
        raise ValueError(f"{drafter_dir} has model_type {kind!r}, not the MTP drafter's {DRAFTER_MODEL_TYPE!r} "
                         f"({DRAFTER})")
    cfg = Config.read(drafter_dir)
    fields = ("hidden", "vocab", "heads", "kv_heads", "head_dim", "rotary_dim", "experts", "top_k", "moe_width",
              "shared_width")
    wrong = [f"{f} {getattr(cfg, f)} (target {getattr(target, f)})" for f in fields
             if getattr(cfg, f) != getattr(target, f)]
    if wrong:
        raise ValueError(f"{drafter_dir} does not match the target model: {', '.join(wrong)}")
    found = checkpoint_tensors(drafter_dir)
    spec = mtp_layout(cfg)
    problems = layout_problems(found, spec, skip=())
    if problems:
        raise ValueError(f"{drafter_dir} does not have the drafter layout the loader reads: " + "; ".join(problems[:6]))
    rd = Reader(drafter_dir, device, spec)
    build = _Builder(rd, cfg, norms_around_one(norm_means(rd, MTP_NORM_W), MTP_NORM_W))
    mtp = MTPW(cfg, build.scale("pre_fc_norm_embedding.weight"), build.scale("pre_fc_norm_hidden.weight"),
               build.qw("fc"), build.layer(0, "layers.0", "attention"), build.scale("norm.weight"), build.around_one)
    if rd.read != set(spec):
        raise RuntimeError(f"drafter loader left {sorted(set(spec) - rd.read)[:4]} unread")
    _settle(rd)
    raw = json.loads((drafter_dir / "config.json").read_text())
    mtp.meta.update(tensors_read=len(rd.read), block_size=raw.get("block_size"))
    return mtp
