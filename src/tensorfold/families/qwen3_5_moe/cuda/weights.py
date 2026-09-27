"""Qwen3.6-35B-A3B weights from the MLX 4-bit checkpoint (affine, groups of 64), and the MTP drafter's.

The loader passes MLX's packing through unchanged; the kernels regroup what they need when they wrap these
tensors. The layout contract with the kernels:

- A projection is a ``QW``: MLX's arrays as stored. ``words`` [..., N, K*bits/32] int32 (the checkpoint's uint32
  bits: torch's uint32 supports few ops), ``scales`` and ``biases`` bf16 [..., N, K/64]. Input k of row n is the
  ``bits``-wide field ``k % (32/bits)`` of word ``k // (32/bits)``, lowest bits first, and its value is
  scale * q + bias with group ``k // 64``'s scale and bias. ``QW.triple()`` is what ``qwen3_5_moe.cuda.qmm``'s
  ``make_q4``, ``make_experts`` and ``embed`` take (``forward.prepare`` regroups the model through them).
- Projections that read the same input are stacked by rows, packing unchanged: Gated DeltaNet
  [in_proj_qkv | in_proj_z | in_proj_b | in_proj_a] and attention [q_proj | k_proj | v_proj] (row counts in
  ``Config.gdn_rows`` and ``Config.attn_rows``; ``QW.split`` gives each part as a view). in_proj_qkv's rows are
  q (16 key heads x 128), k (16 x 128), v (32 value heads x 128), the conv's channel order; q_proj's are per head,
  head h's query (256 rows) then its output gate (256 rows).
- The 256 routed experts and the shared expert are one table per layer: ``gate`` and ``up`` [257, 512, 2048/8
  words], ``down`` [257, 2048, 512/8 words]; the shared expert is expert 256.
- The router is fp32 [257, 2048]: the 256 router rows (``mlp.gate``) then the shared expert's gate row
  (``mlp.shared_expert_gate``), 8-bit in the checkpoint (4-bit in the drafter), dequantized once at load as
  fp32(scale) * q + fp32(bias). The product is exact in fp32, so every device gets the same single rounding.
- Centred norms (input, post-attention, final, attention q/k, and the drafter's) compute x̂ · (1 + w); each is
  returned as that fp32 multiplier. MLX's converter stores 1 + w rounded to bf16 (checked at load from the input
  norms' means, ``around_one``); a checkpoint that stores w gets 1 added in fp32. The Gated DeltaNet output norm
  is not centred: its weight is bf16 as stored (used as w · silu(z)).
- Embedding and head: ``QW`` [248320, 2048/8 words] as stored (the model does not tie them).

The vision tower (``vision_tower.*``) is skipped. The MTP drafter is a separate repository
(mlx-community/Qwen3.6-35B-A3B-MTP-4bit, model_type ``qwen3_5_mtp``, bare names ``fc``, ``pre_fc_norm_embedding``,
``pre_fc_norm_hidden``, ``layers.0.*``, ``norm``): ``load_mtp`` reads it into ``MTPW``. It has no embedding or
head of its own: it uses the target's.
"""

from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .. import DRAFTER, DRAFTER_MODEL_TYPE, eos_ids

VISION_PREFIXES = ("vision_tower.", "model.visual.", "visual.")
# a tensor's dtype and shape as the safetensors header gives them
Spec = dict[str, tuple[str, tuple[int, ...]]]


@dataclass
class Config:
    hidden: int
    layers: int
    layer_types: list[str]              # per layer: "linear" (Gated DeltaNet) or "attention"
    vocab: int
    eps: float
    heads: int
    kv_heads: int
    head_dim: int
    attn_gate: bool                     # q_proj also emits a sigmoid output gate per head
    rope_theta: float
    partial_rotary: float
    rotary_dim: int                     # leading dims of each head that rotate (rotate-half)
    mrope_section: tuple[int, ...]      # text positions are equal on all three axes: plain 1-D RoPE
    nk: int
    nv: int
    dk: int
    dv: int
    conv_kernel: int
    experts: int
    top_k: int
    moe_width: int
    shared_width: int
    norm_topk: bool                     # renormalise the top-k router weights
    tie_embeddings: bool
    mtp_layers: int
    max_position: int
    eos: tuple[int, ...]
    bits: int
    group_size: int
    quant_overrides: dict[str, tuple[int, int]] = field(default_factory=dict)   # module -> (bits, group size)

    @classmethod
    def read(cls, model_dir: str | Path) -> "Config":
        """config.json (its text_config when present) and generation_config.json (the eos ids)."""

        raw = json.loads((Path(model_dir) / "config.json").read_text())
        t = dict(raw.get("text_config") or raw)
        rope = dict(t.get("rope_parameters") or t.get("rope_scaling") or {})
        head_dim = int(t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"])
        partial = float(rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)))
        layers = int(t["num_hidden_layers"])
        kinds = t.get("layer_types")
        if kinds is None:
            every = int(t.get("full_attention_interval", 4))
            kinds = ["full_attention" if (i + 1) % every == 0 else "linear_attention" for i in range(layers)]
        if len(kinds) != layers or set(kinds) - {"linear_attention", "full_attention"}:
            raise ValueError(f"layer_types must name {layers} linear_attention/full_attention layers, got {kinds}")
        quant = next((q for q in (raw.get("quantization"), raw.get("quantization_config"), t.get("quantization"),
                                  t.get("quantization_config")) if isinstance(q, dict) and q), {})
        bits, group = int(quant.get("bits", 4)), int(quant.get("group_size", 64))
        overrides = {k: (int(v.get("bits", bits)), int(v.get("group_size", group)))
                     for k, v in quant.items() if isinstance(v, dict)}
        return cls(
            hidden=int(t["hidden_size"]), layers=layers,
            layer_types=["linear" if k == "linear_attention" else "attention" for k in kinds],
            vocab=int(t["vocab_size"]), eps=float(t.get("rms_norm_eps", 1e-6)), heads=int(t["num_attention_heads"]),
            kv_heads=int(t["num_key_value_heads"]), head_dim=head_dim, attn_gate=bool(t.get("attn_output_gate", True)),
            rope_theta=float(rope.get("rope_theta", t.get("rope_theta", 10_000_000))), partial_rotary=partial,
            rotary_dim=int(head_dim * partial), mrope_section=tuple(int(s) for s in rope.get("mrope_section") or ()),
            nk=int(t["linear_num_key_heads"]), nv=int(t["linear_num_value_heads"]),
            dk=int(t["linear_key_head_dim"]), dv=int(t["linear_value_head_dim"]),
            conv_kernel=int(t["linear_conv_kernel_dim"]), experts=int(t["num_experts"]),
            top_k=int(t["num_experts_per_tok"]), moe_width=int(t["moe_intermediate_size"]),
            shared_width=int(t["shared_expert_intermediate_size"]), norm_topk=bool(t.get("norm_topk_prob", True)),
            tie_embeddings=bool(raw.get("tie_word_embeddings", t.get("tie_word_embeddings", False))),
            mtp_layers=int(t.get("mtp_num_hidden_layers", 0)),
            max_position=int(t.get("max_position_embeddings", 262144)), eos=eos_ids(model_dir),
            bits=bits, group_size=group, quant_overrides=overrides,
        )

    def quant_of(self, module: str) -> tuple[int, int]:
        """(bits, group size) of a module, by its full checkpoint name (without .weight)."""

        return self.quant_overrides.get(module, (self.bits, self.group_size))

    @property
    def conv_dim(self) -> int:
        return 2 * self.nk * self.dk + self.nv * self.dv

    @property
    def gdn_rows(self) -> tuple[int, int, int, int]:
        """Rows of the stacked Gated DeltaNet projection: qkv, z, b, a."""

        return self.conv_dim, self.nv * self.dv, self.nv, self.nv

    @property
    def attn_rows(self) -> tuple[int, int, int]:
        """Rows of the stacked attention projection: q (with its gate), k, v."""

        return self.heads * self.head_dim * (2 if self.attn_gate else 1), self.kv_heads * self.head_dim, \
            self.kv_heads * self.head_dim

    @property
    def attention_layers(self) -> list[int]:
        return [i for i, k in enumerate(self.layer_types) if k == "attention"]


@dataclass
class QW:
    """A quantized matrix in MLX's layout (see the module docstring); leading dims for stacked experts."""

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
    """The MTP drafter: x = fc([pre_fc_norm_embedding(embed(t+1)) | pre_fc_norm_hidden(h)]) with h the target's
    hidden state after its final norm, then one full-attention layer (own KV cache) and ``norm``, then the target's
    head."""

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


# ---------------------------------------------------------------------------------------------------------------
# what the loader reads: every tensor's name, dtype and shape


def _quant_spec(spec: Spec, cfg: Config, name: str, n: int, k: int, lead: tuple[int, ...] = ()) -> None:
    bits, group = cfg.quant_of(name)
    spec[name + ".weight"] = ("U32", (*lead, n, k * bits // 32))
    spec[name + ".scales"] = ("BF16", (*lead, n, k // group))
    spec[name + ".biases"] = ("BF16", (*lead, n, k // group))


def _layer_spec(spec: Spec, cfg: Config, base: str, kind: str) -> None:
    d = cfg.hidden
    spec[base + ".input_layernorm.weight"] = ("BF16", (d,))
    spec[base + ".post_attention_layernorm.weight"] = ("BF16", (d,))
    if kind == "linear":
        a = base + ".linear_attn"
        qkv, z, b, g = cfg.gdn_rows
        _quant_spec(spec, cfg, a + ".in_proj_qkv", qkv, d)
        _quant_spec(spec, cfg, a + ".in_proj_z", z, d)
        _quant_spec(spec, cfg, a + ".in_proj_b", b, d)
        _quant_spec(spec, cfg, a + ".in_proj_a", g, d)
        spec[a + ".conv1d.weight"] = ("BF16", (cfg.conv_dim, cfg.conv_kernel, 1))
        spec[a + ".A_log"] = ("BF16", (cfg.nv,))
        spec[a + ".dt_bias"] = ("BF16", (cfg.nv,))
        spec[a + ".norm.weight"] = ("BF16", (cfg.dv,))
        _quant_spec(spec, cfg, a + ".out_proj", d, cfg.nv * cfg.dv)
    else:
        a = base + ".self_attn"
        q, k, v = cfg.attn_rows
        _quant_spec(spec, cfg, a + ".q_proj", q, d)
        _quant_spec(spec, cfg, a + ".k_proj", k, d)
        _quant_spec(spec, cfg, a + ".v_proj", v, d)
        spec[a + ".q_norm.weight"] = ("BF16", (cfg.head_dim,))
        spec[a + ".k_norm.weight"] = ("BF16", (cfg.head_dim,))
        _quant_spec(spec, cfg, a + ".o_proj", d, cfg.heads * cfg.head_dim)
    m = base + ".mlp"
    _quant_spec(spec, cfg, m + ".gate", cfg.experts, d)
    _quant_spec(spec, cfg, m + ".shared_expert_gate", 1, d)
    for proj, n, k in (("gate_proj", cfg.moe_width, d), ("up_proj", cfg.moe_width, d), ("down_proj", d, cfg.moe_width)):
        _quant_spec(spec, cfg, f"{m}.switch_mlp.{proj}", n, k, (cfg.experts,))
    for proj, n, k in (("gate_proj", cfg.shared_width, d), ("up_proj", cfg.shared_width, d),
                       ("down_proj", d, cfg.shared_width)):
        _quant_spec(spec, cfg, f"{m}.shared_expert.{proj}", n, k)


def layout(cfg: Config, prefix: str = "language_model.") -> Spec:
    """Every tensor ``load`` reads from the main checkpoint, with its dtype and shape."""

    spec: Spec = {}
    _quant_spec(spec, cfg, prefix + "model.embed_tokens", cfg.vocab, cfg.hidden)
    for i, kind in enumerate(cfg.layer_types):
        _layer_spec(spec, cfg, f"{prefix}model.layers.{i}", kind)
    spec[prefix + "model.norm.weight"] = ("BF16", (cfg.hidden,))
    if not cfg.tie_embeddings:
        _quant_spec(spec, cfg, prefix + "lm_head", cfg.vocab, cfg.hidden)
    return spec


def mtp_layout(cfg: Config) -> Spec:
    """Every tensor ``load_mtp`` reads from the drafter, with its dtype and shape (``cfg``: the drafter's)."""

    spec: Spec = {}
    _quant_spec(spec, cfg, "fc", cfg.hidden, 2 * cfg.hidden)
    spec["pre_fc_norm_embedding.weight"] = ("BF16", (cfg.hidden,))
    spec["pre_fc_norm_hidden.weight"] = ("BF16", (cfg.hidden,))
    _layer_spec(spec, cfg, "layers.0", "attention")
    spec["norm.weight"] = ("BF16", (cfg.hidden,))
    return spec


def checkpoint_tensors(model_dir: str | Path) -> Spec:
    """Every tensor in a checkpoint with its dtype and shape, from the safetensors headers (no weights read)."""

    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file() \
        else ["model.safetensors"]
    found: Spec = {}
    for shard in shards:
        _, header = _read_header(model_dir / shard)
        found.update({k: (v["dtype"], tuple(v["shape"])) for k, v in header.items() if k != "__metadata__"})
    return found


def layout_problems(found: Spec, spec: Spec, skip: tuple[str, ...] = VISION_PREFIXES) -> list[str]:
    """What keeps ``spec`` from reading this checkpoint: missing tensors, other dtypes or shapes, and tensors the
    loader would leave unread (other than those under ``skip``)."""

    problems = [f"missing {name}" for name in spec if name not in found]
    problems += [f"{name} is {found[name][0]} {list(found[name][1])}, expected {want[0]} {list(want[1])}"
                 for name, want in spec.items() if name in found and found[name] != want]
    problems += [f"unexpected {name}" for name in found if name not in spec and not name.startswith(skip)]
    return problems


# ---------------------------------------------------------------------------------------------------------------
# reading


def _read_header(path: Path) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


_DT = {"U32": torch.int32, "I32": torch.int32, "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


class _Reader:
    """Tensors by name from a checkpoint's shards, each checked against the loader's ``spec``, read with large
    sequential reads (mmap page faults stream slowly from NVMe once the page cache is dropped). ``read`` records
    the names read; ``release`` drops the read shards' cached pages."""

    def __init__(self, model_dir: Path, device: str | torch.device, spec: Spec) -> None:
        self.dir = Path(model_dir)
        self.device = torch.device(device)
        self.spec = spec
        self.headers: dict[str, tuple[int, dict]] = {}
        self.read: set[str] = set()
        self.touched: set[str] = set()
        index = self.dir / "model.safetensors.index.json"
        if index.is_file():
            self.where = json.loads(index.read_text())["weight_map"]
        else:
            self.where = {k: "model.safetensors" for k in self._header("model.safetensors")[1] if k != "__metadata__"}

    def _header(self, shard: str) -> tuple[int, dict]:
        got = self.headers.get(shard)
        if got is None:
            got = self.headers[shard] = _read_header(self.dir / shard)
        return got

    def get(self, name: str, out: torch.Tensor | None = None) -> torch.Tensor:
        """The tensor (on the reader's device), or copied into ``out`` (which is returned)."""

        want = self.spec.get(name)
        if want is None:
            raise KeyError(f"{name} is not in the loader's layout")
        shard = self.where[name]
        base, header = self._header(shard)
        entry = header[name]
        if (entry["dtype"], tuple(entry["shape"])) != want:
            raise ValueError(f"{name} is {entry['dtype']} {entry['shape']}, expected {want[0]} {list(want[1])}")
        begin, end = entry["data_offsets"]
        raw = torch.empty((end - begin,), dtype=torch.uint8)
        view = memoryview(raw.numpy())
        with open(self.dir / shard, "rb", buffering=0) as f:
            f.seek(base + begin)
            at = 0
            while at < len(view):
                got = f.readinto(view[at:at + (64 << 20)])
                if not got:
                    raise IOError(f"short read of {name}")
                at += got
        self.touched.add(shard)
        self.read.add(name)
        host = raw.view(_DT[entry["dtype"]]).reshape(entry["shape"])
        if out is None:
            return host.to(self.device)
        out.copy_(host)
        return out

    def release(self) -> None:
        """Drop the read shards' cached pages: on unified memory they would sit beside the same bytes on the GPU."""

        for shard in list(self.touched):
            try:
                fd = os.open(self.dir / shard, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except (OSError, AttributeError):
                pass
        self.touched.clear()


def norms_around_one(means: list[float]) -> bool:
    """Whether a checkpoint stores centred norms as 1 + w (MLX), from the input norms' means: 1 + w sits near 1
    (0.74 to 1.08 in the MLX checkpoint), w near 0 (-0.27 to 0.08 in the original)."""

    if not means:
        raise ValueError("no norm weights to tell how they are stored")
    if all(0.5 < m < 2.0 for m in means):
        return True
    if all(-0.5 < m < 0.5 for m in means):
        return False
    raise ValueError(f"cannot tell how the norm weights are stored (input norm means {min(means):.3f} to "
                     f"{max(means):.3f})")


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

    def __init__(self, rd: _Reader, cfg: Config, around_one: bool) -> None:
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
            out = torch.empty((e + 1, n, k), dtype=_DT[dtype], device=self.device)
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


def _settle(rd: _Reader) -> None:
    rd.release()
    if rd.device.type == "cuda":
        torch.cuda.empty_cache()


def load(model_dir: str | Path, device: str | torch.device = "cuda") -> Weights:
    """The main checkpoint on ``device`` in the layout the module docstring states. Refuses a checkpoint whose
    tensors differ from ``layout`` (missing, other dtype or shape, or unexpected, the vision tower aside)."""

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
    rd = _Reader(model_dir, device, spec)
    means = [float(rd.get(f"{prefix}model.layers.{i}.input_layernorm.weight").float().mean())
             for i in range(cfg.layers)]
    build = _Builder(rd, cfg, norms_around_one(means))
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
    """The MTP drafter (``DRAFTER``) on ``device``: its config must match the target's shapes and its tensors
    ``mtp_layout`` exactly."""

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
    rd = _Reader(drafter_dir, device, spec)
    build = _Builder(rd, cfg, norms_around_one([float(rd.get("layers.0.input_layernorm.weight").float().mean())]))
    mtp = MTPW(cfg, build.scale("pre_fc_norm_embedding.weight"), build.scale("pre_fc_norm_hidden.weight"),
               build.qw("fc"), build.layer(0, "layers.0", "attention"), build.scale("norm.weight"), build.around_one)
    if rd.read != set(spec):
        raise RuntimeError(f"drafter loader left {sorted(set(spec) - rd.read)[:4]} unread")
    _settle(rd)
    raw = json.loads((drafter_dir / "config.json").read_text())
    mtp.meta.update(tensors_read=len(rd.read), block_size=raw.get("block_size"))
    return mtp
