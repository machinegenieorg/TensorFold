"""Qwen3.6-35B-A3B's checkpoint: its config, the tensors the loader reads (names, dtypes, shapes) and the reader."""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .. import eos_ids

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
        _, header = read_header(model_dir / shard)
        found.update({k: (v["dtype"], tuple(v["shape"])) for k, v in header.items() if k != "__metadata__"})
    return found


def layout_problems(found: Spec, spec: Spec, skip: tuple[str, ...] = VISION_PREFIXES) -> list[str]:
    """What keeps ``spec`` from reading a checkpoint: missing, mismatched or unread tensors (``skip`` aside)."""

    problems = [f"missing {name}" for name in spec if name not in found]
    problems += [f"{name} is {found[name][0]} {list(found[name][1])}, expected {want[0]} {list(want[1])}"
                 for name, want in spec.items() if name in found and found[name] != want]
    problems += [f"unexpected {name}" for name in found if name not in spec and not name.startswith(skip)]
    return problems


def read_header(path: Path) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


DTYPES = {"U32": torch.int32, "I32": torch.int32, "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


class Reader:
    """Tensors by name, checked against ``spec`` and read sequentially in large blocks (not mmap page faults)."""

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
            got = self.headers[shard] = read_header(self.dir / shard)
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
        host = raw.view(DTYPES[entry["dtype"]]).reshape(entry["shape"])
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
