"""Qwen3.6 MoE's NVFP4 route: ``nvidia/Qwen3.6-35B-A3B-NVFP4`` (ModelOpt, ``MIXED_PRECISION``) read as it ships.

The checkpoint names each quantized layer's algorithm in ``quantized_layers``, and the loader follows the stored
tensors:

* ``W4A16_NVFP4`` (blocks of 16): the routed experts, the shared expert and ``lm_head``. Each is packed E2M1
  nibbles, an e4m3 scale a block and an fp32 scale a tensor, kept as stored and decoded by Flash Next's NVFP4
  kernels (``qwen4_exp/cuda/nvfp4*.py``): the experts through ``nvfp4_moe`` on the plan of
  ``tensorfold.cuda.experts``, the shared expert and the head through ``nvfp4.matmul``. Activations stay bf16.
* ``FP8``: DeltaNet's ``in_proj_qkv``, ``in_proj_z`` and ``out_proj`` and attention's four projections: e4m3
  codes and one fp32 scale a tensor (``fp8.py``, weight-only; ``input_scale`` is vLLM's activation scale and is
  not read).
* bf16 as stored: the embedding, the routers and shared-expert gates, ``in_proj_a`` and ``in_proj_b``, the
  convolution, the norms and the MTP layer (its projections through ``bf16.matmul``; its stacked experts ride the
  NVFP4 tables as exact identity-scaled bf16, as Flash Next's do).

The checkpoint keeps the RMSNorm weights zero-centred (the model scales by ``1 + w``); they are widened to fp32
as ``1 + w`` once, at load, so the norms compute what the model does. DeltaNet's gated norm is stored absolute.
The vision tower and the ``input_scale`` tensors are not read.
"""

from __future__ import annotations

import json
import re
from contextlib import ExitStack
from pathlib import Path
from typing import Callable

import torch

from tensorfold.cuda.moe import Routed
from tensorfold.families.qwen3_5.cuda.weights import GDN, Attention, Config, Layer, Plain, Weights
from tensorfold.families.qwen4_exp.cuda import bf16, nvfp4
from tensorfold.families.qwen4_exp.cuda.nvfp4_moe import Expert4, MoE4, expert4_from_bf16, moe4_from_bf16

from .fp8 import make_fp8

# RMSNorm weights the model applies as ``1 + w`` (Qwen3.5's zero-centred norm); DeltaNet's gated norm is absolute
CENTRED = (".input_layernorm.weight", ".post_attention_layernorm.weight", ".q_norm.weight", ".k_norm.weight",
           "model.norm.weight", "mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight",
           "mtp.pre_fc_norm_hidden.weight")
SKIPPED = re.compile(r"(^|\.)(visual|vision_tower)\.|\.input_scale$")


def quant_block(config: dict) -> dict:
    return config.get("quantization_config") or config.get("quantization") or {}


def is_modelopt(model_dir: str | Path) -> bool:
    config = json.loads((Path(model_dir) / "config.json").read_text())
    return str(quant_block(config).get("quant_method") or "").lower() == "modelopt"


class Dense:
    """A bf16 projection as stored, through Flash Next's row-invariant ``bf16.matmul``."""

    layout = "b16"
    fast = False

    def __init__(self, weight: torch.Tensor) -> None:
        self.b = bf16.make_b16(weight)

    @property
    def n(self) -> int:
        return self.b.n

    @property
    def k(self) -> int:
        return self.b.k

    def nbytes(self) -> int:
        return self.b.nbytes()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return bf16.matmul(x, self.b)

    prefill = __call__


class FP4Linear:
    """An NVFP4 projection through ``nvfp4.matmul`` on the stored bytes; ``rows`` of the table are real (the rest
    pad the table to whole 64-row tiles and are cut from the output)."""

    layout = "nvfp4"
    fast = False

    def __init__(self, fp: nvfp4.FP4, rows: int | None = None) -> None:
        self.fp, self.rows = fp, int(fp.n if rows is None else rows)

    @property
    def n(self) -> int:
        return self.rows

    @property
    def k(self) -> int:
        return self.fp.k

    def nbytes(self) -> int:
        return self.fp.nbytes()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = nvfp4.matmul(x, self.fp)
        return y if self.rows == self.fp.n else y[:, :self.rows]

    prefill = __call__


def fp4_table(parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | float]]) -> tuple[nvfp4.FP4, int]:
    """Row blocks of NVFP4 arrays ([n, k/2] uint8 words, [n, k/16] e4m3 scales, scalar scale) joined as one table,
    each block's per-tensor scale on its own rows, zero rows padding it to whole tiles: (table, real rows)."""

    words = torch.cat([w for w, _, _ in parts])
    scales = torch.cat([s.contiguous().view(torch.uint8) for _, s, _ in parts])
    factor = torch.cat([torch.full((w.shape[0],), float(s2), dtype=torch.float32, device=w.device)
                        for w, _, s2 in parts])
    rows = words.shape[0]
    pad = -rows % nvfp4.BN
    if pad:
        words = torch.cat([words, words.new_zeros((pad, words.shape[1]))])
        scales = torch.cat([scales, scales.new_zeros((pad, scales.shape[1]))])
        factor = torch.cat([factor, factor.new_zeros(pad)])
    fp = nvfp4.make_fp4(words, scales, 1.0)
    fp.scale2 = factor
    return fp, rows


def fp4_linear(words: torch.Tensor, scales: torch.Tensor, scale2: torch.Tensor | float) -> FP4Linear:
    return FP4Linear(*fp4_table([(words, scales, scale2)]))


def shared_fp4(gate: tuple, up: tuple, down: tuple) -> Expert4:
    """An NVFP4 shared expert: gate and up rows in one table (each with its own per-tensor scale), and down."""

    gu, rows = fp4_table([gate, up])
    dn, drows = fp4_table([down])
    if rows != gu.n or drows != dn.n:
        raise ValueError("the shared expert's widths must be whole 64-row tiles")
    return Expert4(gu, dn)


class PooledMoE4(MoE4):
    """Flash Next's ``MoE4``, with the shared expert's output and split-K buffers drawn from one pool that every
    layer shares, a pair for each power of two of rows (slices of it, contiguous, for the rows of a call).

    ``MoE4`` keeps a pair for each row count, in each layer: sized for Flash Next's fixed buffers. Here the MoE runs
    at every prompt chunk's length and every round's width, and those pairs (about 100 KB a row a layer) grew
    without bound. The pool's buffers are never freed, so a captured graph's addresses stay valid; layers use them
    one after another, each reading its output before the next writes."""

    def shared_out(self, x: torch.Tensor, fp: nvfp4.FP4, *, f32: bool = False):
        rows, sk = int(x.shape[0]), nvfp4.split_for(fp.n, fp.k)
        size = 1 << max(4, (rows - 1).bit_length())
        key = (size, fp.n, sk, f32, x.device)
        got = _POOL.get(key)
        if got is None:
            got = _POOL[key] = (torch.empty(size * fp.n, dtype=torch.float32 if f32 else torch.bfloat16,
                                            device=x.device),
                                torch.empty(sk * size * fp.n, dtype=torch.float32, device=x.device) if sk > 1 else None)
        out, part = got
        return (out[:rows * fp.n].view(rows, fp.n),
                part[:sk * rows * fp.n].view(sk, rows, fp.n) if part is not None else None)


_POOL: dict[tuple, tuple] = {}


def pool_bytes(text: dict, rows: int = 4096) -> int:
    """The pool's most (every power of two of rows up to a prompt chunk's, gate/up and down): for admission."""

    width = int(text.get("shared_expert_intermediate_size") or text["moe_intermediate_size"])
    d = int(text["hidden_size"])
    per_row = 0
    for n, k, item in ((2 * width, d, 2), (d, width, 4)):
        sk = nvfp4.split_for(n, k)
        per_row += n * item + (sk * n * 4 if sk > 1 else 0)
    return per_row * sum(1 << b for b in range(4, rows.bit_length()))


def with_pool(geometry, text: dict):
    """``geometry`` plus the shared-expert pool (``PooledMoE4``)."""

    from tensorfold.cuda.capacity import Geometry

    extra = pool_bytes(text)
    return Geometry(lambda slots: geometry.bytes_at(slots) + extra, geometry.reserve, geometry.minimum_slots)


def pooled(m: MoE4) -> PooledMoE4:
    return PooledMoE4(m.gate_up, m.down_proj, m.shared, m.kernel)


def experts_fp4(gate: tuple, up: tuple, down: tuple, shared: Expert4) -> MoE4:
    """One layer's routed experts from stacked checkpoint arrays (each ([E, n, k/2] uint8, [E, n, k/16] e4m3,
    [E] fp32)): gate and up rows joined per expert (gate first), each half with its expert's own scale."""

    (gw, gs, g2), (uw, us, u2), (dw, ds, d2) = gate, up, down
    e, ni, d = int(gw.shape[0]), int(gw.shape[1]), int(dw.shape[1])
    gu = nvfp4.stacked_fp4(torch.cat([gw, uw], dim=1), torch.cat([gs, us], dim=1),
                           torch.cat([g2[:, None].expand(e, ni), u2[:, None].expand(e, ni)], dim=1))
    return PooledMoE4(gu, nvfp4.stacked_fp4(dw, ds, d2[:, None].expand(e, d)), shared)


def centred(w: torch.Tensor) -> torch.Tensor:
    """A zero-centred RMSNorm weight as the scale the model applies, ``1 + w``, in fp32 (exact)."""

    return (1.0 + w.float()).contiguous()


class _Reader:
    """The checkpoint's tensors on the host, a name at a time (the files stay memory-mapped)."""

    def __init__(self, model_dir: Path) -> None:
        from safetensors import safe_open

        self.files, self.where = ExitStack(), {}
        for path in sorted(model_dir.glob("*.safetensors")):
            f = self.files.enter_context(safe_open(str(path), framework="pt", device="cpu"))
            for name in f.keys():
                self.where[name] = f
        if "model.language_model.embed_tokens.weight" in self.where:       # the published layout
            self.key: Callable[[str], str] = (lambda name: "model.language_model." + name[len("model."):]
                                              if name.startswith("model.") else name)
        elif "language_model.model.embed_tokens.weight" in self.where:
            self.key = lambda name: "language_model." + name if name.startswith("model.") else name
        else:
            self.key = lambda name: name
        self.used: set[str] = set()

    def has(self, name: str) -> bool:
        return self.key(name) in self.where

    def get(self, name: str) -> torch.Tensor:
        stored = self.key(name)
        if stored not in self.where:
            raise ValueError(f"the checkpoint has no tensor {stored}")
        self.used.add(stored)
        return self.where[stored].get_tensor(stored)

    def left(self, mtp: bool) -> list[str]:
        return [n for n in self.where if n not in self.used and not SKIPPED.search(n)
                and (n.startswith("mtp.") or ".mtp." in n) == mtp]

    def close(self) -> None:
        self.files.close()


class _Builder:
    """Projections, norms and routed experts of one checkpoint, each as its stored format asks."""

    def __init__(self, model_dir: Path, device: str, cfg: Config) -> None:
        self.rd, self.device, self.cfg = _Reader(model_dir), device, cfg

    def tensor(self, name: str) -> torch.Tensor:
        return self.rd.get(name).to(self.device)

    def norm(self, name: str) -> torch.Tensor:
        w = self.tensor(name)
        return centred(w) if name.endswith(CENTRED) else w.contiguous()

    def linear(self, name: str):
        """FP8, NVFP4 or bf16, by what the checkpoint stores under ``name``."""

        w = self.rd.get(name + ".weight")
        if self.rd.has(name + ".weight_scale_2"):
            return fp4_linear(w.to(self.device), self.tensor(name + ".weight_scale"),
                              self.rd.get(name + ".weight_scale_2"))
        if w.dtype == torch.float8_e4m3fn:
            return make_fp8(w.view(torch.uint8).to(self.device), self.rd.get(name + ".weight_scale"))
        if w.dtype not in (torch.bfloat16, torch.float16, torch.float32) or w.dim() != 2:
            raise ValueError(f"{name}: a {w.dtype} {tuple(w.shape)} weight is neither FP8, NVFP4 nor bf16")
        return Dense(w.to(self.device))

    def _fp4(self, name: str) -> tuple:
        return (self.tensor(name + ".weight"), self.tensor(name + ".weight_scale"),
                self.rd.get(name + ".weight_scale_2"))

    def _stack(self, prefix: str, proj: str) -> tuple:
        """Every routed expert's arrays for one projection, stacked on the host and moved once."""

        e = self.cfg.experts
        words = torch.stack([self.rd.get(f"{prefix}experts.{i}.{proj}.weight") for i in range(e)])
        scales = torch.stack([self.rd.get(f"{prefix}experts.{i}.{proj}.weight_scale").view(torch.uint8)
                              for i in range(e)])
        s2 = torch.stack([self.rd.get(f"{prefix}experts.{i}.{proj}.weight_scale_2").float().reshape(())
                          for i in range(e)])
        return words.to(self.device), scales.to(self.device), s2.to(self.device)

    def routed(self, prefix: str) -> Routed:
        """A layer's router rows (the shared expert's gate row last) and its experts, the shared one beside them."""

        router = torch.cat([self.tensor(prefix + "gate.weight"), self.tensor(prefix + "shared_expert_gate.weight")])
        sp = prefix + "shared_expert."
        fp4_shared = self.rd.has(sp + "gate_proj.weight_scale_2")

        def shared_bf16() -> tuple[torch.Tensor, ...]:
            return tuple(self.tensor(sp + f"{p}_proj.weight").to(torch.bfloat16) for p in ("gate", "up", "down"))

        if self.rd.has(prefix + "experts.0.gate_proj.weight_scale_2"):
            shared = (shared_fp4(self._fp4(sp + "gate_proj"), self._fp4(sp + "up_proj"), self._fp4(sp + "down_proj"))
                      if fp4_shared else expert4_from_bf16(*shared_bf16()))
            experts = experts_fp4(self._stack(prefix, "gate_proj"), self._stack(prefix, "up_proj"),
                                  self._stack(prefix, "down_proj"), shared)
        elif self.rd.has(prefix + "experts.gate_up_proj") and not fp4_shared:     # stacked bf16 (the MTP layer's)
            experts = pooled(moe4_from_bf16(self.tensor(prefix + "experts.gate_up_proj").to(torch.bfloat16),
                                            self.tensor(prefix + "experts.down_proj").to(torch.bfloat16),
                                            shared_bf16()))
        else:
            raise ValueError(f"{prefix}experts: neither NVFP4 experts nor stacked bf16 ones with a bf16 shared expert")
        return Routed(router.to(torch.bfloat16).contiguous(), experts, int(self.cfg.top_k))

    def attention(self, p: str) -> Attention:
        return Attention(q=self.linear(p + "q_proj"), k=self.linear(p + "k_proj"), v=self.linear(p + "v_proj"),
                         o=self.linear(p + "o_proj"), q_norm=self.norm(p + "q_norm.weight"),
                         k_norm=self.norm(p + "k_norm.weight"))


def load(model_dir: str | Path, device: str = "cuda") -> Weights:
    """The checkpoint on the GPU as it ships (the MTP layer is ``load_mtp``'s)."""

    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    b = _Builder(model_dir, device, cfg)
    layers = []
    for i in range(cfg.layers):
        p = f"model.layers.{i}."
        gdn = attn = None
        if cfg.is_linear(i):
            a = p + "linear_attn."
            gdn = GDN(qkv=b.linear(a + "in_proj_qkv"), z=b.linear(a + "in_proj_z"), b=b.linear(a + "in_proj_b"),
                      a=b.linear(a + "in_proj_a"), out=b.linear(a + "out_proj"),
                      conv=b.tensor(a + "conv1d.weight").reshape(-1, cfg.conv_kernel).to(torch.bfloat16).contiguous(),
                      A_log=b.tensor(a + "A_log").float().contiguous(),
                      dt_bias=b.tensor(a + "dt_bias").float().contiguous(), norm=b.norm(a + "norm.weight"))
        else:
            attn = b.attention(p + "self_attn.")
        layers.append(Layer(linear=cfg.is_linear(i), input_norm=b.norm(p + "input_layernorm.weight"),
                            post_norm=b.norm(p + "post_attention_layernorm.weight"), gdn=gdn, attn=attn, gate=None,
                            up=None, down=None, moe=b.routed(p + "mlp.")))
        torch.cuda.empty_cache()
    w = Weights(config=cfg, embed=Plain(b.tensor("model.embed_tokens.weight").to(torch.bfloat16).contiguous()),
                layers=layers, norm=b.norm("model.norm.weight"), head=b.linear("lm_head"), quant="modelopt")
    half = cfg.rope_dims // 2
    w.inv_freq = (cfg.rope_theta ** (-torch.arange(0, half, dtype=torch.float64) / half)).to(torch.float32).to(device)
    left = b.rd.left(mtp=False)
    b.rd.close()
    if left:
        raise ValueError(f"unused checkpoint tensors: {left[:5]} ...")
    torch.cuda.empty_cache()
    return w


def load_mtp(model_dir: str | Path, w: Weights, device: str = "cuda"):
    """The MTP layer (bf16, excluded from quantization) packed like the qwen3_5_moe ``MTP``, or None without one."""

    from .weights import MTP

    model_dir = Path(model_dir)
    b = _Builder(model_dir, device, w.config)
    if not b.rd.has("mtp.fc.weight"):
        b.rd.close()
        return None
    d = w.config.hidden
    fc = b.tensor("mtp.fc.weight").to(torch.bfloat16)                  # [d, 2d]: [embedding | hidden] inputs
    p = "mtp.layers.0."
    m = MTP(norm_e=b.norm("mtp.pre_fc_norm_embedding.weight"), norm_h=b.norm("mtp.pre_fc_norm_hidden.weight"),
            fc_e=Dense(fc[:, :d].contiguous()), fc_h=Dense(fc[:, d:].contiguous()),
            input_norm=b.norm(p + "input_layernorm.weight"), post_norm=b.norm(p + "post_attention_layernorm.weight"),
            attn=b.attention(p + "self_attn."), moe=b.routed(p + "mlp."), norm=b.norm("mtp.norm.weight"))
    del fc
    left = b.rd.left(mtp=True)
    b.rd.close()
    if left:
        raise ValueError(f"unused MTP tensors: {left[:5]}")
    torch.cuda.empty_cache()
    return m


def draft_head(model_dir: str | Path, ids, device: str = "cuda") -> FP4Linear:
    """The draft vocabulary's rows of the NVFP4 ``lm_head``, column j scoring ``ids[j]`` (the target's own head
    bytes, cut to those rows once; drafts pick rows to verify, never output bits)."""

    rd = _Reader(Path(model_dir))
    try:
        rows = torch.as_tensor(ids, dtype=torch.int64)
        words = rd.get("lm_head.weight").index_select(0, rows)
        scales = rd.get("lm_head.weight_scale").view(torch.uint8).index_select(0, rows)
        return fp4_linear(words.to(device), scales.to(device), rd.get("lm_head.weight_scale_2"))
    finally:
        rd.close()


def weight_bytes(draft_rows: int, mtp: bool) -> Callable[[str, dict], tuple[int, int]]:
    """Device bytes a checkpoint tensor takes once loaded (``capacity.admit``'s transform), before any load."""

    import math

    from tensorfold.cuda.capacity import itemsize

    def transform(name: str, info: dict) -> tuple[int, int]:
        if SKIPPED.search(name) or name.endswith(".weight_scale_2"):
            return 0, 0
        in_mtp = name.startswith("mtp.") or ".mtp." in name
        if in_mtp and not mtp:
            return 0, 0
        shape = [int(n) for n in info["shape"]]
        size = math.prod(shape) * itemsize(info, name)
        if info["dtype"] == "U8" and name.endswith(".weight"):         # NVFP4 words, each row's fp32 scale
            size += 4 * shape[0]
            if name == "lm_head.weight" and mtp:                       # the draft vocabulary's rows
                size += draft_rows * (shape[1] + shape[1] // 8 + 4)
        elif info["dtype"] == "F8_E4M3" and name.endswith(".weight"):  # FP8 codes, each row's fp32 scale
            size += 4 * shape[0]
        elif name.endswith((".A_log", ".dt_bias")) or name.endswith(CENTRED):
            size *= 2                                                   # widened to fp32
        elif in_mtp and ".experts." in name:                            # identity-scaled bf16 tables
            size += size // 8 + 4 * math.prod(shape[:2])
        elif name.endswith(".shared_expert.gate_proj.weight") or name.endswith(".shared_expert.up_proj.weight") \
                or name.endswith(".shared_expert.down_proj.weight"):
            size += size // 8 if info["dtype"] == "BF16" else 0
        return size, 0

    return transform
