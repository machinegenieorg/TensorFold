"""Qwen3.6 MoE's NVFP4 route: ``nvidia/Qwen3.6-35B-A3B-NVFP4`` (ModelOpt, ``MIXED_PRECISION``) read as it ships.

The checkpoint names each quantized layer's algorithm in ``quantized_layers``, and the loader follows the stored
tensors:

* ``W4A16_NVFP4`` (blocks of 16): the routed experts, the shared expert and ``lm_head``. Each is packed E2M1
  nibbles, an e4m3 scale a block and an fp32 scale a tensor, kept as stored: the experts, the shared one last, in
  one table of ``tensorfold.cuda.nvfp4_experts`` (the grouped kernel on the experts plan), the head and the draft
  head through its dense form. Activations stay bf16.
* ``FP8``: DeltaNet's ``in_proj_qkv``, ``in_proj_z`` and ``out_proj`` and attention's four projections: e4m3
  codes and one fp32 scale a tensor (``fp8.py``, weight-only; ``input_scale`` is vLLM's activation scale and is
  not read).
* bf16 as stored: the embedding, the routers and shared-expert gates, ``in_proj_a`` and ``in_proj_b``, the
  convolution, the norms and the MTP layer's projections (``bf16.matmul``). The MTP layer's experts only draft: they
  ride the MLX route's 4-bit experts kernel, requantized at load (``affine4``).

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

from tensorfold.cuda import experts as grouped
from tensorfold.cuda import nvfp4_experts
from tensorfold.cuda.moe import Routed
from tensorfold.families.qwen3_5.cuda.weights import GDN, Attention, Config, Layer, Plain, Weights
from tensorfold.families.qwen4_exp.cuda import bf16, nvfp4

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
    """An NVFP4 projection as stored, through ``nvfp4_experts``' dense form (the vocabulary heads): ``rows`` of the
    table are real, the rest pad it to whole 32-row blocks and are cut from the output."""

    layout = "nvfp4"
    fast = False

    def __init__(self, d: nvfp4_experts.Dense, rows: int) -> None:
        self.d, self.rows = d, int(rows)

    @property
    def n(self) -> int:
        return self.rows

    @property
    def k(self) -> int:
        return self.d.k

    def nbytes(self) -> int:
        return self.d.nbytes()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = nvfp4_experts.dense(x, self.d)
        return y if self.rows == self.d.n else y[:, :self.rows]

    prefill = __call__

    def dequantize(self) -> torch.Tensor:
        """The exact fp32 weight [rows, k] (the reference for checks)."""

        words, scales = nvfp4_experts.unpack(self.d.w[:, :, :, 0])
        return nvfp4.dequantize(words[0], scales[0].view(torch.float8_e4m3fn), float(self.d.s2) * 2)[:self.rows]


def fp4_linear(words: torch.Tensor, scales: torch.Tensor, scale2: torch.Tensor | float) -> FP4Linear:
    return FP4Linear(nvfp4_experts.make_dense(words, scales, scale2), words.shape[0])


def experts_table(gate: tuple, up: tuple, down: tuple, shared: tuple) -> nvfp4_experts.Experts:
    """One layer's routed experts from stacked checkpoint arrays (each ([E, n, k/2] uint8, [E, n, k/16] e4m3, [E]
    fp32)) and the NVFP4 shared expert's (the same, unstacked) as expert E of one table."""

    def join(stack: tuple, one: tuple) -> tuple:
        (w, sc, s2), (ow, osc, os2) = stack, one
        return (torch.cat([w, ow[None]]), torch.cat([sc, osc.contiguous().view(torch.uint8)[None]]),
                torch.cat([s2, torch.as_tensor(os2, dtype=torch.float32).reshape(1).to(s2.device)]))

    return nvfp4_experts.make(join(gate, shared[0]), join(up, shared[1]), join(down, shared[2]))


def affine4(w: torch.Tensor, gs: int = 64, chunk: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16 [E, N, K] as MLX affine 4-bit (words [E, N, K/8] int32, bf16 scales and biases a group of ``gs``,
    q = round((w - min) / scale)): for weights that only draft (the MTP layer's experts), where the rounding changes
    speed, never bits."""

    e, n, k = w.shape
    words = torch.empty((e, n, k // 8), dtype=torch.int32, device=w.device)
    scales = torch.empty((e, n, k // gs), dtype=torch.bfloat16, device=w.device)
    biases = torch.empty_like(scales)
    shifts = 4 * torch.arange(8, device=w.device, dtype=torch.int64)
    for a in range(0, e, chunk):
        g = w[a:a + chunk].float().reshape(-1, n, k // gs, gs)
        lo = g.amin(-1).to(torch.bfloat16)
        s = ((g.amax(-1) - lo.float()) / 15).clamp(min=1e-8).to(torch.bfloat16)
        q = ((g - lo.float()[..., None]) / s.float()[..., None]).round().clamp(0, 15).to(torch.int64)
        packed = (q.reshape(-1, n, k // 8, 8) << shifts).sum(-1)
        words[a:a + chunk] = torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed).to(torch.int32)
        scales[a:a + chunk], biases[a:a + chunk] = s, lo
    return words, scales, biases


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
        """A layer's router rows (the shared expert's gate row last) and its experts, the shared one last: NVFP4
        experts in one NVFP4 table, or the MTP layer's stacked bf16 experts on the MLX 4-bit kernels."""

        router = torch.cat([self.tensor(prefix + "gate.weight"), self.tensor(prefix + "shared_expert_gate.weight")])
        sp = prefix + "shared_expert."
        fp4_shared = self.rd.has(sp + "gate_proj.weight_scale_2")
        if self.rd.has(prefix + "experts.0.gate_proj.weight_scale_2") and fp4_shared:
            experts = experts_table(*(self._stack(prefix, f"{p}_proj") for p in ("gate", "up", "down")),
                                    tuple(self._fp4(sp + f"{p}_proj") for p in ("gate", "up", "down")))
        elif self.rd.has(prefix + "experts.gate_up_proj") and not fp4_shared:     # stacked bf16 (the MTP layer's)
            gu = self.tensor(prefix + "experts.gate_up_proj").to(torch.bfloat16)   # [E, 2 ni, d], gate rows first
            dn = self.tensor(prefix + "experts.down_proj").to(torch.bfloat16)
            sg, su, sd = (self.tensor(sp + f"{p}_proj.weight").to(torch.bfloat16) for p in ("gate", "up", "down"))
            ni = gu.shape[1] // 2
            tables = [affine4(torch.cat([part, one[None]])) for part, one in
                      ((gu[:, :ni], sg), (gu[:, ni:], su), (dn, sd))]
            del gu, dn
            experts = grouped.make(tables[:2], tables[2], 64)
        else:
            raise ValueError(f"{prefix}experts: NVFP4 experts with an NVFP4 shared expert, or stacked bf16 experts "
                             "with a bf16 shared expert, are what this route reads")
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
        if SKIPPED.search(name) or name == "lm_head.weight_scale_2":   # the head's is a factor on each row
            return 0, 0
        in_mtp = name.startswith("mtp.") or ".mtp." in name
        if in_mtp and not mtp:
            return 0, 0
        shape = [int(n) for n in info["shape"]]
        size = math.prod(shape) * itemsize(info, name)
        if name == "lm_head.weight":                                   # NVFP4 words, each row's fp32 scale
            size += 4 * shape[0] + (draft_rows * (shape[1] + shape[1] // 8 + 4) if mtp else 0)   # and the draft rows
        elif info["dtype"] == "F8_E4M3" and name.endswith(".weight"):  # FP8 codes, each row's fp32 scale
            size += 4 * shape[0]
        elif name.endswith((".A_log", ".dt_bias")) or name.endswith(CENTRED):
            size *= 2                                                   # widened to fp32
        elif in_mtp and (".experts." in name or ".shared_expert." in name):
            size = size * 9 // 32                                       # bf16 as MLX 4-bit, groups of 64
        return size, 0

    return transform
