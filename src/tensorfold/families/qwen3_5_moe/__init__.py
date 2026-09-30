"""Qwen3.6 MoE (qwen3_5_moe) on CUDA: the 27B's DeltaNet and attention with routed experts, MTP drafts on the lanes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("qwen3_5_moe",)
TITLE = "Qwen3.6 MoE"
LANES = True
# MLX 4-bit, groups of 64, routers 8-bit, MTP layer in mtp-4bit.safetensors (mlx-community's files take it too);
# and NVIDIA's NVFP4 checkpoint as it ships: experts, shared expert and lm_head NVFP4, projections FP8, MTP bf16
MODELS = ("Vontra/Qwen3.6-35B-A3B-MLX-4bit-MTP", "nvidia/Qwen3.6-35B-A3B-NVFP4")
REQUIRED_FILES = {MODELS[0]: ("mtp-4bit.safetensors",)}
QUANT_METHODS = {"cuda": ("mlx", "modelopt")}     # MLX affine 4-bit, or ModelOpt NVFP4 with FP8 projections
# the CUDA engine's kernels read MLX affine weights of this (bits, group size)
CUDA_QUANTIZATION = (4, 64)
# the ModelOpt algorithms the NVFP4 route reads: NVFP4 weights in blocks of 16, FP8 weights with a scale a tensor
# (activations stay bf16 either way: the checkpoint's activation scales are not read)
MODELOPT_ALGOS = {"W4A16_NVFP4": 16, "NVFP4": 16, "FP8": None}


def check(model_dir: str | Path) -> None:
    """One GPU; MLX 4-bit weights in groups of 64, or a ModelOpt checkpoint of NVFP4 (blocks of 16) and FP8 layers."""

    from tensorfold.families import OWN_MODEL_HELP, describe_quantization, quant_method, quantization, read_config

    config = read_config(model_dir)
    if quant_method(config) == "modelopt":
        block = config.get("quantization_config") or config.get("quantization") or {}
        layers = block.get("quantized_layers")
        extra = Path(model_dir) / "hf_quant_config.json"
        if not layers and extra.is_file():
            import json

            layers = (json.loads(extra.read_text()).get("quantization") or {}).get("quantized_layers")
        algos = {(str(v.get("quant_algo")).upper(), v.get("group_size")) for v in (layers or {}).values()}
        if not algos:
            algos = {(str(block.get("quant_algo")).upper(), block.get("group_size"))}
        bad = sorted(f"{a}" + (f" (blocks of {g})" if g else "") for a, g in algos
                     if a not in MODELOPT_ALGOS or (MODELOPT_ALGOS[a] and int(g or 16) != MODELOPT_ALGOS[a]))
        if bad:
            raise ValueError(f"{TITLE}'s CUDA engine reads ModelOpt NVFP4 weights in blocks of 16 and FP8 weights "
                             f"({MODELS[1]}); this checkpoint has {', '.join(bad)}. {OWN_MODEL_HELP}")
        return
    if quantization(config) != CUDA_QUANTIZATION:
        raise ValueError(f"{TITLE}'s CUDA engine reads MLX 4-bit weights in groups of 64 ({MODELS[0]}) or NVIDIA's "
                         f"NVFP4 checkpoint ({MODELS[1]}); this checkpoint has {describe_quantization(config)}. "
                         f"{OWN_MODEL_HELP}")


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None,
                context: int | None = None, **options: Any):
    """The one-GPU engine: MTP chains verified exactly, or the serial reference with ``no_drafts``; ``parallel`` > 1 decodes that many requests together."""

    if drafter:
        raise ValueError(f"{TITLE} drafts with its own MTP layer on CUDA: a separate draft model does not apply")
    if int(tp) != 1:
        raise ValueError(f"{TITLE} runs on one GPU: drop --tp")
    from .cuda import DEPTH
    from .cuda.engine import Qwen36Engine

    depth = 0 if no_drafts else DEPTH if mtp_drafts is None else int(mtp_drafts)
    streams = max(1, int(options.get("parallel") or 1))
    if streams > 1 and not 0 <= depth <= 15:
        raise ValueError(f"--parallel verifies up to 16 rows a stream: --mtp-drafts 0 to 15, not {depth}")
    return Qwen36Engine(Path(model_dir), depth=depth, context=context, context_explicit=options.get("context_explicit"),
                        streams=streams)
