"""Qwen3.6-35B-A3B (model_type qwen3_5_moe) on one NVIDIA GPU: a CUDA-only family drafted by its MTP drafter."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

MODEL_TYPES = ("qwen3_5_moe",)
TITLE = "Qwen3.6-35B-A3B"
MODELS = ("mlx-community/Qwen3.6-35B-A3B-4bit",)
# the checkpoint's MTP head, split out by mlx-vlm: `--drafter auto` passes its directory to cuda_engine once pulled
DRAFTER = "mlx-community/Qwen3.6-35B-A3B-MTP-4bit"
DRAFTER_MODEL_TYPE = "qwen3_5_mtp"

# the CUDA engine's kernels read MLX affine weights of this (bits, group size)
CUDA_QUANTIZATION = (4, 64)
# the prompt-plus-reply window admitted by default, shrunk when memory is short (an explicit --context is refused)
CONTEXT = 32768
MAX_DRAFTS = 15           # --mtp-drafts at most (windows of up to 16 rows)

# the text model's shapes the kernels are built for (config.json text_config)
SHAPE = {"hidden_size": 2048, "num_hidden_layers": 40, "vocab_size": 248320, "num_attention_heads": 16,
         "num_key_value_heads": 2, "head_dim": 256, "linear_num_key_heads": 16, "linear_num_value_heads": 32,
         "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "num_experts": 256,
         "num_experts_per_tok": 8, "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512}
# the only modules quantized apart: the router and shared-expert gate (8-bit here, 4-bit in the drafter), dequantized
_ROUTER = re.compile(r"(^|\.)layers\.\d+\.mlp\.(gate|shared_expert_gate)$")
_QUANT_KEYS = ("bits", "group_size", "mode", "quant_method")


def _text(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("text_config") or config)


def _quant_block(config: dict[str, Any]) -> dict[str, Any]:
    for source in (config, _text(config)):
        for key in ("quantization", "quantization_config"):
            if isinstance(source.get(key), dict) and source[key]:
                return source[key]
    return {}


def eos_ids(model_dir: str | Path) -> tuple[int, ...]:
    """The ids that end a reply, each once: generation_config.json's (``<|im_end|>`` first), then config.json's."""

    from tensorfold.families import read_config

    path = Path(model_dir) / "generation_config.json"
    config = read_config(model_dir)
    sources = ([json.loads(path.read_text())] if path.is_file() else []) + [config, _text(config)]
    found: list[int] = []
    for source in sources:
        value = source.get("eos_token_id")
        for token in value if isinstance(value, list) else [] if value is None else [value]:
            if int(token) not in found:
                found.append(int(token))
    return tuple(found)


def _shape_errors(text: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    return [f"{k} {text.get(k)} (not {SHAPE[k]})" for k in keys if text.get(k) != SHAPE[k]]


def check(model_dir: str | Path) -> None:
    """Refuse from config.json alone a checkpoint the kernels cannot read: another format, bit width or size."""

    from tensorfold.families import OWN_MODEL_HELP, describe_quantization, quantization, read_config

    config = read_config(model_dir)
    if quantization(config) != CUDA_QUANTIZATION:
        raise ValueError(f"{TITLE}'s CUDA kernels read MLX 4-bit weights in groups of 64 ({MODELS[0]}); this "
                         f"checkpoint has {describe_quantization(config)}. {OWN_MODEL_HELP}")
    for key, value in _quant_block(config).items():
        if key in _QUANT_KEYS:
            continue
        if not (isinstance(value, dict) and _ROUTER.search(key) and int(value.get("group_size", 64)) == 64
                and int(value.get("bits", 0)) in (4, 8)):
            raise ValueError(f"{TITLE}'s CUDA kernels read every projection in 4 bits, groups of 64 (only the router "
                             f"and shared-expert gate may be 8-bit); this checkpoint quantizes {key} as {value}. "
                             f"Use {MODELS[0]}. {OWN_MODEL_HELP}")
    wrong = _shape_errors(_text(config), tuple(SHAPE))
    if wrong:
        raise ValueError(f"{TITLE}'s CUDA kernels are built for its shapes; this {config.get('model_type')} "
                         f"checkpoint has {', '.join(wrong)}. Tested checkpoint: {MODELS[0]}. {OWN_MODEL_HELP}")


def check_drafter(drafter_dir: str | Path) -> None:
    """Refuse a draft model that is not this model's MTP drafter (``DRAFTER``) before any weight is read."""

    from tensorfold.families import describe_quantization, quantization, read_config

    config = read_config(drafter_dir)
    kind = config.get("model_type")
    if kind != DRAFTER_MODEL_TYPE:
        raise ValueError(f"{TITLE} drafts with its MTP drafter ({DRAFTER}, model_type {DRAFTER_MODEL_TYPE!r}); "
                         f"{drafter_dir} has model_type {kind!r}. Pass --drafter {DRAFTER}, or --no-drafts")
    if quantization(config) != CUDA_QUANTIZATION:
        raise ValueError(f"{TITLE}'s MTP drafter is read as MLX 4-bit weights in groups of 64 ({DRAFTER}); "
                         f"{drafter_dir} has {describe_quantization(config)}")
    text = _text(config)
    wrong = _shape_errors(text, ("hidden_size", "vocab_size", "num_attention_heads", "num_key_value_heads",
                                 "head_dim", "num_experts", "num_experts_per_tok", "moe_intermediate_size",
                                 "shared_expert_intermediate_size"))
    if int(text.get("mtp_num_hidden_layers", 1)) != 1:
        wrong.append(f"mtp_num_hidden_layers {text.get('mtp_num_hidden_layers')} (not 1)")
    if wrong:
        raise ValueError(f"{drafter_dir} is not {TITLE}'s MTP drafter ({DRAFTER}): it has {', '.join(wrong)}")


def requested_context(model_dir: str | Path, context: int | None, explicit: bool) -> int:
    """The window to admit: ``CONTEXT`` unless --context was given (0: the whole window); a bad one is refused here."""

    from tensorfold.families import read_config

    window = int(_text(read_config(model_dir)).get("max_position_embeddings") or 262144)
    if not explicit or context is None:
        return min(CONTEXT, window)
    if int(context) < 0:
        raise ValueError(f"--context must be 0 (the model's {window}-token window) or a token count, not {context}")
    if int(context) > window:
        raise ValueError(f"--context {context} exceeds {TITLE}'s {window}-token window")
    return int(context)


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None,
                context: int | None = None, **options: Any):
    """Check the GPU count, checkpoint, context and drafter before torch is imported, then build the one-GPU engine."""

    if int(tp) != 1 or int(rank) != 0:
        raise ValueError(f"{TITLE} runs on one GPU (it fits one DGX Spark): serve it without --tp and --rank")
    check(model_dir)
    if mtp_drafts is not None and int(mtp_drafts) < 0:
        raise ValueError(f"--mtp-drafts must be 0 or more, not {mtp_drafts}")
    if mtp_drafts is not None and int(mtp_drafts) > MAX_DRAFTS:
        raise ValueError(f"--mtp-drafts: at most {MAX_DRAFTS} MTP drafts a round, not {mtp_drafts}")
    if drafter and not no_drafts:
        check_drafter(drafter)
    elif mtp_drafts and not no_drafts:
        raise ValueError(f"{TITLE}'s MTP drafts come from its drafter ({DRAFTER}): `tensorfold pull {DRAFTER}` once "
                         "(--drafter auto then uses it), or pass --no-drafts for the serial reference")
    explicit = bool(options.get("context_explicit", context is not None))
    requested = requested_context(model_dir, context, explicit)
    from .cuda.engine import Qwen36Engine

    return Qwen36Engine(Path(model_dir), drafter="" if no_drafts else str(drafter or ""), context=requested,
                        context_explicit=explicit, no_drafts=bool(no_drafts),
                        mtp_drafts=None if mtp_drafts is None else int(mtp_drafts),
                        streams=max(1, int(options.get("parallel") or 1)),
                        **({} if options.get("prompt_cache_gib") is None else
                           {"prompt_cache_gib": float(options["prompt_cache_gib"])}))
