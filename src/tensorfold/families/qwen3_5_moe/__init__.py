"""Qwen3.6-35B-A3B (model_type ``qwen3_5_moe``) on one NVIDIA GPU: a CUDA engine only, for DGX Spark (GB10).

40 decoder layers over a hidden size of 2,048: 30 of Gated DeltaNet and 10 of gated full attention (every fourth
layer), 256 routed experts (top 8, width 512) plus a shared expert in every layer, a 248,320-token vocabulary.
The engine reads the MLX 4-bit checkpoint (affine, groups of 64; the router and shared-expert gate in 8 bits) and
skips its vision tower. MLX's converter drops the checkpoint's MTP head, so drafts come from the MTP drafter, a
separate repository (``DRAFTER``, model_type ``qwen3_5_mtp``) that ``--drafter auto`` passes in once pulled.

There is no MLX engine for this family (no ``load``), so ``tensorfold serve`` refuses the MLX backend for it.
``cuda/``: the weight loader and its layout contract with the kernels (``weights.py``), the forward (``forward.py``),
prefill and serial decoding (``decode.py``) and the engine ``tensorfold serve`` runs (``engine.py``).
"""

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
# the prompt-plus-reply window admitted unless --context says otherwise (smaller when memory is short; an explicit
# --context is refused rather than shrunk, and --context 0 asks for the model's whole window)
CONTEXT = 32768
MAX_DRAFTS = 15           # --mtp-drafts at most (windows of up to 16 rows)

# the text model's shapes the kernels are built for (config.json text_config)
SHAPE = {"hidden_size": 2048, "num_hidden_layers": 40, "vocab_size": 248320, "num_attention_heads": 16,
         "num_key_value_heads": 2, "head_dim": 256, "linear_num_key_heads": 16, "linear_num_value_heads": 32,
         "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "num_experts": 256,
         "num_experts_per_tok": 8, "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512}
# the only modules whose quantization may differ from the default: the router and the shared expert's gate, which
# the loader dequantizes to fp32 once (8-bit in the main checkpoint, 4-bit in the drafter), in groups of 64
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
    """The token ids that end a reply: generation_config.json's first (``<|im_end|>`` 248046, then 248044), then
    config.json's (top level, then text_config), each once. text_config alone names only 248044."""

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
    """Refuse, from config.json alone (before any weight downloads), a checkpoint the kernels do not read: another
    storage format or bit width, per-module quantization beyond the router and shared-expert gate, or another
    model size of the same architecture."""

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
    """The window to admit for ``--context``: ``CONTEXT`` unless it was given (0: the model's whole window). A negative
    context, or one past the model's window, is refused here, before torch is imported."""

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
    """The CUDA engine on one GPU (``tensorfold serve`` on a DGX Spark), set up as the recipe runs it.

    Everything here is checked before torch is imported or a GPU is touched: one GPU (``tp=1``), a checkpoint the
    kernels read (``check``), the context (``requested_context``: 32,768 tokens by default) and, with ``drafter``, the
    MTP drafter (``check_drafter``). With the drafter a round verifies the pending token and up to ``mtp_drafts`` MTP
    drafts (6 by default, at most 15; a chain stops before a later draft the head gives less than 50%), over the
    76,882-token draft vocabulary in ``cuda/draft_vocab.txt``. Without it, with ``no_drafts`` or ``mtp_drafts=0``,
    every round decodes one token, the serial reference. The engine then admits the context against the device's memory
    (``tensorfold.cuda.capacity``) before it loads anything. ``options``: the CLI's ``context_explicit`` (whether
    --context was given) and ``parallel`` (--parallel N; requests are still served one at a time).
    """

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
                        streams=max(1, int(options.get("parallel") or 1)))
