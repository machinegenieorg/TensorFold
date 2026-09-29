"""Qwen3 dense decoders served as last-token embedding models on CUDA: /v1/embeddings, a row-invariant prompt forward."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODEL_TYPES = ("qwen3",)
TITLE = "Qwen3 embeddings"
MODELS = ("Qwen/Qwen3-Embedding-8B",)
# bf16 as shipped, or MLX affine 4-bit in groups of 64 (``python -m tensorfold.families.qwen3.convert``)
QUANT_METHODS = {"cuda": (None, "mlx")}
CUDA_AFFINE_BITS = (4,)
CUDA_AFFINE_GROUPS = (64,)
EMBED_BITS = (4, 8)            # the token table may keep more bits than the projections
HEAD_DIMS = (64, 128, 256)     # the prompt attention kernel's head widths
BATCH_TOKENS = 8192            # tokens one forward step takes by default (--batch-tokens)


def _text(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("text_config") or config


def check(model_dir: str | Path) -> None:
    """Refuse from config.json alone, before any weight downloads, a Qwen3 checkpoint these kernels cannot read."""

    from tensorfold.families import OWN_MODEL_HELP, read_config

    config = read_config(model_dir)
    t = _text(config)
    head_dim = int(t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"])
    problems = []
    if t.get("attention_bias"):
        problems.append("attention biases")
    if t.get("rope_scaling") or (t.get("rope_parameters") or {}).get("rope_type", "default") != "default":
        problems.append("scaled rotary positions")
    if t.get("use_sliding_window"):
        problems.append("sliding-window attention")
    if head_dim not in HEAD_DIMS:
        problems.append(f"head width {head_dim} (the kernels take {', '.join(map(str, HEAD_DIMS))})")
    if int(t["hidden_size"]) % 64 or int(t["intermediate_size"]) % 64:
        problems.append("widths that are not multiples of 64")
    if int(t["num_attention_heads"]) % int(t["num_key_value_heads"]):
        problems.append("query heads that are not a multiple of the key/value heads")
    if problems:
        raise ValueError(f"{TITLE} cannot read this checkpoint: {', '.join(problems)}. Tested checkpoints: "
                         f"{', '.join(MODELS)}. {OWN_MODEL_HELP}")


def check_quantization(config: dict[str, Any], backend: str) -> None:
    """MLX affine 4-bit groups of 64 for the projections; the token table 4- or 8-bit, or unquantized."""

    from tensorfold.quantization import checkpoint_specs

    for path, spec in checkpoint_specs(config).items():
        embed = path.endswith("embed_tokens")
        if spec is None and embed:
            continue
        if spec is None or spec.group_size != 64 or spec.bits not in (EMBED_BITS if embed else CUDA_AFFINE_BITS):
            where = f"'{path}'" if path else "the checkpoint"
            raise ValueError(f"{TITLE}'s CUDA kernels read MLX affine 4-bit weights in groups of 64 (the token "
                             f"table may be 4- or 8-bit or unquantized); {where} declares {spec or 'no quantization'}. "
                             "Convert the bf16 checkpoint with `python -m tensorfold.families.qwen3.convert`")


def pooling(model_dir: str | Path) -> None:
    """A sentence-transformers checkpoint must pool its last token (checked once the full snapshot is here)."""

    path = Path(model_dir) / "1_Pooling" / "config.json"
    if not path.exists():
        raise ValueError(f"{TITLE} serves last-token embedding checkpoints (such as {MODELS[0]}); this folder has no "
                         "1_Pooling/config.json saying how its vectors are pooled")
    config = json.loads(path.read_text())
    if not config.get("pooling_mode_lasttoken"):
        modes = [k.removeprefix("pooling_mode_") for k, v in config.items() if k.startswith("pooling_mode_") and v]
        raise ValueError(f"{TITLE} pools the last token; this checkpoint pools by {', '.join(modes) or 'nothing'}")


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, context: int | None = None,
                batch_tokens: int | None = None, **options: Any):
    """The one-GPU embedding engine; admission runs on the checkpoint's headers before any weight loads."""

    if drafter:
        raise ValueError(f"{TITLE} embeds prompts and decodes nothing: a draft model does not apply")
    if int(tp) != 1:
        raise ValueError(f"{TITLE} runs on one GPU: drop --tp")
    if int(options.get("parallel") or 1) > 1:
        raise ValueError(f"{TITLE} batches embedding requests itself (--batch-tokens): drop --parallel")
    if options.get("vision"):
        raise ValueError(f"{TITLE} embeds text: drop --vision")
    pooling(model_dir)
    from .cuda.engine import Qwen3EmbedEngine

    return Qwen3EmbedEngine(Path(model_dir), context=context, context_explicit=options.get("context_explicit"),
                            batch_tokens=BATCH_TOKENS if batch_tokens is None else int(batch_tokens))
