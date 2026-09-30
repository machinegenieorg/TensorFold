"""Serve options a backend or family has no path for, refused before any weight is downloaded."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any


def _served_name(args: argparse.Namespace) -> str:
    """``--name``, else a repo id's last segment or the model path's name; no I/O (matches ``cli._serve_cuda``)."""

    if getattr(args, "name", ""):
        return args.name
    from tensorfold import hub

    model = str(args.model).rstrip("/")
    return model.split("/")[-1] if hub.is_repo_id(str(args.model)) else Path(model).name


def name_priority(args: argparse.Namespace) -> dict[str, str]:
    """``--name-priority ID=background`` parsed: the id and the priority word it defaults to. No I/O."""

    parsed: dict[str, str] = {}
    for entry in getattr(args, "name_priority", None) or []:
        model_id, sep, priority = str(entry).partition("=")
        if not sep or not model_id.strip() or priority.strip().lower() != "background":
            raise ValueError(f"--name-priority takes ID=background, not {entry!r}")
        parsed[model_id.strip()] = "background"
    return parsed


def check(args: argparse.Namespace, family: Any, backend: str, config_dir: Any = None) -> None:
    """Refuse a KV cache, draft rule, image or share option the backend or family can't serve, before any download."""

    if getattr(args, "vision_urls", False) and not getattr(args, "vision", False):
        raise ValueError("--vision-urls needs --vision")
    if getattr(args, "vision", False):             # only --vision reads the config here
        from tensorfold.families import read_config
        from tensorfold.vision.config import validate_vision_config

        validate_vision_config(read_config(config_dir) if config_dir else {}, family.model_type)
    share = getattr(args, "decode_share", None)
    if share is not None and backend == "cuda":
        raise ValueError("--decode-share sets the Mac server's share; the CUDA engine runs a round after each 1,024 "
                         "prompt rows")
    if share is not None and share < 0:
        raise ValueError(f"--decode-share is 0 (whole prompts first) or more, not {share}")
    kv = getattr(args, "kv_dtype", "bf16")
    if kv != "bf16" and backend != "cuda":
        raise ValueError(f"--kv-dtype {kv} is a CUDA engine option: the MLX path caches keys and values as bf16")
    supported = getattr(family.package, "CUDA_KV_DTYPES", ("bf16",))
    if kv not in supported:
        raise ValueError(f"{family.title} on CUDA serves a {' or '.join(supported)} KV cache, not --kv-dtype {kv}")
    priorities = name_priority(args)
    if priorities and backend != "cuda":
        raise ValueError("--name-priority is a CUDA server option: the MLX server has its own background rule")
    if priorities:
        served = _served_name(args)
        ids = {served, *(str(a).strip() for a in getattr(args, "alias", None) or () if str(a).strip())}
        bad = sorted(set(priorities) - ids)
        if bad:
            raise ValueError(f"--name-priority names {', '.join(bad)}, not --name or an --alias "
                             f"({', '.join(sorted(ids))})")
    confidence = getattr(args, "mtp_confidence", None)
    if confidence is None:
        return
    engine = getattr(family.package, "cuda_engine", None) if backend == "cuda" else None
    if engine is None or "mtp_confidence" not in inspect.signature(engine).parameters:
        raise ValueError(f"--mtp-confidence sets where a CUDA engine's MTP chains stop; {family.title} on "
                         f"{'CUDA' if backend == 'cuda' else 'MLX'} has no such rule")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"--mtp-confidence is a probability from 0 to 1, not {confidence}")


def vision_options(args: argparse.Namespace) -> dict[str, Any]:
    """``--vision`` and ``--vision-urls`` as a family's load options."""

    if not getattr(args, "vision", False):
        return {}
    return {"vision": True, "vision_urls": bool(getattr(args, "vision_urls", False))}


__all__ = ["check", "vision_options"]
