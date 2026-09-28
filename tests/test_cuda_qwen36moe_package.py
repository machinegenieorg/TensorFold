"""Qwen3.6-35B-A3B's package registers a CUDA-only family and refuses what it cannot serve before torch is imported."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tensorfold import families
from tensorfold.families import qwen3_5_moe as family

GENERATION = {"bos_token_id": 248044, "do_sample": True, "eos_token_id": [248046, 248044], "pad_token_id": 248044,
              "temperature": 1.0, "top_k": 20, "top_p": 0.95}


def _config(quant: dict | None = None, **text_changes) -> dict:
    """config.json as mlx-community/Qwen3.6-35B-A3B-4bit ships it (text dims, 8-bit router overrides)."""

    kinds = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(40)]
    text = {"model_type": "qwen3_5_moe_text", **family.SHAPE, "layer_types": kinds, "full_attention_interval": 4,
            "eos_token_id": 248044, "bos_token_id": 248044, "rms_norm_eps": 1e-6, "attn_output_gate": True,
            "partial_rotary_factor": 0.25, "mtp_num_hidden_layers": 1, "tie_word_embeddings": False,
            "max_position_embeddings": 262144,
            "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
                                "rope_theta": 10000000, "rope_type": "default"}}
    text.update(text_changes)
    if quant is None:
        quant = {"group_size": 64, "bits": 4, "mode": "affine"}
        for i in range(40):
            for module in ("gate", "shared_expert_gate"):
                quant[f"language_model.model.layers.{i}.mlp.{module}"] = {"group_size": 64, "bits": 8}
    return {"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe",
            "eos_token_id": 248044, "quantization": quant, "quantization_config": dict(quant), "text_config": text,
            "tie_word_embeddings": False}


def _drafter_config(**text_changes) -> dict:
    config = _config({"group_size": 64, "bits": 4, "mode": "affine"}, **text_changes)
    return {"block_size": 3, "model_type": "qwen3_5_mtp", "quantization": config["quantization"],
            "quantization_config": config["quantization"], "text_config": config["text_config"],
            "tie_word_embeddings": False}


def _write(folder: Path, config: dict, generation: dict | None = GENERATION) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config))
    if generation is not None:
        (folder / "generation_config.json").write_text(json.dumps(generation))
    return folder


def test_the_family_is_registered_for_qwen3_5_moe_on_cuda_only(tmp_path):
    found = families.families()["qwen3_5_moe"]
    assert found.module == "tensorfold.families.qwen3_5_moe"
    assert found.title == "Qwen3.6-35B-A3B" and not found.lanes
    assert families.backends_of(found) == ("cuda",)       # no `load`: the MLX backend is refused
    assert families.detect(_write(tmp_path / "main", _config())).module == found.module
    assert "qwen3_5_mtp" not in families.families()       # the drafter is not a model to serve
    assert family.MODELS == ("mlx-community/Qwen3.6-35B-A3B-4bit",)
    assert family.DRAFTER == "mlx-community/Qwen3.6-35B-A3B-MTP-4bit"
    families.require_readable(found, _config(), "cuda")


def test_the_cli_lists_it_and_refuses_the_mlx_backend(capsys):
    from tensorfold.cli import _backend, main

    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "Qwen3.6-35B-A3B (qwen3_5_moe; CUDA engine)" in out
    assert "model    mlx-community/Qwen3.6-35B-A3B-4bit" in out
    assert "drafter  mlx-community/Qwen3.6-35B-A3B-MTP-4bit" in out
    found = families.families()["qwen3_5_moe"]
    with pytest.raises(ValueError, match="NVIDIA GPUs only"):
        _backend("mlx", found)
    assert _backend("cuda", found) == "cuda"


def test_the_package_and_its_refusals_never_import_torch(tmp_path):
    folder = _write(tmp_path / "main", _config())
    code = f"""
import sys
from tensorfold.families import qwen3_5_moe as family
family.check({str(folder)!r})
assert family.eos_ids({str(folder)!r})[0] == 248046
for kwargs in ({{"tp": 2}}, {{"rank": 1}}, {{"mtp_drafts": 3}}, {{"mtp_drafts": -1}}, {{"mtp_drafts": 16}},
               {{"context": -1}}, {{"context": 300000}}):
    try:
        family.cuda_engine({str(folder)!r}, **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError(kwargs)
loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("torch", "mlx"))
assert not loaded, loaded
"""
    src = str(Path(family.__file__).resolve().parents[3])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")]))}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("kwargs", [{"tp": 2}, {"tp": 2, "rank": 1, "master": "10.0.0.1"}, {"rank": 1}])
def test_cuda_engine_refuses_more_than_one_gpu(tmp_path, kwargs):
    with pytest.raises(ValueError, match="one GPU"):
        family.cuda_engine(_write(tmp_path / "main", _config()), **kwargs)


class _FakeEngine:
    """Stands in for cuda/engine.py's Qwen36Engine (which imports torch and loads the checkpoint)."""

    def __init__(self, model_dir, **kwargs):
        self.model_dir, self.kwargs = Path(model_dir), kwargs


def test_cuda_engine_checks_everything_then_builds_the_engine_as_the_recipe_runs_it(tmp_path, monkeypatch):
    """What `tensorfold serve` passes reaches the engine: the recipe by default, --context, --no-drafts, --parallel."""

    import types

    fake = types.ModuleType("tensorfold.families.qwen3_5_moe.cuda.engine")
    fake.Qwen36Engine = _FakeEngine
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    folder = _write(tmp_path / "main", _config())
    drafter = _write(tmp_path / "mtp", _drafter_config(), generation=None)
    plain = {"drafter": "", "context": 32768, "context_explicit": False, "no_drafts": False, "mtp_drafts": None,
             "streams": 1}
    given = {**plain, "context_explicit": True}
    for kwargs, want in (({}, plain),
                         ({"drafter": str(drafter)}, {**plain, "drafter": str(drafter)}),
                         ({"drafter": str(drafter), "mtp_drafts": 3},
                          {**plain, "drafter": str(drafter), "mtp_drafts": 3}),
                         ({"drafter": str(drafter), "no_drafts": True}, {**plain, "no_drafts": True}),
                         ({"no_drafts": True, "mtp_drafts": 3}, {**plain, "no_drafts": True, "mtp_drafts": 3}),
                         ({"mtp_drafts": 0}, {**plain, "mtp_drafts": 0}),
                         ({"context": 0}, {**given, "context": 0}),
                         ({"context": 4096, "master_port": 29552}, {**given, "context": 4096}),
                         ({"context": 8192, "context_explicit": True, "parallel": 4},
                          {**given, "context": 8192, "streams": 4}),
                         ({"context": 32768, "context_explicit": False}, plain)):
        engine = family.cuda_engine(folder, **kwargs)
        assert isinstance(engine, _FakeEngine) and engine.model_dir == folder, kwargs
        assert engine.kwargs == want, kwargs
    assert family.CONTEXT == 32768 and family.requested_context(folder, None, False) == 32768
    with pytest.raises(ValueError, match="at most 15"):
        family.cuda_engine(folder, drafter=str(drafter), mtp_drafts=16)
    with pytest.raises(ValueError, match="exceeds"):
        family.cuda_engine(folder, context=262145)


def _bad_quant(**changes) -> dict:
    quant = _config()["quantization"]
    quant.update(changes)
    return quant


@pytest.mark.parametrize("config, reason", [
    (_config({"group_size": 64, "bits": 8, "mode": "affine"}), "4-bit weights in groups of 64"),
    (_config({"group_size": 32, "bits": 4, "mode": "affine"}), "4-bit weights in groups of 64"),
    (_config({"quant_method": "modelopt", "quant_algo": "NVFP4"}), "4-bit weights in groups of 64"),
    (_config(_bad_quant(**{"language_model.model.layers.3.self_attn.q_proj": {"group_size": 64, "bits": 8}})),
     "quantizes language_model.model.layers.3.self_attn.q_proj"),
    (_config(_bad_quant(**{"language_model.model.layers.0.mlp.gate": {"group_size": 32, "bits": 8}})),
     "quantizes language_model.model.layers.0.mlp.gate"),
    (_config(hidden_size=3072, num_experts=512), "hidden_size 3072 \\(not 2048\\), num_experts 512"),
    (_config(linear_num_value_heads=64), "linear_num_value_heads 64"),
])
def test_unsupported_checkpoints_are_refused_from_config_alone(tmp_path, config, reason):
    folder = _write(tmp_path / "main", config)
    with pytest.raises(ValueError, match=reason):
        family.check(folder)
    with pytest.raises(ValueError, match=reason):
        family.cuda_engine(folder)


def test_the_drafter_must_be_the_mtp_drafter(tmp_path):
    folder = _write(tmp_path / "main", _config())
    family.check_drafter(_write(tmp_path / "mtp", _drafter_config(), generation=None))
    dflash = _write(tmp_path / "dflash", {"model_type": "qwen3", "quantization": {"bits": 4, "group_size": 64}})
    with pytest.raises(ValueError, match="model_type 'qwen3'"):
        family.cuda_engine(folder, drafter=str(dflash))
    other = _write(tmp_path / "other", _drafter_config(hidden_size=4096), generation=None)
    with pytest.raises(ValueError, match="hidden_size 4096"):
        family.cuda_engine(folder, drafter=str(other))
    with pytest.raises(ValueError, match="tensorfold pull mlx-community/Qwen3.6-35B-A3B-MTP-4bit"):
        family.cuda_engine(folder, mtp_drafts=3)             # MTP drafts asked for, drafter not pulled
    with pytest.raises(ValueError, match="0 or more"):
        family.cuda_engine(folder, mtp_drafts=-1)


def test_eos_includes_im_end(tmp_path):
    # the shipped config.json lists [248046, 248044] at top level; text_config alone names only 248044
    shipped = _config()
    shipped["eos_token_id"] = [248046, 248044]
    assert family.eos_ids(_write(tmp_path / "shipped", shipped)) == (248046, 248044)
    # generation_config.json alone brings in <|im_end|>
    assert family.eos_ids(_write(tmp_path / "text-only", _config())) == (248046, 248044)
    assert 248046 in family.eos_ids(_write(tmp_path / "no-generation", shipped, generation=None))
    assert family.eos_ids(_write(tmp_path / "neither", _config(), generation=None)) == (248044,)
