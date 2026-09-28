"""Qwen3.6-35B-A3B's loader on CPU: the key map against the real checkpoint's and drafter's safetensors headers (no
weights read), the stored norm convention, and a small fake checkpoint through the loader (expert table, router
dequantization, stacked projections, norms). Needs torch, no GPU; the real-checkpoint tests skip when the
checkpoints are not in the Hugging Face cache."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold import families  # noqa: E402
from tensorfold.families import qwen3_5_moe as family  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda import checkpoint as C  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda import weights as W  # noqa: E402

ORIGINAL = "Qwen/Qwen3.6-35B-A3B"


def _cached(repo: str) -> Path:
    from tensorfold import hub

    try:
        found = hub.cached(repo)
    except Exception:  # noqa: BLE001 - no huggingface_hub or no cache: nothing to test against
        found = None
    if found is None or not (found / "model.safetensors.index.json").is_file():
        pytest.skip(f"{repo} is not in the Hugging Face cache")
    return found


# ---------------------------------------------------------------------------------------------------------------
# the real checkpoints: headers and a few norm vectors only


def test_the_real_checkpoint_maps_every_tensor_and_leaves_only_the_vision_tower():
    snap = _cached(family.MODELS[0])
    cfg = W.Config.read(snap)
    found = W.checkpoint_tensors(snap)
    assert set(found) == set(json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"])
    prefix = W._prefix(found)
    assert prefix == "language_model."
    spec = W.layout(cfg, prefix)
    assert W.layout_problems(found, spec) == []                 # every expected tensor, dtype and shape
    left = set(found) - set(spec)
    assert left and all(name.startswith("vision_tower.") for name in left)
    assert not any(name.startswith(("mtp.", "language_model.mtp")) for name in found)   # mlx dropped the head
    # the router and shared-expert gate are 8-bit (4 inputs a word), everything else 4-bit, groups of 64
    assert spec[prefix + "model.layers.0.mlp.gate.weight"] == ("U32", (256, 512))
    assert spec[prefix + "model.layers.0.mlp.shared_expert_gate.weight"] == ("U32", (1, 512))
    assert spec[prefix + "model.layers.0.mlp.switch_mlp.down_proj.weight"] == ("U32", (256, 2048, 64))
    assert cfg.quant_of(prefix + "model.layers.39.mlp.gate") == (8, 64)
    assert cfg.quant_of(prefix + "model.layers.39.mlp.switch_mlp.gate_proj") == (4, 64)
    assert (cfg.hidden, cfg.layers, cfg.vocab, cfg.bits, cfg.group_size) == (2048, 40, 248320, 4, 64)
    assert cfg.attention_layers == list(range(3, 40, 4))
    assert cfg.gdn_rows == (8192, 4096, 32, 32) and cfg.attn_rows == (8192, 512, 512)
    assert (cfg.rotary_dim, cfg.rope_theta, cfg.mrope_section) == (64, 1e7, (11, 11, 10))
    assert (cfg.experts, cfg.top_k, cfg.moe_width, cfg.shared_width, cfg.norm_topk) == (256, 8, 512, 512, True)
    assert cfg.eos[:2] == (248046, 248044)
    assert not cfg.tie_embeddings
    family.check(snap)


def test_the_real_drafter_maps_every_tensor():
    snap = _cached(family.DRAFTER)
    family.check_drafter(snap)
    cfg = W.Config.read(snap)
    spec = W.mtp_layout(cfg)
    assert W.layout_problems(W.checkpoint_tensors(snap), spec, skip=()) == []    # nothing left over
    assert spec["fc.weight"] == ("U32", (2048, 512))           # [hidden, 2*hidden] in 4 bits
    assert spec["layers.0.mlp.gate.weight"] == ("U32", (256, 256))   # the drafter's router is 4-bit
    assert cfg.quant_of("layers.0.mlp.gate") == (4, 64) and not cfg.quant_overrides
    main = W.Config.read(_cached(family.MODELS[0]))
    for name in ("hidden", "vocab", "heads", "kv_heads", "head_dim", "rotary_dim", "experts", "top_k", "moe_width",
                 "shared_width"):
        assert getattr(cfg, name) == getattr(main, name)


def test_the_cli_drafter_auto_finds_the_pulled_drafter():
    snap = _cached(family.DRAFTER)
    from tensorfold.cli import _drafter

    assert Path(_drafter(families.families()["qwen3_5_moe"], "auto")).resolve() == snap.resolve()


def _read(folder: Path, name: str) -> torch.Tensor:
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    if not (folder / index[name]).is_file():
        pytest.skip(f"{name}'s shard is not in the cache")
    base, header = C.read_header(folder / index[name])
    rd = C.Reader(folder, "cpu", {name: (header[name]["dtype"], tuple(header[name]["shape"]))})
    return rd.get(name).float()


def test_mlx_stores_one_plus_w_for_the_centred_norms():
    snap = _cached(family.MODELS[0])
    cfg = W.Config.read(snap)
    spec = W.layout(cfg, "language_model.")
    rd = C.Reader(snap, "cpu", spec)
    means = [float(rd.get(f"language_model.model.layers.{i}.input_layernorm.weight").float().mean())
             for i in range(cfg.layers)]
    assert W.norms_around_one(means)
    drafter = _cached(family.DRAFTER)
    rd = C.Reader(drafter, "cpu", W.mtp_layout(W.Config.read(drafter)))
    assert W.norms_around_one([float(rd.get("layers.0.input_layernorm.weight").float().mean())])


@pytest.mark.parametrize("mlx_repo, mlx_name, original_name", [
    (family.MODELS[0], "language_model.model.layers.39.input_layernorm.weight",
     "model.language_model.layers.39.input_layernorm.weight"),
    (family.MODELS[0], "language_model.model.norm.weight", "model.language_model.norm.weight"),
    (family.DRAFTER, "pre_fc_norm_hidden.weight", "mtp.pre_fc_norm_hidden.weight"),
    (family.DRAFTER, "layers.0.self_attn.q_norm.weight", "mtp.layers.0.self_attn.q_norm.weight"),
])
def test_the_stored_norms_are_the_original_plus_one_rounded_to_bf16(mlx_repo, mlx_name, original_name):
    """Where the original bf16 checkpoint's shard is cached, MLX's value is bf16(w + 1): within one bf16 step."""

    stored = _read(_cached(mlx_repo), mlx_name)
    original = _read(_cached(ORIGINAL), original_name)
    assert float((stored - original).abs().min()) > 0.9                  # not w itself
    assert torch.allclose(stored, original + 1.0, rtol=2.0 ** -7, atol=0)
    # the gated DeltaNet norm is not centred: stored as in the original
    if mlx_repo == family.MODELS[0]:
        name = "layers.38.linear_attn.norm.weight"
        assert torch.equal(_read(_cached(mlx_repo), "language_model.model." + name),
                           _read(_cached(ORIGINAL), "model.language_model." + name))


# ---------------------------------------------------------------------------------------------------------------
# a small fake checkpoint through the loader

TINY = {"hidden_size": 128, "num_hidden_layers": 4, "full_attention_interval": 4, "vocab_size": 256,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64, "linear_num_key_heads": 2,
        "linear_num_value_heads": 4, "linear_key_head_dim": 32, "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4, "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64, "rms_norm_eps": 1e-6, "eos_token_id": 2, "attn_output_gate": True,
        "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000, "mrope_section": [3, 3, 2]},
        "mtp_num_hidden_layers": 1}
CENTRED = ("layernorm.weight", "model.norm.weight", "q_norm.weight", "k_norm.weight", "pre_fc_norm_embedding.weight",
           "pre_fc_norm_hidden.weight")


def _bf16_bits(x: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(x, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _bf16(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header, blobs, at = {}, [], 0
    for name, (dtype, arr) in tensors.items():
        blob = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [at, at + len(blob)]}
        blobs.append(blob)
        at += len(blob)
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))


def _fill(spec: W.Spec, rng: np.random.Generator, around_one: bool) -> dict[str, tuple[str, np.ndarray]]:
    tensors = {}
    for name, (dtype, shape) in spec.items():
        if dtype == "U32":
            arr = rng.integers(0, 2 ** 32, size=shape, dtype=np.uint64).astype(np.uint32)
        elif name.endswith(".scales"):
            arr = _bf16_bits(rng.uniform(0.001, 0.1, size=shape))
        elif name.endswith(".biases"):
            arr = _bf16_bits(rng.uniform(-0.5, 0.5, size=shape))
        elif name.endswith(CENTRED):
            arr = _bf16_bits(rng.normal(0.0, 0.1, size=shape) + (1.0 if around_one else 0.0))
        else:
            arr = _bf16_bits(rng.normal(0.0, 0.5, size=shape))
        tensors[name] = (dtype, arr)
    return tensors


def _fake_checkpoint(folder: Path, *, around_one: bool = True, extra: dict | None = None) -> tuple[Path, dict]:
    folder.mkdir(parents=True, exist_ok=True)
    quant = {"group_size": 64, "bits": 4, "mode": "affine"}
    for i in range(TINY["num_hidden_layers"]):
        for module in ("gate", "shared_expert_gate"):
            quant[f"language_model.model.layers.{i}.mlp.{module}"] = {"group_size": 64, "bits": 8}
    config = {"model_type": "qwen3_5_moe", "quantization": quant, "text_config": dict(TINY),
              "tie_word_embeddings": False}
    (folder / "config.json").write_text(json.dumps(config))
    (folder / "generation_config.json").write_text(json.dumps({"eos_token_id": [3, 2]}))
    cfg = W.Config.read(folder)
    tensors = _fill(W.layout(cfg, "language_model."), np.random.default_rng(7), around_one)
    tensors["vision_tower.patch_embed.proj.weight"] = ("BF16", _bf16_bits(np.ones((4, 4))))
    tensors.update(extra or {})
    names = sorted(tensors)
    shards = {"model-00001-of-00002.safetensors": names[: len(names) // 2],
              "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    for shard, part in shards.items():
        _write_safetensors(folder / shard, {n: tensors[n] for n in part})
    index = {"metadata": {}, "weight_map": {n: s for s, part in shards.items() for n in part}}
    (folder / "model.safetensors.index.json").write_text(json.dumps(index))
    return folder, tensors


def _dequant_reference(words: np.ndarray, scales: np.ndarray, biases: np.ndarray, bits: int) -> np.ndarray:
    """MLX affine values, independently: fields lowest bits first, s*q + b rounded once to fp32."""

    per = 32 // bits
    q = (words[..., None].astype(np.uint64) >> (np.arange(per, dtype=np.uint64) * bits)) & ((1 << bits) - 1)
    q = q.reshape(*words.shape[:-1], words.shape[-1] * per).astype(np.float64)
    s = np.repeat(_bf16(scales).astype(np.float64), 64, axis=-1)
    b = np.repeat(_bf16(biases).astype(np.float64), 64, axis=-1)
    return (q * s + b).astype(np.float32)


def _words(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.view(np.int32))


def _bits(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)


def _same(q: W.QW, tensors: dict, name: str, index=None) -> bool:
    got = [q.words, q.scales, q.biases] if index is None else [q.words[index], q.scales[index], q.biases[index]]
    want = [_words(tensors[name + ".weight"][1]), _bits(tensors[name + ".scales"][1]),
            _bits(tensors[name + ".biases"][1])]
    return all(torch.equal(g, w) for g, w in zip(got, want))


def test_the_loader_builds_the_layout_contract_from_a_fake_checkpoint(tmp_path):
    folder, t = _fake_checkpoint(tmp_path / "main")
    w = W.load(folder, device="cpu")
    cfg = w.cfg
    assert cfg.layer_types == ["linear", "linear", "linear", "attention"]   # from full_attention_interval
    assert cfg.eos == (3, 2) and cfg.rotary_dim == 16 and w.around_one
    assert w.meta["tensors_read"] == len(W.layout(cfg, "language_model.")) and w.meta["skipped"] == 1
    p = "language_model.model."
    assert _same(w.embed, t, p + "embed_tokens") and _same(w.head, t, "language_model.lm_head")
    assert w.embed.words.dtype == torch.int32 and w.embed.scales.dtype == torch.bfloat16
    half = cfg.rotary_dim // 2
    assert torch.equal(w.inv_freq, (1e7 ** (-torch.arange(half, dtype=torch.float64) / half)).float())
    for layer in w.layers:
        base = f"{p}layers.{layer.index}"
        m = layer.moe
        # the expert table: routed experts 0..3 as stored, the shared expert appended as expert 4
        assert m.gate.words.shape == (5, 64, 16) and m.down.words.shape == (5, 128, 8)
        assert m.gate.scales.shape == (5, 64, 2) and m.down.scales.shape == (5, 128, 1)
        for proj, table in (("gate_proj", m.gate), ("up_proj", m.up), ("down_proj", m.down)):
            assert table.words.is_contiguous() and (table.bits, table.group) == (4, 64)
            assert torch.equal(table.words[:4], _words(t[f"{base}.mlp.switch_mlp.{proj}.weight"][1]))
            assert torch.equal(table.scales[:4], _bits(t[f"{base}.mlp.switch_mlp.{proj}.scales"][1]))
            assert torch.equal(table.biases[:4], _bits(t[f"{base}.mlp.switch_mlp.{proj}.biases"][1]))
            assert _same(table, t, f"{base}.mlp.shared_expert.{proj}", 4)
        # the router: 8-bit rows dequantized to fp32, the shared expert's gate row last, exact against numpy
        want = np.concatenate([_dequant_reference(*(t[f"{base}.mlp.{g}.{s}"][1] for s in ("weight", "scales", "biases")),
                                                  8) for g in ("gate", "shared_expert_gate")])
        assert m.router.dtype == torch.float32 and m.router.shape == (5, 128)
        assert np.array_equal(m.router.numpy(), want)
        assert torch.equal(layer.input_scale, torch.from_numpy(_bf16(t[f"{base}.input_layernorm.weight"][1])))
        assert torch.equal(layer.post_scale, torch.from_numpy(_bf16(t[f"{base}.post_attention_layernorm.weight"][1])))
        if layer.linear:
            g = layer.gdn
            assert layer.attn is None and g.proj.words.shape == (256 + 128 + 4 + 4, 16)
            for part, name in zip(g.proj.split(cfg.gdn_rows), ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")):
                assert _same(part, t, f"{base}.linear_attn.{name}")
            assert g.conv.shape == (256, 4) and g.conv.dtype == torch.bfloat16
            assert g.a_log.dtype == torch.float32 and g.dt_bias.dtype == torch.float32
            assert torch.equal(g.norm, _bits(t[f"{base}.linear_attn.norm.weight"][1]))     # not centred: as stored
            assert _same(g.out, t, f"{base}.linear_attn.out_proj")
        else:
            a = layer.attn
            assert layer.gdn is None and a.proj.words.shape == (256 + 64 + 64, 16)
            for part, name in zip(a.proj.split(cfg.attn_rows), ("q_proj", "k_proj", "v_proj")):
                assert _same(part, t, f"{base}.self_attn.{name}")
            assert torch.equal(a.q_scale, torch.from_numpy(_bf16(t[f"{base}.self_attn.q_norm.weight"][1])))
            assert _same(a.o, t, f"{base}.self_attn.o_proj")
    assert torch.equal(w.norm, torch.from_numpy(_bf16(t[p + "norm.weight"][1])))


def test_a_checkpoint_storing_w_gets_one_added_in_fp32(tmp_path):
    folder, t = _fake_checkpoint(tmp_path / "zero", around_one=False)
    w = W.load(folder, device="cpu")
    assert not w.around_one
    stored = torch.from_numpy(_bf16(t["language_model.model.layers.3.self_attn.k_norm.weight"][1]))
    assert torch.equal(w.layers[3].attn.k_scale, stored + 1.0)
    gated = _bits(t["language_model.model.layers.0.linear_attn.norm.weight"][1])
    assert torch.equal(w.layers[0].gdn.norm, gated)                       # never shifted


def test_the_loader_refuses_a_layout_it_does_not_know(tmp_path):
    extra = {"language_model.model.layers.0.mlp.expert_bias": ("BF16", _bf16_bits(np.zeros(4)))}
    folder, _ = _fake_checkpoint(tmp_path / "extra", extra=extra)
    with pytest.raises(ValueError, match="unexpected language_model.model.layers.0.mlp.expert_bias"):
        W.load(folder, device="cpu")
    wrong = {"language_model.model.norm.weight": ("BF16", _bf16_bits(np.ones(64)))}
    folder, _ = _fake_checkpoint(tmp_path / "wrong", extra=wrong)
    with pytest.raises(ValueError, match="language_model.model.norm.weight is BF16 \\[64\\], expected BF16 \\[128\\]"):
        W.load(folder, device="cpu")


def test_the_drafter_loads_with_its_own_4bit_router(tmp_path):
    main, _ = _fake_checkpoint(tmp_path / "main")
    target = W.Config.read(main)
    folder = tmp_path / "mtp"
    folder.mkdir()
    config = {"block_size": 3, "model_type": "qwen3_5_mtp", "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
              "text_config": dict(TINY)}
    (folder / "config.json").write_text(json.dumps(config))
    t = _fill(W.mtp_layout(W.Config.read(folder)), np.random.default_rng(11), True)
    _write_safetensors(folder / "model.safetensors", t)                 # one file, no index
    mtp = W.load_mtp(folder, target, device="cpu")
    assert mtp.around_one and mtp.meta["block_size"] == 3
    assert mtp.fc.words.shape == (128, 32) and _same(mtp.fc, t, "fc")   # [hidden, 2*hidden]
    assert torch.equal(mtp.norm_e, torch.from_numpy(_bf16(t["pre_fc_norm_embedding.weight"][1])))
    assert torch.equal(mtp.norm_h, torch.from_numpy(_bf16(t["pre_fc_norm_hidden.weight"][1])))
    layer = mtp.layer
    assert not layer.linear and layer.attn is not None
    want = np.concatenate([_dequant_reference(*(t[f"layers.0.mlp.{g}.{s}"][1] for s in ("weight", "scales", "biases")), 4)
                           for g in ("gate", "shared_expert_gate")])
    assert np.array_equal(layer.moe.router.numpy(), want)
    assert _same(layer.moe.down, t, "layers.0.mlp.shared_expert.down_proj", 4)
    with pytest.raises(ValueError, match="hidden 64 \\(target 128\\)"):
        bad = dict(TINY, hidden_size=64)
        (folder / "config.json").write_text(json.dumps({**config, "text_config": bad}))
        W.load_mtp(folder, target, device="cpu")
