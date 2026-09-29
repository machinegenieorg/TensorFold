"""A tiny Qwen3.6 MoE in the layout of ``nvidia/Qwen3.6-35B-A3B-NVFP4``: the real formats and names, small shapes.

Routed experts, the shared expert and ``lm_head`` are NVFP4 (packed E2M1 nibbles, e4m3 block scales, an fp32 scale
a tensor, the ``input_scale`` the loader ignores); DeltaNet's qkv, z and out and attention's projections FP8 (e4m3,
an fp32 scale a tensor); the rest bf16, the MTP layer's stacked experts included; RMSNorm weights zero-centred;
one vision tensor the loader skips. The shapes are the MLX tiny model's in ``test_qwen36_moe.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tensorfold.families.qwen4_exp.cuda import nvfp4

D, E, WIDTH, TOP, HEADS, KV, HD = 256, 16, 64, 4, 2, 1, 128
LM = "model.language_model."


def _fp4(n: int, k: int, g: torch.Generator, scale2: float = 2.0 ** -5) -> dict[str, torch.Tensor]:
    codes = torch.randint(0, 16, (n, k), generator=g)
    words = ((codes[:, 1::2] << 4) | codes[:, 0::2]).to(torch.uint8)
    scale = (torch.rand(n, k // nvfp4.GS, generator=g) * 0.8 + 0.1).to(torch.float8_e4m3fn)
    return {"weight": words, "weight_scale": scale, "weight_scale_2": torch.tensor(scale2),
            "input_scale": torch.tensor(1.0)}


def _fp8(n: int, k: int, g: torch.Generator) -> dict[str, torch.Tensor]:
    codes = (torch.randn(n, k, generator=g) * 16).clamp(-448, 448).to(torch.float8_e4m3fn)
    return {"weight": codes, "weight_scale": torch.tensor(0.002), "input_scale": torch.tensor(0.5)}


def tensors(seed: int = 11, vocab: int = 256, *, vision: bool = True) -> dict[str, torch.Tensor]:
    """The checkpoint's tensors by stored name."""

    g = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}

    def bf(*shape, scale=0.03):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    def put(name: str, parts: dict[str, torch.Tensor] | torch.Tensor) -> None:
        if isinstance(parts, torch.Tensor):
            out[name] = parts
        else:
            out.update({f"{name}.{k}": v for k, v in parts.items()})

    def centred(n):
        return bf(n, scale=0.05)

    def attention(p: str, dense: bool) -> None:
        for proj, n, k in (("q_proj", 2 * HEADS * HD, D), ("k_proj", KV * HD, D), ("v_proj", KV * HD, D),
                           ("o_proj", D, HEADS * HD)):
            put(f"{p}self_attn.{proj}", {"weight": bf(n, k)} if dense else _fp8(n, k, g))
        put(f"{p}self_attn.q_norm.weight", centred(HD))
        put(f"{p}self_attn.k_norm.weight", centred(HD))

    def mlp(p: str, stacked: bool) -> None:
        put(f"{p}mlp.gate.weight", bf(E, D, scale=0.05))
        put(f"{p}mlp.shared_expert_gate.weight", bf(1, D, scale=0.05))
        # each projection its own per-tensor scale (gate and up of one expert too), as the kernels must honour
        projs = (("gate_proj", WIDTH, D, 0), ("up_proj", WIDTH, D, 1), ("down_proj", D, WIDTH, 2))
        for proj, n, k, j in projs:
            put(f"{p}mlp.shared_expert.{proj}", {"weight": bf(n, k)} if stacked else _fp4(n, k, g, 2.0 ** -(4 + j)))
        if stacked:
            put(f"{p}mlp.experts.gate_up_proj", bf(E, 2 * WIDTH, D))
            put(f"{p}mlp.experts.down_proj", bf(E, D, WIDTH))
            return
        for e in range(E):
            for proj, n, k, j in projs:
                put(f"{p}mlp.experts.{e}.{proj}", _fp4(n, k, g, 2.0 ** -5 * (1 + (e + j) % 3)))

    put(LM + "embed_tokens.weight", bf(vocab, D))
    for i in range(2):
        p = f"{LM}layers.{i}."
        put(p + "input_layernorm.weight", centred(D))
        put(p + "post_attention_layernorm.weight", centred(D))
        if i == 0:
            a = p + "linear_attn."
            put(a + "in_proj_qkv", _fp8(384, D, g))
            put(a + "in_proj_z", _fp8(128, D, g))
            put(a + "in_proj_b.weight", bf(1, D))
            put(a + "in_proj_a.weight", bf(1, D))
            put(a + "out_proj", _fp8(D, 128, g))
            put(a + "conv1d.weight", bf(384, 1, 4, scale=0.1))
            put(a + "A_log", torch.zeros(1, dtype=torch.bfloat16))
            put(a + "dt_bias", torch.zeros(1, dtype=torch.bfloat16))
            put(a + "norm.weight", (1.0 + bf(128, scale=0.05).float()).to(torch.bfloat16))   # absolute
        else:
            attention(p, dense=False)
        mlp(p, stacked=False)
    put(LM + "norm.weight", centred(D))
    put("lm_head", _fp4(vocab, D, g, 2.0 ** -4))
    put("mtp.fc.weight", bf(D, 2 * D))
    for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
        put(f"mtp.{name}.weight", centred(D))
    put("mtp.layers.0.input_layernorm.weight", centred(D))
    put("mtp.layers.0.post_attention_layernorm.weight", centred(D))
    attention("mtp.layers.0.", dense=True)
    mlp("mtp.layers.0.", stacked=True)
    if vision:
        put("model.visual.blocks.0.attn.qkv.weight", bf(8, 8))
    return out


def config(vocab: int = 256, top_k: int = TOP) -> dict:
    fp8 = [f"{LM}layers.0.linear_attn.{p}" for p in ("in_proj_qkv", "in_proj_z", "out_proj")]
    fp8 += [f"{LM}layers.1.self_attn.{p}_proj" for p in "qkvo"]
    fp4 = ["lm_head"] + [f"{LM}layers.{i}.mlp.{p}" for i in range(2)
                         for p in ("experts", "shared_expert.gate_proj", "shared_expert.up_proj",
                                   "shared_expert.down_proj")]
    layers = {**{n: {"quant_algo": "FP8"} for n in fp8},
              **{n: {"quant_algo": "W4A16_NVFP4", "group_size": 16} for n in fp4}}
    return {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text", "hidden_size": D, "num_hidden_layers": 2, "vocab_size": vocab,
            "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": HD, "linear_num_key_heads": 1,
            "linear_num_value_heads": 1, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "full_attention_interval": 2, "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25},
            "num_experts": E, "num_experts_per_tok": top_k, "moe_intermediate_size": WIDTH,
            "shared_expert_intermediate_size": WIDTH, "mtp_num_hidden_layers": 1, "eos_token_id": 0,
            "max_position_embeddings": 65536},
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                "producer": {"name": "modelopt", "version": "0.44.0"},
                                "quantized_layers": layers, "ignore": ["mtp.layers.0*", "mtp*"]},
    }


def write(folder: Path, seed: int = 11, vocab: int = 256, top_k: int = TOP) -> Path:
    from safetensors.torch import save_file

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config(vocab, top_k)))
    parts = tensors(seed, vocab)
    names = sorted(parts)
    half = len(names) // 2                                  # two shards, the index naming each tensor's file
    shards = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    for file, chunk in shards.items():
        save_file({n: parts[n].contiguous() for n in chunk}, str(folder / file))
    weight_map = {n: f for f, chunk in shards.items() for n in chunk}
    (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return folder


_written: dict[tuple[int, int], Path] = {}


def model(seed: int = 11, vocab: int = 256):
    """(weights, MTP head) loaded by the engine's own loader from a written checkpoint (kept for the session)."""

    import tempfile

    from tensorfold.families.qwen3_5_moe.cuda.mtp import Head
    from tensorfold.families.qwen3_5_moe.cuda.weights import load, load_mtp

    folder = _written.get((seed, vocab))
    if folder is None:
        folder = _written[(seed, vocab)] = write(Path(tempfile.mkdtemp(prefix="q36nvfp4-")), seed, vocab)
    w = load(folder)
    return w, Head(w, load_mtp(folder, w))
