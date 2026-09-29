"""The Qwen3 embedding family without a GPU: config checks, admission before any weight loads, the 4-bit converter."""

import argparse
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold import cli, families, serve_options
from tensorfold.families import qwen3

CONFIG = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 128, "intermediate_size": 256,
          "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
          "vocab_size": 100, "max_position_embeddings": 4096, "rms_norm_eps": 1e-6, "rope_theta": 1000000,
          "attention_bias": False, "rope_scaling": None, "use_sliding_window": False, "tie_word_embeddings": False}


def model_dir(path, config=None, pooling=True, tensors=None):
    (path / "config.json").write_text(json.dumps(config or CONFIG))
    if pooling:
        (path / "1_Pooling").mkdir(exist_ok=True)
        (path / "1_Pooling" / "config.json").write_text(json.dumps({"pooling_mode_lasttoken": True}))
    entries, offset = {}, 0
    for name, dtype, shape, size in tensors or [("embed_tokens.weight", "BF16", [100, 128], 100 * 128 * 2)]:
        entries[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(entries).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


def test_model_type_qwen3_finds_the_embedding_family_on_cuda_only(tmp_path):
    family = families.detect(model_dir(tmp_path))
    assert family.package is qwen3 and families.backends_of(family) == ("cuda",)
    with pytest.raises(ValueError, match="NVIDIA GPUs only"):
        cli._backend("mlx", family)
    families.require_readable(family, CONFIG, "cuda")                        # bf16 as shipped


@pytest.mark.parametrize("change, words", [
    ({"attention_bias": True}, "attention biases"), ({"rope_scaling": {"rope_type": "yarn"}}, "scaled rotary"),
    ({"use_sliding_window": True}, "sliding-window"), ({"head_dim": 96}, "head width 96"),
    ({"intermediate_size": 250}, "multiples of 64"), ({"num_key_value_heads": 3}, "multiple of the key/value"),
])
def test_config_refusals_before_any_download(tmp_path, change, words):
    with pytest.raises(ValueError, match=words):
        qwen3.check(model_dir(tmp_path, {**CONFIG, **change}))


@pytest.mark.parametrize("quant, ok", [
    ({"bits": 4, "group_size": 64}, True),
    ({"bits": 4, "group_size": 64, "embed_tokens": {"bits": 8, "group_size": 64}}, True),
    ({"bits": 4, "group_size": 64, "model.embed_tokens": False}, True),
    ({"bits": 4, "group_size": 32}, True),
    ({"bits": 4, "group_size": 32, "embed_tokens": {"bits": 8, "group_size": 32}}, False),
    ({"bits": 3, "group_size": 64}, False), ({"bits": 4, "group_size": 128}, False),
    ({"bits": 4, "group_size": 64, "layers.0.mlp.down_proj": {"bits": 8, "group_size": 64}}, False),
    ({"bits": 4, "group_size": 64, "layers.0.mlp.down_proj": False}, False),
])
def test_cuda_reads_4bit_projections_in_groups_of_32_or_64(quant, ok):
    family = families.families()["qwen3"]
    config = {**CONFIG, "quantization": quant}
    if ok:
        families.require_readable(family, config, "cuda")
    else:
        with pytest.raises(ValueError, match="4-bit weights in groups of 64 or 32"):
            families.require_readable(family, config, "cuda")


def test_pooling_must_be_the_last_token(tmp_path):
    with pytest.raises(ValueError, match="no 1_Pooling"):
        qwen3.pooling(model_dir(tmp_path, pooling=False))
    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "1_Pooling" / "config.json").write_text(json.dumps({"pooling_mode_mean_tokens": True}))
    with pytest.raises(ValueError, match="pools by mean_tokens"):
        qwen3.pooling(tmp_path)


@pytest.mark.parametrize("options, words", [
    ({"drafter": "/models/draft"}, "draft model"), ({"tp": 2}, "one GPU"), ({"parallel": 4}, "--batch-tokens"),
    ({"vision": True}, "embeds text"),
])
def test_engine_options_that_do_not_apply_are_refused(tmp_path, options, words):
    with pytest.raises(ValueError, match=words):
        qwen3.cuda_engine(model_dir(tmp_path), **options)


def test_batch_tokens_is_an_embedding_option(tmp_path):
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--batch-tokens", "4096", "--alias", "a",
                                          "--alias", "b"])
    assert args.batch_tokens == 4096 and args.alias == ["a", "b"]
    serve_options.check(args, families.families()["qwen3"], "cuda")
    with pytest.raises(ValueError, match="does not embed"):
        serve_options.check(args, families.families()["qwen3_5"], "cuda")
    args.batch_tokens = 0
    with pytest.raises(ValueError, match="positive"):
        serve_options.check(args, families.families()["qwen3"], "cuda")


def test_cuda_serve_passes_batch_tokens_and_aliases(tmp_path, monkeypatch):
    from tensorfold.cuda import server

    seen = {}
    engine = SimpleNamespace(embed=lambda texts: None, dimensions=128, context_window=512)

    def cuda_engine(path, **options):
        seen.update(options)
        return engine

    family = SimpleNamespace(title="Test", model_type="qwen3", package=SimpleNamespace(cuda_engine=cuda_engine))
    made = []

    class App:
        aliases = ()

        def __init__(self, *args, **kwargs):
            made.append(self)
            self.served = args[2]
            self.effective_context_window = 512

        @property
        def model_ids(self):
            return [self.served, *self.aliases]

    monkeypatch.setattr(server, "App", App)
    monkeypatch.setattr(server, "serve", lambda *a: None)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--batch-tokens", "2048",
                                          "--name", "qwen3-embed-8b", "--alias", "nv-embed-v2"])
    assert cli._serve_cuda(args, family, tmp_path, 4096) == 0
    assert seen["batch_tokens"] == 2048 and made[0].model_ids == ["qwen3-embed-8b", "nv-embed-v2"]


def test_a_window_that_cannot_fit_is_refused_before_weights_load(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from tensorfold.families.qwen3.cuda import engine, weights

    loads = []
    monkeypatch.setattr(weights, "load", lambda *a, **k: loads.append(a) or pytest.fail("weights loaded"))
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (12 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: SimpleNamespace(is_integrated=False))
    big = [("layers.0.mlp.up_proj.weight", "BF16", [3584, 1024 * 1024], 7 * 1024**3)]
    path = model_dir(tmp_path, tensors=big)
    with pytest.raises(ValueError, match="cannot fit"):
        engine.Qwen3EmbedEngine(path, context=4096, context_explicit=True)
    assert loads == []


def test_the_step_workspace_grows_with_the_window_past_the_batch():
    pytest.importorskip("torch")
    from tensorfold.families.qwen3.cuda.engine import geometry

    g = geometry(CONFIG, 1024)
    assert g.needed(16) < g.needed(1024) < g.needed(4096)
    assert g.needed(4096) - g.needed(1024) > 10 * (g.needed(1024) - g.needed(16))    # below the batch: rotary rows only


def test_quantize_writes_mlx_affine_words_that_round_trip():
    torch = pytest.importorskip("torch")
    from tensorfold.families.qwen3.convert import dequantize, quantize

    torch.manual_seed(0)
    w = torch.randn(96, 256) * 0.02
    w[3, 5] = 0.4                                      # an outlier: its group's edge stays exact
    for search in (False, True):
        words, scales, biases = quantize(w, 4, 64, search, rows=40)
        assert words.shape == (96, 32) and words.dtype == torch.int32 and scales.shape == biases.shape == (96, 4)
        back = dequantize(words, scales, biases, 4, 64)
        assert (back - w).abs().max() < 0.5 / 15 * 0.51 + 1e-3
        if not search:                                 # MLX's ranges keep the larger edge exact (to bf16)
            assert torch.isclose(back[3, 5], torch.tensor(0.4), atol=2e-3)
        codes = (words.to(torch.int64)[..., None] >> torch.arange(0, 32, 4)) & 15   # lowest nibble first
        assert codes.max() <= 15
    rtn = dequantize(*quantize(w, 4, 64, False), 4, 64)
    best = dequantize(*quantize(w, 4, 64, True), 4, 64)
    assert (best - w).square().sum() <= (rtn - w).square().sum()
    words8, s8, b8 = quantize(w, 8, 64)
    assert words8.shape == (96, 64) and (dequantize(words8, s8, b8, 8, 64) - w).abs().max() < 2e-3


def test_convert_writes_a_checkpoint_the_cuda_family_reads(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors import safe_open
    from safetensors.torch import save_file

    from tensorfold.families.qwen3.convert import main

    source = tmp_path / "bf16"
    source.mkdir()
    model_dir(source)
    (source / "model.safetensors").unlink()
    (source / "tokenizer.json").write_text("{}")
    torch.manual_seed(1)
    c = CONFIG
    tensors = {"embed_tokens.weight": torch.randn(c["vocab_size"], 128), "norm.weight": torch.ones(128)}
    q, kv = 4 * 64, 2 * 64
    for i in range(2):
        p = f"layers.{i}."
        shapes = {"input_layernorm": (128,), "post_attention_layernorm": (128,), "self_attn.q_proj": (q, 128),
                  "self_attn.k_proj": (kv, 128), "self_attn.v_proj": (kv, 128), "self_attn.o_proj": (128, q),
                  "self_attn.q_norm": (64,), "self_attn.k_norm": (64,), "mlp.gate_proj": (256, 128),
                  "mlp.up_proj": (256, 128), "mlp.down_proj": (128, 256)}
        tensors.update({f"{p}{name}.weight": torch.ones(shape) if len(shape) == 1 else torch.randn(*shape)
                        for name, shape in shapes.items()})
    save_file({k: (v * 0.02).to(torch.bfloat16) for k, v in tensors.items()}, str(source / "model.safetensors"))
    target = tmp_path / "q4"
    assert main([str(source), str(target), "--device", "cpu", "--calibration", "none"]) == 0
    config = json.loads((target / "config.json").read_text())
    assert config["quantization"] == {"group_size": 64, "bits": 4, "mode": "affine",
                                      "embed_tokens": {"group_size": 64, "bits": 8}}
    families.require_readable(families.families()["qwen3"], config, "cuda")
    qwen3.pooling(target)
    assert (target / "tokenizer.json").exists() and (target / "tensorfold_convert.json").exists()
    with safe_open(str(target / "model.safetensors"), framework="pt") as f:
        names = set(f.keys())
        assert f.get_tensor("layers.0.mlp.down_proj.weight").shape == (128, 32)
        assert f.get_tensor("norm.weight").dtype == torch.bfloat16
    assert {"layers.1.self_attn.q_proj.scales", "layers.1.self_attn.q_proj.biases", "embed_tokens.scales"} <= names
    with pytest.raises(ValueError, match="already quantized"):
        main([str(target), str(tmp_path / "again"), "--device", "cpu", "--calibration", "none"])


@pytest.mark.parametrize("act_order", [False, True])
def test_gptq_lowers_the_output_error_of_rounding_to_nearest(act_order):
    torch = pytest.importorskip("torch")
    from tensorfold.families.qwen3.convert import dequantize, quantize
    from tensorfold.families.qwen3.gptq import Hessian, gptq, pack_codes

    g = torch.Generator().manual_seed(3)
    w = torch.randn(48, 256, generator=g) * 0.05
    x = torch.randn(2048, 64, generator=g) @ torch.randn(64, 256, generator=g)      # correlated inputs
    x[:, 7] *= 20                                                                     # and one loud channel
    hess = Hessian(256, "cpu")
    hess.add(x, chunk=500)
    codes, scales, biases = gptq(w, hess.value(), act_order=act_order)
    assert codes.min() >= 0 and codes.max() <= 15 and scales.shape == biases.shape == (48, 4)
    words = pack_codes(codes)
    ours = dequantize(words, scales.to(torch.bfloat16), biases.to(torch.bfloat16))
    nearest = dequantize(*quantize(w, 4, 64, True))
    error = lambda q: float(((q - w) @ x.t()).square().sum())
    assert error(ours) < 0.8 * error(nearest)
