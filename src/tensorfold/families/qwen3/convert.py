"""Convert a bf16 Qwen3 embedding checkpoint to MLX affine 4-bit groups of 64, the layout the CUDA prompt kernels read.

    python -m tensorfold.families.qwen3.convert Qwen/Qwen3-Embedding-8B ./Qwen3-Embedding-8B-4bit

The seven projections of every layer become 4-bit words with bf16 scales and biases, as ``mlx.core.quantize``
stores them (eight codes a uint32, lowest nibble first); the token table keeps 8 bits by default (``--embed-bits``);
norms stay bf16. By default the projections are rounded with GPTQ on public calibration text (``--calibration
wikitext``: WikiText-2's training split, fetched from the Hugging Face Hub once), layer by layer through the CUDA
engine's own kernels, so this needs a GPU; ``--calibration none`` rounds each group to the nearest code without
data. Either way each group's range is shrunk to the scale with the least squared error (``--no-search``: MLX's
own ranges). Tokenizer, pooling and license files are copied unchanged, and ``tensorfold_convert.json`` records the
source revision, the settings and each tensor's relative error.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SHRINK = tuple(1.0 - 0.02 * i for i in range(11))          # ranges searched a group: 100% down to 80%
# WikiText-2 (CC BY-SA), raw, training split: public calibration text distinct from any retrieval benchmark
WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet",
            "b08601e04326c79dfdd32d625aee71d232d685c3")


def _codes_for(w, lo, hi, levels: int):
    """Scales, biases (bf16-rounded) and codes for groups spanning [lo, hi], MLX's sign convention included."""

    import torch

    scale = ((hi - lo) / levels).clamp_min(1e-7)
    low_edge = lo.abs() > hi.abs()                        # MLX keeps the larger-magnitude edge exact
    scale = torch.where(low_edge, scale, -scale)
    edge = torch.where(low_edge, lo, hi)
    q0 = torch.round(edge / scale)
    scale = torch.where(q0 != 0, edge / q0, scale)
    bias = torch.where(q0 == 0, torch.zeros_like(edge), edge)
    scale, bias = scale.to(torch.bfloat16).float(), bias.to(torch.bfloat16).float()
    codes = torch.round((w - bias) / scale).clamp_(0, levels)
    return scale, bias, codes


def quantize(weight, bits: int = 4, group: int = 64, search: bool = True, rows: int = 8192):
    """(n, k) float weight -> MLX affine (n, k * bits / 32) uint32 words, (n, k / group) bf16 scales and biases."""

    import torch

    if weight.ndim != 2 or weight.shape[1] % group or (group * bits) % 32:
        raise ValueError(f"cannot quantize a {tuple(weight.shape)} weight in groups of {group}")
    if weight.shape[0] > rows:                     # bounded workspace: rows are quantized independently
        parts = [quantize(weight[a:a + rows], bits, group, search, rows) for a in range(0, weight.shape[0], rows)]
        return tuple(torch.cat(t) for t in zip(*parts))
    n, k = weight.shape
    levels = 2 ** bits - 1
    w = weight.float().reshape(n, k // group, group)
    lo, hi = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
    best = None
    for shrink in (SHRINK if search else (1.0,)):
        scale, bias, codes = _codes_for(w, lo * shrink, hi * shrink, levels)
        err = (codes * scale + bias - w).square().sum(-1, keepdim=True)
        if best is None:
            best = [scale, bias, codes, err]
            continue
        better = err < best[3]
        best = [torch.where(better, a, b) for a, b in zip((scale, bias, codes, err), best)]
    scale, bias, codes = best[:3]
    per = 32 // bits
    codes = codes.to(torch.int64).reshape(n, k // per, per)
    shifts = torch.arange(per, device=codes.device, dtype=torch.int64) * bits
    words = (codes << shifts).sum(-1)
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    return words, scale.reshape(n, -1).to(torch.bfloat16), bias.reshape(n, -1).to(torch.bfloat16)


def dequantize(words, scales, biases, bits: int = 4, group: int = 64):
    """The float weight a quantized one stands for: code * scale + bias."""

    import torch

    per = 32 // bits
    n = words.shape[0]
    shifts = torch.arange(per, device=words.device, dtype=torch.int64) * bits
    codes = ((words.to(torch.int64) & 0xFFFFFFFF)[..., None] >> shifts) & (2 ** bits - 1)
    codes = codes.reshape(n, -1, group).float()
    return (codes * scales.float()[..., None] + biases.float()[..., None]).reshape(n, -1)


def _kind(name: str) -> str:
    base = name.removesuffix(".weight")
    if name.endswith(".weight") and base.rsplit(".", 1)[-1] in PROJECTIONS:
        return "projection"
    if base.endswith("embed_tokens") and name.endswith(".weight"):
        return "embed"
    return "keep"


def calibration_texts(spec: str, count: int, chars: int = 3000) -> list[str]:
    """``count`` texts of public prose, paragraphs joined to ``chars`` characters: WikiText-2's training split, or a
    JSONL / JSON list / text file."""

    if spec == "wikitext":
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        repo, name, revision = WIKITEXT
        rows = pq.read_table(hf_hub_download(repo, name, repo_type="dataset", revision=revision)).column("text")
        paragraphs = [t.strip() for t in rows.to_pylist() if len(t.strip()) > 400 and not t.strip().startswith("=")]
    else:
        text = Path(spec).read_text()
        if text.lstrip().startswith("["):
            paragraphs = [t for t in json.loads(text) if isinstance(t, str)]
        elif Path(spec).suffix == ".jsonl":
            paragraphs = [json.loads(line).get("text", "") for line in text.splitlines() if line.strip()]
        else:
            paragraphs = [p.strip() for p in text.split("\n\n")]
        paragraphs = [p for p in paragraphs if p.strip()]
    if not paragraphs:
        raise ValueError(f"no calibration text in {spec}")
    joined, text = [], ""
    for paragraph in paragraphs:              # consecutive paragraphs joined to a few thousand characters
        text = f"{text}\n\n{paragraph}" if text else paragraph
        if len(text) >= chars:
            joined.append(text)
            text = ""
    joined = joined or [text]
    step = max(1, len(joined) // count)
    return joined[::step][:count]


def _stacked(name: str, config: dict) -> tuple[int, str, slice] | None:
    """Where a projection's rows sit in the engine's stacked matrices: (layer, stacked name, row slice)."""

    import re

    found = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj\.weight$", name)
    if found is None:
        return None
    t = config.get("text_config") or config
    heads, kv, inter = int(t["num_attention_heads"]), int(t["num_key_value_heads"]), int(t["intermediate_size"])
    d = int(t.get("head_dim") or t["hidden_size"] // heads)
    rows = {"q": ("qkv", 0, heads * d), "k": ("qkv", heads * d, (heads + kv) * d),
            "v": ("qkv", (heads + kv) * d, (heads + 2 * kv) * d), "o": ("o", 0, None),
            "gate": ("gate_up", 0, inter), "up": ("gate_up", inter, 2 * inter), "down": ("down", 0, None)}
    stacked, a, b = rows[found.group(2)]
    return int(found.group(1)), stacked, slice(a, b)


def gptq_weights(source: Path, texts: list[str], tokens: int, *, bits: int, group: int, search: bool,
                 act_order: bool, log) -> dict:
    """{(layer, stacked name): (words, scales, biases)} from GPTQ over ``texts`` cut to ``tokens`` tokens each."""

    import torch
    from tokenizers import Tokenizer

    from .cuda.weights import load
    from .gptq import quantize_layers

    if not torch.cuda.is_available():
        raise ValueError("GPTQ runs the CUDA engine's kernels: convert on an NVIDIA GPU, or pass --calibration none")
    tok = Tokenizer.from_file(str(source / "tokenizer.json"))
    ids = []
    for text in texts:
        words = tok.encode(text, add_special_tokens=False)
        words.truncate(tokens - 1)
        ids.append(tok.post_process(words).ids)
    log(f"[tensorfold] GPTQ on {len(ids)} calibration texts, {sum(map(len, ids))} tokens")
    w = load(source, positions=tokens)
    try:
        return quantize_layers(w, ids, bits=bits, group=group, search=search, act_order=act_order, log=log)
    finally:
        del w
        torch.cuda.empty_cache()


def convert(source: Path, target: Path, *, bits: int = 4, group: int = 64, embed_bits: int = 8,
            search: bool = True, device: str | None = None, calibration: str = "none", texts: int = 128,
            tokens: int = 512, act_order: bool = True, log=print) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if bits != 4 or group not in (32, 64):
        raise ValueError("the CUDA prompt kernels read 4-bit weights in groups of 32 or 64")
    if embed_bits not in (4, 8, 16):
        raise ValueError("--embed-bits is 4, 8 or 16 (unquantized)")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = json.loads((source / "config.json").read_text())
    if (config.get("text_config") or config).get("model_type", config.get("model_type")) != "qwen3":
        raise ValueError("this converter reads Qwen3 (model_type qwen3) checkpoints")
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError("the source checkpoint is already quantized; convert the bf16 release")
    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors in {source}")
    solved = {}
    if calibration != "none":
        solved = gptq_weights(source, calibration_texts(calibration, texts, chars=tokens * 6), tokens, bits=bits,
                              group=group, search=search, act_order=act_order, log=log)
    target.mkdir(parents=True, exist_ok=True)
    weight_map, report = {}, {}
    total = 0
    for shard in shards:
        out = {}
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                if name.startswith(("lm_head.", "model.lm_head.")):
                    continue                    # an embedding model never reads a head
                tensor = f.get_tensor(name)
                kind = _kind(name)
                width = bits if kind == "projection" else embed_bits if kind == "embed" else 16
                size = group if kind == "projection" else 64          # the token table keeps groups of 64
                if width == 16:
                    out[name] = tensor.contiguous()
                    continue
                w = tensor.to(device)
                where = _stacked(name, config) if kind == "projection" and solved else None
                if where is not None:
                    words, scales, biases = (t[where[2]].to(device) for t in solved[where[:2]])
                else:
                    words, scales, biases = quantize(w, width, size, search)
                back = dequantize(words, scales, biases, width, size)
                rel = float((back - w.float()).norm() / w.float().norm())
                report[name.removesuffix(".weight")] = {"bits": width, "relative_error": round(rel, 6)}
                base = name.removesuffix(".weight")
                out[name] = words.cpu().view(torch.uint32) if hasattr(torch, "uint32") else words.cpu()
                out[base + ".scales"] = scales.cpu()
                out[base + ".biases"] = biases.cpu()
                del w, back
        path = target / shard.name
        save_file(out, str(path), metadata={"format": "mlx"})
        weight_map.update({name: shard.name for name in out})
        total += sum(t.numel() * t.element_size() for t in out.values())
        log(f"[tensorfold] {shard.name}: {len(out)} tensors")
    (target / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=2))
    quant = {"group_size": group, "bits": bits, "mode": "affine"}
    embed_name = next((n.removesuffix(".weight") for n in weight_map if _kind(n) == "embed"), "embed_tokens")
    quant[embed_name] = False if embed_bits == 16 else {"group_size": 64, "bits": embed_bits}
    config["quantization"] = quant
    config["quantization_config"] = dict(quant)
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for item in source.iterdir():                 # tokenizer, pooling, prompts, license, model card
        if item.name in ("config.json", "model.safetensors.index.json") or item.suffix == ".safetensors":
            continue
        dest = target / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    errors = [v["relative_error"] for k, v in report.items() if v["bits"] == bits]
    receipt = {"source": str(source), "source_revision": source.name if len(source.name) == 40 else None,
               "bits": bits, "group_size": group, "embed_bits": embed_bits, "range_search": search,
               "method": "gptq" if solved else "nearest",
               "calibration": None if not solved else {"text": calibration if calibration != "wikitext" else
                                                       "/".join(WIKITEXT), "texts": texts, "tokens_each": tokens,
                                                       "act_order": act_order},
               "shrink_candidates": list(SHRINK) if search else [1.0], "bytes": total,
               "projection_relative_error": {"mean": sum(errors) / len(errors), "max": max(errors)},
               "tensors": report}
    (target / "tensorfold_convert.json").write_text(json.dumps(receipt, indent=2) + "\n")
    log(f"[tensorfold] wrote {target}: {total / 1024**3:.2f} GiB, projections' relative error mean "
        f"{receipt['projection_relative_error']['mean']:.4f}, max {receipt['projection_relative_error']['max']:.4f}")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tensorfold.families.qwen3.convert",
                                     description=__doc__.split("\n")[0])
    parser.add_argument("source", help="the bf16 checkpoint: a directory or a Hugging Face repo id")
    parser.add_argument("target", help="the directory to write")
    parser.add_argument("--group-size", type=int, default=64, choices=(32, 64),
                        help="inputs a scale and bias cover in the projections (32: closer to bf16, 0.5 GB larger)")
    parser.add_argument("--embed-bits", type=int, default=8, choices=(4, 8, 16),
                        help="bits for the token table (16: keep it bf16)")
    parser.add_argument("--search", action=argparse.BooleanOptionalAction, default=True,
                        help="choose each group's range by least squared error (default); --no-search rounds as MLX")
    parser.add_argument("--calibration", default="wikitext",
                        help="GPTQ calibration text: wikitext (public, fetched once), a JSONL/JSON/text file, or none "
                             "for round-to-nearest without data")
    parser.add_argument("--calibration-texts", type=int, default=128, help="calibration paragraphs")
    parser.add_argument("--calibration-tokens", type=int, default=512, help="tokens kept from each text")
    parser.add_argument("--act-order", action=argparse.BooleanOptionalAction, default=True,
                        help="GPTQ rounds the inputs with the largest activations first (groups keep their order)")
    parser.add_argument("--device", default=None, help="torch device for the arithmetic (default: cuda if present)")
    args = parser.parse_args(argv)
    source = Path(args.source).expanduser()
    if not source.is_dir():
        from tensorfold import hub

        source = Path(hub.resolve(args.source))
    convert(source, Path(args.target).expanduser(), group=args.group_size, embed_bits=args.embed_bits,
            search=args.search, device=args.device,
            calibration=args.calibration, texts=args.calibration_texts, tokens=args.calibration_tokens,
            act_order=args.act_order)
    return 0


if __name__ == "__main__":
    sys.exit(main())
