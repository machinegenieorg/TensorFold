"""Qwen3.6-35B-A3B's memory admission on header-only checkpoints: no GPU, but torch and triton for the byte counts."""

import json
import struct

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.cuda import capacity as cap  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda import engine as E  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.checkpoint import Config, layout, mtp_layout  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.engine import GiB, admission, cache_bytes  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.mtp import draft_token_ids  # noqa: E402
from tests.test_cuda_qwen36moe_package import _config, _drafter_config, _write  # noqa: E402

SIZES = {"U32": 4, "BF16": 2, "F32": 4}


def headers(folder, spec: dict) -> None:
    """A header-only safetensors file naming ``spec``'s tensors (all the admission reads)."""

    header, at = {}, 0
    for name, (dtype, shape) in spec.items():
        n = SIZES[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [at, at + n]}
        at += n
    raw = json.dumps(header).encode()
    (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)


def fake_checkpoint(folder, vision: bool = True):
    """The real config and header-only weights naming every tensor the loader reads (and a vision one)."""

    folder = _write(folder, _config())
    cfg = Config.read(folder)
    spec = dict(layout(cfg, "language_model."))
    if vision:
        spec["vision_tower.blocks.0.attn.qkv.weight"] = ("BF16", (3456, 1152))
    headers(folder, spec)
    return folder, cfg


def fake_drafter(folder):
    """The drafter's real config and header-only weights naming every tensor ``load_mtp`` reads."""

    folder = _write(folder, _drafter_config(), generation=None)
    headers(folder, dict(mtp_layout(Config.read(folder))))
    return folder


def test_the_default_context_shrinks_to_fit_and_an_explicit_one_is_refused(tmp_path):
    folder, cfg = fake_checkpoint(tmp_path / "main")
    weights = cap.estimate_weights(folder, E.weight_transform(cfg.hidden))
    assert 18.0 * GiB < weights.resident < 18.6 * GiB and weights.staging < 2 * GiB
    with pytest.raises(ValueError, match="cannot fit"):                  # not even the weights
        admission(folder, cfg, 32768, False, rows=512, reserve=1, free_memory=16 * GiB)
    room = weights.resident + max(weights.staging, cache_bytes(cfg, 8193, rows=512))
    with pytest.raises(ValueError, match=r"cannot fit requested context 262144.*largest fitting"):
        admission(folder, cfg, 262144, True, rows=512, reserve=1, free_memory=room)
    with pytest.raises(ValueError, match="exceeds the checkpoint's 262144-token native window"):
        admission(folder, cfg, 262145, True, rows=512, reserve=1, free_memory=room)
    got = admission(folder, cfg, 32768, False, rows=512, reserve=1, free_memory=room)      # the default shrinks
    assert 8192 <= got["context_window"] < 8300 and got["cache_slots"] == got["context_window"] + 1, got
    assert got["weight_bytes_estimate"] == weights.resident and got["largest_window"] == got["context_window"]
    got = admission(folder, cfg, 0, True, rows=512, reserve=1, free_memory=10 * room)        # 0: the model's window
    assert got["context_window"] == 262144


def test_the_admission_counts_the_mtp_head(tmp_path):
    folder, cfg = fake_checkpoint(tmp_path / "main")
    drafter = fake_drafter(tmp_path / "mtp")
    ids = draft_token_ids("default")
    rows = int(((ids >= 0) & (ids < cfg.vocab)).sum())
    assert rows == 76882
    plain = cap.estimate_weights(folder, E.weight_transform(cfg.hidden))
    head = cap.estimate_weights(folder, E.weight_transform(cfg.hidden), files=sorted(drafter.glob("*.safetensors")))
    assert 440 < head.resident / 2 ** 20 < 480 and 80 < E.draft_head_bytes(cfg, rows) / 2 ** 20 < 95
    got = admission(folder, cfg, 8192, True, rows=512, reserve=7, drafter=drafter, head_rows=rows,
                    free_memory=100 * GiB)
    assert got["cache_slots"] == 8192 + 7 and got["weight_bytes_estimate"] == plain.resident + head.resident
    assert got["cache_workspace_bytes_estimate"] == cache_bytes(cfg, 8199, rows=512, mtp_layers=1, head_rows=rows)
    alone = plain.resident + max(plain.staging, cache_bytes(cfg, 8193, rows=512))
    with pytest.raises(ValueError, match="cannot fit requested context 8192"):
        admission(folder, cfg, 8192, True, rows=512, reserve=7, drafter=drafter, head_rows=rows, free_memory=alone)
