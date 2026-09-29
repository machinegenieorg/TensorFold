"""The Qwen3 embedding engine: startup admission on the checkpoint's headers, then the packed prompt forward."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from .forward import MLP_ROWS


def token_bytes(hidden: int, heads: int, kv_heads: int, head_dim: int) -> int:
    """Bytes a packed token holds at once in a step: the fp32 residual and a projection's fp32 update, normed and
    attention rows in bf16, and the [q | k | v] projection beside its split copy. For Qwen3-Embedding-8B that is
    72 KB; the allocator's reserve measured 64 KB a token past 32K-token steps (48 KB allocated)."""

    qkv = (heads + 2 * kv_heads) * head_dim
    return 4 * hidden * 2 + 2 * hidden * 2 + 2 * qkv * 2


def geometry(text: dict, batch_tokens: int):
    """Workspace for a step of ``max(window, batch_tokens)`` tokens: an input as long as the window runs alone."""

    from tensorfold.cuda.capacity import Geometry

    hidden, inter = int(text["hidden_size"]), int(text["intermediate_size"])
    heads, kv = int(text["num_attention_heads"]), int(text["num_key_value_heads"])
    head_dim = int(text.get("head_dim") or hidden // heads)
    per_token = token_bytes(hidden, heads, kv, head_dim)
    mlp = MLP_ROWS * (3 * inter) * 2                         # [gate | up] rows and their SwiGLU
    rotary = head_dim * 4                                    # cos and sin a position, fp32

    def bytes_at(slots: int) -> int:           # a step's tokens, the MLP's row block, the rotary table, 512 MiB slack
        return per_token * max(slots, batch_tokens) + mlp + rotary * slots + 512 * 1024**2

    return Geometry(bytes_at, 0)


class Qwen3EmbedEngine:
    """One GPU: ``embed(texts)`` returns each text's pooled hidden state; ``context_window`` bounds one text."""

    def __init__(self, model_dir: Path, *, context: int | None = None, context_explicit: bool | None = None,
                 batch_tokens: int = 8192):
        import torch

        from tensorfold.cuda.capacity import admit

        from .weights import load, weight_transform

        if int(batch_tokens) < 1:
            raise ValueError("--batch-tokens must be a positive token count")
        self.torch = torch
        torch.cuda.set_device(0)
        self.batch_tokens = int(batch_tokens)
        # one admission, before any weight loads: weights as stored plus the step workspace at the window
        self.capacity_plan = admit(model_dir, context, context_explicit, torch,
                                   lambda text: geometry(text, self.batch_tokens), weight_transform(model_dir))
        self.context_window = int(self.capacity_plan["context_window"])
        started = time.perf_counter()
        self.w = load(model_dir, positions=self.context_window)
        c = self.w.config
        self.dimensions = c.hidden
        self.vocab = c.vocab
        self.eos = ()                     # nothing is decoded
        self._warm()
        print(f"[tensorfold] {c.layers} layers ({self.w.quant}, {self.w.nbytes() / 1024**3:.2f} GiB) loaded and "
              f"warmed in {time.perf_counter() - started:.1f}s; steps of up to {self.batch_tokens} tokens, texts of "
              f"up to {self.context_window}", flush=True)

    def _warm(self) -> None:
        """Build the extensions and compile every kernel shape class before the first request."""

        for lengths in ((1,), (3, 70), (17,) * 5):
            self.embed([[0] * n for n in lengths])
        self.torch.cuda.synchronize()

    def embed(self, texts: Sequence[Sequence[int]]):
        """(len(texts), hidden) fp32 numpy rows: each text's last token after the final norm, whatever the batch."""

        from .forward import embed

        self.torch.cuda.set_device(0)
        return embed(self.w, texts).cpu().numpy()
