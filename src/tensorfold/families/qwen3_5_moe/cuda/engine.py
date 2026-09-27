"""The Qwen3.6-35B-A3B CUDA engine behind ``tensorfold.cuda.server``: one GPU (one DGX Spark), one stream.

One request decodes at a time, MTP-drafted when the drafter is present (``decode.mtp_decode``): a round verifies the
pending token and up to ``depth`` chained drafts (6; a chain keeps its first draft and ends before a later one the
head gives less than ``confidence``, 0.5) in one window, and keeps the rows up to the first draft the keyed sampler
disagrees with. Drafted output is token for token serial decoding's (``decode.serial_decode``). Windows of up to
``depth + 1`` rows and the head's steps replay CUDA graphs captured at start (``graphs.py``), with eager's bits.
Without the drafter (or with ``no_drafts``) every round decodes one token.

Memory is checked before anything is loaded (``memory_plan``): the weights (about 18.3 GiB) and the MTP head with its
draft head (about 0.54 GiB), two sequence states (GDN states, conv windows and key/value caches for ``capacity``
positions each, the head's included), the kept snapshots and the window buffers, against the device's free memory. On a GPU that shares the host's memory (GB10) that is the larger of CUDA's
free memory and the kernel's MemAvailable, which counts page cache the kernel gives back on demand.

Prefix reuse: the engine keeps the state after the last request's prompt and after its reply (``State.snapshot``,
with their cache rows in place in the one sequence state), and a prompt that extends either resumes from it. Rows
never depend on their chunk, so a resumed prompt ends in the state and logits of a fresh prefill. A fresh prompt
starts over. A request with ``draft=False`` decodes one token a round from a fresh prefill in a second state and
leaves the kept states as they are: the serial reference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

GiB = 1 << 30
RESERVE = 1 << 30          # allocator slack, Triton's workspace, sampling's temporaries
SNAPSHOTS = 3              # kept for prefix reuse: the prompt's, the reply's and the one being taken
STATES = 2                 # the kept sequence and the serial switch's
WINDOW_ROWS, ATTN_ROWS, LOGIT_ROWS = 32, 64, 32
MAX_DEPTH = 15             # MTP drafts a round at most (the package's MAX_DRAFTS): windows of up to 16 rows
_BYTES = {"U32": 4, "I32": 4, "F32": 4, "BF16": 2, "F16": 2}


# -- memory, before anything is loaded ---------------------------------------------------------------------------
@dataclass
class MemoryPlan:
    """What the engine will hold on the GPU, in bytes."""

    weights: int              # the model in the kernels' layout (the router tables in fp32)
    load_peak: int            # during the load, one layer (or the head) held twice while it is regrouped
    states: int               # every sequence state: GDN states (two buffers), conv windows, key/value caches
    kept: int                 # the kept snapshots (GDN states and conv windows; cache rows stay in the state)
    buffers: int              # window scratch at the engine's rows, the attention partials at the capacity
    reserve: int = RESERVE

    @property
    def total(self) -> int:
        return self.weights + self.load_peak + self.states + self.kept + self.buffers + self.reserve

    def describe(self) -> str:
        parts = ("weights", "load_peak", "states", "kept", "buffers", "reserve")
        return ", ".join(f"{p.replace('_', ' ')} {getattr(self, p) / GiB:.2f}" for p in parts) + " GiB"


def _spec_bytes(spec: dict) -> int:
    total = 0
    for dtype, shape in spec.values():
        n = 1
        for s in shape:
            n *= s
        total += n * _BYTES[dtype]
    return total


def weight_bytes(cfg) -> tuple[int, int]:
    """(the model's bytes on the GPU, the largest single layer or head) from the loader's layout alone: MLX's packing
    as stored (4-bit words, bf16 scales and biases), each layer's router table widened to fp32 [experts + 1, hidden]."""

    from .weights import layout

    spec = layout(cfg, "")
    total = _spec_bytes(spec) + cfg.layers * (cfg.experts + 1) * cfg.hidden * 4
    largest = 0
    for i in range(cfg.layers):
        base = f"model.layers.{i}."
        largest = max(largest, _spec_bytes({k: v for k, v in spec.items() if k.startswith(base)}))
    head = _spec_bytes({k: v for k, v in spec.items() if k.startswith("lm_head")})
    return total, max(largest, head)


def state_bytes(cfg, capacity: int, mtp_layers: int = 0) -> tuple[int, int]:
    """(one sequence state of ``capacity`` positions, one snapshot of it), as ``forward.Pool`` and
    ``State.snapshot`` allocate them."""

    nl = sum(1 for k in cfg.layer_types if k == "linear")
    na = cfg.layers - nl
    rec = nl * cfg.nv * cfg.dv * cfg.dk * 4
    conv = nl * (cfg.conv_kernel - 1) * cfg.conv_dim * 2
    kv = 2 * (na + mtp_layers) * capacity * cfg.kv_heads * cfg.head_dim * 2
    tail = cfg.hidden * 2 if mtp_layers else 0      # the hidden row the head has not absorbed yet
    return 2 * rec + conv + kv + tail, rec + conv + tail


def buffer_bytes(cfg, capacity: int, rows: int, *, window_rows: int = WINDOW_ROWS, attn_rows: int = ATTN_ROWS,
                 logit_rows: int = LOGIT_ROWS) -> int:
    """``forward.Buffers`` for one sequence, from its allocations (the small ones rounded up)."""

    from . import qmm

    d, e, k, width = cfg.hidden, cfg.experts + 1, cfg.top_k + 1, cfg.moe_width
    nl = sum(1 for t in cfg.layer_types if t == "linear")
    conv = cfg.conv_dim
    pw = conv + cfg.nv * cfg.dv + 2 * cfg.nv
    wr, ar = min(window_rows, rows), min(attn_rows, rows)
    qd = cfg.heads * cfg.head_dim
    total = rows * d * 2 * 3 + rows * (d // 64) * 4                       # h, normed, branch; group sums
    total += max(nl * wr, rows) * pw * 2                                   # GDN projection rows
    total += nl * wr * (cfg.nk * cfg.dk * 4 + cfg.nv * cfg.dv * 2 + 2 * cfg.nv * 4)   # replay inputs
    total += nl * (cfg.conv_kernel - 1) * conv * 2                         # next conv windows
    total += rows * cfg.nv * cfg.dv * (2 + 4 / 64)                         # GDN output and its sums
    total += rows * sum(cfg.attn_rows) * 2 + rows * qd * (2 + 2 + 4 / 64)   # attention projection, q, gated
    nch = -(-capacity // 512)
    total += ar * nch * cfg.heads * (cfg.head_dim + 2) * 4 + ar * qd * 2 + ar * (2048 + 8) * 4   # attention partials
    total += rows * (e * 4 + k * 8 + k * width * (2 + 4 / 32) + k * d * 4)  # router logits, picks, act, y
    total += (min(rows * k, e) + 1) * (rows + 1) * 4                       # the expert grouping
    total += 4 * qmm.split_scratch(rows, [(cfg.vocab, d), (sum(cfg.gdn_rows), d), (d, cfg.nv * cfg.dv),
                                          (sum(cfg.attn_rows), d), (d, qd)])        # split-K partials
    total += max(logit_rows, 1) * cfg.vocab * 2 + d * 2 * 2               # logits, last rows
    return int(total * 1.05) + (4 << 20)


def drafter_bytes(drafter_dir: str | Path, cfg, draft_vocab: str | None = "default") -> tuple[int, int, int]:
    """(the MTP head in the kernels' layout with its draft head, the load's transient, the draft head's rows) from the
    drafter's layout and the draft vocabulary: its stored weights, its router table in fp32, the model's head rows at
    the draft vocabulary's ids. While ``mtp.prepare_mtp`` runs, the loaded drafter and the model's head in MLX's
    layout (from which the draft head is cut) sit beside it."""

    from .mtp import draft_token_ids
    from .weights import Config, mtp_layout

    stored = _spec_bytes(mtp_layout(Config.read(drafter_dir)))
    ids = draft_token_ids(draft_vocab)
    rows = cfg.vocab if ids is None else int(((ids >= 0) & (ids < cfg.vocab)).sum())
    row = cfg.hidden // 2 + cfg.hidden // 64 * 4
    return stored + (cfg.experts + 1) * cfg.hidden * 4 + rows * row, stored + cfg.vocab * row, rows


def mtp_buffer_bytes(cfg, capacity: int, rows: int, head_rows: int) -> int:
    """``mtp.MTPBuffers``: steps of up to ``rows`` rows, only the last row past the cache write."""

    from . import qmm

    d, e, k, width = cfg.hidden, cfg.experts + 1, cfg.top_k + 1, cfg.moe_width
    qd = cfg.heads * cfg.head_dim
    total = rows * d * 2 * 6 + rows * (3 * d // 64) * 4 + rows * (sum(cfg.attn_rows) + qd) * 2   # rows, pa, q
    total += -(-capacity // 512) * cfg.heads * (cfg.head_dim + 2) * 4 + qd * 4 + (2048 + 8) * 4 + d * 8   # one row
    total += e * 4 + k * 8 + k * width * (2 + 4 / 32) + k * d * 4 + (k + 1) * 2 * 4                     # its MoE
    total += 4 * qmm.split_scratch(rows, [(d, 2 * d), (sum(cfg.attn_rows), d), (d, qd), (head_rows, d)])
    total += head_rows * 2
    return int(total * 1.05) + (1 << 20)


def memory_plan(cfg, capacity: int, *, rows: int, states: int = STATES, mtp_layers: int = 0,
                drafter: tuple[int, int, int] | None = None, mtp_rows: int = 64) -> MemoryPlan:
    """What the engine holds at ``capacity`` positions with ``rows``-row prefill chunks; ``drafter``: what
    ``drafter_bytes`` gives (the head's cache counts in the states with ``mtp_layers=1``)."""

    weights, peak = weight_bytes(cfg)
    one, snap = state_bytes(cfg, capacity, mtp_layers)
    buffers = buffer_bytes(cfg, capacity, rows)
    head, head_peak, head_rows = drafter or (0, 0, cfg.vocab)
    if mtp_layers:
        buffers += mtp_buffer_bytes(cfg, capacity, mtp_rows, head_rows)
    return MemoryPlan(weights + head, max(peak, head_peak), states * one, SNAPSHOTS * snap, buffers)


def _mem_available() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def device_free_memory() -> tuple[int, str]:
    """(bytes the engine may still allocate, where the number came from). A discrete GPU: CUDA's free memory. A GPU
    that shares the host's memory (``is_integrated``, GB10): the larger of that and /proc/meminfo's MemAvailable,
    since CUDA does not count the page cache the kernel reclaims on demand."""

    import torch

    free, _ = torch.cuda.mem_get_info()
    if getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "is_integrated", False):
        avail = _mem_available()
        if avail is not None and avail > free:
            return avail, "MemAvailable (unified memory)"
    return free, "CUDA free memory"


# -- the engine ----------------------------------------------------------------------------------------------------
class Qwen36Engine:
    """``eos`` and ``generate`` as ``tensorfold.cuda.server`` expects, on one GPU.

    ``capacity``: prompt plus reply tokens a request may hold (the caches' positions). ``drafter``: the MTP drafter's
    directory; ``mtp_drafts``: most drafts a round (default ``decode.DEPTH``; 0 or ``no_drafts``: one token a round);
    ``confidence``: the chain's stop (default ``decode.CONFIDENCE``). ``graphs``: decode windows replay CUDA graphs.
    ``free_memory``: bytes to plan against instead of reading the device. ``model`` and ``mtp``: an already prepared
    ``forward.Model`` and MTP head (tests), which skip the loads.
    """

    def __init__(self, model_dir: str | Path | None, drafter: str = "", *, capacity: int | None = None,
                 no_drafts: bool = False, mtp_drafts: int | None = None, confidence: float | None = None,
                 rows: int | None = None, prefill_chunk: int | None = None, tp: int = 1,
                 free_memory: int | None = None, model=None, mtp=None, graphs: bool = True, warm: bool = True) -> None:
        if int(tp) != 1:
            raise ValueError("Qwen3.6-35B-A3B's CUDA engine runs on one GPU: two ranks do not apply")
        if mtp_drafts is not None and not 0 <= int(mtp_drafts) <= MAX_DEPTH:
            raise ValueError(f"MTP drafts a round: 0 to {MAX_DEPTH}, not {mtp_drafts}")
        if mtp_drafts and not drafter and mtp is None and not no_drafts:
            raise ValueError("MTP drafts come from the drafter (mlx-community/Qwen3.6-35B-A3B-MTP-4bit): pass its "
                             "directory, or no_drafts")
        import gc

        import torch

        from .. import CONTEXT
        from .decode import CONFIDENCE, DEPTH, GRAPH_ROWS, MTP_ROWS, PREFILL_ROWS, Decoder
        from .weights import Config

        started = time.perf_counter()
        torch.cuda.set_device(0)
        cfg = model.cfg if model is not None else Config.read(model_dir)
        self.capacity = int(capacity or CONTEXT)
        if not 0 < self.capacity <= cfg.max_position:
            raise ValueError(f"a context of {self.capacity} tokens: the model reads 1 to {cfg.max_position}")
        rows = int(rows or PREFILL_ROWS)
        self.chunk = prefill_chunk
        has_head = mtp is not None or bool(drafter)
        self.depth = 0 if no_drafts or not has_head else DEPTH if mtp_drafts is None else int(mtp_drafts)
        self.confidence = CONFIDENCE if confidence is None else float(confidence)
        self.drafter = str(drafter) if self.depth and mtp is None else ""
        head = None
        if self.depth and mtp is None:
            head = drafter_bytes(self.drafter, cfg)
        elif self.depth:
            head = (0, 0, mtp.head.n)
        self.plan = memory_plan(cfg, self.capacity, rows=rows, mtp_layers=1 if self.depth else 0, drafter=head,
                                mtp_rows=MTP_ROWS)
        if model is not None:                 # already on the GPU
            self.plan.weights = self.plan.load_peak = 0
        free, source = (int(free_memory), "given") if free_memory is not None else device_free_memory()
        if self.plan.total > free:
            raise ValueError(f"Qwen3.6-35B-A3B at a {self.capacity}-token context needs about "
                             f"{self.plan.total / GiB:.1f} GiB of GPU memory ({self.plan.describe()}); "
                             f"{free / GiB:.1f} GiB is free ({source}). Free memory, or serve a shorter --context")
        if model is None:
            from .forward import prepare
            from .weights import load

            w = load(model_dir, "cuda")
            model = prepare(w, release=True)
            del w
            gc.collect()
            torch.cuda.empty_cache()
        if self.depth and mtp is None:
            from .mtp import prepare_mtp
            from .weights import load_mtp

            mtpw = load_mtp(self.drafter, model.cfg, "cuda")
            mtp = prepare_mtp(mtpw, model, draft_vocab="default")
            del mtpw
            gc.collect()
            torch.cuda.empty_cache()
        loaded_s = time.perf_counter() - started
        wide = self.depth + 1
        self.e = Decoder(model, capacity=self.capacity, rows=rows, window_rows=max(WINDOW_ROWS, wide),
                         attn_rows=ATTN_ROWS, logit_rows=max(LOGIT_ROWS, wide), states=STATES,
                         mtp=mtp if self.depth else None, mtp_rows=MTP_ROWS, graphs=graphs,
                         graph_rows=max(GRAPH_ROWS, wide))
        self.serial = self.e.pool.alloc()             # the serial switch's state; self.e.st keeps the prefixes
        self.cache: list[tuple[list[int], dict]] = []  # (committed ids, State.snapshot of them)
        captured, warm_s = self._warm() if warm and self.capacity >= 128 else (0, 0.0)
        if self.depth:
            mode = (f"1 to {self.depth} MTP drafts a round, a chain stops before a later draft under "
                    f"{self.confidence:.0%} ({mtp.head.n}-token draft head)")
        else:
            mode = "no drafts: the serial reference, one token a round"
        head_gib = mtp.nbytes() / GiB if self.depth else 0.0
        print(f"[tensorfold] Qwen3.6-35B-A3B on CUDA: {mode}; {self.capacity}-token context; weights "
              f"{model.nbytes() / GiB:.2f} GiB + MTP head {head_gib:.2f} GiB, {STATES} states "
              f"{STATES * self.e.pool.nbytes_per_seq() / GiB:.2f} GiB ({free / GiB:.1f} GiB was free, {source}); "
              f"loaded in {loaded_s:.1f}s, {captured} decode graphs captured and kernels warmed in {warm_s:.1f}s",
              flush=True)

    @property
    def eos(self) -> tuple[int, ...]:
        return self.e.eos

    def _warm(self) -> tuple[int, float]:
        """Capture the decode graphs (windows of 1 to depth + 1 rows for the kept state, one-row steps for the serial
        state, the head's steps) and compile what a request runs eagerly (a prefill chunk, a short window, the head's
        prompt absorb) before the first request."""

        import torch

        t0 = time.perf_counter()
        captured = self.e.warm(self.depth + 1, states=[self.e.st]) + self.e.warm(1, states=[self.serial])
        prompt = [(97 * i + 13) % self.e.m.cfg.vocab for i in range(72)]
        chunk, self.chunk = self.chunk, 64
        try:
            self.generate(prompt, 12, None, None)               # a 64-row chunk, an 8-row window, decode rounds
            self.generate(prompt, 3, None, None, draft=False)
        finally:
            self.chunk = chunk
        self.cache = []
        self.e.st.reset()
        self.serial.reset()
        torch.cuda.synchronize()
        return captured, time.perf_counter() - t0

    # -- prefix reuse ----------------------------------------------------------------------------------------------
    def _resume(self, prompt: Sequence[int]):
        """The longest kept state the prompt extends by at least one token, or None."""

        best = None
        for ids, snap in self.cache:
            if len(ids) < len(prompt) and list(prompt[:len(ids)]) == ids and (best is None or len(ids) > len(best[0])):
                best = (ids, snap)
        return best

    def _start_from(self, hit) -> None:
        """Before a prefill: resuming overwrites the cache rows past the kept prefix, so the kept states that extend
        it go; a fresh prompt overwrites them all."""

        if hit is None:
            self.cache = []
        else:
            n = len(hit[0])
            self.cache = [c for c in self.cache if len(c[0]) <= n or c[0][:n] != hit[0]]

    def _remember(self, ids: list[int]) -> None:
        snap = self.e.st.snapshot()
        self.cache = [c for c in self.cache if c[0] != ids][-1:] + [(ids, snap)]

    # -- decoding --------------------------------------------------------------------------------------------------
    def _limit(self, prompt: Sequence[int], max_tokens: int) -> int:
        if not prompt:
            raise ValueError("a prompt needs at least one token")
        room = self.capacity - len(prompt) - self.depth
        if room < 1:
            raise ValueError(f"a prompt of {len(prompt)} tokens leaves no room in the {self.capacity}-token context")
        return max(1, min(int(max_tokens), room))

    def generate(self, prompt: Sequence[int], max_tokens: int, sampling,
                 on_tokens: Callable[[list[int]], bool | None] | None, draft: bool = True) -> dict[str, Any]:
        """Up to ``max_tokens`` reply tokens after ``prompt``, passed to ``on_tokens`` as each round keeps them (a True
        return stops the decode); stops after an eos id. MTP-drafted when the engine has the head; ``draft=False``: one
        token a round from a fresh prefill in the serial state, leaving the kept states alone. Both emit the same
        tokens. Returns the request's stats."""

        import torch

        from .decode import mtp_decode, prefill, serial_decode

        max_tokens = self._limit(prompt, max_tokens)
        prompt = list(prompt)
        drafting = draft and self.depth > 0 and self.e.mtp is not None
        hit = self._resume(prompt) if draft else None
        st = self.e.st if draft else self.serial
        t0 = time.perf_counter()
        if draft:
            self._start_from(hit)
        first = prefill(self.e, prompt, sampling, st=st, chunk=self.chunk, resume=hit[1] if hit else None,
                        mtp=drafting)
        if draft:
            self._remember(prompt)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0
        cached = len(hit[0]) if hit else 0
        stats: dict[str, Any] = {"prompt_tokens": len(prompt), "cached": cached, "prefill_s": round(prefill_s, 4),
                                 "prefill_tps": round((len(prompt) - cached) / prefill_s, 1) if prefill_s else 0.0,
                                 "completion_tokens": 1, "drafts": drafting}
        stop = on_tokens is not None and bool(on_tokens([first]))
        if stop or first in self.eos or max_tokens <= 1:
            return stats
        if drafting:
            res = mtp_decode(self.e, first, max_tokens, sampling, st=st, depth=self.depth,
                             confidence=self.confidence, stop_eos=True, on_tokens=on_tokens)
            stats.update(drafted=res.drafted, accepted=res.accepted, acceptance=round(res.acceptance, 3),
                         tokens_per_round=round(res.tokens_per_round, 2))
        else:
            res = serial_decode(self.e, first, max_tokens, sampling, st=st, stop_eos=True, on_tokens=on_tokens)
        if draft and res.committed:        # the reply's state: every token but the pending last one is committed
            self._remember(prompt + res.committed)
        stats.update(completion_tokens=len(res.tokens), decode_s=round(res.seconds, 4), rounds=res.rounds,
                     decode_tps=round(res.tokens_per_second, 2))
        return stats
