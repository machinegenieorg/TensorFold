"""The Qwen3.6-35B-A3B CUDA engine behind ``tensorfold.cuda.server``: admission, drafting and prefix reuse."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

GiB = 1 << 30
WORKSPACE = 1 << 30        # allocator slack, Triton's workspace, sampling's temporaries
SNAPSHOTS = 3              # kept for prefix reuse: the prompt's, the reply's and the one being taken
STATES = 2                 # the kept sequence and the serial switch's
WINDOW_ROWS, ATTN_ROWS, LOGIT_ROWS = 32, 64, 32
MAX_DEPTH = 15             # MTP drafts a round at most (the package's MAX_DRAFTS): windows of up to 16 rows
ROUTER = (".mlp.gate.", ".mlp.shared_expert_gate.")


# -- memory, before anything is loaded ---------------------------------------------------------------------------
def weight_transform(hidden: int):
    """``capacity.admit``'s view of a tensor: (GPU bytes, mapped host bytes), routers in fp32, no vision tower."""

    from tensorfold.cuda.geometry import padded

    from .checkpoint import VISION_PREFIXES

    def transform(name: str, info: dict) -> tuple[int, int]:
        if name.startswith(VISION_PREFIXES) or ".visual." in name:
            return 0, 0
        if any(r in name for r in ROUTER):
            return (int(info["shape"][0]) * hidden * 4 if name.endswith(".weight") else 0), 0
        return padded(info, list(info["shape"]), float32=name.endswith((".A_log", ".dt_bias", "norm.weight"))), 0

    return transform


def state_bytes(cfg, capacity: int, mtp_layers: int = 0) -> tuple[int, int]:
    """(one sequence state of ``capacity`` positions, one snapshot of it), as ``state.Pool`` allocates them."""

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

    from tensorfold.cuda.experts import SMALL, max_items

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
    pairs = rows * k
    total += rows * (e * 4 + k * 8 + k * width * 2 + k * d * 4)            # router logits, picks, act, y
    total += pairs * 4 + max_items(pairs, e) * 12 + 8                      # the experts' plan
    if pairs > SMALL:
        total += pairs * 4 + -(-pairs // 1024) * e * 4
    total += 4 * qmm.split_scratch(rows, [(cfg.vocab, d), (sum(cfg.gdn_rows), d), (d, cfg.nv * cfg.dv),
                                          (sum(cfg.attn_rows), d), (d, qd)])        # split-K partials
    total += max(logit_rows, 1) * cfg.vocab * 2 + d * 2 * 2               # logits, last rows
    return int(total * 1.05) + (4 << 20)


def mtp_buffer_bytes(cfg, capacity: int, rows: int, head_rows: int) -> int:
    """``mtp.MTPBuffers``: steps of up to ``rows`` rows, only the last row past the cache write."""

    from tensorfold.cuda.experts import max_items

    from . import qmm

    d, e, k, width = cfg.hidden, cfg.experts + 1, cfg.top_k + 1, cfg.moe_width
    qd = cfg.heads * cfg.head_dim
    total = rows * d * 2 * 6 + rows * (3 * d // 64) * 4 + rows * (sum(cfg.attn_rows) + qd) * 2   # rows, pa, q
    total += -(-capacity // 512) * cfg.heads * (cfg.head_dim + 2) * 4 + qd * 4 + (2048 + 8) * 4 + d * 8   # one row
    total += e * 4 + k * 8 + k * width * 2 + k * d * 4 + k * 4 + max_items(k, e) * 12 + 8                # its MoE
    total += 4 * qmm.split_scratch(rows, [(d, 2 * d), (sum(cfg.attn_rows), d), (d, qd), (head_rows, d)])
    total += head_rows * 2
    return int(total * 1.05) + (1 << 20)


def draft_head_bytes(cfg, head_rows: int) -> int:
    """The draft head: the model's head rows at the draft vocabulary's ids, packed (rows padded to 128)."""

    rows = -(-head_rows // 128) * 128
    return rows * (cfg.hidden // 2 + cfg.hidden // 64 * 4)


def cache_bytes(cfg, capacity: int, *, rows: int, states: int = STATES, mtp_layers: int = 0,
                head_rows: int = 0) -> int:
    """Everything but the weights at a cache capacity: states, snapshots, buffers, the head's and a workspace."""

    from .decode import MTP_ROWS

    one, snap = state_bytes(cfg, capacity, mtp_layers)
    total = states * one + SNAPSHOTS * snap + buffer_bytes(cfg, capacity, rows) + WORKSPACE
    if mtp_layers:
        total += mtp_buffer_bytes(cfg, capacity, MTP_ROWS, head_rows) + draft_head_bytes(cfg, head_rows)
    return total


def admission(model_dir: str | Path | None, cfg, context: int | None, explicit: bool, *, rows: int, reserve: int,
              free_memory: int | None = None, loaded: bool = False, drafter: str | Path | None = None,
              head_rows: int = 0) -> dict:
    """``capacity.admit``'s receipt: the window and cache slots (plus ``reserve``) that fit, drafter included."""

    import torch

    from tensorfold.cuda import capacity as cap

    mtp_layers = 1 if head_rows else 0
    geometry = cap.Geometry(lambda slots: cache_bytes(cfg, slots, rows=rows, mtp_layers=mtp_layers,
                                                      head_rows=head_rows), reserve, reserve + 1)
    extra = tuple(sorted(Path(drafter).glob("*.safetensors"))) if drafter and not loaded else ()
    transform = weight_transform(cfg.hidden)
    if free_memory is None and not loaded:
        return cap.admit(model_dir, context, explicit, torch, geometry, transform, extra_files=extra)
    weights = cap.Weights(0, 0)
    if not loaded:
        weights = cap.estimate_weights(model_dir, transform)
        if extra:
            more = cap.estimate_weights(model_dir, transform, files=list(extra))
            weights = cap.Weights(weights.resident + more.resident, max(weights.staging, more.staging))
    budget = int(free_memory) if free_memory is not None else cap.available_bytes(torch)
    plan = cap.make_plan(cfg.max_position, context, explicit, budget, weights, geometry)
    return {**plan.receipt(cap.choose(plan)), "largest_window": plan.largest}


# -- the engine ----------------------------------------------------------------------------------------------------
class Qwen36Engine:
    """``eos``, ``generate`` and ``context_window`` as ``tensorfold.cuda.server`` expects, one request at a time."""

    def __init__(self, model_dir: str | Path | None, drafter: str = "", *, context: int | None = None,
                 context_explicit: bool | None = None, no_drafts: bool = False, mtp_drafts: int | None = None,
                 confidence: float | None = None, rows: int | None = None, prefill_chunk: int | None = None,
                 tp: int = 1, streams: int = 1, free_memory: int | None = None, model=None, mtp=None,
                 graphs: bool = True, warm: bool = True) -> None:
        if int(tp) != 1:
            raise ValueError("Qwen3.6-35B-A3B's CUDA engine runs on one GPU: two ranks do not apply")
        if mtp_drafts is not None and not 0 <= int(mtp_drafts) <= MAX_DEPTH:
            raise ValueError(f"MTP drafts a round: 0 to {MAX_DEPTH}, not {mtp_drafts}")
        if mtp_drafts and not drafter and mtp is None and not no_drafts:
            raise ValueError("MTP drafts come from the drafter (mlx-community/Qwen3.6-35B-A3B-MTP-4bit): pass its "
                             "directory, or no_drafts")
        import gc

        import torch

        from .decode import CONFIDENCE, DEPTH, GRAPH_ROWS, MTP_ROWS, PREFILL_ROWS, Decoder
        from .mtp import draft_token_ids
        from .checkpoint import Config

        started = time.perf_counter()
        torch.cuda.set_device(0)
        cfg = model.cfg if model is not None else Config.read(model_dir)
        rows = int(rows or PREFILL_ROWS)
        self.chunk = prefill_chunk
        # the drafting settings first: the admission reserves depth + 1 speculative positions and counts the head
        has_head = mtp is not None or bool(drafter)
        self.depth = 0 if no_drafts or not has_head else DEPTH if mtp_drafts is None else int(mtp_drafts)
        self.confidence = CONFIDENCE if confidence is None else float(confidence)
        self.drafter = str(drafter) if self.depth and mtp is None else ""
        head_rows = 0
        if self.depth:
            ids = draft_token_ids("default") if mtp is None else None
            head_rows = mtp.head.n if mtp is not None else int(((ids >= 0) & (ids < cfg.vocab)).sum())
        self.concurrent = False              # the server's lock: one request at a time
        if int(streams) > 1:
            print(f"[tensorfold] Qwen3.6-35B-A3B: --parallel {streams} accepted, requests are served one at a time "
                  "(shared rounds are not implemented for this family yet)", flush=True)
        explicit = context is not None if context_explicit is None else bool(context_explicit)
        self.capacity_plan = admission(model_dir, cfg, context, explicit, rows=rows, reserve=self.depth + 1,
                                       free_memory=free_memory, loaded=model is not None,
                                       drafter=self.drafter or None, head_rows=head_rows)
        self.max_len = int(self.capacity_plan["cache_slots"])
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
        self.e = Decoder(model, capacity=self.max_len, rows=rows, window_rows=max(WINDOW_ROWS, wide),
                         attn_rows=ATTN_ROWS, logit_rows=max(LOGIT_ROWS, wide), states=STATES,
                         mtp=mtp if self.depth else None, mtp_rows=MTP_ROWS, graphs=graphs,
                         graph_rows=max(GRAPH_ROWS, wide))
        self.serial = self.e.pool.alloc()             # the serial switch's state; self.e.st keeps the prefixes
        self.cache: list[tuple[list[int], dict]] = []  # (committed ids, State.snapshot of them)
        captured, warm_s = self._warm() if warm and self.max_len >= 128 else (0, 0.0)
        if self.depth:
            mode = (f"1 to {self.depth} MTP drafts a round, a chain stops before a later draft under "
                    f"{self.confidence:.0%} ({mtp.head.n}-token draft head)")
        else:
            mode = "no drafts: the serial reference, one token a round"
        head_gib = mtp.nbytes() / GiB if self.depth else 0.0
        plan = self.capacity_plan
        print(f"[tensorfold] Qwen3.6-35B-A3B on CUDA: {mode}; {self.context_window}-token prompt/reply window, "
              f"{self.max_len}-token cache; weights {model.nbytes() / GiB:.2f} GiB + MTP head {head_gib:.2f} GiB, "
              f"{STATES} states {STATES * self.e.pool.nbytes_per_seq() / GiB:.2f} GiB (estimate "
              f"{plan['total_bytes_estimate'] / GiB:.1f} GiB within {plan['budget_bytes'] / GiB:.1f}); loaded in "
              f"{loaded_s:.1f}s, {captured} decode graphs captured and kernels warmed in {warm_s:.1f}s", flush=True)

    @property
    def context_window(self) -> int:
        """Prompt and reply capacity: the cache slots less the speculative scratch positions."""

        return max(0, self.max_len - self.depth - 1)

    @property
    def eos(self) -> tuple[int, ...]:
        return self.e.eos

    def _warm(self) -> tuple[int, float]:
        """Capture the decode graphs and compile what a request runs eagerly, before the first request."""

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
        """Before a prefill: drop the kept states whose cache rows it overwrites (all of them for a fresh prompt)."""

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
        room = self.context_window - len(prompt)
        if room < 1:
            raise ValueError(f"a prompt of {len(prompt)} tokens leaves no room in the {self.context_window}-token "
                             "prompt/reply window; shorten the prompt or reserve fewer reply tokens")
        return max(1, min(int(max_tokens), room))

    def generate(self, prompt: Sequence[int], max_tokens: int, sampling,
                 on_tokens: Callable[[list[int]], bool | None] | None, draft: bool = True) -> dict[str, Any]:
        """Up to ``max_tokens`` reply tokens to ``on_tokens`` (True stops), drafted or serial; returns the stats."""

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
