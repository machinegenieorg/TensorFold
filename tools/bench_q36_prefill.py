"""Qwen3.6-35B-A3B prefill timing on CUDA: the MoE stages on synthetic weights, and the real checkpoint's prefill.

  python tools/bench_q36_prefill.py moe  [--rows 128 512]     router, top-k and plan, gate/up, down, combine (and the
                                                              shared experts' prefill form, which rounds
                                                              differently, for comparison)
  python tools/bench_q36_prefill.py router [--rows ...]      router launch settings by time (bits checked equal)
  python tools/bench_q36_prefill.py real [--chunks 128 512]   tok/s over a 2048-token prompt (the P6 test's)
  python tools/bench_q36_prefill.py real --profile 512        the kernels of one prompt at that chunk, by GPU time
  python tools/bench_q36_prefill.py real --fresh 23 57 301     first-request latency at new prompt lengths, and the
                                                              Triton kernels that compiled for each (use an empty
                                                              TRITON_CACHE_DIR, or the disk cache hides compiles)

``moe`` and ``router`` time launches replayed from a CUDA graph (GPU time, no launch overhead). Random routing
spreads rows evenly over the experts; the real model's routing is skewed, so ``real`` is the number that counts.
``real`` needs mlx-community/Qwen3.6-35B-A3B-4bit in the Hugging Face cache and reads the prompt from the forward
test's passages.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _events(fn, repeats: int, inner: int = 10) -> float:
    """Median GPU milliseconds of one ``fn`` call: ``inner`` calls captured in a CUDA graph (no launch overhead),
    the graph replayed ``repeats`` times."""

    for _ in range(2):                                  # compile and warm up
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            fn()
    times = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / inner)
    return statistics.median(times)


def bench_moe(rows_list: list[int], repeats: int) -> None:
    from tensorfold.cuda import experts as grouped
    from tensorfold.families.qwen3_5_moe.cuda import moe, qmm
    from tensorfold.families.qwen4_exp.cuda import moe as fn_moe

    dev = "cuda"
    e, k_top, w, d = moe.EXPERTS, moe.TOP_K, moe.WIDTH, moe.HIDDEN

    def mlx(n, k, seed, lead=()):
        g = torch.Generator(device=dev).manual_seed(seed)
        words = torch.randint(-(2**31), 2**31 - 1, (*lead, n, k // 8), generator=g, device=dev,
                              dtype=torch.int64).to(torch.int32)
        scales = (torch.rand((*lead, n, k // 64), generator=g, device=dev) * 0.02 + 0.001).to(torch.bfloat16)
        biases = (torch.randn((*lead, n, k // 64), generator=g, device=dev) * 0.02).to(torch.bfloat16)
        return words, scales, biases

    ex = qmm.make_experts(mlx(w, d, 11, (e + 1,)), mlx(w, d, 12, (e + 1,)), mlx(d, w, 13, (e + 1,)))
    g = torch.Generator(device=dev).manual_seed(1)
    table = (torch.randn((e + 1, d), generator=g, device=dev) * 0.02).contiguous()
    big = moe.buffers(max(rows_list), dev)
    slots = k_top + 1
    print(f"{torch.cuda.get_device_name()}: MoE stages (median of {repeats})")
    for rows in rows_list:
        x = (torch.randn((rows, d), generator=g, device=dev)).to(torch.bfloat16)
        sub = moe.moe(x, None, table, ex, big)
        live = int(big.plan.counts[1].item())
        block = 1 << e.bit_length()
        act, y = big.act.view(-1, w), big.y.view(-1, d)
        pre = grouped.Plan(rows, slots, e + 1, dev, prefill=True)
        grouped.route(sub.pick, pre)
        pact = torch.empty((rows * slots, w), dtype=torch.bfloat16, device=dev)
        py = torch.empty((rows * slots, d), dtype=torch.bfloat16, device=dev)
        t = {
            "router": _events(lambda: moe.router(x, table, sub.logits), repeats),
            "topk": _events(lambda: fn_moe._topk_rows[(rows,)](
                sub.logits, sub.pick, sub.wts, NE=e, NL=e + 1, TOPK=k_top, SLOTS=slots, BLOCK=block,
                SLOTP=16, num_warps=4), repeats),
            "plan": _events(lambda: grouped.route(sub.pick, big.plan), repeats),
            "gateup": _events(lambda: grouped.gate_up(x, ex, big.plan, act, rows), repeats),
            "down": _events(lambda: grouped.down(act, ex, big.plan, y, rows), repeats),
            "combine": _events(lambda: moe.combine(sub.y, sub.wts), repeats),
            "gateup_prefill_form": _events(lambda: grouped.gate_up(x, ex, pre, pact, rows), repeats),
            "down_prefill_form": _events(lambda: grouped.down(pact, ex, pre, py, rows), repeats),
        }
        total = t["router"] + t["topk"] + t["plan"] + t["gateup"] + t["down"] + t["combine"]
        print(f"rows {rows:4d} ({live} experts), us: " + "  ".join(f"{k} {v * 1e3:.1f}" for k, v in t.items())
              + f"  | sum {total * 1e3:.0f} us a layer, x40 = {40 * total:.1f} ms")


def bench_router(rows_list: list[int], repeats: int) -> None:
    import itertools

    from tensorfold.families.qwen3_5_moe.cuda import moe

    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(1)
    table = (torch.randn((moe.EXPERTS + 1, moe.HIDDEN), generator=g, device=dev) * 0.02).contiguous()
    print(f"{torch.cuda.get_device_name()}: router, us (median of {repeats}); every setting checked bit-equal")
    for rows in rows_list:
        x = torch.randn((rows, moe.HIDDEN), generator=g, device=dev).to(torch.bfloat16)
        out = torch.empty((rows, moe.EXPERTS + 1), dtype=torch.float32, device=dev)
        ref = moe.router(x, table).clone()
        base = _events(lambda: moe.router(x, table, out), repeats) * 1e3
        got = []
        for bm, be, bk, w, st in itertools.product((16, 32, 64), (16, 32, 64), (32, 64), (2, 4, 8), (2, 3)):
            if bm > max(16, triton_pow2(rows)):
                continue
            kw = dict(block_m=bm, block_e=be, bk=bk, num_warps=w, num_stages=st)
            try:
                moe.router(x, table, out, **kw)
            except Exception:  # noqa: BLE001 - out of shared memory or registers
                continue
            if not torch.equal(out, ref):
                raise AssertionError(f"router bits changed at {kw}")
            got.append((_events(lambda: moe.router(x, table, out, **kw), repeats) * 1e3, kw))
        got.sort(key=lambda t: t[0])
        named = {tuple(kw.values()): t for t, kw in got}
        print(f"rows {rows}: default {base:.1f}; best " + "; ".join(
            f"{t:.1f} {tuple(kw.values())}" for t, kw in got[:5]) + "; " + "; ".join(
            f"{named[c]:.1f} {c}" for c in ((16, 16, 64, 4, 3), (32, 16, 64, 2, 3), (64, 16, 64, 2, 3)) if c in named))


def triton_pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def _prompt(tok) -> list[int]:
    sys.path.insert(0, str(ROOT / "tests" / "cuda"))
    from test_qwen36moe_reference import PASSAGES  # the forward test's prompt: its passages in order, 2048 tokens

    return [t for text in PASSAGES.values() for t in tok.encode(text, add_special_tokens=False).ids][:2048]


def _compiled() -> dict[str, int]:
    """Compiled variants of every Triton kernel in the loaded tensorfold modules, by 'family/module.kernel'."""

    from triton.runtime.jit import JITFunction

    out = {}
    for name, mod in list(sys.modules.items()):
        if not name.startswith("tensorfold."):
            continue
        parts = name.split(".")
        for k, v in list(vars(mod).items()):
            if isinstance(v, JITFunction):
                out[f"{parts[-3]}/{parts[-1]}.{k}"] = sum(len(c[0]) for c in v.device_caches.values())
    return out


def bench_fresh(e, long: list[int], lengths: list[int]) -> None:
    """First-request latency at prompt lengths not seen before (after a 33-token prompt, 8 one-row steps and the
    2048-token prompt in 512-row chunks), and which kernels compiled for it."""

    from tensorfold.families.qwen3_5_moe.cuda.decode import prefill, serial_decode

    t0 = time.time()
    first = prefill(e, long[:33])
    serial_decode(e, first, 8)
    prefill(e, long, chunk=512)
    torch.cuda.synchronize()
    print(f"warm-up (33-token prompt, 8 steps, 2048 tokens in 512-row chunks): {time.time() - t0:.1f} s")
    for n in lengths:
        before = _compiled()
        times = []
        for _ in range(2):
            torch.cuda.synchronize()
            t0 = time.time()
            prefill(e, long[:n])
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        after = _compiled()
        new = [f"{k} +{after[k] - before.get(k, 0)}" for k in sorted(after) if after[k] != before.get(k, 0)]
        print(f"fresh {n}-token prompt: first {times[0] * 1e3:.0f} ms, again {times[1] * 1e3:.0f} ms; compiled: "
              + (", ".join(new) or "nothing"))


def bench_real(chunks: list[int], profile: int | None, runs: int, fresh: list[int] | None = None) -> None:
    import gc

    from tokenizers import Tokenizer

    from tensorfold import hub
    from tensorfold.families import qwen3_5_moe as family
    from tensorfold.families.qwen3_5_moe.cuda import weights as W
    from tensorfold.families.qwen3_5_moe.cuda.decode import Decoder, run_prompt
    from tensorfold.families.qwen3_5_moe.cuda.forward import prepare

    snap = hub.cached(family.MODELS[0])
    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    long = _prompt(tok)
    t0 = time.time()
    m = prepare(W.load(snap, "cuda"))
    gc.collect()
    torch.cuda.empty_cache()
    e = Decoder(m, capacity=4096, rows=512)
    print(f"{torch.cuda.get_device_name()}: loaded in {time.time() - t0:.0f} s; prompt {len(long)} tokens")
    if fresh:
        with torch.no_grad():
            bench_fresh(e, long, fresh)
        return
    with torch.no_grad():
        for chunk in chunks:
            rates = []
            for i in range(runs + 1):                   # the first run compiles and warms up
                torch.cuda.synchronize()
                t0 = time.time()
                run_prompt(e, long, chunk=chunk)
                torch.cuda.synchronize()
                if i:
                    rates.append(len(long) / (time.time() - t0))
            print(f"{chunk}-row chunks: {statistics.median(rates):.0f} tok/s (median of {runs}; "
                  + ", ".join(f"{r:.0f}" for r in rates) + ")")
        if profile:
            from torch.profiler import ProfilerActivity, profile as prof

            run_prompt(e, long, chunk=profile)
            with prof(activities=[ProfilerActivity.CUDA]) as p:
                run_prompt(e, long, chunk=profile)
                torch.cuda.synchronize()
            print(p.key_averages().table(sort_by="cuda_time_total", row_limit=25))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)
    a = sub.add_parser("moe")
    a.add_argument("--rows", type=int, nargs="+", default=[128, 512])
    a.add_argument("--repeats", type=int, default=20)
    r = sub.add_parser("router")
    r.add_argument("--rows", type=int, nargs="+", default=[1, 16, 32, 64, 128, 512])
    r.add_argument("--repeats", type=int, default=10)
    b = sub.add_parser("real")
    b.add_argument("--chunks", type=int, nargs="+", default=[128, 512])
    b.add_argument("--runs", type=int, default=3)
    b.add_argument("--profile", type=int, default=None, metavar="CHUNK")
    b.add_argument("--fresh", type=int, nargs="+", default=None, metavar="TOKENS",
                   help="only: first-request latency at these new prompt lengths, and what compiled")
    args = p.parse_args()
    if args.what == "moe":
        bench_moe(args.rows, args.repeats)
    elif args.what == "router":
        bench_router(args.rows, args.repeats)
    else:
        bench_real(args.chunks, args.profile, args.runs, args.fresh)


if __name__ == "__main__":
    main()
