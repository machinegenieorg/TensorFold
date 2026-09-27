#!/usr/bin/env python3
"""SPIKE: steady-state continuous batching for the Qwen3.8-27B CUDA engine (families/qwen3_5/cuda/sched.py).

  python tools/spike_serve.py PROMPTS.json [--requests 48] [--concurrency 4,8,16] [--count 512] [--exact 6]

Every request is queued at t=0; up to C run at once (prompt chunks interleaved with decode windows, shared
prompt prefix cached). Reports aggregate output tok/s over the wall time (prefill included), TTFT and latency.
``--exact K``: first check K requests served at concurrency K-2 give each request's own serial decode."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tensorfold.cuda.server import ChatTemplate
from tensorfold.families.qwen3_5.cuda.batch import PROF
from tensorfold.families.qwen3_5.cuda.batch_draft import DPROF
from tensorfold.families.qwen3_5.cuda.decode import prefill, serial_decode
from tensorfold.families.qwen3_5.cuda.sched import Request, run
from tensorfold.families.qwen3_5.cuda.weights import load
from tensorfold.hub import resolve


def h(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("--model", default="Vontra/Qwen3.8-27B-MLX-4bit")
    ap.add_argument("--drafter", default="z-lab/Qwen3.8-27B-DFlash2")
    ap.add_argument("--requests", type=int, default=48)
    ap.add_argument("--concurrency", default="8")
    ap.add_argument("--count", type=int, default=512)
    ap.add_argument("--exact", type=int, default=0)
    ap.add_argument("--exact-count", type=int, default=96)
    ap.add_argument("--max-rows", type=int, default=12)
    ap.add_argument("--row-budget", type=int, default=128)
    ap.add_argument("--prefill-reserve", type=int, default=32)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--sweep", default="", help="comma list of C:max_rows:prefill_reserve; the first is run once "
                    "untimed as a warm-up")
    ap.add_argument("--repeat", type=int, default=1, help="run each concurrency this many times (the first run "
                    "after start-up is slow: first-shape compiles)")
    ap.add_argument("--out", default="spike-serve.json")
    a = ap.parse_args()

    model_dir = resolve(a.model, download=False)
    torch.cuda.set_device(0)
    w = load(model_dir, tiled=True)
    from tensorfold.families.qwen3_5.cuda.dflash2 import DFlash2
    draft = DFlash2(resolve(a.drafter, download=False), w)
    tpl, tok = ChatTemplate(Path(model_dir)), Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    msgs = json.load(open(a.prompts))
    prompts = [tok.encode(tpl.render(m, tools=None, enable_thinking=False), add_special_tokens=False).ids for m in msgs]
    print(f"prompts {len(prompts)}, tokens {min(map(len, prompts))}-{max(map(len, prompts))}", flush=True)
    kw = dict(row_budget=a.row_budget, max_rows=a.max_rows, prefill_reserve=a.prefill_reserve)
    report = {"exact": [], "runs": []}

    if a.exact:
        refs = []
        for p in prompts[:a.exact]:
            st, pending = prefill(w, p, None)
            refs.append(serial_decode(w, st, pending, a.exact_count, None).tokens)
        reqs = [Request(p, a.exact_count) for p in prompts[:a.exact]]
        run(w, reqs, draft, concurrency=max(1, a.exact - 2), **kw)
        for i, (ref, r) in enumerate(zip(refs, reqs)):
            same = ref == r.out
            report["exact"].append({"prompt": i, "serial": h(ref), "served": h(r.out), "identical": same})
            print(f"exact prompt {i}: serial {h(ref)} served {h(r.out)} -> {'IDENTICAL' if same else 'DIFFERENT'}",
                  flush=True)

    # warm-up: compile every kernel bucket
    run(w, [Request(p, 48) for p in prompts[:8]], draft, concurrency=8, **kw)
    n = min(a.requests, len(prompts))
    if a.sweep:
        plan = [tuple(int(v) for v in x.split(":")) for x in a.sweep.split(",")]
        plan = [(*plan[0], False)] + [(*p, True) for p in plan]
    else:
        plan = [(int(x), a.max_rows, a.prefill_reserve, True) for x in a.concurrency.split(",") for _ in range(a.repeat)]
    for c, mr, res, timed_run in plan:
        reqs = [Request(prompts[i % len(prompts)], a.count) for i in range(n)]
        PROF.on, PROF.totals = a.profile, {}
        DPROF.on, DPROF.totals = a.profile, {}
        r = run(w, reqs, draft, concurrency=c, row_budget=a.row_budget, max_rows=mr, prefill_reserve=res)
        if not timed_run:
            print(json.dumps({"warmup_tok_s": round(r["agg_tok_s"], 1)}), flush=True)
            continue
        lat = [d - 0 for d in r["done_s"]]
        summary = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k not in ("ttft_s", "done_s")}
        summary.update(concurrency=c, max_rows=mr, prefill_reserve=res, requests=n, median_ttft_s=statistics.median(r["ttft_s"]),
                       median_done_s=statistics.median(lat), gpu_sections_s={k: round(v, 2) for k, v in PROF.totals.items()},
                       draft_sections_s={k: round(v, 2) for k, v in DPROF.totals.items()})
        report["runs"].append({**summary, "ttft_s": r["ttft_s"], "done_s": r["done_s"]})
        print(json.dumps(summary), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
