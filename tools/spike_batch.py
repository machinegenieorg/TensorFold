#!/usr/bin/env python3
"""SPIKE: concurrency for the Qwen3.8-27B CUDA engine (families/qwen3_5/cuda/batch.py).

  python tools/spike_batch.py PROMPTS.json [--n 1,2,4,8] [--count 512] [--exact 4] [--no-draft]

PROMPTS.json: a list of OpenAI message lists. For each N, the first N prompts decode together (greedy);
aggregate tokens per second over the decode phase is reported. ``--exact K``: first check that K prompts
decoded together give, for every prompt, the same tokens as that prompt's own serial decode."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tensorfold.cuda.server import ChatTemplate
from tensorfold.families.qwen3_5.cuda.batch import PROF, Job, batch_decode
from tensorfold.families.qwen3_5.cuda.batch_draft import DPROF
from tensorfold.families.qwen3_5.cuda.decode import prefill, serial_decode
from tensorfold.families.qwen3_5.cuda.weights import load
from tensorfold.hub import resolve


def h(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("--model", default="Vontra/Qwen3.8-27B-MLX-4bit")
    ap.add_argument("--drafter", default="z-lab/Qwen3.8-27B-DFlash2")
    ap.add_argument("--n", default="1,2,4,8")
    ap.add_argument("--count", type=int, default=512)
    ap.add_argument("--exact", type=int, default=4)
    ap.add_argument("--exact-count", type=int, default=160)
    ap.add_argument("--max-rows", type=int, default=12)
    ap.add_argument("--row-budget", type=int, default=128)
    ap.add_argument("--no-draft", action="store_true")
    ap.add_argument("--serial-draft", action="store_true", help="draft each request on its own (the old path)")
    ap.add_argument("--no-multi", action="store_true", help="per-request GDN launches and commits (the old path)")
    ap.add_argument("--variants", default="new", help="comma list of: new (all batched), old (per-request draft, "
                    "GDN, attention and commit), noattn (new without multi-request attention)")
    ap.add_argument("--out", default="spike-batch.json")
    ap.add_argument("--profile", action="store_true", help="CUDA-event section times inside the verify forward")
    a = ap.parse_args()
    VARIANTS = {"new": dict(batch_draft=True, multi=True, multi_attention=True),
                "old": dict(batch_draft=False, multi=False, multi_attention=False),
                "noattn": dict(batch_draft=True, multi=True, multi_attention=False)}
    variants = a.variants.split(",")

    model_dir = resolve(a.model, download=False)
    torch.cuda.set_device(0)
    w = load(model_dir, tiled=True)
    draft = None
    if not a.no_draft:
        from tensorfold.families.qwen3_5.cuda.dflash2 import DFlash2
        draft = DFlash2(resolve(a.drafter, download=False), w)
    tpl, tok = ChatTemplate(Path(model_dir)), Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    msgs = json.load(open(a.prompts))
    prompts = [tok.encode(tpl.render(m, tools=None, enable_thinking=False), add_special_tokens=False).ids for m in msgs]
    print(f"prompts {len(prompts)}, tokens {min(map(len, prompts))}-{max(map(len, prompts))}", flush=True)
    report = {"exact": [], "runs": []}

    if a.exact:
        # reference: each prompt alone, serial (one token a round, no drafts): the engine's own reference
        refs = []
        for p in prompts[:a.exact]:
            st, pending = prefill(w, p, None)
            refs.append(serial_decode(w, st, pending, a.exact_count, None).tokens)
        jobs = [Job(p, a.exact_count) for p in prompts[:a.exact]]
        batch_decode(w, jobs, draft, row_budget=a.row_budget, max_rows=a.max_rows, **VARIANTS[variants[0]])
        for i, (r, j) in enumerate(zip(refs, jobs)):
            same = r == j.out
            report["exact"].append({"prompt": i, "serial": h(r), "batched": h(j.out), "identical": same, "len": len(r)})
            print(f"exact prompt {i}: serial {h(r)} batched {h(j.out)} -> {'IDENTICAL' if same else 'DIFFERENT'}", flush=True)

    # warm-up: compile every kernel bucket once so the timed runs measure steady state
    for v in variants:
        batch_decode(w, [Job(p, 48) for p in prompts[:8]], draft, row_budget=a.row_budget, max_rows=a.max_rows,
                     **VARIANTS[v])
    for n, v in [(int(x), v) for x in a.n.split(",") for v in variants]:
        jobs = [Job(p, a.count) for p in prompts[:n]]
        PROF.on, PROF.totals = a.profile, {}
        DPROF.on, DPROF.totals = a.profile, {}
        r = batch_decode(w, jobs, draft, row_budget=a.row_budget, max_rows=a.max_rows, **VARIANTS[v])
        r.update(variant=v, n=n, tokens_per_job=[len(j.out) - 1 for j in jobs],
                 accept_per_round=round(sum(j.accepted for j in jobs) / max(1, sum(j.rounds for j in jobs)), 2),
                 rows_per_round=round(r["rows"] / max(1, r["rounds"]), 1),
                 e2e_tok_s=round(r["generated"] / (r["prefill_s"] + r["decode_s"]), 1),
                 gpu_sections_s={k: round(v, 2) for k, v in PROF.totals.items()},
                 draft_sections_s={k: round(v, 2) for k, v in DPROF.totals.items()})
        report["runs"].append(r)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
