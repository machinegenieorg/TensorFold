"""Qwen3.6-35B-A3B measurements on CUDA: MoE stages, prefill, the served engine, drafting and quality."""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
USAGE = """
  moe [--rows 128 512]         MoE stages by GPU time (graph replay), and the shared prefill form for comparison
  router [--rows ...]          router launch settings by GPU time, every setting checked bit-equal
  prefill [--chunks 128 512]   the real checkpoint's prefill of a 2048-token prompt (--profile CHUNK: its kernels)
  prefill --fresh 23 57 301    first-request latency at new prompt lengths, what compiled (empty TRITON_CACHE_DIR)
  serve                        the engine as `tensorfold serve` builds it: start-up, memory, drafted against serial
  drafting                     depth, confidence and draft vocabulary; where a round's time goes; a 32k-cache step
  quality                      the forward and the MTP head against the fp32 reference (TF32 off)

Random routing spreads rows evenly over the experts; the real model's is skewed, so the real prefill is what counts.
Every real run needs mlx-community/Qwen3.6-35B-A3B-4bit (and for drafting its MTP drafter) in the Hugging Face cache.
"""


def _events(fn, repeats: int, inner: int = 10) -> float:
    """Median GPU milliseconds of one ``fn`` call: ``inner`` calls in a CUDA graph, replayed ``repeats`` times."""

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
    _tests()
    from test_qwen36moe_reference import PASSAGES  # the reference test's public passages in order, 2048 tokens

    return [t for text in PASSAGES.values() for t in tok.encode(text, add_special_tokens=False).ids][:2048]


def _tests() -> None:
    if str(ROOT / "tests" / "cuda") not in sys.path:
        sys.path[:0] = [str(ROOT / "tests" / "cuda"), str(ROOT)]


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
    """First-request latency at prompt lengths not seen before, after a warm-up, and which kernels compiled."""

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


def bench_prefill(chunks: list[int], profile: int | None, runs: int, fresh: list[int] | None = None) -> None:
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


# -- the served engine, drafting and quality on the real checkpoint --------------------------------------------------
GIB = 1 << 30
SAMPLINGS = {"greedy": None, "sampled": (20260928, 1.0, 20, 0.95)}     # the checkpoint's generation config, seeded
NAMES = ["chat 1", "chat 2", "chat 3", "json 1", "json 2"]


def _chat(tok, template, text: str) -> list[int]:
    return tok.encode(template.render([{"role": "user", "content": text}], tools=None, enable_thinking=False),
                      add_special_tokens=False).ids


def _sha(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(str(t) for t in tokens).encode()).hexdigest()[:16]


def _sampling(spec):
    from tensorfold.engine.exact_sampling import Sampling

    return None if spec is None else Sampling(*spec)


def bench_serve(tokens: int) -> None:
    """The engine as `tensorfold serve` builds it with no flags: start-up, memory, drafted against serial replies."""

    from tokenizers import Tokenizer

    from tensorfold import families, hub
    from tensorfold.cli import _drafter
    from tensorfold.cuda.server import ChatTemplate
    from tensorfold.families import qwen3_5_moe as family

    _tests()
    from test_qwen36moe_mtp import CHAT, EXTRACT

    snap = hub.cached(family.MODELS[0])
    tok, template = Tokenizer.from_file(str(snap / "tokenizer.json")), ChatTemplate(snap)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    eng = family.cuda_engine(snap, drafter=_drafter(families.families()["qwen3_5_moe"], "auto"), tp=1, rank=0,
                             master="", master_port=29551, no_drafts=False)
    built = time.perf_counter() - t0
    plan = eng.capacity_plan
    free, total = torch.cuda.mem_get_info()
    print(f"{torch.cuda.get_device_name()}: built in {built:.1f} s, depth {eng.depth}, window {eng.context_window}; "
          f"allocated {torch.cuda.memory_allocated() / GIB:.2f} GiB, peak {torch.cuda.max_memory_allocated() / GIB:.2f}"
          f" GiB, estimate {plan['total_bytes_estimate'] / GIB:.2f} GiB within {plan['budget_bytes'] / GIB:.2f} GiB; "
          f"{(total - free) / GIB:.2f} of {total / GIB:.2f} GiB in use on the device")
    for sname, spec in SAMPLINGS.items():
        sampling = _sampling(spec)
        sums = {"tokens": 0, "drafted_s": 0.0, "serial_s": 0.0, "eager_s": 0.0, "drafts": 0, "kept": 0, "rounds": 0}
        for name, text in zip(NAMES, CHAT + EXTRACT):
            prompt = _chat(tok, template, text)
            got, stats = {}, {}
            for run, draft in (("drafted", True), ("serial", False), ("eager", False)):
                graphs = eng.e.graphs
                if run == "eager":
                    eng.e.graphs = None
                try:
                    got[run] = []
                    stats[run] = eng.generate(prompt, tokens, sampling, got[run].extend, draft=draft)
                finally:
                    eng.e.graphs = graphs
            d, s = stats["drafted"], stats["serial"]
            same = "equal" if got["drafted"] == got["serial"] == got["eager"] else "DIFFERENT"
            eager = stats["eager"].get("decode_tps")
            print(f"{sname:7s} {name}: {len(got['serial'])} tokens sha256 {_sha(got['serial'])} {same}; drafted "
                  f"{d.get('decode_tps')} tok/s, serial {s.get('decode_tps')} (eager {eager}); acceptance "
                  f"{d.get('acceptance')}, {d.get('tokens_per_round')} tokens a round")
            sums["tokens"] += len(got["serial"]) - 1
            for run in ("drafted", "serial", "eager"):
                sums[run + "_s"] += stats[run].get("decode_s", 0.0)
            sums["drafts"] += d.get("drafted", 0)
            sums["kept"] += d.get("accepted", 0)
            sums["rounds"] += d.get("rounds", 0)
        n = sums["tokens"]
        print(f"{sname:7s} all: drafted {n / sums['drafted_s']:.1f} tok/s, serial {n / sums['serial_s']:.1f} "
              f"(eager {n / sums['eager_s']:.1f}) = {sums['serial_s'] / sums['drafted_s']:.2f}x; acceptance "
              f"{sums['kept'] / max(1, sums['drafts']):.3f}; {n / max(1, sums['rounds']):.2f} tokens a round")


def _use_head(e, k) -> None:
    from tensorfold.families.qwen3_5_moe.cuda.mtp import MTPBuffers

    e.mtp = k
    e.mbuf = MTPBuffers(e.m, k, e.mbuf.rows, capacity=e.capacity)
    e.graphs.steps.clear()


def _profile(e, ids, sampling, depth: int, confidence: float, rounds: int = 48) -> dict[str, float]:
    """Mean milliseconds a drafted round spends verifying, committing and drafting, and a serial step's."""

    from tensorfold.families.qwen3_5_moe.cuda import decode as D

    st, b = e.st, e.buf
    first = D.prefill(e, ids, sampling)
    t = {"verify": 0.0, "commit": 0.0, "draft": 0.0, "rows": 0.0, "kept": 0.0}
    out = [first]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    drafts = D.draft(e, st, st.mtp_tail[None], [first], st.pos + 1, depth, sampling, confidence)
    t["draft"] += time.perf_counter() - t0
    for _ in range(rounds):
        t0 = time.perf_counter()
        tokens = [out[-1]] + drafts
        lg = e.forward(tokens, st)
        sampled = D.sample_rows(lg[:len(tokens)], [st.pos + 1 + r for r in range(len(tokens))], sampling)
        t1 = time.perf_counter()
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d:
                break
            keep += 1
        e.commit(keep, st)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        out += sampled[:keep]
        drafts = D.draft(e, st, b.hidden[:keep], sampled[:keep], st.pos + 1, depth, sampling, confidence)
        t3 = time.perf_counter()
        t["verify"] += t1 - t0
        t["commit"] += t2 - t1
        t["draft"] += t3 - t2
        t["rows"] += len(tokens)
        t["kept"] += keep
    res = {k: 1e3 * v / rounds for k, v in t.items() if k not in ("rows", "kept")}
    res["rows"], res["kept"] = t["rows"] / rounds, t["kept"] / rounds
    first = D.prefill(e, ids, sampling, mtp=False)
    res["serial step"] = 1e3 * D.serial_decode(e, first, rounds + 1, sampling).seconds / rounds
    return res


def bench_drafting(tokens: int) -> None:
    """Depth, confidence and the draft vocabulary on a chat and a JSON prompt, a round's phases, a 32k-cache step."""

    import dataclasses
    import gc

    from tokenizers import Tokenizer

    from tensorfold import hub
    from tensorfold.cuda.server import ChatTemplate
    from tensorfold.families import qwen3_5_moe as family
    from tensorfold.families.qwen3_5_moe.cuda import decode as D
    from tensorfold.families.qwen3_5_moe.cuda import weights as W
    from tensorfold.families.qwen3_5_moe.cuda.forward import prepare
    from tensorfold.families.qwen3_5_moe.cuda.mtp import draft_head, prepare_mtp

    _tests()
    from test_qwen36moe_mtp import CHAT, EXTRACT

    snap = hub.cached(family.MODELS[0])
    tok, template = Tokenizer.from_file(str(snap / "tokenizer.json")), ChatTemplate(snap)
    w = W.load(snap, "cuda")
    mtpw = W.load_mtp(hub.cached(family.DRAFTER), w.cfg, "cuda")
    m = prepare(w)
    del w
    gc.collect()
    torch.cuda.empty_cache()
    full = prepare_mtp(mtpw, m, draft_vocab=None)
    head, ids, ids_host = draft_head(m, "default")
    heads = {"draft vocabulary": dataclasses.replace(full, head=head, ids=ids, ids_host=ids_host), "full": full}
    e = D.Decoder(m, capacity=4096, rows=512, mtp=heads["draft vocabulary"], graphs=True, states=1)
    e.warm(8)
    sampling = _sampling(SAMPLINGS["sampled"])
    prompts = {"chat": _chat(tok, template, CHAT[0]), "json": _chat(tok, template, EXTRACT[0])}

    def run(ids, draft: bool, depth: int = D.DEPTH, confidence: float = D.CONFIDENCE):
        return D.generate_result(e, ids, tokens, sampling, stop_eos=True, draft=draft, depth=depth,
                                 confidence=confidence)

    base = {name: run(ids, False) for name, ids in prompts.items()}
    vocab = {int(t) for t in heads["draft vocabulary"].ids_host}
    for name, res in base.items():
        inside = sum(t in vocab for t in res.tokens) / len(res.tokens)
        print(f"{name}: serial {res.tokens_per_second:.1f} tok/s; {inside:.3f} of the reply in the draft vocabulary")
    for label in ("draft vocabulary", "full"):
        _use_head(e, heads[label])
        configs = [(3, 0.5), (4, 0.5), (6, 0.3), (6, 0.5), (6, 0.7), (8, 0.5), (15, 0.5)]
        for depth, conf in configs if label == "draft vocabulary" else [(D.DEPTH, D.CONFIDENCE)]:
            parts = []
            for name, ids in prompts.items():
                run(ids, True, depth, conf)                    # captures the new shapes' graphs
                res = run(ids, True, depth, conf)
                same = "" if res.tokens == base[name].tokens else " DIFFERENT"
                parts.append(f"{name} {res.tokens_per_second / base[name].tokens_per_second:.2f}x (acceptance "
                             f"{res.acceptance:.2f}, {res.tokens_per_round:.2f} a round){same}")
            print(f"{label:16s} depth {depth:2d} confidence {conf:.1f}: " + ", ".join(parts))
    _use_head(e, heads["draft vocabulary"])
    for name, ids in prompts.items():
        run(ids, True)                                         # captures the head's steps again
        p = _profile(e, ids, sampling, D.DEPTH, D.CONFIDENCE)
        print(f"{name} round at depth {D.DEPTH}: verify {p['verify']:.2f} ms ({p['rows']:.2f} rows), commit "
              f"{p['commit']:.2f} ms, draft {p['draft']:.2f} ms; {p['kept']:.2f} kept; serial step "
              f"{p['serial step']:.2f} ms")
    big = D.Decoder(m, capacity=32768, rows=512, graphs=True, states=1)
    big.warm(2)
    step = {}
    for d in (e, big):
        for _ in range(2):
            first = D.prefill(d, prompts["chat"], None, mtp=False)
            res = D.serial_decode(d, first, 65, None)
        step[d.capacity] = 1e3 / res.tokens_per_second
    print("serial step with graphs: " + ", ".join(f"{ms:.2f} ms at a {c}-position cache" for c, ms in step.items()))


def bench_quality() -> None:
    """The forward against the fp32 reference (NLL, argmax agreement), and the MTP fc's input order."""

    import dataclasses
    import gc
    import os

    from tokenizers import Tokenizer

    from tensorfold import hub
    from tensorfold.families import qwen3_5_moe as family
    from tensorfold.families.qwen3_5_moe.cuda import decode as D
    from tensorfold.families.qwen3_5_moe.cuda import reference as R
    from tensorfold.families.qwen3_5_moe.cuda import weights as W
    from tensorfold.families.qwen3_5_moe.cuda.forward import prepare

    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise SystemExit("set TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0: the reference's fp32 matmuls would run in TF32")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _tests()
    from test_qwen36moe_reference import PASSAGES, _swap_fc_halves

    snap = hub.cached(family.MODELS[0])
    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    w = W.load(snap, "cuda")
    ref = {}
    for name, text in PASSAGES.items():
        ids = tok.encode(text, add_special_tokens=False).ids
        h = R.hidden(w, torch.tensor(ids), R.new_state(w))
        arg, nll = R.head_top1(w, h, torch.tensor(ids[1:] + [0]))
        ref[name] = (ids, h, arg, nll[:-1])
    mtp = W.load_mtp(hub.cached(family.DRAFTER), w.cfg, "cuda")
    for label, head in (("[embedding | hidden]", mtp),
                        ("[hidden | embedding]", dataclasses.replace(mtp, fc=_swap_fc_halves(mtp.fc, w.cfg.hidden)))):
        agree = n = 0
        for ids, h, arg, _ in ref.values():
            hm = R.mtp_hidden(head, w, torch.tensor(ids[1:]), h[:-1], R.new_mtp_state(head, w))
            agree += int((R.head_top1(w, hm[:-1])[0] == arg[1:-1]).sum())
            n += len(ids) - 2
        print(f"MTP fc {label}: its top-1 is the model's next token at {agree / n:.3f} of {n} rows")
    ref = {name: (ids, arg.cpu(), nll.cpu()) for name, (ids, _, arg, nll) in ref.items()}
    del mtp
    m = prepare(w)
    del w
    gc.collect()
    torch.cuda.empty_cache()
    e = D.Decoder(m, capacity=4096, rows=512)
    agree_all = n_all = 0
    ours_sum = ref_sum = 0.0
    for name, (ids, arg_r, nll_r) in ref.items():
        arg, nll = D.score(e, ids)
        n = len(ids) - 1
        agree = int((arg[:-1].cpu() == arg_r[:-1]).sum())
        print(f"{name:12s} {n:5d} tokens: NLL {float(nll.mean()):.4f}, reference {float(nll_r.mean()):.4f}; argmax "
              f"agreement {agree / n:.4f}")
        agree_all, n_all = agree_all + agree, n_all + n
        ours_sum, ref_sum = ours_sum + float(nll.sum()), ref_sum + float(nll_r.sum())
    print(f"{'all':12s} {n_all:5d} tokens: NLL {ours_sum / n_all:.4f}, reference {ref_sum / n_all:.4f}; argmax "
          f"agreement {agree_all / n_all:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)
    a = sub.add_parser("moe")
    a.add_argument("--rows", type=int, nargs="+", default=[128, 512])
    a.add_argument("--repeats", type=int, default=20)
    r = sub.add_parser("router")
    r.add_argument("--rows", type=int, nargs="+", default=[1, 16, 32, 64, 128, 512])
    r.add_argument("--repeats", type=int, default=10)
    b = sub.add_parser("prefill")
    b.add_argument("--chunks", type=int, nargs="+", default=[128, 512])
    b.add_argument("--runs", type=int, default=3)
    b.add_argument("--profile", type=int, default=None, metavar="CHUNK")
    b.add_argument("--fresh", type=int, nargs="+", default=None, metavar="TOKENS",
                   help="only: first-request latency at these new prompt lengths, and what compiled")
    for name in ("serve", "drafting"):
        sub.add_parser(name).add_argument("--tokens", type=int, default=256)
    sub.add_parser("quality")
    args = p.parse_args()
    with torch.no_grad():
        if args.what == "moe":
            bench_moe(args.rows, args.repeats)
        elif args.what == "router":
            bench_router(args.rows, args.repeats)
        elif args.what == "prefill":
            bench_prefill(args.chunks, args.profile, args.runs, args.fresh)
        elif args.what == "serve":
            bench_serve(args.tokens)
        elif args.what == "drafting":
            bench_drafting(args.tokens)
        else:
            bench_quality()


if __name__ == "__main__":
    main()
