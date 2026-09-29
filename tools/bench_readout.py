"""Readout scoring latency and memory: N requests in flight, sharing a long token-id prefix, on /v1/completions.

Mirrors the readout wrapper's traffic shape: a token-id prompt (max_tokens 1, temperature 0,
return_tokens_as_token_ids, logprobs, allowed_token_ids), most of the prompt shared across requests so the
server's prefix cache does the work. Standard library only, so it runs on a bare host.

  python3 tools/bench_readout.py http://127.0.0.1:8080 qwen3.5-4b-readout --vocab 248320 \\
      --concurrency 4 --total 4096 --prefix 3500 --reps 5 --mem --output out.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import threading
import time
import urllib.request

MAX_LOGPROBS = 64


def build_requests(model: str, vocab: int, total: int, prefix: int, n: int, allowed: int, seed: int) -> list[dict]:
    """``n`` requests sharing the first ``prefix`` token ids, each with its own random tail and allowed set."""

    rng = random.Random(seed)
    shared = [rng.randrange(1, vocab) for _ in range(prefix)]
    requests = []
    for i in range(n):
        tail = [rng.randrange(1, vocab) for _ in range(total - prefix)]
        allowed_ids = sorted(rng.sample(range(vocab), min(allowed, vocab)))
        requests.append({"model": model, "prompt": shared + tail, "max_tokens": 1, "temperature": 0,
                         "return_tokens_as_token_ids": True, "logprobs": min(len(allowed_ids), MAX_LOGPROBS),
                         "allowed_token_ids": allowed_ids})
    return requests


def one(base: str, body: dict) -> dict:
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    sent = time.perf_counter()
    out = {"sent": sent, "latency_s": None, "error": None, "chosen": None}
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read())
        out["latency_s"] = time.perf_counter() - sent
        choice = payload["choices"][0]
        out["chosen"] = choice["text"]
        out["top_logprobs"] = choice["logprobs"]["top_logprobs"][0]
    except Exception as exc:  # noqa: BLE001 - a failed request is reported, not raised
        out["latency_s"] = time.perf_counter() - sent
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


class Memory(threading.Thread):
    """Samples nvidia-smi's per-process compute memory every 0.5s (``tools/shared_prefix_load.py``'s technique)."""

    def __init__(self, ignore: tuple[str, ...]) -> None:
        super().__init__(daemon=True)
        self.ignore, self.stop, self.samples = ignore, threading.Event(), []

    def gpu_gib(self) -> float:
        rows = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        total = 0.0
        for row in rows.strip().splitlines():
            parts = [p.strip() for p in row.split(",")]
            if len(parts) == 3 and not any(name in parts[1] for name in self.ignore):
                try:
                    total += float(parts[2]) / 1024
                except ValueError:
                    pass
        return total

    def run(self) -> None:
        while not self.stop.is_set():
            self.samples.append((time.perf_counter(), self.gpu_gib()))
            self.stop.wait(0.5)


def _percentile(sorted_values: list[float], p: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(p * (len(sorted_values) - 1)))]


def run_round(base: str, requests: list[dict]) -> list[dict]:
    results: list[dict] = [None] * len(requests)  # type: ignore[list-item]

    def worker(i: int, body: dict) -> None:
        results[i] = one(base, body)

    threads = [threading.Thread(target=worker, args=(i, body)) for i, body in enumerate(requests)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for r in results:
        r["round_wall_s"] = time.perf_counter() - start
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--vocab", type=int, required=True, help="the served model's vocabulary size")
    p.add_argument("--concurrency", type=int, default=4, help="requests in flight per round")
    p.add_argument("--total", type=int, default=4096, help="tokens per prompt")
    p.add_argument("--prefix", type=int, default=3500, help="tokens every request in a round shares")
    p.add_argument("--allowed", type=int, default=4, help="allowed_token_ids per request")
    p.add_argument("--reps", type=int, default=10, help="rounds of --concurrency requests each")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--mem", action="store_true")
    p.add_argument("--ignore", default="", help="comma list of process names nvidia-smi should not count")
    p.add_argument("--label", default="")
    p.add_argument("--output")
    args = p.parse_args()
    if args.prefix >= args.total:
        raise SystemExit("--prefix must be smaller than --total")

    memory = Memory(tuple(s for s in args.ignore.split(",") if s)) if args.mem else None
    idle_gib = memory.gpu_gib() if memory else None
    if memory:
        memory.start()

    all_results: list[dict] = []
    for rep in range(args.reps):
        requests = build_requests(args.model, args.vocab, args.total, args.prefix, args.concurrency, args.allowed,
                                  args.seed + rep)
        all_results.extend(run_round(args.base, requests))

    if memory:
        memory.stop.set()
        memory.join()

    ok = [r for r in all_results if not r["error"]]
    latencies = sorted(r["latency_s"] for r in ok)
    summary = {
        "label": args.label, "concurrency": args.concurrency, "reps": args.reps, "total_tokens": args.total,
        "prefix_tokens": args.prefix, "requests": len(all_results), "ok": len(ok),
        "failed": len(all_results) - len(ok),
        "latency_s_p50": round(_percentile(latencies, 0.5), 4) if latencies else None,
        "latency_s_p90": round(_percentile(latencies, 0.9), 4) if latencies else None,
        "latency_s_max": round(max(latencies), 4) if latencies else None,
        "latency_s_min": round(min(latencies), 4) if latencies else None,
    }
    if memory and memory.samples:
        summary.update(idle_gpu_gib=round(idle_gib, 2), peak_gpu_gib=round(max(s[1] for s in memory.samples), 2))
    print(json.dumps(summary), flush=True)
    if args.output:
        json.dump({"summary": summary, "results": all_results,
                   "memory": [list(s) for s in memory.samples] if memory else []}, open(args.output, "w"), indent=1)


if __name__ == "__main__":
    main()
