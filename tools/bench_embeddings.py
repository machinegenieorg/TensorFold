"""Measure an OpenAI-compatible /v1/embeddings server: throughput by batch and text length, query latency, and
whether a text's vector depends on the batch it arrives in. Works against TensorFold or vLLM.

    python3 tools/bench_embeddings.py http://127.0.0.1:8080 qwen3-embed-8b --corpus texts.jsonl --output bench.json

``--corpus`` is public text, one JSON object a line with a "text" field (or "title" and "text", as BEIR's
corpus.jsonl), or a JSON list of strings; without it the repository's own documentation is used. Texts of an exact
token length come from ``truncate_prompt_tokens``, which keeps a text's start (the server reports the count).
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import re
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

QUERY = ("Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
         "How does a GPU multiply matrices?")


def post(url: str, body: dict, timeout: float = 600) -> dict:
    request = urllib.request.Request(url.rstrip("/") + "/v1/embeddings", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def vectors(reply: dict) -> list[np.ndarray]:
    out = []
    for item in sorted(reply["data"], key=lambda d: d["index"]):
        e = item["embedding"]
        out.append(np.frombuffer(base64.b64decode(e), dtype="<f4") if isinstance(e, str) else np.array(e, np.float32))
    return out


def corpus(path: str | None) -> list[str]:
    if path is None:
        root = Path(__file__).resolve().parents[1]
        files = [root / "README.md", root / "RUNBOOK.md", *sorted((root / "docs").rglob("*.md"))]
        return [p.strip() for f in files for p in re.split(r"\n\s*\n", f.read_text()) if len(p.strip()) > 80]
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        return [t for t in json.loads(text) if isinstance(t, str) and t.strip()]
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [(r.get("title", "") + "\n" + r.get("text", "")).strip() for r in rows]


def long_texts(parts: list[str], count: int, tokens: int, seed: int) -> list[str]:
    """``count`` distinct texts of at least ``tokens`` tokens (about four characters a token) from the corpus."""

    rnd = random.Random(seed)
    out = []
    for _ in range(count):
        chosen, size = [], 0
        while size < tokens * 6:
            chosen.append(rnd.choice(parts))
            size += len(chosen[-1])
        out.append("\n\n".join(chosen))
    return out


def throughput(url: str, model: str, parts: list[str], lengths: list[int], batches: list[int], reps: int,
               clients: int, seed: int, base64_out: bool) -> list[dict]:
    rows = []
    for length in lengths:
        for batch in batches:
            texts = long_texts(parts, batch * clients * (reps + 1), length, seed + length + batch)

            def one(chunk: list[str]) -> tuple[float, int]:
                start = time.perf_counter()
                reply = post(url, {"model": model, "input": chunk, "truncate_prompt_tokens": length,
                                   **({"encoding_format": "base64"} if base64_out else {})})
                return time.perf_counter() - start, int(reply["usage"]["prompt_tokens"])

            one(texts[:batch])                                        # warm this shape
            walls, tokens = [], 0
            for r in range(reps):
                chunks = [texts[(1 + r * clients + c) * batch:(2 + r * clients + c) * batch] for c in range(clients)]
                start = time.perf_counter()
                with ThreadPoolExecutor(clients) as pool:
                    done = list(pool.map(one, chunks))
                walls.append(time.perf_counter() - start)
                tokens = sum(t for _, t in done)
            wall = statistics.median(walls)
            rows.append({"length": length, "batch": batch, "clients": clients, "tokens": tokens, "seconds": wall,
                         "tokens_per_s": tokens / wall, "texts_per_s": batch * clients / wall})
            print(f"length {length:5d} batch {batch:3d} clients {clients}: {tokens / wall:9.0f} tok/s "
                  f"({batch * clients / wall:7.1f} texts/s, {wall * 1000:8.1f} ms a round)", flush=True)
    return rows


def latency(url: str, model: str, reps: int) -> dict:
    post(url, {"model": model, "input": QUERY})
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        post(url, {"model": model, "input": QUERY})
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    out = {"p50_ms": statistics.median(times), "p90_ms": times[int(0.9 * (len(times) - 1))], "min_ms": times[0],
           "reps": reps}
    print(f"query latency: p50 {out['p50_ms']:.1f} ms, p90 {out['p90_ms']:.1f} ms, min {out['min_ms']:.1f} ms",
          flush=True)
    return out


def batching(url: str, model: str, parts: list[str], count: int, seed: int) -> dict:
    """Each text alone, then in shuffled batches of 2 to 64: identical bytes and the least cosine."""

    rnd = random.Random(seed)
    texts = [t[:rnd.choice([200, 800, 2000, 6000])] for t in rnd.sample(parts, min(count, len(parts)))]
    alone = [vectors(post(url, {"model": model, "input": [t], "truncate_prompt_tokens": 2000}))[0] for t in texts]
    same, total, least = 0, 0, 1.0
    for size in (2, 3, 8, 16, 32, 64):
        order = list(range(len(texts)))
        rnd.shuffle(order)
        for at in range(0, len(order), size):
            group = order[at:at + size]
            body = {"model": model, "input": [texts[i] for i in group], "truncate_prompt_tokens": 2000}
            got = vectors(post(url, body))
            for i, v in zip(group, got):
                total += 1
                same += v.tobytes() == alone[i].tobytes()
                a, b = v.astype(np.float64), alone[i].astype(np.float64)
                least = min(least, float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
    print(f"batching: {same}/{total} vectors identical to the text alone; least cosine {least:.8f}", flush=True)
    return {"identical": same, "total": total, "least_cosine": least}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("url")
    parser.add_argument("model")
    parser.add_argument("--corpus", default=None, help="public text: JSONL with text (and title), or a JSON list")
    parser.add_argument("--lengths", default="100,500,1000,2000")
    parser.add_argument("--batches", default="1,8,32")
    parser.add_argument("--clients", type=int, default=1, help="concurrent clients sending batches")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--latency-reps", type=int, default=50)
    parser.add_argument("--batching", type=int, default=64, help="texts for the batch-dependence check (0: skip)")
    parser.add_argument("--base64", action="store_true", help="ask for base64 vectors in the throughput runs")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    parts = corpus(args.corpus)
    result = {"url": args.url, "model": args.model, "corpus_texts": len(parts)}
    result["latency"] = latency(args.url, args.model, args.latency_reps)
    result["throughput"] = throughput(args.url, args.model, parts, [int(x) for x in args.lengths.split(",")],
                                      [int(x) for x in args.batches.split(",")], args.reps, args.clients, args.seed,
                                      args.base64)
    if args.batching:
        result["batching"] = batching(args.url, args.model, parts, args.batching, args.seed)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
