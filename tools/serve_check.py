#!/usr/bin/env python3
"""Check a TensorFold server over HTTP: exactness under load, streaming, cancellation.

  serve_check.py ref  PROMPTS.json OUT.json [--url URL] [--k 6] [--max-tokens 200]
      the reference: each request alone with "draft": false (greedy, and seeded sampling)
  serve_check.py load PROMPTS.json REF.json [--url URL] [--load 16]
      the same requests all at once, drafted, among --load other streaming requests; one extra stream is
      dropped mid-way. Every reply must equal its reference."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def stream(url, body, timeout=1800, drop_after=None):
    req = urllib.request.Request(url, json.dumps({**body, "stream": True}).encode(), {"content-type": "application/json"})
    text, n = [], 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            ev = json.loads(line[6:])
            for c in ev.get("choices") or []:
                text.append((c.get("delta") or {}).get("content") or "")
            n += 1
            if drop_after is not None and n >= drop_after:
                r.fp.raw._sock.shutdown(socket.SHUT_RDWR)       # a client that goes away mid-reply
                return None
    return "".join(text)


def cases(msgs, k, max_tokens):
    out = []
    for i, m in enumerate(msgs[:k]):
        base = {"model": "qwen3.8-27b", "messages": m, "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False}}
        out.append({"id": f"greedy-{i}", "body": {**base, "temperature": 0}})
        out.append({"id": f"sampled-{i}", "body": {**base, "temperature": 0.7, "top_p": 0.95, "top_k": 20, "seed": 1000 + i}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("ref", "load"))
    ap.add_argument("prompts")
    ap.add_argument("ref")
    ap.add_argument("--url", default="http://127.0.0.1:8891/v1/chat/completions")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--load", type=int, default=16)
    a = ap.parse_args()
    msgs = json.load(open(a.prompts))
    if a.mode == "ref":
        res = {}
        for c in cases(msgs, a.k, a.max_tokens):
            t = time.perf_counter()
            r = post(a.url, {**c["body"], "draft": False})
            res[c["id"]] = r["choices"][0]["message"].get("content") or ""
            print(f"{c['id']}: {len(res[c['id']])} chars in {time.perf_counter() - t:.1f}s", flush=True)
        json.dump(res, open(a.ref, "w"), indent=1)
        return
    ref = json.load(open(a.ref))
    todo = cases(msgs, a.k, a.max_tokens)
    load = [{"model": "qwen3.8-27b", "messages": m, "max_tokens": 300, "temperature": 0,
             "chat_template_kwargs": {"enable_thinking": False}} for m in msgs[a.k:a.k + a.load]]
    t = time.perf_counter()
    with ThreadPoolExecutor(len(todo) + len(load) + 1) as ex:
        bg = [ex.submit(stream, a.url, b) for b in load]
        dropped = ex.submit(stream, a.url, load[0] if load else todo[0]["body"], drop_after=5)
        # half the checked requests stream, half do not
        futs = {c["id"]: ex.submit(stream if n % 2 else lambda u, b: post(u, b)["choices"][0]["message"].get("content") or "",
                                   a.url, c["body"]) for n, c in enumerate(todo)}
        got = {k: f.result() for k, f in futs.items()}
        bg_done = sum(1 for f in bg if f.result() is not None)
        dropped.result()
    ok = 0
    for k, v in got.items():
        same = v == ref[k]
        ok += same
        print(f"{k}: {'IDENTICAL' if same else 'DIFFERENT'} ({len(v)} chars)", flush=True)
    print(json.dumps({"identical": f"{ok}/{len(got)}", "background_completed": f"{bg_done}/{len(load)}",
                      "wall_s": round(time.perf_counter() - t, 1)}))
    sys.exit(0 if ok == len(got) else 1)


if __name__ == "__main__":
    main()
