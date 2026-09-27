#!/usr/bin/env python3
"""Follow-up calls resume from the end of the call they extend, and stay exact.

A second pass over the same document appends to the first call's user message ("already found: ...; what else?").
Each follow-up should resume from near the end of the call it extends (``tensorfold.cached``), and reply exactly as
it does on a server that keeps no such states.

  followup_check.py PROMPTS.json [--url URL] [--n 8] [--save F]      # with the follow-up cache
  followup_check.py PROMPTS.json [--url URL] --compare F             # on a server started with --followup-gib 0"""

from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    return d["choices"][0]["message"].get("content") or "", d.get("tensorfold") or {}, d["usage"]["prompt_tokens"]


def followup(m, text):
    m2 = [dict(x) for x in m]
    m2[-1]["content"] += ("\n\nALREADY EXTRACTED FROM THIS EVENT: do not repeat these. Extract only what is "
                          "missing.\n- " + text[:200].replace("\n", " "))
    return m2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("--url", default="http://127.0.0.1:8891/v1/chat/completions")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--save", help="write the follow-ups and their replies here")
    ap.add_argument("--compare", help="replay follow-ups saved by --save (serially) and compare the replies")
    a = ap.parse_args()
    convs = json.load(open(a.prompts))
    base = {"model": "qwen3.8-27b", "max_tokens": 96, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    run = lambda ms, **kw: list(ThreadPoolExecutor(len(ms)).map(lambda m: post(a.url, {**base, "messages": m, **kw}), ms))
    if a.compare:                               # replay saved follow-ups one at a time, no drafts: same replies?
        saved = json.load(open(a.compare))
        same = sum(post(a.url, {**base, "messages": x["messages"], "draft": False})[0] == x["reply"] for x in saved)
        print(json.dumps({"identical_to_saved": f"{same}/{len(saved)}"}))
        return

    # set A: first calls, then their follow-ups together; each follow-up should resume near its first call's end
    A = convs[:a.n]
    first = run(A)
    second = run([followup(m, t) for m, (t, _, _) in zip(A, first)])
    near = 0
    for i, ((_, s1, p1), (_, s2, p2)) in enumerate(zip(first, second)):
        ok = s2.get("cached", 0) >= p1 - 40
        near += ok
        print(f"A{i}: first prompt {p1} (resumed {s1.get('cached')}), follow-up prompt {p2} (resumed {s2.get('cached')})"
              f"{'' if ok else ' NOT near the first call end'}", flush=True)

    print(json.dumps({"resumed_near_first_end": f"{near}/{len(A)}"}))
    if a.save:                                  # follow-ups and their replies, to replay on a server without the cache
        json.dump([{"messages": followup(m, t), "reply": r[0]} for m, (t, _, _), r in zip(A, first, second)],
                  open(a.save, "w"))


if __name__ == "__main__":
    main()
