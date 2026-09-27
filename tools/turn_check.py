#!/usr/bin/env python3
"""A growing conversation (an agent loop) resumes from the previous call's state; stop sequences; /health.

  turn_check.py PROMPTS.json [--url http://127.0.0.1:8891] [--burst 24]"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def call(base, body):
    t = time.perf_counter()
    first, usage, text, finish = None, None, [], None
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps({**body, "stream": True,
                                 "stream_options": {"include_usage": True}}).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            ev = json.loads(line[6:])
            usage = ev.get("usage") or usage
            for c in ev.get("choices") or []:
                d = (c.get("delta") or {}).get("content")
                if d:
                    first = first or time.perf_counter() - t
                    text.append(d)
                finish = c.get("finish_reason") or finish
    return "".join(text), finish, (usage or {}).get("prompt_tokens"), round(first or -1, 1), round(time.perf_counter() - t, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("--url", default="http://127.0.0.1:8891")
    ap.add_argument("--burst", type=int, default=24)
    a = ap.parse_args()
    msgs = json.load(open(a.prompts))
    block = "\n\n".join(x["content"] for m in msgs[30:40] for x in m)[:100000]      # a long fixed context
    conv = [{"role": "system", "content": block}, {"role": "user", "content": "List three dates mentioned above."}]
    base = {"model": "qwen3.8-27b", "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": False}}
    for turn in range(3):
        text, finish, pt, ttft, total = call(a.url, {**base, "messages": conv})
        print(f"turn {turn}: prompt {pt} tokens, first token {ttft}s, total {total}s, finish {finish}", flush=True)
        conv = conv + [{"role": "assistant", "content": text}, {"role": "user", "content": f"Now give one more date (turn {turn + 1})."}]
    # a new conversation over the same long context (a new agent call): resumes from the shared prefix
    for n in range(2):
        other = [{"role": "system", "content": block}, {"role": "user", "content": f"Name two people mentioned above ({n})."}]
        _, finish, pt, ttft, total = call(a.url, {**base, "messages": other})
        print(f"new conversation {n}: prompt {pt} tokens, first token {ttft}s, total {total}s", flush=True)
    # a burst of other work (short prompts sharing their own prefixes) must not push the long block out
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda m: call(a.url, {**base, "max_tokens": 16, "messages": m}), msgs[:a.burst]))
    other = [{"role": "system", "content": block}, {"role": "user", "content": "Name one organisation mentioned above."}]
    _, finish, pt, ttft, total = call(a.url, {**base, "messages": other})
    print(f"after a burst of {a.burst}: new conversation, prompt {pt} tokens, first token {ttft}s, total {total}s", flush=True)
    text, finish, _, _, _ = call(a.url, {**base, "messages": [{"role": "user", "content": "Count from 1 to 20, one number per line."}],
                                         "stop": ["\n7"]})
    print(f"stop ['\\n7']: finish {finish}, ends with {text[-12:]!r}, contains '7': {'7' in text}", flush=True)
    print("health:", urllib.request.urlopen(a.url + "/health", timeout=10).read().decode())


if __name__ == "__main__":
    main()
