#!/usr/bin/env python3
"""Structured output over HTTP: each constrained reply must parse, follow its schema, and equal its own serial
("draft": false) reply when drafted among other requests.

  grammar_check.py PROMPTS.json [--url URL] [--k 6] [--load 10]"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["meeting", "email", "document", "chat", "other"]},
        "when": {"type": "string", "enum": ["past", "upcoming", "undated"]},
        "people": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "summary": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["kind", "when", "people", "summary", "confidence"],
    "additionalProperties": False,
}
ASK = ("Label the event above. Reply with JSON only: kind, when (relative to its own date), up to five people, "
       "a one-sentence summary, and your confidence from 0 to 1.")


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]["message"].get("content") or ""


def valid(text, schema):
    try:
        o = json.loads(text)
    except Exception:
        return False
    if schema is None:
        return isinstance(o, (dict, list))
    if set(o) != set(schema["required"]):
        return False
    p = schema["properties"]
    return (o["kind"] in p["kind"]["enum"] and o["when"] in p["when"]["enum"] and isinstance(o["people"], list)
            and len(o["people"]) <= 5 and isinstance(o["summary"], str) and isinstance(o["confidence"], (int, float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("--url", default="http://127.0.0.1:8891/v1/chat/completions")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--load", type=int, default=10)
    a = ap.parse_args()
    msgs = json.load(open(a.prompts))
    reqs = []
    for i, m in enumerate(msgs[:a.k]):
        m = m + [{"role": "user", "content": ASK}]
        base = {"model": "qwen3.8-27b", "messages": m, "max_tokens": 300, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        if i % 3 == 2:
            reqs.append((f"json_object-{i}", {**base, "response_format": {"type": "json_object"}}, None))
        elif i % 3 == 1:
            reqs.append((f"guided_json-{i}", {**base, "guided_json": SCHEMA}, SCHEMA))
        else:
            reqs.append((f"json_schema-{i}", {**base, "response_format": {"type": "json_schema",
                         "json_schema": {"name": "label", "schema": SCHEMA}}}, SCHEMA))
    serial = {rid: post(a.url, {**b, "draft": False}) for rid, b, _ in reqs}
    load = [{"model": "qwen3.8-27b", "messages": m, "max_tokens": 200, "temperature": 0,
             "chat_template_kwargs": {"enable_thinking": False}} for m in msgs[a.k:a.k + a.load]]
    with ThreadPoolExecutor(len(reqs) + len(load)) as ex:
        bg = [ex.submit(post, a.url, b) for b in load]
        futs = {rid: ex.submit(post, a.url, b) for rid, b, _ in reqs}
        drafted = {k: f.result() for k, f in futs.items()}
        [f.result() for f in bg]
    same = ok = 0
    for rid, _, schema in reqs:
        s, d = serial[rid], drafted[rid]
        same += s == d
        ok += valid(d, schema)
        print(f"{rid}: {'IDENTICAL' if s == d else 'DIFFERENT'}, valid={valid(d, schema)}: {d[:120]!r}", flush=True)
    print(json.dumps({"identical": f"{same}/{len(reqs)}", "valid": f"{ok}/{len(reqs)}"}))
    sys.exit(0 if same == ok == len(reqs) else 1)


if __name__ == "__main__":
    main()
