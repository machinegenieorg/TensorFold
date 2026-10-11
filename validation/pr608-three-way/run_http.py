"""Preserve raw SSE evidence for the fresh three-way PR608 greedy code/chat comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import sys
import time
from typing import Any

import bench_concurrent as bench


SCHEMA = "pr608_three_way_http_v1"
PARAMETERS = {
    "prompts": ["code", "chat"],
    "levels": [1, 16],
    "repetitions": 3,
    "uncached_first_use_tokens": 256,
    "measured_tokens": 256,
    "temperature": 0.0,
    "seed": 1234,
}
EXPECTED_PROMPTS = {
    "code": ("completion", "Write a short Python function that computes the Fibonacci sequence and explain it.", 14),
    "chat": ("chat", "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.", 31),
}


def sha(path: Path) -> str:
    """Hash the exact source bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    """Bind the runner and its copied SSE/evidence helpers."""
    root = Path(__file__).resolve().parent
    return {name: sha(root / name) for name in ("run_http.py", "bench_concurrent.py", "bench_evidence.py")}


def prompts() -> dict[str, dict[str, Any]]:
    """Require the upstream prompt text, kind and order exactly."""
    items = bench.PROMPTS
    if [item.get("name") for item in items] != PARAMETERS["prompts"]:
        raise ValueError("original prompt order changed")
    for item in items:
        kind, text, _ = EXPECTED_PROMPTS[item["name"]]
        if item.get("kind") != kind or item.get("prompt") != text:
            raise ValueError("original prompt text or route changed")
    return {item["name"]: item for item in items}


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically preserve partial evidence after each burst."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def expected_body(model: str, item: dict[str, Any], tokens: int) -> tuple[str, dict[str, Any]]:
    """Rebuild the unchanged upstream greedy request body and route."""
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "seed": 1234,
    }
    if item["kind"] == "chat":
        body["messages"] = [{"role": "user", "content": item["prompt"]}]
        body["chat_template_kwargs"] = {"enable_thinking": False}
        return "/v1/chat/completions", body
    body["prompt"] = item["prompt"]
    return "/v1/completions", body


def validate_request(
    row: dict[str, Any], *, item: dict[str, Any], model: str, base: str,
    tokens: int, cached: int,
) -> None:
    """Require one complete SSE response with literal work, cache and clock evidence."""
    if row.get("error") or row.get("complete") is not True or row.get("unmeasured"):
        raise ValueError("failed, incomplete or unmeasured SSE request")
    if row.get("prompt") != item["name"] or row.get("seed") != 1234:
        raise ValueError("prompt/seed drift")
    suffix, body = expected_body(model, item, tokens)
    if row.get("request_body") != body or row.get("request_url") != base + suffix:
        raise ValueError("literal HTTP request drift")
    if row.get("request_body_sha256") != hashlib.sha256(json.dumps(body).encode()).hexdigest():
        raise ValueError("HTTP body hash drift")
    if row.get("finish_reason") != "length" or row.get("tokens") != tokens:
        raise ValueError("response did not complete forced output length")
    usage = row.get("usage")
    _, _, prompt_tokens = EXPECTED_PROMPTS[item["name"]]
    if not isinstance(usage, dict) or usage.get("prompt_tokens") != prompt_tokens:
        raise ValueError("prompt token count drift")
    if usage.get("completion_tokens") != tokens or row.get("prompt_tokens") != prompt_tokens:
        raise ValueError("completion/prompt usage drift")
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict) or details.get("cached_tokens") != cached:
        raise ValueError("actual cached-token count differs from predeclared arm")
    if row.get("cached_tokens") != cached or row.get("cache_state") != ("reused" if cached else "cold"):
        raise ValueError("normalized cache state drift")
    if not isinstance(row.get("response_id"), str) or not row["response_id"]:
        raise ValueError("missing response identity")
    if not isinstance(row.get("output_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", row["output_sha256"]):
        raise ValueError("missing text output hash")
    if not isinstance(row.get("token_sha"), str) or not re.fullmatch(r"(?:[0-9a-f]{12}|[0-9a-f]{64})", row["token_sha"]):
        raise ValueError("missing token fingerprint")
    runtime = row.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("token_sha") != row["token_sha"]:
        raise ValueError("runtime token fingerprint mismatch")
    work = row.get("work")
    raw_work = row.get("speculative") or runtime
    origin = "speculative" if row.get("speculative") else "tensorfold"
    if row.get("work_source") != origin or work != {key: raw_work.get(key) for key in ("rounds", "accepted", "drafted")}:
        raise ValueError("speculative work/raw provenance mismatch")
    if not isinstance(work, dict) or any(type(work.get(key)) is not int or work[key] < 0
                                         for key in ("rounds", "accepted", "drafted")):
        raise ValueError("missing actual speculative work")
    if work["rounds"] <= 0 or work["accepted"] > work["drafted"]:
        raise ValueError("invalid speculative work")
    clocks = [row.get(key) for key in ("sent", "first", "last", "done")]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in clocks):
        raise ValueError("missing SSE or DONE timestamp")
    sent, first, last, done = clocks
    if not sent < first < last <= done:
        raise ValueError("nonmonotonic SSE timestamps")
    for key, actual in (("ttft_s", first - sent), ("full_response_s", done - sent),
                        ("decode_tps", (tokens - 1) / (last - first))):
        value = row.get(key)
        if type(value) not in (int, float) or not math.isclose(value, actual, rel_tol=1e-12):
            raise ValueError("SSE timing arithmetic mismatch")
    pieces = row.get("pieces")
    if not isinstance(pieces, list) or len(pieces) < bench.MIN_PIECES or pieces[0][0] != first or pieces[-1][0] != last:
        raise ValueError("missing nonempty SSE pieces")
    previous = sent
    for arrived, characters in pieces:
        if type(arrived) not in (int, float) or not previous <= arrived <= done:
            raise ValueError("invalid SSE piece arrival")
        if type(characters) is not int or characters <= 0:
            raise ValueError("empty or invalid SSE piece")
        previous = arrived


def validate_report(report: dict[str, Any]) -> dict[str, int]:
    """Validate two uncached first uses and 102 primed requests, all 256 outputs."""
    if report.get("schema") != SCHEMA or report.get("complete") is not True:
        raise ValueError("incomplete three-way comparison arm")
    if report.get("parameters") != PARAMETERS or report.get("source_sha256") != source_hashes():
        raise ValueError("parameters or source changed")
    label = report.get("label")
    if label not in ("stock", "candidate", "python"):
        raise ValueError("expected stock, candidate or Python label")
    items = prompts()
    cells = report.get("cells", [])
    if [cell.get("prompt") for cell in cells] != PARAMETERS["prompts"]:
        raise ValueError("missing prompt cell")
    ids: set[str] = set()
    requests = 0
    total_prompt = 0
    total_output = 0
    for cell in cells:
        item = items[cell["prompt"]]
        prompt_n = EXPECTED_PROMPTS[item["name"]][2]
        expected_cached = 0 if label == "stock" else prompt_n - 1 if label == "python" else prompt_n
        uncached_first_use = cell.get("uncached_first_use")
        if not isinstance(uncached_first_use, dict):
            raise ValueError("missing 256-token uncached first-use request")
        validate_request(uncached_first_use, item=item, model=report["model"], base=report["base"], tokens=256, cached=0)
        ids.add(uncached_first_use["response_id"])
        requests += 1
        total_prompt += uncached_first_use["prompt_tokens"]
        total_output += 256
        levels = cell.get("levels", [])
        if [level.get("streams") for level in levels] != [1, 16]:
            raise ValueError("original n1/n16 order changed")
        for level in levels:
            bursts = level.get("bursts", [])
            if len(bursts) != 3:
                raise ValueError("missing measured repetition")
            for index, burst in enumerate(bursts):
                if (burst.get("index") != index
                        or burst.get("phase") != ("primed_first_burst" if index == 0 else "repeat")):
                    raise ValueError("repetition order changed")
                rows = burst.get("requests", [])
                if len(rows) != level["streams"]:
                    raise ValueError("missing simultaneous request")
                for row in rows:
                    validate_request(row, item=item, model=report["model"], base=report["base"],
                                     tokens=256, cached=expected_cached)
                    ids.add(row["response_id"])
                    requests += 1
                    total_prompt += row["prompt_tokens"]
                    total_output += 256
    if requests != 104 or len(ids) != 104 or total_output != 26624:
        raise ValueError("request identity/count drift")
    return {"requests": requests, "prompt_tokens": total_prompt, "output_tokens": total_output}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Derive TTFT, decode-span rate and complete wall time from retained clocks."""
    return {
        "median_ttft_s": statistics.median(row["ttft_s"] for row in rows),
        "slowest_ttft_s": max(row["ttft_s"] for row in rows),
        "median_decode_tps": statistics.median(row["decode_tps"] for row in rows),
        "minimum_decode_tps": min(row["decode_tps"] for row in rows),
        "aggregate_decode_tps": sum(row["tokens"] - 1 for row in rows)
        / (max(row["last"] for row in rows) - min(row["first"] for row in rows)),
        "median_full_response_s": statistics.median(row["full_response_s"] for row in rows),
        "batch_wall_s": max(row["done"] for row in rows) - min(row["sent"] for row in rows),
    }


def compare_three_reports(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Require all 104 literal request/output pairs for all three backends."""
    if set(reports) != {"stock", "python", "candidate"}:
        raise ValueError("missing three-way arm")
    for label, report in reports.items():
        validate_report(report)
        if report["label"] != label:
            raise ValueError("three-way arm identity mismatch")
    observations = []
    unequal_work = {"stock_python": 0, "stock_candidate": 0, "python_candidate": 0}
    for cell_index, name in enumerate(PARAMETERS["prompts"]):
        cells = {arm: report["cells"][cell_index] for arm, report in reports.items()}
        groups = [("uncached_first_use", 1, None,
                   {arm: [cell["uncached_first_use"]] for arm, cell in cells.items()})]
        for level_index, n in enumerate(PARAMETERS["levels"]):
            for burst_index in range(PARAMETERS["repetitions"]):
                groups.append(("primed_first_burst" if burst_index == 0 else "repeat", n,
                               burst_index, {arm: cell["levels"][level_index]["bursts"]
                                             [burst_index]["requests"] for arm, cell in cells.items()}))
        for phase, n, repetition, grouped in groups:
            for left_arm, right_arm in (("stock", "python"), ("stock", "candidate"),
                                        ("python", "candidate")):
                for left, right in zip(grouped[left_arm], grouped[right_arm]):
                    if (left["request_body"] != right["request_body"] or
                            left["request_body_sha256"] != right["request_body_sha256"]):
                        raise ValueError("three-way literal request mismatch")
                    if (left["output_sha256"] != right["output_sha256"] or
                            left["token_sha"] != right["token_sha"]):
                        raise ValueError("three-way output mismatch")
                    unequal_work[f"{left_arm}_{right_arm}"] += left["work"] != right["work"]
            observations.append({"prompt": name, "streams": n, "phase": phase,
                                 "repetition": repetition,
                                 "arms": {arm: summarize_rows(rows) for arm, rows in grouped.items()}})
    return {"schema": "pr608_three_way_http_comparison_v1", "complete": True,
            "requests_per_backend": 104, "total_requests": 312,
            "uncached_first_use_per_backend": 2, "primed_requests_per_backend": 102,
            "unequal_speculative_work_requests": unequal_work, "observations": observations}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run one uncached prompt request, then the unchanged primed n1/n16 panel."""
    items = prompts()
    base = args.base.rstrip("/")
    report: dict[str, Any] = {
        "schema": SCHEMA, "complete": False, "label": args.label, "model": args.model,
        "base": base, "parameters": PARAMETERS, "source_sha256": source_hashes(), "cells": [],
    }
    write_report(args.output, report)
    deadline = time.monotonic() + args.max_seconds
    for name in PARAMETERS["prompts"]:
        item = items[name]
        cell: dict[str, Any] = {"prompt": name, "uncached_first_use": None, "levels": []}
        report["cells"].append(cell)
        if deadline - time.monotonic() <= 0:
            raise TimeoutError("client deadline exceeded")
        bench.REQUEST_TIMEOUT = min(1800, deadline - time.monotonic())
        cell["uncached_first_use"] = bench.together(base, args.model, [(item, 1234)], 256, 0.0)[0]
        validate_request(cell["uncached_first_use"], item=item, model=args.model, base=base, tokens=256, cached=0)
        cell["uncached_first_use_metrics"] = summarize_rows([cell["uncached_first_use"]])
        write_report(args.output, report)
        for streams in PARAMETERS["levels"]:
            level: dict[str, Any] = {"streams": streams, "bursts": []}
            cell["levels"].append(level)
            for index in range(3):
                if deadline - time.monotonic() <= 0:
                    raise TimeoutError("client deadline exceeded")
                bench.REQUEST_TIMEOUT = min(1800, deadline - time.monotonic())
                rows = bench.together(base, args.model, [(item, 1234)] * streams, 256, 0.0)
                burst = {"index": index, "phase": "primed_first_burst" if index == 0 else "repeat",
                         "requests": rows, "aggregate": bench.aggregates(rows)}
                level["bursts"].append(burst)
                write_report(args.output, report)
                prompt_n = EXPECTED_PROMPTS[name][2]
                expected_cached = 0 if args.label == "stock" else prompt_n - 1 if args.label == "python" else prompt_n
                for row in rows:
                    validate_request(row, item=item, model=args.model, base=base, tokens=256, cached=expected_cached)
                burst["metrics"] = summarize_rows(rows)
                write_report(args.output, report)
                print(json.dumps({"label": args.label, "prompt": name, "streams": streams,
                                  "repetition": index, "median_ttft_s": statistics.median(row["ttft_s"] for row in rows),
                                  "slowest_ttft_s": max(row["ttft_s"] for row in rows)}), flush=True)
    report["complete"] = True
    report["health_totals"] = validate_report(report)
    write_report(args.output, report)
    return report


def main() -> None:
    """Run one fresh three-way arm with an explicit output and bounded deadline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", choices=("stock", "candidate", "python"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=1200)
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error("max-seconds must be positive")

    def expired(signum: int, frame: Any) -> None:
        """Retain the partial report when the whole client window expires."""
        print("three-way panel deadline exceeded; partial report retained", file=sys.stderr, flush=True)
        os._exit(1)

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.max_seconds)
    try:
        run(args)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}; partial report retained", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
