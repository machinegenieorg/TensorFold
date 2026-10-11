"""Reconstruct three-way metrics from sanitized portable request evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROMPTS = {
    "code": ("/v1/completions", "Write a short Python function that computes the Fibonacci sequence and explain it.", 14),
    "chat": ("/v1/chat/completions", "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.", 31),
}
ARMS = ("stock", "python", "candidate")
ORDERS = (("python", "stock", "candidate"), ("candidate", "stock", "python"))
PARAMETERS = {
    "prompts": ["code", "chat"], "levels": [1, 16], "repetitions": 3,
    "uncached_first_use_tokens": 256, "measured_tokens": 256,
    "temperature": 0.0, "seed": 1234,
}


def require(condition: bool, message: str) -> None:
    """Reject incomplete or inconsistent evidence."""
    if not condition:
        raise ValueError(message)


def digest(value: bytes) -> str:
    """Hash literal bytes."""
    return hashlib.sha256(value).hexdigest()


def expected_body(model: str, prompt: str) -> tuple[str, dict[str, Any]]:
    """Build the fixed upstream greedy request without client imports."""
    route, text, _ = PROMPTS[prompt]
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "seed": 1234,
    }
    if prompt == "chat":
        body["messages"] = [{"role": "user", "content": text}]
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        body["prompt"] = text
    return route, body


def check_request(row: dict[str, Any], *, prompt: str, arm: str, cold: bool, model: str) -> None:
    """Require literal request, usage, work, output and SSE clock evidence."""
    route, body = expected_body(model, prompt)
    require(row.get("complete") is True and not row.get("error") and not row.get("unmeasured"), "incomplete request")
    require(row.get("prompt") == prompt and row.get("seed") == 1234, "prompt/seed drift")
    require(row.get("request_body") == body, "body drift")
    require(row.get("request_body_sha256") == digest(json.dumps(body).encode()), "body SHA drift")
    require(urlparse(row.get("request_url", "")).path == route, "route drift")
    require(row.get("tokens") == 256 and row.get("finish_reason") == "length", "output length/finish drift")
    prompt_n = PROMPTS[prompt][2]
    cached = 0 if cold or arm == "stock" else prompt_n - 1 if arm == "python" else prompt_n
    usage = row.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    require((row.get("prompt_tokens"), usage.get("prompt_tokens"), usage.get("completion_tokens"))
            == (prompt_n, prompt_n, 256), "token usage drift")
    require((row.get("cached_tokens"), details.get("cached_tokens")) == (cached, cached), "cache count drift")
    require(row.get("cache_state") == ("reused" if cached else "cold"), "cache state drift")
    require(isinstance(row.get("response_id"), str) and row["response_id"], "response ID missing")
    for field, lengths in (("output_sha256", {64}), ("token_sha", {12, 64})):
        value = row.get(field)
        require(isinstance(value, str) and len(value) in lengths
                and all(c in "0123456789abcdef" for c in value), f"{field} missing")
    runtime = row.get("runtime") or {}
    require(runtime.get("token_sha") == row["token_sha"], "runtime token SHA drift")
    raw_work = row.get("speculative") or runtime
    work = row.get("work") or {}
    require(row.get("work_source") == ("speculative" if row.get("speculative") else "tensorfold"), "work origin drift")
    require(work == {key: raw_work.get(key) for key in ("rounds", "accepted", "drafted")}, "work raw drift")
    require(all(type(work.get(key)) is int and work[key] >= 0 for key in ("rounds", "accepted", "drafted"))
            and work["rounds"] > 0 and work["accepted"] <= work["drafted"], "invalid work")
    clocks = [row.get(key) for key in ("sent", "first", "last", "done")]
    require(all(type(value) in (int, float) and math.isfinite(value) for value in clocks), "clock missing")
    sent, first, last, done = clocks
    require(sent < first < last <= done, "clock order drift")
    pieces = row.get("pieces")
    require(isinstance(pieces, list) and len(pieces) >= 4
            and pieces[0][0] == first and pieces[-1][0] == last, "SSE pieces missing")
    previous = sent
    for arrived, characters in pieces:
        require(type(arrived) in (int, float) and math.isfinite(arrived)
                and previous <= arrived <= done and type(characters) is int
                and characters > 0, "SSE piece drift")
        previous = arrived
    for field, value in (("ttft_s", first - sent), ("full_response_s", done - sent),
                         ("decode_tps", 255 / (last - first))):
        observed = row.get(field)
        require(type(observed) in (int, float) and math.isclose(observed, value, rel_tol=1e-12),
                f"{field} arithmetic drift")


def report_rows(report: dict[str, Any]) -> dict[tuple[str, int, int | None, int], dict[str, Any]]:
    """Flatten one fresh process's cold first use and three warm bursts."""
    require(report.get("schema") == "pr608_three_way_http_v1" and report.get("complete") is True, "report incomplete")
    require(report.get("parameters") == PARAMETERS, "parameters drift")
    arm = report.get("label")
    require(arm in ARMS, "arm label drift")
    cells = report.get("cells")
    require(isinstance(cells, list) and [cell.get("prompt") for cell in cells] == list(PROMPTS), "prompt order drift")
    model = report.get("model")
    require(isinstance(model, str) and model, "model missing")
    rows: dict[tuple[str, int, int | None, int], dict[str, Any]] = {}
    ids: set[str] = set()
    for cell in cells:
        prompt = cell["prompt"]
        cold = cell.get("uncached_first_use")
        require(isinstance(cold, dict), "uncached first use missing")
        check_request(cold, prompt=prompt, arm=arm, cold=True, model=model)
        rows[(prompt, 1, None, 0)] = cold
        require(cold["response_id"] not in ids, "duplicate cold response ID")
        ids.add(cold["response_id"])
        levels = cell.get("levels")
        require(isinstance(levels, list) and [level.get("streams") for level in levels] == [1, 16], "levels drift")
        for level in levels:
            n = level["streams"]
            bursts = level.get("bursts")
            require(isinstance(bursts, list) and len(bursts) == 3, "burst count drift")
            for b, burst in enumerate(bursts):
                require(burst.get("index") == b, "burst index drift")
                require(burst.get("phase") == ("primed_first_burst" if b == 0 else "repeat"),
                        "burst phase drift")
                requests = burst.get("requests")
                require(isinstance(requests, list) and len(requests) == n, "request count drift")
                for i, row in enumerate(requests):
                    check_request(row, prompt=prompt, arm=arm, cold=False, model=model)
                    rows[(prompt, n, b, i)] = row
                    require(row["response_id"] not in ids, "duplicate response ID")
                    ids.add(row["response_id"])
    require(len(rows) == len(ids) == 104, "104-request panel drift")
    require(all(row["sent"] > rows[("code", 1, None, 0)]["done"]
                for key, row in rows.items() if key[0] == "code" and key[2] is not None),
            "code cold request was not first")
    require(rows[("chat", 1, None, 0)]["sent"]
            > max(row["done"] for key, row in rows.items() if key[0] == "code"),
            "chat first use did not follow code panel")
    for prompt in PROMPTS:
        previous = rows[(prompt, 1, None, 0)]["done"]
        for n in (1, 16):
            for b in range(3):
                group = [rows[(prompt, n, b, i)] for i in range(n)]
                require(min(row["sent"] for row in group) > previous, "burst ordering drift")
                previous = max(row["done"] for row in group)
    totals = report.get("health_totals") or {}
    require(totals.get("requests") == 104 and totals.get("prompt_tokens") == 2340
            and totals.get("output_tokens") == 26624, "report aggregate drift")
    return rows


def burst_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute client timing and decode metrics for one literal burst."""
    return {
        "median_ttft_s": statistics.median(row["ttft_s"] for row in rows),
        "slowest_ttft_s": max(row["ttft_s"] for row in rows),
        "batch_wall_s": max(row["done"] for row in rows) - min(row["sent"] for row in rows),
        "median_per_stream_decode_tps": statistics.median(row["decode_tps"] for row in rows),
        "aggregate_decode_tps": sum(row["tokens"] - 1 for row in rows)
        / (max(row["last"] for row in rows) - min(row["first"] for row in rows)),
    }


def phase_metrics(rows: dict[tuple[str, int, int | None, int], dict[str, Any]],
                  prompt: str, n: int) -> dict[str, Any]:
    """Separate uncached first use from primed first/repeated bursts."""
    cold = rows[(prompt, 1, None, 0)]
    bursts = [burst_metrics([rows[(prompt, n, b, i)] for i in range(n)]) for b in range(3)]
    fields = bursts[0].keys()
    return {
        "uncached_first_use": {key: cold[key] for key in ("ttft_s", "full_response_s", "decode_tps")},
        "bursts": bursts,
        "primed_first_burst": bursts[0],
        "warm_repeated_mean": {key: statistics.mean(b[key] for b in bursts[1:]) for key in fields},
        "all_three_mean": {key: statistics.mean(b[key] for b in bursts) for key in fields},
        "max_three_slowest_ttft_s": max(b["slowest_ttft_s"] for b in bursts),
    }


def geometric_ratio(values: list[float]) -> float:
    """Balance two within-order ratios without treating requests as replicates."""
    require(len(values) == 2 and all(value > 0 for value in values), "ratio count drift")
    return math.sqrt(values[0] * values[1])


def read_order(root: Path, label: str, expected: tuple[str, ...],
               provenance: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one derived receipt and reconstruct reports in memory."""
    receipt = json.loads((root / f"evidence-{label}.json").read_text())
    require(receipt.get("schema") == "pr608_three_way_public_order_v1"
            and receipt.get("sanitized") is True and receipt.get("complete") is True,
            "wrong or incomplete public receipt")
    require(receipt["process_order"] == list(expected), "process order drift")
    info = receipt["requests"]
    require(info["file"] == f"requests-{label}.jsonl", "request file drift")
    payload = (root / info["file"]).read_bytes()
    require(digest(payload) == info["sha256"], "request file hash drift")
    records = [json.loads(line) for line in payload.decode().splitlines()]
    require(len(records) == info["count"] == 312, "312-request evidence drift")
    require(set(receipt["reports"]) == set(ARMS), "arm inventory drift")
    indexed = {arm: {} for arm in ARMS}
    for record in records:
        require(record.get("schema") == "pr608_three_way_public_request_v1", "request schema drift")
        arm, prompt = record["arm"], record["prompt"]
        n, burst, stream = record["streams"], record["burst"], record["stream"]
        require(arm in ARMS and prompt in PROMPTS and type(n) is int and n in (1, 16)
                and type(stream) is int and 0 <= stream < n, "request identity drift")
        require((burst is None and n == 1 and stream == 0
                 and record["phase"] == "uncached_first_use") or
                (type(burst) is int and burst in (0, 1, 2)
                 and record["phase"] == ("primed_first_burst" if burst == 0 else "repeat")),
                "request phase drift")
        key = (prompt, n, burst, stream)
        require(key not in indexed[arm], "duplicate request slot")
        row = record["request"]
        require("request_url" not in row and row["request_path"] == PROMPTS[prompt][0],
                "unsanitized or incorrect route")
        indexed[arm][key] = dict(row, request_url=row["request_path"])
    reports = {}
    for arm in ARMS:
        meta = receipt["reports"][arm]
        require(meta["source_sha256"] == provenance["tested_client_source_sha256"],
                "tested client source drift")
        cells = []
        for prompt in PROMPTS:
            cell = {"prompt": prompt, "uncached_first_use": indexed[arm].get((prompt, 1, None, 0)),
                    "levels": []}
            for n in (1, 16):
                bursts = [{"index": b, "phase": "primed_first_burst" if b == 0 else "repeat",
                           "requests": [indexed[arm].get((prompt, n, b, i)) for i in range(n)]}
                          for b in range(3)]
                require(all(row is not None for burst in bursts for row in burst["requests"]),
                        "missing primed request")
                cell["levels"].append({"streams": n, "bursts": bursts})
            cells.append(cell)
        reports[arm] = {"schema": "pr608_three_way_http_v1", "complete": True,
                        "label": arm, "model": meta["model"], "parameters": meta["parameters"],
                        "health_totals": meta["health_totals"], "source_sha256": meta["source_sha256"],
                        "cells": cells}
    return receipt, reports


def analyse(root: Path, source_root: Path | None = None) -> dict[str, Any]:
    """Check all 624 rows and reproduce the independent metric definitions."""
    provenance = json.loads((root / "provenance.json").read_text())
    require(provenance["schema"] == "pr608_three_way_public_provenance_v1"
            and provenance["sanitized"] is True, "provenance drift")
    require(provenance["stock_binary_sha256"] ==
            "73a0e0434ce5adf3031a22fe2a70cbc9c671150464e4a6ecc4016b8defc12c8d"
            and provenance["native_binary_sha256"] ==
            "469b917ca87959fbf3f38964cd91388da433e1395a5773556560d4982fc7d761", "binary identity drift")
    require(provenance["models"] == [
        {"name": "target", "repo": "Vontra/Qwen3.8-27B-MLX-4bit",
         "revision": "70ae7fac63274ff2eac54152031433374cb80f2f"},
        {"name": "drafter", "repo": "z-lab/Qwen3.8-27B-DFlash2",
         "revision": "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"}], "model identity drift")
    expected_flags = {"TF_PR608_EXACT_PROMPT_CACHE", "TF_PR608_EXACT_CACHE_SEQ_REUSE",
                      "TF_PR608_EXACT_CACHE_LOGITS", "TF_PR608_EXACT_CACHE_DRAFT_REUSE",
                      "TF_PR608_EXACT_CACHE_SKIP_RESET"}
    flags = provenance["flags"]
    require(set(flags) == set(ARMS) and flags["python"] == {}, "arm flag inventory drift")
    for arm in ("stock", "candidate"):
        require(set(flags[arm]) == expected_flags and
                set(flags[arm].values()) == {"1" if arm == "candidate" else "0"},
                "runtime flag values drift")
    qualification = (root / "qualification.json").read_bytes()
    require(digest(qualification) == provenance["qualification_sha256"]
            and json.loads(qualification)["passed"] is True, "qualification attestation drift")
    sources = provenance["source_sha256"]
    require(len(sources) == provenance["source_count"] == 19, "runtime source count drift")
    if source_root is not None:
        for name, expected_hash in sources.items():
            require(name.startswith("zig/src/families/qwen27/cuda/")
                    and ".." not in Path(name).parts, "unsafe source path")
            require(digest((source_root / name).read_bytes()) == expected_hash, "runtime source changed")
    orders: dict[str, Any] = {}
    for label, expected in (("forward", ORDERS[0]), ("reverse", ORDERS[1])):
        receipt, reports = read_order(root, label, expected, provenance)
        rows = {arm: report_rows(report) for arm, report in reports.items()}
        require({report["model"] for report in reports.values()} == {"qwen27-pr608"}, "model alias drift")
        for key in rows["stock"]:
            bodies = {json.dumps(rows[arm][key]["request_body"], sort_keys=True) for arm in ARMS}
            outputs = {(rows[arm][key]["output_sha256"], rows[arm][key]["token_sha"]) for arm in ARMS}
            require(len(bodies) == len(outputs) == 1, f"three-arm parity drift: {key}")
        results = {}
        for prompt in PROMPTS:
            for n in (1, 16):
                key = f"{prompt}_{n}"
                results[key] = {arm: phase_metrics(rows[arm], prompt, n) for arm in ARMS}
        state = receipt["lifecycle_attestation"]
        require(state.get("complete") is True and state.get("restored") is True
                and state.get("all_jobs_passed") is True, "lifecycle attestation incomplete")
        require(all(state["jobs"][arm]["passed"] is True for arm in ARMS), "arm attestation failed")
        orders[label] = {"process_order_expected": list(expected), "requests_per_arm": 104,
                         "status_sha256": receipt["status_sha256"],
                         "provenance_sha256": receipt["provenance_sha256"],
                         "source_report_sha256": {arm: receipt["reports"][arm]["source_report_sha256"] for arm in ARMS},
                         "work_difference_counts": {
                             "stock_candidate": sum(rows["stock"][key]["work"] != rows["candidate"][key]["work"]
                                                    for key in rows["stock"]),
                             "python_candidate": sum(rows["python"][key]["work"] != rows["candidate"][key]["work"]
                                                     for key in rows["stock"]),
                         }, "cells": results}
    balanced = {}
    for cell in orders["forward"]["cells"]:
        balanced[cell] = {}
        for arm in ARMS:
            phases = [orders[o]["cells"][cell][arm] for o in orders]
            values = {}
            for name in ("uncached_first_use", "warm_repeated_mean", "all_three_mean"):
                values[name] = {key: statistics.mean(p[name][key] for p in phases)
                                for key in phases[0][name]}
            values["primed_first_burst"] = {key: statistics.mean(p["bursts"][0][key] for p in phases)
                                             for key in phases[0]["bursts"][0]}
            values["max_three_slowest_ttft_s"] = statistics.mean(p["max_three_slowest_ttft_s"] for p in phases)
            balanced[cell][arm] = values
        for other in ("stock", "python"):
            ratio = {}
            for name, field in (("uncached_first_use", "ttft_s"),
                                ("uncached_first_use", "full_response_s"),
                                ("primed_first_burst", "median_ttft_s"),
                                ("warm_repeated_mean", "median_ttft_s"),
                                ("warm_repeated_mean", "median_per_stream_decode_tps"),
                                ("warm_repeated_mean", "aggregate_decode_tps"),
                                ("warm_repeated_mean", "batch_wall_s"),
                                ("all_three_mean", "median_ttft_s")):
                vals = [orders[o]["cells"][cell]["candidate"][name][field]
                        / orders[o]["cells"][cell][other][name][field] for o in orders]
                ratio[f"{name}.{field}"] = geometric_ratio(vals)
            vals = [orders[o]["cells"][cell]["candidate"]["max_three_slowest_ttft_s"]
                    / orders[o]["cells"][cell][other]["max_three_slowest_ttft_s"] for o in orders]
            ratio["max_three_slowest_ttft_s"] = geometric_ratio(vals)
            balanced[cell][f"candidate_to_{other}_geometric_ratio"] = ratio
    return {"schema": "pr608_three_way_independent_analysis_v1", "complete": True,
            "orders": orders, "balanced": balanced,
            "limits": ["Cold here means no cached prompt tokens after server readiness, not model startup time.",
                       "Code is first request in each process; chat first-use follows code traffic.",
                       "Bursts share a process and their requests are correlated; two orders are not confidence intervals.",
                       "Different speculative work is retained and does not invalidate literal output parity, but limits causal attribution."]}



def main() -> None:
    """Validate offline; optionally check the nineteen runtime source bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyse(args.root, args.source_root)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
