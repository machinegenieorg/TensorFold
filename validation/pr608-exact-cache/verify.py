"""Validate sanitized HTTP evidence and reconstruct per-order TTFT/full-wall tables."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

import run_http

ARMS = ("reset_off", "python", "candidate")
FLAGS = {
    "TF_PR608_EXACT_PROMPT_CACHE", "TF_PR608_EXACT_CACHE_SEQ_REUSE",
    "TF_PR608_EXACT_CACHE_LOGITS", "TF_PR608_EXACT_CACHE_DRAFT_REUSE",
    "TF_PR608_EXACT_CACHE_SKIP_RESET",
}


def require(ok: bool, message: str) -> None:
    """Reject a missing or contradictory evidence gate."""
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    """Hash retained bytes without changing them."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_order(path: Path) -> dict[str, Any]:
    """Validate one derived receipt and all 312 retained requests."""
    evidence = json.loads(path.read_text())
    require(evidence.get("schema") == "pr608_exact_cache_public_order_v1"
            and evidence.get("complete") is True and evidence.get("sanitized") is True,
            "incomplete or wrong sanitized receipt")
    order = evidence["order_name"]
    expected = list(ARMS) if order == "forward" else ["candidate", "python", "reset_off"]
    require(order in ("forward", "reverse") and evidence["process_order"] == expected,
            "process-order drift")
    require(set(evidence["reports"]) == set(ARMS), "missing arm")
    info = evidence["requests"]
    require(info["file"] == f"requests-{order}.jsonl", "unexpected evidence path")
    data_path = path.parent / info["file"]
    require(sha(data_path) == info["sha256"], "request evidence hash mismatch")
    records = [json.loads(line) for line in data_path.read_text().splitlines()]
    require(len(records) == info["count"] == 312, "request count drift")
    slots: dict[tuple[Any, ...], dict[str, Any]] = {}
    ids: dict[str, set[str]] = {arm: set() for arm in ARMS}
    for record in records:
        require(record.get("schema") == "pr608_exact_cache_public_request_v1",
                "wrong request schema")
        arm, prompt = record["arm"], record["prompt"]
        require(arm in ARMS and prompt in ("code", "chat"), "unknown request identity")
        n, burst, stream = record["streams"], record["burst"], record["stream"]
        phase = record["phase"]
        warmup = burst is None
        require(type(n) is int and n in (1, 16) and type(stream) is int
                and 0 <= stream < n, "bad stream geometry")
        require((warmup and n == 1 and stream == 0 and phase == "warmup") or
                (type(burst) is int and burst in (0, 1, 2)
                 and phase == ("first_measured" if burst == 0 else "repeat")),
                "burst/warmup identity drift")
        key = (arm, prompt, n, burst, stream)
        require(key not in slots, "duplicate request slot")
        row = record["request"]
        require("request_url" not in row, "unsanitized endpoint")
        report = evidence["reports"][arm]
        require(report["parameters"] == run_http.PARAMETERS, "protocol drift")
        require(report["reported_cache_label"] == ("python" if arm == "python" else "on"),
                "cache-label drift")
        item = run_http.prompts()[prompt]
        route, _ = run_http.expected_body(report["model"], item, 16 if warmup else 256)
        require(row["request_path"] == route, "route drift")
        # The copied validator sees a synthetic string base; it never opens a connection.
        checked = dict(row, request_url="receipt-base" + route)
        prompt_n = run_http.EXPECTED_PROMPTS[prompt][2]
        cached = 0 if warmup else prompt_n - 1 if arm == "python" else prompt_n
        run_http.validate_request(checked, item=item, model=report["model"],
                                  base="receipt-base", tokens=16 if warmup else 256,
                                  cached=cached)
        require(row["response_id"] not in ids[arm], "duplicate response identity")
        ids[arm].add(row["response_id"])
        slots[key] = row
    for arm in ARMS:
        require(evidence["reports"][arm]["health_totals"] ==
                {"requests": 104, "prompt_tokens": 2340, "output_tokens": 26144},
                "health-count drift")
        require(len(ids[arm]) == 104, "arm identity count drift")
        for prompt in ("code", "chat"):
            require((arm, prompt, 1, None, 0) in slots, "missing warmup")
            for n in (1, 16):
                for burst in range(3):
                    require(all((arm, prompt, n, burst, s) in slots for s in range(n)),
                            "missing measured request")
    return {"receipt": evidence, "slots": slots}


def paired(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Require literal prompt/settings and both output fingerprints; retain work differences."""
    require(left["request_body"] == right["request_body"]
            and left["request_body_sha256"] == right["request_body_sha256"],
            "paired literal request mismatch")
    require(left["output_sha256"] == right["output_sha256"]
            and left["token_sha"] == right["token_sha"], "paired output mismatch")
    return left["work"] != right["work"]


def metrics(slots: dict[tuple[Any, ...], dict[str, Any]], arm: str,
            prompt: str, n: int) -> dict[str, Any]:
    """Keep each repetition and arithmetic means of the two warmed burst metrics."""
    repetitions = []
    for burst in range(3):
        rows = [slots[(arm, prompt, n, burst, s)] for s in range(n)]
        repetitions.append({
            "burst": burst, "phase": "first_measured" if burst == 0 else "repeat",
            "median_ttft_s": statistics.median(row["ttft_s"] for row in rows),
            "slowest_ttft_s": max(row["ttft_s"] for row in rows),
            "batch_wall_s": max(row["done"] for row in rows) - min(row["sent"] for row in rows),
        })
    return {
        "repetitions": repetitions,
        "first_median_ttft_s": repetitions[0]["median_ttft_s"],
        "first_slowest_ttft_s": repetitions[0]["slowest_ttft_s"],
        "first_batch_wall_s": repetitions[0]["batch_wall_s"],
        "max_three_slowest_ttft_s": max(row["slowest_ttft_s"] for row in repetitions),
        "warm_median_ttft_mean_s": statistics.mean(row["median_ttft_s"] for row in repetitions[1:]),
        "warm_slowest_ttft_mean_s": statistics.mean(row["slowest_ttft_s"] for row in repetitions[1:]),
        "warm_batch_wall_mean_s": statistics.mean(row["batch_wall_s"] for row in repetitions[1:]),
    }


def verify(root: Path, source_root: Path | None = None) -> dict[str, Any]:
    """Check provenance, complete work, output pairing and both process orders."""
    provenance = json.loads((root / "provenance.json").read_text())
    require(provenance["schema"] == "pr608_exact_cache_public_provenance_v1"
            and provenance["sanitized"] is True, "bad provenance schema")
    require(provenance["models"] == [
        {"name": "target", "repo": "Vontra/Qwen3.8-27B-MLX-4bit",
         "revision": "70ae7fac63274ff2eac54152031433374cb80f2f"},
        {"name": "drafter", "repo": "z-lab/Qwen3.8-27B-DFlash2",
         "revision": "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"},
    ], "model identity drift")
    sources = provenance["source_sha256"]
    require(len(sources) == provenance["source_count"] == 19
            and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in sources.values()),
            "source inventory drift")
    if source_root is not None:
        for relative, digest in sources.items():
            require(relative.startswith("zig/src/families/qwen27/cuda/")
                    and ".." not in Path(relative).parts, "unsafe source path")
            require(sha(source_root / relative) == digest, f"source changed: {relative}")
    flags = provenance["flag_values"]
    for arm in ("reset_off", "candidate"):
        require(set(flags[arm]) == FLAGS and all(value == "1" for name, value in flags[arm].items()
                if name != "TF_PR608_EXACT_CACHE_SKIP_RESET"), "preceding flags drift")
        require(flags[arm]["TF_PR608_EXACT_CACHE_SKIP_RESET"] ==
                ("1" if arm == "candidate" else "0"), "reset arm drift")
    qualification = json.loads((root / "qualification.json").read_text())
    require(qualification["passed"] is True and qualification["sanitized"] is True,
            "missing device gate attestation")
    reset = qualification["reset_skip"]
    require(all(reset[k] is True for k in (
        "fresh_poisoned_full_state_raw_and_continuation_equal",
        "pooled_poisoned_full_state_raw_and_continuation_equal",
        "positive_hint_actual_miss_reset_before_target", "public_open_reset_domains_zero")),
        "reset device gate failed")
    require(reset["restored_fresh"] == 2 and reset["restored_pool"] == 1
            and reset["fallback_resets"] == 1 and reset["failed_deferred_transactions"] == 0
            and reset["avoided_fill_calls"] == 288 and reset["avoided_fill_bytes"] == 461832192,
            "structural counter drift")
    output: dict[str, Any] = {"schema": "pr608_exact_cache_public_summary_v1", "complete": True,
                              "requests": 624, "orders": {}, "balanced": {}}
    for order in ("forward", "reverse"):
        data = read_order(root / f"evidence-{order}.json")
        slots = data["slots"]
        work_diff = {"reset_off_candidate": 0, "candidate_python_measured": 0,
                     "candidate_python_warmup": 0}
        for key, candidate in slots.items():
            arm, prompt, n, burst, stream = key
            if arm != "candidate":
                continue
            work_diff["reset_off_candidate"] += paired(
                slots[("reset_off", prompt, n, burst, stream)], candidate)
            pair_type = "candidate_python_warmup" if burst is None else "candidate_python_measured"
            work_diff[pair_type] += paired(
                candidate, slots[("python", prompt, n, burst, stream)])
        output["orders"][order] = {"work_differences": work_diff,
            "paired_reset_requests": 104, "paired_python_measured_requests": 102,
            "paired_python_warmup_requests": 2, "paired_python_all_requests": 104,
            "cells": {f"{prompt}_{n}": {arm: metrics(slots, arm, prompt, n) for arm in ARMS}
                      for prompt in ("code", "chat") for n in (1, 16)}}
    fields = ("first_median_ttft_s", "first_slowest_ttft_s", "first_batch_wall_s",
              "max_three_slowest_ttft_s", "warm_median_ttft_mean_s",
              "warm_slowest_ttft_mean_s", "warm_batch_wall_mean_s")
    for prompt in ("code", "chat"):
        for n in (1, 16):
            key = f"{prompt}_{n}"
            cells = [output["orders"][order]["cells"][key] for order in ("forward", "reverse")]
            output["balanced"][key] = {
                "arithmetic_means": {arm: {field: statistics.mean(cell[arm][field] for cell in cells)
                                            for field in fields} for arm in ARMS},
                "geometric_candidate_to_python": {
                    field: math.sqrt(math.prod(cell["candidate"][field] / cell["python"][field]
                                               for cell in cells)) for field in fields},
            }
    return output


def main() -> None:
    """Run offline; optional source-root additionally rehashes all nineteen runtime files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.root, args.source_root)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
