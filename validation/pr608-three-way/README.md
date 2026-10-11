## What this changes

This is a fresh matched comparison of pinned Python v0.6.6, unmodified PR608 (`99bcc5b2`) and the complete exact-prompt-cache candidate (`474bd89b`). It supersedes the historical cross-run comparison for performance claims. Philip's Qwen27 engine and Python reference are the baseline; the candidate includes the earlier first-token handoff fix (`7acdc544`) and nineteen pinned cache/reuse runtime files.

Every request uses the same literal greedy code/chat body, seed 1234, ignore_eos and 256 outputs. Each backend starts fresh: code first-use n1, then three n1 and three n16 bursts; chat first-use n1 follows that code traffic and has the same subsequent panel. Cold means prompt cached_tokens=0 after server readiness, not model-load time. There is no cold n16 measurement. After first use, stock remains uncached, Python reuses N−1 (13/30 tokens), candidate restores full N (14/31).

The candidate's cache is opt-in and defaults off. It hits only an exact whole identical short prompt, not a growing conversation or arbitrary matching prefix. Changing/appending conversation turns does not automatically reuse this snapshot. The other reuse/reset mechanisms have their own short-capacity and ownership gates; this experiment is not general prefix-cache support.

## Receipt

All 624 requests preserve literal bodies/body hashes, output fingerprints, usage/work and sent/first/last/DONE clocks in sanitized JSONL. They include 312 requests per order, 104 per backend. Orders are Python→stock→candidate and candidate→stock→Python. The offline checker requires all three outputs to match, reports differing speculative work, and reproduces the independent summary. Its source revision, binaries, model revisions, source hashes and prior device gates are in provenance.json and qualification.json. Raw physical identities and operational restore logs are excluded.

Tables are arithmetic means across the two orders. The cold first-use rows are single requests. Warm repeated rows average bursts 1 and 2 within each order; full per-repetition values, primed first bursts and maximum-three-burst TTFT remain in summary.json.

Cold n1: **TTFT ms / decode tok/s / full response s**:

| Prompt | Stock PR608 | Python | Candidate |
| --- | ---: | ---: | ---: |
| Code | 145.8 / 69.30 / 3.826 | 205.9 / 66.80 / 4.024 | 148.8 / 69.44 / 3.821 |
| Chat | 131.4 / 49.45 / 5.289 | 123.5 / 50.25 / 5.199 | 162.0 / 49.38 / 5.327 |

Warm repeated: **median / slowest TTFT ms**:

| Prompt / streams | Stock PR608 | Python | Candidate |
| --- | ---: | ---: | ---: |
| Code / 1 | 128.3 / 128.3 | 111.3 / 111.3 | 3.4 / 3.4 |
| Code / 16 | 681.8 / 683.2 | 219.5 / 221.0 | 126.2 / 127.5 |
| Chat / 1 | 135.5 / 135.5 | 114.2 / 114.2 | 3.8 / 3.8 |
| Chat / 16 | 833.5 / 835.4 | 233.0 / 235.0 | 126.8 / 128.5 |

Warm repeated: **median per-stream decode tok/s / aggregate decode tok/s / full burst wall s**:

| Prompt / streams | Stock PR608 | Python | Candidate |
| --- | ---: | ---: | ---: |
| Code / 1 | 68.80 / 68.80 / 3.836 | 68.11 / 68.11 / 3.856 | 69.45 / 69.45 / 3.675 |
| Code / 16 | 25.13 / 381.38 / 10.833 | 25.08 / 401.15 / 10.408 | 25.51 / 403.21 / 10.124 |
| Chat / 1 | 49.49 / 49.49 / 5.288 | 50.37 / 50.37 / 5.178 | 49.49 / 49.49 / 5.157 |
| Chat / 16 | 18.78 / 285.81 / 14.418 | 19.06 / 304.83 / 13.634 | 18.81 / 295.42 / 13.819 |

Reconstruct offline with Python 3.11+ (stdlib only):

```sh
python3 validation/pr608-three-way/verify.py --source-root .
python3 -B -m unittest discover -s validation/pr608-three-way -p test_verify.py
```

The standalone client requires your already-running isolated server and downloads/loads no model:

```sh
python3 validation/pr608-three-way/run_http.py --base "$BASE_URL" \
  --model qwen27-pr608 --label candidate --output candidate.json --max-seconds 1200
```

Repeat with stock and python labels on separate fresh processes, in both orders above. Geometry is parallel 16 / context 32768, BF16 KV/prefill, thinking off, same target/drafter revisions and 49-kernel AOT/native assets. Use the candidate configuration documented in ../pr608-exact-cache/README.md with all five flags 1; stock is the exact original binary with those flags absent/0. Python uses --drafter-bits 4 --kv-dtype bf16 --no-prefill-fp8 and its actual prefix-cache behavior. Client/cache gates distinguish backend states; matching CLI flags do not imply identical cache internals.

## Limits

TTFT is send→first nonempty SSE content; decode rate is 255/(last−first); full wall is last DONE−earliest send. Aggregate decode uses the sum of actual output-token intervals over the shared first-to-last span. It is not the character-estimated steady_tps field. No capture-time subtraction, equality-of-work assumption or universal no-regression claim is made. Cold candidate TTFT can be slower than stock: all three methods' cold TTFT and complete response time remain in the first table. Decode rate and wall results are retained even when an arm loses. This is a trade-off for review, not an automatic recommendation to enable the flags for every workload.

Requests in a burst are correlated. Two fresh process orders are point estimates, not confidence intervals. Only two short prompts, greedy output and n1/n16 were measured; no general-prefix, long-context, sampled HTTP, other-model or FlashNext claim follows. Hash attestations do not reconstruct absent binaries/model weights or the omitted device qualifier. The content/reasoning composite SHA and server token fingerprint are retained; full token-ID sequences are not distributed. Driver/library versions were not independently retained beyond the pinned image/environment attestations.

Jonathan Regan-Beasley contributed this work in a personal capacity.
