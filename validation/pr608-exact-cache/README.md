## What this changes

This contribution builds on Philip's [Qwen27 CUDA engine, PR608](https://github.com/ashhart/TensorFold/pull/608) and the earlier [first-token handoff fix, 7acdc544](https://github.com/jregan-beasley-smc/TensorFold/commit/7acdc54473b5c9f54582b5d3c465e89b92daf886). Runtime commit `474bd89b68dee96149898ad303626d5d9dce0291` contains 19 pinned files: four existing files modified and 15 helpers/tests added. The tested parent `311de580` differs from publication parent `7acdc544` only by earlier validation material. The runtime bytes are unchanged.

Five independent diagnostic flags accept `0` or `1` and default to `0`. The latter four require the exact-prompt cache:

| Flag | Mechanism |
| --- | --- |
| `TF_PR608_EXACT_PROMPT_CACHE` | Snapshot the full identical prompt's target, head and packed DFlash state; restore into independently owned normal allocations. |
| `TF_PR608_EXACT_CACHE_SEQ_REUSE` | Reuse settled target Seq storage after a proven full-prompt restore, at exactly the requested capacity. |
| `TF_PR608_EXACT_CACHE_LOGITS` | Cache the unchanged first BF16 logits before sampling; recheck the active key or run the ordinary head. |
| `TF_PR608_EXACT_CACHE_DRAFT_REUSE` | Reuse fixed Context storage across eligible small prompts, with fresh wrapper/CopyIndex state. |
| `TF_PR608_EXACT_CACHE_SKIP_RESET` | Defer initialization fills in a private prefill transaction; a successful full-state restore replaces them, a miss performs the original reset before target work. |

The cache has one active and one pending entry under a combined 256 MiB cap, including the 496,640-byte logits row. Target retention is capped at 3 GiB; Context retention at 1.5 GiB. Active plus retained owners stay within configured streams, at most 16. Pool/reset eligibility also requires normal capacity at most 512. Capacity remains `min(context, prompt_length + max_tokens + 16)`. Public initialized factories, sampling, target/draft arithmetic and planner behavior remain unchanged. Clean completion, successful settle/cache release and fences gate ownership transfer. Cancellation and errors invalidate pending state; uncertain CUDA use cannot enter a reusable pool.

## Receipt

`provenance.json` pins all runtime files, model revisions, Python v0.6.6 (`cb2ebf0540f42604e2759b2ddef497861e928248`), native binary and build attestations. The retained environment is DGX Spark GB10 `sm_121`, Linux aarch64, Zig 0.17.0 and NVIDIA's PyTorch 26.07 container (image hash retained). Driver/library versions were not independently retained. Both native arms used the same 49-kernel AOT/native assets. CPU checks passed 334 host tests, six native build steps and five qualifier build steps.

`qualification.json` is a sanitized actual-CUDA gate attestation, not a distributed device checker. It records poisoned fresh/pooled state restoration, immutable snapshots, 16 nonaliasing owners, raw first logits, direct continuation and hint-to-miss fallback. Three resolved restores avoided 288 initialization fills / 461,832,192 bytes; public factories still reset. The 968-line device checker, model weights, compiled binaries and operational logs are not included.

Each order has 312 retained requests: three fresh server arms, each with two 16-output prompt warmups and 102 measured 256-output requests. Order 1 is reset-off / Python / candidate; order 2 is candidate / Python / reset-off. Both native arms enable the preceding four flags; only the final reset flag differs. Literal bodies and output fingerprints match for 104 native pairs and 104 native/Python pairs per order, including two untimed warmups. Work counters differ and remain visible.

These are client TTFT medians and slowest values in milliseconds, and complete burst wall time in seconds. Each cell averages the two orders. “First” means the first measured burst after warmup; “warm” averages bursts 1 and 2, not process cold. `verify.py` reconstructs every repetition and table from the sanitized request evidence.

| Prompt / streams / phase | Python median / slowest TTFT | Candidate median / slowest TTFT | Python / candidate wall |
| --- | ---: | ---: | ---: |
| Code / 1 / first | 111.0 / 111.0 | 9.6 / 9.6 | 3.869 / 3.683 |
| Code / 1 / warm | 113.7 / 113.7 | 3.1 / 3.1 | 3.865 / 3.678 |
| Chat / 1 / first | 113.9 / 113.9 | 9.2 / 9.2 | 5.188 / 5.169 |
| Chat / 1 / warm | 114.8 / 114.8 | 3.6 / 3.6 | 5.192 / 5.165 |
| Code / 16 / first | 302.6 / 304.4 | 245.0 / 246.5 | 10.561 / 10.200 |
| Code / 16 / warm | 220.3 / 222.3 | 128.6 / 130.2 | 10.468 / 10.182 |
| Chat / 16 / first | 215.4 / 217.9 | 201.6 / 203.8 | 13.635 / 13.823 |
| Chat / 16 / warm | 237.1 / 239.4 | 130.7 / 132.0 | 13.703 / 13.797 |

Recheck the bundled evidence offline with Python 3.11+ and no third-party packages:

```sh
python3 validation/pr608-exact-cache/verify.py --source-root .
python3 -B -m unittest discover -s validation/pr608-exact-cache -p 'test_verify.py'
zig build test-cuda-host -j2 --summary all
```

For a new GPU run, independently prepare the pinned models and upstream Qwen27 AOT assets. Start each arm in a fresh server process. Supply your own model directories, binary, address and port; this client does not start services or download models. The native candidate configuration is:

```sh
export TF_PR608_EXACT_PROMPT_CACHE=1
export TF_PR608_EXACT_CACHE_SEQ_REUSE=1
export TF_PR608_EXACT_CACHE_LOGITS=1
export TF_PR608_EXACT_CACHE_DRAFT_REUSE=1
export TF_PR608_EXACT_CACHE_SKIP_RESET=1
export TENSORFOLD_CUDA_KERNELS="$AOT_DIR"
export TENSORFOLD_MEMORY_RESERVE_GIB=8
"$NATIVE_BIN" serve "$TARGET_DIR" --backend cuda --drafter "$DRAFTER_DIR" \
  --host "$BIND_HOST" --port "$PORT" --name qwen27-pr608 \
  --parallel 16 --context 32768 --prompt-cache-gib 0 \
  --temperature 0 --no-thinking --no-update-check
python3 validation/pr608-exact-cache/run_http.py --base "$BASE_URL" \
  --model qwen27-pr608 --label on --output candidate.json --max-seconds 1200
```

Use reset flag `0`, the same preceding four flags, and client label `on` for reset-off. That label denotes full-N cache expectations, not reset policy. For the pinned Python reference use its upstream CUDA serving CLI with the same models, geometry and alias:

```sh
PYTHONPATH="$PYTHON_SOURCE/src" python3 -m tensorfold serve "$TARGET_DIR" \
  --backend cuda --drafter "$DRAFTER_DIR" --drafter-bits 4 \
  --host "$BIND_HOST" --port "$PORT" --name qwen27-pr608 \
  --parallel 16 --context 32768 --prompt-cache-gib 0 \
  --temperature 0 --no-thinking --no-update-check \
  --kv-dtype bf16 --no-prefill-fp8
python3 validation/pr608-exact-cache/run_http.py --base "$BASE_URL" \
  --model qwen27-pr608 --label python --output python.json --max-seconds 1200
```

The Python source and dependencies must already be installed separately; the bundled client imports no engine code. Python's actual prefix cache retains N−1 tokens; native retains N. The client requires observed counts 13/30 versus 14/31 after uncached warmups. Do not assume matching cache internals from matching CLI flags. Run both process orders above. Live client reports contain the endpoint you supply; the bundled JSONL removes endpoints and preserves routes, bodies, usage/work, SSE clocks and hashes.

## Limits

This is an exact full-prompt experiment for fresh short text, not generic prefix matching, resume/vision support or Flash Next qualification. HTTP qualification is greedy, two short prompts, 1/16 streams and fixed 256 outputs. Long-context, sampled HTTP and diverse prompts remain outside the measured scope. Direct device continuation does not substitute for every speculative verify/keep path.

TTFT is send to first nonempty SSE content; full wall ends at final `[DONE]`. There is no capture-time subtraction. Responses in a burst are correlated; two process orders are point estimates, not confidence intervals. The five-feature/Python comparison does not isolate each feature. Full-wall changes are much smaller than TTFT changes; chat wall is slightly slower here. No throughput or universal speedup claim is made.

`output_sha256` combines content/reasoning digests independently of chunk boundaries; `token_sha` is the server's retained token fingerprint (12 hex digits here), not a distributed full token-ID sequence. Sanitized receipts are derived evidence, not raw operational logs. Hash attestations cannot independently reconstruct absent binaries or the omitted device checker.

Philip's engine and Python reference provide the baseline implementation. Jonathan Regan-Beasley contributed this validation and cache work in a personal capacity.
