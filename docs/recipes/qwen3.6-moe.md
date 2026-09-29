# Qwen3.6-35B-A3B

The `qwen3_5_moe` family serves Qwen3.6-35B-A3B on one NVIDIA GPU. Its layers are the 27B's (Gated DeltaNet
and gated full attention, every fourth layer attention) with routed experts in place of the dense MLP, and it
drafts with the checkpoint's own MTP layer. It reads the MLX 4-bit conversion and NVIDIA's NVFP4 checkpoint.

## Checkpoint

```bash
tensorfold pull Vontra/Qwen3.6-35B-A3B-MLX-4bit-MTP
tensorfold serve Vontra/Qwen3.6-35B-A3B-MLX-4bit-MTP --name bench
```

Tested revision: `81169a9bc511a27c1b4eedb77a2cd98ced431847` (20.9 GB). Its weights are
`mlx-community/Qwen3.6-35B-A3B-4bit` (MLX affine 4-bit in groups of 64, routers and the shared-expert gate at
8 bits), converted from `Qwen/Qwen3.6-35B-A3B`; that conversion drops the MTP layer, which ships beside it as
`mtp-4bit.safetensors` (converted by the same rules: experts split into gate and up projections, norms
shifted by one, projections 4-bit, gates 8-bit). The mlx-community folder serves the same way once that file
is placed in it. After the weights, one Spark keeps about 75 GB for caches: attention holds 20 KB a token and
DeltaNet 63 MB a stream.

`--no-drafts` or request field `"draft": false` selects serial decoding, the reference drafted output equals.

### NVIDIA's NVFP4 checkpoint

```bash
tensorfold pull nvidia/Qwen3.6-35B-A3B-NVFP4
tensorfold serve nvidia/Qwen3.6-35B-A3B-NVFP4 --name bench
```

Tested revision: `1355db6a052410cfd62085d94b58866fd0f2c3c5` (22 GB, ModelOpt 0.44, `MIXED_PRECISION`), the weights
vLLM serves. The engine reads them as they ship, on `tensorfold/cuda/nvfp4`; each layer's format is the
checkpoint's:

| Tensors | Stored as | Read by |
| --- | --- | --- |
| Routed experts, shared expert | `W4A16_NVFP4`: E2M1 nibbles, an e4m3 scale per 16 values, an fp32 scale per tensor | one table a layer on the experts plan, the shared expert its last expert: `nvfp4/experts_split` (K in slices; `nvfp4/experts` reads the same bytes) |
| `lm_head` and the draft head (its draft vocabulary's rows) | `W4A16_NVFP4` | `nvfp4.linear.Fp4Linear` |
| DeltaNet `in_proj_qkv`, `in_proj_z`, `out_proj`; attention `q/k/v/o_proj` | `FP8`: e4m3 codes, an fp32 scale per tensor | `nvfp4.linear.Fp8Linear`: decode's bf16 rows on the exact weights (W8A16), prompt chunks' rows FP8 with a scale a row (the FP8 GEMM) |
| Embedding, routers, shared-expert gates, `in_proj_a/b`, conv, norms, the MTP layer | bf16 | as stored (`bf16.matmul`); `in_proj_a/b` take prompt chunks' FP8 rows exactly (`rows8`); the MTP layer's experts only draft, and are quantized to NVFP4 at load |

Decode's rows stay bf16. A prompt chunk's rows reach the FP8 projections and the DeltaNet gates as FP8 with a
scale a row, as the 27B's NVFP4 route and the MLX route's prompt path have them; the experts read bf16 rows in
both. `TF_NVFP4_PROMPT_ROWS=bf16` (read at load) keeps prompt rows bf16 too, the FP8 projections then loaded as
`fp8.py`'s row-major W8A16 matmul: closer to the checkpoint's arithmetic, slower. Nothing reads the checkpoint's
`input_scale` tensors (vLLM's static FP8 activation scales) or its `kv_cache_quant_algo: FP8` (vLLM's cache
format), so the KV cache stays bf16. The RMSNorm weights are stored zero-centred (the model applies `1 + w`, which
MLX conversions store instead) and become that scale in fp32 at load; DeltaNet's gated norm is stored as applied.
The vision tower is not read. Exactness is the MLX route's: drafted replies equal `"draft": false`, `--parallel N`
equals solo, a resumed prompt equals a fresh prefill, and startup admits the window before loading.

Fidelity on eight public passages of 1,024 tokens (`pydoc_data` topics and standard-library source): next-token
NLL, and the share of positions whose top token equals the fp32 forward of the bf16 release
(`Qwen/Qwen3.6-35B-A3B` at `995ad96`) or of this checkpoint's exactly dequantized weights. Verify windows give the
tokens serial decoding gives; the prompt path fills the context.

| Scores | NLL | Top 1 = bf16 release | Top 1 = NVFP4, fp32 |
| --- | ---: | ---: | ---: |
| bf16 release, fp32 | 0.4464 | 100% | 93.62% |
| NVFP4 checkpoint, fp32 | 0.5041 | 93.62% | 100% |
| NVFP4 route, verify windows | 0.4970 | 93.44% | 98.25% |
| NVFP4 route, prompt path (FP8 rows) | 0.5137 | 93.27% | 96.93% |
| `TF_NVFP4_PROMPT_ROWS=bf16`, verify windows | 0.5002 | 93.51% | 98.08% |
| `TF_NVFP4_PROMPT_ROWS=bf16`, prompt path | 0.4993 | 93.56% | 98.36% |
| MLX 4-bit route, verify windows | 0.5183 | 90.69% | 89.67% |
| MLX 4-bit route, prompt path (FP8 rows) | 0.5444 | 89.63% | 88.67% |

```bash
python -m tensorfold.families.qwen3_5_moe.cuda.reference reference <bf16 or NVFP4 folder> out.pt <tokenizer folder>
python -m tensorfold.families.qwen3_5_moe.cuda.reference route <any checkpoint folder> out.pt <tokenizer folder>
python -m tensorfold.families.qwen3_5_moe.cuda.reference compare reference.pt route.pt ...
```

On one RTX PRO 6000 Blackwell Max-Q (NGC 26.07, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), with 8 client
threads over HTTP and the workloads of [Concurrent requests](#concurrent-requests):

| | `--parallel 1` label | chat | `--parallel 8` label | chat | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVFP4 route | 910 tok/s | 438 tok/s | 1,642 tok/s | 1,083 tok/s | 24.0-25.1 GB |
| `TF_NVFP4_PROMPT_ROWS=bf16` | 941 | 461 | 1,637 | 1,131 | 24.1-24.4 GB |
| MLX 4-bit route | 907 | 428 | 1,529 | 1,099 | 22.7-22.9 GB |

Every NVFP4 reply's token SHA-256 is the same at `--parallel 1`, at `--parallel 8` and with `"draft": false`
(80 of 80, in either prompt row form), and without expandable segments; with bf16 prompt rows they are the
previous build's (bf16 prompt rows only) too. Drafts keep 9.8 tokens a round on the labels and 3.0 on the chats
(MLX: 9.3 and 2.9).

Prompt prefill (the engine's, random tokens, fresh state each; the MTP head absorbs every row but the last), on
an RTX PRO 6000 Max-Q and on a GB10 (office hardware, measured while its other GPU work was paused):

| Prompt tokens | 3,072 | 6,144 | 12,288 | 24,576 |
| --- | ---: | ---: | ---: | ---: |
| NVFP4 route, Max-Q | 23,200 tok/s | 20,000 tok/s | 18,800 tok/s | 16,100 tok/s |
| `TF_NVFP4_PROMPT_ROWS=bf16`, Max-Q | 19,800 | 17,500 | 16,800 | 14,600 |
| MLX 4-bit route, Max-Q | 22,400 | 19,700 | 18,500 | 15,800 |
| NVFP4 route, GB10 | 8,735 | 8,357 | 7,826 | 6,589 |
| `TF_NVFP4_PROMPT_ROWS=bf16`, GB10 | 7,665 | 7,328 | 6,684 | 5,723 |
| MLX 4-bit route, GB10 | 8,267 | 7,950 | 7,453 | 6,300 |

Time to first token through the server on that GB10 (`--parallel 16 --context 32768`, one client, unique chat
prompts, 16-token replies), against vLLM serving the same checkpoint on a second Spark at the same time:

| Prompt tokens | 2,922 | 5,804 | 11,568 | 23,482 |
| --- | ---: | ---: | ---: | ---: |
| NVFP4 route | 0.363 s (8,052 tok/s) | 0.727 s (7,985) | 1.511 s (7,657) | 3.628 s (6,472) |
| MLX 4-bit route | 0.359 s (8,144) | 0.746 s (7,778) | 1.889 s (6,124) | 3.747 s (6,267) |
| vLLM, NVFP4 | 0.524 s (5,618) | 0.924 s (6,330) | 1.917 s (6,082) | 4.250 s (5,567) |

A 4,096-row chunk on the GB10, by kernel (ms, the MLX route's in brackets): experts gate/up 110 (121) and down 77
(74), the FP8 projections 73 (88), DeltaNet 103 (102), attention 39 (40), routing and the slots' sum 36 (38), norms
20 (21); the MTP head's absorption 3 (5). With bf16 prompt rows the projections take 144. The GB10 is where the
prompt path's forms were chosen, while its GPU was shared (a serving engine kept it 85-96% busy): FP8 rows took
the projections from 306 ms to 160, the staged expert kernel the experts from 465 to 382, and keys-only
absorption through the wide form the head from 77 to 5; prompts of 3,072 tokens went from 3,251 tok/s to 4,337.
On the Max-Q, whose 128 MB L2 holds what the GB10's 24 MB re-reads, the same changes gave 17%.

On these experts (257 of 512 x 2,048) a layer's gate/up and down take 12 and 6 us for a row on `experts_split`
(54 and 11 on `nvfp4.experts`), 32 and 17 for four (55 and 17), and 1.13 and 0.73 ms for a 4,096-row prompt chunk
on the Max-Q (2.06 and 1.19); from 16 rows up the decode forms are within 5%.

## CUDA execution

Verify windows run the 27B's shared kernels (4-bit matmul, DeltaNet tree and replay, tree attention) with
routed experts from `tensorfold/cuda/experts.py`: the router's top 8 of 256 by fp32 logit (ties to the lower
id), weights renormalized over the eight, the shared expert as expert 256 with a sigmoid gate, and the slots
summed in slot order. Each (row, expert) pair gets the same bits in any window, so a drafted row equals the
serial step.

Each round first verifies a copied continuation when the context repeats eight or more tokens, and otherwise
a chain of up to three MTP drafts: the head reads the target's final normed state and the next token's
embedding, drafts with the target's keyed sampling rule over a draft vocabulary, and a chain stops after a
draft it gives under 30%. Decoding runs in buffers the engine keeps between requests, so verify chains and
head steps replay CUDA graphs captured once per width and context bucket.
Prompts prefill in chunks; the head absorbs every prompt row but the last, as the keys and values later rows
attend to (their queries, attention, experts and outputs are never computed). States are kept at the second
message's start and at prompt ends, so a prompt sharing a system block resumes there with a fresh prefill's
bits.

### Concurrent requests

```bash
tensorfold serve Vontra/Qwen3.6-35B-A3B-MLX-4bit-MTP --parallel 8 --name bench
```

`--parallel N` decodes up to N requests in shared rounds, and every reply equals the same request served alone
and its `"draft": false` run. A round verifies every stream's MTP chain or copied continuation in one forward;
the head absorbs every stream's kept rows in one call, then the chains advance a step at a time for all streams
by the one-stream rule, so a stream drafts what it drafts alone. A new prompt prefills 1,024 tokens a round
while the others decode, states are kept at message starts and prompt ends, and each stream's caches are sized
once, at admission. Startup admits N full prompt/reply windows, three kept prompt ends and the graph buffers
before loading; an explicit `--context` that does not fit is refused with the window that does. Rounds over
several streams run eagerly, as the 27B's do; a stream decoding alone replays the one-stream CUDA graphs, so a
lone request runs as fast as without `--parallel`.

On one RTX PRO 6000 Blackwell Max-Q (NGC 26.07, torch 2.13, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
checkpoint revision 81169a9), with 8 client threads over HTTP:

| | `--parallel 1` | `--parallel 4` | `--parallel 8` |
| --- | ---: | ---: | ---: |
| Label JSON: 32 requests, public-domain passages, up to 1,410 tokens, default sampling | 904 tok/s | 1,255 tok/s | 1,476 tok/s |
| p50 / p95 latency | 8.6 / 9.1 s | 5.9 / 8.1 s | 5.1 / 7.7 s |
| Chat: 48 requests, up to 256 tokens, half greedy, half seeded | 427 tok/s | 859 tok/s | 1,094 tok/s |
| p50 / p95 latency | 3.3 / 3.9 s | 1.6 / 2.5 s | 1.2 / 2.0 s |
| Peak memory (nvidia-smi) | 22.3 GiB | 22.1 GiB | 22.9 GiB |

Every reply's token SHA-256 is the same at each N and with `"draft": false`. The label replies repeat their JSON,
so copied continuations keep 9.3 tokens a round per stream; chats keep 2.9. One client at a time gets 912 tok/s
on the label requests at `--parallel 8`, as a lone stream replays the graphs.

## Measurements

One DGX Spark (GB10) in NVIDIA's `pytorch:26.07-py3` container, checkpoint revision 81169a9, against vLLM serving
`nvidia/Qwen3.6-35B-A3B-NVFP4` with MTP=3 on the same Spark (`vllm/vllm-openai`, prefix caching, chunked prefill,
`--max-num-batched-tokens 8192`, `--gpu-memory-utilization 0.60`).

Decode with the [public benchmark command](README.md#measurements); drafted replies equal `"draft": false` ones:

| | Code sampled | Chat sampled | Code greedy | Chat greedy |
| --- | ---: | ---: | ---: | ---: |
| TensorFold | 179.4 tok/s | 141.3 tok/s | 166.6 tok/s | 162.8 tok/s |
| vLLM, MTP=3 | 120.6 tok/s | 100.6 tok/s | 122.1 tok/s | 117.0 tok/s |

Serial decoding (`--no-drafts`) runs at 86-88 tok/s. A round verifies up to four rows, and each row brings its
own eight experts, so a round reads about twice the bytes of one serial step; longer drafts pay off only when
most of their rows are kept.

Cold prefill with `tools/prefill_cold.py` (chat prompts from the Python standard library at exact rendered
lengths, a unique first line each so nothing resumes; median time to first token of three):

| Prompt tokens | 2,048 | 8,192 | 16,384 | 32,768 | 65,536 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TensorFold | 7,161 tok/s | 7,353 tok/s | 6,541 tok/s | 5,257 tok/s | 3,688 tok/s |
| vLLM, MTP=3 | 5,907 tok/s | 5,881 tok/s | 5,090 tok/s | 3,951 tok/s | 2,693 tok/s |

The server process peaks at 31.3 GiB (nvidia-smi) during the 65,536-token prompts; vLLM holds its memory
reservation, 72.9 GB at 0.60.
