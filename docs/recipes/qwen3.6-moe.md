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
vLLM serves. The engine reads them as they ship; each layer's format is the checkpoint's:

| Tensors | Stored as | Read by |
| --- | --- | --- |
| Routed experts, shared expert, `lm_head` | `W4A16_NVFP4`: E2M1 nibbles, an e4m3 scale per 16 values, an fp32 scale per tensor | `tensorfold/cuda/nvfp4_experts.cu` on the stored bytes: a layer's experts and its shared expert in one table on the experts plan, the heads in its dense form; bf16 activations |
| DeltaNet `in_proj_qkv`, `in_proj_z`, `out_proj`; attention `q/k/v/o_proj` | `FP8`: e4m3 codes, an fp32 scale per tensor | `fp8.py`: the codes (exact in bf16) are the tensor-core operand and the scale multiplies the fp32 sums; bf16 activations |
| Embedding, routers, shared-expert gates, `in_proj_a/b`, conv, norms, the MTP layer | bf16 | as stored (`bf16.matmul`), except the MTP layer's experts: they only draft, and ride the MLX route's 4-bit experts kernel, requantized at load |

The expert kernel is `experts.cu`'s decode form on the FP4 format: the nibbles in its fragment order, and in place
of each 64-input group's scales and biases the group's e4m3 block scales, a word a lane. Lane t of a quad carries
one NVFP4 block's inputs, so it decodes its fragment to exact bf16 values (2 x code x block scale: at most six
significant bits), and each (row, column) is one fp32 mma chain over K, split into slices fixed by K and added in
order, times the per-tensor scale. A row is one mma row, so its bits never depend on the other rows of a call.
The FP8 projections stay one byte a weight: widened to bf16 at load they would take 2.38 GiB instead of 1.19 GiB,
and a decode step's 130 FP8 matmuls 2.06 ms instead of 1.31 ms. Prompt chunks take K in one slice in both kernels
(chunk-invariant bits of their own, as the MLX route's prompt path has). The checkpoint's `input_scale` tensors are
calibration scales for vLLM's FP8 activations, and `kv_cache_quant_algo: FP8` names vLLM's cache format: neither
is part of the weights, so activations and the KV cache stay bf16. The RMSNorm weights are stored zero-centred
(the model applies `1 + w`, which MLX conversions store instead) and become that scale in fp32 at load; DeltaNet's
gated norm is stored as applied. The MTP head drafts over the draft vocabulary's rows of the NVFP4 head. The
vision tower is not read. Exactness is the MLX route's: drafted replies equal `"draft": false`, `--parallel N`
equals solo, a resumed prompt equals a fresh prefill, and startup admits the window before loading.

Fidelity on eight public passages of 1,024 tokens (`pydoc_data` topics and standard-library source): next-token
NLL and the share of positions whose top token equals the fp32 forward of the original bf16 release
(`Qwen/Qwen3.6-35B-A3B` at `995ad96`) or of this checkpoint's exactly dequantized weights. Verify windows give
the tokens serial decoding gives; the prompt path fills the context.

| Scores | NLL (nats/token) | Top 1 = bf16 release | Top 1 = NVFP4, fp32 |
| --- | ---: | ---: | ---: |
| bf16 release, fp32 | 0.4464 | 100% | 93.62% |
| NVFP4 checkpoint, fp32 | 0.5041 | 93.62% | 100% |
| NVFP4 route, verify windows | 0.5002 | 93.51% | 98.08% |
| NVFP4 route, prompt path | 0.4993 | 93.56% | 98.36% |
| MLX 4-bit route, verify windows | 0.5183 | 90.69% | 89.67% |
| MLX 4-bit route, prompt path (FP8 activations) | 0.5444 | 89.63% | 88.67% |

```bash
python -m tensorfold.families.qwen3_5_moe.cuda.reference reference <bf16 or NVFP4 folder> out.pt <tokenizer folder>
python -m tensorfold.families.qwen3_5_moe.cuda.reference route <any checkpoint folder> out.pt <tokenizer folder>
python -m tensorfold.families.qwen3_5_moe.cuda.reference compare reference.pt route.pt ...
```

On one RTX PRO 6000 Blackwell Max-Q (NGC 26.07, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), with 8 client
threads over HTTP and the workloads of [Concurrent requests](#concurrent-requests), both checkpoints measured the
same day:

| | NVFP4, `--parallel 1` | MLX 4-bit, `--parallel 1` | NVFP4, `--parallel 8` | MLX 4-bit, `--parallel 8` |
| --- | ---: | ---: | ---: | ---: |
| Label JSON, 32 requests | 922 tok/s | 903 tok/s | 1,578 tok/s | 1,496 tok/s |
| p50 / p95 latency | 9.7 / 10.3 s | 8.5 / 9.1 s | 5.8 / 7.5 s | 5.1 / 7.6 s |
| Chat, 48 requests | 457 tok/s | 427 tok/s | 1,107 tok/s | 1,100 tok/s |
| p50 / p95 latency | 3.0 / 3.5 s | 3.3 / 4.0 s | 1.1 / 2.0 s | 1.2 / 2.0 s |
| Peak memory (nvidia-smi) | 24.2 GB | 22.7 GB | 24.9 GB | 23.4 GB |

Every NVFP4 reply's token SHA-256 is the same at `--parallel 1`, at `--parallel 8` and with `"draft": false`
(80 of 80), and without expandable segments. Drafts keep 9.8 tokens a round on the labels and 3.0 on the chats
(MLX: 9.3 and 2.9). A verify forward of four rows takes 5.1 ms of GPU time (MLX: 5.0 ms): the experts' gate/up
0.84 ms and down 0.47 ms (an NVFP4 and an MLX 4-bit g64 weight are both 0.5625 bytes), the FP8 projections
1.05 ms and their split-K sums 0.55 ms, the head 0.21 ms. A 4,096-token prompt prefills at 19,500 tok/s (MLX:
22,100), the FP8 projections' bf16 inputs the difference. GB10 figures are still to be measured.

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
Prompts prefill in chunks; the head absorbs every prompt row but the last. States are kept at the second
message's start, the last assistant turn's start and prompt ends, so a prompt sharing a system block or
extending a conversation resumes there with a fresh prefill's bits.

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

### Structured output

With `pip install 'tensorfold[grammar]'` (xgrammar), the engine enforces `response_format` JSON schemas, alone and
with `--parallel N` ([API reference](../api.md#structured-output)). A chain or copied continuation loses its first
draft the grammar rejects, or a stop token, and every row after it. The verify forward runs as without a schema (a
graph replay or a shared eager round); its logits are then masked by each row's path before sampling, and the grammar
follows the kept tokens. The head drafts as it does without a schema. A constrained reply equals its `"draft": false`
reply and, with `--parallel N`, its solo run.

The label requests above with a JSON schema (an array of `{id, themes, tone, evidence}` objects, themes from the
allowed list), on the same GPU and settings:

| | `--parallel 1` | `--parallel 4` | `--parallel 8` |
| --- | ---: | ---: | ---: |
| Label JSON with the schema | 852 tok/s | 1,162 tok/s | 1,340 tok/s |
| p50 / p95 latency | 8.1 / 8.9 s | 5.6 / 7.5 s | 4.8 / 8.3 s |
| The same requests without it | 907 tok/s | 1,248 tok/s | 1,489 tok/s |

26 of the 32 replies end and validate; the other six reach the 1,410-token limit and stop as incomplete JSON
(`finish_reason: "length"`). Every reply's token SHA-256 is the same at each N and with `"draft": false`. One stream
decodes 1,097 tok/s against 1,143 without the schema: the grammar's windows and masks add about 0.2 ms to an 8.1 ms
round. The rest of the gap is prefill spread over shorter replies (27,857 tokens against 32,032).

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
