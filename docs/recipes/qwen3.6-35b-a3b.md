# Qwen3.6-35B-A3B (`qwen3_5_moe`)

A CUDA engine only, for one DGX Spark (GB10, 128 GB unified memory). It reads
`mlx-community/Qwen3.6-35B-A3B-4bit` as stored (affine 4-bit, groups of 64; the router and shared-expert gate in
8 bits) and skips its vision tower. Package: `src/tensorfold/families/qwen3_5_moe/` (the kernels are listed in
`cuda/README.md`; most are TensorFold's shared CUDA kernels or Flash Next's). There is no MLX engine for this
family.

Status: phase 1, one stream, every round decodes one token (the serial reference). MTP drafting from
`mlx-community/Qwen3.6-35B-A3B-MTP-4bit` comes next. Numbers marked TODO wait for the GB10 run. The development
numbers below come from an RTX 5090 (sm_120, 32 GB) in `tensorfold-dev:26.07`, the x86 build of the same NVIDIA
PyTorch 26.07 base (PyTorch 2.13, CUDA 13.3, Triton 3.7.1). The 5090 has about six times GB10's memory bandwidth,
so its speeds do not predict GB10's.

## What decides the speed

- 35B total and 3B active parameters. Hidden size 2,048.
- 40 layers: 10 repeats of 3 Gated DeltaNet layers and 1 gated full-attention layer, each followed by an MoE.
- DeltaNet: 32 value heads and 16 query-key heads of dim 128, a 4-tap convolution, the output norm gated by
  silu(z).
- Attention: 16 query heads over 2 KV heads, head dim 256, a sigmoid output gate per head, q/k norms, rotary on the
  first 64 of each head's 256 dims. Every row reads all its keys (no sparse attention).
- MoE: 256 routed experts, top 8, width 512, plus a gated shared expert of width 512 in every layer. The router
  runs in fp32.
- Vocabulary 248,320, the head not tied to the embedding.
- About 1.73 GB of weights read per token at one row: 0.64 GB of experts (9 of 257 a layer), 0.72 GB of DeltaNet and
  attention projections, 0.29 GB of head and 84 MB of fp32 router tables. At 240 GB/s that is a floor of about
  7.2 ms, about 139 tok/s.
- Each extra verify row brings up to 8 new experts a layer, up to 14 MB of weights a layer (0.57 GB over the 40
  layers) that no other row shares. Wide draft trees are therefore expensive; chains of MTP drafts are the plan.

## CUDA

```bash
tensorfold serve mlx-community/Qwen3.6-35B-A3B-4bit --host 0.0.0.0 --port 8080
```

With no flags this is the recipe: one GPU, a 32,768-token prompt/reply window (smaller if memory is short), and
the MTP drafter when it has been pulled (`tensorfold pull mlx-community/Qwen3.6-35B-A3B-MTP-4bit`; `--drafter auto`
finds it). The drafter is checked at start. Until drafting is wired in, the engine says so and every round decodes
one token. `--context N` asks for a prompt/reply window of N tokens and is refused at start when it does not fit;
`--context 0` asks for the model's whole 262,144-token window. `--tp 2` is refused, because the model fits one Spark.
`--parallel N` is accepted but requests still take turns: this engine does not share rounds yet.

A request with `"draft": false` decodes from a fresh prefill in a second state and leaves the kept states alone:
the serial reference. The server keeps the state after the last request's prompt and after its reply, and a prompt
that extends either resumes from it: a second chat turn, or a longer completion. The response's `tensorfold` field
reports `cached` (the tokens resumed from), the prefill and decode times and their rates.

Before loading anything, the engine admits its context with `tensorfold.cuda.capacity`, as the other CUDA engines
do: the checkpoint's tensors in the kernels' layout (read from the safetensors headers; the router tables in fp32,
the vision tower skipped) plus the caches, snapshots, buffers and a workspace at the window, against the memory the
device has left after a 10% (at least 4 GiB) margin. On GB10 that is MemAvailable, which counts the page cache the
kernel gives back on demand. The server refuses a request whose prompt plus reply passes the window before it
streams anything. The estimate at the default window:

| | GiB |
| --- | ---: |
| Weights in the kernels' layout (the router tables in fp32) | 18.23 |
| Loading: three times the largest layer | 1.33 |
| Two sequence states: GDN states (two buffers each), conv windows, keys and values (20 KiB a token) | 1.49 |
| Three snapshots for prefix reuse (GDN states and conv windows) | 0.18 |
| Window buffers (512-row prefill chunks) and attention partials | 0.23 |
| Workspace | 1.00 |
| Total (weights plus the larger of loading and the rest) | 21.13 |

The states grow by 40 KiB a token of context (two states). On the 5090 the engine held 20.11 GiB after start-up at
the default window, with a peak of 20.17 GiB. GB10: TODO.

The first start builds the DeltaNet, lane-matmul and expert extensions with the container's `nvcc` and compiles
the Triton kernels. Later
starts read the weights with large sequential reads and drop the shards from the page cache as they go (on unified
memory the cached pages would sit beside the same bytes on the GPU), then run a short warm-up that compiles a
prefill chunk, a short window and one-row steps. On the 5090 with the checkpoint in the page cache: 9.2 s to load,
0.3 s to warm up. GB10 start-up time: TODO.

### Measured

| | 5090 (development) | GB10 |
| --- | ---: | ---: |
| Serial decode, one row a step, eager (no CUDA graphs) | 208 tok/s | TODO |
| Prefill of a 2,048-token prompt, 128-row chunks | 6,326 tok/s | TODO |
| Prefill of a 2,048-token prompt, 512-row chunks | 8,262 tok/s | TODO |
| Drafted decode, code and chat, sampled and greedy | not built yet | TODO |
| vLLM on the same prompts (`tools/bench_openai.py`, medians over seeds) | | TODO |

`tools/bench_q36_prefill.py` times the MoE stages and the real prefill. Prompt chunks run the decode kernels, so
prefill gives serial decoding's bits; the chunk size will be chosen on GB10.

### Exactness

Every kernel on the verify path gives a row the same bits whether it runs alone or as one row of a window, and
sampling is the keyed rule every CUDA family shares (`tensorfold/cuda/sampling.py`). Once drafting lands, a draft is
kept exactly when it is the token serial decoding samples there. Serial decoding runs through the same kernels.

- Matmuls: TensorFold's lane matmul, each output the same chain of tensor-core steps over the same 64-input groups
  in the same order at any row count. The K split is a constant of the weight's shape, pinned in `cuda/qmm.py`.
- Router: fp32 logits on CUDA cores in a fixed K order. The top 8 are taken on the fp32 logits, the lower expert id
  on ties.
- Experts: TensorFold's grouped expert kernels (decode form): a (row, slot) pair gets the same arithmetic whatever
  other pairs share its expert. The combine adds a row's 8 slots in pick order, then the shared expert, and rounds
  once.
- DeltaNet: a window's rows run in order inside one kernel from the committed state. Keeping a prefix replays
  those rows with the same update routine, compiled without FMA contraction.
- Attention: fixed 512-key chunks by absolute position, merged in position order.
- Prefill chunks and resumed prompts: rows never depend on their chunk, so any chunking, and a prompt resumed
  from a kept state, ends in the state of one fresh prefill.

What was checked, on the 5090 (sm_120). GB10 hashes differ from sm_120's, and each machine is its own reference:

| Check | Result |
| --- | --- |
| `tests/cuda/test_qwen36moe_{qmm,moe,gdn,attention}.py`: each kernel's rows alone against in a window (matmuls at 1 to 200 rows, MoE at 1 to 512 rows and in any order, DeltaNet at 1 to 128 rows; attention windows from key positions 0, 450 and 3,990) | bit-identical |
| `tests/cuda/test_qwen36moe_forward.py`: windows of 2 to 16 rows keeping any prefix, then continuing; prefill chunks of 1, 7, 64, 128 and 512 rows; resumed against fresh prompts; two sequences in one window | bit-identical logits and states |
| The real checkpoint: a 16-row window against 16 serial steps | 16 of 16 rows bit-identical |
| `tests/cuda/test_qwen36moe_engine.py`, random weights: the engine's streamed tokens against serial decoding, greedy and sampled; a prompt resumed from the kept prompt or reply (after a 5-row window and a 40-row chunk, and after a reply stopped early) against a fresh prefill; `draft=False` against the kept path | equal |
| Flash Next with the shared GDN and attention changes: its GDN and attention digests, and 659 hashes of its kernels and of a random-weight model's prefill, windows, serial, graph and MTP-drafted decoding, against v0.3.5 on the same GPU | byte-identical |
| The released server in-process, the real checkpoint: OpenAI chat requests with thinking off, greedy and sampled (seed 1234), 64 tokens: the served tokens against the engine's serial decoding of the same prompt; streamed against non-streamed; `"draft": false`; a request past the window refused before streaming | equal in all three, greedy and sampled; refused |
| The real checkpoint: a follow-up turn resumed from the kept reply (67 of 87 tokens) against a fresh prefill, greedy and sampled | equal |
| Drafted against serial by token-ID SHA-256 on every benchmark run, GB10 | TODO (with drafting) |

### Quality

Against a plain fp32 PyTorch forward with bf16 roundings (`cuda/reference.py`, TF32 off), teacher-forced over the
reference test's passages on the 5090: argmax agreement 99.3% over 3,114 tokens. On a passage the model has not
memorised, the NLL is 1.934 against the reference's 1.927. Greedy chat replies are coherent and name what the
prompts ask for. The MTP drafter's fc reads `[normed embedding | normed hidden]` in that order: 90.3% top-1
agreement with the next token, against 0.1% with the halves swapped. GB10: TODO.

### Limits

- One stream: requests decode one at a time, whatever `--parallel` says. The forward already takes a table of
  sequences (the batching seam).
- No drafting yet: every round decodes one token, with no CUDA graphs.
- The context capacity is fixed at start. Prefix reuse keeps two states, the last prompt's and the last reply's, and
  the caches hold one sequence, so a prompt that extends neither starts over.
- A prompt that does not fit the window is refused, and a reply stops where the window ends.

### Next

MTP chains with a confidence stop (depth 3 to 4) and a draft vocabulary rebuilt for this tokenizer, CUDA graphs for
the decode windows, then the GB10 measurements, the vLLM comparison and the SHA-256 exactness runs.
