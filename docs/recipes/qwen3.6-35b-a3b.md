# Qwen3.6-35B-A3B (`qwen3_5_moe`)

A CUDA engine only, for one DGX Spark (GB10, 128 GB unified memory). It reads
`mlx-community/Qwen3.6-35B-A3B-4bit` as stored (affine 4-bit, groups of 64; the router and shared-expert gate in
8 bits) and skips its vision tower. Package: `src/tensorfold/families/qwen3_5_moe/` (the kernels are listed in
`cuda/README.md`). There is no MLX engine for this family.

Status: phase 1, one stream, MTP-drafted from `mlx-community/Qwen3.6-35B-A3B-MTP-4bit` with CUDA graphs. Numbers
marked TODO wait for the GB10 run. The development
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
  layers) that no other row shares. Wide draft trees are therefore expensive, so the engine drafts chains.
- MTP: one draft layer (gated attention over its own cache and the same 257-expert MoE) that reads the model's
  hidden row after the final norm and the next token's embedding, from a separate repository (MLX's converter drops
  it from the main checkpoint). Its head scores 76,882 of the 248,320 token ids (`cuda/draft_vocab.txt`), 31%
  of a full head's bytes a draft.

## DGX Spark (CUDA)

```bash
tensorfold serve mlx-community/Qwen3.6-35B-A3B-4bit --host 0.0.0.0 --port 8080
```

With no flags this is the recipe: one GPU, a 32,768-token context, and MTP drafts once the drafter has been pulled
(`tensorfold pull mlx-community/Qwen3.6-35B-A3B-MTP-4bit`; `--drafter auto` finds it). A round verifies the pending
token and up to 6 MTP drafts in one window. A chain always keeps its first draft and stops before a later draft the
head gives less than 50%. `--mtp-drafts N` sets the most drafts a round (at most 15), and `--no-drafts`, or serving
without the drafter, decodes one token a round. Windows of up to 8 rows, and the head's steps, replay CUDA graphs
captured at start. `--context N` sets the caches' positions (prompt plus reply), and `--context 0` gives the
model's whole 262,144-token window. `--tp 2` is refused, because the model fits one Spark.

A request with `"draft": false` decodes one token a round from a fresh prefill in a second state and leaves the
kept states alone: the serial reference. The server keeps the state after the last request's prompt and after its
reply, the head's cache included, and a prompt that extends either resumes from it: a second chat turn, or a longer
completion. The response's `tensorfold` field reports `cached` (the tokens resumed from), the prefill and decode
times and their rates, and for a drafted reply the drafts verified and kept, the acceptance and the tokens a round.

Before loading anything, the engine checks that the model fits in free memory. On GB10, free memory is the larger
of CUDA's free memory and `/proc/meminfo`'s MemAvailable, because the kernel gives back page cache on demand and
CUDA does not count it. The plan at the default context:

| | GiB |
| --- | ---: |
| Weights in the kernels' layout (the router tables in fp32): the model 18.24, the MTP head with its draft head 0.53 | 18.76 |
| While loading: the loaded drafter and the model's head in MLX's layout, from which the draft head is cut | 0.71 |
| Two sequence states: GDN states (two buffers each), conv windows, keys and values (22 KiB a token, the head's included) | 1.61 |
| Three snapshots for prefix reuse (GDN states, conv windows, the head's waiting row) | 0.18 |
| Window buffers (512-row prefill chunks), the head's step buffers and attention partials | 0.22 |
| Reserve | 1.00 |
| Total | 22.49 |

The states grow by 44 KiB a token of context (two states), so the model's whole window needs about 33 GiB. On the
5090 the engine held 20.72 GiB after start-up at the default context, with a peak of 20.78 GiB against the plan's
21.49 GiB without the reserve; the device reported 23.06 GiB in use (the CUDA context and graphs sit outside
PyTorch's count, inside the reserve). GB10: TODO.

The first start builds the DeltaNet extension with the container's `nvcc` and compiles the Triton kernels. Later
starts read the weights with large sequential reads and drop the shards from the page cache as they go (on unified
memory the cached pages would sit beside the same bytes on the GPU), read the drafter, capture the decode graphs
(32: windows of 1 to 7 rows at both GDN buffer parities, the head's steps, one-row steps for the serial state) and
run a short warm-up request. On the 5090 with the checkpoints in the page cache: 8.7 s to load, 3.5 s to capture and
warm up. GB10 start-up time: TODO.

### Measured

Decode speed after the first token, one stream, through the released server in-process (`tests/cuda/test_qwen36moe_engine.py`):
three chat prompts and two JSON-extraction prompts with thinking off, 256 tokens each, greedy and sampled (the
checkpoint's generation config: temperature 1, top-k 20, top-p 0.95). On the 5090 only the ratios mean anything
for GB10, and even they will move: a verify row costs relatively more there.

| | 5090 (development) | GB10 |
| --- | ---: | ---: |
| Serial decode with CUDA graphs, greedy / sampled | 206 / 200 tok/s | TODO |
| Serial decode eager (no graphs), greedy / sampled | 176 / 169 tok/s | TODO |
| Drafted against serial, all five prompts, greedy / sampled | 1.98x / 1.85x | TODO |
| Drafted against serial, chat prompts | 1.40-1.85x | TODO |
| Drafted against serial, JSON prompts | 2.90-3.34x | TODO |
| Draft acceptance, greedy / sampled; tokens a round | 0.757 / 0.791; 3.02 / 2.79 | TODO |
| Prefill of a 2,048-token prompt, 128-row / 512-row chunks | 1,395 / 1,134 tok/s | TODO |
| vLLM on the same prompts (`tools/bench_openai.py`, medians over seeds) | | TODO |

A one-row step with graphs takes 4.76 ms at a 4,096- and at a 32,768-position cache. JSON replies draft deep (about
6 tokens a round) because the head's chains rarely fall under the 50% stop there. Chat chains stop earlier. Depth
and the stop were chosen on the 5090 and need retuning on GB10. Prefill is slower at 512-row chunks than at 128
because the grouped expert kernels still visit empty tiles. An early exit for those tiles is the next prefill change,
and the chunk size will be chosen on GB10 after it lands.

### Exactness

Every kernel on the verify path gives a row the same bits whether it runs alone or as one row of a window, and
sampling is the keyed rule of `engine/exact_sampling.py`, so a draft is kept exactly when it is the token serial
decoding samples there. Serial decoding runs through the same kernels. A draft is sampled with the same keyed rule at
its position, so a sampled draft shares its position's noise with the verify.

- Matmuls: each output is the same chain of tensor-core steps over the same 64-input groups in the same order at
  any row count. The K split is a constant of the weight's shape, frozen in `cuda/qmm.py`.
- Router: fp32 logits on CUDA cores in a fixed K order. The top 8 are taken on the fp32 logits, the lower expert id
  on ties.
- Experts: a (row, expert) pair gets the same arithmetic whatever other rows share the expert. The combine adds a
  row's 8 slots in pick order, then the shared expert, and rounds once.
- DeltaNet: a window's rows run in order inside one kernel from the committed state. Keeping a prefix replays
  those rows with the same update routine, compiled without FMA contraction.
- Attention: fixed 512-key chunks by absolute position, merged in position order.
- Prefill chunks and resumed prompts: rows never depend on their chunk, so any chunking, and a prompt resumed
  from a kept state, ends in the state of one fresh prefill.
- A round never drafts past the reply's length, so the caches end where serial decoding leaves them.
- CUDA graphs replay the eager kernels with the same arguments: eager and graph decoding give the same bits.

What was checked, on the 5090 (sm_120). GB10 hashes differ from sm_120's, and each machine is its own reference:

| Check | Result |
| --- | --- |
| `tests/cuda/test_qwen36moe_{qmm,moe,gdn,attention}.py`: each kernel's rows alone against in a window (matmuls, MoE and DeltaNet at 1 to 128 rows, MoE rows in any order; attention windows from key positions 0, 450 and 3,990) | bit-identical |
| `tests/cuda/test_qwen36moe_forward.py`: windows of 2 to 16 rows keeping any prefix, then continuing; prefill chunks of 1, 7, 64, 128 and 512 rows; resumed against fresh prompts; two sequences in one window | bit-identical logits and states |
| The real checkpoint: a 16-row window against 16 serial steps | 16 of 16 rows bit-identical |
| `tests/cuda/test_qwen36moe_mtp.py`, random weights: drafted against serial decoding, greedy and sampled, eager and graphs, depths 1 to 6, with and without the confidence stop, with a draft vocabulary, from a resumed prompt, from a state the head lags behind, and with an oracle drafter that keeps long prefixes; graph replays against eager | equal tokens, bit-identical logits and states |
| `tests/cuda/test_qwen36moe_engine.py`, random weights, three engines (serial, drafted as served, 4 drafts without a stop): the engine's streamed tokens against serial decoding, greedy and sampled, with eos and early stops mid-round; a prompt resumed from the kept prompt or reply (after a 5-row window and a 40-row chunk, and after a reply stopped early) against a fresh prefill; `draft=False` against the kept path | equal |
| The released server in-process, the real checkpoint and drafter: 3 chat and 2 JSON prompts, greedy and sampled, 256 tokens: the served (drafted) reply and the served `"draft": false` reply against the engine's serial decoding, by token-ID SHA-256 | 10 of 10 equal |
| The same server: a 64-token chat reply, greedy and sampled (seed 1234), streamed against non-streamed against `"draft": false` against serial decoding | equal |
| The real checkpoint: a drafted follow-up turn resumed from the kept reply (67 of 87 tokens) against a fresh prefill, greedy and sampled | equal |
| The real MTP head against the fp32 reference head on the same hidden rows | relative error 0.6-0.8%, the same top-1 on 99.1% of rows |
| Drafted against serial by token-ID SHA-256 on every benchmark run, GB10 | TODO |

### Quality

Against a plain fp32 PyTorch forward with bf16 roundings (`cuda/reference.py`, TF32 off), teacher-forced over the
reference test's passages on the 5090: argmax agreement 99.3% over 3,114 tokens. On a passage the model has not
memorised, the NLL is 1.934 against the reference's 1.927. Greedy chat replies are coherent and name what the
prompts ask for. The MTP drafter's fc reads `[normed embedding | normed hidden]` in that order: 90.3% top-1
agreement with the next token, against 0.1% with the halves swapped. GB10: TODO.

### Limits

- One stream: requests decode one at a time. The forward already takes a table of sequences (the batching seam).
- The MoE grouping kernel is compiled for each distinct window or chunk size (the row count is one of its compile-time
  constants). A request whose prompt tail has a new size waits for one compile, about 1 s on the 5090. Triton caches
  it on disk, so it happens once per size per machine.
- The context capacity is fixed at start. Prefix reuse keeps two states, the last prompt's and the last reply's, and
  the caches hold one sequence, so a prompt that extends neither starts over.
- A prompt that does not fit the context is refused, and a reply stops where the context ends.

### Next

The GB10 measurements (serial and drafted speed, prefill, memory, start-up), retuning the depth and the chain's
stop there, the expert early exit for prefill, the vLLM comparison and the SHA-256 exactness runs on GB10.
