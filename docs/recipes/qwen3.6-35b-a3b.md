# Qwen3.6-35B-A3B (`qwen3_5_moe`)

A CUDA engine only, for one DGX Spark (GB10, 128 GB unified memory). It reads
`mlx-community/Qwen3.6-35B-A3B-4bit` as stored (affine 4-bit, groups of 64; the router and shared-expert gate in
8 bits) and skips its vision tower. Package: `src/tensorfold/families/qwen3_5_moe/` (the kernels are listed in
`cuda/README.md`; most are TensorFold's shared CUDA kernels or Flash Next's). There is no MLX engine for this
family.

Status: one stream, MTP-drafted from `mlx-community/Qwen3.6-35B-A3B-MTP-4bit` with CUDA graphs. Numbers marked
TODO wait for the GB10 run. The development numbers below come from an RTX 5090 (sm_120, 32 GB) in an x86 build of
the same NVIDIA PyTorch 26.07 container (PyTorch 2.13, CUDA 13.3, Triton 3.7.1). The 5090 has about six times
GB10's memory bandwidth, so its speeds do not predict GB10's.

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

## Draft vocabulary provenance

The MTP head's draft head reads the public list in `src/tensorfold/families/qwen3_5_moe/cuda/draft_vocab.txt`:
76,882 sorted IDs, 31% of the 248,320-row head. The target sampler still reads the full vocabulary, so the list
affects proposals only; a token outside it can never be a draft, which costs speed, never correctness.

The corpus is Flash Next's: Homebrew CPython 3.14.5's standard-library `*.py` files, excluding `site-packages` and
`__pycache__` (see [its recipe](qwen3.8-flash-next.md#draft-vocabulary-provenance); with Flash Next's tokenizer the
same corpus and command reproduce its list, SHA-256 `88d5b483…`). Use `tokenizers==0.22.2` and `tools/draft_vocab.py`
with SHA-256 `1baf0dd08669355cf9cf6e32998e5436ce8712c3ab1bffe38d369f3fa3a86b56`. The tokenizer is the one the
server encodes with, `tokenizer.json` from `mlx-community/Qwen3.6-35B-A3B-4bit` (33 added tokens), with SHA-256:

```text
87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4
```

Place that tokenizer at `tokenizer.json` and copy the clean stdlib into an empty `cpython` directory, preserving
relative paths, as Flash Next's recipe describes. Then run from the repository root:

```bash
TOKENIZERS_PARALLELISM=false python3 -B tools/draft_vocab.py tokenizer.json draft_vocab.txt --size 76882 --keep-below 65536 --min-count 1 --added-tokens 'cpython/**/*.py'
```

The generator keeps every ID below 65,536 and the tokenizer's added IDs (248,044 to 248,076), then adds corpus IDs
by frequency and fills the remaining places with the lowest unused IDs. Expected output SHA-256:

```text
0fdfd41d8d7f310ee25240e81de7310f71d70f98136e51a9e172139365f10f94
```

On the reference test's three public passages the list holds 98.2% of the tokens. The drafting rates measured
below used the previous list (the same size, built from a local corpus); they do not qualify the current one.

## CUDA

```bash
tensorfold serve mlx-community/Qwen3.6-35B-A3B-4bit --host 0.0.0.0 --port 8080
```

With no flags this is the recipe: one GPU, a 32,768-token prompt/reply window (smaller if memory is short), and
MTP drafts once the drafter has been pulled (`tensorfold pull mlx-community/Qwen3.6-35B-A3B-MTP-4bit`; `--drafter
auto` finds it). A round verifies the pending token and up to 6 MTP drafts in one window. A chain always keeps its
first draft and stops before a later draft the head gives less than 50%. `--mtp-drafts N` sets the most drafts a
round (at most 15), and `--no-drafts`, or serving without the drafter, decodes one token a round. Windows of up to 8
rows, and the head's steps, replay CUDA graphs captured at start (a sequence that passes 8,192 keys captures the next
context bucket's graphs when it first needs them). `--context N` asks for a prompt/reply window of N tokens and is
refused at start when it does not fit;
`--context 0` asks for the model's whole 262,144-token window. `--tp 2` is refused, because the model fits one Spark.
`--parallel N` is accepted but requests still take turns: this engine does not share rounds yet.

A request with `"draft": false` decodes one token a round from a fresh prefill in a second state and leaves the
kept states alone: the serial reference. The server keeps the state after the last request's prompt and after its
reply, the head's cache included, and a prompt that extends either resumes from it: a second chat turn, or a longer
completion. The response's `tensorfold` field reports `cached` (the tokens resumed from), the prefill and decode
times and their rates, `token_sha` (drafted and `"draft": false` replies must match), and for a drafted reply the
drafts verified and kept, the acceptance and the tokens a round.

Before loading anything, the engine admits its context with `tensorfold.cuda.capacity`, as the other CUDA engines
do: the checkpoint's and the drafter's tensors in the kernels' layout (read from the safetensors headers; the router
tables in fp32, the vision tower skipped) plus the caches, snapshots, buffers and a workspace at the window and its
7 speculative slots (the depth plus one), against the memory the
device has left after a 10% (at least 4 GiB) margin. On GB10 that is MemAvailable, which counts the page cache the
kernel gives back on demand. The server refuses a request whose prompt plus reply passes the window before it
streams anything. The estimate at the default window:

| | GiB |
| --- | ---: |
| Weights in the kernels' layout (the router tables in fp32): the model 18.23, the MTP head 0.44 | 18.67 |
| Loading: three times the largest layer (the drafter loads afterwards and needs less) | 1.33 |
| Two sequence states: GDN states (two buffers each), conv windows, keys and values (22 KiB a token, the head's included) | 1.61 |
| Three snapshots for prefix reuse (GDN states, conv windows, the head's waiting row) | 0.18 |
| Window buffers (512-row prefill chunks), the head's step buffers, the 76,882-row draft head, attention partials | 0.33 |
| Workspace | 1.00 |
| Total (weights plus the larger of loading and the rest) | 21.79 |

The states grow by 44 KiB a token of context (two states). On the 5090 (`tools/bench_q36.py serve`) the engine held
20.77 GiB after start-up at the default window, with a peak of 20.83 GiB; the device reported 22.92 GiB in use (the
CUDA context and graphs sit outside PyTorch's count, inside the workspace). GB10: TODO.

The first start builds the DeltaNet, lane-matmul and expert extensions with the container's `nvcc` and compiles the
Triton kernels. Later starts read the weights with large sequential reads and drop the shards from the page cache as
they go (on unified memory the cached pages would sit beside the same bytes on the GPU), read the drafter, capture
the decode graphs (32: windows of 1 to 7 rows at both GDN buffer parities, the head's steps, one-row steps for the
serial state) and run a short warm-up request. On the 5090 with the checkpoints in the page cache: 9.7 s to load,
2.1 s to capture and warm up. GB10 start-up time: TODO.

### Measured

Decode speed after the first token, one stream, from the engine as `tensorfold serve` builds it
(`python tools/bench_q36.py serve`): three chat prompts and two JSON-extraction prompts with thinking off, 256
tokens each, greedy and sampled (the checkpoint's generation config: temperature 1, top-k 20, top-p 0.95). Prefill:
`python tools/bench_q36.py prefill`. On the 5090 only the ratios mean anything for GB10, and even they will move: a
verify row costs relatively more there. The 5090 column predates the current draft vocabulary.

| | 5090 (development) | GB10 |
| --- | ---: | ---: |
| Serial decode with CUDA graphs, greedy / sampled | 220 / 214 tok/s | TODO |
| Serial decode eager (no graphs), greedy / sampled | 210 / 201 tok/s | TODO |
| Drafted against serial, all five prompts, greedy / sampled | 2.29x / 2.09x | TODO |
| Drafted against serial, chat prompts | 1.53-2.03x | TODO |
| Drafted against serial, JSON prompts | 3.63-4.17x | TODO |
| Draft acceptance, greedy / sampled; tokens a round | 0.758 / 0.793; 2.98 / 2.74 | TODO |
| Prefill of a 2,048-token prompt, 128-row / 512-row chunks | 6,326 / 8,262 tok/s | TODO |
| vLLM on the same prompts (`tools/bench_openai.py`, medians over seeds) | | TODO |

A one-row step with graphs takes 4.44 ms at a 4,096- and at a 32,768-position cache. JSON replies draft deep (about
6 tokens a round) because the head's chains rarely fall under the 50% stop there; chat chains stop earlier. Depth
and the stop were chosen on the 5090 and need retuning on GB10 (`tools/bench_q36.py drafting`). `tools/bench_q36.py
prefill` times the real prefill and `moe` the MoE stages. Prompt chunks run the decode kernels, so prefill gives
serial decoding's bits; the chunk size will be chosen on GB10.

### Exactness

Every kernel on the verify path gives a row the same bits whether it runs alone or as one row of a window, and
sampling is the keyed rule every CUDA family shares (`tensorfold/cuda/sampling.py`), so a draft is kept exactly when
it is the token serial decoding samples there. Serial decoding runs through the same kernels. A draft is sampled with
the same keyed rule at its position, so a sampled draft shares its position's noise with the verify.

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
- A round never drafts past the reply's length, so the caches end where serial decoding leaves them.
- CUDA graphs replay the eager kernels with the same arguments. The context bucket only bounds which key chunks
  attention launches; chunks past a row's keys write nothing, so every bucket, and eager, give the same bits.

What was checked, on the 5090 (sm_120). GB10 hashes differ from sm_120's, and each machine is its own reference:

| Check | Result |
| --- | --- |
| `tests/cuda/test_qwen36moe_{qmm,moe,gdn,attention}.py`: each kernel's rows alone against in a window (matmuls at 1 to 200 rows, MoE at 1 to 512 rows and in any order, DeltaNet at 1 to 128 rows; attention windows from key positions 0, 450 and 3,990) | bit-identical |
| `tests/cuda/test_qwen36moe_forward.py`: windows of 2 to 16 rows keeping any prefix, then continuing; prefill chunks of 1, 7, 64, 128 and 512 rows; resumed against fresh prompts; two sequences in one window | bit-identical logits and states |
| The real checkpoint: a 16-row window against 16 serial steps | 16 of 16 rows bit-identical |
| `tests/cuda/test_qwen36moe_mtp.py`, random weights: drafted against serial decoding, greedy and sampled, eager and graphs, depths 1 to 6, with and without the confidence stop, with a draft vocabulary, from a resumed prompt, from a state the head lags behind, with an oracle drafter that keeps long prefixes, and across the 8,192-key context bucket; graph replays against eager | equal tokens, bit-identical logits and states |
| `tests/cuda/test_qwen36moe_engine.py`, random weights, three engines (serial, drafted as served, 4 drafts without a stop): the engine's streamed tokens against serial decoding, greedy and sampled, with eos and early stops mid-round; a prompt resumed from the kept prompt or reply (after a 5-row window and a 40-row chunk, and after a reply stopped early) against a fresh prefill; `draft=False` against the kept path | equal |
| The released server in-process, the real checkpoint and drafter: 3 chat and 2 JSON prompts, greedy and sampled, 256 tokens: the served (drafted) reply and the served `"draft": false` reply against the engine's serial decoding, by token-ID SHA-256 | 10 of 10 equal |
| Flash Next with the shared GDN and attention changes: its GDN and attention digests, and 659 hashes of its kernels and of a random-weight model's prefill, windows, serial, graph and MTP-drafted decoding, against v0.3.5 on the same GPU | byte-identical |
| The released server in-process, the real checkpoint: OpenAI chat requests with thinking off, greedy and sampled (seed 1234), 64 tokens: the served tokens against the engine's serial decoding of the same prompt; streamed against non-streamed; `"draft": false`; a request past the window refused before streaming | equal in all three, greedy and sampled; refused |
| The real checkpoint: a drafted follow-up turn resumed from the kept reply (67 of 87 tokens) against a fresh prefill, greedy and sampled | equal |
| The real MTP head against the fp32 reference head on the same hidden rows | relative error 0.6-0.9%, the same top-1 on 98.6-99.3% of rows |
| `tests/cuda/test_qwen36moe_engine.py`: 15 drafts a round verifying 16-row windows from CUDA graphs; an exactly repeated prompt; two concurrent requests to the served engine, each against its solo reply | TODO (GB10 run) |
| `tests/test_cuda_qwen36moe_{package,load,admission,reference}.py`, no GPU: the package's refusals, the loader's key map and centred-norm groups, the memory admission, the reference on tiny weights | package, load and reference pass on a Mac; admission TODO (GB10 run) |
| Drafted against serial by token-ID SHA-256 on every benchmark run, GB10 | TODO |

#### Rerunning the hash checks

Two tools hash kernel and model outputs on fixed random inputs, one JSON entry per output. Run them from the
repository root in the CUDA container, on the same GPU and image for both sides of a comparison, and diff the files:

```bash
python tools/hash_flashnext.py fn.json   # 659 Flash Next hashes: lane matmuls, experts, GDN, gdn_io, attention, a model
python tools/hash_qwen36.py q36.json     # 363 Qwen3.6 hashes: every matmul shape, the MoE stages, a model
```

For Flash Next the reference is v0.3.5 itself: in a v0.3.5 checkout, copy this branch's
`tools/hash_flashnext.py` and `tests/cuda/test_qwen36moe_attention.py` (whose `fn_outputs` uses v0.3.5's APIs
only), run the tool there, then on this branch, and require every hash equal. For Qwen3.6 the reference is the
previous commit's output. `FN_HASHES` in `tests/cuda/test_qwen36moe_attention.py` holds the tool's `attn/` entries
and `FLASHNEXT` in `tests/cuda/test_qwen36moe_gdn.py` holds `flashnext_digests()` from v0.3.5, per compute
capability; the GB10 (12, 1) entries come from the GB10 run.

### Quality

Against a plain fp32 PyTorch forward with bf16 roundings (`cuda/reference.py`, TF32 off), teacher-forced over the
reference test's public passages on the 5090 (`python tools/bench_q36.py quality`): argmax agreement 99.3% over
3,114 tokens. On a passage the model has not
memorised, the NLL is 1.934 against the reference's 1.927. Greedy chat replies are coherent and name what the
prompts ask for. The MTP drafter's fc reads `[normed embedding | normed hidden]` in that order: 90.3% top-1
agreement with the next token, against 0.1% with the halves swapped. GB10: TODO.

### Limits

- One stream: requests decode one at a time, whatever `--parallel` says. The forward already takes a table of
  sequences.
- The context capacity is fixed at start. Prefix reuse keeps two states, the last prompt's and the last reply's, and
  the caches hold one sequence, so a prompt that extends neither starts over. An exactly repeated prompt prefills
  again: a kept state holds no logits for its last position.
- A prompt that does not fit the window is refused, and a reply stops where the window ends.

### Next

The GB10 measurements (serial and drafted speed, prefill, memory, start-up), retuning the depth and the chain's
stop there, the vLLM comparison and the SHA-256 exactness runs on GB10.
