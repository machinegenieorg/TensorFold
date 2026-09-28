# Qwen3.6-35B-A3B on CUDA

The CUDA engine for Qwen3.6-35B-A3B (`model_type` `qwen3_5_moe`) on one DGX Spark (GB10). It reads the MLX 4-bit
checkpoint (`mlx-community/Qwen3.6-35B-A3B-4bit`: affine 4-bit, groups of 64, the router and shared-expert gate in
8 bits) as stored and skips the vision tower. MLX's converter drops the checkpoint's MTP head, so drafts come from
the MTP drafter split out of the original checkpoint (`mlx-community/Qwen3.6-35B-A3B-MTP-4bit`, `model_type`
`qwen3_5_mtp`), which `--drafter auto` passes in once pulled. There is no MLX engine for this family (no `load`),
so `tensorfold serve` refuses the MLX backend for it.

Every kernel on the verify path gives a row the same bits whether it runs alone or as one row of a window. Serial
decoding runs through the same kernels, so a drafted token is the token serial decoding produces on this machine.

## Kernels

Qwen3.6 writes few kernels of its own. The dense matmuls and the experts run TensorFold's shared CUDA kernels
(`tensorfold/cuda/`) at MLX's groups of 64; the GDN chain, attention and the top-k are Flash Next's
(`families/qwen4_exp/cuda/`), shared and run at Qwen3.6's shapes: GDN heads (16, 32) with the SiLU gate, dense
attention (`sparse=False`, no indexer).

| File | Kernel | What it computes | Why the bits do not depend on the row count |
| --- | --- | --- | --- |
| `tensorfold/cuda/kernels/qmm.py` (`qmm.cu`), via `qmm.py` | `qmm_kernel`, `reduce_kernel` | the dense 4-bit matmuls (the stacked GDN and attention projections, GDN `out_proj`, `o_proj`, the head) on words packed once at load: per 64-input group a tensor-core dot of the bf16 rows and the integer-valued weights, then `acc + p * scale + xs * bias`, groups in order | a row's sums never depend on its row tile (16, 32 or 64) or the other rows; the K split is pinned per weight shape (`SPLITS`) and its slices are added in slice order |
| | `_group_sums` | fp32 sums of each 64 bf16 inputs (the bias term of the next matmul) | one row at a time |
| `qwen3_5/cuda/glue.py`, via `qmm.py` | `_embed` | an embedding row gathered from MLX's layout and dequantized | one program per row and group |
| `tensorfold/cuda/experts.py` (`experts.cu`), via `moe.py` | `plan`, `run` (decode form) | the 256 routed experts and the shared expert as one table of 257: the (row, slot) pairs grouped by expert into items of 16, then per item and column block `bf16(bf16(silu(bf16 gate)) * bf16 up)`, and the down projection to fp32 `y[row, slot]` | a pair gets the same arithmetic whatever other pairs share its item; prompt chunks use the decode form too |
| `moe.py` | `_router` | fp32 router logits `[R, 257]` from the loader's fp32 table (256 router rows, then the shared expert's gate row) | an IEEE dot on CUDA cores, one fused multiply-add chain per (row, expert) over K in order |
| Flash Next's `moe.py`, via `moe.py` | `_topk_rows` | each row's top 8 by fp32 logit (the lower id on ties), weights `exp(l_k - l_0)` renormalised over the 8 and rounded to bf16, the shared gate `bf16(sigmoid(bf16(logit)))` in slot 8 | a row's picks read its own logits |
| `moe.py` | `_combine` | `bf16(sum of w_k * y_k)` over the 8 routed slots in pick order, then the shared expert, one rounding; fused with the residual add, the next RMSNorm (`1 + w`, fp32) and its 64-sums | one row at a time, slots in a fixed order |
| `forward.py` | `_add_norm` | residual add (a bf16 branch, or a matmul's split-K slices added in slice order and rounded once), RMSNorm with the fp32 `1 + w`, and the 64-sums | one row at a time |
| | `_conv_tail` | a GDN layer's next conv window (the last 3 of the old window and a prefill chunk's q, k, v rows) | copies |
| Flash Next's `forward.py` | `_shift_windows` | the conv windows after keeping the first rows of a window | copies |
| Flash Next's `gdn.cu` (`gdn.py`) | `chain_kernel` (`NK=16, NV=32`, SiLU) | a window's rows in order from the committed state in one kernel: depthwise conv and SiLU, q/k L2 norms, `g = -exp(A_log) softplus(a + dt_bias)` and `beta = sigmoid(b)`, the delta rule in fp32, the gated RMSNorm `w * norm(y) * silu(z)`; writes the state after the last row and what a replay needs | each row updates the state its predecessors left, in row order, the same whether the window has one row or 32 |
| | `replay_kernel` | the commit of a shorter prefix: the kept rows replayed from the committed state | the chain's update routine, compiled with `--fmad=false`, so a kept prefix has the bits of serial steps |
| Flash Next's `glue.py` | `_attn_prep` | q and k RMSNorm (`1 + w`, fp32), rotate-half RoPE on the first 64 of 256 dims at the row's absolute position (fp32, one rounding), keys and values written to the cache at their positions; no indexer (`index_heads=0`) | one program per row and head |
| | `_attn_gate` | `bf16(o * sigmoid(gate))` with the gate from the second half of each head's q_proj rows, and the 64-sums for `o_proj` | one program per row and head |
| Flash Next's `attention.py` | `_chunks`, `_tile`, `_merge` | full attention of 16 query heads over 2 KV heads of 256: keys in fixed 512-key chunks by absolute position, eight 64-key tensor-core tiles each, the chunks merged in position order; long windows run in blocks of `attn_rows` rows | a row's chunks hold the same keys alone or in a window; `sparse=False` keeps Flash Next's sparse attention off at any context |
| `tensorfold/cuda/sampling.py` (`decode.sample_rows`) | `sample_rows` | the argmax (lowest id on ties), or the top-k plus a margin on the GPU and the keyed draw of `engine/exact_sampling.py` on the host | each row from its own logits and absolute position |
| `mtp.py` | `_norm_into` | the MTP head's two pre-fc norms (the next token's embedding and the model's hidden row) into the one row its fc reads; the head then runs the same matmul, attention and MoE kernels over its own cache | draft-only: the head proposes and the verify window decides, so it needs no row invariance; it is deterministic, so drafts, and speeds, repeat |

No cuBLAS, `torch.matmul` or `F.linear` on the verify path: their algorithms depend on the row count.

## Matmuls (`qmm.py`)

Dense projections run the shared lane matmul (`tensorfold.cuda.kernels.qmm`) at group size 64. For weight group g
(64 inputs, one scale s and one bias b per output column):

```text
P[m, n, g] = x[m, g-block] . q[n, g-block]     tensor cores, bf16 x integer-valued bf16 -> fp32
y[m, n]    = sum over g, in order, of  s[n, g] * P[m, n, g] + b[n, g] * xs[m, g]
```

where xs[m, g] is the fp32 sum of the group's 64 bf16 inputs. The K groups are split into SK slices that are a
constant of the weight's shape (`SPLITS`, pinned at the shared rule's choice for these shapes, about 192 programs a
matrix at one row), and the slices are added in slice order. The K slices are part of the arithmetic: a new value
changes every row's bits alike, so serial and windows stay equal, but stored hashes move; retime them on GB10 before
hashes are stored there. `matmul` dispatches on the weight's type; `Q4` is the only format today, and an NVFP4 type
would add its own branch without touching the callers. The embedding stays in MLX's row layout for the gather.

## MoE (`moe.py`)

```text
router    fp32 logits [R, 257] = x . W for the fp32 table W [257, 2048] (256 router rows, then the shared
          expert's gate row): bf16 x widened to fp32, one fused multiply-add chain per (row, expert) over K in
          order (tl.dot at input_precision "ieee" runs on CUDA cores), so a logit never depends on the other
          rows, the tile or the launch settings; the row tile follows the row count (ROUTER_CFG, timed on an
          RTX 5090: 16 x 16 tiles up to 128 rows, 32 x 16 to 256, 64 x 16 beyond; retime on GB10's 48 SMs)
select    Flash Next's top-k: each row's top 8 by fp32 logit, largest first, the lower id on ties; weights
          exp(l_k - l_0) / sum over the 8 (the softmax over 256 renormalised over the 8), rounded to bf16;
          slot 8 the shared expert with weight bf16(sigmoid(bf16(gate logit))); then the plan groups the
          window's (row, slot) pairs by expert
experts   the shared grouped kernels over the 257-expert table at group 64, decode form:
          act = bf16(bf16(silu(bf16 gate)) * bf16 up), then the down projection in fp32 per (row, slot)
combine   branch = bf16(sum over the 9 slots, routed in pick order and then the shared expert, of fp32 w_k y_k),
          one rounding, fused with the residual add and the next RMSNorm (combine_add_rmsnorm)
```

Prompt chunks use the decode form too, so a prefilled row gets the bits of a serial step (the shared prefill form
rounds its outputs differently). A pair's bits never depend on the other rows, and the row count is a runtime
argument, so a new prompt length compiles nothing.

## The forward (`forward.py`, `state.py`)

The model is plain pre-norm: embed, then per layer

```text
normed = RMSNorm(h) (1 + w)            fp32, one bf16 rounding (fused into the previous layer's combine)
h      = h + mixer(normed)             Gated DeltaNet (30 layers) or gated attention (10 layers)
normed = RMSNorm(h) (1 + w_post)       fused with the residual add (and the out projection's K slices)
h      = h + MoE(normed)               256 routed experts (top 8) and the shared expert
normed = RMSNorm(h) (1 + w_next)       the next layer's input norm, or the final norm after the last layer
```

and the head over the requested rows. A window is a table of sequences (`Seq`: a state, its rows' offset and
count, its first position). The row-shared kernels (every matmul, the router, the experts, the norms) run once over
all rows; the kernels that read a sequence's committed state (the GDN chain, attention's cache write and reads)
run once per sequence on its rows. The decoders pass one sequence today; nothing in the forward assumes it.

The committed state (`State`, a slot of a `Pool`) is read-only during a forward, except for attention cache rows at
and past the committed length. `commit` keeps a sequence's first `keep` rows:

- Gated DeltaNet: the forward writes the state after the sequence's last row into the layer's other state buffer;
  a shorter keep replays the kept rows from the committed state into it (`gdn.replay`, the chain's update
  routine, the same bits). The buffer parity flips.
- Conv windows: rows [keep, keep + 3) of [old window; the rows' q | k | v].
- Attention: keys and values were written at their absolute positions; the committed length advances by `keep`.

A window of at most `window_rows` rows keeps what a partial keep needs (each GDN layer's projection rows and the
replay inputs). A longer window is a prefill chunk: it keeps every row, saves no replay inputs, and writes each
layer's next conv window during the forward (`Buffers.tail`), which bounds the scratch at 512-row chunks.
Attention runs a long window in row blocks of `attn_rows` (each row's keys are chunked by absolute position and
merged in order, so the blocking changes no bits); a prefill chunk's blocks launch only the key chunks their rows
read, while a window launches every chunk up to its context bound, so a captured graph stays valid as the sequence
grows. `State.snapshot` keeps what lies outside the cache rows (GDN states, conv windows, length, the head's
bookkeeping, about 64 MB); with the rows below the length in place, `restore` brings the sequence back.

## Weights and the layout contract (`checkpoint.py`, `weights.py`)

The loader passes MLX's packing through unchanged; the kernels regroup what they need when they wrap it.

- A projection is a `QW`: MLX's arrays as stored. `words` [..., N, K*bits/32] int32 (the checkpoint's uint32 bits),
  `scales` and `biases` bf16 [..., N, K/64]. Input k of row n is the `bits`-wide field `k % (32/bits)` of word
  `k // (32/bits)`, lowest bits first, and its value is scale * q + bias with group `k // 64`'s scale and bias.
- Projections that read the same input are stacked by rows: Gated DeltaNet [in_proj_qkv | in_proj_z | in_proj_b |
  in_proj_a] and attention [q_proj | k_proj | v_proj] (`Config.gdn_rows`, `Config.attn_rows`). in_proj_qkv's rows
  are q (16 key heads x 128), k (16 x 128), v (32 value heads x 128), the conv's channel order; q_proj's are per
  head, the query (256 rows) then its output gate (256 rows).
- The 256 routed experts and the shared expert are one table per layer: `gate` and `up` [257, 512, 2048/8 words],
  `down` [257, 2048, 512/8 words]; the shared expert is expert 256.
- The router is fp32 [257, 2048]: the 256 router rows (`mlp.gate`), then the shared expert's gate row
  (`mlp.shared_expert_gate`), 8-bit in the checkpoint (4-bit in the drafter), dequantized once at load as
  fp32(scale) * q + fp32(bias). The product is exact in fp32, so every device gets the same single rounding.
- Centred norms (input, post-attention, final, attention q/k, and the drafter's pre-fc norms) compute x̂ (1 + w);
  each is returned as that fp32 multiplier. MLX stores 1 + w rounded to bf16; a checkpoint that stores w gets 1
  added in fp32. The loader judges each group of norms against its own mean in the original checkpoint (`NORM_W`,
  `MTP_NORM_W`: a group is 1 + w when its mean lies nearer w + 1 than w) and refuses a checkpoint that mixes the
  two. The Gated DeltaNet output norm is not centred: its weight is used as stored (w silu(z)).
- Embedding and head: `QW` [248320, 2048/8 words] as stored (the model does not tie them).
- The MTP drafter is a separate repository with bare names (`fc`, `pre_fc_norm_embedding`, `pre_fc_norm_hidden`,
  `layers.0.*`, `norm`); `load_mtp` reads it into `MTPW`. It uses the target's embedding and head.

`checkpoint.py` holds the config, every tensor the loader reads with its dtype and shape (`layout`, `mtp_layout`,
checked against the safetensors headers before a byte is read) and the reader, which reads with large sequential
reads (mmap page faults stream slowly once the page cache is dropped) and drops the read shards' pages (on unified
memory they would sit beside the same bytes on the GPU).

## The MTP head (`mtp.py`)

Row t of the head reads the model's hidden state at position t (after the final norm, `Buffers.hidden`) and the
token at position t + 1, and scores the token at position t + 2:

```text
x = fc([RMSNorm(embed(t_{t+1})) (1 + w_e) | RMSNorm(h_t) (1 + w_h)])      one 2048 x 4096 matmul, bf16 out
x = x + attention(RMSNorm(x) (1 + w_in))    one gated attention layer over the head's own cache, the model's RoPE
x = x + MoE(RMSNorm(x) (1 + w_post))        256 routed experts (top 8) and the shared expert
h = RMSNorm(x) (1 + w_norm);  logits = draft head(h)
```

Row t sits at position t of the head's cache (`State.mtp_kc`), which holds the rows the head has absorbed
(`State.mtp_len`). A chained draft feeds the head's own output back as the next row's hidden state; its cache
entries sit past `mtp_len` and the next absorb overwrites them. Only the last row of a step needs its output, so a
step runs the attention, MoE and head on its last row alone (and a prompt absorb stops after the cache write). The
head proposes and never decides a token, so it needs no row invariance; every kernel is deterministic, so drafts,
and therefore speeds, repeat. The draft head is the model's head rows at the 76,882 ids of `draft_vocab.txt` (a
full head is 248,320 x 2,048 4-bit weights, about 290 MB a draft); a token outside the list can never be a draft,
which costs speed, never correctness. The list's provenance is in the recipe.

## Decoding (`decode.py`, `graphs.py`)

Every emitted token is the keyed sample (`tensorfold.cuda.sampling.sample_rows`: seeded Gumbel over top-k/top-p,
ties by token id) of this engine's logits at its position, or the argmax (the lowest id among equal logits) when
greedy. A serial step is a one-row window of `forward` followed by a one-row `commit`.

A drafted round verifies the pending token and up to `depth` MTP drafts as one window through the same kernels and
the same sampler, keeps the rows up to the first draft that differs from what the sampler picks there, then the
head absorbs the kept rows and chains the next drafts. Drafts are sampled with the same keyed rule at their
positions, so a sampled draft shares its position's noise with the verify. With `confidence` > 0 a chain always
keeps its first draft and ends before a later draft the head gives less than `confidence`. A round never drafts
past the reply's length, so the caches end as serial decoding leaves them; from a state the head lags behind (a
serially decoded one), the first round runs without drafts while the head catches up.

A prompt runs in chunks of up to 512 rows; with the MTP head, the head absorbs each position whose next token is
known and keeps the last position's hidden row (`State.mtp_tail`) until the first reply token. Rows never depend
on their chunk, so any chunking, and a resumed prompt, ends in the state and logits of one fresh prefill.

A forward's GPU work reads only static buffers, device-side positions and its sequence's state, so `graphs.py`
captures it once and replays it with new inputs staged beforehand. A capture is keyed by the window's rows, the
sequence's pool slot and GDN buffer parity (a layer reads its state from one of two buffers and every commit flips
them), and the context bucket, as Flash Next keys it: attention launches the key chunks below max(8192, the next
power of two at or above the window's last key + 1), capped at the capacity. Chunks past a row's keys write
nothing and the merge never reads them, so every bucket gives the same bits. The head's steps have one graph per
(rows, with or without logits, slot, bucket). Larger windows (prefill chunks) run eagerly.

## The engine (`engine.py`)

One request decodes at a time (`--parallel` above one is accepted, and requests take turns), MTP-drafted when the
drafter is present: a round verifies the pending token and up to 6 chained drafts (at most 15), and a chain stops
before a later draft the head gives less than 50%. Windows of up to depth + 1 rows and the head's steps replay CUDA
graphs. Without the drafter, or with `--no-drafts`, every round decodes one token.

Capacity is fixed at start-up by `tensorfold.cuda.capacity.admit` before anything is loaded: the checkpoint's and
the drafter's tensors in the kernels' layout (the router tables widened to fp32, the vision tower skipped), plus,
at a cache capacity, two sequence states, the kept snapshots, the window buffers, the head's step buffers and draft
head, and a workspace (`cache_bytes`), against the memory the device has (on GB10, MemAvailable, which counts the
page cache the kernel gives back). The drafter loads after the model has been regrouped, so its transient stays
below the model's own. An explicit `--context` that cannot fit is refused with the largest that does; the default
(32,768 tokens) shrinks to fit. `context_window` (the cache slots less depth + 1 speculative positions) is what the
server checks requests against before streaming.

Prefix reuse: the engine keeps the state after the last request's prompt and after its reply (`State.snapshot`,
with their cache rows in place), and a prompt that extends either by at least one token resumes from it. An exactly
repeated prompt prefills afresh: the kept state holds no logits for its last position. A request with
`"draft": false` decodes one token a round from a fresh prefill in a second state and leaves the kept states alone.

## The fp32 reference (`reference.py`)

A plain PyTorch forward following transformers' `modeling_qwen3_5_moe.py`, and vLLM's `qwen3_5_mtp.py` for the MTP
head: q_proj gives [query | gate] per head; RoPE rotates the first 64 of each head's 256 dims (text positions reduce
mrope to 1-D RoPE); the DeltaNet decay is g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b), and value head h
reads key head h // (nv / nk); the MoE takes the fp32 softmax over 256 experts, the top 8 (ties to the lower id)
renormalised, and adds the shared expert times sigmoid(its gate row . x). Arithmetic is fp32 (run with
`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`); with `State(bf16=True)` activations are rounded to bf16 where the kernels
store them. It dequantizes the weights it uses a layer at a time, so the model fits a 32 GB GPU beside them.

## Tests and tools

`tests/cuda/test_qwen36moe_*.py` need a GPU: row invariance of every kernel, windows against serial steps, commit
and resume against serial state, the forward and the MTP head against the fp32 reference, drafted against serial
decoding, graphs against eager, and the engine behind the server. The top-level `test_cuda_qwen36moe_*.py` files
need none: the package's refusals (`package`), the loader on fake checkpoints (`load`), the memory admission
(`admission`, with torch and triton) and the reference on tiny weights (`reference`). `tools/hash_qwen36.py` and `tools/hash_flashnext.py` hash
outputs for before-and-after comparisons; `tools/bench_q36.py` measures. Measured numbers are in
[the recipe](../../../../../docs/recipes/qwen3.6-35b-a3b.md).
