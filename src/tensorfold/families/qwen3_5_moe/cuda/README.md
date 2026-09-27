# Qwen3.6-35B-A3B on CUDA

The CUDA engine for Qwen3.6-35B-A3B (`model_type` `qwen3_5_moe`) on one DGX Spark (GB10). It reads the MLX 4-bit
checkpoint (`mlx-community/Qwen3.6-35B-A3B-4bit`: affine 4-bit, groups of 64, the router and shared-expert gate in
8 bits) as stored, skips the vision tower, and drafts with the MTP drafter split out of the original checkpoint
(`mlx-community/Qwen3.6-35B-A3B-MTP-4bit`, `model_type` `qwen3_5_mtp`).

Every kernel on the verify path gives a row the same bits whether it runs alone or as one row of a window. Serial
decoding runs through the same kernels, so a drafted token is the token serial decoding produces on this machine.

## Kernels

Several are Flash Next's (`families/qwen4_exp/cuda/`), shared and run at Qwen3.6's shapes: its GDN heads (16, 32)
with the SiLU gate, dense attention (`sparse=False`) and MLX's groups of 64.

| File | Kernel | What it computes | Why the bits do not depend on the row count |
| --- | --- | --- | --- |
| `qmm.py` (Flash Next's `qwen4_exp/cuda/qmm.py` at `gs=64`) | `_qmm`, `_reduce` | the dense 4-bit matmuls (the stacked GDN and attention projections, GDN `out_proj`, `o_proj`, the head) on words regrouped once at load: per 64-input group a tensor-core dot of the bf16 rows and the integer-valued weights, then `acc + p * scale + xs * bias`, groups in order | a row's sums never depend on its tile or the other rows; the K split is frozen per weight shape (`SHAPES`), and its slices are added in slice order; launch settings change the schedule, never the sums |
| | `_group_sums` | fp32 sums of each 64 bf16 inputs (the bias term of the next matmul) | one row at a time |
| `qmm.py` | `_embed` | an embedding row gathered from MLX's layout and dequantized | one program per row and group |
| | `_moe_gateup`, `_moe_down` (Flash Next's) | the 256 routed experts and the shared expert as one table of 257: one program per distinct expert and column tile; `bf16(silu(bf16(gate)) * bf16(up))` with its 32-sums, then the down projection to fp32 `y[row, slot]` | a (row, expert) pair gets the same arithmetic whatever rows share the expert; the K order is fixed (the narrower tiles chosen for 1-2 member windows change the schedule only) |
| `moe.py` | `_router` | fp32 router logits `[R, 257]` from the loader's fp32 table (256 router rows, then the shared expert's gate row) | an IEEE dot on CUDA cores, one fused multiply-add chain per (row, expert) over K in order |
| | `_topk_rows`, `_group` (Flash Next's) | each row's top 8 by fp32 logit (the lower id on ties), weights `exp(l_k - l_0)` renormalised over the 8 and rounded to bf16, the shared gate `bf16(sigmoid(bf16(logit)))` in slot 8; then the window's distinct experts in increasing id order with the (row, slot) pairs that picked each, in row order | a row's picks read its own logits; the grouping is one program in a fixed order |
| | `_combine` | `bf16(sum of w_k * y_k)` over the 8 routed slots in pick order, then the shared expert, one rounding; fused with the residual add, the next RMSNorm (`1 + w`, fp32) and its 64-sums | one row at a time, slots in a fixed order |
| `forward.py` | `_add_norm` | residual add (a bf16 branch, or a matmul's split-K slices added in slice order and rounded once, the bits of `_reduce`), RMSNorm with the fp32 `1 + w`, and the 64-sums | one row at a time |
| | `_conv_tail` | a GDN layer's next conv window (the last 3 of the old window and a prefill chunk's q, k, v rows) | copies |
| Flash Next's `forward.py` | `_shift_windows` | the conv windows after keeping the first rows of a window | copies |
| Flash Next's `gdn.cu` (`gdn.py`) | `chain_kernel` (`NK=16, NV=32`, SiLU) | a window's rows in order from the committed state in one kernel: depthwise conv and SiLU, q/k L2 norms, `g = -exp(A_log) softplus(a + dt_bias)` and `beta = sigmoid(b)`, the delta rule in fp32, the gated RMSNorm `w * norm(y) * silu(z)`; writes the state after the last row and what a replay needs | each row updates the state its predecessors left, in row order, the same whether the window has one row or 32 |
| | `replay_kernel` | the commit of a shorter prefix: the kept rows replayed from the committed state | the chain's update routine, compiled with `--fmad=false`, so a kept prefix has the bits of serial steps |
| Flash Next's `glue.py` | `_attn_prep` | q and k RMSNorm (`1 + w`, fp32), rotate-half RoPE on the first 64 of 256 dims at the row's absolute position (fp32, one rounding), keys and values written to the cache at their positions; no indexer (`index_heads=0`) | one program per row and head |
| | `_attn_gate` | `bf16(o * sigmoid(gate))` with the gate from the second half of each head's q_proj rows, and the 64-sums for `o_proj` | one program per row and head |
| Flash Next's `attention.py` | `_chunks`, `_tile`, `_merge` | full attention of 16 query heads over 2 KV heads of 256: keys in fixed 512-key chunks by absolute position, eight 64-key tensor-core tiles each, the chunks merged in position order; long windows run in blocks of `attn_rows` rows | a row's chunks hold the same keys alone or in a window; `sparse=False` keeps Flash Next's sparse attention off at any context |
| `decode.py` | `sample_rows` | the argmax (lowest id on ties), or the top-k plus a margin on the GPU and the keyed draw of `engine/exact_sampling.py` on the host | each row from its own logits and absolute position |

No cuBLAS, `torch.matmul` or `F.linear` on the verify path: their algorithms depend on the row count.

## The rest of the package

- `weights.py` reads the checkpoint and the drafter and states the layout the kernels take: MLX's packing
  unchanged (uint32 words `[N, K/8]`, bf16 scales and biases `[N, K/64]`), projections that share an input stacked
  by rows, the 256 routed experts and the shared expert as one table of 257 (the shared expert is expert 256), the
  router and shared-expert gate dequantized once to fp32 `[257, 2048]`, and every centred norm as its fp32
  multiplier `1 + w` (MLX stores `1 + w`, rounded to bf16; checked at load).
- `forward.py` regroups the loaded weights for the kernels (`prepare`), runs a window of rows per sequence
  (`forward`, `forward_many`) and keeps a prefix (`commit`); `State` is one sequence's committed caches, from a
  `Pool`, with `snapshot` and `restore`.
- `decode.py` has `prefill` (512-row chunks), `serial_decode` and `score` (teacher-forced NLL).
- `engine.py` is what `tensorfold serve` runs: the memory check before loading, prefix reuse from the last prompt
  and reply, and the serial switch (`"draft": false`).
- `reference.py` is a plain fp32 PyTorch forward (with optional bf16 roundings) for the quality tests.

The tests are in `tests/cuda/test_qwen36moe_*.py`: row invariance of every kernel, windows against serial
steps, commit and resume against serial state, the forward against the fp32 reference, and the engine behind
the server. The package and loader tests need no GPU. Measured numbers: [the recipe](../../../../../docs/recipes/qwen3.6-35b-a3b.md).
