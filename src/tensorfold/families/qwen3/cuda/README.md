# Qwen3 embeddings on CUDA

The CUDA engine for Qwen3 dense decoders served as last-token embedding models (model type `qwen3`, tested with
`Qwen/Qwen3-Embedding-8B`). It runs the prompt forward only: no decoding, and no key/value cache outlives a step.
See [the recipe](../../../../../docs/recipes/qwen3-embedding.md) for serving, the endpoint and measurements.

The server packs waiting texts from any requests back to back into one forward step. Every kernel on the path gives
a text's rows the bits they get when the text runs alone, so a vector never depends on the batch, the other texts'
lengths or its position in the step.

## Kernels

| File | Kernel | What it computes | Why a text's bits do not depend on the batch |
| --- | --- | --- | --- |
| `tensorfold/cuda/kernels/dense.py` | `_matmul` | a bf16 projection `x @ W.T`: bf16 tensor-core products, one fp32 chain over K in k16 steps | each output adds its K in the same order at any row count; the four block shapes change only which block computes it |
| `tensorfold/cuda/kernels/qmm_prefill.cu` (shared with the 27B) | `prefill_kernel` | a 4-bit projection: each weight `bf16(fma(q, s, b))`, then the same fp32 chain | the same, per group of 64 inputs in order |
| `tensorfold/cuda/kernels/prefill_attention.py` | `_attend_texts` (beside the 27B's `_attend`, sharing `_tile`) | causal attention within each text of a packed step, 64-key tiles | a block's rows read only their text's keys, tiled from the text's own start, exactly as `attention` tiles that text alone |
| `glue.py` | `_add_rmsnorm` | adds a projection's fp32 output into the fp32 residual, then RMSNorm to bf16 rows | one program a row |
| | `_qkv` | q/k RMSNorm and rotate-half rotary from a cos/sin table (Hugging Face's fp32 angles), v copied out | one program a row and head |
| | `_swiglu`, `_pool` | silu(gate) * up; the last layer's sum and the final RMSNorm in fp32 | one program a row |
| `tensorfold/families/qwen3_5/cuda/glue.py` (shared) | `embedding` | token rows from a bf16, 8-bit or 4-bit table | one row a token |

The residual stream stays in fp32: the output and down projections write fp32 and add into it unrounded, and only
the projections' inputs are rounded to bf16. The last layer runs its output projection and MLP for the pooled rows
only. The server L2-normalizes each pooled row on the host in float64, after cutting it to the requested
`dimensions`.

## The rest of the package

- `weights.py` loads bf16 as shipped or MLX affine 4-bit groups of 64 (packed once for `qmm_prefill`), stacking
  `[q | k | v]` and `[gate | up]`; stacking never changes a row's bits.
- `forward.py` packs a step's texts, runs the layers and pools each text's last token.
- `engine.py` is what `tensorfold serve` runs: startup admission on the checkpoint's headers before any weight
  loads, then `embed(texts)` for `tensorfold/cuda/embeddings.py`'s queue.
- `../convert.py` and `../gptq.py` write the 4-bit checkpoint.

The tests in `tests/cuda/test_qwen3_embed.py` cover the kernels' row invariance, packed texts against texts alone,
a random model against an fp32 forward, the 4-bit path against its dequantized bf16 forward and, with the
checkpoint, batch invariance, truncation and agreement with Hugging Face's fp32 model.
