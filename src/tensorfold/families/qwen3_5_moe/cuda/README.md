# Qwen3.6-35B-A3B on CUDA

The CUDA engine for Qwen3.6-35B-A3B (`model_type` `qwen3_5_moe`) on one DGX Spark (GB10). It reads the MLX 4-bit
checkpoint (`mlx-community/Qwen3.6-35B-A3B-4bit`: affine 4-bit, groups of 64, the router and shared-expert gate in
8 bits) as stored, skips the vision tower, and drafts with the MTP drafter split out of the original checkpoint
(`mlx-community/Qwen3.6-35B-A3B-MTP-4bit`, `model_type` `qwen3_5_mtp`).

Every kernel on the verify path gives a row the same bits whether it runs alone or as one row of a window. Serial
decoding runs through the same kernels, so a drafted token is the token serial decoding produces on this machine.

## Kernels

To be listed as they land, with what each computes and why its bits do not depend on the row count (see
`families/qwen3_5/cuda/README.md` for the format).

## The rest of the package

- `weights.py` reads the checkpoint and the drafter and states the layout the kernels take: MLX's packing
  unchanged (uint32 words `[N, K/8]`, bf16 scales and biases `[N, K/64]`), projections that share an input stacked
  by rows, the 256 routed experts and the shared expert as one table of 257 (the shared expert is expert 256), the
  router and shared-expert gate dequantized once to fp32 `[257, 2048]`, and every centred norm as its fp32
  multiplier `1 + w` (MLX stores `1 + w`, rounded to bf16; checked at load).

The tests are in `tests/cuda/test_qwen36moe_*.py`. The package and loader tests need no GPU.
