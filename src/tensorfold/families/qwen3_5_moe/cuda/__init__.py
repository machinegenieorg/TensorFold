"""Qwen3.6-35B-A3B (qwen3_5_moe) on CUDA: row-invariant kernels, the forward and MTP drafting, on one GPU.

Same contract as the other CUDA engines: every kernel on the verify path gives a row the same bits whether it runs
alone (serial decoding) or as one row of a verify window, so a drafted token is the token serial decoding would
have sampled on this machine. Weights: the MLX 4-bit checkpoint as shipped (affine, groups of 64), passed to the
kernels in MLX's packing (``weights.py`` states the layout); drafts from the separate MTP drafter repository.
"""
