"""Qwen3.8 Flash Next (qwen4_exp) on CUDA: row-invariant kernels, the forward, MTP drafting, two-GPU tensor parallel.

The contract of the Apple Silicon engine: every kernel on the verify path gives a row the same bits whether it
runs alone (serial decoding) or as one row of a verify window, so a drafted token is the token serial decoding
would have sampled on this machine. Serial is defined by these kernels; their bits differ from the Mac's.
Weights: the MLX 4-bit checkpoint as shipped (affine, groups of 32), regrouped at load.

``engine.FlashNextEngine`` is what ``tensorfold serve`` runs (the family's ``cuda_engine``).
"""

# the recipe measured on DGX Spark (docs/recipes/qwen3.8-flash-next.md)
DEPTH = 6            # most MTP drafts a round
CONFIDENCE = 0.3     # a chain ends before a draft the MTP head gives less than this
CONTEXT = 8192       # prompt plus reply tokens the caches hold
