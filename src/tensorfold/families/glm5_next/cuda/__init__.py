"""GLM-5.3-Flash on CUDA, tensor parallel over two GPUs: row-invariant kernels, forward, MTP and DFlash2 drafting.

Same contract as the other CUDA engines: every kernel on the verify path gives a row the same bits whether it runs
alone (serial decoding) or as one row of a verify window, and every cross-rank reduction is an all-gather of fp32
partials summed in rank order, so a drafted token is the token serial decoding samples on the same two ranks.
Weights: the MLX 4-bit checkpoint (affine, groups of 64), each rank reading its share (``split.py``) and
regrouping it once at load (``qmm.py``).
"""
