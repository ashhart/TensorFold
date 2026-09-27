"""TensorFold: fast, exact LLM decoding on Apple Silicon and NVIDIA GPUs behind an OpenAI-compatible endpoint."""

import os

# MLX runs float32 matmuls as TF32 on M5-generation GPUs unless told otherwise (median relative error 4e-4 against
# 3e-8 in fp32). The row kernels repeat MLX's fp32 arithmetic, so prefill (MLX's matmuls) and decode (the kernels)
# only agree bit for bit in fp32. Set before mlx is imported; an explicit MLX_ENABLE_TF32 in the environment wins.
# (kingjamez, M5 Ultra, 2026-09-27: 0 of 16,384 hc_expand values differ with it, ~11% in the last bf16 bit without.)
os.environ.setdefault("MLX_ENABLE_TF32", "0")

__version__ = "0.3.4.1"
