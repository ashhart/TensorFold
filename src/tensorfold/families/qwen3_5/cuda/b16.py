"""A plain fp16/bf16 weight linear on CUDA (``b16.cu``), row-invariant on the verify path.

The engines read most weights quantized; the few tensors an EXL3 checkpoint leaves as they are (in_proj_a and
in_proj_b of the GDN layers) come through here. One warp owns one output element and sums k in a fixed order
in fp32 (each lane its own strided run, then a fixed butterfly), so an element's bits depend on its own row and
column alone - the contract's row invariance by construction rather than by tuning, and no cuBLAS on the verify
path.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    here = Path(__file__).parent
    return load(name="tensorfold_qwen_b16_v1", sources=[str(here / "b16.cpp"), str(here / "b16.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def matmul(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """x [M, K] @ w [N, K]^T (+ bias [N]), x cast to the weight's dtype when they differ."""

    if x.dtype != w.dtype:
        x = x.to(w.dtype)
    b = bias if bias is not None and bias.numel() else torch.empty(0, dtype=w.dtype, device=w.device)
    return _ext().b16_linear(x.contiguous(), w.contiguous(), b)
