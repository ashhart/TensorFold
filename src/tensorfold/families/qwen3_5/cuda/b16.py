"""A plain fp16/bf16 linear (``b16.cu``) for what an EXL3 pack leaves unquantized: one warp an output, fixed order, row-invariant."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_qwen_b16_v1", sources=[str(here / "b16.cpp"), str(here / "b16.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def matmul(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """x [M, K] @ w [N, K]^T (+ bias [N]), x cast to the weight's dtype when they differ."""

    if x.dtype != w.dtype:
        x = x.to(w.dtype)
    b = bias if bias is not None and bias.numel() else torch.empty(0, dtype=w.dtype, device=w.device)
    return _ext().b16_linear(x.contiguous(), w.contiguous(), b)
