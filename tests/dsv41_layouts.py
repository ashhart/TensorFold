"""Test-side builders of the layouts DeepSeek-V4.1's loader makes: an FP8 block linear and a trellis's rank columns."""

from __future__ import annotations

import torch

from tensorfold.cuda.nvfp4.linear import Mx8Linear
from tensorfold.families.deepseek_v41.cuda import split
from tensorfold.families.deepseek_v41.cuda.convert import fp8_block_rows

TRELLIS = "layers.0.ffn.experts.0.w1.trellis"     # a gate/up trellis: the loader cuts its tile columns


def mx8_from_block(weight: torch.Tensor, scale: torch.Tensor) -> Mx8Linear:
    """e4m3 [N, K] with 32x32 E8M0 block scales -> ``Mx8Linear``, as the loader builds a single projection."""

    return Mx8Linear.from_checkpoint(weight, fp8_block_rows(scale, int(weight.shape[0])))


def exl3_dim1_half(trellis: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    """EXL3 trellis [K/16, N/16, 32] -> the rank's tile columns as the loader cuts them, a contiguous copy."""

    return trellis[split.slice_for(rank, world, TRELLIS, tuple(trellis.shape))].clone(
        memory_format=torch.contiguous_format)
