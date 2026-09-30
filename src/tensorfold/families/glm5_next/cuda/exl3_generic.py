"""GLM routed experts through the generic mixed-width EXL3 path.

Serves any bit width (k2 2..16) with bit-identical epilogue arithmetic
(ACT_BF16, GLM tile configs). Uses the exl3_mm.Scratch's rows/slots (passed
explicitly) to size its own generic Scratch once per key.

The shared expert (pick == E) is excluded by the generic kernels, as in
exl3_mm; the caller combines its slot from the BF16 shared MLP.
"""
from __future__ import annotations

import torch

from tensorfold.cuda.exl3 import experts as generic

_scratch: dict = {}


def routed(x, pick, ex, rows, limit):
    """ey[pair] fp32 for routed pairs; the shared slot's row is stale (caller overwrites)."""
    key = (rows, pick.shape[1], ex.count)
    s = _scratch.get(key)
    if s is None:
        s = generic.Scratch(ex, rows, pick.shape[1], device=str(x.device))
        _scratch[key] = s
    return generic.routed(x, pick, None, ex, s, None, rows, limit, act_mode=generic.ACT_BF16)
