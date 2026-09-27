"""Run a row-exact DeltaNet chain and retain normalized k, v, g and beta so replaying any accepted prefix reconstructs serial state with the same update routine."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

NK, NV, DK, DV = 16, 48, 128, 128
CONV = 2 * NK * DK + NV * DV
PW = CONV + NV * DV + 2 * NV


def widths(nk: int, nv: int) -> tuple[int, int]:
    """(conv channels, projection row width) for nk key heads and nv value heads."""

    conv = 2 * nk * DK + nv * DV
    return conv, conv + nv * DV + 2 * nv


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    here = Path(__file__).parent
    return load(name="tensorfold_qwen4_exp_gdn", sources=[str(here / "gdn.cpp"), str(here / "gdn.cu")],
                extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False)


class GDNScratch:
    """A sequence's replay inputs for its last window of up to ``rows`` rows."""

    def __init__(self, rows: int, device, nk: int = NK, nv: int = NV) -> None:
        self.k = torch.empty((rows, nk, DK), dtype=torch.float32, device=device)
        self.v = torch.empty((rows, nv, DV), dtype=torch.bfloat16, device=device)
        self.g = torch.empty((rows, nv), dtype=torch.float32, device=device)
        self.b = torch.empty((rows, nv), dtype=torch.float32, device=device)


def chain(p: torch.Tensor, conv_state: torch.Tensor, conv_w: torch.Tensor, state_in: torch.Tensor,
          a_log: torch.Tensor, dt_bias: torch.Tensor, norm_w: torch.Tensor, eps: float, rows: int,
          scratch: GDNScratch, state_out: torch.Tensor, out: torch.Tensor, xs: torch.Tensor) -> None:
    """``out`` [rows, NV*DV] bf16 and ``xs`` (its 32-group sums) may be row views of a larger buffer."""

    _ext().chain(p, conv_state, conv_w, state_in, a_log, dt_bias, norm_w, float(eps), int(rows), out, xs,
                 state_out, scratch.k, scratch.v, scratch.g, scratch.b)


def replay(state_in: torch.Tensor, scratch: GDNScratch, rows: int, state_out: torch.Tensor) -> None:
    _ext().replay(state_in, scratch.k, scratch.v, scratch.g, scratch.b, int(rows), state_out)
