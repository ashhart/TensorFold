"""Flash Next's DeltaNet ``front`` and ``back`` for many streams' rows, bit-equal to gdn.cu's chain kernel."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from .gdn import DK, DV


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_qwen4_exp_gdn_io", sources=[str(here / "gdn_io.cpp"), str(here / "gdn_io.cu")],
                extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False)


def front(p: torch.Tensor, conv_ptrs: torch.Tensor, sid: torch.Tensor, windows: torch.Tensor, conv_w: torch.Tensor,
          a_log: torch.Tensor, dt_bias: torch.Tensor, nk: int, out: tuple | None = None) -> tuple[torch.Tensor, ...]:
    """``windows`` taps: < 3 a row of stream ``sid``'s conv state, else projection row tap - 3; fills ``out``."""

    if out is None:
        rows, nv = windows.shape[0], a_log.numel()
        dev = p.device
        q = torch.empty((rows, nk, DK), dtype=torch.float32, device=dev)
        v = torch.empty((rows, nv, DV), dtype=torch.bfloat16, device=dev)
        g = torch.empty((rows, nv), dtype=torch.float32, device=dev)
        out = (q, torch.empty_like(q), v, g, torch.empty_like(g))
    _ext().front(p, conv_ptrs, sid, windows, conv_w, a_log, dt_bias, *out)
    return out


def back(y: torch.Tensor, p: torch.Tensor, norm_w: torch.Tensor, eps: float, out: torch.Tensor,
         xs: torch.Tensor) -> None:
    """Gated RMSNorm of the read-out ``y`` (z from ``p``) into bf16 ``out`` and its 32-channel group sums ``xs``."""

    _ext().back(y, p, norm_w, float(eps), out, xs)
