"""GLM-5.3-Flash's KDA chain on CUDA (``kda.cu``): one fused kernel per layer and window.

``chain`` runs R consecutive rows from the committed state: conv + SiLU, q/k L2 norms, per-channel decay,
beta, the delta rule and the gated RMSNorm, writing the rows' outputs, the state after the last row, and
what a replay needs (normalized k, v, the decay and beta per row). ``replay`` rebuilds the state after the
first ``keep`` rows of a window with the same update routine, so keeping a prefix of a window gives the
bits of serial steps.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

DK = DV = 128


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    here = Path(__file__).parent
    return load(name="tensorfold_glm_kda_v1", sources=[str(here / "kda.cpp"), str(here / "kda.cu")],
                extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False)


class KDAScratch:
    """Static per-window buffers (outputs and replay inputs) for up to ``rows`` rows of ``heads`` heads. With
    ``parent``/``index``: views into one allocation for all layers (so a commit replays every layer at once)."""

    def __init__(self, rows: int, heads: int, device, parent: "KDAScratchSet | None" = None, index: int = 0) -> None:
        if parent is None:
            self.out = torch.empty((rows, heads * DV), dtype=torch.bfloat16, device=device)
            self.k = torch.empty((rows, heads, DK), dtype=torch.float32, device=device)
            self.v = torch.empty((rows, heads, DV), dtype=torch.bfloat16, device=device)
            self.g = torch.empty((rows, heads, DK), dtype=torch.float32, device=device)
            self.b = torch.empty((rows, heads), dtype=torch.float32, device=device)
        else:
            self.out = parent.out[index]
            self.k, self.v, self.g, self.b = parent.k[index], parent.v[index], parent.g[index], parent.b[index]


class KDAScratchSet:
    """KDAScratch for ``layers`` layers in one allocation each."""

    def __init__(self, layers: int, rows: int, heads: int, device) -> None:
        self.layers, self.rows, self.heads = layers, rows, heads
        self.out = torch.empty((layers, rows, heads * DV), dtype=torch.bfloat16, device=device)
        self.k = torch.empty((layers, rows, heads, DK), dtype=torch.float32, device=device)
        self.v = torch.empty((layers, rows, heads, DV), dtype=torch.bfloat16, device=device)
        self.g = torch.empty((layers, rows, heads, DK), dtype=torch.float32, device=device)
        self.b = torch.empty((layers, rows, heads), dtype=torch.float32, device=device)
        self.views = [KDAScratch(rows, heads, device, self, i) for i in range(layers)]


def replay_layers(state_in: torch.Tensor, scratch: KDAScratchSet, rows: int, state_out: torch.Tensor) -> None:
    """Every layer's state after the first ``rows`` rows: state_in/state_out [layers, H, 128, 128]."""

    L, H = scratch.layers, scratch.heads
    _ext().replay_layers(state_in, H * DV * DK, scratch.k, scratch.v, scratch.g, scratch.b, scratch.rows * H * DK,
                         scratch.rows * H, L, H, int(rows), state_out)


def chain(p: torch.Tensor, b_off: int, a: torch.Tensor, g: torch.Tensor, conv_state: torch.Tensor,
          conv_w: torch.Tensor, state_in: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
          norm_w: torch.Tensor, eps: float, lower: float, rows: int, scratch: KDAScratch,
          state_out: torch.Tensor) -> torch.Tensor:
    """p: projection rows [q | k | v | ... | b at b_off ...] (row stride p.stride(0)); a, g: the forget-gate and
    output-gate rows (bf16, row strides their own). Returns scratch.out[:rows]."""

    _ext().chain(p, p.stride(0), int(b_off), a, a.stride(0), g, g.stride(0), conv_state, conv_w, state_in, a_log,
                 dt_bias, norm_w, float(eps), float(lower), int(rows), scratch.out, state_out, scratch.k, scratch.v,
                 scratch.g, scratch.b)
    return scratch.out[:rows]


def replay(state_in: torch.Tensor, scratch: KDAScratch, rows: int, state_out: torch.Tensor) -> None:
    _ext().replay(state_in, scratch.k, scratch.v, scratch.g, scratch.b, int(rows), state_out)
