"""Fused KDA chains and prefix replay use the same state update routine so a kept prefix preserves serial bits."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

DK = DV = 128


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_glm_kda_v1", sources=[str(here / "kda.cpp"), str(here / "kda.cu")],
                extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False)


class KDAScratch:
    """Static window outputs and replay inputs, with optional views into shared storage so all layers can replay together."""

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
    """Run projection rows p [q | k | v | ... | b at b_off ...] and bf16 gate rows a and g using their own strides; return scratch.out[:rows]."""

    _ext().chain(p, p.stride(0), int(b_off), a, a.stride(0), g, g.stride(0), conv_state, conv_w, state_in, a_log,
                 dt_bias, norm_w, float(eps), float(lower), int(rows), scratch.out, state_out, scratch.k, scratch.v,
                 scratch.g, scratch.b)
    return scratch.out[:rows]


def replay(state_in: torch.Tensor, scratch: KDAScratch, rows: int, state_out: torch.Tensor) -> None:
    _ext().replay(state_in, scratch.k, scratch.v, scratch.g, scratch.b, int(rows), state_out)
