"""Grouped block-FP8 experts on ``tensorfold.cuda.experts``' plan: a pair's bits never depend on the others."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from tensorfold.cuda import experts as grouped

COLS = 32                 # output columns a warp
BLOCK = 128               # the checkpoint's scale block, both ways
PROMPT_TILE = 64          # pairs a prompt item holds: four warps of 16 share each staged block


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import MIN_CAPABILITY, load

    here = Path(__file__).parent
    return load(name="tensorfold_fp8_experts_v6", sources=[str(here / "experts.cpp"), str(here / "experts.cu")],
                need=MIN_CAPABILITY, extra_cuda_cflags=["-O3"], verbose=False)


def pack(weight: torch.Tensor) -> torch.Tensor:
    """e4m3 [E, N, K] -> int32 [E, N/32, K/32, 32, 2, 4]: lane (gq, t)'s half h, tile j: inputs 16h + 4t + 0..3."""

    e, n, k = weight.shape
    if n % BLOCK or k % BLOCK:
        raise ValueError(f"FP8 experts [{e}, {n}, {k}]: N and K must be multiples of {BLOCK}")
    words = weight.contiguous().view(torch.int32).view(e, n // COLS, 4, 8, k // 32, 2, 4)   # [E, cb, j, gq, g, h, t]
    return words.permute(0, 1, 4, 3, 6, 5, 2).reshape(e, n // COLS, k // 32, 32, 2, 4).contiguous()


@dataclass
class Experts8:
    """One layer's experts (the shared one last): gate and up (SwiGLU) and down, each with fp32 128 x 128 scales."""

    up: torch.Tensor          # [E, NI/32, D/32, 2, 32, 2, 4] int32
    down: torch.Tensor        # [E, D/32, NI/32, 1, 32, 2, 4]
    up_scale: torch.Tensor    # [E, 2, NI/128, D/128] fp32 (gate, up)
    down_scale: torch.Tensor  # [E, 1, D/128, NI/128]
    width: int                # NI
    dims: int                 # D
    limit: float = 0.0

    @property
    def count(self) -> int:
        return int(self.up.shape[0])

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.up, self.down, self.up_scale, self.down_scale))


def make(gate: tuple, up: tuple, down: tuple, *, limit: float = 0.0) -> Experts8:
    """Each of gate, up, down: (e4m3 weights [E, N, K], fp32 scales [E, N/128, K/128]), as the checkpoint has them."""

    width, dims = int(gate[0].shape[1]), int(gate[0].shape[2])
    for name, (w, s) in (("gate", gate), ("up", up), ("down", down)):
        n, k = w.shape[1:]
        if tuple(s.shape) != (w.shape[0], n // BLOCK, k // BLOCK):
            raise ValueError(f"{name}: scales {tuple(s.shape)} do not tile [{n}, {k}] in {BLOCK} x {BLOCK} blocks")
    u = torch.stack([pack(gate[0]), pack(up[0])], dim=3)
    d = pack(down[0]).unsqueeze(3)
    us = torch.stack([gate[1], up[1]], dim=1).to(torch.float32).contiguous()
    ds = down[1].to(torch.float32).unsqueeze(1).contiguous()
    return Experts8(u, d, us, ds, width, dims, float(limit))


def _run(epi: int, x: torch.Tensor, slots: int, w: torch.Tensor, scale: torch.Tensor, kg: int, nb: int,
         plan: grouped.Plan, out: torch.Tensor, n: int, limit: float, skip: int, rows: int) -> None:
    units = grouped.max_items(rows * plan.slots, plan.experts, plan.tile) * nb
    if plan.prefill and plan.tile == PROMPT_TILE:      # an item a CTA, its blocks staged once
        _ext().prompt(epi, x, x.stride(0), slots, w, scale, kg, nb, plan.items, plan.counts, plan.members, out, n,
                      limit, skip, units)
        return
    _ext().experts(epi, x, x.stride(0), slots, w, scale, kg, nb, plan.items, plan.counts, plan.members, out, n,
                   limit, skip, units)


def gate_up(x: torch.Tensor, ex: Experts8, plan: grouped.Plan, out: torch.Tensor, rows: int, skip: int = -1) -> None:
    """x [R, D] bf16 -> out [R * slots, NI] bf16, each routed pair's SwiGLU; pairs of expert ``skip`` untouched."""

    _run(2, x, plan.slots, ex.up, ex.up_scale, ex.dims // 32, ex.width // COLS, plan, out, ex.width, ex.limit, skip,
         rows)


def down(act: torch.Tensor, ex: Experts8, plan: grouped.Plan, out: torch.Tensor, rows: int, skip: int = -1) -> None:
    """act [R * slots, NI] bf16 -> out [R * slots, D] in ``out``'s dtype; pairs of expert ``skip`` untouched."""

    _run(0 if out.dtype == torch.float32 else 3, act, 0, ex.down, ex.down_scale, ex.width // 32, ex.dims // COLS,
         plan, out, ex.dims, 0.0, skip, rows)


def dense(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp32 [..., N, K]: e4m3 times its block's scale (a reference)."""

    n, k = weight.shape[-2:]
    full = scale.float().repeat_interleave(BLOCK, -2)[..., :n, :].repeat_interleave(BLOCK, -1)[..., :k]
    return weight.float() * full
