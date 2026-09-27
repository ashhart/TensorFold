"""Grouped MoE experts on MLX 4-bit weights; a (row, slot) pair's bits never depend on the other rows of the call."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

NTW = 4                  # n8 tiles a warp
COLS = 8 * NTW           # output columns a warp
TILE = 16                # pairs an item holds (decode form)
PREFILL_TILE = 64        # pairs an item holds (prefill form)
SMALL = 1024             # pairs the one-block plan takes; wider plans rank in blocks of 1024 pairs


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    here = Path(__file__).parent
    return load(name="tensorfold_experts_v6", sources=[str(here / "experts.cpp"), str(here / "experts.cu"),
                                                        str(here / "experts_prefill.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def _nibbles(w: torch.Tensor) -> torch.Tensor:
    """Nibbles to (i0, i2, i4, i6, i1, i3, i5, i7), so one shift and mask give two adjacent inputs as a bf16 pair."""

    w = w.to(torch.int64) & 0xFFFFFFFF
    out = torch.zeros_like(w)
    for i in range(8):
        out |= ((w >> (4 * i)) & 0xF) << (4 * (i // 2 + 4 * (i % 2)))
    return torch.where(out >= 2 ** 31, out - 2 ** 32, out).to(torch.int32)


def pack(words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, gs: int, chunk: int = 32) -> torch.Tensor:
    """MLX words [E, N, K/8], scales and biases -> [E, N/32, K/gs, block]: B-fragments, then scales and biases."""

    if gs not in (32, 64):
        raise ValueError(f"groups of 32 or 64 inputs, not {gs}")
    e, n, k8 = words.shape
    k = k8 * 8
    if n % COLS or k % gs or scales.shape != (e, n, k // gs) or biases.shape != scales.shape:
        raise ValueError(f"experts: shape {tuple(words.shape)} with scales {tuple(scales.shape)} does not pack")
    kg, nb, h = k // gs, n // COLS, gs // 32
    wpl = NTW * h
    out = torch.empty((e, nb, kg, 32 * wpl + 8 * NTW), dtype=torch.int32, device=words.device)
    for e0 in range(0, e, chunk):
        w = _nibbles(words[e0:e0 + chunk].view(torch.int32))
        c = w.shape[0]
        w = w.view(c, nb, NTW, 8, kg, 4, h).permute(0, 1, 4, 2, 6, 3, 5).reshape(c, nb, kg, wpl // 4, 4, 32)
        out[e0:e0 + c, :, :, :32 * wpl] = w.permute(0, 1, 2, 3, 5, 4).reshape(c, nb, kg, 32 * wpl)
        sb = []
        for t in (scales, biases):
            v = t[e0:e0 + c].reshape(c, nb, NTW, 4, 2, kg).permute(0, 1, 5, 3, 2, 4).contiguous()
            sb.append(v.view(torch.int32).reshape(c, nb, kg, 4, NTW))
        out[e0:e0 + c, :, :, 32 * wpl:] = torch.cat(sb, dim=-1).reshape(c, nb, kg, 8 * NTW)
    return out


def unpack(blocks: torch.Tensor, gs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``pack``'s inverse: blocks [E, N/32, K/gs, block] -> MLX words [E, N, K/8], scales and biases [E, N, K/gs]."""

    e, nb, kg, _ = blocks.shape
    h = gs // 32
    wpl = NTW * h
    w = blocks[..., :32 * wpl].reshape(e, nb, kg, wpl // 4, 32, 4).permute(0, 1, 2, 3, 5, 4)
    w = w.reshape(e, nb, kg, NTW, h, 8, 4).permute(0, 1, 3, 5, 2, 6, 4).reshape(e, nb * COLS, kg * 4 * h)
    v = w.to(torch.int64) & 0xFFFFFFFF
    words = torch.zeros_like(v)
    for s in range(8):                                  # nibble slot s holds input 2 (s % 4) + s // 4
        words |= ((v >> (4 * s)) & 0xF) << (4 * (2 * (s % 4) + s // 4))
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    sb = blocks[..., 32 * wpl:].contiguous().view(torch.bfloat16).reshape(e, nb, kg, 4, 2, NTW, 2)
    scales, biases = (sb[:, :, :, :, i].permute(0, 1, 4, 3, 5, 2).reshape(e, nb * COLS, kg) for i in (0, 1))
    return words, scales.contiguous(), biases.contiguous()


@dataclass
class Experts:
    """One layer's experts (the shared ones included): gate and up (SwiGLU) or up (relu^2), and down."""

    up: torch.Tensor          # [E, NI/32, D/gs, M, block] int32, M = 2 (gate, then up) or 1
    down: torch.Tensor        # [E, D/32, NI/gs, 1, block]
    gs: int
    width: int                # NI: the expert's (or this rank's) intermediate width
    dims: int                 # D
    limit: float = 0.0        # SwiGLU clip (0: none)

    @property
    def count(self) -> int:
        return self.up.shape[0]

    @property
    def swiglu(self) -> bool:
        return self.up.shape[3] == 2

    def bytes_per_expert(self) -> int:
        return (self.up[0].numel() + self.down[0].numel()) * 4


def make(up: list[tuple], down: tuple, gs: int, *, limit: float = 0.0) -> Experts:
    """One layer's MLX arrays: ``up`` [gate, up] (SwiGLU) or [up] (relu^2), ``down``, each (words, scales, biases)."""

    if len(up) not in (1, 2):
        raise ValueError("experts take a gate and an up projection, or an up projection alone")
    width, dims = up[0][0].shape[1], up[0][0].shape[2] * 8
    u = torch.stack([pack(*m, gs) for m in up], dim=3)
    d = pack(*down, gs).unsqueeze(3)
    return Experts(u, d, gs, width, dims, float(limit))


def max_items(pairs: int, experts: int, tile: int = TILE) -> int:
    """Items a plan of ``pairs`` can hold: an item per used expert, plus one per ``tile`` pairs past its first."""

    return min(pairs, experts) + pairs // tile


class Plan:
    """Scratch grouping pairs by expert: members, items (expert, first, count), counts [items, distinct experts]."""

    def __init__(self, rows: int, slots: int, experts: int, device: torch.device | str, *,
                 prefill: bool = False) -> None:
        pairs = rows * slots
        self.rows, self.slots, self.experts, self.prefill = rows, slots, experts, prefill
        self.tile = PREFILL_TILE if prefill else TILE
        self.members = torch.zeros((pairs,), dtype=torch.int32, device=device)
        self.items = torch.zeros((max_items(pairs, experts, self.tile), 3), dtype=torch.int32, device=device)
        self.counts = torch.zeros((2,), dtype=torch.int32, device=device)
        wide = pairs > SMALL
        self.rank = torch.zeros((pairs if wide else 1,), dtype=torch.int32, device=device)
        self.hist = torch.zeros((-(-pairs // 1024) * experts if wide else 1,), dtype=torch.int32, device=device)


def route(picks: torch.Tensor, plan: Plan) -> None:
    """``picks`` [R, slots] int32, contiguous: each (row, slot) pair's expert id (shared experts included)."""

    rows, slots = picks.shape
    if slots != plan.slots or rows > plan.rows:
        raise ValueError(f"picks {tuple(picks.shape)} do not fit a plan of {plan.rows} x {plan.slots}")
    _ext().plan(picks, rows * slots, plan.experts, plan.tile, plan.members, plan.items, plan.counts, plan.rank,
                plan.hist)


def gate_up(x: torch.Tensor, ex: Experts, plan: Plan, out: torch.Tensor, rows: int) -> None:
    """x [R, D] bf16 (unit-stride rows) -> out [R * slots, NI] bf16, each pair's activation."""

    items = max_items(rows * plan.slots, plan.experts, plan.tile)
    epi = 2 if ex.swiglu else 1
    if plan.prefill:
        _ext().prefill(ex.gs, epi, x, plan.slots, ex.up, ex.dims // ex.gs, ex.width // COLS, plan.items, plan.counts,
                       plan.members, out, ex.width, ex.limit, items)
    else:
        _ext().run(ex.gs, epi, x, plan.slots, ex.up, ex.dims // ex.gs, ex.width // COLS, plan.items, plan.counts,
                   plan.members, out, ex.width, ex.limit, items * (ex.width // COLS))


def down(act: torch.Tensor, ex: Experts, plan: Plan, out: torch.Tensor, rows: int) -> None:
    """act [R * slots, NI] bf16 -> out [R * slots, D] per pair: fp32, or bf16 on a prefill plan when ``out`` is bf16."""

    items = max_items(rows * plan.slots, plan.experts, plan.tile)
    if plan.prefill:
        _ext().prefill(ex.gs, 3 if out.dtype == torch.bfloat16 else 0, act, 0, ex.down, ex.width // ex.gs,
                       ex.dims // COLS, plan.items, plan.counts, plan.members, out, ex.dims, 0.0, items)
    else:
        _ext().run(ex.gs, 0, act, 0, ex.down, ex.width // ex.gs, ex.dims // COLS, plan.items, plan.counts,
                   plan.members, out, ex.dims, 0.0, items * (ex.dims // COLS))
