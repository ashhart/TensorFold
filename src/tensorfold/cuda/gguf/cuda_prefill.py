"""CUDA staged grouped prefill for prepared IQ2/Q2 SoA weights."""

from __future__ import annotations

import torch

from .kernels import codebook
from .prepare import _ext


def iq2_soa_group_prefill(x: torch.Tensor, weight, plan, out: torch.Tensor, *, slots: int, input_slots: bool) -> None:
    """Write grouped prefill into ``out`` [rows, slots, N] using the staged CUDA kernel."""

    k, n = weight.shape[:2]
    rows = x.shape[0]
    x_in = x.reshape(rows * slots, k).contiguous() if input_slots and x.ndim == 3 else x.contiguous()
    slot_arg = 0 if input_slots else slots
    flat = out.reshape(rows * slots, n)
    _ext().iq2_soa_prefill(
        x_in,
        slot_arg,
        weight.data,
        int(weight.bn),
        codebook(x.device),
        plan.items,
        plan.counts,
        plan.members,
        flat,
        n,
        k,
        weight.shape[2],
    )


def q2_soa_group_prefill(x: torch.Tensor, weight, plan, out: torch.Tensor, *, slots: int, input_slots: bool) -> None:
    """Write Q2_K SoA grouped prefill into ``out``."""

    k, n = weight.shape[:2]
    rows = x.shape[0]
    x_in = x.reshape(rows * slots, k).contiguous() if input_slots and x.ndim == 3 else x.contiguous()
    slot_arg = 0 if input_slots else slots
    dm_bytes = int(weight.bn) & 0xFFFFFFFF
    sc_bytes = int(weight.bn) >> 32
    _ext().q2_soa_prefill(
        x_in,
        slot_arg,
        weight.data,
        dm_bytes,
        sc_bytes,
        plan.items,
        plan.counts,
        plan.members,
        out.reshape(rows * slots, n),
        n,
        k,
        weight.shape[2],
    )


def iq2_soa_gate_up_prefill(x: torch.Tensor, gate, up, plan, *, slots: int) -> torch.Tensor:
    """Fused gate+up+SwiGLU into ``[rows, slots, N]`` via the staged CUDA kernel."""

    k, n = gate.shape[:2]
    rows = x.shape[0]
    out = torch.empty((rows, slots, n), dtype=torch.bfloat16, device=x.device)
    _ext().iq2_soa_gate_up(
        x.contiguous(),
        slots,
        gate.data,
        up.data,
        int(gate.bn),
        codebook(x.device),
        plan.items,
        plan.counts,
        plan.members,
        out.reshape(rows * slots, n),
        n,
        k,
        gate.shape[2],
    )
    return out
