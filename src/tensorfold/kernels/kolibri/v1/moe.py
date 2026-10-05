"""Kolibri 1's routed experts for any number of rows: Gemma 4's expert kernels with SwiGLU in place of GeGLU."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.gemma.v1 import moe as gemma
from tensorfold.kernels.gemma.v1.base import Kernel
from tensorfold.kernels.inputs import padded

_SILU = r"""
// x sigmoid(x) in fp32
inline float silu(float x) {
  return x / (1.0f + metal::exp(-x));
}
"""

# Gemma's gate|up kernel, bf16(silu(bf16(gate)) * bf16(up)) for every (row, slot)
_gateup = Kernel("kolibri_expert_gateup", gemma._EXPERT_GATEUP.replace("gelu_tanh(", "silu("),
                 ["X", "IDX", "GW", "GSC", "GBI", "UW", "USC", "UBI"], ["ACT"], header=gemma._QDOT_HEADER + _SILU)


def route(logits: mx.array, bias: mx.array, top_k: int) -> tuple[mx.array, mx.array]:
    """fp32 router logits [R, E]: top k of ``logits + bias`` (best first), weighted by bf16(sigmoid(logits)), no
    renormalisation; ids and weights flat, row r's at [r K, r K + K), padded for the expert kernels."""

    ids = mx.argsort(-(logits + bias), axis=-1)[:, :top_k]
    weights = mx.sigmoid(mx.take_along_axis(logits, ids, axis=-1)).astype(mx.bfloat16)
    return padded(ids.astype(mx.uint32).reshape(-1)), padded(weights.reshape(-1))


def expert_gateup(x: mx.array, ids: mx.array, top_k: int, gate: Any, up: Any, *, simdgroups: int = 2,
                  rows_per_simdgroup: int = 4) -> mx.array:
    """x [R, K] and ``route``'s ids -> bf16(silu(x W_gate^T) * (x W_up^T)) for every (row, slot): [R * TOPK, N]."""

    gemma.check_q4(gate)
    rows, dims = x.shape
    width = gate.weight.shape[1]
    per_tg = simdgroups * rows_per_simdgroup
    if width % per_tg or dims % 16:
        raise ValueError("expert_gateup: expert width must split into threadgroups, inputs into chunks of 16")
    consts = (("K", dims), ("N", width), ("TOPK", top_k), ("GS", gate.group_size), ("SG", simdgroups),
              ("RPS", rows_per_simdgroup))
    return _gateup(consts, inputs=[x, ids, gate.weight, gate.scales, gate.biases, up.weight, up.scales, up.biases],
                   grid=(32 * simdgroups, width // per_tg, rows * top_k), threadgroup=(32 * simdgroups, 1, 1),
                   output_shapes=[(rows * top_k, width)], output_dtypes=[mx.bfloat16])[0]


expert_down = gemma.expert_down
check_q4 = gemma.check_q4

__all__ = ["check_q4", "expert_down", "expert_gateup", "route"]
