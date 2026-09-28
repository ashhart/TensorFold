"""The MoE with experts streamed into a slot pool: routing ids for the host, then gate/up and down through the slots."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.qwen.flash_next.v1 import experts
from tensorfold.kernels.qwen.flash_next.v1.base import count, kernel

# one simdgroup per row: simd_topk's ids (comparisons only, so the resident kernels' ids) into the host's mailbox
_ROUTE = r"""
  const uint lane = thread_index_in_simdgroup;
  const int r = int(threadgroup_position_in_grid.x);
  if (r >= rows[0]) return;
  int ids[TOPK];
  float picked[TOPK];
  simd_topk_all<NE, TOPK>(LOGITS + r * NL, lane, ids, picked);
  if (lane == 0) {
    device uint32_t* box = (device uint32_t*)BOX;
    for (int k = 0; k < TOPK; k++) box[r * TOPK + k] = uint32_t(ids[k]);
    OUT[0] = uint32_t(rows[0]);
  }
"""

_GATEUP_PICK = "const size_t e = shared ? 0 : size_t(simd_topk<NE>(LOGITS + r * NL, slot, lane, picked));"
_DOWN_PICK = "const size_t e = shared ? 0 : size_t(PICK[r * TOPK + k]);"


def _gateup_source() -> str:
    """expert_gateup with only its weight address changed: expert e's rows live in pool slot SLOTOF[layer][e]."""

    if experts._EXPERT_GATEUP.count(_GATEUP_PICK) != 1:
        raise RuntimeError("stream_experts: expert_gateup's pick line changed")
    return experts._EXPERT_GATEUP.replace(_GATEUP_PICK, _GATEUP_PICK.replace("const size_t e =", "const size_t id =")
                                          + "\n  const size_t e = shared ? 0 : size_t(SLOTOF[LAYER[0] * NE + int(id)]);"
                                          ).replace("PICK[r * TOPK + slot] = uint32_t(e);",
                                                    "PICK[r * TOPK + slot] = uint32_t(id);")


def _down_source() -> str:
    """expert_down_y with only its weight address changed, through the same slot table."""

    if experts._EXPERT_DOWN_Y.count(_DOWN_PICK) != 1:
        raise RuntimeError("stream_experts: expert_down_y's pick line changed")
    return experts._EXPERT_DOWN_Y.replace(
        _DOWN_PICK, "const size_t e = shared ? 0 : size_t(SLOTOF[LAYER[0] * NE + int(PICK[r * TOPK + k])]);")


def route(logits: mx.array, top_k: int, box: mx.array) -> mx.array:
    """Write each row's top-k expert ids into ``box`` (the host reads it after the GPU's signal); returns a token."""

    rows = int(logits.shape[0])
    run = kernel("q4_stream_route", _ROUTE, ["LOGITS", "BOX", "rows"], ["OUT"])
    return run(inputs=[logits, box, count(rows)],
               template=[("NE", experts_of(logits)), ("NL", int(logits.shape[-1])), ("TOPK", top_k)],
               grid=(32 * rows, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.uint32])[0]


def experts_of(logits: mx.array) -> int:
    """Routed experts in a logits row: the router's rows carry the shared gate as one extra column."""

    return int(logits.shape[-1]) - 1


def gateup(x: mx.array, logits: mx.array, top_k: int, n_experts: int, pool: Any, slot_of: mx.array, layer: mx.array,
           shared: tuple[Any, Any], *, rows_per_simdgroup: int = 4, simdgroups: int = 2) -> tuple[mx.array, ...]:
    """expert_gateup over the pool's slots: the same activations, picks and weights as the resident kernel."""

    rows, dims = x.shape
    width = int(pool.gate.weight.shape[1])
    sg, su = shared
    run = kernel("q4_stream_gateup", _gateup_source,
                 ["X", "LOGITS", "GW", "GS", "GB", "UW", "US", "UB", "SGW", "SGS", "SGB", "SUW", "SUS", "SUB",
                  "SLOTOF", "LAYER"], ["ACT", "PICK", "WTS"])
    return tuple(run(inputs=[x, logits, pool.gate.weight, pool.gate.scales, pool.gate.biases, pool.up.weight,
                             pool.up.scales, pool.up.biases, sg.weight, sg.scales, sg.biases, su.weight, su.scales,
                             su.biases, slot_of, layer],
                     template=[("K", dims), ("N", width), ("TOPK", top_k), ("SHARED", 1), ("NE", n_experts),
                               ("NL", int(logits.shape[-1])), ("RPS", rows_per_simdgroup), ("SG", simdgroups)],
                     grid=(32 * simdgroups, width // (rows_per_simdgroup * simdgroups), rows * (top_k + 1)),
                     threadgroup=(32 * simdgroups, 1, 1),
                     output_shapes=[(rows, top_k + 1, width), (rows, top_k), (rows, top_k)],
                     output_dtypes=[mx.bfloat16, mx.uint32, mx.float32]))


def down(act: mx.array, picks: mx.array, n_experts: int, pool: Any, slot_of: mx.array, layer: mx.array,
         shared: Any, *, simdgroups: int = 2) -> mx.array:
    """expert_down_y over the pool's slots."""

    rows, slots, width = act.shape
    top_k = int(picks.shape[-1])
    dims = int(pool.down.weight.shape[1])
    run = kernel("q4_stream_down_y", _down_source,
                 ["ACT", "PICK", "DW", "DS", "DB", "SDW", "SDS", "SDB", "rows", "SLOTOF", "LAYER"], ["Y"])
    return run(inputs=[act, picks, pool.down.weight, pool.down.scales, pool.down.biases, shared.weight,
                       shared.scales, shared.biases, count(rows), slot_of, layer],
               template=[("NI", width), ("D", dims), ("TOPK", top_k), ("SG", simdgroups), ("NE", n_experts)],
               grid=(32 * simdgroups, dims // 8, -(-rows * slots // simdgroups)), threadgroup=(32 * simdgroups, 1, 1),
               output_shapes=[(rows, slots, dims)], output_dtypes=[mx.bfloat16])[0]


__all__ = ["down", "gateup", "route"]
