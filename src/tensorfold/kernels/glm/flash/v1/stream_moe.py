"""GLM's fused MoE with the routed experts in a slot pool: picks to the host between marks, rows read through slots."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels import inputs
from tensorfold.kernels.glm.flash.v1 import moe as MK
from tensorfold.kernels.glm.flash.v1 import widths as W
from tensorfold.kernels.glm.flash.v1.fused import MAX_ROWS, _kernel

# a thread a pick: the route kernel's picks into the host's mailbox (written through the input), then a token
_BOX = r"""
  const uint i = thread_position_in_grid.x;
  device uint32_t* box = (device uint32_t*)BOX;
  if (i < uint(N)) box[i] = uint32_t(PICK[i]);
  if (i == 0) OUT[0] = 1u;
"""

# one threadgroup: the box's first NE words become 1 for each expert some prompt row picked, else 0
PRESENT = r"""
  const uint t = thread_position_in_threadgroup.x;
  device uint32_t* box = (device uint32_t*)BOX;
  for (uint e = t; e < uint(NE); e += 1024) box[e] = 0u;
  threadgroup_barrier(mem_flags::mem_device);
  for (uint i = t; i < uint(N); i += 1024) box[IDS[i]] = 1u;
  threadgroup_barrier(mem_flags::mem_device);
  if (t == 0) OUT[0] = 1u;
"""

_GATEUP_AT = "const size_t e = shared ? 0 : size_t(UIDS[u]);"
_DOWN_AT = "const size_t at = shared ? size_t(row0) : size_t(UIDS[u]) * N + row0;"
_SLOT = "SLOTOF[LAYER[0] * NE + UIDS[u]]"
_GATEUP_IN = ["X", "GW", "GS", "GB", "UW", "US", "UB", "SGU", "SGUS", "SGUB", "UIDS", "UMEM", "UCOUNT", "LIM",
              "SLOTOF", "LAYER"]
_DOWN_IN = ["ACT", "DW", "DS", "DB", "SDW", "SDS", "SDB", "UIDS", "UMEM", "UCOUNT", "SLOTOF", "LAYER"]


def slotted(source: str, line: str) -> str:
    """A resident MoE kernel whose routed expert rows are addressed through the slot table; nothing else changes."""

    if source.count(line) != 1:
        raise RuntimeError("stream_moe: a resident MoE kernel's expert address changed")
    return source.replace(line, line.replace("UIDS[u]", _SLOT))


def present(ids: mx.array, box: mx.array, experts: int) -> mx.array:
    """Flags in ``box`` for the experts ``ids`` names (a prompt chunk's picks); returns a token."""

    run = _kernel("moe_present", PRESENT, ["IDS", "BOX"], ["OUT"])
    return run(inputs=[ids, box], template=[("NE", experts), ("N", int(ids.size))], grid=(1024, 1, 1),
               threadgroup=(1024, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.uint32])[0]


def moe_rows(moe: Any, x: mx.array, streamer: Any, layer: int, *, rps: int = 4) -> mx.array:
    """``moe.moe_rows`` (shared expert apart) with the routed experts read through ``streamer``'s slots."""

    if not MK.SPLIT_SHARED:
        raise RuntimeError("streamed experts take the MoE block with its shared expert apart")
    rows, dims = x.shape
    cfg = moe.cfg
    top, experts = cfg.num_experts_per_tok, cfg.n_routed_experts
    inter, sh, maxu = moe.gate.outs, moe.shared, rows * top
    g, u, d = streamer.pool["gate"], streamer.pool["up"], streamer.pool["down"]
    gateup = _kernel("moe_gateup_slots", slotted(MK._MOE_GATEUP, _GATEUP_AT), _GATEUP_IN, ["ACT"])
    down = _kernel("moe_down_slots", slotted(MK._MOE_DOWN, _DOWN_AT), _DOWN_IN, ["Y"])
    (gu_v, gu_lb), (dn_v, dn_lb) = W.QFAST_ALL[sh.gate_up.bits], W.QFAST_ALL[sh.down.bits]
    none = moe.__dict__.get("_no_group")
    if none is None:
        none = moe._no_group = inputs.ints(())
        mx.eval(none)

    def gu(part: int, slots: int, zs: int, group: tuple[mx.array, ...]) -> mx.array:
        return gateup(inputs=[x, g.weight, g.scales, g.biases, u.weight, u.scales, u.biases, sh.gate_up.weight,
                              sh.gate_up.scales, sh.gate_up.biases, *group[:3], moe.limit_arr, *group[3:]],
                      template=[("K", dims), ("N", inter), ("RPS", rps), ("TOPK", top), ("MAXR", MAX_ROWS),
                                ("MAXU", maxu), ("PART", part), ("SB", sh.gate_up.bits), ("SV", gu_v),
                                ("SLB", gu_lb), ("NE", experts)],
                      grid=(32 * rows, inter // rps, zs), threadgroup=(32 * rows, 1, 1),
                      output_shapes=[(rows, slots, inter)], output_dtypes=[mx.bfloat16])[0]

    def dn(act: mx.array, part: int, slots: int, zs: int, group: tuple[mx.array, ...]) -> mx.array:
        return down(inputs=[act, d.weight, d.scales, d.biases, sh.down.weight, sh.down.scales, sh.down.biases,
                            *group],
                    template=[("K", inter), ("N", dims), ("RPS", rps), ("TOPK", top), ("MAXR", MAX_ROWS),
                              ("MAXU", maxu), ("PART", part), ("SB", sh.down.bits), ("SV", dn_v), ("SLB", dn_lb),
                              ("NE", experts)],
                    grid=(32 * rows, dims // rps, zs), threadgroup=(32 * rows, 1, 1),
                    output_shapes=[(rows, slots, dims)], output_dtypes=[mx.bfloat16])[0]

    blank = (none,) * 5
    ys = dn(gu(1, 1, 1, blank), 1, 1, 1, blank)
    logits = MK.router_rows(x.astype(mx.float32), moe)
    threads = max(32 * MAX_ROWS, -(-experts // 32) * 32)
    route = _kernel("moe_route", MK._MOE_ROUTE, ["LOGITS", "BIAS", "SCALE"], ["PICK", "WTS", "UIDS", "UMEM", "UCOUNT"])
    pick, wts, uids, umem, ucount = route(
        inputs=[logits, moe.bias, moe.scale_arr],
        template=[("NE", experts), ("TOPK", top), ("MAXR", MAX_ROWS), ("NT", threads)],
        grid=(threads, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(rows, top), (rows, top), (max(rows * top, inputs.MIN_ELEMENTS),), (rows * top, MAX_ROWS),
                       (inputs.MIN_ELEMENTS,)],
        output_dtypes=[mx.int32, mx.float32, mx.int32, mx.int32, mx.int32])
    box = _kernel("moe_box", _BOX, ["PICK", "BOX"], ["OUT"])
    token = box(inputs=[pick, streamer.box], template=[("N", maxu)], grid=(-(-maxu // 32) * 32, 1, 1),
                threadgroup=(32, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.uint32])[0]
    token = streamer.hold(token, "rows", layer, rows)
    table, lid = mx.depends([streamer.slot_of, streamer.layer_ids[layer]], [token])
    group = (uids, umem, ucount, table, lid)
    y = dn(gu(2, top, maxu, group), 2, top, maxu, group)
    combine = _kernel("moe_combine_split", MK._MOE_COMBINE_SPLIT, ["YS", "Y", "WTS"], ["OUT"])
    return combine(inputs=[ys.reshape(rows, dims), y, wts], template=[("D", dims), ("TOPK", top)],
                   grid=(rows * dims, 1, 1), threadgroup=(256, 1, 1),
                   output_shapes=[(rows, dims)], output_dtypes=[mx.bfloat16])[0]


__all__ = ["PRESENT", "moe_rows", "present", "slotted"]
