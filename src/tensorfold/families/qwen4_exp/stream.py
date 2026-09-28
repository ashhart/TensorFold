"""Flash Next with its routed experts streamed from the checkpoint into a GPU slot pool; the rest stays resident."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from tensorfold.streaming.pool import PARTS, Streamer, expert_nbytes, sources

PROJS = ("gate", "up", "down")
PREFILL_ROWS = 4096          # the largest prompt chunk a window call takes

# one threadgroup: the box's first E words become 1 for each expert some routed row picked, else 0
_PRESENT = r"""
  const uint t = thread_position_in_threadgroup.x;
  device uint32_t* box = (device uint32_t*)BOX;
  for (uint e = t; e < uint(NE); e += 1024) box[e] = 0u;
  threadgroup_barrier(mem_flags::mem_device);
  for (uint i = t; i < uint(N); i += 1024) box[IDS[i]] = 1u;
  threadgroup_barrier(mem_flags::mem_device);
  if (t == 0) OUT[0] = 1u;
"""


def expert_names(layers: int) -> dict[tuple[int, str, str], str]:
    return {(i, proj, part): f"language_model.model.layers.{i}.mlp.switch_mlp.{proj}_proj.{part}"
            for i in range(layers) for proj in PROJS for part in PARTS}


def switch_keys(key: str) -> bool:
    """Weights the pool serves instead of the model: the decoder layers' routed stacks (not the MTP head's)."""

    return key.startswith("model.layers.") and ".mlp.switch_mlp." in key


def attach(model: Any, model_dir: Path, gib: float) -> Streamer:
    """Build the pool for ``gib`` GiB of experts and route every decoder MoE through it."""

    from tensorfold.streaming.build import load as hostsync

    cfg = model.args
    found, shapes = sources(model_dir, expert_names(cfg.num_hidden_layers))
    slots = int(gib * 2**30) // expert_nbytes(found, 0)
    streamer = Streamer(found, shapes, layers=cfg.num_hidden_layers, experts=cfg.num_experts,
                        top_k=cfg.num_experts_per_tok, slots=slots, box_rows=PREFILL_ROWS, hostsync=hostsync())
    for index, layer in enumerate(model.layers):
        moe = layer.mlp
        moe.__dict__["streamer"] = streamer
        moe.__dict__["stream_layer"] = index
        for proj in PROJS:                       # the stacks stay on disk: placeholders keep the modules' names
            linear = getattr(moe.switch_mlp, f"{proj}_proj")
            for part in PARTS:
                setattr(linear, part, mx.zeros((1,), dtype=getattr(linear, part).dtype))
    model.__dict__["streamer"] = streamer
    return streamer


def moe_rows(moe: Any, x: mx.array, logits: mx.array, shared: tuple[Any, Any, Any]) -> tuple[str, tuple[mx.array, ...]]:
    """FusedDecode's MoE for streamed experts: the resident kernels' arithmetic, weights read through slots."""

    from tensorfold.kernels.qwen.flash_next.v1 import stream_experts

    streamer, layer = moe.__dict__["streamer"], moe.__dict__["stream_layer"]
    cfg_top = moe.top_k
    token = stream_experts.route(logits, cfg_top, streamer.box)
    token = streamer.hold(token, "rows", layer, int(x.shape[0]))
    slot_of, layer_id = mx.depends([streamer.slot_of, streamer.layer_ids[layer]], [token])
    pool = _Pool(streamer.pool)
    gate, up, down = shared
    act, picks, weights = stream_experts.gateup(x, logits, cfg_top, streamer.experts, pool, slot_of, layer_id,
                                                (gate, up))
    return "grouped", (stream_experts.down(act, picks, streamer.experts, pool, slot_of, layer_id, down), weights,
                       logits)


def moe_chunk(moe: Any, x: mx.array) -> mx.array:
    """SparseMoE on a prompt chunk: its routing, then the layer's picked experts loaded into the window slots."""

    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm
    from tensorfold.kernels.qwen.flash_next.v1.base import kernel

    streamer, layer = moe.__dict__["streamer"], moe.__dict__["stream_layer"]
    experts, weights = moe.route(x)
    ids = experts.reshape(-1).astype(mx.uint32)
    present = kernel("q4_stream_present", _PRESENT, ["IDS", "BOX"], ["OUT"])
    token = present(inputs=[ids, streamer.box], template=[("NE", streamer.experts), ("N", int(ids.size))],
                    grid=(1024, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.uint32])[0]
    token = streamer.hold(token, "window", layer, int(x.shape[1]))
    views = streamer.window_views()
    switch = copy.copy(moe.switch_mlp)
    for proj in PROJS:
        linear = copy.copy(getattr(moe.switch_mlp, f"{proj}_proj"))
        for part, array in zip(PARTS, mx.depends([getattr(views[proj], p) for p in PARTS], [token])):
            setattr(linear, part, array)
        setattr(switch, f"{proj}_proj", linear)
    if prefill_mm.active(int(x.shape[0]) * int(x.shape[1])) and x.shape[0] == 1 and x.dtype == mx.bfloat16:
        return prefill_mm.moe(moe, x, route=(experts, weights), switch=switch)
    routed = (switch(x, experts) * weights[..., None]).sum(axis=-2)
    return routed + moe.shared_expert(x) * mx.sigmoid(moe.shared_expert_gate(x))


class _Pool:
    """The pool's projections as the kernels' gate/up/down arguments."""

    def __init__(self, pool: dict) -> None:
        self.gate, self.up, self.down = pool["gate"], pool["up"], pool["down"]


__all__ = ["attach", "expert_names", "moe_chunk", "moe_rows", "switch_keys"]
