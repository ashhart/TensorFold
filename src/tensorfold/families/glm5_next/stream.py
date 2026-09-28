"""GLM-5.3-Flash with its decoder layers' routed experts streamed from the checkpoint into a GPU slot pool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from tensorfold.families.glm5_next import config as C
from tensorfold.families.glm5_next.layouts import routed_expert
from tensorfold.families.glm5_next.linear import Q
from tensorfold.streaming.pool import PARTS, Streamer, expert_nbytes, sources

PROJS = ("gate", "up", "down")


def expert_names(model_dir: Path, layers: int) -> dict[tuple, str]:
    """The decoder layers' routed expert tensors by (layer, proj, part[, expert]), as the checkpoint names them."""

    index = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    names = {}
    for name in index:
        key = routed_expert(name, layers)
        if key is not None:
            names[key] = name
    return names


def attach(model: Any, model_dir: Path, gib: float) -> Streamer:
    """Build the pool for ``gib`` GiB of experts and route every decoder MoE block through it."""

    from tensorfold.kernels.glm.flash.v1 import kernels as K
    from tensorfold.kernels.glm.flash.v1 import moe as MK
    from tensorfold.streaming.build import load as hostsync

    cfg = model.args
    blocks = [(i, layer.mlp) for i, layer in enumerate(model.layers) if hasattr(layer.mlp, "router")]
    if not blocks:
        raise ValueError("--ssd-experts: this checkpoint has no routed experts to stream")
    if not (K.metal() and "moe" in C.FUSED and MK.SPLIT_SHARED and all(moe.fused_ok for _, moe in blocks)):
        raise ValueError("--ssd-experts streams GLM-5.3-Flash's experts through its fused Metal MoE kernels, which "
                         "take 4-bit experts in groups of 64 on a Mac GPU")
    found, shapes = sources(model_dir, expert_names(model_dir, cfg.num_hidden_layers))
    slots = int(gib * 2**30) // expert_nbytes(found, blocks[0][0])
    streamer = Streamer(found, shapes, layers=cfg.num_hidden_layers, experts=cfg.n_routed_experts,
                        top_k=cfg.num_experts_per_tok, slots=slots, box_rows=C.DECODE_ROWS, hostsync=hostsync())
    for i, moe in blocks:
        moe.streamer, moe.stream_layer = streamer, i
    model.streamer = streamer
    print(f"[glm5] routed experts stream from SSD into {slots} slots ({gib:g} GiB)", flush=True)
    return streamer


def moe(block: Any, x: mx.array, rows_exact: bool) -> mx.array:
    """``MoE.__call__`` with the routed experts in the pool: decode windows fused, prompt chunks through a window."""

    from tensorfold.kernels.glm.flash.v1 import stream_moe

    streamer, layer = block.streamer, block.stream_layer
    if rows_exact:
        return stream_moe.moe_rows(block, x, streamer, layer)
    idx, w = block.select(x)
    token = stream_moe.present(idx.reshape(-1).astype(mx.uint32), streamer.box, block.cfg.n_routed_experts)
    token = streamer.hold(token, "window", layer, int(x.shape[0]))
    views = streamer.window_views()
    qs = tuple(Q(*mx.depends([getattr(views[p], part) for part in PARTS], [token]), bits=block.gate.bits,
                 group=block.gate.group) for p in PROJS)
    out = block.combine(w, block.experts(x, idx, qs), x.dtype)
    if block.shared is not None:
        out = out + block.shared(x, rows_exact)
    return out


__all__ = ["attach", "expert_names", "moe"]
