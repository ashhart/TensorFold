"""Real Nemotron tensors and one mlx_lm layer of each type, built as the engine runs prompt chunks over 16 tokens."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

LAYERS = {"M": 0, "E": 1, "*": 5}            # the first Mamba, MoE and attention layers


def tensors(model: Path, names: tuple[str, ...]) -> dict[str, mx.array]:
    """Checkpoint tensors by name, read lazily from their shards and evaluated."""

    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    out: dict[str, mx.array] = {}
    for shard in sorted({index[n] for n in names}):
        loaded = mx.load(str(model / shard))
        out.update({n: loaded[n] for n in names if index[n] == shard})
    mx.eval(list(out.values()))
    return out


def args(model: Path):
    from mlx_lm.models.nemotron_h import ModelArgs

    return ModelArgs.from_dict(json.loads((model / "config.json").read_text()))


def block(model: Path, kind: str):
    """mlx_lm's NemotronHBlock for the first layer of `kind` with its 4-bit weights, lane_qmm in its linears."""

    from mlx_lm.models.nemotron_h import NemotronHBlock

    from tensorfold.kernels.qwen.dense.v1 import lane_qmm

    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"backbone.layers.{LAYERS[kind]}."
    found = tensors(model, tuple(n for n in index if n.startswith(prefix)))
    w = {n[len(prefix):]: a for n, a in found.items()}
    b = NemotronHBlock(args(model), kind)
    nn.quantize(b, group_size=64, bits=4, class_predicate=lambda p, m: f"{p}.scales" in w)
    b.load_weights(list(w.items()), strict=True)
    mx.eval(b.parameters())
    holder = nn.Module()
    holder.layer = b
    lane_qmm.install(holder, rows=lane_qmm.MAX_ROWS, tile=True, wide=True)
    return b


def cache(kind: str):
    """The engine's cache for a layer: RowStateCache for Mamba, AlternatingKVCache for attention, none for MoE."""

    if kind == "M":
        from tensorfold.families.nemotron_h.state_cache import RowStateCache

        return RowStateCache(2)
    if kind == "*":
        from tensorfold.engine.alternating_kv import AlternatingKVCache

        return AlternatingKVCache()
    return None
