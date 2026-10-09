"""Write edited projections into the model's own shards in place, as bf16 modules marked unquantized."""
from __future__ import annotations

import json
import os

import mlx.core as mx

SUFFIX = ".safetensors"


def write_json(path: str, data: dict):
    with open(path + ".tmp", "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(path + ".tmp", path)


def bake(path: str, weights: dict[str, mx.array]):
    """Rewrite each shard holding an edited module, then config.json and the index; every write lands by rename."""
    index_path, config_path = os.path.join(path, "model.safetensors.index.json"), os.path.join(path, "config.json")
    with open(index_path) as f:
        index = json.load(f)
    with open(config_path) as f:
        config = json.load(f)
    shards: dict[str, list[str]] = {}
    for module in weights:
        shards.setdefault(index["weight_map"][module + ".weight"], []).append(module)
    total = index.setdefault("metadata", {}).get("total_size", 0)
    for shard, modules in shards.items():
        file = os.path.join(path, shard)
        tensors, meta = mx.load(file, return_metadata=True)
        for module in modules:
            w = weights[module].astype(mx.bfloat16)
            total -= sum(tensors.pop(module + s).nbytes for s in (".weight", ".scales", ".biases") if module + s in tensors)
            index["weight_map"].pop(module + ".scales", None)
            index["weight_map"].pop(module + ".biases", None)
            tensors[module + ".weight"] = w
            total += w.nbytes
        tmp = file[: -len(SUFFIX)] + ".tmp" + SUFFIX
        mx.save_safetensors(tmp, tensors, metadata=meta)
        os.replace(tmp, file)
    index["metadata"]["total_size"] = total
    for key in ("quantization", "quantization_config"):
        if key in config:
            config[key].update({module: False for module in weights})
    write_json(config_path, config)
    write_json(index_path, index)
