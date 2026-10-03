"""oMLX's own DeepSeek-V4.1 model as the numerical reference; needs an oMLX checkout on ``OMLX_SRC``."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def available() -> bool:
    src = os.environ.get("OMLX_SRC", "")
    return bool(src) and (Path(src) / "omlx" / "patches" / "deepseek_v41" / "language.py").is_file()


def set_module(model: Any, path: str, module: Any) -> None:
    """oMLX's ``loading.set_module`` (that module imports mlx_vlm)."""

    parent = model
    parts = path.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if isinstance(parent, list) else getattr(parent, part)
    if isinstance(parent, list):
        parent[int(parts[-1])] = module
    else:
        setattr(parent, parts[-1], module)


def load_reference(folder: Path, token_map: list[int], layers: int | None = None) -> Any:
    """oMLX's LanguageModel with the checkpoint's weights, loaded as oMLX's ``loading.load`` places them."""

    src = os.environ["OMLX_SRC"]
    if src not in sys.path:
        sys.path.insert(0, src)
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.language import LanguageModel
    from omlx.patches.deepseek_v41.quantization import QuantizedProjection
    from omlx.patches.deepseek_v41.storage import DiskEngramEmbedding

    raw = json.loads((folder / "config.json").read_text())
    spec = raw["omlx_deepseek_v41"]
    cfg = ModelConfig.from_dict({**raw, "omlx_deepseek_v41": {**spec, "preserve_mtp": False}})
    if layers is not None:
        cfg.n_layers = int(layers)
        cfg.engram_layer_ids = tuple(i for i in cfg.engram_layer_ids if i < layers)
        cfg.engram_num_embeddings = tuple(cfg.engram_num_embeddings[:len(cfg.engram_layer_ids)])
    model = LanguageModel(cfg)
    prefix = "language_model."
    mapping = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    engram_keys = set()
    for name, table in spec["engram_tables"].items():
        if layers is not None and int(name.split(".")[2]) >= layers:
            continue
        engram_keys |= {table["weight_key"], table["scale_key"], table["bias_key"]}
        set_module(model, name[len(prefix):], DiskEngramEmbedding(
            folder / table["weight_file"], table["weight_key"], table["scale_key"], folder / table["scale_file"],
            bias_key=table["bias_key"], bits=table["bits"], group_size=table["group_size"]))
    values = {}
    for shard in sorted(set(mapping.values())):
        for key, value in mx.load(str(folder / shard)).items():
            if key in engram_keys or not key.startswith(prefix) or key.startswith(prefix + "mtp."):
                continue
            values[key[len(prefix):]] = value
    for name, q in spec["quantized_modules"].items():
        short = name[len(prefix):]
        if short.startswith("mtp.") or short + ".weight" not in values:
            continue
        if layers is not None and short.startswith("layers.") and int(short.split(".")[1]) >= layers:
            continue
        set_module(model, short, QuantizedProjection(values.pop(short + ".weight"), values.pop(short + ".scales"),
                                                     bits=q["bits"], mode=q["mode"],
                                                     group_size=q.get("group_size", 32)))
    expected = {n for n, _ in tree_flatten(model.parameters())}
    model.load_weights([(k, v) for k, v in values.items() if k in expected], strict=False)
    mx.eval(model.parameters())
    model.set_token_map(token_map)
    for layer in model.layers:
        if "engram" in layer:
            layer.engram.embed.make_resident()
    return model


def reference_logits(model: Any, ids: list[int]) -> Any:
    """Teacher-forced fp32 logits [L, V] of one prompt, one forward."""

    import mlx.core as mx

    logits = model(mx.array([ids], dtype=mx.int32), cache=model.make_cache())
    mx.eval(logits)
    return logits[0]
