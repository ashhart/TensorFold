"""Merging a LoRA adapter into the Qwen-Image-2.1 transformer's weights."""

# Adapters in the diffusers layout: `transformer.<module path>.lora_A.weight` and `.lora_B.weight`, with the
# rank and alpha in the file's `lora_adapter_metadata`. The update is formed and added in float32: a bfloat16
# sum would round most of a small update away. Quantize to int8 after merging, or call `settle` to return the
# merged weights to bfloat16.

from __future__ import annotations

import json
import struct

PREFIXES = ("transformer.", "diffusion_model.")


def header(path) -> tuple[dict, dict]:
    """(tensor table, metadata) of a safetensors file, without reading the tensors."""

    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        table = json.loads(handle.read(size))
    return {k: v for k, v in table.items() if k != "__metadata__"}, table.get("__metadata__") or {}


def _alpha(metadata: dict) -> float | None:
    raw = metadata.get("lora_adapter_metadata")
    if not raw:
        return None
    return json.loads(raw).get("transformer.lora_alpha")


def _resolve(model, dotted: str):
    target = model
    for part in dotted.split("."):
        target = target[int(part)] if part.isdigit() else getattr(target, part)
    return target


def merge(model, path, strength: float = 1.0) -> dict:
    """Add every low-rank update in ``path`` to its projection, leaving the merged weights in float32."""

    import mlx.core as mx

    table, metadata = header(path)
    alpha = _alpha(metadata)
    tensors = mx.load(str(path))
    pairs: dict[str, dict[str, mx.array]] = {}
    for name in table:
        stem = name
        for prefix in PREFIXES:
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
        for tag in ("lora_A", "lora_B"):
            marker = f".{tag}.weight"
            alternate = f".{tag}.default.weight"
            for suffix in (marker, alternate):
                if stem.endswith(suffix):
                    pairs.setdefault(stem[: -len(suffix)], {})[tag] = tensors[name]
    if not pairs:
        raise ValueError(f"{path} holds no lora_A / lora_B pairs")
    merged = 0
    for module_path, pair in pairs.items():
        if set(pair) != {"lora_A", "lora_B"}:
            raise ValueError(f"{module_path} has only {sorted(pair)}")
        try:
            module = _resolve(model, module_path)
        except (AttributeError, IndexError, TypeError) as error:
            raise KeyError(f"the adapter targets {module_path}, which the model does not have") from error
        down, up = pair["lora_A"].astype(mx.float32), pair["lora_B"].astype(mx.float32)
        rank = down.shape[0]
        if up.shape != (module.weight.shape[0], rank) or down.shape[1] != module.weight.shape[1]:
            raise ValueError(f"{module_path}: update {tuple(up.shape)} x {tuple(down.shape)} does not fit "
                             f"{tuple(module.weight.shape)}")
        scale = strength * (float(alpha) / rank if alpha is not None else 1.0)
        module.weight = module.weight.astype(mx.float32) + scale * (up @ down)
        mx.eval(module.weight)
        merged += 1
    mx.clear_cache()
    return {"matrices": merged, "alpha": alpha, "strength": strength}


def settle(model) -> int:
    """Return every float32 weight left by ``merge`` to bfloat16; the number of tensors changed."""

    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    changed = [(name, value.astype(mx.bfloat16)) for name, value in tree_flatten(model.parameters())
               if value.dtype == mx.float32]
    if changed:
        model.update(tree_unflatten(changed))
        mx.eval(model.parameters())
    return len(changed)
