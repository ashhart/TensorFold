"""Applying adapters to the H3 transformer: low-rank updates and small parameter corrections, in float32."""

# Name and layout mapping for diffusers-namespace adapters follows the conversion in minimax-h3-mlx's
# `reference/diffusers/convert_minimax_h3_to_diffusers.py` and its `lora.py` (Apache-2.0,
# https://github.com/mrbizarro/minimax-h3-mlx, revision 7919020).

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

RENAMES = (
    ("transformer_blocks.", "blocks."), ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
    ("time_embedder.linear_1", "time_embedder.proj_in"), ("time_embedder.linear_2", "time_embedder.proj_out"),
    ("audio_proj_in", "audio_patch_proj"), ("audio_proj_out", "final_layer.audio_out"),
    ("context_embedder", "condition_proj"), ("norm_out.linear", "final_layer.adaln_proj.linear"),
    ("norm_out.norm", "final_layer.norm"), ("proj_in", "video_patch_proj"), ("proj_out", "final_layer.video_out"),
    (".attn.to_out.0", ".attn.out_proj"), (".ff.net.0.proj", ".mlp.fc1"), (".ff.net.2", ".mlp.fc2"),
)


@dataclass
class Adapter:
    """Updates under checkpoint module names: ``W += sum(B A)`` per module, then additive ``diffs``."""

    name: str
    pairs: dict = field(default_factory=dict)  # module -> [(A (rank, in), B (out, rank))]
    diffs: dict = field(default_factory=dict)  # parameter -> array


def header(path) -> tuple[dict, dict]:
    """(tensors, metadata) of a safetensors file without reading its payload."""

    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        tensors = json.loads(handle.read(size))
    return tensors, tensors.pop("__metadata__", {}) or {}


def _rename(name: str) -> str:
    # `proj_in`/`proj_out` are substrings of the audio names and of each other's targets; apply each rule once,
    # anchored where it can only mean the diffusers name
    for old, new in RENAMES:
        if old.startswith(".") or old.endswith("."):
            name = name.replace(old, new)
        elif name == old or name.startswith(old + "."):
            name = new + name[len(old):]
    return name


def load(path, config) -> Adapter:
    """Read an adapter into checkpoint names and layouts.

    Two formats are read. The runner layout has ``<module>.lora_A/lora_B.weight`` under checkpoint names with the
    alpha/rank factor folded into B and fused QKV rows in contiguous ``(3, heads, head_dim)`` order. FastVideo's
    ``fastvideo-lora-v2`` uses diffusers names: separate ``to_q``/``to_k``/``to_v``, a SwiGLU projection stored
    ``[value; gate]``, alpha equal to rank, and ``.diff``/``.diff_b`` corrections. Both are mapped onto the
    checkpoint's per-head ``[q, k, v]`` rows and ``[gate; value]`` halves.
    """

    import mlx.core as mx

    path = Path(path)
    tensors, metadata = header(path)
    weights = mx.load(str(path))
    heads, dim = config.num_attention_heads, config.attention_head_dim
    adapter = Adapter(path.name)

    def add(module, a, b):
        adapter.pairs.setdefault(module, []).append((a.astype(mx.float32), b.astype(mx.float32)))

    if metadata.get("format") == "fastvideo-lora-v2":
        if int(metadata.get("set_weight_tensors", "0")):
            raise ValueError(f"{path.name} replaces weights; only low-rank and diff adapters are read")
        for key in sorted(weights):
            if key.endswith(".lora_A.weight"):
                source = key[: -len(".lora_A.weight")]
                a, b = weights[key], weights[source + ".lora_B.weight"]
                kind = source.rsplit(".", 1)[-1]
                if kind in ("to_q", "to_k", "to_v"):
                    # rows of one of q, k, v for every head, placed at that slot of each head's [q, k, v] rows
                    slot = ("to_q", "to_k", "to_v").index(kind)
                    full = mx.zeros((heads, 3, dim, b.shape[1]), dtype=mx.float32)
                    full[:, slot] = b.astype(mx.float32).reshape(heads, dim, b.shape[1])
                    add(_rename(source[: -len(kind)] + "qkv_proj"), a, full.reshape(3 * heads * dim, b.shape[1]))
                elif source.endswith(".ff.net.0.proj"):
                    half = b.shape[0] // 2
                    add(_rename(source), a, mx.concatenate([b[half:], b[:half]], axis=0))
                else:
                    add(_rename(source), a, b)
            elif key.endswith(".diff"):
                adapter.diffs[_rename(key[: -len(".diff")]) + ".weight"] = weights[key].astype(mx.float32)
            elif key.endswith(".diff_b"):
                adapter.diffs[_rename(key[: -len(".diff_b")]) + ".bias"] = weights[key].astype(mx.float32)
            elif not key.endswith(".lora_B.weight"):
                raise ValueError(f"{path.name}: unsupported tensor {key}")
        return adapter

    if any(name.endswith(".alpha") for name in tensors) or metadata.get("peft_scale_folded_into_B") != "1":
        raise ValueError(f"{path.name} is neither fastvideo-lora-v2 nor the runner layout with its scale in lora_B; "
                         "loaded at 1.0 such a file renders noise")
    if metadata.get("qkv_layout", "").split()[0] != "contiguous":
        raise ValueError(f"{path.name}: unknown qkv_layout {metadata.get('qkv_layout')!r}")
    for key in sorted(weights):
        if not key.endswith(".lora_A.weight"):
            continue
        module = key[: -len(".lora_A.weight")]
        a, b = weights[key], weights[module + ".lora_B.weight"]
        if module.endswith("qkv_proj"):
            b = b.reshape(3, heads, dim, b.shape[1]).transpose(1, 0, 2, 3).reshape(3 * heads * dim, b.shape[-1])
        add(module, a, b)
    return adapter


def _resolve(model, dotted: str):
    """(owner, attribute) for a checkpoint parameter or module name."""

    parts = dotted.split(".")
    owner = model
    for part in parts[:-1]:
        owner = owner[int(part)] if part.isdigit() else getattr(owner, part)
    return owner, parts[-1]


def merge(model, path, strength: float = 1.0) -> dict:
    """Add an adapter to the model's weights in float32 and leave the touched weights float32.

    Adapter updates are about a thousandth of the weight they ride on; rounding the sum back to bfloat16 keeps
    only 50-85% of the update and adds noise of the same size. The float32 weights are meant to be quantized by
    the int8 kernels (which keeps the update in expectation) or settled with :func:`settle`.
    """

    import mlx.core as mx

    adapter = load(path, model.config)
    changed = 0
    for module, pairs in adapter.pairs.items():
        owner, leaf = _resolve(model, module)
        linear = getattr(owner, leaf)
        weight = linear.weight.astype(mx.float32)
        for a, b in pairs:
            if (b.shape[0], a.shape[1]) != tuple(weight.shape):
                raise ValueError(f"{module}: adapter is {b.shape[0]}x{a.shape[1]}, weight is {weight.shape}")
            weight = weight + strength * (b @ a)
        mx.eval(weight)
        linear.weight = weight
        changed += 1
    for parameter, diff in adapter.diffs.items():
        owner, leaf = _resolve(model, parameter)
        current = getattr(owner, leaf)
        if tuple(current.shape) != tuple(diff.shape):
            raise ValueError(f"{parameter}: correction is {diff.shape}, parameter is {current.shape}")
        updated = current.astype(mx.float32) + strength * diff
        # vectors multiply the sequence and would lift it to float32; matrices stay float32 until settled
        setattr(owner, leaf, updated.astype(current.dtype) if updated.ndim == 1 and leaf == "weight" else updated)
        mx.eval(getattr(owner, leaf))
    mx.clear_cache()
    return {"adapter": adapter.name, "matrices": changed, "corrections": len(adapter.diffs)}


def settle(model) -> int:
    """Round any block projection still in float32 back to bfloat16; returns how many were rounded.

    This is the lossy path for projections that are not run through an int8 kernel.
    """

    import mlx.core as mx

    rounded = 0
    for block in model.blocks:
        for owner, leaf in ((block.attn, "qkv_proj"), (block.attn, "out_proj"), (block.mlp, "fc1"),
                            (block.mlp, "fc2")):
            linear = getattr(owner, leaf, None)
            weight = getattr(linear, "weight", None)
            if weight is not None and weight.dtype == mx.float32:
                linear.weight = weight.astype(mx.bfloat16)
                mx.eval(linear.weight)
                rounded += 1
    mx.clear_cache()
    return rounded
