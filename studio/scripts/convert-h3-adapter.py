"""Usage: convert-h3-adapter.py ADAPTER OUT: a LightX2V H3 Turbo adapter (diffusers PEFT) in --lora's layout."""

from __future__ import annotations

import json
import struct
import sys

import mlx.core as mx

RENAMES = (("transformer_blocks.", "blocks."), ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
           (".attn.to_out.0", ".attn.out_proj"), (".ff.net.0.proj", ".mlp.fc1"), (".ff.net.2", ".mlp.fc2"))
A, B = ".lora_A.default.weight", ".lora_B.default.weight"


def main(source: str, target: str) -> None:
    with open(source, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        metadata = json.loads(handle.read(size)).get("__metadata__") or {}
    weights = mx.load(source)
    modules = sorted(key[: -len(A)] for key in weights if key.endswith(A))
    if len(modules) * 2 != len(weights) or "alpha" not in metadata:
        raise SystemExit(f"{source}: not a diffusers PEFT adapter with its alpha in the metadata")
    alpha = float(metadata["alpha"])
    out, ranks = {}, set()

    def rename(name: str) -> str:
        for old, new in RENAMES:
            name = name.replace(old, new)
        return name

    def put(module: str, a, b, rank: int) -> None:
        out[module + ".lora_A.weight"] = a.astype(mx.bfloat16)
        out[module + ".lora_B.weight"] = (b.astype(mx.float32) * (alpha / rank)).astype(mx.bfloat16)

    for module in modules:
        a, b = weights[module + A], weights[module + B]
        rank = a.shape[0]
        ranks.add(rank)
        kind = module.rsplit(".", 1)[-1]
        if kind in ("to_k", "to_v"):
            continue  # fused with to_q below
        if kind == "to_q":
            stem = module[: -len("to_q")]
            parts = [(weights[stem + name + A], weights[stem + name + B]) for name in ("to_q", "to_k", "to_v")]
            rows = parts[0][1].shape[0]
            fused = mx.zeros((3 * rows, 3 * rank), dtype=mx.float32)
            for slot, (_, part) in enumerate(parts):
                fused[slot * rows:(slot + 1) * rows, slot * rank:(slot + 1) * rank] = part.astype(mx.float32)
            put(rename(stem + "qkv_proj"), mx.concatenate([part for part, _ in parts], axis=0), fused, rank)
        elif module.endswith(".ff.net.0.proj"):
            half = b.shape[0] // 2
            put(rename(module), a, mx.concatenate([b[half:], b[:half]], axis=0), rank)
        else:
            put(rename(module), a, b, rank)
    if len(ranks) != 1:
        raise SystemExit(f"{source}: mixed ranks {sorted(ranks)}")
    rank = ranks.pop()
    mx.save_safetensors(target, out, metadata={
        "converted_from": source.rsplit("/", 1)[-1], "source_namespace": "diffusers PEFT split q/k/v",
        "source_rank": str(rank), "lora_alpha": str(alpha), "peft_scale_folded_into_B": "1",
        "qkv_fused_rank": str(3 * rank), "qkv_layout": "contiguous [q_all;k_all;v_all]; loader interleaves exactly once",
        "fc1_layout": "[gate;value]"})
    print(f"wrote {target}: {len(out) // 2} modules, rank {rank}, scale {alpha / rank:g} in lora_B")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
