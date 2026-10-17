"""Usage: convert-lora.py LORA OUT H3_MODEL_DIR: an H3 LoRA (ComfyUI, kohya, PEFT) in h3_generate.py --lora's layout."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import lora_layout  # noqa: E402

PAIRS = ((".lora_A.weight", ".lora_B.weight"), (".lora_down.weight", ".lora_up.weight"),
         (".lora_A.default.weight", ".lora_B.default.weight"))
SECOND = tuple(pair[1] for pair in PAIRS)


def main(source: str, target: str, model_dir: str) -> None:
    with open(source, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        metadata = json.loads(handle.read(size)).get("__metadata__") or {}
    # FL2VA and Ref2VA are the same transformer in two trainings: either partition's config gives the head layout
    found = sorted(Path(model_dir).glob("FL2VA/transformer/config.json")) + sorted(Path(model_dir).glob("Ref2VA/transformer/config.json"))
    if not found:
        raise SystemExit(f"no MiniMax H3 transformer config under {model_dir} (FL2VA or Ref2VA)")
    config = json.loads(found[0].read_text())
    heads, dim = int(config["num_attention_heads"]), int(config["attention_head_dim"])
    weights = mx.load(source)
    out, ranks, scales = {}, set(), set()
    for key in sorted(weights):
        pair = next((p for p in PAIRS if key.endswith(p[0])), None)
        if pair is None:
            if key.endswith(SECOND + (".alpha",)):
                continue
            raise SystemExit(f"{source}: unsupported tensor {key} (a low-rank LoRA with lora_A/lora_B is read)")
        stem = key[: -len(pair[0])]
        module = stem[len("diffusion_model."):] if stem.startswith("diffusion_model.") else stem
        if module.startswith("lora_unet_"):
            module = lora_layout.kohya_module(module[len("lora_unet_"):]) or module
            if module.startswith("lora_unet_"):
                raise SystemExit(f"{source}: {module} is not a layer of MiniMax H3's transformer")
        a, b = weights[key], weights[stem + pair[1]].astype(mx.float32)
        rank = a.shape[0]
        if a.ndim != 2 or b.ndim != 2:
            raise SystemExit(f"{source}: {key} is not a linear layer's update")
        alpha = float(weights[stem + ".alpha"].item()) if stem + ".alpha" in weights else float(metadata.get("ss_network_alpha") or rank)
        if module.endswith("qkv_proj"):
            if b.shape[0] != 3 * heads * dim:
                raise SystemExit(f"{source}: {module} has {b.shape[0]} rows, the model's QKV has {3 * heads * dim}")
            b = b.reshape(heads, 3, dim, rank).transpose(1, 0, 2, 3).reshape(3 * heads * dim, rank)
        out[module + ".lora_A.weight"] = a.astype(mx.bfloat16)
        out[module + ".lora_B.weight"] = (b * (alpha / rank)).astype(mx.bfloat16)
        ranks.add(rank)
        scales.add(alpha / rank)
    if not out:
        raise SystemExit(f"{source}: no lora_A/lora_B tensors found")
    mx.save_safetensors(target, out, metadata={
        "converted_from": Path(source).name, "source_namespace": "checkpoint names",
        "source_rank": ",".join(map(str, sorted(ranks))), "peft_scale_folded_into_B": "1",
        "qkv_layout": "contiguous [q_all;k_all;v_all]; loader interleaves exactly once", "fc1_layout": "[gate;value]",
        **{k: str(v) for k, v in metadata.items() if k.startswith("modelspec.") or k == "notice"}})
    print(f"wrote {target}: {len(out) // 2} modules, rank {sorted(ranks)}, scale {sorted(scales)} in lora_B")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
