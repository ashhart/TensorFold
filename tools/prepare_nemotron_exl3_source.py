#!/usr/bin/env python3
"""Create a non-mutating EXL3-compatible view of the pinned BF16 source.

ExLlamaV3 v1.5.3 reads hybrid_override_pattern while NVIDIA's original
config supplies the equivalent layers_block_type list with older HF names.
This view keeps all original files untouched, adds the two equivalent pattern
strings, and translates layer-type aliases for current Transformers tokenizers.
"""
import argparse
import hashlib
import json
from pathlib import Path

MAPPING = {"mamba": "M", "moe": "E", "attention": "*"}
HF_TYPE_ALIAS = {"mamba": "linear_attention", "moe": "moe", "attention": "full_attention"}
CHECKS = {
    "mamba": ("mixer.in_proj.weight", "mixer.conv1d.weight"),
    "moe": ("mixer.experts.0.up_proj.weight", "mixer.experts.0.down_proj.weight"),
    "attention": ("mixer.q_proj.weight", "mixer.k_proj.weight"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--require-weights", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    overlay = args.overlay.absolute()
    if overlay == source or source in overlay.parents or overlay in source.parents:
        raise ValueError("Source and overlay must be separate directories")
    original_bytes = (source / "config.json").read_bytes()
    original = json.loads(original_bytes)
    if original.get("architectures") != ["NemotronHForCausalLM"] or original.get("model_type") != "nemotron_h":
        raise ValueError("Unexpected source architecture")
    if original.get("dtype") != "bfloat16" or "quantization_config" in original:
        raise ValueError("Source must be unquantized BF16")
    blocks = original["layers_block_type"]
    if len(blocks) != original["num_hidden_layers"] or set(blocks) - MAPPING.keys():
        raise ValueError("Unexpected trunk block layout")
    trunk = "".join(MAPPING[b] for b in blocks)
    mtp_blocks = original.get("mtp_layers_block_type", [])
    if original.get("num_nextn_predict_layers") != 1 or mtp_blocks != ["attention", "moe"]:
        raise ValueError("Unvalidated MTP layout")
    mtp = "".join(MAPPING[b] for b in mtp_blocks)
    for key, value in (("hybrid_override_pattern", trunk), ("mtp_hybrid_override_pattern", mtp)):
        if key in original and original[key] != value:
            raise ValueError(f"Conflicting {key}")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weights = index["weight_map"]
    for layer, kind in enumerate(blocks):
        for suffix in CHECKS[kind]:
            name = f"backbone.layers.{layer}.{suffix}"
            if name not in weights:
                raise ValueError(f"Missing expected tensor {name}")
    for name in ("mtp.layers.0.eh_proj.weight", "mtp.layers.0.mixer.q_proj.weight",
                 "mtp.layers.1.mixer.experts.0.up_proj.weight",
                 "mtp.layers.1.final_layernorm.weight"):
        if name not in weights:
            raise ValueError(f"Missing expected MTP tensor {name}")
    shard_names = set(weights.values())
    if len(shard_names) != 14 or any(not n.startswith("model-") or not n.endswith(".safetensors") for n in shard_names):
        raise ValueError("Unexpected source shard map")
    if args.require_weights:
        missing = [n for n in shard_names if not (source / n).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} source shards: {missing}")
    legacy = original.copy()
    legacy["hybrid_override_pattern"] = trunk
    legacy["mtp_hybrid_override_pattern"] = mtp
    adapted = legacy.copy()
    # Current Transformers validates the newer names when loading the tokenizer.
    # These aliases are semantically identical; ExLlamaV3 reads the pattern above.
    adapted["layers_block_type"] = [HF_TYPE_ALIAS[b] for b in blocks]
    adapted["mtp_layers_block_type"] = [HF_TYPE_ALIAS[b] for b in mtp_blocks]
    overlay.mkdir(parents=True, exist_ok=True)
    config_out = overlay / "config.json"
    payload = (json.dumps(adapted, indent=2, ensure_ascii=False) + "\n").encode()
    if config_out.exists():
        if config_out.read_bytes() != payload:
            prior_payload = (json.dumps(legacy, indent=2, ensure_ascii=False) + "\n").encode()
            if config_out.read_bytes() != prior_payload:
                raise FileExistsError(f"Refusing to replace unexpected overlay config: {config_out}")
            config_out.write_bytes(payload)  # migrate our own precisely matched first-pass view
    else:
        config_out.write_bytes(payload)
    linked = []
    for item in source.iterdir():
        if not item.is_file() or item.name == "config.json":
            continue
        target = overlay / item.name
        if target.is_symlink():
            if target.resolve() != item.resolve():
                raise FileExistsError(f"Unexpected symlink at {target}")
        elif target.exists():
            raise FileExistsError(f"Unexpected file at {target}")
        else:
            target.symlink_to(item)
            linked.append(item.name)
    print(json.dumps({
        "source": str(source), "overlay": str(overlay),
        "source_config_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "overlay_config_sha256": hashlib.sha256(payload).hexdigest(),
        "trunk_pattern": trunk, "mtp_pattern": mtp,
        "source_shards_present": sum((source / n).is_file() for n in shard_names),
        "source_shards_expected": len(shard_names),
        "linked_now": linked,
    }, indent=2))


if __name__ == "__main__":
    main()
