"""Convert an EXL3 Qwen vision sidecar once; serving loads the cached floating tower.

Usage: python -m tensorfold.vision.exl3_convert vision_k6.safetensors vision-f16.safetensors
Set TENSORFOLD_VISION_WEIGHTS to the output when starting a CUDA vision server.
The source remains unchanged. The artifact records its hash, codec and conversion version.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

VERSION = "2"


def convert_tensors(tensors, config=None):
    from tensorfold.cuda.exl3 import format as fmt
    from .qwen_checkpoint import vision_key

    local = {}
    for name, value in tensors.items():
        key = vision_key(name)
        if key is None:
            raise ValueError(f"non-vision tensor in sidecar: {name}")
        if key in local:
            raise ValueError(f"duplicate vision tensor: {key}")
        local[key] = value
    groups = {name[:-8] for name in local if name.endswith(".trellis")}
    consumed, result = set(), {}
    for group in sorted(groups):
        parts = {part: local[group + "." + part] for part in fmt.PARTS if group + "." + part in local}
        if "suh" not in parts and "su" not in parts or "svh" not in parts and "sv" not in parts:
            raise ValueError(f"missing EXL3 scales: {group}")
        codebook = "mul1" if "mul1" in parts else "mcg" if "mcg" in parts else "3inst"
        suh = parts["suh"] if "suh" in parts else fmt.unpack_signs(parts["su"])
        svh = parts["svh"] if "svh" in parts else fmt.unpack_signs(parts["sv"])
        result[group + ".weight"] = fmt.dequantize(parts["trellis"], suh, svh,
                                                   fmt.bits_of(parts["trellis"].shape), codebook).T.astype(np.float16)
        consumed.update(group + "." + part for part in parts if part != "bias")
    for name, value in local.items():
        if name in consumed:
            continue
        if name in result or np.asarray(value).dtype.kind != "f":
            raise ValueError(f"unsupported or duplicate vision tensor: {name}")
        result[name] = np.asarray(value).astype(np.float16)
    blocks = {name.split(".attn.")[0] for name in result if ".attn.q_proj." in name}
    for block in sorted(blocks):
        for part in ("weight", "bias"):
            names = [f"{block}.attn.{proj}_proj.{part}" for proj in ("q", "k", "v")]
            if any(name not in result for name in names):
                raise ValueError(f"incomplete split QKV: {block}")
            combined = f"{block}.attn.qkv.{part}"
            # Some packs retain the original fused float QKV beside quantized split projections.
            # ExLlamaV3 loads the split projections; reconstruct those instead of the stale fused copy.
            result[combined] = np.concatenate([result.pop(name) for name in names], axis=0)
    if config is not None:
        hidden, mid = config["hidden_size"], config["intermediate_size"]
        padded = ((mid + 127) // 128) * 128
        for layer in range(config["depth"]):
            prefix = f"blocks.{layer}.mlp."
            fc1, bias, fc2 = (result[prefix + part] for part in
                             ("linear_fc1.weight", "linear_fc1.bias", "linear_fc2.weight"))
            if (fc1.shape not in ((mid, hidden), (padded, hidden))
                    or bias.shape != (fc1.shape[0],) or fc2.shape != (hidden, fc1.shape[0])):
                raise ValueError(f"unexpected EXL3 vision MLP padding: {prefix}")
            result[prefix + "linear_fc1.weight"] = fc1[:mid]
            result[prefix + "linear_fc1.bias"] = bias[:mid]
            result[prefix + "linear_fc2.weight"] = fc2[:, :mid]
    return {"vision_tower." + name: np.ascontiguousarray(value) for name, value in result.items()}


def convert(source: Path, output: Path):
    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file

    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("output must differ from the immutable source")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    config_path = source.parent / "config.json"
    config_raw = config_path.read_bytes() if config_path.exists() else b""
    config = json.loads(config_raw).get("vision_config") if config_raw else None
    metadata = {"tensorfold_converter": VERSION, "source_sha256": digest.hexdigest(), "dtype": "F16",
                "config_sha256": hashlib.sha256(config_raw).hexdigest()}
    if output.exists():
        with safe_open(str(output), framework="np") as existing:
            if existing.metadata() != metadata:
                raise ValueError("existing artifact belongs to a different source or converter; choose a new output")
        return output
    tensors = convert_tensors(load_file(str(source)), config)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp-" + str(os.getpid()))
    try:
        save_file(tensors, str(temporary), metadata=metadata)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(convert(args.source, args.output))


if __name__ == "__main__":
    main()
