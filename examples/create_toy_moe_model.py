#!/usr/bin/env python3
"""Create a tiny MoE-style safetensors model directory for TensorFold demos."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import sys


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: create_toy_moe_model.py OUTPUT_DIR", file=sys.stderr)
        return 2

    output_dir = Path(argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    shard = output_dir / "model-00001-of-00001.safetensors"
    tensors = {
        "model.embed_tokens.weight": ("F32", [2, 4], byte_range(0, 32)),
        "model.layers.0.mlp.experts.gate_up_proj.weight": ("F32", [2, 4], byte_range(32, 64)),
        "model.layers.0.mlp.experts.down_proj.weight": ("F32", [2, 4], byte_range(64, 96)),
        "model.norm.weight": ("F32", [4], byte_range(96, 112)),
        "lm_head.weight": ("F32", [2, 4], byte_range(112, 144)),
    }

    offset = 0
    header = {"__metadata__": {"format": "toy-tensorfold-moe"}}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        payload.extend(data)

    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)
    print(shard)
    return 0


def byte_range(start: int, end: int) -> bytes:
    return bytes(value % 256 for value in range(start, end))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
