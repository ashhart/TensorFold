#!/usr/bin/env python3
"""Create a tiny safetensors file for TensorFold demos and tests."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import sys


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: create_toy_safetensors.py OUTPUT.safetensors", file=sys.stderr)
        return 2

    output = Path(argv[1])
    tensors = {
        "model.embed_tokens.weight": ("F32", [2, 4], byte_range(0, 32)),
        "model.layers.0.self_attn.q_proj.weight": ("F32", [4, 4], byte_range(32, 96)),
        "model.layers.0.mlp.up_proj.weight": ("F32", [4, 4], byte_range(96, 160)),
        "model.layers.1.self_attn.q_proj.weight": ("F32", [4, 4], byte_range(160, 224)),
        "model.norm.weight": ("F32", [4], byte_range(224, 240)),
        "lm_head.weight": ("F32", [2, 4], byte_range(240, 272)),
    }

    offset = 0
    header = {"__metadata__": {"format": "toy-tensorfold"}}
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
    output.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)
    print(output)
    return 0


def byte_range(start: int, end: int) -> bytes:
    return bytes(value % 256 for value in range(start, end))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
