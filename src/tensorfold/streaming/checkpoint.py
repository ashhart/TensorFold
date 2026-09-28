"""A checkpoint's safetensors headers, read without MLX: where each tensor's bytes start and how many there are."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Callable


def headers(model_dir: Path) -> dict[Path, tuple[int, dict]]:
    """Each shard's (first data byte, header) for the model*.safetensors files of ``model_dir``."""

    out: dict[Path, tuple[int, dict]] = {}
    for path in sorted(Path(model_dir).glob("model*.safetensors")):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            out[path] = (8 + n, json.loads(f.read(n)))
    return out


def tensor_bytes(model_dir: Path, keep: Callable[[str], bool]) -> int:
    """Bytes of the tensors whose names ``keep`` accepts."""

    return sum(entry["data_offsets"][1] - entry["data_offsets"][0]
               for _, header in headers(model_dir).values() for name, entry in header.items()
               if name != "__metadata__" and keep(name))


__all__ = ["headers", "tensor_bytes"]
