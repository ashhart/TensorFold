"""Read only a local checkpoint's vision tensors, including shards shared with language weights."""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np

PREFIXES = ("model.language_model.visual.", "model.visual.", "vision_tower.", "visual.")
DTYPES = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "BF16": "<u2", "I64": "<i8", "I32": "<i4",
          "I16": "<i2", "I8": "i1", "U64": "<u8", "U32": "<u4", "U16": "<u2", "U8": "u1", "BOOL": "?"}


def vision_key(name: str) -> str | None:
    for prefix in PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return None


def _header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as stream:
        size = stream.read(8)
        if len(size) != 8:
            raise ValueError(f"Incomplete safetensors header: {path.name}")
        length = struct.unpack("<Q", size)[0]
        if not 2 <= length <= min(64 * 1024**2, path.stat().st_size - 8):
            raise ValueError(f"Invalid safetensors header length: {path.name}")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header: {path.name}")
    return header, 8 + length


def vision_tensors(model_dir: Path) -> dict[str, tuple[Path, dict, int]]:
    """Inspect headers only and return local tower names with their file, tensor metadata and data start."""
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    expected = None
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
        expected = {name: shard for name, shard in mapping.items() if vision_key(name) is not None}
        shards = sorted(set(expected.values()))
        if any(Path(p).is_absolute() or ".." in Path(p).parts for p in shards):
            raise ValueError("Vision checkpoint index contains an invalid shard path")
        files = [model_dir / shard for shard in shards]
    else:
        files = sorted(model_dir.glob("*.safetensors"))
    result, seen = {}, set()
    for path in files:
        header, begin = _header(path)
        for name, item in header.items():
            local = vision_key(name)
            if local is None or "position_ids" in local:
                continue
            if expected is not None and name not in expected:
                continue
            if local in result:
                raise ValueError(f"Duplicate vision tensor: {local}")
            result[local] = (path, item, begin)
            seen.add(name)
    missing = set(expected or ()) - seen
    missing = {name for name in missing if "position_ids" not in name}
    if missing:
        raise ValueError(f"Vision checkpoint is missing indexed tensors: {sorted(missing)[:3]}")
    if not result:
        raise ValueError("This local checkpoint has no vision tower weights; use a complete multimodal checkpoint")
    return result


def load_vision_weights(tensors: dict[str, tuple[Path, dict, int]], mx: Any) -> dict[str, Any]:
    """Read selected byte ranges rather than materializing the language tensors in mixed shards."""
    weights = {}
    for name, (path, item, begin) in tensors.items():
        dtype = item.get("dtype")
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported vision tensor dtype {dtype}: {name}")
        shape = item.get("shape", ())
        if any(not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError(f"Invalid vision tensor shape: {name}")
        offsets = item.get("data_offsets", ())
        if len(offsets) != 2 or any(not isinstance(n, int) for n in offsets):
            raise ValueError(f"Invalid vision tensor offsets: {name}")
        start, end = offsets
        dt = np.dtype(DTYPES[dtype])
        if start < 0 or end - start != math.prod(shape) * dt.itemsize or begin + end > path.stat().st_size:
            raise ValueError(f"Invalid vision tensor range: {name}")
        with path.open("rb") as stream:
            stream.seek(begin + start)
            raw = stream.read(end - start)
        if len(raw) != end - start:
            raise ValueError(f"Incomplete vision tensor: {name}")
        array = mx.array(np.frombuffer(raw, dtype=dt).reshape(shape).copy())
        weights[name] = array.view(mx.bfloat16) if dtype == "BF16" else array
    return weights


def quantization_predicate(config: dict, weights: dict[str, Any]):
    """Respect per-module overrides only where the checkpoint actually contains packed tensors."""
    quant = config.get("quantization") or config.get("quantization_config") or {}
    overrides = {vision_key(name): value for name, value in quant.items() if vision_key(name) is not None}

    def predicate(path: str, module: Any):
        if f"{path}.scales" not in weights:
            return False
        if not hasattr(module, "to_quantized"):
            raise ValueError(f"Vision module cannot load quantized weights: {path}")
        value = overrides.get(path)
        if value is False:
            raise ValueError(f"Vision quantization metadata contradicts packed weights: {path}")
        settings = {key: quant[key] for key in ("bits", "group_size", "mode") if key in quant}
        if isinstance(value, dict):
            settings.update({key: value[key] for key in ("bits", "group_size", "mode") if key in value})
        if "bits" not in settings or "group_size" not in settings:
            raise ValueError(f"Vision quantization metadata is missing bits or group_size: {path}")
        settings.setdefault("mode", "affine")
        return settings

    return predicate
