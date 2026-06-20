"""Small no-model demo fixtures for TensorFold Runtime."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct


@dataclass(frozen=True)
class DemoPaths:
    root: Path
    toy_safetensors: Path
    toy_moe_dir: Path
    toy_moe_shard: Path


def create_demo(output_dir: Path, *, force: bool = False) -> DemoPaths:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"{output_dir} already exists and is not empty; use --force to overwrite")

    output_dir.mkdir(parents=True, exist_ok=True)
    toy_safetensors = output_dir / "toy.safetensors"
    toy_moe_dir = output_dir / "toy-moe"
    toy_moe_dir.mkdir(parents=True, exist_ok=True)
    toy_moe_shard = toy_moe_dir / "model-00001-of-00001.safetensors"

    _write_safetensors(
        toy_safetensors,
        {
            "model.embed_tokens.weight": ("F32", [2, 4], _byte_range(0, 32)),
            "model.layers.0.self_attn.q_proj.weight": ("F32", [4, 4], _byte_range(32, 96)),
            "model.layers.0.mlp.up_proj.weight": ("F32", [4, 4], _byte_range(96, 160)),
            "model.layers.1.self_attn.q_proj.weight": ("F32", [4, 4], _byte_range(160, 224)),
            "model.norm.weight": ("F32", [4], _byte_range(224, 240)),
            "lm_head.weight": ("F32", [2, 4], _byte_range(240, 272)),
        },
        metadata_format="toy-tensorfold",
    )
    _write_safetensors(
        toy_moe_shard,
        {
            "model.embed_tokens.weight": ("F32", [2, 4], _byte_range(0, 32)),
            "model.layers.0.mlp.experts.gate_up_proj.weight": ("F32", [2, 4], _byte_range(32, 64)),
            "model.layers.0.mlp.experts.down_proj.weight": ("F32", [2, 4], _byte_range(64, 96)),
            "model.norm.weight": ("F32", [4], _byte_range(96, 112)),
            "lm_head.weight": ("F32", [2, 4], _byte_range(112, 144)),
        },
        metadata_format="toy-tensorfold-moe",
    )
    return DemoPaths(
        root=output_dir,
        toy_safetensors=toy_safetensors,
        toy_moe_dir=toy_moe_dir,
        toy_moe_shard=toy_moe_shard,
    )


def _write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, list[int], bytes]],
    *,
    metadata_format: str,
) -> None:
    offset = 0
    header: dict[str, object] = {"__metadata__": {"format": metadata_format}}
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
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)


def _byte_range(start: int, end: int) -> bytes:
    return bytes(value % 256 for value in range(start, end))
