"""Layer-aware SmartTensor manifests."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from smarttensor.safetensors import SafeTensorFile

LAYER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|\.)model\.layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)transformer\.h\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)gpt_neox\.layers\.(\d+)(?:\.|$)"),
    re.compile(r"^layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)blocks\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)blk\.(\d+)(?:\.|$)"),
)

SIDE_TOWER_PREFIXES = (
    "audio_tower.layers.",
    "vision_tower.blocks.",
    "vision_tower.layers.",
    "multi_modal_projector.",
)


@dataclass(frozen=True)
class TensorRecord:
    """Portable tensor metadata used by the runtime planner."""

    name: str
    file: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    absolute_offsets: tuple[int, int]
    nbytes: int
    layer: int | None = None
    role: str | None = None
    residency_hint: str = "stream"

    @property
    def element_count(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "file": self.file,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "data_offsets": list(self.data_offsets),
            "absolute_offsets": list(self.absolute_offsets),
            "nbytes": self.nbytes,
            "layer": self.layer,
            "role": self.role,
            "residency_hint": self.residency_hint,
        }


@dataclass(frozen=True)
class LayerRecord:
    """Summary of tensors assigned to one transformer layer."""

    index: int
    tensor_names: tuple[str, ...]
    nbytes: int
    dtypes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "tensor_names": list(self.tensor_names),
            "nbytes": self.nbytes,
            "dtypes": list(self.dtypes),
        }


@dataclass
class SmartTensorManifest:
    """A model-level index for tensors across one or more safetensors shards."""

    format: str
    files: tuple[str, ...]
    tensors: dict[str, TensorRecord]
    layers: dict[int, LayerRecord]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_safetensors(cls, paths: list[str | Path]) -> "SmartTensorManifest":
        if not paths:
            raise ValueError("at least one safetensors path is required")

        tensor_records: dict[str, TensorRecord] = {}
        files: list[str] = []
        metadata: dict[str, Any] = {"source_format": "safetensors", "file_metadata": {}}

        for raw_path in paths:
            path = Path(raw_path)
            with SafeTensorFile(path) as safe_file:
                files.append(str(path))
                metadata["file_metadata"][str(path)] = safe_file.user_metadata
                for tensor in safe_file:
                    if tensor.name in tensor_records:
                        raise ValueError(f"duplicate tensor name across shards: {tensor.name}")

                    layer = infer_layer_index(tensor.name)
                    record = TensorRecord(
                        name=tensor.name,
                        file=str(path),
                        dtype=tensor.dtype,
                        shape=tensor.shape,
                        data_offsets=tensor.data_offsets,
                        absolute_offsets=tensor.absolute_offsets,
                        nbytes=tensor.nbytes,
                        layer=layer,
                        role=infer_tensor_role(tensor.name),
                        residency_hint=infer_residency_hint(tensor.name, layer),
                    )
                    tensor_records[tensor.name] = record

        layers = build_layers(tensor_records)
        return cls(
            format="safetensors",
            files=tuple(files),
            tensors=tensor_records,
            layers=layers,
            metadata=metadata,
        )

    @property
    def total_bytes(self) -> int:
        return sum(record.nbytes for record in self.tensors.values())

    def bytes_by_dtype(self) -> dict[str, int]:
        totals: dict[str, int] = defaultdict(int)
        for record in self.tensors.values():
            totals[record.dtype] += record.nbytes
        return dict(sorted(totals.items()))

    def unlayered_tensors(self) -> list[str]:
        return sorted(name for name, record in self.tensors.items() if record.layer is None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "files": list(self.files),
            "total_bytes": self.total_bytes,
            "tensor_count": len(self.tensors),
            "layer_count": len(self.layers),
            "bytes_by_dtype": self.bytes_by_dtype(),
            "metadata": self.metadata,
            "layers": [layer.to_dict() for _, layer in sorted(self.layers.items())],
            "tensors": [
                record.to_dict() for _, record in sorted(self.tensors.items(), key=lambda item: item[0])
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def build_layers(tensors: dict[str, TensorRecord]) -> dict[int, LayerRecord]:
    grouped: dict[int, list[TensorRecord]] = defaultdict(list)
    for record in tensors.values():
        if record.layer is not None:
            grouped[record.layer].append(record)

    layers: dict[int, LayerRecord] = {}
    for index, records in grouped.items():
        records = sorted(records, key=lambda record: record.name)
        layers[index] = LayerRecord(
            index=index,
            tensor_names=tuple(record.name for record in records),
            nbytes=sum(record.nbytes for record in records),
            dtypes=tuple(sorted({record.dtype for record in records})),
        )
    return dict(sorted(layers.items()))


def infer_layer_index(tensor_name: str) -> int | None:
    if tensor_name.startswith(SIDE_TOWER_PREFIXES):
        return None
    for pattern in LAYER_PATTERNS:
        match = pattern.search(tensor_name)
        if match:
            return int(match.group(1))
    return None


def infer_tensor_role(tensor_name: str) -> str:
    lowered = tensor_name.lower()
    if "embed" in lowered:
        return "embedding"
    if lowered.startswith("lm_head.") or ".lm_head." in lowered:
        return "output"
    if "self_attn" in lowered or ".attn" in lowered or "attention" in lowered:
        return "attention"
    if "mlp" in lowered or "feed_forward" in lowered or "ffn" in lowered:
        return "mlp"
    if "norm" in lowered:
        return "norm"
    return "unknown"


def infer_residency_hint(tensor_name: str, layer: int | None) -> str:
    role = infer_tensor_role(tensor_name)
    if layer is None and role in {"embedding", "output"}:
        return "pin"
    if role == "norm":
        return "pin-small"
    return "stream"
