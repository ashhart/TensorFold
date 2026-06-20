"""Memory-budget planning for layer-wise tensor streaming."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from smarttensor.manifest import SmartTensorManifest, TensorRecord

BYTE_UNITS = {
    "b": 1,
    "kb": 1000,
    "kib": 1024,
    "mb": 1000**2,
    "mib": 1024**2,
    "gb": 1000**3,
    "gib": 1024**3,
}


@dataclass(frozen=True)
class PlanStep:
    """One high-level action in a streaming execution plan."""

    action: str
    tensor_names: tuple[str, ...]
    nbytes: int
    layer: int | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "layer": self.layer,
            "tensor_names": list(self.tensor_names),
            "nbytes": self.nbytes,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class StreamingPlan:
    """A simple exact-first layer streaming plan."""

    budget_bytes: int
    pinned_bytes: int
    largest_layer_bytes: int
    peak_estimated_bytes: int
    prefetch_window: int
    fits_budget: bool
    steps: tuple[PlanStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_bytes": self.budget_bytes,
            "pinned_bytes": self.pinned_bytes,
            "largest_layer_bytes": self.largest_layer_bytes,
            "peak_estimated_bytes": self.peak_estimated_bytes,
            "prefetch_window": self.prefetch_window,
            "fits_budget": self.fits_budget,
            "steps": [step.to_dict() for step in self.steps],
        }


def build_streaming_plan(
    manifest: SmartTensorManifest,
    *,
    budget_bytes: int,
    prefetch_window: int = 1,
) -> StreamingPlan:
    """Build a conservative layer-streaming plan from a manifest.

    The plan keeps high-value unlayered tensors resident and streams layer
    tensors in index order. Peak memory is estimated as pinned tensors plus the
    current layer and the requested number of prefetched future layers.
    """

    if budget_bytes < 0:
        raise ValueError("budget_bytes must be non-negative")
    if prefetch_window < 0:
        raise ValueError("prefetch_window must be non-negative")

    pinned = pinned_tensors(manifest)
    steps: list[PlanStep] = []
    pinned_bytes = sum(record.nbytes for record in pinned)
    if pinned:
        steps.append(
            PlanStep(
                action="pin",
                tensor_names=tuple(record.name for record in pinned),
                nbytes=pinned_bytes,
                reason="embedding/output/small normalization tensors are reused across tokens",
            )
        )

    layer_indices = sorted(manifest.layers)
    largest_layer_bytes = 0
    peak_estimated_bytes = pinned_bytes

    for position, layer_index in enumerate(layer_indices):
        layer = manifest.layers[layer_index]
        largest_layer_bytes = max(largest_layer_bytes, layer.nbytes)

        active_indices = layer_indices[position : position + prefetch_window + 1]
        active_bytes = sum(manifest.layers[index].nbytes for index in active_indices)
        peak_estimated_bytes = max(peak_estimated_bytes, pinned_bytes + active_bytes)

        steps.append(
            PlanStep(
                action="load-layer",
                layer=layer.index,
                tensor_names=layer.tensor_names,
                nbytes=layer.nbytes,
                reason="current transformer layer must be resident for exact dense execution",
            )
        )

        for prefetch_index in active_indices[1:]:
            prefetch_layer = manifest.layers[prefetch_index]
            steps.append(
                PlanStep(
                    action="prefetch-layer",
                    layer=prefetch_layer.index,
                    tensor_names=prefetch_layer.tensor_names,
                    nbytes=prefetch_layer.nbytes,
                    reason=f"prefetch window {prefetch_window}",
                )
            )

        steps.append(
            PlanStep(
                action="evict-layer",
                layer=layer.index,
                tensor_names=layer.tensor_names,
                nbytes=layer.nbytes,
                reason="layer output has advanced to the next transformer block",
            )
        )

    return StreamingPlan(
        budget_bytes=budget_bytes,
        pinned_bytes=pinned_bytes,
        largest_layer_bytes=largest_layer_bytes,
        peak_estimated_bytes=peak_estimated_bytes,
        prefetch_window=prefetch_window,
        fits_budget=peak_estimated_bytes <= budget_bytes,
        steps=tuple(steps),
    )


def pinned_tensors(manifest: SmartTensorManifest) -> list[TensorRecord]:
    return sorted(
        (
            record
            for record in manifest.tensors.values()
            if record.residency_hint.startswith("pin")
        ),
        key=lambda record: record.name,
    )


def parse_bytes(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)?\s*", value)
    if not match:
        raise ValueError(f"invalid byte value: {value!r}")

    number = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    if unit not in BYTE_UNITS:
        raise ValueError(f"unsupported byte unit {unit!r}; use B, MB, MiB, GB, or GiB")
    return int(number * BYTE_UNITS[unit])


def format_bytes(value: int) -> str:
    if value >= 1024**3:
        return f"{value / 1024**3:.2f} GiB"
    if value >= 1024**2:
        return f"{value / 1024**2:.2f} MiB"
    if value >= 1024:
        return f"{value / 1024:.2f} KiB"
    return f"{value} B"
