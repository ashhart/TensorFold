"""Static frontier-model profiling for TensorFold.

This module deliberately avoids importing MLX or loading tensor payloads. It
uses model config plus safetensors headers to answer the first question for a
too-large model: what resident expert arena sizes are even plausible?
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import statistics
from typing import Any, Iterable

from smarttensor.planner import format_bytes, parse_bytes
from smarttensor.safetensors import SafeTensorFile


_SHARD_RE = re.compile(r"model-(\d+)-of-(\d+)\.safetensors$")
_GLM_SWITCH_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.switch_mlp\."
    r"(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)


@dataclass(frozen=True)
class ShardCompleteness:
    present: int
    expected: int | None
    missing: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return self.expected is not None and not self.missing and self.present == self.expected

    def to_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "expected": self.expected,
            "missing": list(self.missing),
            "complete": self.complete,
        }


@dataclass(frozen=True)
class GlmMoEProfile:
    model_dir: str
    model_type: str
    architecture: str | None
    total_file_bytes: int
    shard_completeness: ShardCompleteness
    hidden_size: int
    moe_intermediate_size: int
    routed_experts: int
    experts_per_token: int
    routed_layer_count: int
    routed_layers: tuple[int, ...]
    switch_tensor_count: int
    observed_switch_tensor_count: int
    complete_observed_layers: int
    bytes_per_expert: int
    active_routed_bytes_per_token: int
    budget_fits: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_dir": self.model_dir,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "total_file_bytes": self.total_file_bytes,
            "total_file_human": format_bytes(self.total_file_bytes),
            "shards": self.shard_completeness.to_dict(),
            "hidden_size": self.hidden_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "routed_experts": self.routed_experts,
            "experts_per_token": self.experts_per_token,
            "routed_layer_count": self.routed_layer_count,
            "routed_layers": list(self.routed_layers),
            "switch_tensor_count": self.switch_tensor_count,
            "observed_switch_tensor_count": self.observed_switch_tensor_count,
            "complete_observed_layers": self.complete_observed_layers,
            "bytes_per_expert": self.bytes_per_expert,
            "bytes_per_expert_human": format_bytes(self.bytes_per_expert),
            "active_routed_bytes_per_token": self.active_routed_bytes_per_token,
            "active_routed_bytes_per_token_human": format_bytes(
                self.active_routed_bytes_per_token
            ),
            "budget_fits": list(self.budget_fits),
        }

    def to_human(self) -> str:
        lines = [
            "TensorFold frontier profile",
            f"model_dir: {self.model_dir}",
            f"model_type: {self.model_type}",
            f"architecture: {self.architecture or 'unknown'}",
            f"model files: {format_bytes(self.total_file_bytes)}",
            (
                "shards: "
                f"{self.shard_completeness.present}/"
                f"{self.shard_completeness.expected or '?'}"
                f" ({'complete' if self.shard_completeness.complete else 'incomplete'})"
            ),
        ]
        if self.shard_completeness.missing:
            lines.append(f"missing shards: {list(self.shard_completeness.missing)}")
        lines.extend(
            [
                f"routed layers: {self.routed_layer_count}",
                f"routed experts/layer: {self.routed_experts}",
                f"experts/token: {self.experts_per_token}",
                f"packed switch tensors: {self.switch_tensor_count}",
                f"observed switch tensors: {self.observed_switch_tensor_count}",
                f"complete observed routed layers: {self.complete_observed_layers}",
                f"bytes/expert: {format_bytes(self.bytes_per_expert)}",
                (
                    "all-cold routed bytes/token: "
                    f"{format_bytes(self.active_routed_bytes_per_token)}"
                ),
            ]
        )
        if self.budget_fits:
            lines.append("budget fits:")
            for fit in self.budget_fits:
                lines.append(
                    "  "
                    f"{fit['budget_human']}: "
                    f"K{fit['experts_per_layer']} per routed layer "
                    f"({fit['arena_bytes_human']}, "
                    f"{fit['coverage_of_topk']:.2f}x top-k)"
                )
        return "\n".join(lines)


def profile_frontier_model(
    model_dir: str | Path,
    *,
    budgets: Iterable[str | int] = (),
) -> GlmMoEProfile:
    """Profile a supported frontier MoE model without loading tensor data."""

    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = str(config.get("model_type", ""))
    if model_type != "glm_moe_dsa":
        raise ValueError(f"unsupported frontier model_type {model_type!r}")
    return _profile_glm_moe_dsa(model_dir, config, budgets=budgets)


def _profile_glm_moe_dsa(
    model_dir: Path,
    config: dict[str, Any],
    *,
    budgets: Iterable[str | int],
) -> GlmMoEProfile:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json is missing a weight_map object")

    switch_names = sorted(name for name in weight_map if _GLM_SWITCH_RE.match(name))
    layer_from_index = sorted(
        {
            int(match.group(1))
            for name in switch_names
            if (match := _GLM_SWITCH_RE.match(name)) is not None
        }
    )
    if not layer_from_index:
        raise ValueError("glm_moe_dsa model has no switch_mlp tensors in weight_map")

    metadata_by_name = _load_present_switch_metadata(model_dir, weight_map, switch_names)
    per_layer_bytes: dict[int, int] = {}
    routed_experts = int(config["n_routed_experts"])
    for name, meta in metadata_by_name.items():
        match = _GLM_SWITCH_RE.match(name)
        if match is None or not meta.shape or meta.shape[0] != routed_experts:
            continue
        per_layer_bytes[int(match.group(1))] = per_layer_bytes.get(int(match.group(1)), 0) + (
            meta.nbytes // routed_experts
        )

    complete_layers = [
        layer
        for layer in layer_from_index
        if sum(1 for name in metadata_by_name if name.startswith(f"model.layers.{layer}.mlp.switch_mlp.")) == 9
    ]
    if complete_layers:
        bytes_per_expert = int(statistics.median(per_layer_bytes[layer] for layer in complete_layers))
    else:
        bytes_per_expert = _estimate_glm_bytes_per_expert(config)

    experts_per_token = int(config["num_experts_per_tok"])
    routed_layer_count = len(layer_from_index)
    active_routed_bytes = bytes_per_expert * experts_per_token * routed_layer_count
    budget_fits = tuple(
        _budget_fit(parse_bytes(str(budget)), bytes_per_expert, routed_layer_count, experts_per_token, routed_experts)
        for budget in budgets
    )

    return GlmMoEProfile(
        model_dir=str(model_dir),
        model_type=str(config.get("model_type", "")),
        architecture=_first_string(config.get("architectures")),
        total_file_bytes=_total_file_bytes(model_dir),
        shard_completeness=_shard_completeness(model_dir, weight_map.values()),
        hidden_size=int(config["hidden_size"]),
        moe_intermediate_size=int(config["moe_intermediate_size"]),
        routed_experts=routed_experts,
        experts_per_token=experts_per_token,
        routed_layer_count=routed_layer_count,
        routed_layers=tuple(layer_from_index),
        switch_tensor_count=len(switch_names),
        observed_switch_tensor_count=len(metadata_by_name),
        complete_observed_layers=len(complete_layers),
        bytes_per_expert=bytes_per_expert,
        active_routed_bytes_per_token=active_routed_bytes,
        budget_fits=budget_fits,
    )


def _load_present_switch_metadata(model_dir: Path, weight_map: dict[str, str], switch_names: list[str]):
    names_by_file: dict[str, list[str]] = {}
    for name in switch_names:
        file_name = weight_map.get(name)
        if isinstance(file_name, str):
            names_by_file.setdefault(file_name, []).append(name)

    metadata = {}
    for file_name, names in sorted(names_by_file.items()):
        path = model_dir / file_name
        if not path.exists():
            continue
        with SafeTensorFile(path) as safe_file:
            for name in names:
                if name in safe_file.tensors:
                    metadata[name] = safe_file.tensors[name]
    return metadata


def _estimate_glm_bytes_per_expert(config: dict[str, Any]) -> int:
    hidden = int(config["hidden_size"])
    intermediate = int(config["moe_intermediate_size"])
    quant = config.get("quantization") or {}
    bits = int(quant.get("bits", 8))
    group_size = int(quant.get("group_size", 64))
    weight_bytes = (3 * hidden * intermediate * bits) // 8
    gate_up_groups = 2 * intermediate * (hidden // group_size)
    down_groups = hidden * (intermediate // group_size)
    scale_bias_bytes = (gate_up_groups + down_groups) * 2 * 2
    return weight_bytes + scale_bias_bytes


def _budget_fit(
    budget_bytes: int,
    bytes_per_expert: int,
    routed_layer_count: int,
    top_k: int,
    routed_experts: int,
) -> dict[str, Any]:
    experts_per_layer = max(0, min(routed_experts, budget_bytes // (bytes_per_expert * routed_layer_count)))
    arena_bytes = experts_per_layer * bytes_per_expert * routed_layer_count
    return {
        "budget_bytes": budget_bytes,
        "budget_human": format_bytes(budget_bytes),
        "experts_per_layer": experts_per_layer,
        "arena_bytes": arena_bytes,
        "arena_bytes_human": format_bytes(arena_bytes),
        "coverage_of_topk": experts_per_layer / top_k if top_k else 0.0,
    }


def _shard_completeness(model_dir: Path, referenced_files: Iterable[str]) -> ShardCompleteness:
    referenced = set(referenced_files)
    present_numbers = set()
    expected_numbers = set()
    for file_name in referenced | {path.name for path in model_dir.glob("model-*.safetensors")}:
        match = _SHARD_RE.match(file_name)
        if match is None:
            continue
        number = int(match.group(1))
        expected = int(match.group(2))
        expected_numbers.add(expected)
        if (model_dir / file_name).exists():
            present_numbers.add(number)

    expected = max(expected_numbers) if expected_numbers else None
    missing = tuple(
        number
        for number in range(1, expected + 1)
        if expected is not None and number not in present_numbers
    )
    return ShardCompleteness(
        present=len(present_numbers),
        expected=expected,
        missing=missing,
    )


def _total_file_bytes(model_dir: Path) -> int:
    return sum(path.stat().st_size for path in model_dir.glob("model-*.safetensors"))


def _first_string(value: object) -> str | None:
    if isinstance(value, list) and value and isinstance(value[0], str):
        return value[0]
    if isinstance(value, str):
        return value
    return None
