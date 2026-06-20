"""Shared MLX adapter utility functions."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np

from smarttensor.manifest import SmartTensorManifest, TensorRecord
from smarttensor.planner import pinned_tensors
from smarttensor.safetensors import SafeTensorFile

from .records import MlxStreamEvent

def load_mlx_config(model_dir: Path) -> dict[str, Any]:
    from mlx_lm.utils import load_config

    config = load_config(model_dir)
    if "quantization_config" not in config:
        text_config = config.get("text_config", {})
        if "quantization_config" in text_config:
            config["quantization_config"] = text_config["quantization_config"]
    return config


def load_native_mlx_array(safe_file: SafeTensorFile, record: TensorRecord) -> Any:
    import mlx.core as mx

    tensor = safe_file.tensor(record.name)
    try:
        dtype = numpy_dtype(record.dtype)
        np_array = np.frombuffer(tensor.view, dtype=dtype).reshape(record.shape)
        mlx_array = mx.array(np_array)
        if record.dtype == "BF16":
            mlx_array = mlx_array.view(mx.bfloat16)
        return mlx_array
    finally:
        tensor.release()


def contiguous_runs(indices: tuple[int, ...]) -> list[tuple[int, int]]:
    """Collapse a strictly-ascending index tuple into [start, stop) runs.

    Adjacent ascending ids merge into one run, so the concatenation of
    ``arr[start:stop]`` over the runs reproduces ``arr`` restricted to those
    ids in the given order. A request for adjacent experts (or a single
    expert) yields one run -> a pure basic slice with no gather. Callers must
    pass an ascending, duplicate-free tuple (see ``gather_first_dim_rows``).
    """

    runs: list[tuple[int, int]] = []
    for index in indices:
        if runs and index == runs[-1][1]:
            start, _ = runs[-1]
            runs[-1] = (start, index + 1)
        else:
            runs.append((index, index + 1))
    return runs


def gather_first_dim_rows(np_array: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    """Extract first-dim rows, preferring contiguous basic slices over a gather.

    numpy fancy indexing (``np_array[[e0, e1, ...]]``) forces a strided copy
    that measured ~1.4-2.0x slower than a contiguous basic slice of the same
    bytes (Sprint 5 forensics; re-measured this lane). When ``indices`` is
    already strictly ascending -- which every runtime caller is, since expert
    ids arrive via ``selected_expert_ids`` (sorted unique) -- we collapse it
    into contiguous runs and extract each run as ``np_array[start:stop]``. A
    single run (the common top-k case) needs no concatenation at all.

    The returned rows are in exactly ``indices`` order, so this is a bitwise
    drop-in for ``np_array[list(indices)]``. If ``indices`` is not strictly
    ascending we fall back to the fancy-index gather to preserve that order.
    """

    is_ascending = all(
        indices[i] < indices[i + 1] for i in range(len(indices) - 1)
    )
    if not is_ascending:
        return np_array[list(indices)]
    if not indices:
        return np.ascontiguousarray(np_array[:0])

    runs = contiguous_runs(indices)
    if len(runs) == 1:
        start, stop = runs[0]
        # Basic slice: contiguous source rows, no fancy-index gather.
        return np.ascontiguousarray(np_array[start:stop])
    rows = np.empty((len(indices), *np_array.shape[1:]), dtype=np_array.dtype)
    position = 0
    for start, stop in runs:
        count = stop - start
        rows[position : position + count] = np_array[start:stop]
        position += count
    return rows


def load_native_mlx_array_first_dim_indices(
    safe_file: SafeTensorFile,
    record: TensorRecord,
    indices: tuple[int, ...],
) -> Any:
    import mlx.core as mx

    tensor = safe_file.tensor(record.name)
    try:
        dtype = numpy_dtype(record.dtype)
        np_array = np.frombuffer(tensor.view, dtype=dtype).reshape(record.shape)
        selected = gather_first_dim_rows(np_array, indices)
        mlx_array = mx.array(selected)
        if record.dtype == "BF16":
            mlx_array = mlx_array.view(mx.bfloat16)
        return mlx_array
    finally:
        tensor.release()


def numpy_dtype(dtype: str) -> np.dtype:
    mapping = {
        "BOOL": np.dtype("bool"),
        "U8": np.dtype("u1"),
        "I8": np.dtype("i1"),
        "U16": np.dtype("<u2"),
        "I16": np.dtype("<i2"),
        "U32": np.dtype("<u4"),
        "I32": np.dtype("<i4"),
        "U64": np.dtype("<u8"),
        "I64": np.dtype("<i8"),
        "F16": np.dtype("<f2"),
        "BF16": np.dtype("<u2"),
        "F32": np.dtype("<f4"),
        "F64": np.dtype("<f8"),
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported native MLX dtype: {dtype}") from exc


def compact_event(event: MlxStreamEvent) -> dict[str, Any]:
    return {
        "action": event.action,
        "layer": event.layer,
        "seconds": event.seconds,
        "resident_bytes": event.resident_bytes,
        "requested_count": len(event.requested),
        "loaded_count": len(event.loaded),
        "skipped_count": len(event.skipped),
        "evicted_count": len(event.evicted),
        "nbytes_loaded": event.nbytes_loaded,
        "transient_page_bytes": event.transient_page_bytes,
    }


def kv_cache_nbytes(cache: list[Any]) -> int:
    total = 0
    for item in cache:
        total += int(getattr(item, "nbytes", 0) or 0)
    return total


def select_retained_layers_for_budget(
    manifest: SmartTensorManifest,
    resident_budget_bytes: int,
    *,
    pin_policy: str = "all",
    warm_embeddings: bool = False,
) -> set[int]:
    """Select a prefix of layers that fits within the runtime overlap budget."""

    if resident_budget_bytes < 0:
        raise ValueError("resident_budget_bytes must be non-negative")
    if pin_policy not in {"all", "phase"}:
        raise ValueError("pin_policy must be 'all' or 'phase'")
    if warm_embeddings and pin_policy != "phase":
        raise ValueError("warm_embeddings only applies to pin_policy='phase'")

    base_bytes = pinned_budget_bytes(
        manifest,
        pin_policy=pin_policy,
        warm_embeddings=warm_embeddings,
    )
    phase_role_bytes = (
        phase_role_budget_bytes(manifest, warm_embeddings=warm_embeddings)
        if pin_policy == "phase"
        else 0
    )
    retained: set[int] = set()
    retained_bytes = 0
    layers = tuple(sorted(manifest.layers.items()))
    if resident_budget_bytes <= base_bytes:
        return retained

    for layer_index, layer in layers:
        candidate = retained | {layer_index}
        candidate_retained_bytes = retained_bytes + layer.nbytes
        streamed_layer_bytes = max(
            (candidate_layer.nbytes for candidate_index, candidate_layer in layers if candidate_index not in candidate),
            default=0,
        )
        runtime_overlap_bytes = max(streamed_layer_bytes, phase_role_bytes)
        if base_bytes + candidate_retained_bytes + runtime_overlap_bytes > resident_budget_bytes:
            break
        retained.add(layer_index)
        retained_bytes = candidate_retained_bytes
    return retained


def pinned_budget_bytes(
    manifest: SmartTensorManifest,
    *,
    pin_policy: str,
    warm_embeddings: bool = False,
) -> int:
    if pin_policy == "phase":
        total = sum(
            record.nbytes
            for record in pinned_tensors(manifest)
            if record.residency_hint == "pin-small"
        )
        if warm_embeddings:
            total += role_budget_bytes(manifest, "embedding")
        return total
    return sum(record.nbytes for record in pinned_tensors(manifest))


def phase_role_budget_bytes(
    manifest: SmartTensorManifest,
    *,
    warm_embeddings: bool = False,
) -> int:
    roles = ("output",) if warm_embeddings else ("embedding", "output")
    return max(
        (
            role_budget_bytes(manifest, role)
            for role in roles
        ),
        default=0,
    )


def role_budget_bytes(manifest: SmartTensorManifest, role: str) -> int:
    return sum(record.nbytes for record in manifest.tensors.values() if record.role == role)


def selected_expert_ids(indices: Any) -> list[int]:
    values = np.asarray(indices).reshape(-1)
    return sorted({int(value) for value in values})


def selected_expert_counts(indices: Any) -> dict[int, int]:
    values = np.asarray(indices).reshape(-1)
    counts: dict[int, int] = {}
    for value in values:
        expert = int(value)
        counts[expert] = counts.get(expert, 0) + 1
    return counts


def should_refresh_hot_set(
    current_order: list[int],
    missing: list[int],
    counts: dict[int, int],
    hysteresis: int,
) -> bool:
    """Gate hot-set membership rebuilds behind a count margin.

    A refresh copies the whole resident table (16 rows x 12 tensors), so a
    cold candidate must beat the weakest resident's frequency count by
    ``hysteresis`` before a rebuild is even considered. Without the margin,
    early noisy counts churned 142 rebuilds in 65 passes for no measured
    row-hit-rate gain.
    """

    if not current_order:
        return True
    if not missing:
        return False
    weakest = min(counts.get(expert, 0) for expert in current_order)
    strongest = max(counts.get(expert, 0) for expert in missing)
    return strongest >= weakest + hysteresis


def choose_frequency_hot_order(
    current_order: list[int],
    missing: list[int],
    counts: dict[int, int],
    cap: int,
) -> list[int]:
    if cap <= 0:
        return []

    current_set = set(current_order)
    missing_unique = [expert for expert in dict.fromkeys(missing) if expert not in current_set]
    candidates = list(current_order) + missing_unique
    current_position = {expert: index for index, expert in enumerate(current_order)}
    missing_position = {expert: index for index, expert in enumerate(missing_unique)}

    def key(expert: int) -> tuple[int, int, int]:
        return (
            -counts.get(expert, 0),
            0 if expert in current_set else 1,
            current_position.get(expert, len(current_order) + missing_position.get(expert, 0)),
        )

    return sorted(candidates, key=key)[:cap]


def remap_expert_indices(indices: Any, selected_experts: list[int]) -> Any:
    """Map global expert ids to row positions in ``selected_experts``.

    Vectorized position-lookup (callers guarantee every value appears in the
    table). Accepts an MLX array or an already-fetched numpy array, so hot
    loops can reuse one host copy of the indices.
    """

    import mlx.core as mx

    values = np.asarray(indices)
    table = np.asarray(selected_experts, dtype=np.int64)
    positions = np.zeros(int(table.max()) + 1 if table.size else 1, dtype=np.int32)
    positions[table] = np.arange(table.size, dtype=np.int32)
    return mx.array(positions[values])


def remap_expert_indices_to_slots(indices: Any, expert_to_slot: dict[int, int]) -> Any:
    """Map global expert ids to persistent arena slots."""

    import mlx.core as mx

    values = np.asarray(indices)
    if not expert_to_slot:
        return mx.array(np.zeros_like(values, dtype=np.int32))
    max_expert = max(expert_to_slot)
    positions = np.zeros(max_expert + 1, dtype=np.int32)
    for expert, slot in expert_to_slot.items():
        positions[expert] = slot
    return mx.array(positions[values])


def select_qwen_base_layers_for_budget(
    manifest: SmartTensorManifest,
    resident_budget_bytes: int,
    *,
    top_k: int,
    expert_marker: str = ".mlp.switch_mlp.",
) -> set[int]:
    if resident_budget_bytes < 0:
        raise ValueError("resident_budget_bytes must be non-negative")

    phase_role_bytes = phase_role_budget_bytes(manifest)
    pin_small_bytes = pinned_budget_bytes(manifest, pin_policy="phase")
    selected_expert_bytes = qwen_selected_expert_bytes(
        manifest, top_k=top_k, expert_marker=expert_marker
    )
    retained: set[int] = set()
    layers = tuple(sorted(manifest.layers.items()))
    if resident_budget_bytes <= phase_role_bytes + pin_small_bytes + selected_expert_bytes:
        return retained

    retained_bytes = 0
    for layer_index, layer in layers:
        base_bytes = qwen_layer_base_bytes(manifest, layer_index, expert_marker=expert_marker)
        candidate = retained | {layer_index}
        streamed_base_bytes = max(
            (
                qwen_layer_base_bytes(manifest, candidate_index, expert_marker=expert_marker)
                for candidate_index, _ in layers
                if candidate_index not in candidate
            ),
            default=0,
        )
        peak_bytes = (
            pin_small_bytes
            + phase_role_bytes
            + selected_expert_bytes
            + retained_bytes
            + base_bytes
            + streamed_base_bytes
        )
        if peak_bytes > resident_budget_bytes:
            break
        retained.add(layer_index)
        retained_bytes += base_bytes
    return retained


def reserve_weight_page_budget_for_base_retention(
    resident_budget_bytes: int,
    weight_page_budget_bytes: int | None,
) -> int:
    if resident_budget_bytes < 0:
        raise ValueError("resident_budget_bytes must be non-negative")
    if weight_page_budget_bytes is None:
        return resident_budget_bytes
    if weight_page_budget_bytes < 0:
        raise ValueError("weight_page_budget_bytes must be non-negative")
    return max(resident_budget_bytes - weight_page_budget_bytes, 0)


def qwen_layer_base_bytes(
    manifest: SmartTensorManifest,
    layer_index: int,
    *,
    expert_marker: str = ".mlp.switch_mlp.",
) -> int:
    layer = manifest.layers[layer_index]
    return sum(
        manifest.tensors[name].nbytes
        for name in layer.tensor_names
        if expert_marker not in name
    )


def qwen_selected_expert_bytes(
    manifest: SmartTensorManifest,
    *,
    top_k: int,
    expert_marker: str = ".mlp.switch_mlp.",
) -> int:
    first_layer = next(
        (
            layer
            for _, layer in sorted(manifest.layers.items())
            if any(expert_marker in name for name in layer.tensor_names)
        ),
        None,
    )
    if first_layer is None:
        return 0
    total = 0
    for name in first_layer.tensor_names:
        if expert_marker not in name:
            continue
        record = manifest.tensors[name]
        if not record.shape:
            continue
        total += record.nbytes * top_k // record.shape[0]
    return total


def build_mlx_model_shell(config: dict[str, Any], manifest: SmartTensorManifest) -> Any:
    from mlx import nn
    from mlx_lm.utils import _get_classes

    model_class, model_args_class = _get_classes(config=config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    quantization = config.get("quantization")
    if quantization is None and (quantization_config := config.get("quantization_config")):
        if all(key in quantization_config for key in ("group_size", "bits")):
            quantization = quantization_config

    if quantization is not None:
        weight_names = set(manifest.tensors)
        group_size = quantization.get("group_size")
        bits = quantization.get("bits")
        mode = quantization.get("mode", "affine")

        def class_predicate(path: str, module: Any) -> bool | dict[str, Any]:
            custom = quantization.get(path)
            if isinstance(custom, dict):
                return custom
            if not hasattr(module, "to_quantized"):
                return False
            if config.get("model_type") == "deepseek_v3" and (
                path.endswith(".self_attn.embed_q")
                or path.endswith(".self_attn.unembed_out")
            ):
                return any(
                    candidate in weight_names
                    for candidate in (
                        path.replace(".embed_q", ".kv_b_proj") + ".scales",
                        path.replace(".unembed_out", ".kv_b_proj") + ".scales",
                    )
                )
            return f"{path}.scales" in weight_names

        nn.quantize(
            model,
            group_size=group_size,
            bits=bits,
            mode=mode,
            class_predicate=class_predicate,
        )

    model.eval()
    return model


def sanitize_weights(model: Any, arrays: dict[str, Any]) -> dict[str, Any]:
    if hasattr(model, "sanitize"):
        return model.sanitize(arrays)
    return arrays


def placeholder_for_record(record: Any) -> Any:
    import mlx.core as mx

    return mx.zeros(record.shape, dtype=mlx_dtype(record.dtype))


def mlx_dtype(dtype: str) -> Any:
    import mlx.core as mx

    mapping = {
        "BOOL": mx.bool_,
        "U8": mx.uint8,
        "I8": mx.int8,
        "U16": mx.uint16,
        "I16": mx.int16,
        "U32": mx.uint32,
        "I32": mx.int32,
        "U64": mx.uint64,
        "I64": mx.int64,
        "F16": mx.float16,
        "BF16": mx.bfloat16,
        "F32": mx.float32,
        "F64": mx.float64,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported MLX dtype for placeholder: {dtype}") from exc


def clear_mlx_memory() -> None:
    gc.collect()
    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        else:
            mx.metal.clear_cache()
    except Exception:
        pass
