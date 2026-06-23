"""Production-facing Qwen layer-union scheduling helpers.

These helpers are intentionally pure Python. The measured fast diagnostic path
uses one layer-union expert slab at a time with bounded lookahead; production
needs the same planning semantics without importing the benchmark harness.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _normalize_block_range(block: Any) -> tuple[int, int]:
    if isinstance(block, dict):
        start = int(block["start"])
        count = int(block["count"])
    else:
        start = int(block[0])
        count = int(block[1])
    if start < 0:
        raise ValueError("block start must be non-negative")
    if count < 1:
        raise ValueError("block count must be positive")
    return start, count


def _chunk_token_steps(chunk: dict[str, Any]) -> list[int]:
    token_steps = chunk.get("token_steps")
    if token_steps is None:
        start = chunk.get("start")
        count = chunk.get("count")
        if start is None or count is None:
            return []
        return list(range(int(start), int(start) + int(count)))
    return [int(token_step) for token_step in token_steps]


def _chunk_experts(chunk: dict[str, Any], *, expert_key: str) -> list[int]:
    experts = chunk.get(expert_key)
    if experts is None:
        experts = chunk.get("experts")
    if experts is None:
        experts = chunk.get("group_experts")
    return [int(expert) for expert in experts or []]


def build_block_scoped_route_union_preload_plan_from_layer_chunks(
    layer_results: list[dict[str, Any]],
    *,
    blocks: Iterable[Any],
    expert_key: str = "group_experts",
) -> dict[str, Any]:
    """Build conservative verifier-block preload plans from layer chunk telemetry.

    Diagnostic layer-union artifacts currently expose chunk-level route unions
    rather than exact per-token expert sets. A production verifier block should
    therefore preload every chunk union that overlaps the block. That can over-
    read slightly, but it avoids under-planning from the available trace.
    """

    normalized_blocks = [_normalize_block_range(block) for block in blocks]
    block_items: list[dict[str, Any]] = []
    total_rows = 0
    max_block_rows = 0
    for start, count in normalized_blocks:
        stop = start + count
        resident_per_layer: dict[str, list[int]] = {}
        rows = 0
        for layer_item in layer_results:
            if "layer" not in layer_item:
                continue
            layer_key = str(int(layer_item["layer"]))
            experts_for_layer: set[int] = set()
            for chunk in layer_item.get("windowed_chunks") or []:
                token_steps = _chunk_token_steps(chunk)
                if not token_steps:
                    continue
                if not any(start <= token_step < stop for token_step in token_steps):
                    continue
                experts_for_layer.update(_chunk_experts(chunk, expert_key=expert_key))
            if not experts_for_layer:
                continue
            experts = sorted(experts_for_layer)
            resident_per_layer[layer_key] = experts
            rows += len(experts)
        total_rows += rows
        max_block_rows = max(max_block_rows, rows)
        block_items.append(
            {
                "start": start,
                "count": count,
                "rows": rows,
                "resident_per_layer": resident_per_layer,
            }
        )
    return {
        "format": "qwen_route_union_preload_blocks_v1",
        "source": "layer_union_windowed_chunks",
        "conservative_from_chunk_groups": True,
        "expert_key": expert_key,
        "block_count": len(block_items),
        "total_rows": total_rows,
        "max_block_rows": max_block_rows,
        "blocks": block_items,
    }


def build_layer_overlap_read_groups(
    records: list[Any],
    *,
    batch_chunk_size: int,
    window_read_chunks: int,
) -> list[list[list[Any]]]:
    """Split token records into bounded read groups for layer-overlap execution."""

    if batch_chunk_size < 1:
        raise ValueError("batch_chunk_size must be positive")
    if window_read_chunks < 1:
        raise ValueError("window_read_chunks must be positive")
    window_chunks = [
        records[offset : offset + batch_chunk_size]
        for offset in range(0, len(records), batch_chunk_size)
    ]
    return [
        window_chunks[offset : offset + window_read_chunks]
        for offset in range(0, len(window_chunks), window_read_chunks)
    ]


def build_layer_window_route_union_schedule(
    layer_chunks: list[dict[str, Any]],
    *,
    window_read_chunks: int,
    bytes_per_expert_row: int | dict[int | str, int] = 0,
) -> list[dict[str, Any]]:
    """Build coalesced layer-window raw-load groups from routed chunk unions.

    The fast hard-quarter diagnostic executor wins by loading one bounded raw
    expert batch per layer-window group, not one tiny adopted batch per routed
    chunk. This helper is the production-facing description of that load unit.
    """

    if window_read_chunks < 1:
        raise ValueError("window_read_chunks must be positive")

    def bytes_per_row_for(layer_index: int) -> int:
        if isinstance(bytes_per_expert_row, dict):
            value = bytes_per_expert_row.get(layer_index)
            if value is None:
                value = bytes_per_expert_row.get(str(layer_index), 0)
            return int(value or 0)
        return int(bytes_per_expert_row or 0)

    schedule: list[dict[str, Any]] = []
    sorted_layers = sorted(layer_chunks, key=lambda item: int(item["layer"]))
    for layer_item in sorted_layers:
        layer_index = int(layer_item["layer"])
        chunks = list(layer_item.get("chunks") or layer_item.get("windowed_chunks") or [])
        per_row = bytes_per_row_for(layer_index)
        groups: list[dict[str, Any]] = []
        for read_group, offset in enumerate(range(0, len(chunks), window_read_chunks)):
            chunk_group = chunks[offset : offset + window_read_chunks]
            token_steps = sorted(
                {
                    token_step
                    for chunk in chunk_group
                    for token_step in _chunk_token_steps(chunk)
                }
            )
            experts = sorted(
                {
                    expert
                    for chunk in chunk_group
                    for expert in _chunk_experts(chunk, expert_key="group_experts")
                }
            )
            rows = len(experts)
            groups.append(
                {
                    "read_group": read_group,
                    "chunk_count": len(chunk_group),
                    "token_steps": token_steps,
                    "experts": experts,
                    "rows": rows,
                    "nbytes": rows * per_row,
                }
            )
        schedule.append(
            {
                "layer": layer_index,
                "groups": groups,
                "rows": sum(int(group["rows"]) for group in groups),
                "nbytes": sum(int(group["nbytes"]) for group in groups),
            }
        )
    return schedule


def build_layer_union_read_plan(
    route_union_summary: dict[str, Any],
    *,
    layer: str,
) -> dict[str, Any]:
    """Select one layer-union slab from a route-union summary."""

    plans = build_layer_union_read_plans(route_union_summary, layer=layer)
    if len(plans) != 1:
        raise ValueError("layer must select one layer")
    return plans[0]


def build_layer_union_read_plans(
    route_union_summary: dict[str, Any],
    *,
    layer: str,
) -> list[dict[str, Any]]:
    """Select one or all layer-union slabs from a route-union summary."""

    per_layer = route_union_summary.get("per_layer") or {}
    if not per_layer:
        raise ValueError("route union summary has no per-layer expert unions")
    if layer == "all":
        selected_keys = sorted(per_layer, key=lambda key: int(key))
    elif layer == "largest":
        selected_keys = [
            max(
                per_layer,
                key=lambda key: (
                    int(per_layer[key].get("union_rows") or 0),
                    -int(key),
                ),
            )
        ]
    else:
        try:
            selected_key = str(int(layer))
        except ValueError as exc:
            raise ValueError("layer must be an integer, 'largest', or 'all'") from exc
        if selected_key not in per_layer:
            raise ValueError(f"layer {selected_key} is not present in the route union summary")
        selected_keys = [selected_key]

    plans: list[dict[str, Any]] = []
    for selected_key in selected_keys:
        selected = per_layer[selected_key]
        experts = [int(expert) for expert in selected.get("experts") or []]
        plans.append(
            {
                "layer": int(selected_key),
                "experts": experts,
                "union_rows": int(selected.get("union_rows") or len(experts)),
                "union_bytes": int(selected.get("union_bytes") or 0),
            }
        )
    return plans


def expand_layer_union_read_plan(
    read_plan: dict[str, Any],
    *,
    extra_experts: Iterable[int],
) -> dict[str, Any]:
    """Return a read plan expanded with experts found by a recomputed route."""

    original_experts = [int(expert) for expert in read_plan.get("experts") or []]
    merged_experts = sorted(set(original_experts).union(int(expert) for expert in extra_experts))
    original_rows = int(read_plan.get("union_rows") or len(original_experts))
    original_bytes = int(read_plan.get("union_bytes") or 0)
    bytes_per_row = original_bytes // original_rows if original_rows else 0
    expanded = dict(read_plan)
    expanded["experts"] = merged_experts
    expanded["union_rows"] = len(merged_experts)
    expanded["union_bytes"] = len(merged_experts) * bytes_per_row
    expanded["original_union_rows"] = original_rows
    expanded["expanded_union_rows"] = len(merged_experts)
    expanded["route_union_expanded"] = len(merged_experts) != original_rows
    return expanded


def layer_overlap_preload_fits(
    preload_byte_cap: int | None,
    *,
    reserved_bytes: int,
    queued_bytes: int,
    future_bytes: int,
) -> bool:
    """Return whether another preload fits the bounded overlap byte budget."""

    if preload_byte_cap is None or int(preload_byte_cap) <= 0:
        return True
    total = int(reserved_bytes) + int(queued_bytes) + int(future_bytes)
    return total <= int(preload_byte_cap)


def layer_overlap_preload_admissions(
    future_bytes: list[int],
    *,
    preload_byte_cap: int | None,
    reserved_bytes: int = 0,
    queued_bytes: int = 0,
    stop_after_first_miss: bool = False,
    order: str = "deadline",
) -> list[bool]:
    """Return bounded preload admissions in scan order."""

    if order not in {"deadline", "smallest-fit"}:
        raise ValueError("order must be 'deadline' or 'smallest-fit'")
    admissions: list[bool] = []
    running_queued_bytes = int(queued_bytes)
    indexed_sizes = list(enumerate(int(size) for size in future_bytes))
    if order == "smallest-fit":
        indexed_sizes.sort(key=lambda item: (item[1], item[0]))
        admissions = [False] * len(future_bytes)
    for original_index, future_size in indexed_sizes:
        fits = layer_overlap_preload_fits(
            preload_byte_cap,
            reserved_bytes=int(reserved_bytes),
            queued_bytes=running_queued_bytes,
            future_bytes=future_size,
        )
        if order == "smallest-fit":
            admissions[original_index] = fits
        else:
            admissions.append(fits)
        if not fits:
            if stop_after_first_miss:
                break
            continue
        running_queued_bytes += future_size
    return admissions


def should_release_layer_window(
    *,
    layer_offset: int,
    layer_count: int,
    window_release: str,
    period: int,
) -> bool:
    """Return whether a layer-union executor should release the current layer."""

    if window_release != "layer":
        return False
    if period < 1:
        raise ValueError("period must be positive")
    if layer_offset < 0 or layer_count < 1:
        raise ValueError("layer_offset and layer_count must describe a layer sequence")
    if layer_offset >= layer_count:
        raise ValueError("layer_offset must be within layer_count")
    return ((layer_offset + 1) % period == 0) or (layer_offset + 1 == layer_count)


def summarize_read_breakdown_seconds(
    *,
    load_seconds: float,
    assign_seconds: float,
    eval_seconds: float,
) -> dict[str, float]:
    """Return timed sub-buckets for one route-union table feed."""

    return {
        "load_seconds": float(load_seconds),
        "assign_seconds": float(assign_seconds),
        "eval_seconds": float(eval_seconds),
        "total_seconds": float(load_seconds) + float(assign_seconds) + float(eval_seconds),
    }


def summarize_windowed_overlap_ceiling(
    layer_results: list[dict[str, Any]],
) -> dict[str, float | int]:
    """Estimate a two-stage read/compute pipeline ceiling for windowed execution."""

    tasks: list[tuple[float, float]] = []
    token_records_per_layer = 0
    for item in layer_results:
        token_steps = {
            int(record["token_step"])
            for record in (item.get("records") or [])
            if record.get("token_step") is not None
        }
        token_records_per_layer = max(token_records_per_layer, len(token_steps))
        chunks = item.get("windowed_chunks") or []
        if chunks:
            for chunk in chunks:
                read_seconds = float(chunk.get("read_seconds") or 0.0)
                compute_seconds = float(chunk.get("compute_seconds") or 0.0)
                release_seconds = float(chunk.get("release_seconds") or 0.0)
                tasks.append((read_seconds, compute_seconds + release_seconds))
            continue

        passes = item.get("compute_passes") or []
        if passes:
            latest = passes[-1]
            tasks.append(
                (
                    float(latest.get("read_seconds") or item.get("read_seconds") or 0.0),
                    float(latest.get("seconds") or 0.0),
                )
            )

    serial_read_seconds = sum(read for read, _compute in tasks)
    serial_compute_seconds = sum(compute for _read, compute in tasks)
    serial_total_seconds = serial_read_seconds + serial_compute_seconds
    read_available = 0.0
    compute_available = 0.0
    for read_seconds, compute_seconds in tasks:
        read_available += read_seconds
        compute_available = max(compute_available, read_available) + compute_seconds
    ideal_overlap_seconds = compute_available
    overlap_saved_seconds = max(0.0, serial_total_seconds - ideal_overlap_seconds)
    return {
        "tasks": len(tasks),
        "token_records_per_layer": token_records_per_layer,
        "serial_read_seconds": serial_read_seconds,
        "serial_compute_seconds": serial_compute_seconds,
        "serial_total_seconds": serial_total_seconds,
        "ideal_overlap_seconds": ideal_overlap_seconds,
        "overlap_saved_seconds": overlap_saved_seconds,
        "overlap_speedup": (
            serial_total_seconds / ideal_overlap_seconds
            if ideal_overlap_seconds > 0
            else 0.0
        ),
        "diagnostic_tok_s_ideal_overlap": (
            token_records_per_layer / ideal_overlap_seconds
            if ideal_overlap_seconds > 0
            else 0.0
        ),
    }
