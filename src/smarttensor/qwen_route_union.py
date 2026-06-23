"""Reusable Qwen route-union MoE compute helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


ProjectionFn = Callable[..., Any]
ProjectionMode = str


def choose_next_route_union_chunk(
    *,
    pending_indices: Any,
    expected_index: int,
    ready_indices: set[int] | frozenset[int],
    ready_order: bool,
) -> int:
    """Select the next chunk to compute from pending load futures."""

    if not ready_order:
        return int(expected_index)
    for index in sorted(int(value) for value in ready_indices):
        if index in pending_indices:
            return int(index)
    return int(expected_index)


def _default_hotcold_projection(
    *,
    projection_name: str,
    route_x: Any,
    route_source: Any,
    local_indices: Any,
    batch: Any,
    prefix: str,
) -> Any:
    from smarttensor.hotcold_qmm import hotcold_gather_qmm_affine4

    weight = batch.arrays[f"{prefix}.{projection_name}.weight"]
    scales = batch.arrays[f"{prefix}.{projection_name}.scales"]
    biases = batch.arrays[f"{prefix}.{projection_name}.biases"]
    return hotcold_gather_qmm_affine4(
        route_x,
        weight,
        scales,
        biases,
        weight,
        scales,
        biases,
        route_source,
        local_indices,
    )


def _batch_projection_arrays(
    *,
    batch: Any,
    prefix: str,
    projection_name: str,
) -> tuple[Any, Any, Any]:
    return (
        batch.arrays[f"{prefix}.{projection_name}.weight"],
        batch.arrays[f"{prefix}.{projection_name}.scales"],
        batch.arrays[f"{prefix}.{projection_name}.biases"],
    )


def _batch_hotcold_projection_arrays(
    *,
    batch: Any,
    prefix: str,
    projection_name: str,
) -> tuple[tuple[Any, Any, Any], tuple[Any, Any, Any]]:
    cold = _batch_projection_arrays(
        batch=batch,
        prefix=prefix,
        projection_name=projection_name,
    )
    hot_arrays = getattr(batch, "hot_arrays", None)
    if hot_arrays is None:
        return cold, cold
    hot = (
        hot_arrays[f"{prefix}.{projection_name}.weight"],
        hot_arrays[f"{prefix}.{projection_name}.scales"],
        hot_arrays[f"{prefix}.{projection_name}.biases"],
    )
    return hot, cold


def _batch_expert_page_tables(*, batch: Any) -> tuple[Any, Any]:
    arrays = getattr(batch, "arrays", {})
    expert_page_ids = getattr(batch, "expert_page_ids", None)
    if expert_page_ids is None:
        expert_page_ids = arrays.get("__expert_page_ids")
    expert_row_offsets = getattr(batch, "expert_row_offsets", None)
    if expert_row_offsets is None:
        expert_row_offsets = arrays.get("__expert_row_offsets")
    if expert_page_ids is None or expert_row_offsets is None:
        raise ValueError("page_gather_qmm mode requires expert page tables")
    return expert_page_ids, expert_row_offsets


def _batch_qmm_options(*, batch: Any) -> dict[str, Any]:
    return {
        "group_size": int(getattr(batch, "group_size", 64)),
        "bits": int(getattr(batch, "bits", 4)),
        "mode": str(getattr(batch, "qmm_mode", "affine")),
        "validate_resident": bool(getattr(batch, "validate_resident", False)),
    }


def _compute_page_gather_qmm_moe(
    *,
    mlp: Any,
    route_x: Any,
    scores: Any,
    expert_indices_host: Any | None,
    expert_indices_device: Any | None,
    route_shape: tuple[int, ...],
    batch: Any,
    prefix: str,
    shared_y: Any,
    hidden_dims: int,
) -> Any:
    import mlx.core as mx

    from smarttensor.page_gather_qmm import expert_page_gather_qmm

    expert_page_ids, expert_row_offsets = _batch_expert_page_tables(batch=batch)
    qmm_options = _batch_qmm_options(batch=batch)
    if expert_indices_device is not None:
        flat_experts = expert_indices_device.reshape((-1,)).astype(mx.uint32)
    else:
        flat_experts = mx.array(
            np.asarray(expert_indices_host).reshape(-1),
            dtype=mx.uint32,
        )
    route_count = int(flat_experts.shape[0])

    def project(projection_name: str, projection_x: Any) -> Any:
        weight, scales, biases = _batch_projection_arrays(
            batch=batch,
            prefix=prefix,
            projection_name=projection_name,
        )
        output = expert_page_gather_qmm(
            projection_x.reshape((route_count, 1, int(projection_x.shape[-1]))),
            weight,
            scales,
            biases,
            flat_experts,
            expert_page_ids,
            expert_row_offsets,
            transpose=True,
            group_size=qmm_options["group_size"],
            bits=qmm_options["bits"],
            mode=qmm_options["mode"],
            validate_resident=qmm_options["validate_resident"],
        )
        return output.reshape((route_count, -1))

    up = project("up_proj", route_x)
    gate = project("gate_proj", route_x)
    activated = mlp.switch_mlp.activation(up, gate)
    routed_y = project("down_proj", activated)
    expert_y = routed_y.reshape((*route_shape, hidden_dims))
    expert_y = (expert_y * scores[..., None]).sum(axis=-2)
    return expert_y + shared_y


def _compute_hotcold_pair_reduce_moe(
    *,
    mlp: Any,
    input_x: Any,
    scores: Any,
    route_shape: tuple[int, ...],
    batch: Any,
    prefix: str,
    route_source: Any,
    local_indices: Any,
    shared_y: Any,
    single_input: bool,
    norepeat_input: bool = False,
) -> Any:
    import mlx.core as mx

    from smarttensor.hotcold_qmm import (
        hotcold_gather_qmv_affine4_pair,
        hotcold_gather_qmv_affine4_pair_norepeat,
        hotcold_gather_qmv_affine4_pair_single,
        hotcold_gather_qmv_affine4_weighted_sum,
    )

    top_k = int(route_shape[-1])
    hidden_dims = int(input_x.shape[-1])
    input_rows = int(input_x.shape[0])
    route_rows = 1
    for dim in route_shape[:-1]:
        route_rows *= int(dim)
    if route_rows != input_rows:
        raise ValueError("route_shape leading dimensions must match input rows")

    up_hot, up_cold = _batch_hotcold_projection_arrays(
        batch=batch,
        prefix=prefix,
        projection_name="up_proj",
    )
    gate_hot, gate_cold = _batch_hotcold_projection_arrays(
        batch=batch,
        prefix=prefix,
        projection_name="gate_proj",
    )
    down_hot, down_cold = _batch_hotcold_projection_arrays(
        batch=batch,
        prefix=prefix,
        projection_name="down_proj",
    )
    if single_input and norepeat_input:
        raise ValueError("single_input and norepeat_input are mutually exclusive")
    if single_input:
        if input_rows != 1:
            raise ValueError(
                "hotcold_qmv_pair_single_reduce requires exactly one input row"
            )
        up_y, gate_y = hotcold_gather_qmv_affine4_pair_single(
            input_x,
            *up_hot,
            *up_cold,
            *gate_hot,
            *gate_cold,
            route_source,
            local_indices,
        )
    elif norepeat_input:
        up_y, gate_y = hotcold_gather_qmv_affine4_pair_norepeat(
            input_x,
            *up_hot,
            *up_cold,
            *gate_hot,
            *gate_cold,
            route_source,
            local_indices,
            top_k=top_k,
        )
    else:
        route_x = mx.repeat(input_x, top_k, axis=0)
        up_y, gate_y = hotcold_gather_qmv_affine4_pair(
            route_x,
            *up_hot,
            *up_cold,
            *gate_hot,
            *gate_cold,
            route_source,
            local_indices,
        )

    activated = mlp.switch_mlp.activation(up_y, gate_y)
    routed_y = hotcold_gather_qmv_affine4_weighted_sum(
        activated,
        *down_hot,
        *down_cold,
        route_source,
        local_indices,
        scores.reshape((-1,)),
        top_k=top_k,
    )
    return routed_y.reshape((*route_shape[:-1], hidden_dims)) + shared_y


def _compute_serial_switch_mlp_moe(
    *,
    mlp: Any,
    x: Any,
    scores: Any,
    route_shape: tuple[int, ...],
    local_indices: Any,
    shared_y: Any,
    hidden_dims: int,
) -> Any:
    """Compute a route-union slab with stock per-row SwitchMLP semantics."""

    import mlx.core as mx

    top_k = int(route_shape[-1])
    route_rows = 1
    for dim in route_shape[:-1]:
        route_rows *= int(dim)
    flat_x = x.reshape((route_rows, hidden_dims))
    flat_scores = scores.reshape((route_rows, top_k))
    flat_shared_y = shared_y.reshape((route_rows, hidden_dims))
    flat_local_indices = local_indices.reshape((route_rows, top_k))
    row_outputs = []
    for row_index in range(route_rows):
        row_x = flat_x[row_index : row_index + 1].reshape((1, 1, hidden_dims))
        row_local_indices = flat_local_indices[row_index : row_index + 1].reshape(
            (1, 1, top_k)
        )
        row_scores = flat_scores[row_index : row_index + 1].reshape((1, 1, top_k))
        expert_y = mlp.switch_mlp(row_x, row_local_indices)
        expert_y = (expert_y * row_scores[..., None]).sum(axis=-2)
        row_output = expert_y.reshape((1, hidden_dims)) + flat_shared_y[
            row_index : row_index + 1
        ]
        mx.eval(row_output)
        row_outputs.append(row_output)
    return mx.concatenate(row_outputs, axis=0).reshape(tuple(x.shape))


def compute_qwen_route_union_moe(
    *,
    mlp: Any,
    x: Any,
    scores: Any,
    expert_indices_host: Any | None,
    expert_indices_device: Any | None = None,
    route_shape: tuple[int, ...] | None = None,
    batch: Any,
    prefix: str,
    projection_fn: ProjectionFn | None = None,
    shared_y: Any | None = None,
    route_source: Any | None = None,
    local_indices: Any | None = None,
    projection_mode: ProjectionMode = "default",
    batch_chunk_size: int | None = None,
) -> Any:
    """Compute one Qwen MoE block from a loaded route-union expert slab."""

    import mlx.core as mx

    if expert_indices_host is None:
        if route_shape is None:
            if expert_indices_device is None:
                raise ValueError(
                    "route_shape is required when expert_indices_host is not supplied"
                )
            route_shape = tuple(int(dim) for dim in expert_indices_device.shape)
        route_shape = tuple(int(dim) for dim in route_shape)
    else:
        route_shape = tuple(int(dim) for dim in np.asarray(expert_indices_host).shape)
    if not route_shape:
        raise ValueError("expert_indices_host must not be scalar")
    top_k = int(route_shape[-1])
    hidden_dims = int(x.shape[-1])
    if projection_mode not in {
        "default",
        "page_gather_qmm",
        "hotcold_qmv_pair_reduce",
        "hotcold_qmv_pair_norepeat_reduce",
        "hotcold_qmv_pair_single_reduce",
        "serial_switch_mlp",
    }:
        raise ValueError(f"unknown Qwen route-union projection mode: {projection_mode}")
    if projection_fn is not None and projection_mode != "default":
        raise ValueError("projection_fn cannot be combined with a fused projection_mode")
    if (route_source is None) != (local_indices is None):
        raise ValueError("route_source and local_indices must be supplied together")
    if batch_chunk_size is not None:
        batch_chunk_size = int(batch_chunk_size)
        if batch_chunk_size < 1:
            raise ValueError("batch_chunk_size must be positive when set")
    route_rows = 1
    for dim in route_shape[:-1]:
        route_rows *= int(dim)
    if batch_chunk_size is not None and 0 < batch_chunk_size < route_rows:
        if shared_y is None:
            shared_y = mlp.shared_expert(x)
            shared_y = mx.sigmoid(mlp.shared_expert_gate(x)) * shared_y
        flat_x = x.reshape((route_rows, hidden_dims))
        flat_scores = scores.reshape((route_rows, top_k))
        flat_shared_y = shared_y.reshape((route_rows, hidden_dims))
        flat_experts = (
            np.asarray(expert_indices_host).reshape((route_rows, top_k))
            if expert_indices_host is not None
            else None
        )
        flat_experts_device = (
            expert_indices_device.reshape((route_rows, top_k))
            if expert_indices_device is not None
            else None
        )
        flat_route_source = (
            route_source.reshape((route_rows * top_k,))
            if route_source is not None
            else None
        )
        flat_local_indices = (
            local_indices.reshape((route_rows * top_k,))
            if local_indices is not None
            else None
        )
        chunks = []
        for start in range(0, route_rows, batch_chunk_size):
            stop = min(start + batch_chunk_size, route_rows)
            route_start = start * top_k
            route_stop = stop * top_k
            chunks.append(
                compute_qwen_route_union_moe(
                    mlp=mlp,
                    x=flat_x[start:stop],
                    scores=flat_scores[start:stop],
                    expert_indices_host=(
                        flat_experts[start:stop] if flat_experts is not None else None
                    ),
                    expert_indices_device=(
                        flat_experts_device[start:stop]
                        if flat_experts_device is not None
                        else None
                    ),
                    route_shape=(stop - start, top_k),
                    batch=batch,
                    prefix=prefix,
                    projection_fn=projection_fn,
                    shared_y=flat_shared_y[start:stop],
                    route_source=(
                        flat_route_source[route_start:route_stop]
                        if flat_route_source is not None
                        else None
                    ),
                    local_indices=(
                        flat_local_indices[route_start:route_stop]
                        if flat_local_indices is not None
                        else None
                    ),
                    projection_mode=projection_mode,
                    batch_chunk_size=None,
                )
            )
        return mx.concatenate(chunks, axis=0).reshape(tuple(x.shape))

    needs_local_route_map = projection_mode != "page_gather_qmm"
    if needs_local_route_map and route_source is None:
        if expert_indices_host is None:
            raise ValueError(
                "expert_indices_host is required when route_source/local_indices are absent"
            )
        flat_experts = np.asarray(expert_indices_host).reshape(-1)
        batch_row_order = (
            tuple(int(index) for index in batch.first_dim_indices)
            if getattr(batch, "first_dim_indices", None) is not None
            else tuple(sorted({int(expert) for expert in flat_experts}))
        )
        row_by_expert = {
            int(expert): row for row, expert in enumerate(batch_row_order)
        }
        local_rows = [row_by_expert[int(expert)] for expert in flat_experts]
        local_indices = mx.array(np.asarray(local_rows, dtype=np.int32), dtype=mx.int32)
        route_source = mx.array(
            np.ones((int(flat_experts.size),), dtype=np.int32),
            dtype=mx.int32,
        )
    route_x = mx.repeat(x.reshape((-1, hidden_dims)), top_k, axis=0)
    project = projection_fn or _default_hotcold_projection

    if shared_y is None:
        shared_y = mlp.shared_expert(x)
        shared_y = mx.sigmoid(mlp.shared_expert_gate(x)) * shared_y

    if projection_mode == "page_gather_qmm":
        if expert_indices_host is None and expert_indices_device is None:
            raise ValueError(
                "page_gather_qmm mode requires expert_indices_host or expert_indices_device"
            )
        return _compute_page_gather_qmm_moe(
            mlp=mlp,
            route_x=route_x,
            scores=scores,
            expert_indices_host=expert_indices_host,
            expert_indices_device=expert_indices_device,
            route_shape=route_shape,
            batch=batch,
            prefix=prefix,
            shared_y=shared_y,
            hidden_dims=hidden_dims,
        )

    input_x = x.reshape((-1, hidden_dims))
    if projection_mode == "hotcold_qmv_pair_reduce":
        return _compute_hotcold_pair_reduce_moe(
            mlp=mlp,
            input_x=input_x,
            scores=scores,
            route_shape=route_shape,
            batch=batch,
            prefix=prefix,
            route_source=route_source,
            local_indices=local_indices,
            shared_y=shared_y,
            single_input=False,
        )
    if projection_mode == "hotcold_qmv_pair_norepeat_reduce":
        return _compute_hotcold_pair_reduce_moe(
            mlp=mlp,
            input_x=input_x,
            scores=scores,
            route_shape=route_shape,
            batch=batch,
            prefix=prefix,
            route_source=route_source,
            local_indices=local_indices,
            shared_y=shared_y,
            single_input=False,
            norepeat_input=True,
        )
    if projection_mode == "hotcold_qmv_pair_single_reduce":
        return _compute_hotcold_pair_reduce_moe(
            mlp=mlp,
            input_x=input_x,
            scores=scores,
            route_shape=route_shape,
            batch=batch,
            prefix=prefix,
            route_source=route_source,
            local_indices=local_indices,
            shared_y=shared_y,
            single_input=True,
        )
    if projection_mode == "serial_switch_mlp":
        return _compute_serial_switch_mlp_moe(
            mlp=mlp,
            x=x,
            scores=scores,
            route_shape=route_shape,
            local_indices=local_indices,
            shared_y=shared_y,
            hidden_dims=hidden_dims,
        )

    up = project(
        projection_name="up_proj",
        route_x=route_x,
        route_source=route_source,
        local_indices=local_indices,
        batch=batch,
        prefix=prefix,
    )
    gate = project(
        projection_name="gate_proj",
        route_x=route_x,
        route_source=route_source,
        local_indices=local_indices,
        batch=batch,
        prefix=prefix,
    )
    activated = mlp.switch_mlp.activation(up, gate)
    routed_y = project(
        projection_name="down_proj",
        route_x=activated,
        route_source=route_source,
        local_indices=local_indices,
        batch=batch,
        prefix=prefix,
    )
    expert_y = routed_y.reshape((*route_shape, hidden_dims))
    expert_y = (expert_y * scores[..., None]).sum(axis=-2)
    return expert_y + shared_y
