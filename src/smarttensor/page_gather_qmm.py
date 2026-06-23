"""Page-backed gather-QMM prototypes for SmartTensor route slabs.

These helpers do not add a new MLX primitive yet. They define the runtime
contract SmartTensor needs: route rows select persistent page/row slots, then
the helper maps those selections onto the dense batch axis that stock
``mx.gather_qmm`` already accepts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _shape(array: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(array, "shape", ()))


def expert_slot_page_tables(
    *,
    expert_count: int,
    slot_to_expert: Sequence[int | None],
    rows_per_page: int = 1,
    missing_value: int = -1,
) -> tuple[Any, Any]:
    """Build global-expert page tables from a resident arena slot map."""

    import mlx.core as mx
    import numpy as np

    if expert_count < 1:
        raise ValueError("expert_count must be positive")
    if rows_per_page < 1:
        raise ValueError("rows_per_page must be positive")

    page_ids = np.full((int(expert_count),), int(missing_value), dtype=np.int32)
    row_offsets = np.full((int(expert_count),), int(missing_value), dtype=np.int32)
    for slot, expert in enumerate(slot_to_expert):
        if expert is None:
            continue
        expert_id = int(expert)
        if expert_id < 0 or expert_id >= expert_count:
            raise ValueError("slot_to_expert contains an expert outside expert_count")
        page_ids[expert_id] = int(slot) // int(rows_per_page)
        row_offsets[expert_id] = int(slot) % int(rows_per_page)

    return mx.array(page_ids, dtype=mx.int32), mx.array(row_offsets, dtype=mx.int32)


def page_gather_qmm(
    x: Any,
    weight_pages: Any,
    scale_pages: Any,
    bias_pages: Any | None,
    rhs_page_ids: Any,
    rhs_row_offsets: Any,
    *,
    lhs_indices: Any | None = None,
    transpose: bool = True,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "affine",
    sorted_indices: bool = False,
    validate_resident: bool = False,
) -> Any:
    """Gather quantized RHS matrices from persistent page-row slabs.

    ``weight_pages``, ``scale_pages``, and ``bias_pages`` must share the first
    two dense batch dimensions: ``(page_count, rows_per_page, ...)``. The route
    supplies page ids and row offsets. The function flattens that page-row
    coordinate into the RHS index batch expected by stock MLX ``gather_qmm``.
    """

    import mlx.core as mx
    import numpy as np

    weight_shape = _shape(weight_pages)
    scale_shape = _shape(scale_pages)
    bias_shape = None if bias_pages is None else _shape(bias_pages)
    if len(weight_shape) < 4:
        raise ValueError("weight_pages must have page and row batch dimensions")
    if len(scale_shape) < 4:
        raise ValueError("scale_pages must have page and row batch dimensions")
    if weight_shape[:2] != scale_shape[:2]:
        raise ValueError("weight_pages and scale_pages must share page-row dimensions")
    if bias_shape is not None and weight_shape[:2] != bias_shape[:2]:
        raise ValueError("weight_pages and bias_pages must share page-row dimensions")

    rows_per_page = int(weight_shape[1])
    page_ids = rhs_page_ids.astype(mx.int32)
    row_offsets = rhs_row_offsets.astype(mx.int32)

    if validate_resident:
        missing_page = bool(np.asarray(mx.any(page_ids < 0)))
        invalid_offset = bool(
            np.asarray(
                mx.any(
                    (row_offsets < 0)
                    | (row_offsets >= mx.array(rows_per_page, dtype=mx.int32))
                )
            )
        )
        if missing_page or invalid_offset:
            raise ValueError("rhs page ids or row offsets contain non-resident entries")

    flat_indices = (
        page_ids * mx.array(rows_per_page, dtype=mx.int32) + row_offsets
    ).astype(mx.uint32)
    flat_rows = int(weight_shape[0]) * rows_per_page
    flat_weight_shape = (flat_rows, *weight_shape[2:])
    flat_scale_shape = (flat_rows, *scale_shape[2:])
    flat_bias_shape = None if bias_shape is None else (flat_rows, *bias_shape[2:])

    return mx.gather_qmm(
        x,
        weight_pages.reshape(flat_weight_shape),
        scale_pages.reshape(flat_scale_shape),
        None if bias_pages is None else bias_pages.reshape(flat_bias_shape),
        lhs_indices=lhs_indices,
        rhs_indices=flat_indices,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        sorted_indices=sorted_indices,
    )


def mixed_bit_gather_qmm(
    x: Any,
    rhs_global_indices: Any,
    tiers: Sequence[dict[str, Any]],
    *,
    transpose: bool = True,
    mode: str = "affine",
) -> Any:
    """Mixed-precision MoE gather-QMM via partition-and-sum over bit-width tiers.

    ``mx.gather_qmm`` requires a single ``bits``/``group_size`` per call, so a
    mixed-precision expert set cannot go through one call. Each tier holds a
    compact quantized slab of just its experts plus ``global_to_local`` (an int
    array mapping every global expert id to its row in this tier's slab, or -1
    if the expert lives in another tier). For each tier we run one uniform-bit
    ``gather_qmm`` over the routed experts, zero the contributions of experts
    that belong to other tiers, and sum across tiers. Each routed expert lives
    in exactly one tier, so the masked sum reconstructs the full MoE output with
    each expert computed at its own precision.
    """

    import mlx.core as mx

    out: Any = None
    for tier in tiers:
        global_to_local = tier["global_to_local"]
        local = mx.take(global_to_local, rhs_global_indices).astype(mx.int32)
        in_tier = local >= 0
        safe = mx.where(in_tier, local, mx.zeros_like(local)).astype(mx.uint32)
        out_t = mx.gather_qmm(
            x,
            tier["wq"],
            tier["scales"],
            tier["biases"],
            rhs_indices=safe,
            transpose=transpose,
            group_size=tier["group_size"],
            bits=tier["bits"],
            mode=mode,
        )
        mask = in_tier.reshape(_shape(out_t)[:-1]).astype(out_t.dtype)[..., None]
        contrib = out_t * mask
        out = contrib if out is None else out + contrib
    if out is None:
        raise ValueError("tiers must contain at least one tier")
    return out


def expert_page_gather_qmm(
    x: Any,
    weight_pages: Any,
    scale_pages: Any,
    bias_pages: Any | None,
    rhs_global_indices: Any,
    expert_page_ids: Any,
    expert_row_offsets: Any,
    *,
    lhs_indices: Any | None = None,
    transpose: bool = True,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "affine",
    sorted_indices: bool = False,
    validate_resident: bool = False,
) -> Any:
    """Map global expert ids through page tables, then run page gather-QMM."""

    import mlx.core as mx

    if len(_shape(expert_page_ids)) != 1 or len(_shape(expert_row_offsets)) != 1:
        raise ValueError("expert page tables must be 1D")
    page_ids = mx.take(expert_page_ids, rhs_global_indices).astype(mx.int32)
    row_offsets = mx.take(expert_row_offsets, rhs_global_indices).astype(mx.int32)
    return page_gather_qmm(
        x,
        weight_pages,
        scale_pages,
        bias_pages,
        page_ids,
        row_offsets,
        lhs_indices=lhs_indices,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        sorted_indices=sorted_indices,
        validate_resident=validate_resident,
    )
