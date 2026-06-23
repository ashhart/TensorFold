"""Experimental hot/cold quantized projection primitives for MLX.

The functions here are deliberately small and decode-only shaped. They prove a
contract that ``mx.gather_qmm`` cannot currently express directly: one logical
route-ordered RHS where each route lane can read either a resident hot slot or
a freshly loaded cold row without first concatenating those rows into a table.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


def _shape(array: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(array, "shape", ()))


def _dtype(array: Any) -> Any:
    return getattr(array, "dtype")


def _as_dtype(array: Any, dtype: Any) -> Any:
    return array if _dtype(array) == dtype else array.astype(dtype)


@lru_cache(maxsize=32)
def _affine4_kernel(
    *,
    route_count: int,
    input_dims: int,
    output_dims: int,
    packed_cols: int,
    groups: int,
    group_size: int,
) -> Any:
    import mlx.core as mx

    packs_per_group = group_size // 8
    source = f"""
        uint elem = thread_position_in_grid.x;
        if (elem >= {int(route_count * output_dims)}) {{
            return;
        }}

        uint route = elem / {int(output_dims)};
        uint out_col = elem - route * {int(output_dims)};
        int src = route_source[route];
        int row = local_indices[route];
        float acc = 0.0f;

        for (uint group = 0; group < {int(groups)}; ++group) {{
            uint scale_offset = (uint(row) * {int(output_dims)} + out_col)
                * {int(groups)} + group;
            float scale = src == 0
                ? float(hot_scales[scale_offset])
                : float(cold_scales[scale_offset]);
            float bias = src == 0
                ? float(hot_biases[scale_offset])
                : float(cold_biases[scale_offset]);
            float group_sum = 0.0f;
            float group_accum = 0.0f;

            for (uint pack = 0; pack < {int(packs_per_group)}; ++pack) {{
                uint weight_offset = (uint(row) * {int(output_dims)} + out_col)
                    * {int(packed_cols)} + group * {int(packs_per_group)} + pack;
                uint packed = src == 0
                    ? hot_weight[weight_offset]
                    : cold_weight[weight_offset];
                uint in_col = group * {int(group_size)} + pack * 8;
                float x0 = float(x[route * {int(input_dims)} + in_col]);
                float x1 = float(x[route * {int(input_dims)} + in_col + 1]);
                float x2 = float(x[route * {int(input_dims)} + in_col + 2]);
                float x3 = float(x[route * {int(input_dims)} + in_col + 3]);
                float x4 = float(x[route * {int(input_dims)} + in_col + 4]);
                float x5 = float(x[route * {int(input_dims)} + in_col + 5]);
                float x6 = float(x[route * {int(input_dims)} + in_col + 6]);
                float x7 = float(x[route * {int(input_dims)} + in_col + 7]);

                group_sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
                group_accum +=
                    x0 * float(packed & 0x0000000fu) +
                    (x1 / 16.0f) * float(packed & 0x000000f0u) +
                    (x2 / 256.0f) * float(packed & 0x00000f00u) +
                    (x3 / 4096.0f) * float(packed & 0x0000f000u) +
                    x4 * float((packed >> 16) & 0x000fu) +
                    (x5 / 16.0f) * float((packed >> 16) & 0x00f0u) +
                    (x6 / 256.0f) * float((packed >> 16) & 0x0f00u) +
                    (x7 / 4096.0f) * float((packed >> 16) & 0xf000u);
            }}
            acc += scale * group_accum + group_sum * bias;
        }}
        out[elem] = T(acc);
    """
    return mx.fast.metal_kernel(
        name=(
            "smarttensor_hotcold_qmm_affine4_"
            f"r{route_count}_i{input_dims}_o{output_dims}_g{group_size}"
        ),
        input_names=[
            "x",
            "hot_weight",
            "hot_scales",
            "hot_biases",
            "cold_weight",
            "cold_scales",
            "cold_biases",
            "route_source",
            "local_indices",
        ],
        output_names=["out"],
        source=source,
    )


def hotcold_gather_qmm_affine4(
    x: Any,
    hot_weight: Any,
    hot_scales: Any,
    hot_biases: Any,
    cold_weight: Any,
    cold_scales: Any,
    cold_biases: Any,
    route_source: Any,
    local_indices: Any,
    *,
    group_size: int = 64,
) -> Any:
    """Run a route-ordered hot/cold 4-bit affine projection.

    Args:
        x: Float input with shape ``(route_count, input_dims)``.
        hot_*: Resident quantized expert rows. ``hot_weight`` must have shape
            ``(hot_slots, output_dims, packed_cols)`` and hot scales/biases must
            have shape ``(hot_slots, output_dims, input_dims // group_size)``.
        cold_*: Freshly loaded quantized expert rows with the same trailing
            dimensions as ``hot_*``.
        route_source: ``int32`` array of shape ``(route_count,)`` where ``0``
            reads hot rows and ``1`` reads cold rows.
        local_indices: ``int32`` array of shape ``(route_count,)``. Each value
            is a slot index when the matching source is hot, or a row index when
            the matching source is cold.

    Returns:
        Array with shape ``(route_count, output_dims)`` and the same dtype as
        ``x`` for MLX-compatible decode-shaped projections.
    """

    import mlx.core as mx

    x_shape = _shape(x)
    hot_w_shape = _shape(hot_weight)
    cold_w_shape = _shape(cold_weight)
    hot_s_shape = _shape(hot_scales)
    cold_s_shape = _shape(cold_scales)
    route_shape = _shape(route_source)
    local_shape = _shape(local_indices)

    if len(x_shape) != 2:
        raise ValueError("x must have shape (route_count, input_dims)")
    if len(hot_w_shape) != 3 or len(cold_w_shape) != 3:
        raise ValueError("hot_weight and cold_weight must be rank-3 arrays")
    if len(hot_s_shape) != 3 or len(cold_s_shape) != 3:
        raise ValueError("hot_scales and cold_scales must be rank-3 arrays")
    if route_shape != (x_shape[0],) or local_shape != (x_shape[0],):
        raise ValueError("route_source and local_indices must match x's route count")

    route_count, input_dims = x_shape
    _hot_slots, output_dims, packed_cols = hot_w_shape
    if cold_w_shape[1:] != hot_w_shape[1:]:
        raise ValueError("cold_weight must match hot_weight trailing dimensions")
    if cold_s_shape[1:] != hot_s_shape[1:]:
        raise ValueError("cold scales/biases must match hot scales/biases trailing dimensions")
    if _shape(hot_biases) != hot_s_shape or _shape(cold_biases) != cold_s_shape:
        raise ValueError("bias shapes must match scale shapes")
    if hot_s_shape[1] != output_dims:
        raise ValueError("scale output dimension must match weight output dimension")
    groups = hot_s_shape[2]
    if group_size <= 0 or group_size % 8 != 0:
        raise ValueError("group_size must be a positive multiple of 8 for affine4")
    if groups * group_size != input_dims:
        raise ValueError("input_dims must equal scale groups * group_size")
    if packed_cols * 8 != input_dims:
        raise ValueError("packed_cols must equal input_dims / 8 for 4-bit weights")
    if group_size == 64 and input_dims % 512 == 0 and output_dims % 8 == 0:
        return hotcold_gather_qmv_affine4(
            x,
            hot_weight,
            hot_scales,
            hot_biases,
            cold_weight,
            cold_scales,
            cold_biases,
            route_source,
            local_indices,
        )

    kernel = _affine4_kernel(
        route_count=route_count,
        input_dims=input_dims,
        output_dims=output_dims,
        packed_cols=packed_cols,
        groups=groups,
        group_size=group_size,
    )
    return kernel(
        inputs=[
            x,
            hot_weight,
            hot_scales,
            hot_biases,
            cold_weight,
            cold_scales,
            cold_biases,
            route_source,
            local_indices,
        ],
        template=[("T", _dtype(x))],
        grid=(route_count * output_dims, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(route_count, output_dims)],
        output_dtypes=[_dtype(x)],
    )[0]


@lru_cache(maxsize=32)
def _affine4_qmv_fast_kernel(
    *,
    route_count: int,
    input_dims: int,
    output_dims: int,
    packed_cols: int,
    groups: int,
) -> Any:
    import mlx.core as mx

    output_tiles = output_dims // 8
    source = f"""
        constexpr int packs_per_thread = 2;
        constexpr int pack_factor = 8;
        constexpr int bytes_per_pack = 4;
        constexpr int values_per_thread = 16;
        constexpr int block_size = values_per_thread * 32;
        constexpr int scale_step_per_thread = 4;
        constexpr int results_per_simdgroup = 4;

        uint tile = threadgroup_position_in_grid.z;
        if (tile >= {int(route_count * output_tiles)}) {{
            return;
        }}

        uint route = tile / {int(output_tiles)};
        uint out_tile = tile - route * {int(output_tiles)};
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;
        uint out_row = out_tile * 8 + simd_gid * results_per_simdgroup;
        if (out_row >= {int(output_dims)}) {{
            return;
        }}

        int src = route_source[route];
        int expert_row = local_indices[route];
        const device uint32_t* weight = src == 0 ? hot_weight : cold_weight;
        const device T* scales = src == 0 ? hot_scales : cold_scales;
        const device T* biases = src == 0 ? hot_biases : cold_biases;

        const device uint8_t* ws = (const device uint8_t*)weight;
        size_t matrix_base = (size_t(expert_row) * {int(output_dims)} + out_row);
        ws += matrix_base * {int(packed_cols)} * bytes_per_pack
            + simd_lid * packs_per_thread * bytes_per_pack;
        scales += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
        biases += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
        const device T* x_route = x + route * {int(input_dims)}
            + simd_lid * values_per_thread;
        device T* y_route = out + route * {int(output_dims)} + out_row;

        typedef float U;
        thread U x_thread[values_per_thread];
        thread U result[results_per_simdgroup] = {{0}};

        for (int k = 0; k < {int(input_dims)}; k += block_size) {{
            U sum = 0;
            for (int i = 0; i < values_per_thread; i += 4) {{
                sum += x_route[i] + x_route[i + 1] + x_route[i + 2] + x_route[i + 3];
                x_thread[i] = x_route[i];
                x_thread[i + 1] = x_route[i + 1] / 16.0f;
                x_thread[i + 2] = x_route[i + 2] / 256.0f;
                x_thread[i + 3] = x_route[i + 3] / 4096.0f;
            }}

            for (int row = 0; row < results_per_simdgroup; row++) {{
                const device uint16_t* wl = (const device uint16_t*)(
                    ws + row * {int(packed_cols)} * bytes_per_pack);
                const device T* sl = scales + row * {int(groups)};
                const device T* bl = biases + row * {int(groups)};
                U s = sl[0];
                U b = bl[0];
                U accum = 0;
                for (int i = 0; i < values_per_thread / 4; i++) {{
                    uint16_t packed = wl[i];
                    accum +=
                        x_thread[4 * i] * (packed & 0x000f) +
                        x_thread[4 * i + 1] * (packed & 0x00f0) +
                        x_thread[4 * i + 2] * (packed & 0x0f00) +
                        x_thread[4 * i + 3] * (packed & 0xf000);
                }}
                result[row] += s * accum + sum * b;
            }}

            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / 64;
            biases += block_size / 64;
            x_route += block_size;
        }}

        for (int row = 0; row < results_per_simdgroup; row++) {{
            U reduced = simd_sum(result[row]);
            if (simd_lid == 0) {{
                y_route[row] = static_cast<T>(reduced);
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=(
            "smarttensor_hotcold_qmv_affine4_"
            f"r{route_count}_i{input_dims}_o{output_dims}"
        ),
        input_names=[
            "x",
            "hot_weight",
            "hot_scales",
            "hot_biases",
            "cold_weight",
            "cold_scales",
            "cold_biases",
            "route_source",
            "local_indices",
        ],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=32)
def _affine4_qmv_pair_fast_kernel(
    *,
    route_count: int,
    input_dims: int,
    output_dims: int,
    packed_cols: int,
    groups: int,
    single_input: bool = False,
    norepeat_top_k: int | None = None,
) -> Any:
    import mlx.core as mx

    output_tiles = output_dims // 8
    if single_input and norepeat_top_k is not None:
        raise ValueError("single_input and norepeat_top_k are mutually exclusive")
    if single_input:
        name_suffix = "_single"
        x_route_offset = "simd_lid * values_per_thread"
    elif norepeat_top_k is not None:
        name_suffix = f"_norepeat_k{int(norepeat_top_k)}"
        x_route_offset = (
            f"(route / {int(norepeat_top_k)}) * {int(input_dims)} "
            "+ simd_lid * values_per_thread"
        )
    else:
        name_suffix = ""
        x_route_offset = f"route * {int(input_dims)} + simd_lid * values_per_thread"
    source = f"""
        constexpr int packs_per_thread = 2;
        constexpr int pack_factor = 8;
        constexpr int bytes_per_pack = 4;
        constexpr int values_per_thread = 16;
        constexpr int block_size = values_per_thread * 32;
        constexpr int scale_step_per_thread = 4;
        constexpr int results_per_simdgroup = 4;

        uint tile = threadgroup_position_in_grid.z;
        if (tile >= {int(route_count * output_tiles)}) {{
            return;
        }}

        uint route = tile / {int(output_tiles)};
        uint out_tile = tile - route * {int(output_tiles)};
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;
        uint out_row = out_tile * 8 + simd_gid * results_per_simdgroup;
        if (out_row >= {int(output_dims)}) {{
            return;
        }}

        int src = route_source[route];
        int expert_row = local_indices[route];
        const device uint32_t* weight_a = src == 0 ? hot_a_weight : cold_a_weight;
        const device T* scales_a = src == 0 ? hot_a_scales : cold_a_scales;
        const device T* biases_a = src == 0 ? hot_a_biases : cold_a_biases;
        const device uint32_t* weight_b = src == 0 ? hot_b_weight : cold_b_weight;
        const device T* scales_b = src == 0 ? hot_b_scales : cold_b_scales;
        const device T* biases_b = src == 0 ? hot_b_biases : cold_b_biases;

        const device uint8_t* ws_a = (const device uint8_t*)weight_a;
        const device uint8_t* ws_b = (const device uint8_t*)weight_b;
        size_t matrix_base = (size_t(expert_row) * {int(output_dims)} + out_row);
        ws_a += matrix_base * {int(packed_cols)} * bytes_per_pack
            + simd_lid * packs_per_thread * bytes_per_pack;
        ws_b += matrix_base * {int(packed_cols)} * bytes_per_pack
            + simd_lid * packs_per_thread * bytes_per_pack;
        scales_a += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
        biases_a += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
        scales_b += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
        biases_b += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
        const device T* x_route = x + {x_route_offset};
        device T* y_a_route = out_a + route * {int(output_dims)} + out_row;
        device T* y_b_route = out_b + route * {int(output_dims)} + out_row;

        typedef float U;
        thread U x_thread[values_per_thread];
        thread U result_a[results_per_simdgroup] = {{0}};
        thread U result_b[results_per_simdgroup] = {{0}};

        for (int k = 0; k < {int(input_dims)}; k += block_size) {{
            U sum = 0;
            for (int i = 0; i < values_per_thread; i += 4) {{
                sum += x_route[i] + x_route[i + 1] + x_route[i + 2] + x_route[i + 3];
                x_thread[i] = x_route[i];
                x_thread[i + 1] = x_route[i + 1] / 16.0f;
                x_thread[i + 2] = x_route[i + 2] / 256.0f;
                x_thread[i + 3] = x_route[i + 3] / 4096.0f;
            }}

            for (int row = 0; row < results_per_simdgroup; row++) {{
                const device uint16_t* wl_a = (const device uint16_t*)(
                    ws_a + row * {int(packed_cols)} * bytes_per_pack);
                const device T* sl_a = scales_a + row * {int(groups)};
                const device T* bl_a = biases_a + row * {int(groups)};
                U s_a = sl_a[0];
                U b_a = bl_a[0];
                U accum_a = 0;
                const device uint16_t* wl_b = (const device uint16_t*)(
                    ws_b + row * {int(packed_cols)} * bytes_per_pack);
                const device T* sl_b = scales_b + row * {int(groups)};
                const device T* bl_b = biases_b + row * {int(groups)};
                U s_b = sl_b[0];
                U b_b = bl_b[0];
                U accum_b = 0;
                for (int i = 0; i < values_per_thread / 4; i++) {{
                    uint16_t packed_a = wl_a[i];
                    accum_a +=
                        x_thread[4 * i] * (packed_a & 0x000f) +
                        x_thread[4 * i + 1] * (packed_a & 0x00f0) +
                        x_thread[4 * i + 2] * (packed_a & 0x0f00) +
                        x_thread[4 * i + 3] * (packed_a & 0xf000);
                    uint16_t packed_b = wl_b[i];
                    accum_b +=
                        x_thread[4 * i] * (packed_b & 0x000f) +
                        x_thread[4 * i + 1] * (packed_b & 0x00f0) +
                        x_thread[4 * i + 2] * (packed_b & 0x0f00) +
                        x_thread[4 * i + 3] * (packed_b & 0xf000);
                }}
                result_a[row] += s_a * accum_a + sum * b_a;
                result_b[row] += s_b * accum_b + sum * b_b;
            }}

            ws_a += block_size * bytes_per_pack / pack_factor;
            ws_b += block_size * bytes_per_pack / pack_factor;
            scales_a += block_size / 64;
            biases_a += block_size / 64;
            scales_b += block_size / 64;
            biases_b += block_size / 64;
            x_route += block_size;
        }}

        for (int row = 0; row < results_per_simdgroup; row++) {{
            U reduced_a = simd_sum(result_a[row]);
            U reduced_b = simd_sum(result_b[row]);
            if (simd_lid == 0) {{
                y_a_route[row] = static_cast<T>(reduced_a);
                y_b_route[row] = static_cast<T>(reduced_b);
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=(
            f"smarttensor_hotcold_qmv_pair{name_suffix}_affine4_"
            f"r{route_count}_i{input_dims}_o{output_dims}"
        ),
        input_names=[
            "x",
            "hot_a_weight",
            "hot_a_scales",
            "hot_a_biases",
            "cold_a_weight",
            "cold_a_scales",
            "cold_a_biases",
            "hot_b_weight",
            "hot_b_scales",
            "hot_b_biases",
            "cold_b_weight",
            "cold_b_scales",
            "cold_b_biases",
            "route_source",
            "local_indices",
        ],
        output_names=["out_a", "out_b"],
        source=source,
    )


@lru_cache(maxsize=32)
def _affine4_qmv_norepeat_fast_kernel(
    *,
    route_count: int,
    top_k: int,
    input_dims: int,
    output_dims: int,
    packed_cols: int,
    groups: int,
) -> Any:
    import mlx.core as mx

    output_tiles = output_dims // 8
    source = f"""
        constexpr int packs_per_thread = 2;
        constexpr int pack_factor = 8;
        constexpr int bytes_per_pack = 4;
        constexpr int values_per_thread = 16;
        constexpr int block_size = values_per_thread * 32;
        constexpr int scale_step_per_thread = 4;
        constexpr int results_per_simdgroup = 4;

        uint tile = threadgroup_position_in_grid.z;
        if (tile >= {int(route_count * output_tiles)}) {{
            return;
        }}

        uint route = tile / {int(output_tiles)};
        uint out_tile = tile - route * {int(output_tiles)};
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;
        uint out_row = out_tile * 8 + simd_gid * results_per_simdgroup;
        if (out_row >= {int(output_dims)}) {{
            return;
        }}

        int src = route_source[route];
        int expert_row = local_indices[route];
        const device uint32_t* weight = src == 0 ? hot_weight : cold_weight;
        const device T* scales = src == 0 ? hot_scales : cold_scales;
        const device T* biases = src == 0 ? hot_biases : cold_biases;

        const device uint8_t* ws = (const device uint8_t*)weight;
        size_t matrix_base = (size_t(expert_row) * {int(output_dims)} + out_row);
        ws += matrix_base * {int(packed_cols)} * bytes_per_pack
            + simd_lid * packs_per_thread * bytes_per_pack;
        scales += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
        biases += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
        uint input_row = route / {int(top_k)};
        const device T* x_route = x + input_row * {int(input_dims)}
            + simd_lid * values_per_thread;
        device T* y_route = out + route * {int(output_dims)} + out_row;

        typedef float U;
        thread U x_thread[values_per_thread];
        thread U result[results_per_simdgroup] = {{0}};

        for (int k = 0; k < {int(input_dims)}; k += block_size) {{
            U sum = 0;
            for (int i = 0; i < values_per_thread; i += 4) {{
                sum += x_route[i] + x_route[i + 1] + x_route[i + 2] + x_route[i + 3];
                x_thread[i] = x_route[i];
                x_thread[i + 1] = x_route[i + 1] / 16.0f;
                x_thread[i + 2] = x_route[i + 2] / 256.0f;
                x_thread[i + 3] = x_route[i + 3] / 4096.0f;
            }}

            for (int row = 0; row < results_per_simdgroup; row++) {{
                const device uint16_t* wl = (const device uint16_t*)(
                    ws + row * {int(packed_cols)} * bytes_per_pack);
                const device T* sl = scales + row * {int(groups)};
                const device T* bl = biases + row * {int(groups)};
                U s = sl[0];
                U b = bl[0];
                U accum = 0;
                for (int i = 0; i < values_per_thread / 4; i++) {{
                    uint16_t packed = wl[i];
                    accum +=
                        x_thread[4 * i] * (packed & 0x000f) +
                        x_thread[4 * i + 1] * (packed & 0x00f0) +
                        x_thread[4 * i + 2] * (packed & 0x0f00) +
                        x_thread[4 * i + 3] * (packed & 0xf000);
                }}
                result[row] += s * accum + sum * b;
            }}

            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / 64;
            biases += block_size / 64;
            x_route += block_size;
        }}

        for (int row = 0; row < results_per_simdgroup; row++) {{
            U reduced = simd_sum(result[row]);
            if (simd_lid == 0) {{
                y_route[row] = static_cast<T>(reduced);
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=(
            "smarttensor_hotcold_qmv_norepeat_affine4_"
            f"r{route_count}_k{top_k}_i{input_dims}_o{output_dims}"
        ),
        input_names=[
            "x",
            "hot_weight",
            "hot_scales",
            "hot_biases",
            "cold_weight",
            "cold_scales",
            "cold_biases",
            "route_source",
            "local_indices",
        ],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=32)
def _affine4_qmv_expertmap_fast_kernel(
    *,
    route_count: int,
    input_dims: int,
    output_dims: int,
    packed_cols: int,
    groups: int,
    cold_rows: int,
) -> Any:
    import mlx.core as mx

    output_tiles = output_dims // 8
    source = f"""
        constexpr int packs_per_thread = 2;
        constexpr int pack_factor = 8;
        constexpr int bytes_per_pack = 4;
        constexpr int values_per_thread = 16;
        constexpr int block_size = values_per_thread * 32;
        constexpr int scale_step_per_thread = 4;
        constexpr int results_per_simdgroup = 4;

        uint tile = threadgroup_position_in_grid.z;
        if (tile >= {int(route_count * output_tiles)}) {{
            return;
        }}

        uint route = tile / {int(output_tiles)};
        uint out_tile = tile - route * {int(output_tiles)};
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;
        uint out_row = out_tile * 8 + simd_gid * results_per_simdgroup;
        if (out_row >= {int(output_dims)}) {{
            return;
        }}

        int expert = expert_indices[route];
        int expert_row = hot_slot_map[expert];
        int src = 0;
        if (expert_row < 0) {{
            src = 1;
            expert_row = 0;
            for (uint cold_row = 0; cold_row < {int(cold_rows)}; ++cold_row) {{
                if (cold_experts[cold_row] == expert) {{
                    expert_row = int(cold_row);
                    break;
                }}
            }}
        }}
        const device uint32_t* weight = src == 0 ? hot_weight : cold_weight;
        const device T* scales = src == 0 ? hot_scales : cold_scales;
        const device T* biases = src == 0 ? hot_biases : cold_biases;

        const device uint8_t* ws = (const device uint8_t*)weight;
        size_t matrix_base = (size_t(expert_row) * {int(output_dims)} + out_row);
        ws += matrix_base * {int(packed_cols)} * bytes_per_pack
            + simd_lid * packs_per_thread * bytes_per_pack;
        scales += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
        biases += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
        const device T* x_route = x + route * {int(input_dims)}
            + simd_lid * values_per_thread;
        device T* y_route = out + route * {int(output_dims)} + out_row;

        typedef float U;
        thread U x_thread[values_per_thread];
        thread U result[results_per_simdgroup] = {{0}};

        for (int k = 0; k < {int(input_dims)}; k += block_size) {{
            U sum = 0;
            for (int i = 0; i < values_per_thread; i += 4) {{
                sum += x_route[i] + x_route[i + 1] + x_route[i + 2] + x_route[i + 3];
                x_thread[i] = x_route[i];
                x_thread[i + 1] = x_route[i + 1] / 16.0f;
                x_thread[i + 2] = x_route[i + 2] / 256.0f;
                x_thread[i + 3] = x_route[i + 3] / 4096.0f;
            }}

            for (int row = 0; row < results_per_simdgroup; row++) {{
                const device uint16_t* wl = (const device uint16_t*)(
                    ws + row * {int(packed_cols)} * bytes_per_pack);
                const device T* sl = scales + row * {int(groups)};
                const device T* bl = biases + row * {int(groups)};
                U s = sl[0];
                U b = bl[0];
                U accum = 0;
                for (int i = 0; i < values_per_thread / 4; i++) {{
                    uint16_t packed = wl[i];
                    accum +=
                        x_thread[4 * i] * (packed & 0x000f) +
                        x_thread[4 * i + 1] * (packed & 0x00f0) +
                        x_thread[4 * i + 2] * (packed & 0x0f00) +
                        x_thread[4 * i + 3] * (packed & 0xf000);
                }}
                result[row] += s * accum + sum * b;
            }}

            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / 64;
            biases += block_size / 64;
            x_route += block_size;
        }}

        for (int row = 0; row < results_per_simdgroup; row++) {{
            U reduced = simd_sum(result[row]);
            if (simd_lid == 0) {{
                y_route[row] = static_cast<T>(reduced);
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=(
            "smarttensor_hotcold_qmv_expertmap_affine4_"
            f"r{route_count}_i{input_dims}_o{output_dims}_c{cold_rows}"
        ),
        input_names=[
            "x",
            "hot_weight",
            "hot_scales",
            "hot_biases",
            "cold_weight",
            "cold_scales",
            "cold_biases",
            "expert_indices",
            "hot_slot_map",
            "cold_experts",
        ],
        output_names=["out"],
        source=source,
    )


def hotcold_gather_qmv_affine4_expertmap(
    x: Any,
    hot_weight: Any,
    hot_scales: Any,
    hot_biases: Any,
    cold_weight: Any,
    cold_scales: Any,
    cold_biases: Any,
    expert_indices: Any,
    hot_slot_map: Any,
    cold_experts: Any,
) -> Any:
    """Run hot/cold QMV using routed expert ids and a frozen hot-slot map."""

    import mlx.core as mx

    x_shape = _shape(x)
    hot_w_shape = _shape(hot_weight)
    cold_w_shape = _shape(cold_weight)
    hot_s_shape = _shape(hot_scales)
    cold_s_shape = _shape(cold_scales)
    expert_shape = _shape(expert_indices)
    slot_map_shape = _shape(hot_slot_map)
    cold_expert_shape = _shape(cold_experts)

    if len(x_shape) != 2:
        raise ValueError("x must have shape (route_count, input_dims)")
    if len(hot_w_shape) != 3 or len(cold_w_shape) != 3:
        raise ValueError("hot_weight and cold_weight must be rank-3 arrays")
    if expert_shape != (x_shape[0],):
        raise ValueError("expert_indices must match x's route count")
    if len(slot_map_shape) != 1 or len(cold_expert_shape) != 1:
        raise ValueError("hot_slot_map and cold_experts must be 1D arrays")

    route_count, input_dims = x_shape
    _hot_slots, output_dims, packed_cols = hot_w_shape
    if cold_w_shape[1:] != hot_w_shape[1:]:
        raise ValueError("cold_weight must match hot_weight trailing dimensions")
    if cold_s_shape[1:] != hot_s_shape[1:]:
        raise ValueError("cold scales/biases must match hot scales/biases trailing dimensions")
    if _shape(hot_biases) != hot_s_shape or _shape(cold_biases) != cold_s_shape:
        raise ValueError("bias shapes must match scale shapes")
    groups = hot_s_shape[2]
    if hot_s_shape[1] != output_dims:
        raise ValueError("scale output dimension must match weight output dimension")
    if groups * 64 != input_dims:
        raise ValueError("hotcold_gather_qmv_affine4_expertmap requires group_size=64")
    if packed_cols * 8 != input_dims:
        raise ValueError("packed_cols must equal input_dims / 8 for 4-bit weights")
    if input_dims % 512 != 0:
        raise ValueError("input_dims must be a multiple of 512 for the fast expert-map QMV probe")
    if output_dims % 8 != 0:
        raise ValueError("output_dims must be a multiple of 8 for the fast expert-map QMV probe")

    kernel = _affine4_qmv_expertmap_fast_kernel(
        route_count=route_count,
        input_dims=input_dims,
        output_dims=output_dims,
        packed_cols=packed_cols,
        groups=groups,
        cold_rows=cold_expert_shape[0],
    )
    return kernel(
        inputs=[
            x,
            hot_weight,
            hot_scales,
            hot_biases,
            cold_weight,
            cold_scales,
            cold_biases,
            expert_indices,
            hot_slot_map,
            cold_experts,
        ],
        template=[("T", _dtype(x))],
        grid=(32, 2, route_count * (output_dims // 8)),
        threadgroup=(32, 2, 1),
        output_shapes=[(route_count, output_dims)],
        output_dtypes=[_dtype(x)],
    )[0]


@lru_cache(maxsize=32)
def _affine4_qmv_single_fast_kernel(
    *,
    route_count: int,
    input_dims: int,
    output_dims: int,
    packed_cols: int,
    groups: int,
) -> Any:
    import mlx.core as mx

    output_tiles = output_dims // 8
    source = f"""
        constexpr int packs_per_thread = 2;
        constexpr int pack_factor = 8;
        constexpr int bytes_per_pack = 4;
        constexpr int values_per_thread = 16;
        constexpr int block_size = values_per_thread * 32;
        constexpr int scale_step_per_thread = 4;
        constexpr int results_per_simdgroup = 4;

        uint tile = threadgroup_position_in_grid.z;
        if (tile >= {int(route_count * output_tiles)}) {{
            return;
        }}

        uint route = tile / {int(output_tiles)};
        uint out_tile = tile - route * {int(output_tiles)};
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;
        uint out_row = out_tile * 8 + simd_gid * results_per_simdgroup;
        if (out_row >= {int(output_dims)}) {{
            return;
        }}

        int src = route_source[route];
        int expert_row = local_indices[route];
        const device uint32_t* weight = src == 0 ? hot_weight : cold_weight;
        const device T* scales = src == 0 ? hot_scales : cold_scales;
        const device T* biases = src == 0 ? hot_biases : cold_biases;

        const device uint8_t* ws = (const device uint8_t*)weight;
        size_t matrix_base = (size_t(expert_row) * {int(output_dims)} + out_row);
        ws += matrix_base * {int(packed_cols)} * bytes_per_pack
            + simd_lid * packs_per_thread * bytes_per_pack;
        scales += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
        biases += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
        const device T* x_route = x + simd_lid * values_per_thread;
        device T* y_route = out + route * {int(output_dims)} + out_row;

        typedef float U;
        thread U x_thread[values_per_thread];
        thread U result[results_per_simdgroup] = {{0}};

        for (int k = 0; k < {int(input_dims)}; k += block_size) {{
            U sum = 0;
            for (int i = 0; i < values_per_thread; i += 4) {{
                sum += x_route[i] + x_route[i + 1] + x_route[i + 2] + x_route[i + 3];
                x_thread[i] = x_route[i];
                x_thread[i + 1] = x_route[i + 1] / 16.0f;
                x_thread[i + 2] = x_route[i + 2] / 256.0f;
                x_thread[i + 3] = x_route[i + 3] / 4096.0f;
            }}

            for (int row = 0; row < results_per_simdgroup; row++) {{
                const device uint16_t* wl = (const device uint16_t*)(
                    ws + row * {int(packed_cols)} * bytes_per_pack);
                const device T* sl = scales + row * {int(groups)};
                const device T* bl = biases + row * {int(groups)};
                U s = sl[0];
                U b = bl[0];
                U accum = 0;
                for (int i = 0; i < values_per_thread / 4; i++) {{
                    uint16_t packed = wl[i];
                    accum +=
                        x_thread[4 * i] * (packed & 0x000f) +
                        x_thread[4 * i + 1] * (packed & 0x00f0) +
                        x_thread[4 * i + 2] * (packed & 0x0f00) +
                        x_thread[4 * i + 3] * (packed & 0xf000);
                }}
                result[row] += s * accum + sum * b;
            }}

            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / 64;
            biases += block_size / 64;
            x_route += block_size;
        }}

        for (int row = 0; row < results_per_simdgroup; row++) {{
            U reduced = simd_sum(result[row]);
            if (simd_lid == 0) {{
                y_route[row] = static_cast<T>(reduced);
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=(
            "smarttensor_hotcold_qmv_single_affine4_"
            f"r{route_count}_i{input_dims}_o{output_dims}"
        ),
        input_names=[
            "x",
            "hot_weight",
            "hot_scales",
            "hot_biases",
            "cold_weight",
            "cold_scales",
            "cold_biases",
            "route_source",
            "local_indices",
        ],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=32)
def _affine4_qmv_weighted_sum_fast_kernel(
    *,
    route_count: int,
    top_k: int,
    input_dims: int,
    output_dims: int,
    packed_cols: int,
    groups: int,
) -> Any:
    import mlx.core as mx

    input_rows = route_count // top_k
    output_tiles = output_dims // 8
    source = f"""
        constexpr int packs_per_thread = 2;
        constexpr int pack_factor = 8;
        constexpr int bytes_per_pack = 4;
        constexpr int values_per_thread = 16;
        constexpr int block_size = values_per_thread * 32;
        constexpr int scale_step_per_thread = 4;
        constexpr int results_per_simdgroup = 4;

        uint tile = threadgroup_position_in_grid.z;
        if (tile >= {int(input_rows * output_tiles)}) {{
            return;
        }}

        uint input_row = tile / {int(output_tiles)};
        uint out_tile = tile - input_row * {int(output_tiles)};
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;
        uint out_row = out_tile * 8 + simd_gid * results_per_simdgroup;
        if (out_row >= {int(output_dims)}) {{
            return;
        }}

        typedef float U;
        thread S weighted[results_per_simdgroup] = {{0}};

        for (uint lane = 0; lane < {int(top_k)}; ++lane) {{
            uint route = input_row * {int(top_k)} + lane;
            int src = route_source[route];
            int expert_row = local_indices[route];
            S score = scores[route];
            const device uint32_t* weight = src == 0 ? hot_weight : cold_weight;
            const device T* scales = src == 0 ? hot_scales : cold_scales;
            const device T* biases = src == 0 ? hot_biases : cold_biases;

            const device uint8_t* ws = (const device uint8_t*)weight;
            size_t matrix_base = (size_t(expert_row) * {int(output_dims)} + out_row);
            ws += matrix_base * {int(packed_cols)} * bytes_per_pack
                + simd_lid * packs_per_thread * bytes_per_pack;
            scales += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
            biases += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
            const device T* x_route = x + route * {int(input_dims)}
                + simd_lid * values_per_thread;

            thread U x_thread[values_per_thread];
            thread U result[results_per_simdgroup] = {{0}};

            for (int k = 0; k < {int(input_dims)}; k += block_size) {{
                U sum = 0;
                for (int i = 0; i < values_per_thread; i += 4) {{
                    sum += x_route[i] + x_route[i + 1] + x_route[i + 2] + x_route[i + 3];
                    x_thread[i] = x_route[i];
                    x_thread[i + 1] = x_route[i + 1] / 16.0f;
                    x_thread[i + 2] = x_route[i + 2] / 256.0f;
                    x_thread[i + 3] = x_route[i + 3] / 4096.0f;
                }}

                for (int row = 0; row < results_per_simdgroup; row++) {{
                    const device uint16_t* wl = (const device uint16_t*)(
                        ws + row * {int(packed_cols)} * bytes_per_pack);
                    const device T* sl = scales + row * {int(groups)};
                    const device T* bl = biases + row * {int(groups)};
                    U s = sl[0];
                    U b = bl[0];
                    U accum = 0;
                    for (int i = 0; i < values_per_thread / 4; i++) {{
                        uint16_t packed = wl[i];
                        accum +=
                            x_thread[4 * i] * (packed & 0x000f) +
                            x_thread[4 * i + 1] * (packed & 0x00f0) +
                            x_thread[4 * i + 2] * (packed & 0x0f00) +
                            x_thread[4 * i + 3] * (packed & 0xf000);
                    }}
                    result[row] += s * accum + sum * b;
                }}

                ws += block_size * bytes_per_pack / pack_factor;
                scales += block_size / 64;
                biases += block_size / 64;
                x_route += block_size;
            }}

            for (int row = 0; row < results_per_simdgroup; row++) {{
                U reduced = simd_sum(result[row]);
                if (simd_lid == 0) {{
                    T rounded = static_cast<T>(reduced);
                    S projected = static_cast<S>(rounded);
                    S product = projected * score;
                    weighted[row] = weighted[row] + product;
                }}
            }}
        }}

        device S* y_row = out + input_row * {int(output_dims)} + out_row;
        for (int row = 0; row < results_per_simdgroup; row++) {{
            if (simd_lid == 0) {{
                y_row[row] = weighted[row];
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=(
            "smarttensor_hotcold_qmv_weighted_sum_affine4_"
            f"r{route_count}_k{top_k}_i{input_dims}_o{output_dims}"
        ),
        input_names=[
            "x",
            "hot_weight",
            "hot_scales",
            "hot_biases",
            "cold_weight",
            "cold_scales",
            "cold_biases",
            "route_source",
            "local_indices",
            "scores",
        ],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=32)
def _affine4_qmv_weighted_sum_expertmap_fast_kernel(
    *,
    route_count: int,
    top_k: int,
    input_dims: int,
    output_dims: int,
    packed_cols: int,
    groups: int,
    cold_rows: int,
) -> Any:
    import mlx.core as mx

    input_rows = route_count // top_k
    output_tiles = output_dims // 8
    source = f"""
        constexpr int packs_per_thread = 2;
        constexpr int pack_factor = 8;
        constexpr int bytes_per_pack = 4;
        constexpr int values_per_thread = 16;
        constexpr int block_size = values_per_thread * 32;
        constexpr int scale_step_per_thread = 4;
        constexpr int results_per_simdgroup = 4;

        uint tile = threadgroup_position_in_grid.z;
        if (tile >= {int(input_rows * output_tiles)}) {{
            return;
        }}

        uint input_row = tile / {int(output_tiles)};
        uint out_tile = tile - input_row * {int(output_tiles)};
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;
        uint out_row = out_tile * 8 + simd_gid * results_per_simdgroup;
        if (out_row >= {int(output_dims)}) {{
            return;
        }}

        typedef float U;
        thread S weighted[results_per_simdgroup] = {{0}};

        for (uint lane = 0; lane < {int(top_k)}; ++lane) {{
            uint route = input_row * {int(top_k)} + lane;
            int expert = expert_indices[route];
            int expert_row = hot_slot_map[expert];
            int src = 0;
            if (expert_row < 0) {{
                src = 1;
                expert_row = 0;
                for (uint cold_row = 0; cold_row < {int(cold_rows)}; ++cold_row) {{
                    if (cold_experts[cold_row] == expert) {{
                        expert_row = int(cold_row);
                        break;
                    }}
                }}
            }}
            S score = scores[route];
            const device uint32_t* weight = src == 0 ? hot_weight : cold_weight;
            const device T* scales = src == 0 ? hot_scales : cold_scales;
            const device T* biases = src == 0 ? hot_biases : cold_biases;

            const device uint8_t* ws = (const device uint8_t*)weight;
            size_t matrix_base = (size_t(expert_row) * {int(output_dims)} + out_row);
            ws += matrix_base * {int(packed_cols)} * bytes_per_pack
                + simd_lid * packs_per_thread * bytes_per_pack;
            scales += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
            biases += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
            const device T* x_route = x + route * {int(input_dims)}
                + simd_lid * values_per_thread;

            thread U x_thread[values_per_thread];
            thread U result[results_per_simdgroup] = {{0}};

            for (int k = 0; k < {int(input_dims)}; k += block_size) {{
                U sum = 0;
                for (int i = 0; i < values_per_thread; i += 4) {{
                    sum += x_route[i] + x_route[i + 1] + x_route[i + 2] + x_route[i + 3];
                    x_thread[i] = x_route[i];
                    x_thread[i + 1] = x_route[i + 1] / 16.0f;
                    x_thread[i + 2] = x_route[i + 2] / 256.0f;
                    x_thread[i + 3] = x_route[i + 3] / 4096.0f;
                }}

                for (int row = 0; row < results_per_simdgroup; row++) {{
                    const device uint16_t* wl = (const device uint16_t*)(
                        ws + row * {int(packed_cols)} * bytes_per_pack);
                    const device T* sl = scales + row * {int(groups)};
                    const device T* bl = biases + row * {int(groups)};
                    U s = sl[0];
                    U b = bl[0];
                    U accum = 0;
                    for (int i = 0; i < values_per_thread / 4; i++) {{
                        uint16_t packed = wl[i];
                        accum +=
                            x_thread[4 * i] * (packed & 0x000f) +
                            x_thread[4 * i + 1] * (packed & 0x00f0) +
                            x_thread[4 * i + 2] * (packed & 0x0f00) +
                            x_thread[4 * i + 3] * (packed & 0xf000);
                    }}
                    result[row] += s * accum + sum * b;
                }}

                ws += block_size * bytes_per_pack / pack_factor;
                scales += block_size / 64;
                biases += block_size / 64;
                x_route += block_size;
            }}

            for (int row = 0; row < results_per_simdgroup; row++) {{
                U reduced = simd_sum(result[row]);
                if (simd_lid == 0) {{
                    T rounded = static_cast<T>(reduced);
                    S projected = static_cast<S>(rounded);
                    S product = projected * score;
                    weighted[row] = weighted[row] + product;
                }}
            }}
        }}

        device S* y_row = out + input_row * {int(output_dims)} + out_row;
        for (int row = 0; row < results_per_simdgroup; row++) {{
            if (simd_lid == 0) {{
                y_row[row] = weighted[row];
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=(
            "smarttensor_hotcold_qmv_weighted_sum_expertmap_affine4_"
            f"r{route_count}_k{top_k}_i{input_dims}_o{output_dims}_c{cold_rows}"
        ),
        input_names=[
            "x",
            "hot_weight",
            "hot_scales",
            "hot_biases",
            "cold_weight",
            "cold_scales",
            "cold_biases",
            "expert_indices",
            "hot_slot_map",
            "cold_experts",
            "scores",
        ],
        output_names=["out"],
        source=source,
    )


def hotcold_gather_qmv_affine4(
    x: Any,
    hot_weight: Any,
    hot_scales: Any,
    hot_biases: Any,
    cold_weight: Any,
    cold_scales: Any,
    cold_biases: Any,
    route_source: Any,
    local_indices: Any,
) -> Any:
    """Run a Qwen-decode-shaped hot/cold affine-4 gather-QMV.

    This mirrors MLX's fast gather-QMV kernel shape instead of the slower scalar
    proof kernel. It intentionally supports the shape class we care about first:
    ``bits=4``, ``group_size=64``, ``input_dims % 512 == 0`` and
    ``output_dims % 8 == 0``.
    """

    import mlx.core as mx

    x_shape = _shape(x)
    hot_w_shape = _shape(hot_weight)
    cold_w_shape = _shape(cold_weight)
    hot_s_shape = _shape(hot_scales)
    cold_s_shape = _shape(cold_scales)
    route_shape = _shape(route_source)
    local_shape = _shape(local_indices)

    if len(x_shape) != 2:
        raise ValueError("x must have shape (route_count, input_dims)")
    if len(hot_w_shape) != 3 or len(cold_w_shape) != 3:
        raise ValueError("hot_weight and cold_weight must be rank-3 arrays")
    if route_shape != (x_shape[0],) or local_shape != (x_shape[0],):
        raise ValueError("route_source and local_indices must match x's route count")

    route_count, input_dims = x_shape
    _hot_slots, output_dims, packed_cols = hot_w_shape
    if cold_w_shape[1:] != hot_w_shape[1:]:
        raise ValueError("cold_weight must match hot_weight trailing dimensions")
    if cold_s_shape[1:] != hot_s_shape[1:]:
        raise ValueError("cold scales/biases must match hot scales/biases trailing dimensions")
    if _shape(hot_biases) != hot_s_shape or _shape(cold_biases) != cold_s_shape:
        raise ValueError("bias shapes must match scale shapes")
    groups = hot_s_shape[2]
    if hot_s_shape[1] != output_dims:
        raise ValueError("scale output dimension must match weight output dimension")
    if groups * 64 != input_dims:
        raise ValueError("hotcold_gather_qmv_affine4 requires group_size=64")
    if packed_cols * 8 != input_dims:
        raise ValueError("packed_cols must equal input_dims / 8 for 4-bit weights")
    if input_dims % 512 != 0:
        raise ValueError("input_dims must be a multiple of 512 for the fast QMV probe")
    if output_dims % 8 != 0:
        raise ValueError("output_dims must be a multiple of 8 for the fast QMV probe")

    kernel = _affine4_qmv_fast_kernel(
        route_count=route_count,
        input_dims=input_dims,
        output_dims=output_dims,
        packed_cols=packed_cols,
        groups=groups,
    )
    return kernel(
        inputs=[
            x,
            hot_weight,
            hot_scales,
            hot_biases,
            cold_weight,
            cold_scales,
            cold_biases,
            route_source,
            local_indices,
        ],
        template=[("T", _dtype(x))],
        grid=(32, 2, route_count * (output_dims // 8)),
        threadgroup=(32, 2, 1),
        output_shapes=[(route_count, output_dims)],
        output_dtypes=[_dtype(x)],
    )[0]


def hotcold_gather_qmv_affine4_weighted_sum(
    x: Any,
    hot_weight: Any,
    hot_scales: Any,
    hot_biases: Any,
    cold_weight: Any,
    cold_scales: Any,
    cold_biases: Any,
    route_source: Any,
    local_indices: Any,
    scores: Any,
    *,
    top_k: int,
) -> Any:
    """Run hot/cold down-QMV and route-score reduction in one Metal pass.

    This is equivalent to:

    ``hotcold_gather_qmv_affine4(...).reshape(rows, top_k, -1)
    * scores.reshape(rows, top_k, 1).sum(axis=-2)``

    The per-route projection is rounded to the activation dtype before score
    multiplication to match the existing materialized QMV path.
    """

    import mlx.core as mx

    x_shape = _shape(x)
    hot_w_shape = _shape(hot_weight)
    cold_w_shape = _shape(cold_weight)
    hot_s_shape = _shape(hot_scales)
    cold_s_shape = _shape(cold_scales)
    route_shape = _shape(route_source)
    local_shape = _shape(local_indices)
    score_shape = _shape(scores)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if len(x_shape) != 2:
        raise ValueError("x must have shape (route_count, input_dims)")
    if len(hot_w_shape) != 3 or len(cold_w_shape) != 3:
        raise ValueError("hot_weight and cold_weight must be rank-3 arrays")
    if route_shape != local_shape or len(route_shape) != 1:
        raise ValueError("route_source and local_indices must be 1D arrays")
    if score_shape != route_shape:
        raise ValueError("scores must be a 1D array matching route_source")

    route_count, input_dims = x_shape
    if route_count != route_shape[0]:
        raise ValueError("x route count must match route_source")
    if route_count % int(top_k) != 0:
        raise ValueError("route count must be divisible by top_k")
    _hot_slots, output_dims, packed_cols = hot_w_shape
    if cold_w_shape[1:] != hot_w_shape[1:]:
        raise ValueError("cold_weight must match hot_weight trailing dimensions")
    if cold_s_shape[1:] != hot_s_shape[1:]:
        raise ValueError("cold scales/biases must match hot scales/biases trailing dimensions")
    if _shape(hot_biases) != hot_s_shape or _shape(cold_biases) != cold_s_shape:
        raise ValueError("bias shapes must match scale shapes")
    groups = hot_s_shape[2]
    if hot_s_shape[1] != output_dims:
        raise ValueError("scale output dimension must match weight output dimension")
    if groups * 64 != input_dims:
        raise ValueError("hotcold_gather_qmv_affine4_weighted_sum requires group_size=64")
    if packed_cols * 8 != input_dims:
        raise ValueError("packed_cols must equal input_dims / 8 for 4-bit weights")
    if input_dims % 512 != 0:
        raise ValueError("input_dims must be a multiple of 512 for the fast weighted QMV probe")
    if output_dims % 8 != 0:
        raise ValueError("output_dims must be a multiple of 8 for the fast weighted QMV probe")

    output_dtype = _dtype(scores) if _dtype(scores) == mx.float32 else _dtype(x)
    kernel = _affine4_qmv_weighted_sum_fast_kernel(
        route_count=route_count,
        top_k=int(top_k),
        input_dims=input_dims,
        output_dims=output_dims,
        packed_cols=packed_cols,
        groups=groups,
    )
    return kernel(
        inputs=[
            x,
            hot_weight,
            hot_scales,
            hot_biases,
            cold_weight,
            cold_scales,
            cold_biases,
            route_source,
            local_indices,
            scores,
        ],
        template=[("T", _dtype(x)), ("S", output_dtype)],
        grid=(32, 2, (route_count // int(top_k)) * (output_dims // 8)),
        threadgroup=(32, 2, 1),
        output_shapes=[(route_count // int(top_k), output_dims)],
        output_dtypes=[output_dtype],
    )[0]


@lru_cache(maxsize=32)
def _affine8_qmv_norepeat_fast_kernel(
    *,
    route_count: int,
    top_k: int,
    input_dims: int,
    output_dims: int,
    packed_cols: int,
    groups: int,
) -> Any:
    import mlx.core as mx

    output_tiles = output_dims // 8
    source = f"""
        constexpr int packs_per_thread = 4;
        constexpr int pack_factor = 4;
        constexpr int bytes_per_pack = 4;
        constexpr int values_per_thread = 16;
        constexpr int block_size = values_per_thread * 32;
        constexpr int scale_step_per_thread = 4;
        constexpr int results_per_simdgroup = 4;

        uint tile = threadgroup_position_in_grid.z;
        if (tile >= {int(route_count * output_tiles)}) {{
            return;
        }}

        uint route = tile / {int(output_tiles)};
        uint out_tile = tile - route * {int(output_tiles)};
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;
        uint out_row = out_tile * 8 + simd_gid * results_per_simdgroup;
        if (out_row >= {int(output_dims)}) {{
            return;
        }}

        int src = route_source[route];
        int expert_row = local_indices[route];
        const device uint32_t* weight = src == 0 ? hot_weight : cold_weight;
        const device T* scales = src == 0 ? hot_scales : cold_scales;
        const device T* biases = src == 0 ? hot_biases : cold_biases;

        const device uint8_t* ws = (const device uint8_t*)weight;
        size_t matrix_base = (size_t(expert_row) * {int(output_dims)} + out_row);
        ws += matrix_base * {int(packed_cols)} * bytes_per_pack
            + simd_lid * packs_per_thread * bytes_per_pack;
        scales += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
        biases += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
        uint input_row = route / {int(top_k)};
        const device T* x_route = x + input_row * {int(input_dims)}
            + simd_lid * values_per_thread;
        device T* y_route = out + route * {int(output_dims)} + out_row;

        typedef float U;
        thread U x_thread[values_per_thread];
        thread U result[results_per_simdgroup] = {{0}};

        for (int k = 0; k < {int(input_dims)}; k += block_size) {{
            U sum = 0;
            for (int i = 0; i < values_per_thread; i++) {{
                sum += x_route[i];
                x_thread[i] = x_route[i];
            }}

            for (int row = 0; row < results_per_simdgroup; row++) {{
                const device uint32_t* wl = (const device uint32_t*)(
                    ws + row * {int(packed_cols)} * bytes_per_pack);
                const device T* sl = scales + row * {int(groups)};
                const device T* bl = biases + row * {int(groups)};
                U s = sl[0];
                U b = bl[0];
                U accum = 0;
                for (int i = 0; i < values_per_thread / 4; i++) {{
                    uint packed = wl[i];
                    accum +=
                        x_thread[4 * i] * float(packed & 0x000000ffu) +
                        x_thread[4 * i + 1] * float((packed >> 8) & 0x000000ffu) +
                        x_thread[4 * i + 2] * float((packed >> 16) & 0x000000ffu) +
                        x_thread[4 * i + 3] * float((packed >> 24) & 0x000000ffu);
                }}
                result[row] += s * accum + sum * b;
            }}

            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / 64;
            biases += block_size / 64;
            x_route += block_size;
        }}

        for (int row = 0; row < results_per_simdgroup; row++) {{
            U reduced = simd_sum(result[row]);
            if (simd_lid == 0) {{
                y_route[row] = static_cast<T>(reduced);
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=(
            "smarttensor_hotcold_qmv_norepeat_affine8_"
            f"r{route_count}_k{top_k}_i{input_dims}_o{output_dims}"
        ),
        input_names=[
            "x",
            "hot_weight",
            "hot_scales",
            "hot_biases",
            "cold_weight",
            "cold_scales",
            "cold_biases",
            "route_source",
            "local_indices",
        ],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=32)
def _affine8_qmv_weighted_sum_fast_kernel(
    *,
    route_count: int,
    top_k: int,
    input_dims: int,
    output_dims: int,
    packed_cols: int,
    groups: int,
) -> Any:
    import mlx.core as mx

    input_rows = route_count // top_k
    output_tiles = output_dims // 8
    source = f"""
        constexpr int packs_per_thread = 4;
        constexpr int pack_factor = 4;
        constexpr int bytes_per_pack = 4;
        constexpr int values_per_thread = 16;
        constexpr int block_size = values_per_thread * 32;
        constexpr int scale_step_per_thread = 4;
        constexpr int results_per_simdgroup = 4;

        uint tile = threadgroup_position_in_grid.z;
        if (tile >= {int(input_rows * output_tiles)}) {{
            return;
        }}

        uint input_row = tile / {int(output_tiles)};
        uint out_tile = tile - input_row * {int(output_tiles)};
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;
        uint out_row = out_tile * 8 + simd_gid * results_per_simdgroup;
        if (out_row >= {int(output_dims)}) {{
            return;
        }}

        typedef float U;
        thread S weighted[results_per_simdgroup] = {{0}};

        for (uint lane = 0; lane < {int(top_k)}; ++lane) {{
            uint route = input_row * {int(top_k)} + lane;
            int src = route_source[route];
            int expert_row = local_indices[route];
            S score = scores[route];
            const device uint32_t* weight = src == 0 ? hot_weight : cold_weight;
            const device T* scales = src == 0 ? hot_scales : cold_scales;
            const device T* biases = src == 0 ? hot_biases : cold_biases;

            const device uint8_t* ws = (const device uint8_t*)weight;
            size_t matrix_base = (size_t(expert_row) * {int(output_dims)} + out_row);
            ws += matrix_base * {int(packed_cols)} * bytes_per_pack
                + simd_lid * packs_per_thread * bytes_per_pack;
            scales += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
            biases += matrix_base * {int(groups)} + simd_lid / scale_step_per_thread;
            const device T* x_route = x + route * {int(input_dims)}
                + simd_lid * values_per_thread;

            thread U x_thread[values_per_thread];
            thread U result[results_per_simdgroup] = {{0}};

            for (int k = 0; k < {int(input_dims)}; k += block_size) {{
                U sum = 0;
                for (int i = 0; i < values_per_thread; i++) {{
                    sum += x_route[i];
                    x_thread[i] = x_route[i];
                }}

                for (int row = 0; row < results_per_simdgroup; row++) {{
                    const device uint32_t* wl = (const device uint32_t*)(
                        ws + row * {int(packed_cols)} * bytes_per_pack);
                    const device T* sl = scales + row * {int(groups)};
                    const device T* bl = biases + row * {int(groups)};
                    U s = sl[0];
                    U b = bl[0];
                    U accum = 0;
                    for (int i = 0; i < values_per_thread / 4; i++) {{
                        uint packed = wl[i];
                        accum +=
                            x_thread[4 * i] * float(packed & 0x000000ffu) +
                            x_thread[4 * i + 1] * float((packed >> 8) & 0x000000ffu) +
                            x_thread[4 * i + 2] * float((packed >> 16) & 0x000000ffu) +
                            x_thread[4 * i + 3] * float((packed >> 24) & 0x000000ffu);
                    }}
                    result[row] += s * accum + sum * b;
                }}

                ws += block_size * bytes_per_pack / pack_factor;
                scales += block_size / 64;
                biases += block_size / 64;
                x_route += block_size;
            }}

            for (int row = 0; row < results_per_simdgroup; row++) {{
                U reduced = simd_sum(result[row]);
                if (simd_lid == 0) {{
                    T rounded = static_cast<T>(reduced);
                    S projected = static_cast<S>(rounded);
                    weighted[row] = weighted[row] + projected * score;
                }}
            }}
        }}

        device S* y_row = out + input_row * {int(output_dims)} + out_row;
        for (int row = 0; row < results_per_simdgroup; row++) {{
            if (simd_lid == 0) {{
                y_row[row] = weighted[row];
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=(
            "smarttensor_hotcold_qmv_weighted_sum_affine8_"
            f"r{route_count}_k{top_k}_i{input_dims}_o{output_dims}"
        ),
        input_names=[
            "x",
            "hot_weight",
            "hot_scales",
            "hot_biases",
            "cold_weight",
            "cold_scales",
            "cold_biases",
            "route_source",
            "local_indices",
            "scores",
        ],
        output_names=["out"],
        source=source,
    )


def hotcold_gather_qmv_affine8_norepeat(
    x: Any,
    hot_weight: Any,
    hot_scales: Any,
    hot_biases: Any,
    cold_weight: Any,
    cold_scales: Any,
    cold_biases: Any,
    route_source: Any,
    local_indices: Any,
    *,
    top_k: int,
) -> Any:
    x_shape = _shape(x)
    hot_w_shape = _shape(hot_weight)
    cold_w_shape = _shape(cold_weight)
    hot_s_shape = _shape(hot_scales)
    cold_s_shape = _shape(cold_scales)
    route_shape = _shape(route_source)
    local_shape = _shape(local_indices)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if len(x_shape) != 2:
        raise ValueError("x must have shape (input_rows, input_dims)")
    if len(hot_w_shape) != 3 or len(cold_w_shape) != 3:
        raise ValueError("hot_weight and cold_weight must be rank-3 arrays")
    if route_shape != local_shape or len(route_shape) != 1:
        raise ValueError("route_source and local_indices must be 1D arrays")
    input_rows, input_dims = x_shape
    route_count = route_shape[0]
    if input_rows * int(top_k) != route_count:
        raise ValueError("route count must equal input rows * top_k")
    _hot_slots, output_dims, packed_cols = hot_w_shape
    if cold_w_shape[1:] != hot_w_shape[1:]:
        raise ValueError("cold_weight must match hot_weight trailing dimensions")
    if cold_s_shape[1:] != hot_s_shape[1:]:
        raise ValueError("cold scales/biases must match hot scales/biases trailing dimensions")
    if _shape(hot_biases) != hot_s_shape or _shape(cold_biases) != cold_s_shape:
        raise ValueError("bias shapes must match scale shapes")
    groups = hot_s_shape[2]
    if hot_s_shape[1] != output_dims:
        raise ValueError("scale output dimension must match weight output dimension")
    if groups * 64 != input_dims:
        raise ValueError("hotcold_gather_qmv_affine8_norepeat requires group_size=64")
    if packed_cols * 4 != input_dims:
        raise ValueError("packed_cols must equal input_dims / 4 for 8-bit weights")
    if input_dims % 512 != 0:
        raise ValueError("input_dims must be a multiple of 512 for the fast affine8 QMV probe")
    if output_dims % 8 != 0:
        raise ValueError("output_dims must be a multiple of 8 for the fast affine8 QMV probe")

    input_dtype = _dtype(x)
    hot_scales = _as_dtype(hot_scales, input_dtype)
    hot_biases = _as_dtype(hot_biases, input_dtype)
    cold_scales = _as_dtype(cold_scales, input_dtype)
    cold_biases = _as_dtype(cold_biases, input_dtype)
    kernel = _affine8_qmv_norepeat_fast_kernel(
        route_count=route_count,
        top_k=int(top_k),
        input_dims=input_dims,
        output_dims=output_dims,
        packed_cols=packed_cols,
        groups=groups,
    )
    return kernel(
        inputs=[
            x,
            hot_weight,
            hot_scales,
            hot_biases,
            cold_weight,
            cold_scales,
            cold_biases,
            route_source,
            local_indices,
        ],
        template=[("T", input_dtype)],
        grid=(32, 2, route_count * (output_dims // 8)),
        threadgroup=(32, 2, 1),
        output_shapes=[(route_count, output_dims)],
        output_dtypes=[input_dtype],
    )[0]


def hotcold_gather_qmv_affine8_weighted_sum(
    x: Any,
    hot_weight: Any,
    hot_scales: Any,
    hot_biases: Any,
    cold_weight: Any,
    cold_scales: Any,
    cold_biases: Any,
    route_source: Any,
    local_indices: Any,
    scores: Any,
    *,
    top_k: int,
) -> Any:
    import mlx.core as mx

    x_shape = _shape(x)
    hot_w_shape = _shape(hot_weight)
    cold_w_shape = _shape(cold_weight)
    hot_s_shape = _shape(hot_scales)
    cold_s_shape = _shape(cold_scales)
    route_shape = _shape(route_source)
    local_shape = _shape(local_indices)
    score_shape = _shape(scores)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if len(x_shape) != 2:
        raise ValueError("x must have shape (route_count, input_dims)")
    if len(hot_w_shape) != 3 or len(cold_w_shape) != 3:
        raise ValueError("hot_weight and cold_weight must be rank-3 arrays")
    if route_shape != local_shape or len(route_shape) != 1:
        raise ValueError("route_source and local_indices must be 1D arrays")
    if score_shape != route_shape:
        raise ValueError("scores must be a 1D array matching route_source")
    route_count, input_dims = x_shape
    if route_count % int(top_k) != 0:
        raise ValueError("route count must be divisible by top_k")
    _hot_slots, output_dims, packed_cols = hot_w_shape
    if cold_w_shape[1:] != hot_w_shape[1:]:
        raise ValueError("cold_weight must match hot_weight trailing dimensions")
    if cold_s_shape[1:] != hot_s_shape[1:]:
        raise ValueError("cold scales/biases must match hot scales/biases trailing dimensions")
    if _shape(hot_biases) != hot_s_shape or _shape(cold_biases) != cold_s_shape:
        raise ValueError("bias shapes must match scale shapes")
    groups = hot_s_shape[2]
    if hot_s_shape[1] != output_dims:
        raise ValueError("scale output dimension must match weight output dimension")
    if groups * 64 != input_dims:
        raise ValueError("hotcold_gather_qmv_affine8_weighted_sum requires group_size=64")
    if packed_cols * 4 != input_dims:
        raise ValueError("packed_cols must equal input_dims / 4 for 8-bit weights")
    if input_dims % 512 != 0:
        raise ValueError("input_dims must be a multiple of 512 for the fast affine8 weighted QMV probe")
    if output_dims % 8 != 0:
        raise ValueError("output_dims must be a multiple of 8 for the fast affine8 weighted QMV probe")

    input_dtype = _dtype(x)
    hot_scales = _as_dtype(hot_scales, input_dtype)
    hot_biases = _as_dtype(hot_biases, input_dtype)
    cold_scales = _as_dtype(cold_scales, input_dtype)
    cold_biases = _as_dtype(cold_biases, input_dtype)
    output_dtype = _dtype(scores) if _dtype(scores) == mx.float32 else input_dtype
    kernel = _affine8_qmv_weighted_sum_fast_kernel(
        route_count=route_count,
        top_k=int(top_k),
        input_dims=input_dims,
        output_dims=output_dims,
        packed_cols=packed_cols,
        groups=groups,
    )
    return kernel(
        inputs=[
            x,
            hot_weight,
            hot_scales,
            hot_biases,
            cold_weight,
            cold_scales,
            cold_biases,
            route_source,
            local_indices,
            scores,
        ],
        template=[("T", input_dtype), ("S", output_dtype)],
        grid=(32, 2, (route_count // int(top_k)) * (output_dims // 8)),
        threadgroup=(32, 2, 1),
        output_shapes=[(route_count // int(top_k), output_dims)],
        output_dtypes=[output_dtype],
    )[0]


def hotcold_gather_qmv_affine4_weighted_sum_expertmap(
    x: Any,
    hot_weight: Any,
    hot_scales: Any,
    hot_biases: Any,
    cold_weight: Any,
    cold_scales: Any,
    cold_biases: Any,
    expert_indices: Any,
    hot_slot_map: Any,
    cold_experts: Any,
    scores: Any,
    *,
    top_k: int,
) -> Any:
    """Run hot/cold down-QMV+score reduction from expert ids and slot maps."""

    import mlx.core as mx

    x_shape = _shape(x)
    hot_w_shape = _shape(hot_weight)
    cold_w_shape = _shape(cold_weight)
    hot_s_shape = _shape(hot_scales)
    cold_s_shape = _shape(cold_scales)
    expert_shape = _shape(expert_indices)
    slot_map_shape = _shape(hot_slot_map)
    cold_expert_shape = _shape(cold_experts)
    score_shape = _shape(scores)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if len(x_shape) != 2:
        raise ValueError("x must have shape (route_count, input_dims)")
    if len(hot_w_shape) != 3 or len(cold_w_shape) != 3:
        raise ValueError("hot_weight and cold_weight must be rank-3 arrays")
    if expert_shape != score_shape or len(expert_shape) != 1:
        raise ValueError("expert_indices and scores must be matching 1D arrays")
    if len(slot_map_shape) != 1 or len(cold_expert_shape) != 1:
        raise ValueError("hot_slot_map and cold_experts must be 1D arrays")

    route_count, input_dims = x_shape
    if route_count != expert_shape[0]:
        raise ValueError("x route count must match expert_indices")
    if route_count % int(top_k) != 0:
        raise ValueError("route count must be divisible by top_k")
    _hot_slots, output_dims, packed_cols = hot_w_shape
    if cold_w_shape[1:] != hot_w_shape[1:]:
        raise ValueError("cold_weight must match hot_weight trailing dimensions")
    if cold_s_shape[1:] != hot_s_shape[1:]:
        raise ValueError("cold scales/biases must match hot scales/biases trailing dimensions")
    if _shape(hot_biases) != hot_s_shape or _shape(cold_biases) != cold_s_shape:
        raise ValueError("bias shapes must match scale shapes")
    groups = hot_s_shape[2]
    if hot_s_shape[1] != output_dims:
        raise ValueError("scale output dimension must match weight output dimension")
    if groups * 64 != input_dims:
        raise ValueError("hotcold_gather_qmv_affine4_weighted_sum_expertmap requires group_size=64")
    if packed_cols * 8 != input_dims:
        raise ValueError("packed_cols must equal input_dims / 8 for 4-bit weights")
    if input_dims % 512 != 0:
        raise ValueError("input_dims must be a multiple of 512 for the fast expert-map weighted QMV probe")
    if output_dims % 8 != 0:
        raise ValueError("output_dims must be a multiple of 8 for the fast expert-map weighted QMV probe")

    output_dtype = _dtype(scores) if _dtype(scores) == mx.float32 else _dtype(x)
    kernel = _affine4_qmv_weighted_sum_expertmap_fast_kernel(
        route_count=route_count,
        top_k=int(top_k),
        input_dims=input_dims,
        output_dims=output_dims,
        packed_cols=packed_cols,
        groups=groups,
        cold_rows=cold_expert_shape[0],
    )
    return kernel(
        inputs=[
            x,
            hot_weight,
            hot_scales,
            hot_biases,
            cold_weight,
            cold_scales,
            cold_biases,
            expert_indices,
            hot_slot_map,
            cold_experts,
            scores,
        ],
        template=[("T", _dtype(x)), ("S", output_dtype)],
        grid=(32, 2, (route_count // int(top_k)) * (output_dims // 8)),
        threadgroup=(32, 2, 1),
        output_shapes=[(route_count // int(top_k), output_dims)],
        output_dtypes=[output_dtype],
    )[0]


def hotcold_gather_qmv_affine4_single(
    x: Any,
    hot_weight: Any,
    hot_scales: Any,
    hot_biases: Any,
    cold_weight: Any,
    cold_scales: Any,
    cold_biases: Any,
    route_source: Any,
    local_indices: Any,
) -> Any:
    """Run hot/cold QMV for one input row shared by all route lanes."""

    import mlx.core as mx

    x_shape = _shape(x)
    hot_w_shape = _shape(hot_weight)
    cold_w_shape = _shape(cold_weight)
    hot_s_shape = _shape(hot_scales)
    cold_s_shape = _shape(cold_scales)
    route_shape = _shape(route_source)
    local_shape = _shape(local_indices)
    if x_shape[:1] != (1,) or len(x_shape) != 2:
        raise ValueError("x must have shape (1, input_dims)")
    if len(hot_w_shape) != 3 or len(cold_w_shape) != 3:
        raise ValueError("hot_weight and cold_weight must be rank-3 arrays")
    if route_shape != local_shape or len(route_shape) != 1:
        raise ValueError("route_source and local_indices must be 1D arrays")

    route_count = route_shape[0]
    _input_rows, input_dims = x_shape
    _hot_slots, output_dims, packed_cols = hot_w_shape
    if cold_w_shape[1:] != hot_w_shape[1:]:
        raise ValueError("cold_weight must match hot_weight trailing dimensions")
    if cold_s_shape[1:] != hot_s_shape[1:]:
        raise ValueError("cold scales/biases must match hot scales/biases trailing dimensions")
    if _shape(hot_biases) != hot_s_shape or _shape(cold_biases) != cold_s_shape:
        raise ValueError("bias shapes must match scale shapes")
    groups = hot_s_shape[2]
    if hot_s_shape[1] != output_dims:
        raise ValueError("scale output dimension must match weight output dimension")
    if groups * 64 != input_dims:
        raise ValueError("hotcold_gather_qmv_affine4_single requires group_size=64")
    if packed_cols * 8 != input_dims:
        raise ValueError("packed_cols must equal input_dims / 8 for 4-bit weights")
    if input_dims % 512 != 0:
        raise ValueError("input_dims must be a multiple of 512 for the fast QMV single probe")
    if output_dims % 8 != 0:
        raise ValueError("output_dims must be a multiple of 8 for the fast QMV single probe")

    kernel = _affine4_qmv_single_fast_kernel(
        route_count=route_count,
        input_dims=input_dims,
        output_dims=output_dims,
        packed_cols=packed_cols,
        groups=groups,
    )
    return kernel(
        inputs=[
            x,
            hot_weight,
            hot_scales,
            hot_biases,
            cold_weight,
            cold_scales,
            cold_biases,
            route_source,
            local_indices,
        ],
        template=[("T", _dtype(x))],
        grid=(32, 2, route_count * (output_dims // 8)),
        threadgroup=(32, 2, 1),
        output_shapes=[(route_count, output_dims)],
        output_dtypes=[_dtype(x)],
    )[0]


def hotcold_gather_qmv_affine4_pair(
    x: Any,
    hot_a_weight: Any,
    hot_a_scales: Any,
    hot_a_biases: Any,
    cold_a_weight: Any,
    cold_a_scales: Any,
    cold_a_biases: Any,
    hot_b_weight: Any,
    hot_b_scales: Any,
    hot_b_biases: Any,
    cold_b_weight: Any,
    cold_b_scales: Any,
    cold_b_biases: Any,
    route_source: Any,
    local_indices: Any,
) -> tuple[Any, Any]:
    """Run two Qwen-decode-shaped hot/cold affine-4 QMV projections together.

    The paired primitive is exact-contract sugar for the up+gate half of a
    SwiGLU expert: it shares the route metadata, Metal launch, and input-vector
    loads, then writes two independent projection outputs.
    """

    import mlx.core as mx

    x_shape = _shape(x)
    a_w_shape = _shape(hot_a_weight)
    b_w_shape = _shape(hot_b_weight)
    cold_a_w_shape = _shape(cold_a_weight)
    cold_b_w_shape = _shape(cold_b_weight)
    a_s_shape = _shape(hot_a_scales)
    b_s_shape = _shape(hot_b_scales)
    route_shape = _shape(route_source)
    local_shape = _shape(local_indices)

    if len(x_shape) != 2:
        raise ValueError("x must have shape (route_count, input_dims)")
    if len(a_w_shape) != 3 or len(b_w_shape) != 3:
        raise ValueError("hot projection weights must be rank-3 arrays")
    if a_w_shape != b_w_shape:
        raise ValueError("paired projections must have matching hot weight shapes")
    if cold_a_w_shape[1:] != a_w_shape[1:] or cold_b_w_shape[1:] != b_w_shape[1:]:
        raise ValueError("cold weights must match hot trailing dimensions")
    if route_shape != (x_shape[0],) or local_shape != (x_shape[0],):
        raise ValueError("route_source and local_indices must match x's route count")
    if _shape(cold_a_scales)[1:] != a_s_shape[1:] or _shape(cold_b_scales)[1:] != b_s_shape[1:]:
        raise ValueError("cold scales/biases must match hot scales/biases trailing dimensions")
    if _shape(hot_a_biases) != a_s_shape or _shape(cold_a_biases) != _shape(cold_a_scales):
        raise ValueError("projection A bias shapes must match scale shapes")
    if _shape(hot_b_biases) != b_s_shape or _shape(cold_b_biases) != _shape(cold_b_scales):
        raise ValueError("projection B bias shapes must match scale shapes")
    if a_s_shape != b_s_shape:
        raise ValueError("paired projections must have matching scale shapes")

    route_count, input_dims = x_shape
    _hot_slots, output_dims, packed_cols = a_w_shape
    groups = a_s_shape[2]
    if a_s_shape[1] != output_dims:
        raise ValueError("scale output dimension must match weight output dimension")
    if groups * 64 != input_dims:
        raise ValueError("hotcold_gather_qmv_affine4_pair requires group_size=64")
    if packed_cols * 8 != input_dims:
        raise ValueError("packed_cols must equal input_dims / 8 for 4-bit weights")
    if input_dims % 512 != 0:
        raise ValueError("input_dims must be a multiple of 512 for the fast QMV pair probe")
    if output_dims % 8 != 0:
        raise ValueError("output_dims must be a multiple of 8 for the fast QMV pair probe")

    kernel = _affine4_qmv_pair_fast_kernel(
        route_count=route_count,
        input_dims=input_dims,
        output_dims=output_dims,
        packed_cols=packed_cols,
        groups=groups,
    )
    out_a, out_b = kernel(
        inputs=[
            x,
            hot_a_weight,
            hot_a_scales,
            hot_a_biases,
            cold_a_weight,
            cold_a_scales,
            cold_a_biases,
            hot_b_weight,
            hot_b_scales,
            hot_b_biases,
            cold_b_weight,
            cold_b_scales,
            cold_b_biases,
            route_source,
            local_indices,
        ],
        template=[("T", _dtype(x))],
        grid=(32, 2, route_count * (output_dims // 8)),
        threadgroup=(32, 2, 1),
        output_shapes=[(route_count, output_dims), (route_count, output_dims)],
        output_dtypes=[_dtype(x), _dtype(x)],
    )
    return out_a, out_b


def hotcold_gather_qmv_affine4_pair_single(
    x: Any,
    hot_a_weight: Any,
    hot_a_scales: Any,
    hot_a_biases: Any,
    cold_a_weight: Any,
    cold_a_scales: Any,
    cold_a_biases: Any,
    hot_b_weight: Any,
    hot_b_scales: Any,
    hot_b_biases: Any,
    cold_b_weight: Any,
    cold_b_scales: Any,
    cold_b_biases: Any,
    route_source: Any,
    local_indices: Any,
) -> tuple[Any, Any]:
    """Run paired hot/cold QMV for one input row shared by all route lanes."""

    import mlx.core as mx

    x_shape = _shape(x)
    a_w_shape = _shape(hot_a_weight)
    b_w_shape = _shape(hot_b_weight)
    cold_a_w_shape = _shape(cold_a_weight)
    cold_b_w_shape = _shape(cold_b_weight)
    a_s_shape = _shape(hot_a_scales)
    b_s_shape = _shape(hot_b_scales)
    route_shape = _shape(route_source)
    local_shape = _shape(local_indices)

    if x_shape[:1] != (1,) or len(x_shape) != 2:
        raise ValueError("x must have shape (1, input_dims)")
    if len(a_w_shape) != 3 or len(b_w_shape) != 3:
        raise ValueError("hot projection weights must be rank-3 arrays")
    if a_w_shape != b_w_shape:
        raise ValueError("paired projections must have matching hot weight shapes")
    if cold_a_w_shape[1:] != a_w_shape[1:] or cold_b_w_shape[1:] != b_w_shape[1:]:
        raise ValueError("cold weights must match hot trailing dimensions")
    if route_shape != local_shape or len(route_shape) != 1:
        raise ValueError("route_source and local_indices must be 1D arrays")
    if _shape(cold_a_scales)[1:] != a_s_shape[1:] or _shape(cold_b_scales)[1:] != b_s_shape[1:]:
        raise ValueError("cold scales/biases must match hot scales/biases trailing dimensions")
    if _shape(hot_a_biases) != a_s_shape or _shape(cold_a_biases) != _shape(cold_a_scales):
        raise ValueError("projection A bias shapes must match scale shapes")
    if _shape(hot_b_biases) != b_s_shape or _shape(cold_b_biases) != _shape(cold_b_scales):
        raise ValueError("projection B bias shapes must match scale shapes")
    if a_s_shape != b_s_shape:
        raise ValueError("paired projections must have matching scale shapes")

    route_count = route_shape[0]
    _input_rows, input_dims = x_shape
    _hot_slots, output_dims, packed_cols = a_w_shape
    groups = a_s_shape[2]
    if a_s_shape[1] != output_dims:
        raise ValueError("scale output dimension must match weight output dimension")
    if groups * 64 != input_dims:
        raise ValueError("hotcold_gather_qmv_affine4_pair_single requires group_size=64")
    if packed_cols * 8 != input_dims:
        raise ValueError("packed_cols must equal input_dims / 8 for 4-bit weights")
    if input_dims % 512 != 0:
        raise ValueError("input_dims must be a multiple of 512 for the fast QMV pair-single probe")
    if output_dims % 8 != 0:
        raise ValueError("output_dims must be a multiple of 8 for the fast QMV pair-single probe")

    kernel = _affine4_qmv_pair_fast_kernel(
        route_count=route_count,
        input_dims=input_dims,
        output_dims=output_dims,
        packed_cols=packed_cols,
        groups=groups,
        single_input=True,
    )
    out_a, out_b = kernel(
        inputs=[
            x,
            hot_a_weight,
            hot_a_scales,
            hot_a_biases,
            cold_a_weight,
            cold_a_scales,
            cold_a_biases,
            hot_b_weight,
            hot_b_scales,
            hot_b_biases,
            cold_b_weight,
            cold_b_scales,
            cold_b_biases,
            route_source,
            local_indices,
        ],
        template=[("T", _dtype(x))],
        grid=(32, 2, route_count * (output_dims // 8)),
        threadgroup=(32, 2, 1),
        output_shapes=[(route_count, output_dims), (route_count, output_dims)],
        output_dtypes=[_dtype(x), _dtype(x)],
    )
    return out_a, out_b


def hotcold_gather_qmv_affine4_pair_norepeat(
    x: Any,
    hot_a_weight: Any,
    hot_a_scales: Any,
    hot_a_biases: Any,
    cold_a_weight: Any,
    cold_a_scales: Any,
    cold_a_biases: Any,
    hot_b_weight: Any,
    hot_b_scales: Any,
    hot_b_biases: Any,
    cold_b_weight: Any,
    cold_b_scales: Any,
    cold_b_biases: Any,
    route_source: Any,
    local_indices: Any,
    *,
    top_k: int,
) -> tuple[Any, Any]:
    """Run paired hot/cold QMV without materializing repeated per-route inputs."""

    import mlx.core as mx

    x_shape = _shape(x)
    a_w_shape = _shape(hot_a_weight)
    b_w_shape = _shape(hot_b_weight)
    cold_a_w_shape = _shape(cold_a_weight)
    cold_b_w_shape = _shape(cold_b_weight)
    a_s_shape = _shape(hot_a_scales)
    b_s_shape = _shape(hot_b_scales)
    route_shape = _shape(route_source)
    local_shape = _shape(local_indices)

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if len(x_shape) != 2:
        raise ValueError("x must have shape (input_rows, input_dims)")
    if len(a_w_shape) != 3 or len(b_w_shape) != 3:
        raise ValueError("hot projection weights must be rank-3 arrays")
    if a_w_shape != b_w_shape:
        raise ValueError("paired projections must have matching hot weight shapes")
    if cold_a_w_shape[1:] != a_w_shape[1:] or cold_b_w_shape[1:] != b_w_shape[1:]:
        raise ValueError("cold weights must match hot trailing dimensions")
    if route_shape != local_shape or len(route_shape) != 1:
        raise ValueError("route_source and local_indices must be 1D arrays")
    if _shape(cold_a_scales)[1:] != a_s_shape[1:] or _shape(cold_b_scales)[1:] != b_s_shape[1:]:
        raise ValueError("cold scales/biases must match hot scales/biases trailing dimensions")
    if _shape(hot_a_biases) != a_s_shape or _shape(cold_a_biases) != _shape(cold_a_scales):
        raise ValueError("projection A bias shapes must match scale shapes")
    if _shape(hot_b_biases) != b_s_shape or _shape(cold_b_biases) != _shape(cold_b_scales):
        raise ValueError("projection B bias shapes must match scale shapes")
    if a_s_shape != b_s_shape:
        raise ValueError("paired projections must have matching scale shapes")

    input_rows, input_dims = x_shape
    route_count = route_shape[0]
    if input_rows * int(top_k) != route_count:
        raise ValueError("route count must equal input rows * top_k")
    _hot_slots, output_dims, packed_cols = a_w_shape
    groups = a_s_shape[2]
    if a_s_shape[1] != output_dims:
        raise ValueError("scale output dimension must match weight output dimension")
    if groups * 64 != input_dims:
        raise ValueError("hotcold_gather_qmv_affine4_pair_norepeat requires group_size=64")
    if packed_cols * 8 != input_dims:
        raise ValueError("packed_cols must equal input_dims / 8 for 4-bit weights")
    if input_dims % 512 != 0:
        raise ValueError("input_dims must be a multiple of 512 for the fast QMV pair-norepeat probe")
    if output_dims % 8 != 0:
        raise ValueError("output_dims must be a multiple of 8 for the fast QMV pair-norepeat probe")

    kernel = _affine4_qmv_pair_fast_kernel(
        route_count=route_count,
        input_dims=input_dims,
        output_dims=output_dims,
        packed_cols=packed_cols,
        groups=groups,
        norepeat_top_k=int(top_k),
    )
    out_a, out_b = kernel(
        inputs=[
            x,
            hot_a_weight,
            hot_a_scales,
            hot_a_biases,
            cold_a_weight,
            cold_a_scales,
            cold_a_biases,
            hot_b_weight,
            hot_b_scales,
            hot_b_biases,
            cold_b_weight,
            cold_b_scales,
            cold_b_biases,
            route_source,
            local_indices,
        ],
        template=[("T", _dtype(x))],
        grid=(32, 2, route_count * (output_dims // 8)),
        threadgroup=(32, 2, 1),
        output_shapes=[(route_count, output_dims), (route_count, output_dims)],
        output_dtypes=[_dtype(x), _dtype(x)],
    )
    return out_a, out_b


def hotcold_gather_qmv_affine4_norepeat(
    x: Any,
    hot_weight: Any,
    hot_scales: Any,
    hot_biases: Any,
    cold_weight: Any,
    cold_scales: Any,
    cold_biases: Any,
    route_source: Any,
    local_indices: Any,
    *,
    top_k: int,
) -> Any:
    """Run hot/cold QMV without materializing repeated per-route inputs.

    ``x`` has shape ``(input_rows, input_dims)`` and each input row owns
    ``top_k`` adjacent route lanes. This is equivalent to calling
    ``hotcold_gather_qmv_affine4(mx.repeat(x, top_k, axis=0), ...)`` but skips
    that repeat table.
    """

    import mlx.core as mx

    x_shape = _shape(x)
    hot_w_shape = _shape(hot_weight)
    cold_w_shape = _shape(cold_weight)
    hot_s_shape = _shape(hot_scales)
    cold_s_shape = _shape(cold_scales)
    route_shape = _shape(route_source)
    local_shape = _shape(local_indices)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if len(x_shape) != 2:
        raise ValueError("x must have shape (input_rows, input_dims)")
    if len(hot_w_shape) != 3 or len(cold_w_shape) != 3:
        raise ValueError("hot_weight and cold_weight must be rank-3 arrays")
    if route_shape != local_shape or len(route_shape) != 1:
        raise ValueError("route_source and local_indices must be 1D arrays")

    input_rows, input_dims = x_shape
    route_count = route_shape[0]
    if input_rows * int(top_k) != route_count:
        raise ValueError("route count must equal input rows * top_k")
    _hot_slots, output_dims, packed_cols = hot_w_shape
    if cold_w_shape[1:] != hot_w_shape[1:]:
        raise ValueError("cold_weight must match hot_weight trailing dimensions")
    if cold_s_shape[1:] != hot_s_shape[1:]:
        raise ValueError("cold scales/biases must match hot scales/biases trailing dimensions")
    if _shape(hot_biases) != hot_s_shape or _shape(cold_biases) != cold_s_shape:
        raise ValueError("bias shapes must match scale shapes")
    groups = hot_s_shape[2]
    if hot_s_shape[1] != output_dims:
        raise ValueError("scale output dimension must match weight output dimension")
    if groups * 64 != input_dims:
        raise ValueError("hotcold_gather_qmv_affine4_norepeat requires group_size=64")
    if packed_cols * 8 != input_dims:
        raise ValueError("packed_cols must equal input_dims / 8 for 4-bit weights")
    if input_dims % 512 != 0:
        raise ValueError("input_dims must be a multiple of 512 for the fast QMV norepeat probe")
    if output_dims % 8 != 0:
        raise ValueError("output_dims must be a multiple of 8 for the fast QMV norepeat probe")

    kernel = _affine4_qmv_norepeat_fast_kernel(
        route_count=route_count,
        top_k=int(top_k),
        input_dims=input_dims,
        output_dims=output_dims,
        packed_cols=packed_cols,
        groups=groups,
    )
    return kernel(
        inputs=[
            x,
            hot_weight,
            hot_scales,
            hot_biases,
            cold_weight,
            cold_scales,
            cold_biases,
            route_source,
            local_indices,
        ],
        template=[("T", _dtype(x))],
        grid=(32, 2, route_count * (output_dims // 8)),
        threadgroup=(32, 2, 1),
        output_shapes=[(route_count, output_dims)],
        output_dtypes=[_dtype(x)],
    )[0]


def materialized_hotcold_gather_qmm_affine4(
    x: Any,
    hot_weight: Any,
    hot_scales: Any,
    hot_biases: Any,
    cold_weight: Any,
    cold_scales: Any,
    cold_biases: Any,
    route_source: Any,
    local_indices: Any,
    *,
    group_size: int = 64,
) -> Any:
    """Reference implementation using row materialization plus ``mx.gather_qmm``."""

    import mlx.core as mx
    import numpy as np

    rows_w = []
    rows_s = []
    rows_b = []
    for src, row in zip(
        np.asarray(route_source).reshape(-1).tolist(),
        np.asarray(local_indices).reshape(-1).tolist(),
        strict=True,
    ):
        if int(src) == 0:
            rows_w.append(hot_weight[int(row) : int(row) + 1])
            rows_s.append(hot_scales[int(row) : int(row) + 1])
            rows_b.append(hot_biases[int(row) : int(row) + 1])
        else:
            rows_w.append(cold_weight[int(row) : int(row) + 1])
            rows_s.append(cold_scales[int(row) : int(row) + 1])
            rows_b.append(cold_biases[int(row) : int(row) + 1])
    table_w = mx.concatenate(rows_w, axis=0)
    table_s = mx.concatenate(rows_s, axis=0)
    table_b = mx.concatenate(rows_b, axis=0)
    return mx.gather_qmm(
        x[:, None, :],
        table_w,
        table_s,
        table_b,
        transpose=True,
        group_size=group_size,
        bits=4,
        mode="affine",
        sorted_indices=False,
    ).squeeze(1)


def materialized_hotcold_gather_qmm_affine8(
    x: Any,
    hot_weight: Any,
    hot_scales: Any,
    hot_biases: Any,
    cold_weight: Any,
    cold_scales: Any,
    cold_biases: Any,
    route_source: Any,
    local_indices: Any,
    *,
    group_size: int = 64,
) -> Any:
    """Reference affine-8 hot/cold projection using row materialization."""

    import mlx.core as mx
    import numpy as np

    rows_w = []
    rows_s = []
    rows_b = []
    for src, row in zip(
        np.asarray(route_source).reshape(-1).tolist(),
        np.asarray(local_indices).reshape(-1).tolist(),
        strict=True,
    ):
        if int(src) == 0:
            rows_w.append(hot_weight[int(row) : int(row) + 1])
            rows_s.append(hot_scales[int(row) : int(row) + 1])
            rows_b.append(hot_biases[int(row) : int(row) + 1])
        else:
            rows_w.append(cold_weight[int(row) : int(row) + 1])
            rows_s.append(cold_scales[int(row) : int(row) + 1])
            rows_b.append(cold_biases[int(row) : int(row) + 1])
    table_w = mx.concatenate(rows_w, axis=0)
    table_s = mx.concatenate(rows_s, axis=0)
    table_b = mx.concatenate(rows_b, axis=0)
    return mx.gather_qmm(
        x[:, None, :],
        table_w,
        table_s,
        table_b,
        transpose=True,
        group_size=group_size,
        bits=8,
        mode="affine",
        sorted_indices=False,
    ).squeeze(1)
