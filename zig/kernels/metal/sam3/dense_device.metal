constant uint sam_dense_device_rows = 32;
constant uint sam_dense_device_columns = 64;

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_device_clear(device uint* domain, uint at [[thread_position_in_grid]]) {
    if (at == 0) domain[0] = 0;
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_device_domain(const device float* x, const device float* weight, const device float* bias,
    device atomic_uint* domain, constant SamDenseShape& shape, uint at [[thread_position_in_grid]],
    uint width [[threads_per_grid]]) {
    const ulong nx = ulong(shape.rows) * shape.depth, nw = ulong(shape.columns) * shape.depth;
    uint invalid = 0;
    for (ulong i = at; i < nx + nw + shape.columns; i += width) {
        const float value = i < nx ? x[i] : i < nx + nw ? weight[i - nx] : bias[i - nx - nw];
        const uint bits = as_type<uint>(value) & 0x7fffffffu;
        // This domain keeps even exact product cancellation above the FP32 subnormal range.
        invalid |= uint(bits != 0 && (bits < 0x2d800000u || bits > 0x51800000u));
    }
    if (invalid) atomic_fetch_or_explicit(domain, invalid, memory_order_relaxed);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_device_plan(device uint* domain, constant SamDenseShape& shape, uint at [[thread_position_in_grid]]) {
    if (at != 0) return;
    const bool fallback = domain[0] != 0;
    domain[1] = fallback ? 0 : (shape.columns + sam_dense_device_columns - 1) / sam_dense_device_columns;
    domain[2] = (shape.rows + sam_dense_device_rows - 1) / sam_dense_device_rows;
    domain[3] = 1;
    domain[4] = fallback ? (shape.columns + 3) / 4 : 0;
    domain[5] = shape.rows;
    domain[6] = 1;
}

template<bool KN>
inline void sam_dense_device_impl(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* domain,
    uint2 group) {
    if (domain[0]) return;
    using namespace mpp::tensor_ops;
    using Input = tensor<device float, dextents<int, 2>, tensor_inline>;
    Input a((device float*)x, dextents<int, 2>(shape.depth, shape.rows), array<int, 2>({1, int(shape.depth)}));
    Input b((device float*)weight, dextents<int, 2>(KN ? shape.columns : shape.depth, KN ? shape.depth : shape.columns), array<int, 2>({1, KN ? int(shape.columns) : int(shape.depth)}));
    auto left = a.slice(0, int(group.y) * sam_dense_device_rows);
    auto right = b.slice(KN ? int(group.x) * sam_dense_device_columns : 0, KN ? 0 : int(group.x) * sam_dense_device_columns);
    constexpr auto descriptor = matmul2d_descriptor(sam_dense_device_rows, sam_dense_device_columns, int(dynamic_extent), false, !KN, false);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    auto result = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>, metal::remove_addrspace_t<decltype(right)>, float>();
    op.run(left, right, result);
    TF_UNROLL
    for (ushort e = 0; e < result.get_capacity(); ++e) {
        const auto at = result.get_multidimensional_index(e);
        const uint row = group.y * sam_dense_device_rows + at[1], col = group.x * sam_dense_device_columns + at[0];
        if (row < shape.rows && col < shape.columns) {
            out[long(row) * shape.columns + col] = result[e] + bias[col];
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_device(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* domain,
    uint2 group [[threadgroup_position_in_grid]]) {
    sam_dense_device_impl<false>(x, weight, bias, out, shape, domain, group);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_device_nn(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* domain,
    uint2 group [[threadgroup_position_in_grid]]) {
    sam_dense_device_impl<true>(x, weight, bias, out, shape, domain, group);
}

template<bool KN, ushort CHUNK = 64, bool GELU = false>
inline void sam_dense_device_compensated_impl(const device float*, const device float*, const device float*,
    device float*, constant SamDenseShape&, const device uint*, uint2);

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_device_gelu(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* domain,
    uint2 group [[threadgroup_position_in_grid]]) {
    sam_dense_device_compensated_impl<false, 32, true>(x, weight, bias, out, shape, domain, group);
}

template<bool KN, ushort CHUNK, bool GELU>
inline void sam_dense_device_compensated_impl(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* domain,
    uint2 group) {
    if (domain[0]) return;
    using namespace mpp::tensor_ops;
    using Input = tensor<device float, dextents<int, 2>, tensor_inline>;
    Input a((device float*)x, dextents<int, 2>(shape.depth, shape.rows), array<int, 2>({1, int(shape.depth)}));
    Input b((device float*)weight, dextents<int, 2>(KN ? shape.columns : shape.depth, KN ? shape.depth : shape.columns), array<int, 2>({1, KN ? int(shape.columns) : int(shape.depth)}));
    auto left = a.slice(0, int(group.y) * sam_dense_device_rows);
    auto right = b.slice(KN ? int(group.x) * sam_dense_device_columns : 0, KN ? 0 : int(group.x) * sam_dense_device_columns);
    constexpr auto descriptor = matmul2d_descriptor(sam_dense_device_rows, sam_dense_device_columns, CHUNK, false, !KN, false);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    auto product = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>, metal::remove_addrspace_t<decltype(right)>, float>();
    float2 sums[32];
    TF_UNROLL
    for (ushort e = 0; e < 32; ++e) sums[e] = float2(0);
    for (uint first = 0; first < shape.depth; first += CHUNK) {
        auto x_part = a.slice(first, int(group.y) * sam_dense_device_rows);
        auto w_part = b.slice(KN ? int(group.x) * sam_dense_device_columns : int(first), KN ? int(first) : int(group.x) * sam_dense_device_columns);
        op.run(x_part, w_part, product);
        TF_UNROLL
        for (ushort e = 0; e < product.get_capacity(); ++e) sam_pair_add(sums[e], product[e]);
    }
    TF_UNROLL
    for (ushort e = 0; e < product.get_capacity(); ++e) {
        const auto at = product.get_multidimensional_index(e);
        const uint row = group.y * sam_dense_device_rows + at[1], col = group.x * sam_dense_device_columns + at[0];
        if (row < shape.rows && col < shape.columns) {
            sam_pair_add(sums[e], bias[col]);
            const float value = sums[e].x + sums[e].y;
            if constexpr (GELU) out[long(row) * shape.columns + col] = sam_gelu_fp32(value);
            else out[long(row) * shape.columns + col] = value;
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_device_gelu_linear(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* domain,
    uint2 group [[threadgroup_position_in_grid]]) {
    sam_dense_device_compensated_impl<false, 32>(x, weight, bias, out, shape, domain, group);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_device_compensated(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* domain,
    uint2 group [[threadgroup_position_in_grid]]) {
    sam_dense_device_compensated_impl<false>(x, weight, bias, out, shape, domain, group);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_device_nn_compensated(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* domain,
    uint2 group [[threadgroup_position_in_grid]]) {
    sam_dense_device_compensated_impl<true>(x, weight, bias, out, shape, domain, group);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_device_fallback(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* domain,
    uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (!domain[0]) return;
    sam_dense_small<false>(x, weight, bias, out, shape, group, sg, lane);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_device_nn_fallback(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* domain,
    uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (!domain[0]) return;
    sam_dense_small<false, true>(x, weight, bias, out, shape, group, sg, lane);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_device_gelu_fallback(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* domain,
    uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (!domain[0]) return;
    sam_dense_small<false>(x, weight, bias, out, shape, group, sg, lane);
    const uint column = group.x * 4 + sg;
    if (lane == 0 && column < shape.columns && group.y < shape.rows) {
        const long at = long(group.y) * shape.columns + column;
        const float value = out[at];
        out[at] = sam_gelu_fp32(value);
    }
}
