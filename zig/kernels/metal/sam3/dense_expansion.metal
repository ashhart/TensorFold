struct SamDenseResiduals { atomic_uint count; uint indices[4]; float values[4]; };

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_expansion_clear(device uint* domain, constant SamDenseShape& shape,
    device SamDenseResiduals* residuals, uint at [[thread_position_in_grid]]) {
    if (at < 8) domain[at] = 0;
    if (at < shape.rows) atomic_store_explicit(&residuals[at].count, 0u, memory_order_relaxed);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_expansion_pack(const device float* x, const device float* weight, device half* packed,
    device atomic_uint* domain, constant SamDenseShape& shape, uint at [[thread_position_in_grid]],
    uint width [[threads_per_grid]], device SamDenseResiduals* residuals [[buffer(5)]]) {
    if (atomic_load_explicit(domain, memory_order_relaxed)) return;
    const ulong nx = ulong(shape.rows) * shape.depth, nw = ulong(shape.columns) * shape.depth;
    uint invalid = 0, present = 0;
    for (ulong i = at; i < nx + nw; i += width) {
        const float value = i < nx ? x[i] : weight[i - nx];
        const half high = half(value);
        const float remainder = value - float(high);
        if (i >= nx) {
            packed[3 * nx + i - nx] = high;
            present |= uint(value != 0) << 3;
            invalid |= uint(!isfinite(float(high)) || (value != 0 && as_type<uint>(float(high)) != as_type<uint>(value)));
        } else {
            const half middle = half(ldexp(remainder, 11));
            const float tail = remainder - ldexp(float(middle), -11);
            const half low = half(ldexp(tail, 22));
            packed[i] = high;
            packed[i + nx] = middle;
            packed[i + 2 * nx] = low;
            present |= uint(float(high) != 0);
            present |= uint(float(middle) != 0) << 1;
            present |= uint(float(low) != 0) << 2;
            const float reconstructed = (float(high) + ldexp(float(middle), -11)) + ldexp(float(low), -22);
            const float residual = value - reconstructed;
            invalid |= uint(!isfinite(residual) || (value != 0 && as_type<uint>(reconstructed + residual) != as_type<uint>(value)));
            if (residual != 0 && isfinite(residual)) {
                device SamDenseResiduals& row = residuals[i / shape.depth];
                const uint slot = atomic_fetch_add_explicit(&row.count, 1u, memory_order_relaxed);
                if (slot < 4) { row.indices[slot] = uint(i % shape.depth); row.values[slot] = residual; }
                else invalid |= 1u;
            }
        }
    }
    if (invalid) atomic_fetch_or_explicit(domain, 1u, memory_order_relaxed);
    present = simd_or(present);
    if ((at & 31u) == 0 && present) atomic_fetch_or_explicit(domain + 7, present, memory_order_relaxed);
}

template<bool GELU, bool RESIDUAL = true>
inline void sam_dense_expansion_impl(const device half* packed, const device float* bias,
    device float* output, constant SamDenseShape& shape, const device uint* domain,
    device SamDenseResiduals* residuals, uint2 group) {
    if (domain[0]) return;
    using namespace mpp::tensor_ops;
    using Input = tensor<device half, dextents<int, 2>, tensor_inline>;
    const ulong nx = ulong(shape.rows) * shape.depth;
    Input x((device half*)packed, dextents<int, 2>(shape.depth, shape.rows), array<int, 2>({1, int(shape.depth)}));
    Input w((device half*)packed + 3 * nx, dextents<int, 2>(shape.depth, shape.columns), array<int, 2>({1, int(shape.depth)}));
    const int row = int(group.y) * 32, column = int(group.x) * 64;
    auto left = x.slice(0, row), right = w.slice(0, column);
    constexpr auto high_descriptor = matmul2d_descriptor(32, 64, 64, false, true, false);
    constexpr auto correction_descriptor = matmul2d_descriptor(32, 64, int(dynamic_extent), false, true, false);
    matmul2d<high_descriptor, execution_simdgroups<2>> high_op;
    matmul2d<correction_descriptor, execution_simdgroups<2>> correction_op;
    auto product = high_op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>, metal::remove_addrspace_t<decltype(right)>, float>();
    float2 sums[32];
    TF_UNROLL
    for (ushort e = 0; e < 32; ++e) sums[e] = float2(0);
    const uint present = domain[7];
    if ((present & 9u) == 9u) for (uint first = 0; first < shape.depth; first += 64) {
        auto x_part = x.slice(first, row), w_part = w.slice(first, column);
        high_op.run(x_part, w_part, product);
        TF_UNROLL
        for (ushort e = 0; e < product.get_capacity(); ++e) sam_pair_add(sums[e], product[e]);
    }
    TF_UNROLL
    for (ushort x_part = 1; x_part < 3; ++x_part) {
        if (!(present & (1u << x_part))) continue;
        Input xc((device half*)packed + x_part * nx, dextents<int, 2>(shape.depth, shape.rows), array<int, 2>({1, int(shape.depth)}));
        auto xl = xc.slice(0, row), wr = w.slice(0, column);
        auto correction = correction_op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(xl)>, metal::remove_addrspace_t<decltype(wr)>, float>();
        correction_op.run(xl, wr, correction);
        TF_UNROLL
        for (ushort e = 0; e < correction.get_capacity(); ++e) sam_pair_add(sums[e], ldexp(correction[e], -11 * int(x_part)));
    }
    TF_UNROLL
    for (ushort e = 0; e < product.get_capacity(); ++e) {
        const auto at = product.get_multidimensional_index(e);
        const uint r = row + at[1], c = column + at[0];
        if (r < shape.rows && c < shape.columns) {
            if constexpr (RESIDUAL) {
                device SamDenseResiduals& rare = residuals[r];
                const uint count = min(atomic_load_explicit(&rare.count, memory_order_relaxed), 4u);
                uint first = 0;
                for (uint j = 0; j < count; ++j) {
                    uint selected = 0, index = 0xffffffffu;
                    for (uint k = 0; k < count; ++k) if (rare.indices[k] >= first && rare.indices[k] < index) {
                        selected = k;
                        index = rare.indices[k];
                    }
                    const float value = rare.values[selected], weight = float(packed[3 * nx + ulong(c) * shape.depth + index]);
                    const float correction = value * weight;
                    sam_pair_add(sums[e], correction);
                    sam_pair_add(sums[e], fma(value, weight, -correction));
                    first = index + 1;
                }
            }
            sam_pair_add(sums[e], bias[c]);
            const float value = sums[e].x + sums[e].y;
            if constexpr (GELU) output[long(r) * shape.columns + c] = sam_gelu_fp32(value);
            else output[long(r) * shape.columns + c] = value;
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(64)]]
void sam_dense_expansion(const device half* packed, const device float* bias, device float* output,
    constant SamDenseShape& shape, const device uint* domain, device SamDenseResiduals* residuals,
    uint2 group [[threadgroup_position_in_grid]]) {
    sam_dense_expansion_impl<false>(packed, bias, output, shape, domain, residuals, group);
}

[[kernel, max_total_threads_per_threadgroup(64)]]
void sam_dense_expansion_gelu(const device half* packed, const device float* bias, device float* output,
    constant SamDenseShape& shape, const device uint* domain, device SamDenseResiduals* residuals,
    uint2 group [[threadgroup_position_in_grid]]) {
    sam_dense_expansion_impl<true>(packed, bias, output, shape, domain, residuals, group);
}

[[kernel, max_total_threads_per_threadgroup(64)]]
void sam_dense_expansion_drop_residual(const device half* packed, const device float* bias, device float* output,
    constant SamDenseShape& shape, const device uint* domain, device SamDenseResiduals* residuals,
    uint2 group [[threadgroup_position_in_grid]]) {
    sam_dense_expansion_impl<false, false>(packed, bias, output, shape, domain, residuals, group);
}
