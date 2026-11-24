// Products below 2^-126 flush to zero, so only non-finite operands or magnitudes above 2^36 set the fallback bit.
[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_wide_domain(const device float* x, const device float* weight, const device float* bias,
    device atomic_uint* domain, constant SamDenseShape& shape, uint at [[thread_position_in_grid]],
    uint width [[threads_per_grid]]) {
    const ulong nx = ulong(shape.rows) * shape.depth, nw = ulong(shape.columns) * shape.depth;
    uint invalid = 0;
    for (ulong i = at; i < nx + nw + shape.columns; i += width) {
        const float value = i < nx ? x[i] : i < nx + nw ? weight[i - nx] : bias[i - nx - nw];
        invalid |= uint((as_type<uint>(value) & 0x7fffffffu) > 0x51800000u);
    }
    if (invalid) atomic_fetch_or_explicit(domain, invalid, memory_order_relaxed);
}

inline uint sam_dense_wide_outside(float4 value) {
    const uint4 bits = as_type<uint4>(value) & 0x7fffffffu;
    return uint(any(bits > 0x51800000u));
}

// The weight and bias half of the domain, once per constant weight: `flags[0]` must start at zero.
[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_wide_operands(const device float* weight, const device float* bias, device atomic_uint* flags,
    constant SamDenseShape& shape, uint at [[thread_position_in_grid]], uint width [[threads_per_grid]]) {
    const ulong nw = ulong(shape.columns) * shape.depth;
    uint invalid = 0;
    for (ulong i = at; i < nw + shape.columns; i += width)
        invalid |= uint((as_type<uint>(i < nw ? weight[i] : bias[i - nw]) & 0x7fffffffu) > 0x51800000u);
    if (invalid) atomic_fetch_or_explicit(flags, invalid, memory_order_relaxed);
}

// The input half of the domain on top of the cached operand flags; `domain[0]` starts as `flags[0]`.
[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_wide_seed(device uint* domain, const device uint* flags, uint at [[thread_position_in_grid]]) {
    if (at == 0) domain[0] = flags[0];
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam_dense_wide_rows(const device float4* x, device atomic_uint* domain, constant uint& count,
    uint at [[thread_position_in_grid]], uint width [[threads_per_grid]]) {
    uint invalid = 0;
    for (uint i = at; i < count / 4; i += width) invalid |= sam_dense_wide_outside(x[i]);
    if (at < count % 4) invalid |= sam_dense_wide_outside(float4(((const device float*)x)[count - count % 4 + at]));
    invalid = simd_or(invalid);
    if (invalid && simd_is_first()) atomic_fetch_or_explicit(domain, invalid, memory_order_relaxed);
}

// Merges precomputed weight and input flags into the indirect plan; sets out_flag[0] = fallback when emit_flag != 0.
[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_wide_plan(device uint* domain, const device uint* weight_flags, const device uint* input_flags,
    device uint* out_flag, constant SamDenseShape& shape, constant uint& emit_flag,
    uint at [[thread_position_in_grid]]) {
    if (at != 0) return;
    const bool fallback = (weight_flags[0] | input_flags[0]) != 0;
    domain[0] = uint(fallback);
    domain[1] = fallback ? 0 : (shape.columns + 63) / 64;
    domain[2] = (shape.rows + 31) / 32;
    domain[3] = 1;
    domain[4] = fallback ? (shape.columns + 3) / 4 : 0;
    domain[5] = shape.rows;
    domain[6] = 1;
    if (emit_flag) out_flag[0] = uint(fallback);
}

// KC 0: one FP32 accumulation over depth, else KC-deep compensated chunks. EPI 0 plain, 2 residual, 3 tiled residual.
template<ushort KC, bool GELU, ushort EPI = 0, bool OUT_FLAG = false>
inline void sam_dense_wide_impl(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* domain, const device float* residual,
    device atomic_uint* out_flag, uint2 group) {
    if (domain[0]) return;
    using namespace mpp::tensor_ops;
    using Input = tensor<device float, dextents<int, 2>, tensor_inline>;
    const uint gx = (shape.columns + 63) / 64, gy = (shape.rows + 31) / 32;
    const uint linear = group.y * gx + group.x, band = 8u;
    const uint band_start = (linear / (gx * band)) * band;
    const uint band_rows = min(band, gy - band_start);
    const uint in_band = linear % (gx * band);
    const int row = int(band_start + (in_band % band_rows)) * 32;
    const int column = int(in_band / band_rows) * 64, depth = int(shape.depth);
    Input a((device float*)x, dextents<int, 2>(depth, int(shape.rows)), array<int, 2>({1, depth}));
    Input b((device float*)weight, dextents<int, 2>(depth, int(shape.columns)), array<int, 2>({1, depth}));
    auto left = a.slice(0, row);
    auto right = b.slice(0, column);
    constexpr auto descriptor = matmul2d_descriptor(32, 64, KC == 0 ? int(dynamic_extent) : int(KC), false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    auto product = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>, metal::remove_addrspace_t<decltype(right)>, float>();
    float2 sums[16];
    TF_UNROLL
    for (ushort e = 0; e < 16; ++e) sums[e] = float2(0);
    if constexpr (KC == 0) {
        op.run(left, right, product);
    } else {
        for (int first = 0; first < depth; first += KC) {
            auto x_part = a.slice(first, row);
            auto w_part = b.slice(first, column);
            op.run(x_part, w_part, product);
            TF_UNROLL
            for (ushort e = 0; e < 16; ++e) {
                const float y = product[e] - sums[e].y, t = sums[e].x + y;
                sums[e].y = (t - sums[e].x) - y;
                sums[e].x = t;
            }
        }
    }
    uint invalid = 0;
    TF_UNROLL
    for (ushort e = 0; e < 16; ++e) {
        const auto at = product.get_multidimensional_index(e);
        const uint r = row + at[1], c = column + at[0];
        if (r >= shape.rows || c >= shape.columns) continue;
        float value = product[e] + bias[c];
        if constexpr (KC != 0) {
            float2 sum = sam_two_sum(sums[e].x, -sums[e].y);
            sam_pair_add(sum, bias[c]);
            value = sum.x + sum.y;
        }
        const float out_val = GELU ? sam_gelu_fp32(value) : value;
        const ulong target = ulong(EPI == 3 ? sam_planes_tile_row(r) : r) * shape.columns + c;
        const float written = EPI >= 2 ? residual[target] + out_val : out_val;
        out[target] = written;
        if constexpr (OUT_FLAG) invalid |= uint((as_type<uint>(written) & 0x7fffffffu) > 0x51800000u);
    }
    if constexpr (OUT_FLAG) {
        invalid = simd_or(invalid);
        if (invalid && simd_is_first()) atomic_fetch_or_explicit(out_flag, 1u, memory_order_relaxed);
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_wide_plain(const device float* x, const device float* weight, const device float* bias, device float* out,
    constant SamDenseShape& shape, const device uint* domain, uint2 group [[threadgroup_position_in_grid]]) {
    sam_dense_wide_impl<0, false>(x, weight, bias, out, shape, domain, nullptr, nullptr, group);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_wide_plain_residual(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* domain, const device float* residual,
    uint2 group [[threadgroup_position_in_grid]]) {
    sam_dense_wide_impl<0, false, 2>(x, weight, bias, out, shape, domain, residual, nullptr, group);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_wide_plain_tiled(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* domain, const device float* residual,
    uint2 group [[threadgroup_position_in_grid]]) {
    sam_dense_wide_impl<0, false, 3>(x, weight, bias, out, shape, domain, residual, nullptr, group);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_wide_compensated(const device float* x, const device float* weight, const device float* bias, device float* out,
    constant SamDenseShape& shape, const device uint* domain, uint2 group [[threadgroup_position_in_grid]]) {
    if (shape.depth == 2304 && shape.columns == 256) sam_dense_wide_impl<256, false>(x, weight, bias, out, shape, domain, nullptr, nullptr, group);
    else sam_dense_wide_impl<64, false>(x, weight, bias, out, shape, domain, nullptr, nullptr, group);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_wide_compensated_residual(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* domain, const device float* residual,
    uint2 group [[threadgroup_position_in_grid]]) {
    sam_dense_wide_impl<64, false, 2>(x, weight, bias, out, shape, domain, residual, nullptr, group);
}

// 32-deep chunks keep the expansion within PyTorch's per-column FP64 error; 64 does not.
[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_wide_gelu(const device float* x, const device float* weight, const device float* bias, device float* out,
    constant SamDenseShape& shape, const device uint* domain, uint2 group [[threadgroup_position_in_grid]]) {
    sam_dense_wide_impl<32, true>(x, weight, bias, out, shape, domain, nullptr, nullptr, group);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_wide_gelu_flag(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* domain, device atomic_uint* out_flag,
    uint2 group [[threadgroup_position_in_grid]]) {
    sam_dense_wide_impl<32, true, 0, true>(x, weight, bias, out, shape, domain, nullptr, out_flag, group);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_wide_residual_fallback(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* domain, const device float* residual,
    uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (!domain[0]) return;
    sam_dense_small<false>(x, weight, bias, out, shape, group, sg, lane);
    const uint column = group.x * 4 + sg;
    if (lane == 0 && column < shape.columns && group.y < shape.rows) {
        const ulong target = ulong(group.y) * shape.columns + column;
        out[target] = residual[target] + out[target];
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_wide_tiled_fallback(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* domain, const device float* residual,
    uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (!domain[0]) return;
    const uint dst_row = sam_planes_tile_row(group.y);
    sam_dense_small_at<false>(x, weight, bias, out, shape, group, sg, lane, dst_row);
    const uint column = group.x * 4 + sg;
    if (lane == 0 && column < shape.columns && group.y < shape.rows) {
        const ulong target = ulong(dst_row) * shape.columns + column;
        out[target] = residual[target] + out[target];
    }
}
