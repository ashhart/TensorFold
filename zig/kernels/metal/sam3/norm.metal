inline float2 sam_norm_product(float2 a, float2 b) {
    const float high = a.x * b.x;
    return sam_two_sum(high, fma(a.x, b.x, -high) + a.x * b.y + a.y * b.x + a.y * b.y);
}

inline float2 sam_norm_sum(float2 value) {
    TF_UNROLL
    for (ushort offset = 16; offset; offset >>= 1) {
        const float high = simd_shuffle_xor(value.x, offset), low = simd_shuffle_xor(value.y, offset);
        sam_pair_add(value, high);
        sam_pair_add(value, low);
    }
    return value;
}

inline float sam_norm_apply(float x, float2 mean, float2 inv, float w, float b) {
    float2 delta = sam_two_sum(x, -mean.x);
    sam_pair_add(delta, -mean.y);
    delta = sam_norm_product(delta, inv);
    if (!isfinite(w) || !isfinite(b)) return (delta.x + delta.y) * w + b;
    sam_pair_scale(delta, w);
    sam_pair_add(delta, b);
    return delta.x + delta.y;
}

// OUT_MODE: 0 plain FP32, 1 dense_planes, 2 FP32 + wide flag, 3 dense_pair; tiled reads each tile row's grid row.
template<ushort OUT_MODE>
inline void sam_norm_compensated_impl(const device float* x, const device float* weight, const device float* bias,
    device float* out, device atomic_uint* out_flag, constant uint2& shape, constant float& eps, bool tiled,
    uint group, uint sg, uint lane) {
    const uint row = group * 4 + sg, width = shape.y;
    if (row >= shape.x) return;
    const ulong target = ulong(row) * width, base = tiled ? ulong(sam_planes_tile_row(row)) * width : target;
    const ulong total = ulong(shape.x) * width;
    uint invalid = 0;
    float vals[32];
    float2 sum(0);
    if (width == 1024) {
        TF_UNROLL
        for (ushort k = 0; k < 32; ++k) {
            const float v = x[base + uint(k) * 32u + lane];
            vals[k] = v;
            sam_pair_add(sum, v);
        }
    } else {
        for (uint col = lane; col < width; col += 32) sam_pair_add(sum, x[base + col]);
    }
    float2 mean = sam_norm_sum(sum);
    const float inverse_width = 1.0f / float(width);
    const float2 divisor(inverse_width, fma(-inverse_width, float(width), 1.0f) / float(width));
    mean = sam_norm_product(mean, divisor);
    float2 squares(0);
    if (width == 1024) {
        TF_UNROLL
        for (ushort k = 0; k < 32; ++k) {
            float2 delta = sam_two_sum(vals[k], -mean.x);
            sam_pair_add(delta, -mean.y);
            const float2 square = sam_norm_product(delta, delta);
            sam_pair_add(squares, square.x);
            sam_pair_add(squares, square.y);
        }
    } else {
        for (uint col = lane; col < width; col += 32) {
            float2 delta = sam_two_sum(x[base + col], -mean.x);
            sam_pair_add(delta, -mean.y);
            const float2 square = sam_norm_product(delta, delta);
            sam_pair_add(squares, square.x);
            sam_pair_add(squares, square.y);
        }
    }
    float2 variance = sam_norm_sum(squares);
    variance = sam_norm_product(variance, divisor);
    sam_pair_add(variance, eps);
    const float inverse = precise::rsqrt(variance.x), square = inverse * inverse;
    const float residual = fma(-variance.x, square, 1.0f) - variance.x * fma(inverse, inverse, -square) - variance.y * square;
    const float2 inv = sam_two_sum(inverse, 0.5f * inverse * residual);
    if (width == 1024) {
        TF_UNROLL
        for (ushort k = 0; k < 32; ++k) {
            const uint col = uint(k) * 32u + lane;
            const float value = sam_norm_apply(vals[k], mean, inv, weight[col], bias[col]);
            if constexpr (OUT_MODE == 1) sam_planes_store((device half*)out, total, target + col, value, invalid);
            else if constexpr (OUT_MODE == 3) sam_pair_store((device half*)out, total, target + col, value, invalid);
            else {
                out[target + col] = value;
                if constexpr (OUT_MODE == 2) invalid |= uint((as_type<uint>(value) & 0x7fffffffu) > 0x51800000u);
            }
        }
    } else {
        for (uint col = lane; col < width; col += 32) {
            const float value = sam_norm_apply(x[base + col], mean, inv, weight[col], bias[col]);
            if constexpr (OUT_MODE == 1) sam_planes_store((device half*)out, total, target + col, value, invalid);
            else if constexpr (OUT_MODE == 3) sam_pair_store((device half*)out, total, target + col, value, invalid);
            else {
                out[target + col] = value;
                if constexpr (OUT_MODE == 2) invalid |= uint((as_type<uint>(value) & 0x7fffffffu) > 0x51800000u);
            }
        }
    }
    if constexpr (OUT_MODE == 1) {
        if (invalid) atomic_fetch_or_explicit(sam_planes_flags((device half*)out, total), 1u, memory_order_relaxed);
    } else if constexpr (OUT_MODE == 3) {
        invalid = simd_or(invalid);
        if (invalid && lane == 0) atomic_fetch_or_explicit(sam_pair_flags((device half*)out, total), 1u, memory_order_relaxed);
    } else if constexpr (OUT_MODE == 2) {
        invalid = simd_or(invalid);
        if (invalid && lane == 0) atomic_fetch_or_explicit(out_flag, 1u, memory_order_relaxed);
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_norm_compensated(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant uint2& shape, constant float& eps,
    uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    sam_norm_compensated_impl<0>(x, weight, bias, out, nullptr, shape, eps, false, group, sg, lane);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_norm_compensated_wide(const device float* x, const device float* weight, const device float* bias,
    device float* out, device atomic_uint* out_flag, constant uint2& shape, constant float& eps, constant uint& tiled,
    uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    sam_norm_compensated_impl<2>(x, weight, bias, out, out_flag, shape, eps, tiled != 0, group, sg, lane);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_norm_wide_clear(device uint* flag, uint at [[thread_position_in_grid]]) {
    if (at == 0) flag[0] = 0;
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_norm_compensated_planes(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant uint2& shape, constant float& eps, constant uint& tiled,
    uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    sam_norm_compensated_impl<1>(x, weight, bias, out, nullptr, shape, eps, tiled != 0, group, sg, lane);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_norm_planes_clear(device half* planes, constant uint& count, uint at [[thread_position_in_grid]]) {
    if (at < 8) atomic_store_explicit(sam_planes_flags(planes, count) + at, 0u, memory_order_relaxed);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_norm_compensated_pair(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant uint2& shape, constant float& eps, constant uint& tiled,
    uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    sam_norm_compensated_impl<3>(x, weight, bias, out, nullptr, shape, eps, tiled != 0, group, sg, lane);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_norm_pair_clear(device half* pair, constant uint& count, uint at [[thread_position_in_grid]]) {
    if (at < 8) atomic_store_explicit(sam_pair_flags(pair, count) + at, 0u, memory_order_relaxed);
}
