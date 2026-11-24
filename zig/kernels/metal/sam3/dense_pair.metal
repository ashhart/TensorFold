// 3-pass FP16-plane NAX GEMM for FP32 weights: X_hi*W_hi + 2^-11*(X_hi*W_lo + X_lo*W_hi), with exact FP32 fallback.
[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_pair_prepare_weights(const device float* weight, const device float* bias, device half* w_pair,
    constant SamDenseShape& shape, uint at [[thread_position_in_grid]], uint width [[threads_per_grid]]) {
    const ulong nw = ulong(shape.columns) * shape.depth;
    uint invalid = 0;
    for (ulong i = at; i < nw + shape.columns; i += width) {
        if (i < nw) {
            const float v = weight[i];
            const bool bad = (as_type<uint>(v) & 0x7fffffffu) > 0x47000000u;
            invalid |= uint(bad);
            const half h = half(v);
            w_pair[i] = h;
            w_pair[i + nw] = bad ? half(0.0f) : half((v - float(h)) * 2048.0f);
        } else {
            invalid |= uint(!isfinite(bias[i - nw]));
        }
    }
    invalid = simd_or(invalid);
    if (invalid && simd_is_first())
        atomic_fetch_or_explicit((device atomic_uint*)((device float*)w_pair + nw), 1u, memory_order_relaxed);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_pair_clear(device half* pair, constant uint& count, uint at [[thread_position_in_grid]]) {
    if (at < 8) atomic_store_explicit(sam_pair_flags(pair, count) + at, 0u, memory_order_relaxed);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam_dense_pair_pack(const device float* x, device half* pair, constant uint& count,
    uint at [[thread_position_in_grid]], uint width [[threads_per_grid]]) {
    uint invalid = 0;
    const uint count4 = count / 4;
    const device float4* src4 = (const device float4*)x;
    device half4* hi4 = (device half4*)pair;
    device half4* lo4 = (device half4*)(pair + count);
    for (uint i = at; i < count4; i += width) {
        const float4 v = src4[i];
        const uint4 bits = as_type<uint4>(v) & 0x7fffffffu;
        if (any(bits > 0x47000000u)) {
            for (uint j = 0; j < 4; ++j) sam_pair_store(pair, count, ulong(i) * 4 + j, v[j], invalid);
        } else {
            const half4 h = half4(v);
            hi4[i] = h;
            lo4[i] = half4((v - float4(h)) * 2048.0f);
        }
    }
    const uint tail = count4 * 4 + at;
    if (tail < count) sam_pair_store(pair, count, tail, x[tail], invalid);
    invalid = simd_or(invalid);
    if (invalid && simd_is_first())
        atomic_fetch_or_explicit(sam_pair_flags(pair, count), 1u, memory_order_relaxed);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_pair_plan(device half* x_pair, const device half* w_pair, device uint* plan, device half* out_pair,
    constant SamDenseShape& shape, constant uint2& tile, constant uint& emit_pair,
    uint at [[thread_position_in_grid]]) {
    if (at != 0) return;
    const ulong nx = ulong(shape.rows) * shape.depth;
    const ulong nw = ulong(shape.columns) * shape.depth;
    const ulong outputs = ulong(shape.rows) * shape.columns;
    const uint x_flag = atomic_load_explicit(sam_pair_flags(x_pair, nx), memory_order_relaxed);
    const uint w_flag = ((const device uint*)((const device float*)w_pair + nw))[0];
    const bool fallback = (x_flag | w_flag) != 0;
    const uint grid[13] = {
        uint(fallback),
        fallback ? 0u : (shape.columns + tile.y - 1) / tile.y,
        (shape.rows + tile.x - 1) / tile.x,
        1u,
        fallback ? (shape.columns + 3) / 4 : 0u,
        shape.rows,
        1u,
        fallback ? uint((nx + 127) / 128) : 0u,
        1u,
        1u,
        fallback ? uint((outputs + 127) / 128) : 0u,
        1u,
        1u,
    };
    for (ushort i = 0; i < 13; ++i) plan[i] = grid[i];
    if (emit_pair) atomic_store_explicit(sam_pair_flags(out_pair, outputs), 0u, memory_order_relaxed);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_pair_restore(device half* pair, constant uint& count, uint at [[thread_position_in_grid]]) {
    if (at >= count) return;
    const half h = pair[at];
    if ((as_type<ushort>(h) & 0x7c00u) == 0x7c00u) return;
    ((device float*)pair)[count + at] = float(h) + ldexp(float(pair[at + count]), -11);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_pair_gelu_fallback(const device float* projected, device half* out_pair, constant uint& count,
    uint at [[thread_position_in_grid]]) {
    if (at >= count) return;
    uint invalid = 0;
    sam_pair_store(out_pair, count, at, sam_gelu_fp32(projected[at]), invalid);
    if (invalid) atomic_fetch_or_explicit(sam_pair_flags(out_pair, count), 1u, memory_order_relaxed);
}

// 2-piece degree-8 Chebyshev-in-z erfcx minimax GELU (<= 2.8 ULP on negative tails, zero lookup table).
inline float sam_gelu_minimax(float value) {
    const uint bits = as_type<uint>(value), mag = bits & 0x7fffffffu, sign = bits & 0x80000000u;
    if (mag >= 0x7f800000u) return mag > 0x7f800000u ? value + value : (sign ? NAN : value);
    if (mag < 0x24800000u) return 0.5f * value;
    const float ax = as_type<float>(mag);
    const float t_hi = ax * 0.7071067690849304f;
    const float t_lo = fma(ax, 0.7071067690849304f, -t_hi) + ax * 1.2101617104478969e-8f;
    if (t_hi >= 6.0f) return sign ? as_type<float>(0x80000000u) : value;
    const bool hi = t_hi >= 1.5f;
    const float z = 2.0f / (2.0f + t_hi);
    const float u = hi ? fma(6.222222328186035f, z, -2.555555582046509f)
                       : fma(4.666666507720947f, z, -3.6666667461395264f);
    float p = hi ? 3.642034600e-8f : -2.363153000e-7f;
    p = fma(p, u, hi ? 2.397083600e-7f : 3.782315600e-7f);
    p = fma(p, u, hi ? -2.148603500e-6f : 1.070726650e-5f);
    p = fma(p, u, hi ? -1.875571000e-5f : -6.358368000e-5f);
    p = fma(p, u, hi ? 1.001983400e-4f : -4.959912000e-4f);
    p = fma(p, u, hi ? 2.464470000e-3f : 5.108922000e-3f);
    p = fma(p, u, hi ? 2.070917000e-2f : 6.816980000e-2f);
    p = fma(p, u, hi ? 1.119584740e-1f : 3.341615800e-1f);
    p = fma(p, u, hi ? 1.863737400e-1f : 5.931084000e-1f);
    const float th = as_type<float>(as_type<uint>(t_hi) & 0xfffff000u);
    const float rem = fma(t_hi - th, t_hi + th, 2.0f * t_hi * t_lo);
    const float e = precise::exp(-(th * th)) * precise::exp(-rem);
    const float corr = 0.5f * value * (e * p);
    return sign ? corr : (value - corr);
}

// EPI: 0 FP32 out, 1 GELU -> pair out, 2 FP32 residual, 3 FP32 tiled residual.
template<ushort BM, ushort BN, ushort SG, ushort EPI>
inline void sam_dense_pair_impl(const device half* x_pair, const device half* w_pair, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* plan, const device float* residual,
    threadgroup float* tg, uint2 group) {
    if (plan[0]) return;
    using namespace mpp::tensor_ops;
    using Mat = tensor<device half, dextents<int, 2>, tensor_inline>;
    const ulong nx = ulong(shape.rows) * shape.depth;
    const ulong nw = ulong(shape.columns) * shape.depth;
    const ulong outputs = ulong(shape.rows) * shape.columns;
    const uint gx = (shape.columns + BN - 1) / BN, gy = (shape.rows + BM - 1) / BM;
    const uint linear = group.y * gx + group.x, band = 8u;
    const uint bstart = (linear / (gx * band)) * band;
    const uint brows = min(band, gy - bstart), inb = linear % (gx * band);
    const int row = int(bstart + (inb % brows)) * BM, col = int(inb / brows) * BN;
    const int depth = int(shape.depth), rows = int(shape.rows), cols = int(shape.columns);
    Mat ah((device half*)x_pair, dextents<int, 2>(depth, rows), array<int, 2>({1, depth}));
    Mat al((device half*)(x_pair + nx), dextents<int, 2>(depth, rows), array<int, 2>({1, depth}));
    Mat bh((device half*)w_pair, dextents<int, 2>(depth, cols), array<int, 2>({1, depth}));
    Mat bl((device half*)(w_pair + nw), dextents<int, 2>(depth, cols), array<int, 2>({1, depth}));
    auto xh = ah.slice(0, row), xl = al.slice(0, row);
    auto wh = bh.slice(0, col), wl = bl.slice(0, col);
    constexpr auto desc_init = matmul2d_descriptor(BM, BN, int(dynamic_extent), false, true, false);
    constexpr auto desc_acc = matmul2d_descriptor(BM, BN, int(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_init, execution_simdgroups<SG>> op_init;
    matmul2d<desc_acc, execution_simdgroups<SG>> op_acc;
    auto acc = op_init.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(xh)>,
        metal::remove_addrspace_t<decltype(wl)>, float>();
    op_init.run(xh, wl, acc);
    op_acc.run(xl, wh, acc);
    TF_UNROLL
    for (ushort e = 0; e < acc.get_capacity(); ++e) {
        const auto at = acc.get_multidimensional_index(e);
        tg[at[1] * BN + at[0]] = acc[e];
    }
    op_init.run(xh, wh, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint invalid = 0;
    TF_UNROLL
    for (ushort e = 0; e < acc.get_capacity(); ++e) {
        const auto at = acc.get_multidimensional_index(e);
        const uint r = row + at[1], c = col + at[0];
        if (r >= shape.rows || c >= shape.columns) continue;
        const float val = fma(tg[at[1] * BN + at[0]], 0x1p-11f, acc[e]) + bias[c];
        if constexpr (EPI == 1) {
            sam_pair_store((device half*)out, outputs, ulong(r) * shape.columns + c, sam_gelu_minimax(val), invalid);
        } else {
            const ulong target = ulong(EPI == 3 ? sam_planes_tile_row(r) : r) * shape.columns + c;
            out[target] = EPI >= 2 ? residual[target] + val : val;
        }
    }
    if constexpr (EPI == 1) {
        invalid = simd_or(invalid);
        if (invalid && simd_is_first())
            atomic_fetch_or_explicit(sam_pair_flags((device half*)out, outputs), 1u, memory_order_relaxed);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam_dense_pair_plain(const device half* xp, const device half* wp, const device float* bias, device float* out,
    constant SamDenseShape& shape, const device uint* plan, const device float* residual,
    uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float tg[64 * 128];
    sam_dense_pair_impl<64, 128, 8, 0>(xp, wp, bias, out, shape, plan, residual, tg, group);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_pair_gelu_pair(const device half* xp, const device half* wp, const device float* bias, device float* out,
    constant SamDenseShape& shape, const device uint* plan, const device float* residual,
    uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float tg[32 * 128];
    sam_dense_pair_impl<32, 128, 4, 1>(xp, wp, bias, out, shape, plan, residual, tg, group);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_pair_residual(const device half* xp, const device half* wp, const device float* bias, device float* out,
    constant SamDenseShape& shape, const device uint* plan, const device float* residual,
    uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float tg[32 * 128];
    sam_dense_pair_impl<32, 128, 4, 2>(xp, wp, bias, out, shape, plan, residual, tg, group);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_pair_tiled(const device half* xp, const device half* wp, const device float* bias, device float* out,
    constant SamDenseShape& shape, const device uint* plan, const device float* residual,
    uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float tg[32 * 128];
    sam_dense_pair_impl<32, 128, 4, 3>(xp, wp, bias, out, shape, plan, residual, tg, group);
}
