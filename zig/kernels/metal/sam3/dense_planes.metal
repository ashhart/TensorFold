// Flags in the plan: 1 some inputs only in the restore area, 4 non-finite weight or bias.
[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_planes_clear(device half* planes, device uint* plan, device half* emitted, constant SamDenseShape& shape,
    constant uint2& mode, uint at [[thread_position_in_grid]]) {
    if (at < 16) plan[at] = 0;
    if (mode.x && at < 8) atomic_store_explicit(sam_planes_flags(planes, ulong(shape.rows) * shape.depth) + at, 0u, memory_order_relaxed);
    if (mode.y && at < 8) atomic_store_explicit(sam_planes_flags(emitted, ulong(shape.rows) * shape.columns) + at, 0u, memory_order_relaxed);
}

inline void sam_dense_planes_operands(const device half* weight, const device float* bias, device uint* plan,
    constant SamDenseShape& shape, uint at, uint width) {
    const ulong nw = ulong(shape.columns) * shape.depth;
    uint operands = 0;
    for (ulong i = at; i < nw + shape.columns; i += width)
        operands |= uint(i < nw ? (as_type<ushort>(weight[i]) & 0x7c00u) == 0x7c00u : !isfinite(bias[i - nw]));
    if (operands) atomic_fetch_or_explicit((device atomic_uint*)plan, 4u, memory_order_relaxed);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_planes_pack(const device float* x, const device half* weight, const device float* bias,
    device half* planes, device uint* plan, constant SamDenseShape& shape, uint at [[thread_position_in_grid]],
    uint width [[threads_per_grid]]) {
    const ulong n = ulong(shape.rows) * shape.depth;
    uint invalid = 0;
    for (ulong i = at; i < n; i += width) sam_planes_store(planes, n, i, x[i], invalid);
    if (invalid) atomic_fetch_or_explicit(sam_planes_flags(planes, n), 1u, memory_order_relaxed);
    sam_dense_planes_operands(weight, bias, plan, shape, at, width);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_planes_domain(const device half* weight, const device float* bias, device uint* plan,
    constant SamDenseShape& shape, uint at [[thread_position_in_grid]], uint width [[threads_per_grid]]) {
    sam_dense_planes_operands(weight, bias, plan, shape, at, width);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_planes_plan(device half* planes, device uint* plan, constant SamDenseShape& shape, constant uint2& tile,
    uint at [[thread_position_in_grid]]) {
    if (at != 0) return;
    const ulong n = ulong(shape.rows) * shape.depth, outputs = ulong(shape.rows) * shape.columns;
    const uint flags = plan[0] | atomic_load_explicit(sam_planes_flags(planes, n), memory_order_relaxed);
    const bool fallback = flags != 0;
    const uint grid[13] = {flags, fallback ? 0 : (shape.columns + tile.y - 1) / tile.y, (shape.rows + tile.x - 1) / tile.x, 1,
        fallback ? (shape.columns + 3) / 4 : 0, shape.rows, 1, fallback ? uint((n + 127) / 128) : 0, 1, 1,
        fallback ? uint((outputs + 127) / 128) : 0, 1, 1};
    for (ushort i = 0; i < 13; ++i) plan[i] = grid[i];
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_planes_restore(device half* planes, constant SamDenseShape& shape, uint at [[thread_position_in_grid]]) {
    const ulong n = ulong(shape.rows) * shape.depth;
    if (at >= n) return;
    const float high = float(planes[at]);
    if (!isfinite(high)) return;
    ((device float*)planes)[sam_planes_floats(n) + at] = (high + ldexp(float(planes[at + n]), -11)) + ldexp(float(planes[at + 2 * n]), -22);
}

// Adds the residual to a fallback projection, at the grid row of a tile-ordered row when tiled is set.
[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_planes_finish(const device float* projected, device float* output, const device float* residual,
    constant SamDenseShape& shape, constant uint& tiled, uint at [[thread_position_in_grid]]) {
    if (at >= ulong(shape.rows) * shape.columns) return;
    const uint r = uint(at / shape.columns), c = uint(at % shape.columns);
    const ulong target = ulong(tiled ? sam_planes_tile_row(r) : r) * shape.columns + c;
    output[target] = residual[target] + projected[at];
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_planes_gelu(const device float* x, device half* planes, constant uint& count,
    uint at [[thread_position_in_grid]], uint width [[threads_per_grid]]) {
    uint invalid = 0;
    for (ulong i = at; i < count; i += width) sam_planes_store(planes, count, i, sam_gelu_fp32(x[i]), invalid);
    if (invalid) atomic_fetch_or_explicit(sam_planes_flags(planes, count), 1u, memory_order_relaxed);
}

// KC 0 accumulates each plane over the whole depth; otherwise the high plane is compensated per KC-deep chunk.
// EPI 0: FP32 output; 2: residual added; 3: residual added at the grid row of a tile-ordered row.
template<ushort M, ushort N, ushort SG, ushort KC, ushort EPI>
inline void sam_dense_planes_impl(const device half* planes, const device half* weight, const device float* bias,
    device float* output, constant SamDenseShape& shape, const device uint* plan, const device float* residual, uint2 group) {
    if (plan[0]) return;
    using namespace mpp::tensor_ops;
    using Input = tensor<device half, dextents<int, 2>, tensor_inline>;
    const ulong n = ulong(shape.rows) * shape.depth;
    const int row = int(group.y) * M, column = int(group.x) * N, depth = int(shape.depth);
    const int rows = min(int(M), int(shape.rows) - row), cols = min(int(N), int(shape.columns) - column);
    Input w((device half*)weight + long(column) * depth, dextents<int, 2>(depth, cols), array<int, 2>({1, depth}));
    Input x0((device half*)planes + long(row) * depth, dextents<int, 2>(depth, rows), array<int, 2>({1, depth}));
    Input x1((device half*)planes + n + long(row) * depth, dextents<int, 2>(depth, rows), array<int, 2>({1, depth}));
    Input x2((device half*)planes + 2 * n + long(row) * depth, dextents<int, 2>(depth, rows), array<int, 2>({1, depth}));
    constexpr ushort count = M * N / (SG * 32);
    constexpr auto descriptor = matmul2d_descriptor(M, N, int(dynamic_extent), false, true, false);
    matmul2d<descriptor, execution_simdgroups<SG>> op;
    auto product = op.template get_destination_cooperative_tensor<decltype(x0), decltype(w), float>();
    float2 sums[count];
    if constexpr (KC != 0) {
        constexpr auto chunk = matmul2d_descriptor(M, N, KC, false, true, false);
        matmul2d<chunk, execution_simdgroups<SG>> chunk_op;
        TF_UNROLL
        for (ushort e = 0; e < count; ++e) sums[e] = float2(0);
        for (int first = 0; first < depth; first += KC) {
            auto a0 = x0.slice(first, 0), wk = w.slice(first, 0);
            chunk_op.run(a0, wk, product);
            TF_UNROLL
            for (ushort e = 0; e < count; ++e) {
                const float y = product[e] - sums[e].y, t = sums[e].x + y;
                sums[e].y = (t - sums[e].x) - y;
                sums[e].x = t;
            }
        }
        TF_UNROLL
        for (ushort e = 0; e < count; ++e) sums[e] = sam_two_sum(sums[e].x, -sums[e].y);
    } else {
        op.run(x0, w, product);
        TF_UNROLL
        for (ushort e = 0; e < count; ++e) sums[e] = float2(product[e], 0);
    }
    op.run(x1, w, product);
    TF_UNROLL
    for (ushort e = 0; e < count; ++e) sam_pair_add(sums[e], ldexp(product[e], -11));
    op.run(x2, w, product);
    TF_UNROLL
    for (ushort e = 0; e < count; ++e) {
        sam_pair_add(sums[e], ldexp(product[e], -22));
        const auto at = product.get_multidimensional_index(e);
        const uint r = row + at[1], c = column + at[0];
        if (r < shape.rows && c < shape.columns) {
            sam_pair_add(sums[e], bias[c]);
            const float value = sums[e].x + sums[e].y;
            const ulong target = ulong(EPI == 3 ? sam_planes_tile_row(r) : r) * shape.columns + c;
            if constexpr (EPI >= 2) output[target] = residual[target] + value;
            else output[target] = value;
        }
    }
}

#define SAM_DENSE_PLANES(NAME, KC, EPI) \
[[kernel, max_total_threads_per_threadgroup(128)]] \
void NAME(const device half* planes, const device half* weight, const device float* bias, device float* output, \
    constant SamDenseShape& shape, const device uint* plan, const device float* residual, \
    uint2 group [[threadgroup_position_in_grid]]) { \
    sam_dense_planes_impl<32, 128, 4, KC, EPI>(planes, weight, bias, output, shape, plan, residual, group); \
}

// 64-deep chunks keep the trunk FC1 within PyTorch's per-column FP64 error; the FC2 residual kernel needs only 128.
SAM_DENSE_PLANES(sam_dense_planes, 0, 0)
SAM_DENSE_PLANES(sam_dense_planes_residual, 0, 2)
SAM_DENSE_PLANES(sam_dense_planes_tiled, 0, 3)
SAM_DENSE_PLANES(sam_dense_planes_compensated, 64, 0)
SAM_DENSE_PLANES(sam_dense_planes_compensated_residual, 128, 2)
