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

// KC 0: one FP32 accumulation over the whole depth, then the bias. Else KC-deep chunks under a running compensation.
template<ushort KC, bool GELU>
inline void sam_dense_wide_impl(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant SamDenseShape& shape, const device uint* domain, uint2 group) {
    if (domain[0]) return;
    using namespace mpp::tensor_ops;
    using Input = tensor<device float, dextents<int, 2>, tensor_inline>;
    const int row = int(group.y) * 32, column = int(group.x) * 64, depth = int(shape.depth);
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
        out[ulong(r) * shape.columns + c] = GELU ? sam_gelu_fp32(value) : value;
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_wide_plain(const device float* x, const device float* weight, const device float* bias, device float* out,
    constant SamDenseShape& shape, const device uint* domain, uint2 group [[threadgroup_position_in_grid]]) {
    sam_dense_wide_impl<0, false>(x, weight, bias, out, shape, domain, group);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_wide_compensated(const device float* x, const device float* weight, const device float* bias, device float* out,
    constant SamDenseShape& shape, const device uint* domain, uint2 group [[threadgroup_position_in_grid]]) {
    sam_dense_wide_impl<64, false>(x, weight, bias, out, shape, domain, group);
}

// 32-deep chunks keep the expansion within PyTorch's per-column FP64 error; 64 does not.
[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_dense_wide_gelu(const device float* x, const device float* weight, const device float* bias, device float* out,
    constant SamDenseShape& shape, const device uint* domain, uint2 group [[threadgroup_position_in_grid]]) {
    sam_dense_wide_impl<32, true>(x, weight, bias, out, shape, domain, group);
}
