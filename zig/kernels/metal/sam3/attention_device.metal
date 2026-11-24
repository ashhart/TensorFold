[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_device_clear(device uint* flags, constant uint& count, uint at [[thread_position_in_grid]]) {
    if (at < count) flags[at] = 0;
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_device_domain(const device float* q, const device float* k, const device float* v,
    device atomic_uint* flags, constant int* p, uint at [[thread_position_in_grid]], uint width [[threads_per_grid]]) {
    const ulong nq = (ulong(p[0]) * p[1] * p[2] * p[13]) >> 2u;
    const ulong nk = (ulong(p[0]) * p[1] * p[3] * p[13]) >> 2u;
    const device float4* q4 = (const device float4*)q;
    const device float4* k4 = (const device float4*)k;
    const device float4* v4 = (const device float4*)v;
    uint invalid = 0;
    for (ulong i = at; i < nq + 2 * nk; i += width) {
        const float4 val = i < nq ? q4[i] : i < nq + nk ? k4[i - nq] : v4[i - nq - nk];
        const uint4 bits = as_type<uint4>(val) & 0x7fffffffu;
        const bool4 bad = (bits != 0u) & ((bits < 0x2d800000u) | (bits > 0x51800000u));
        if (any(bad)) {
            if (i < nq + nk || any(bits >= 0x7f800000u)) invalid |= 1u;
            else invalid |= 2u;
        }
    }
    invalid = simd_or(invalid);
    if (invalid && simd_is_first()) atomic_fetch_or_explicit(flags, invalid, memory_order_relaxed);
}

template<ushort DIM>
inline void sam_attention_device_impl(const device float* Q, const device float* K, const device float* V,
    device float* O, constant int* p, device atomic_uint* flags, uint3 group, ushort sg, ushort lane) {
    if (atomic_load_explicit(flags, memory_order_relaxed)) {
        if (p[11] > 0 && lane == 0 && sg == 0 && group.x == 0 && group.y == 0 && group.z == 0)
            atomic_store_explicit(flags + p[11], 1u, memory_order_relaxed);
        return;
    }
    using namespace mpp::tensor_ops;
    using Input = tensor<device float, dextents<int, 2>, tensor_inline>;
    constexpr ushort QUERY_ROWS = 16;
    constexpr ushort KEY_ROWS = 64;
    const int first_query = int(group.x) * 64 + int(sg) * QUERY_ROWS, rows = p[2], key_count = p[3];
    if (first_query >= rows) return;
    Q += long(group.z) * p[12] + long(group.y) * p[13];
    K += long(group.z) * p[15] + long(group.y) * p[16];
    V += long(group.z) * p[18] + long(group.y) * p[19];
    O += long(group.z) * p[21] + long(group.y) * p[22];
    Input queries((device float*)Q, dextents<int, 2>(DIM, rows), array<int, 2>({1, p[14]}));
    Input keys((device float*)K, dextents<int, 2>(DIM, key_count), array<int, 2>({1, p[17]}));
    Input values((device float*)V, dextents<int, 2>(DIM, key_count), array<int, 2>({1, p[20]}));
    auto query = queries.slice(0, first_query);
    constexpr auto score_descriptor = matmul2d_descriptor(QUERY_ROWS, KEY_ROWS, DIM, false, true, false);
    constexpr auto value_descriptor = matmul2d_descriptor(QUERY_ROWS, DIM, KEY_ROWS, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<score_descriptor, execution_simdgroup> score_op;
    matmul2d<value_descriptor, execution_simdgroup> value_op;
    const float scale_log2 = as_type<float>(p[24]) * 1.4426950408889634f;
    float maximum[2] = {-INFINITY, -INFINITY};
    float minimum[2] = {INFINITY, INFINITY};
    float total[2] = {0, 0};
    auto key0 = keys.slice(0, 0);
    auto scores = score_op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(query)>, metal::remove_addrspace_t<decltype(key0)>, float>();
    auto probs0 = value_op.template get_left_input_cooperative_tensor<float, float, float>(scores);
    auto val0 = values.slice(0, 0);
    auto acc = value_op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(probs0)>, metal::remove_addrspace_t<decltype(val0)>, float>();
    TF_UNROLL
    for (ushort e = 0; e < acc.get_capacity(); ++e) acc[e] = 0;
    for (int first_key = 0; first_key < key_count; first_key += KEY_ROWS) {
        auto key = keys.slice(0, first_key);
        score_op.run(query, key, scores);
        float next[2] = {maximum[0], maximum[1]};
        if (first_key + KEY_ROWS <= key_count) {
            TF_UNROLL
            for (ushort e = 0; e < scores.get_capacity(); ++e) {
                const int r = scores.get_multidimensional_index(e)[1] >> 3;
                const float s = scores[e] * scale_log2;
                scores[e] = s;
                next[r] = max(next[r], s);
                minimum[r] = min(minimum[r], s);
            }
        } else {
            TF_UNROLL
            for (ushort e = 0; e < scores.get_capacity(); ++e) {
                const auto at = scores.get_multidimensional_index(e);
                const int r = at[1] >> 3;
                const bool valid = first_key + at[0] < key_count;
                const float s = valid ? scores[e] * scale_log2 : -INFINITY;
                scores[e] = s;
                next[r] = max(next[r], s);
                if (valid) minimum[r] = min(minimum[r], s);
            }
        }
        TF_UNROLL
        for (ushort r = 0; r < 2; ++r) {
            next[r] = max(next[r], simd_shuffle_xor(next[r], ushort(1)));
            next[r] = max(next[r], simd_shuffle_xor(next[r], ushort(8)));
        }
        float added[2] = {0, 0};
        TF_UNROLL
        for (ushort e = 0; e < scores.get_capacity(); ++e) {
            const int r = scores.get_multidimensional_index(e)[1] >> 3;
            const float prob = fast::exp2(scores[e] - next[r]);
            scores[e] = prob;
            added[r] += prob;
        }
        float alpha[2];
        TF_UNROLL
        for (ushort r = 0; r < 2; ++r) {
            added[r] += simd_shuffle_xor(added[r], ushort(1));
            added[r] += simd_shuffle_xor(added[r], ushort(8));
            alpha[r] = fast::exp2(maximum[r] - next[r]);
            total[r] = fma(total[r], alpha[r], added[r]);
            maximum[r] = next[r];
        }
        auto probabilities = value_op.template get_left_input_cooperative_tensor<float, float, float>(scores);
        auto value = values.slice(0, first_key);
        TF_UNROLL
        for (ushort e = 0; e < acc.get_capacity(); ++e)
            acc[e] *= alpha[acc.get_multidimensional_index(e)[1] >> 3];
        value_op.run(probabilities, value, acc);
    }
    bool tiny_output[2] = {false, false};
    uint wide_bad = 0;
    TF_UNROLL
    for (ushort e = 0; e < acc.get_capacity(); ++e) {
        const auto at = acc.get_multidimensional_index(e);
        const int row = first_query + at[1];
        if (row < rows) {
            const int r = at[1] >> 3;
            const float value = acc[e] / total[r];
            O[long(row) * p[23] + at[0]] = value;
            tiny_output[r] |= abs(value) <= 0x1p-48f;
            if (p[11] > 0) wide_bad |= uint((as_type<uint>(value) & 0x7fffffffu) > 0x51800000u);
        }
    }
    bool fallback = false;
    TF_UNROLL
    for (ushort r = 0; r < 2; ++r) {
        minimum[r] = min(minimum[r], simd_shuffle_xor(minimum[r], ushort(1)));
        minimum[r] = min(minimum[r], simd_shuffle_xor(minimum[r], ushort(8)));
        fallback |= (maximum[r] - minimum[r] > 28.8539008f) && simd_any(tiny_output[r]);
    }
    const bool need_rescale = simd_any(fallback);
    if (need_rescale && lane == 0) {
        const uint index = (group.z * uint(p[1]) + group.y) * uint(p[5]) + group.x + 1;
        atomic_store_explicit(flags + index, 1u, memory_order_relaxed);
    }
    if (p[11] > 0) {
        wide_bad = simd_or(wide_bad) | uint(need_rescale);
        if (wide_bad && lane == 0) atomic_fetch_or_explicit(flags + p[11], 1u, memory_order_relaxed);
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_device64(const device float* q, const device float* k, const device float* v,
    device float* out, constant int* p, device atomic_uint* flags, uint3 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    sam_attention_device_impl<64>(q, k, v, out, p, flags, group, sg, lane);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_device32(const device float* q, const device float* k, const device float* v,
    device float* out, constant int* p, device atomic_uint* flags, uint3 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    sam_attention_device_impl<32>(q, k, v, out, p, flags, group, sg, lane);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_device_fallback(const device float* q, const device float* k, const device float* v,
    device float* out, const device int* p, const device uint* flags, uint3 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    if ((flags[0] & 1u) == 0) return;
    tfp::attention_nax<float, false, false, false, 64>(q, k, v, out, p, (const device float*)(p + 24), group, sg, lane);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_device32_fallback(const device float* q, const device float* k, const device float* v,
    device float* out, const device int* p, const device uint* flags, uint3 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    if ((flags[0] & 1u) == 0) return;
    tfp::attention_nax<float, false, false, false, 32>(q, k, v, out, p, (const device float*)(p + 24), group, sg, lane);
}

template<ushort DIM>
inline float sam_attention_device_score(const device float* q, const device float* k, uint lane, float scale) {
    float2 sum(0);
    TF_UNROLL
    for (ushort d = 0; d < DIM / 32; ++d) {
        const float a = q[lane + d * 32], b = k[lane + d * 32], product = a * b;
        sam_pair_add(sum, product);
        sam_pair_add(sum, fma(a, b, -product));
    }
    for (ushort offset = 16; offset != 0; offset >>= 1) {
        const float high = simd_shuffle_xor(sum.x, offset), low = simd_shuffle_xor(sum.y, offset);
        sam_pair_add(sum, high);
        sam_pair_add(sum, low);
    }
    return (sum.x + sum.y) * scale;
}

template<ushort DIM>
inline void sam_attention_device_rescale_impl(const device float* q, const device float* k, const device float* v,
    device float* out, const device int* p, const device uint* flags, uint3 group, uint sg, uint lane) {
    const uint index = (group.z * uint(p[1]) + group.y) * uint(p[5]) + group.x + 1;
    if (!flags[0] && !flags[index]) return;
    q += long(group.z) * p[12] + long(group.y) * p[13];
    k += long(group.z) * p[15] + long(group.y) * p[16];
    v += long(group.z) * p[18] + long(group.y) * p[19];
    out += long(group.z) * p[21] + long(group.y) * p[22];
    if (flags[0] & 1u) {
        bool key_invalid = false;
        for (uint at = lane; at < uint(p[3]) * DIM; at += 32)
            key_invalid |= (as_type<uint>(k[long(at / DIM) * p[17] + at % DIM]) & 0x7fffffffu) >= 0x7f800000u;
        key_invalid = simd_any(key_invalid);
        for (int r = int(group.x) * 64 + int(sg) * 16; r < min(p[2], int(group.x) * 64 + int(sg + 1) * 16); ++r) {
            bool invalid = key_invalid;
            TF_UNROLL
            for (ushort d = 0; d < DIM / 32; ++d)
                invalid |= (as_type<uint>(q[long(r) * p[14] + d * 32 + lane]) & 0x7fffffffu) >= 0x7f800000u;
            if (simd_any(invalid)) {
                TF_UNROLL
                for (ushort d = 0; d < DIM / 32; ++d) out[long(r) * p[23] + lane + d * 32] = NAN;
            }
        }
        return;
    }
    for (int r = int(group.x) * 64 + int(sg) * 16; r < min(p[2], int(group.x) * 64 + int(sg + 1) * 16); ++r) {
        const device float* query = q + long(r) * p[14];
        float maximum = -INFINITY;
        for (int j = 0; j < p[3]; ++j) maximum = max(maximum, sam_attention_device_score<DIM>(query, k + long(j) * p[17], lane, as_type<float>(p[24])));
        int exponents[DIM / 32];
        float2 denominator(0), sums[DIM / 32];
        TF_UNROLL
        for (ushort d = 0; d < DIM / 32; ++d) { exponents[d] = -2048; sums[d] = float2(0); }
        for (int j = 0; j < p[3]; ++j) {
            const float logp = (sam_attention_device_score<DIM>(query, k + long(j) * p[17], lane, as_type<float>(p[24])) - maximum) * 1.4426950408889634f;
            if (logp < -512) continue;
            const int whole = int(floor(logp));
            const float fraction = precise::exp2(logp - float(whole));
            sam_pair_add(denominator, ldexp(fraction, whole));
            TF_UNROLL
            for (ushort d = 0; d < DIM / 32; ++d) {
                const uint bits = as_type<uint>(v[long(j) * p[20] + d * 32 + lane]);
                if ((bits & 0x7fffffffu) != 0) exponents[d] = max(exponents[d], whole + sam_binary_exponent(bits));
            }
        }
        for (int j = 0; j < p[3]; ++j) {
            const float logp = (sam_attention_device_score<DIM>(query, k + long(j) * p[17], lane, as_type<float>(p[24])) - maximum) * 1.4426950408889634f;
            if (logp < -512) continue;
            const int whole = int(floor(logp));
            const float fraction = precise::exp2(logp - float(whole));
            TF_UNROLL
            for (ushort d = 0; d < DIM / 32; ++d) {
                const uint bits = as_type<uint>(v[long(j) * p[20] + d * 32 + lane]);
                if ((bits & 0x7fffffffu) == 0) continue;
                const float mantissa = sam_binary_mantissa(bits), product = fraction * mantissa;
                const int shift = whole + sam_binary_exponent(bits) - exponents[d];
                sam_pair_add(sums[d], ldexp(product, shift));
                sam_pair_add(sums[d], ldexp(fma(fraction, mantissa, -product), shift));
            }
        }
        const float divisor = denominator.x + denominator.y;
        TF_UNROLL
        for (ushort d = 0; d < DIM / 32; ++d) {
            const float high = sums[d].x / divisor;
            const float low = (fma(-high, divisor, sums[d].x) + sums[d].y) / divisor;
            const uint bits = exponents[d] == -2048 ? 0 : sam_scaled_bits(sam_two_sum(high, low), exponents[d]);
            out[long(r) * p[23] + d * 32 + lane] = as_type<float>(bits);
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_device_rescale(const device float* q, const device float* k, const device float* v,
    device float* out, const device int* p, const device uint* flags, uint3 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    sam_attention_device_rescale_impl<64>(q, k, v, out, p, flags, group, sg, lane);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_device32_rescale(const device float* q, const device float* k, const device float* v,
    device float* out, const device int* p, const device uint* flags, uint3 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    sam_attention_device_rescale_impl<32>(q, k, v, out, p, flags, group, sg, lane);
}
