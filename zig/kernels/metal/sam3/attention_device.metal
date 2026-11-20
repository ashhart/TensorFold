[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_device_clear(device uint* flags, constant uint& count, uint at [[thread_position_in_grid]]) {
    if (at < count) flags[at] = 0;
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_device_domain(const device float* q, const device float* k, const device float* v,
    device atomic_uint* flags, constant int* p, uint at [[thread_position_in_grid]], uint width [[threads_per_grid]]) {
    const ulong nq = ulong(p[0]) * p[1] * p[2] * p[13], nk = ulong(p[0]) * p[1] * p[3] * p[13];
    uint invalid = 0;
    for (ulong i = at; i < nq + 2 * nk; i += width) {
        const float value = i < nq ? q[i] : i < nq + nk ? k[i - nq] : v[i - nq - nk];
        const uint bits = as_type<uint>(value) & 0x7fffffffu;
        if (bits != 0 && (bits < 0x2d800000u || bits > 0x51800000u))
            invalid |= i < nq + nk || bits >= 0x7f800000u ? 1u : 2u;
    }
    if (invalid) atomic_fetch_or_explicit(flags, invalid, memory_order_relaxed);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_device_center(const device float* keys, const device float* values, device float* centered, device atomic_uint* flags,
    constant int* p, uint group [[threadgroup_position_in_grid]], ushort thread_index [[thread_index_in_threadgroup]]) {
    if (atomic_load_explicit(flags, memory_order_relaxed)) return;
    threadgroup float2 partial[256];
    threadgroup float anchors[128];
    const uint batch = group / p[1], head = group % p[1], channel = thread_index % 64;
    const long base = long(batch) * p[15] + long(head) * p[16];
    const long count = long(p[0]) * p[3] * p[1] * 64;
    float2 sum(0), value_sum(0);
    for (int j = thread_index / 64; j < p[3]; j += 2) {
        sam_pair_add(sum, keys[base + long(j) * p[17] + channel]);
        sam_pair_add(value_sum, values[base + long(j) * p[20] + channel]);
    }
    partial[thread_index] = sum;
    partial[128 + thread_index] = value_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < 64) {
        sam_pair_add(sum, partial[thread_index + 64].x);
        sam_pair_add(sum, partial[thread_index + 64].y);
        anchors[thread_index] = (sum.x + sum.y) / float(p[3]);
        sam_pair_add(value_sum, partial[192 + thread_index].x);
        sam_pair_add(value_sum, partial[192 + thread_index].y);
        anchors[64 + thread_index] = (value_sum.x + value_sum.y) / float(p[3]);
        centered[count * 2 + group * 64 + thread_index] = anchors[64 + thread_index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint invalid = 0;
    for (uint i = thread_index; i < uint(p[3]) * 64; i += 128) {
        const long at = base + long(i / 64) * p[17] + i % 64;
        const float difference = keys[at] - anchors[i % 64];
        const float value_difference = values[at] - anchors[64 + i % 64];
        centered[at] = difference;
        centered[count + at] = value_difference;
        const uint bits = as_type<uint>(difference) & 0x7fffffffu;
        invalid |= uint(bits != 0 && (bits < 0x2d800000u || bits > 0x51800000u));
        const uint value_bits = as_type<uint>(value_difference) & 0x7fffffffu;
        invalid |= uint(value_bits != 0 && (value_bits < 0x2d800000u || value_bits > 0x51800000u));
    }
    if (invalid) atomic_fetch_or_explicit(flags, 1u, memory_order_relaxed);
}

template<ushort DIM>
inline void sam_attention_device_impl(const device float* Q, const device float* K, const device float* V,
    device float* O, constant int* p, device atomic_uint* flags, uint3 group, ushort sg, ushort lane) {
    if (atomic_load_explicit(flags, memory_order_relaxed)) return;
    using namespace mpp::tensor_ops;
    using Input = tensor<device float, dextents<int, 2>, tensor_inline>;
    Q += long(group.z) * p[12] + long(group.y) * p[13];
    K += long(group.z) * p[15] + long(group.y) * p[16];
    V += long(group.z) * p[18] + long(group.y) * p[19];
    O += long(group.z) * p[21] + long(group.y) * p[22];
    const int first_query = int(group.x) * 64 + int(sg) * 16;
    Input queries((device float*)Q, dextents<int, 2>(DIM, p[2]), array<int, 2>({1, p[14]}));
    Input keys((device float*)K, dextents<int, 2>(DIM, p[3]), array<int, 2>({1, p[17]}));
    Input values((device float*)V, dextents<int, 2>(DIM, p[3]), array<int, 2>({1, p[20]}));
    auto query = queries.slice(0, first_query);
    constexpr auto score_descriptor = matmul2d_descriptor(16, 32, 16, false, true, false);
    constexpr auto value_descriptor = matmul2d_descriptor(16, DIM, 32, false, false, false);
    matmul2d<score_descriptor, execution_simdgroup> score_op;
    matmul2d<value_descriptor, execution_simdgroup> value_op;
    float maximum[2] = {-INFINITY, -INFINITY};
    float minimum[2] = {INFINITY, INFINITY};
    bool tiny_output[2] = {false, false};
    float2 total[2] = {float2(0), float2(0)}, output[DIM / 2];
    TF_UNROLL
    for (ushort e = 0; e < DIM / 2; ++e) output[e] = float2(0);
    for (int first_key = 0; first_key < p[3]; first_key += 32) {
        auto key = keys.slice(0, first_key);
        auto scores = score_op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(query)>, metal::remove_addrspace_t<decltype(key)>, float>();
        float2 score_sums[16];
        TF_UNROLL
        for (ushort e = 0; e < 16; ++e) score_sums[e] = float2(0);
        TF_UNROLL
        for (ushort d = 0; d < DIM; d += 16) {
            auto q_part = queries.slice(d, first_query);
            auto k_part = keys.slice(d, first_key);
            score_op.run(q_part, k_part, scores);
            TF_UNROLL
            for (ushort e = 0; e < scores.get_capacity(); ++e) sam_pair_add(score_sums[e], scores[e]);
        }
        TF_UNROLL
        for (ushort e = 0; e < scores.get_capacity(); ++e) scores[e] = score_sums[e].x + score_sums[e].y;
        float next[2] = {maximum[0], maximum[1]};
        TF_UNROLL
        for (ushort e = 0; e < scores.get_capacity(); ++e) {
            const auto at = scores.get_multidimensional_index(e);
            const int row = at[1] >> 3;
            scores[e] = first_key + at[0] < p[3] ? scores[e] * as_type<float>(p[24]) : -INFINITY;
            next[row] = max(next[row], scores[e]);
            if (first_key + at[0] < p[3]) minimum[row] = min(minimum[row], scores[e]);
        }
        TF_UNROLL
        for (ushort row = 0; row < 2; ++row) {
            next[row] = max(next[row], simd_shuffle_xor(next[row], ushort(1)));
            next[row] = max(next[row], simd_shuffle_xor(next[row], ushort(8)));
        }
        float2 added[2] = {float2(0), float2(0)};
        TF_UNROLL
        for (ushort e = 0; e < scores.get_capacity(); ++e) {
            const auto at = scores.get_multidimensional_index(e);
            const int row = at[1] >> 3;
            scores[e] = precise::exp(scores[e] - next[row]);
            sam_pair_add(added[row], scores[e]);
        }
        float alpha[2];
        TF_UNROLL
        for (ushort row = 0; row < 2; ++row) {
            for (ushort offset = 1; offset <= 8; offset *= 8) {
                const float high = simd_shuffle_xor(added[row].x, offset), low = simd_shuffle_xor(added[row].y, offset);
                sam_pair_add(added[row], high);
                sam_pair_add(added[row], low);
            }
            alpha[row] = precise::exp(maximum[row] - next[row]);
            sam_pair_scale(total[row], alpha[row]);
            sam_pair_add(total[row], added[row].x);
            sam_pair_add(total[row], added[row].y);
            maximum[row] = next[row];
        }
        auto probabilities = value_op.template get_left_input_cooperative_tensor<float, float, float>(scores);
        auto value = values.slice(0, first_key);
        auto product = value_op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(probabilities)>, metal::remove_addrspace_t<decltype(value)>, float>();
        value_op.run(probabilities, value, product);
        TF_UNROLL
        for (ushort e = 0; e < product.get_capacity(); ++e) {
            const auto at = product.get_multidimensional_index(e);
            sam_pair_scale(output[e], alpha[at[1] >> 3]);
            sam_pair_add(output[e], product[e]);
            const int row = first_query + at[1];
            if (first_key + 32 >= p[3] && row < p[2]) {
                const float2 denominator = total[at[1] >> 3];
                const float value = (output[e].x + output[e].y) / (denominator.x + denominator.y);
                O[long(row) * p[23] + at[0]] = value;
                tiny_output[at[1] >> 3] |= abs(value) <= 0x1p-48f;
            }
        }
    }
    bool fallback = false;
    TF_UNROLL
    for (ushort row = 0; row < 2; ++row) {
        minimum[row] = min(minimum[row], simd_shuffle_xor(minimum[row], ushort(1)));
        minimum[row] = min(minimum[row], simd_shuffle_xor(minimum[row], ushort(8)));
        // Outside this range, flushed probabilities contribute at most 2^-74 under the input/K bounds.
        fallback |= maximum[row] - minimum[row] > 20 && simd_any(tiny_output[row]);
    }
    if (simd_any(fallback) && lane == 0) {
        const uint index = (group.z * uint(p[1]) + group.y) * uint(p[5]) + group.x + 1;
        atomic_store_explicit(flags + index, 1u, memory_order_relaxed);
    }
}

inline void sam_attention_window(const device float* Q, const device float* K, const device float* V,
    device float* O, constant int* p, device atomic_uint* flags, uint3 group, uint thread_index,
    threadgroup float* probabilities, threadgroup float* score_errors, threadgroup float* maxima, threadgroup float* minima,
    threadgroup float* alpha, threadgroup float2* denominators, const device float* value_means) {
    if (atomic_load_explicit(flags, memory_order_relaxed) || int(group.x) * 32 >= p[2]) return;
    using namespace mpp::tensor_ops;
    using Input = tensor<device float, dextents<int, 2>, tensor_inline>;
    using Probabilities = tensor<threadgroup float, dextents<int, 2>, tensor_inline>;
    Q += long(group.z) * p[12] + long(group.y) * p[13];
    K += long(group.z) * p[15] + long(group.y) * p[16];
    V += long(group.z) * p[18] + long(group.y) * p[19];
    O += long(group.z) * p[21] + long(group.y) * p[22];
    value_means += (long(group.z) * p[1] + group.y) * 64;
    const int first_query = int(group.x) * 32;
    const uint row = thread_index / 4, column_part = thread_index % 4;
    if (thread_index < 32) {
        maxima[thread_index] = -INFINITY;
        minima[thread_index] = INFINITY;
        denominators[thread_index] = float2(0);
    }
    Input queries((device float*)Q, dextents<int, 2>(64, p[2]), array<int, 2>({1, p[14]}));
    Input keys((device float*)K, dextents<int, 2>(64, p[3]), array<int, 2>({1, p[17]}));
    Input values((device float*)V, dextents<int, 2>(64, p[3]), array<int, 2>({1, p[20]}));
    Probabilities weights(probabilities, dextents<int, 2>(64, 32), array<int, 2>({1, 64}));
    auto query = queries.slice(0, first_query);
    constexpr auto score_descriptor = matmul2d_descriptor(32, 64, 32, false, true, false);
    constexpr auto value_descriptor = matmul2d_descriptor(32, 64, 64, false, false, false);
    matmul2d<score_descriptor, execution_simdgroups<4>> score_op;
    matmul2d<value_descriptor, execution_simdgroups<4>> value_op;
    float output[32];
    TF_UNROLL
    for (ushort e = 0; e < 32; ++e) output[e] = 0;
    for (int first_key = 0; first_key < p[3]; first_key += 64) {
        auto key = keys.slice(0, first_key);
        auto scores = score_op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(query)>, metal::remove_addrspace_t<decltype(key)>, float>();
        float2 sums[32];
        TF_UNROLL
        for (ushort e = 0; e < 32; ++e) sums[e] = float2(0);
        TF_UNROLL
        for (ushort d = 0; d < 64; d += 32) {
            auto q_part = queries.slice(d, first_query);
            auto k_part = keys.slice(d, first_key);
            score_op.run(q_part, k_part, scores);
            TF_UNROLL
            for (ushort e = 0; e < scores.get_capacity(); ++e) sam_pair_add(sums[e], scores[e]);
        }
        TF_UNROLL
        for (ushort e = 0; e < scores.get_capacity(); ++e) {
            const auto at = scores.get_multidimensional_index(e);
            probabilities[at[1] * 64 + at[0]] = first_key + at[0] < p[3]
                ? sums[e].x * as_type<float>(p[24]) : -INFINITY;
            score_errors[at[1] * 64 + at[0]] = sums[e].y * as_type<float>(p[24]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float previous = maxima[row];
        float next = previous, minimum = minima[row];
        TF_UNROLL
        for (ushort j = 0; j < 16; ++j) {
            const uint column = column_part + j * 4;
            const float score = probabilities[row * 64 + column];
            next = max(next, score);
            if (first_key + column < uint(p[3])) minimum = min(minimum, score);
        }
        for (ushort offset = 1; offset <= 2; offset *= 2) {
            next = max(next, simd_shuffle_xor(next, offset));
            minimum = min(minimum, simd_shuffle_xor(minimum, offset));
        }
        float2 added(0);
        TF_UNROLL
        for (ushort j = 0; j < 16; ++j) {
            const uint column = column_part + j * 4, at = row * 64 + column;
            float probability = 0;
            if (first_key + column < uint(p[3])) {
                float2 difference = sam_two_sum(probabilities[at], -next);
                sam_pair_add(difference, score_errors[at]);
                const float exponential = precise::exp(difference.x);
                probability = fma(exponential, difference.y, exponential);
            }
            probabilities[at] = probability;
            sam_pair_add(added, probability);
        }
        for (ushort offset = 1; offset <= 2; offset *= 2) {
            const float high = simd_shuffle_xor(added.x, offset), low = simd_shuffle_xor(added.y, offset);
            sam_pair_add(added, high);
            sam_pair_add(added, low);
        }
        if (column_part == 0) {
            alpha[row] = precise::exp(previous - next);
            float2 total = denominators[row];
            sam_pair_scale(total, alpha[row]);
            sam_pair_add(total, added.x);
            sam_pair_add(total, added.y);
            denominators[row] = total;
            maxima[row] = next;
            minima[row] = minimum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto value = values.slice(0, first_key);
        auto product = value_op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(weights)>, metal::remove_addrspace_t<decltype(value)>, float>();
        value_op.run(weights, value, product);
        bool fallback = false;
        TF_UNROLL
        for (ushort e = 0; e < product.get_capacity(); ++e) {
            const auto at = product.get_multidimensional_index(e);
            output[e] = fma(output[e], alpha[at[1]], product[e]);
            if (first_key + 64 >= p[3] && first_query + at[1] < p[2]) {
                const float2 denominator = denominators[at[1]];
                const float quotient = output[e] / denominator.x;
                const float remainder = fma(-quotient, denominator.x, output[e]) - quotient * denominator.y;
                float2 shifted = sam_two_sum(quotient, remainder / denominator.x);
                sam_pair_add(shifted, value_means[at[0]]);
                const float result = shifted.x + shifted.y;
                O[long(first_query + at[1]) * p[23] + at[0]] = result;
                fallback |= maxima[at[1]] - minima[at[1]] > 20 && abs(result) <= 0x1p-48f;
            }
        }
        if (simd_any(fallback) && thread_index % 32 == 0) {
            const uint index = (group.z * uint(p[1]) + group.y) * uint(p[5]) + group.x / 2 + 1;
            atomic_store_explicit(flags + index, 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_device64(const device float* q, const device float* k, const device float* v,
    device float* out, constant int* p, device atomic_uint* flags, uint3 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    const device float* centered [[buffer(6)]]) {
    threadgroup float probabilities[32 * 64], score_errors[32 * 64], maxima[32], minima[32], alpha[32];
    threadgroup float2 denominators[32];
    const long count = long(p[0]) * p[3] * p[1] * 64;
    if (p[1] == 16 && p[2] == 576 && p[3] == 576) sam_attention_window(q, centered, centered + count, out, p, flags, group, sg * 32 + lane,
        probabilities, score_errors, maxima, minima, alpha, denominators, centered + count * 2);
    else sam_attention_device_impl<64>(q, k, v, out, p, flags, group, sg, lane);
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
