template<ushort DIM>
inline void sam_attention_device_split_impl(const device float* Q, const device float* K, const device float* V,
    device float* partial, device float* stats, device float* O, device atomic_uint* flags, constant int* P,
    uint3 tg, uint sg, uint lane) {
    if (atomic_load_explicit(flags, memory_order_relaxed)) {
        if (P[11] > 0 && lane == 0 && sg == 0 && tg.x == 0 && tg.y == 0 && tg.z == 0)
            atomic_store_explicit(flags + P[11], 1u, memory_order_relaxed);
        return;
    }
    using namespace mpp::tensor_ops;
    using Input = tensor<device float, dextents<int, 2>, tensor_inline>;
    constexpr ushort QUERY_ROWS = 16;
    constexpr ushort KEY_ROWS = 64;
    constexpr ushort ROW_PARTS = QUERY_ROWS / 8;
    const int splits = P[24], split = int(tg.z) % splits, batch = int(tg.z) / splits;
    const int row = int(tg.x) * 64 + int(sg) * QUERY_ROWS, rows = P[2];
    const int first = split * P[25] * 32, end = min(first + P[25] * 32, P[3]);
    if (row >= rows || first >= end) return;
    Q += long(batch) * P[12] + long(tg.y) * P[13];
    K += long(batch) * P[15] + long(tg.y) * P[16];
    const long v_head = long(tg.y) * P[19];
    V += long(batch) * P[18] + v_head;
    O += long(batch) * P[21] + long(tg.y) * P[22];
    const long base = ((long(batch) * P[1] + tg.y) * splits + split) * rows + row;
    partial += base * DIM;
    stats += base * 3;
    Input queries((device float*)Q, dextents<int, 2>(DIM, rows), array<int, 2>({1, P[14]}));
    Input keys((device float*)K, dextents<int, 2>(DIM, P[3]), array<int, 2>({1, P[17]}));
    Input values((device float*)V, dextents<int, 2>(DIM, P[3]), array<int, 2>({1, P[20]}));
    auto query = queries.slice(0, row);
    constexpr auto score_descriptor = matmul2d_descriptor(QUERY_ROWS, KEY_ROWS, DIM, false, true, false);
    constexpr auto value_descriptor = matmul2d_descriptor(QUERY_ROWS, DIM, KEY_ROWS, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<score_descriptor, execution_simdgroup> score_op;
    matmul2d<value_descriptor, execution_simdgroup> value_op;
    const float scale_log2 = as_type<float>(P[26]) * 1.4426950408889634f;
    float maximum[ROW_PARTS], minimum[ROW_PARTS], total[ROW_PARTS];
    TF_UNROLL
    for (ushort r = 0; r < ROW_PARTS; ++r) { maximum[r] = -INFINITY; minimum[r] = INFINITY; total[r] = 0; }
    auto key0 = keys.slice(0, first);
    auto scores = score_op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(query)>, metal::remove_addrspace_t<decltype(key0)>, float>();
    auto probs0 = value_op.template get_left_input_cooperative_tensor<float, float, float>(scores);
    auto val0 = values.slice(0, first);
    auto acc = value_op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(probs0)>, metal::remove_addrspace_t<decltype(val0)>, float>();
    TF_UNROLL
    for (ushort e = 0; e < acc.get_capacity(); ++e) acc[e] = 0;
    for (int first_key = first; first_key < end; first_key += KEY_ROWS) {
        auto key = keys.slice(0, first_key);
        score_op.run(query, key, scores);
        float next[ROW_PARTS];
        TF_UNROLL
        for (ushort r = 0; r < ROW_PARTS; ++r) next[r] = maximum[r];
        if (first_key + KEY_ROWS <= end) {
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
                const bool valid = first_key + at[0] < end;
                const float s = valid ? scores[e] * scale_log2 : -INFINITY;
                scores[e] = s;
                next[r] = max(next[r], s);
                if (valid) minimum[r] = min(minimum[r], s);
            }
        }
        TF_UNROLL
        for (ushort r = 0; r < ROW_PARTS; ++r) {
            next[r] = max(next[r], simd_shuffle_xor(next[r], ushort(1)));
            next[r] = max(next[r], simd_shuffle_xor(next[r], ushort(8)));
        }
        float added[ROW_PARTS];
        TF_UNROLL
        for (ushort r = 0; r < ROW_PARTS; ++r) added[r] = 0;
        TF_UNROLL
        for (ushort e = 0; e < scores.get_capacity(); ++e) {
            const int r = scores.get_multidimensional_index(e)[1] >> 3;
            const float p = fast::exp2(scores[e] - next[r]);
            scores[e] = p;
            added[r] += p;
        }
        float alpha[ROW_PARTS];
        TF_UNROLL
        for (ushort r = 0; r < ROW_PARTS; ++r) {
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
    TF_UNROLL
    for (ushort r = 0; r < ROW_PARTS; ++r) {
        minimum[r] = min(minimum[r], simd_shuffle_xor(minimum[r], ushort(1)));
        minimum[r] = min(minimum[r], simd_shuffle_xor(minimum[r], ushort(8)));
    }
    if (splits == 1) {
        bool tiny[ROW_PARTS];
        TF_UNROLL
        for (ushort r = 0; r < ROW_PARTS; ++r) tiny[r] = false;
        uint wide_bad = 0;
        TF_UNROLL
        for (ushort e = 0; e < acc.get_capacity(); ++e) {
            const auto at = acc.get_multidimensional_index(e);
            if (row + at[1] < rows) {
                const int r = at[1] >> 3;
                const float val = acc[e] / total[r];
                O[long(row + at[1]) * P[23] + at[0]] = val;
                tiny[r] |= abs(val) <= 0x1p-48f;
                if (P[11] > 0) wide_bad |= uint((as_type<uint>(val) & 0x7fffffffu) > 0x51800000u);
            }
        }
        bool fallback = false;
        TF_UNROLL
        for (ushort r = 0; r < ROW_PARTS; ++r)
            fallback |= (maximum[r] - minimum[r] > 28.8539008f) && simd_any(tiny[r]);
        const bool need_rescale = simd_any(fallback);
        if (need_rescale && lane == 0) {
            const uint index = (uint(batch) * uint(P[1]) + uint(tg.y)) * uint(P[5]) + uint(tg.x) + 1;
            atomic_store_explicit(flags + index, 1u, memory_order_relaxed);
        }
        if (P[11] > 0) {
            wide_bad = simd_or(wide_bad) | uint(need_rescale);
            if (wide_bad && lane == 0) atomic_fetch_or_explicit(flags + P[11], 1u, memory_order_relaxed);
        }
    } else {
        TF_UNROLL
        for (ushort e = 0; e < acc.get_capacity(); ++e) {
            const auto at = acc.get_multidimensional_index(e);
            if (row + at[1] < rows) {
                const int r = at[1] >> 3;
                partial[at[1] * DIM + at[0]] = acc[e];
                if (at[0] == 0) {
                    stats[at[1] * 3] = maximum[r];
                    stats[at[1] * 3 + 1] = total[r];
                    stats[at[1] * 3 + 2] = -minimum[r];
                }
            }
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_split32(const device float* q, const device float* k, const device float* v,
    device float* partial, device float* stats, device float* out, device atomic_uint* flags, constant int* p,
    uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    sam_attention_device_split_impl<32>(q, k, v, partial, stats, out, flags, p, group, sg, lane);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_split64(const device float* q, const device float* k, const device float* v,
    device float* partial, device float* stats, device float* out, device atomic_uint* flags, constant int* p,
    uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    sam_attention_device_split_impl<64>(q, k, v, partial, stats, out, flags, p, group, sg, lane);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_split32_merge(const device float* partial, const device float* stats,
    device float* out, device atomic_uint* flags, constant int* P,
    uint3 at [[thread_position_in_grid]]) {
    if (atomic_load_explicit(flags, memory_order_relaxed)) return;
    const int dim = P[13], channel = int(at.x) % dim;
    const int row = int(at.x) / dim, head = int(at.y), batch = int(at.z);
    if (row >= P[2]) return;
    const long plane = (long(batch) * P[1] + head) * P[24] * P[2];
    float maximum = -INFINITY, negative_minimum = -INFINITY;
    for (int split = 0; split < P[24]; ++split) {
        const long base = plane + long(split) * P[2] + row;
        maximum = max(maximum, stats[base * 3]);
        negative_minimum = max(negative_minimum, stats[base * 3 + 2]);
    }
    float2 denominator(0), sum(0);
    uint exceptional = 0;
    for (int split = 0; split < P[24]; ++split) {
        const long base = plane + long(split) * P[2] + row;
        const float weight = fast::exp2(stats[base * 3] - maximum);
        const float total = stats[base * 3 + 1], term = total * weight;
        sam_pair_add(denominator, term);
        sam_pair_add(denominator, fma(total, weight, -term));
        const float value = partial[base * dim + channel], product = value * weight;
        const uint flag = sam_exception(as_type<uint>(value), 0x3f800000u);
        exceptional |= flag;
        if (!flag) {
            sam_pair_add(sum, product);
            sam_pair_add(sum, fma(value, weight, -product));
        }
    }
    const float divisor = denominator.x + denominator.y;
    const float value = !isfinite(divisor) || divisor <= 0 || (exceptional & 4u) || (exceptional & 3u) == 3u ? NAN
        : exceptional ? as_type<float>((exceptional & 2u) ? 0xff800000u : 0x7f800000u) : (sum.x + sum.y) / divisor;
    out[long(batch) * P[21] + long(head) * P[22] + long(row) * P[23] + channel] = value;
    if (maximum + negative_minimum > 28.8539008f && abs(value) <= 0x1p-48f) {
        const uint index = (uint(batch) * uint(P[1]) + uint(head)) * uint(P[5]) + uint(row) / 64 + 1;
        atomic_store_explicit(flags + index, 1u, memory_order_relaxed);
    }
}
