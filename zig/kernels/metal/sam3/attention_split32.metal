template<ushort DIM>
inline void sam_attention_device_split_impl(const device float* Q, const device float* K, const device float* V,
    device float* partial, device float* stats, const device uint* flags, constant int* P,
    uint3 tg, uint sg, uint lane) {
    if (flags[0]) return;
    using namespace mpp::tensor_ops;
    using Input = tensor<device float, dextents<int, 2>, tensor_inline>;
    constexpr ushort QUERY_ROWS = 16;
    constexpr ushort KEY_ROWS = DIM == 32 ? 64 : 32;
    constexpr ushort ROW_PARTS = QUERY_ROWS / 8;
    const int split = int(tg.z) % P[24], batch = int(tg.z) / P[24];
    const int row = int(tg.x) * 64 + int(sg) * QUERY_ROWS, rows = P[2];
    const int first = split * P[25] * 32, end = min(first + P[25] * 32, P[3]);
    Q += long(batch) * P[12] + long(tg.y) * P[13];
    K += long(batch) * P[15] + long(tg.y) * P[16];
    const long v_head = long(tg.y) * P[19];
    V += long(batch) * P[18] + v_head;
    const long base = ((long(batch) * P[1] + tg.y) * P[24] + split) * rows + row;
    partial += base * DIM;
    stats += base * 3;
    Input queries((device float*)Q, dextents<int, 2>(DIM, rows), array<int, 2>({1, P[14]}));
    Input keys((device float*)K, dextents<int, 2>(DIM, P[3]), array<int, 2>({1, P[17]}));
    Input values((device float*)V, dextents<int, 2>(DIM, P[3]), array<int, 2>({1, P[20]}));
    auto query = queries.slice(0, row);
    constexpr ushort SCORE_DEPTH = DIM == 64 ? 32 : 16;
    constexpr auto score_descriptor = matmul2d_descriptor(QUERY_ROWS, KEY_ROWS, SCORE_DEPTH, false, true, false);
    constexpr auto value_descriptor = matmul2d_descriptor(QUERY_ROWS, DIM, KEY_ROWS, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<score_descriptor, execution_simdgroup> score_op;
    matmul2d<value_descriptor, execution_simdgroup> value_op;
    float maximum[ROW_PARTS], minimum[ROW_PARTS];
    float2 total[ROW_PARTS];
    TF_UNROLL
    for (ushort r = 0; r < ROW_PARTS; ++r) { maximum[r] = -INFINITY; minimum[r] = INFINITY; total[r] = float2(0); }
    float output[DIM * QUERY_ROWS / 32];
    TF_UNROLL
    for (ushort e = 0; e < DIM * QUERY_ROWS / 32; ++e) output[e] = 0;
    for (int first_key = first; first_key < end; first_key += KEY_ROWS) {
        auto key = keys.slice(0, first_key);
        auto scores = score_op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(query)>, metal::remove_addrspace_t<decltype(key)>, float>();
        score_op.run(query, key, scores);
        float first_scores[QUERY_ROWS * KEY_ROWS / 32];
        TF_UNROLL
        for (ushort e = 0; e < scores.get_capacity(); ++e) first_scores[e] = scores[e];
        auto q_tail = queries.slice(SCORE_DEPTH, row);
        auto k_tail = keys.slice(SCORE_DEPTH, first_key);
        score_op.run(q_tail, k_tail, scores);
        TF_UNROLL
        for (ushort e = 0; e < scores.get_capacity(); ++e) scores[e] += first_scores[e];
        float next[ROW_PARTS];
        TF_UNROLL
        for (ushort r = 0; r < ROW_PARTS; ++r) next[r] = maximum[r];
        TF_UNROLL
        for (ushort e = 0; e < scores.get_capacity(); ++e) {
            const auto at = scores.get_multidimensional_index(e);
            const int r = at[1] >> 3;
            scores[e] = first_key + at[0] < P[3] ? scores[e] * as_type<float>(P[26]) : -INFINITY;
            next[r] = max(next[r], scores[e]);
            if (first_key + at[0] < P[3]) minimum[r] = min(minimum[r], scores[e]);
        }
        TF_UNROLL
        for (ushort r = 0; r < ROW_PARTS; ++r) {
            next[r] = max(next[r], simd_shuffle_xor(next[r], ushort(1)));
            next[r] = max(next[r], simd_shuffle_xor(next[r], ushort(8)));
        }
        float2 added[ROW_PARTS];
        TF_UNROLL
        for (ushort r = 0; r < ROW_PARTS; ++r) added[r] = float2(0);
        TF_UNROLL
        for (ushort e = 0; e < scores.get_capacity(); ++e) {
            const auto at = scores.get_multidimensional_index(e);
            const int r = at[1] >> 3;
            if constexpr (DIM == 32) scores[e] = fast::exp2((scores[e] - next[r]) * 1.4426950408889634f);
            else scores[e] = precise::exp(scores[e] - next[r]);
            if constexpr (DIM == 32) added[r].x += scores[e];
            else sam_pair_add(added[r], scores[e]);
        }
        float alpha[ROW_PARTS];
        TF_UNROLL
        for (ushort r = 0; r < ROW_PARTS; ++r) {
            for (ushort offset = 1; offset <= 8; offset *= 8) {
                if constexpr (DIM == 32) added[r].x += simd_shuffle_xor(added[r].x, offset);
                else {
                    const float high = simd_shuffle_xor(added[r].x, offset), low = simd_shuffle_xor(added[r].y, offset);
                    sam_pair_add(added[r], high);
                    sam_pair_add(added[r], low);
                }
            }
            if constexpr (DIM == 32) alpha[r] = fast::exp2((maximum[r] - next[r]) * 1.4426950408889634f);
            else alpha[r] = precise::exp(maximum[r] - next[r]);
            if constexpr (DIM == 32) total[r].x = fma(total[r].x, alpha[r], added[r].x);
            else {
                sam_pair_scale(total[r], alpha[r]);
                sam_pair_add(total[r], added[r].x);
                sam_pair_add(total[r], added[r].y);
            }
            maximum[r] = next[r];
        }
        auto probabilities = value_op.template get_left_input_cooperative_tensor<float, float, float>(scores);
        auto value = values.slice(0, first_key);
        auto product = value_op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(probabilities)>, metal::remove_addrspace_t<decltype(value)>, float>();
        TF_UNROLL
        for (ushort e = 0; e < product.get_capacity(); ++e)
            product[e] = output[e] * alpha[product.get_multidimensional_index(e)[1] >> 3];
        value_op.run(probabilities, value, product);
        if (first_key + KEY_ROWS >= end) {
            TF_UNROLL
            for (ushort r = 0; r < ROW_PARTS; ++r) {
                minimum[r] = min(minimum[r], simd_shuffle_xor(minimum[r], ushort(1)));
                minimum[r] = min(minimum[r], simd_shuffle_xor(minimum[r], ushort(8)));
            }
        }
        TF_UNROLL
        for (ushort e = 0; e < product.get_capacity(); ++e) {
            const auto at = product.get_multidimensional_index(e);
            output[e] = product[e];
            if (first_key + KEY_ROWS >= end && row + at[1] < rows) {
                const int r = at[1] >> 3;
                partial[at[1] * DIM + at[0]] = output[e];
                if (at[0] == 0) {
                    stats[at[1] * 3] = maximum[r];
                    stats[at[1] * 3 + 1] = total[r].x + total[r].y;
                    stats[at[1] * 3 + 2] = -minimum[r];
                }
            }
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_split32(const device float* q, const device float* k, const device float* v,
    device float* partial, device float* stats, const device uint* flags, constant int* p,
    uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    sam_attention_device_split_impl<32>(q, k, v, partial, stats, flags, p, group, sg, lane);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_split64(const device float* q, const device float* k, const device float* v,
    device float* partial, device float* stats, const device uint* flags, constant int* p,
    uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    sam_attention_device_split_impl<64>(q, k, v, partial, stats, flags, p, group, sg, lane);
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
        const float shift = stats[base * 3] - maximum;
        const float weight = dim == 32 ? fast::exp2(shift * 1.4426950408889634f) : precise::exp(shift);
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
    if (maximum + negative_minimum > 20.0f && abs(value) <= 0x1p-48f) {
        const uint index = (uint(batch) * uint(P[1]) + uint(head)) * uint(P[5]) + uint(row) / 64 + 1;
        atomic_store_explicit(flags + index, 1u, memory_order_relaxed);
    }
}
