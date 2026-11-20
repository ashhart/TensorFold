[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_split_scores(const device float* Q, const device float* K,
    device float* probabilities, device float* stats, constant int* P,
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    using namespace tfp;
    const int split = int(tg.z) % P[24], batch = int(tg.z) / P[24];
    const int row = int(tg.x) * 64 + int(sg) * 16, rows = P[2], columns = P[6] * 32;
    const int first = split * P[25], blocks = min(P[25], P[6] - first), stride = P[25] + 2;
    Q += long(batch) * P[12] + long(tg.y) * P[13] + long(row) * P[14];
    K += long(batch) * P[15] + long(tg.y) * P[16] + long(first) * 32 * P[17];
    probabilities += ((long(batch) * P[1] + tg.y) * rows + row) * columns + first * 32;
    stats += (((long(batch) * P[1] + tg.y) * P[24] + split) * rows + row) * stride;
    const short2 home = frag_home(ushort(lane));
    const float scale2 = as_type<float>(P[26]) * 1.44269504089f;
    float top[2] = {kMasked, kMasked}, total[2] = {0, 0};
    for (int kb = 0; kb < blocks; ++kb) {
        const bool last = first + kb == P[8];
        frag<float> scores[2] = {frag<float>(0), frag<float>(0)};
        #pragma clang loop unroll_count(4)
        for (short d = 0; d < 16; ++d) {
            frag<float> q, k0, k1;
            frag_get_in(q, Q, P[14], 0, 16 * d, home, rows - row, kAnyCols);
            if (last) {
                frag_get_t_in(k0, K, P[17], 0, 16 * d, home, P[10], kAnyCols);
                frag_get_t_in(k1, K, P[17], 16, 16 * d, home, P[10], kAnyCols);
            } else {
                frag_get_t(k0, K, P[17], 0, 16 * d, home);
                frag_get_t(k1, K, P[17], 16, 16 * d, home);
            }
            mma_16x32<false, true>(scores[0], scores[1], q, k0, k1);
        }
        TF_UNROLL
        for (short f = 0; f < 2; ++f) {
            TF_UNROLL
            for (short e = 0; e < 8; ++e) {
                scores[f][e] *= scale2;
                if (last && 16 * f + home.x + TF_COL(e) >= P[10]) scores[f][e] = kMasked;
            }
        }
        float next[2] = {top[0], top[1]};
        row_fold<true>(scores[0], next);
        row_fold<true>(scores[1], next);
        TF_UNROLL
        for (short f = 0; f < 2; ++f) {
            TF_UNROLL
            for (short e = 0; e < 8; ++e) scores[f][e] = fast::exp2(scores[f][e] - next[e >> 2]);
        }
        TF_UNROLL
        for (short h = 0; h < 2; ++h) {
            const float previous = fast::exp2(top[h] - next[h]);
            top[h] = next[h];
            total[h] *= previous;
            const int r = home.y + h * 8;
            if (home.x == 0 && row + r < rows) stats[r * stride + kb] = previous;
        }
        row_fold<false>(scores[0], total);
        row_fold<false>(scores[1], total);
        frag_put_in(scores[0], probabilities, columns, 0, kb * 32, home, rows - row, columns);
        frag_put_in(scores[1], probabilities, columns, 0, kb * 32 + 16, home, rows - row, columns);
        K += 32 * P[17];
        threadgroup_barrier(mem_flags::mem_none);
    }
    TF_UNROLL
    for (short h = 0; h < 2; ++h) {
        const int r = home.y + h * 8;
        if (home.x == 0 && row + r < rows) {
            stats[r * stride + P[25]] = top[h];
            stats[r * stride + P[25] + 1] = total[h];
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_split_values(const device float* V, const device float* probabilities,
    const device float* stats, device float* partial, constant int* P,
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    using namespace tfp;
    const int part = int(tg.z) % SAM_SPLIT_VALUE_PARTS, split = int(tg.z) / SAM_SPLIT_VALUE_PARTS % P[24], batch = int(tg.z) / (SAM_SPLIT_VALUE_PARTS * P[24]);
    const int row = int(tg.x) * 64 + int(sg) * 16, rows = P[2], columns = P[6] * 32;
    const int first = split * P[25], blocks = min(P[25], P[6] - first), stride = P[25] + 2;
    V += long(batch) * P[18] + long(tg.y) * P[19] + long(first) * 32 * P[20] + part * (256 / SAM_SPLIT_VALUE_PARTS);
    probabilities += ((long(batch) * P[1] + tg.y) * rows + row) * columns + first * 32;
    stats += (((long(batch) * P[1] + tg.y) * P[24] + split) * rows + row) * stride;
    partial += (((long(batch) * P[1] + tg.y) * P[24] + split) * rows + row) * 256 + part * (256 / SAM_SPLIT_VALUE_PARTS);
    const short2 home = frag_home(ushort(lane));
    frag<float> acc[16 / SAM_SPLIT_VALUE_PARTS];
    TF_UNROLL
    for (short d = 0; d < 16 / SAM_SPLIT_VALUE_PARTS; ++d) acc[d] = frag<float>(0);
    for (int kb = 0; kb < blocks; ++kb) {
        frag<float> scores[2];
        frag_get_in(scores[0], probabilities, columns, 0, kb * 32, home, rows - row, columns);
        frag_get_in(scores[1], probabilities, columns, 0, kb * 32 + 16, home, rows - row, columns);
        float previous[2];
        TF_UNROLL
        for (short h = 0; h < 2; ++h) {
            const int r = home.y + h * 8;
            previous[h] = row + r < rows ? stats[r * stride + kb] : 0;
        }
        TF_UNROLL
        for (short d = 0; d < 16 / SAM_SPLIT_VALUE_PARTS; ++d) {
            TF_UNROLL
            for (short e = 0; e < 8; ++e) acc[d][e] *= previous[e >> 2];
        }
        simdgroup_barrier(mem_flags::mem_none);
        TF_UNROLL
        for (short d = 0; d < 16 / SAM_SPLIT_VALUE_PARTS; d += 2) {
            if (d == 4) threadgroup_barrier(mem_flags::mem_none);
            TF_UNROLL
            for (short k = 0; k < 2; ++k) {
                frag<float> v0, v1;
                if (first + kb == P[8]) {
                    frag_get_in(v0, V, P[20], k * 16, d * 16, home, P[10], kAnyCols);
                    frag_get_in(v1, V, P[20], k * 16, d * 16 + 16, home, P[10], kAnyCols);
                } else {
                    frag_get(v0, V, P[20], k * 16, d * 16, home);
                    frag_get(v1, V, P[20], k * 16, d * 16 + 16, home);
                }
                tensor_attention_pv(acc[d], acc[d + 1], scores[k], v0, v1, ushort(lane));
            }
        }
        V += 32 * P[20];
    }
    TF_UNROLL
    for (short d = 0; d < 16 / SAM_SPLIT_VALUE_PARTS; ++d) frag_put_in(acc[d], partial, 256, 0, d * 16, home, rows - row, 256);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_attention_split_merge(const device float* partial, const device float* stats,
    device float* out, constant int* P, uint3 at [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const int row = int(at.x) / 32, head = int(at.y), batch = int(at.z);
    if (row >= P[2]) return;
    const long plane = (long(batch) * P[1] + head) * P[24] * P[2];
    const int stride = P[25] + 2;
    float maximum = -INFINITY;
    for (int split = 0; split < P[24]; ++split) maximum = max(maximum, stats[(plane + long(split) * P[2] + row) * stride + P[25]]);
    float2 denominator = float2(0), sums[8];
    uint exceptional[8];
    TF_UNROLL
    for (short d = 0; d < 8; ++d) {
        sums[d] = float2(0);
        exceptional[d] = 0;
    }
    for (int split = 0; split < P[24]; ++split) {
        const long base = plane + long(split) * P[2] + row;
        const float weight = precise::exp2(stats[base * stride + P[25]] - maximum);
        const float total = stats[base * stride + P[25] + 1], term = total * weight;
        sam_pair_add(denominator, term);
        sam_pair_add(denominator, fma(total, weight, -term));
        TF_UNROLL
        for (short d = 0; d < 8; ++d) {
            const float value = partial[base * 256 + d * 32 + lane], product = value * weight;
            const uint flag = sam_exception(as_type<uint>(value), 0x3f800000u);
            exceptional[d] |= flag;
            if (flag) continue;
            sam_pair_add(sums[d], product);
            sam_pair_add(sums[d], fma(value, weight, -product));
        }
    }
    const float divisor = denominator.x + denominator.y;
    TF_UNROLL
    for (short d = 0; d < 8; ++d) {
        const uint flag = exceptional[d];
        const float value = !isfinite(divisor) || divisor <= 0 || (flag & 4u) || (flag & 3u) == 3u ? NAN
            : flag ? as_type<float>((flag & 2u) ? 0xff800000u : 0x7f800000u) : (sums[d].x + sums[d].y) / divisor;
        out[long(batch) * P[21] + long(head) * P[22] + long(row) * P[23] + d * 32 + lane] = value;
    }
}
