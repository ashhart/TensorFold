// Recall complete index groups in logical token order, including the causal part of the current group.
#include <metal_stdlib>
using namespace metal;

struct Hy4RecallShape { uint rows, score_groups, group_budget; };
struct Hy4RecallRow { uint group_base, token; };

kernel void tf_hy4_group_scores(
    const device bfloat *query [[buffer(0)]],
    const device bfloat *keys [[buffer(1)]],
    const device bfloat *head_weights [[buffer(2)]],
    const device Hy4RecallRow *rows [[buffer(3)]],
    device float *scores [[buffer(4)]],
    constant Hy4RecallShape &a [[buffer(5)]],
    constant float &head_scale [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint r = tg.y, group = tg.x * 4 + simd;
    const Hy4RecallRow row = rows[r];
    const uint closed = row.token / 4;
    if (closed < a.group_budget || closed > a.score_groups || group >= closed) return;
    const device bfloat *key = keys + (ulong(row.group_base) + group) * 128;
    float cached[4];
    for (uint d = 0; d < 4; ++d) cached[d] = float(key[lane + d * 32]);
    float score = 0.0f;
    for (uint head = 0; head < 16; ++head) {
        const device bfloat *q = query + (ulong(r) * 16 + head) * 128;
        float dot = 0.0f;
        for (uint d = 0; d < 4; ++d) dot = fma(float(q[lane + d * 32]), cached[d], dot);
        for (ushort d = 16; d > 0; d >>= 1) dot += simd_shuffle_xor(dot, d);
        const float w = float(head_weights[ulong(r) * 16 + head]) * head_scale;
        score = fma(w, max(dot, 0.0f), score);
    }
    if (lane == 0) scores[ulong(r) * a.score_groups + group] = score;
}

inline uint hy4_order(float x) {
    const uint bits = as_type<uint>(x + 0.0f);
    return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}

kernel void tf_hy4_recall_groups(
    const device float *scores [[buffer(0)]],
    const device Hy4RecallRow *rows [[buffer(1)]],
    device int *tokens [[buffer(2)]],
    device int *counts [[buffer(3)]],
    constant Hy4RecallShape &a [[buffer(4)]],
    uint r [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint picked[511], digit, next_need, included[8], equal[8], run_gt, run_eq;
    const uint position = rows[r].token, closed = position / 4, keep = min(closed, a.group_budget - 1);
    if (closed > a.score_groups) {
        if (t == 0) counts[r] = 0;
        return;
    }
    if (closed > keep) {
        const device float *row_scores = scores + ulong(r) * a.score_groups;
        uint prefix = 0, mask = 0, need = keep;
        for (int shift = 24; shift >= 0; shift -= 8) {
            atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = t; i < closed; i += 256) {
                const uint k = hy4_order(row_scores[i]);
                if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (t == 0) {
                uint above = 0, b = 255;
                for (;;) {
                    const uint n = atomic_load_explicit(&hist[b], memory_order_relaxed);
                    if (above + n >= need || b == 0) break;
                    above += n;
                    --b;
                }
                digit = b;
                next_need = need - above;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            prefix |= digit << shift;
            mask |= 255u << shift;
            need = next_need;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (t == 0) run_gt = run_eq = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint base = 0; base < closed; base += 256) {
            const uint i = base + t, k = i < closed ? hy4_order(row_scores[i]) : 0u;
            const uint gt = i < closed && k > prefix ? 1u : 0u;
            const uint eq = i < closed && k == prefix ? 1u : 0u;
            const uint ix_gt = simd_prefix_exclusive_sum(gt), ix_eq = simd_prefix_exclusive_sum(eq);
            if (lane == 31) { included[simd] = ix_gt + gt; equal[simd] = ix_eq + eq; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint prior_gt = run_gt + ix_gt, prior_eq = run_eq + ix_eq;
            for (uint j = 0; j < simd; ++j) { prior_gt += included[j]; prior_eq += equal[j]; }
            if (gt != 0 || (eq != 0 && prior_eq < need)) picked[prior_gt + min(prior_eq, need)] = i;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (t == 0) {
                for (uint j = 0; j < 8; ++j) { run_gt += included[j]; run_eq += equal[j]; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (uint i = t; i < keep; i += 256) picked[i] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device int *out = tokens + ulong(r) * a.group_budget * 4;
    const uint count = keep * 4 + position % 4 + 1;
    for (uint i = t; i < count; i += 256) {
        const uint group = i / 4 < keep ? picked[i / 4] : closed;
        out[i] = int(group * 4 + i % 4);
    }
    if (t == 0) counts[r] = int(count);
}
