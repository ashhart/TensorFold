// Hy4-flash's compressed indexer, the parts before scoring: each row's normalized token key into the layer's key
// cache, and each four-slot group's pooled, normalized key (fp32 inside, BF16 out; RoPE runs after, on the cache).
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct Hy4IndexKeys { uint rows, capacity; float epsilon; };
struct Hy4IndexGroups { uint first, count, capacity_groups; float epsilon; };

inline float hy4_ix_value(ushort word) { return as_type<float>(uint(word) << 16); }

inline ushort hy4_ix_round(float value) {
    const uint bits = as_type<uint>(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return ushort((bits >> 16) | 0x0040u);
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// One threadgroup of 128 a row: RMS over the 128 key columns in fp32, the normalized value rounded to BF16, then
// times the BF16 gain (a BF16 product), written at the row's cache slot.
kernel void tf_hy4_index_keys(
    const device ushort *keys [[buffer(0)]],
    const device ushort *gain [[buffer(1)]],
    const device int *slots [[buffer(2)]],
    device ushort *cache [[buffer(3)]],
    constant Hy4IndexKeys &a [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[4];
    const int slot = slots[row];
    const float x = hy4_ix_value(keys[ulong(row) * 128 + t]);
    float sq = simd_sum(x * x);
    if (lane == 0) partial[simd] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = ((partial[0] + partial[1]) + partial[2]) + partial[3];
    if (slot < 0 || uint(slot) >= a.capacity) return;
    const float inv = precise::rsqrt(total / 128.0f + a.epsilon);
    const float normed = hy4_ix_value(hy4_ix_round(x * inv));
    cache[ulong(slot) * 128 + t] = hy4_ix_round(hy4_ix_value(gain[t]) * normed);
}

// One threadgroup of 128 a group (thread = output column): the four token keys through the fp32 value and gate
// projections ([out][in]), the slot softmax of the gates plus the intra-group position bias, the weighted sum, and
// the fp32 key norm, rounded to BF16 at the group's cache row. The four slots of a group are its tokens' cache
// slots 4g..4g+3.
kernel void tf_hy4_index_groups(
    const device ushort *token_keys [[buffer(0)]],
    const device float *value_w [[buffer(1)]],
    const device float *gate_w [[buffer(2)]],
    const device float *bias [[buffer(3)]],
    const device float *norm_w [[buffer(4)]],
    device ushort *group_keys [[buffer(5)]],
    constant Hy4IndexGroups &a [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint c [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float x[4][128];
    threadgroup float partial[4];
    const uint g = a.first + tg;
    if (g >= a.capacity_groups) return;
    for (uint s = 0; s < 4; ++s) x[s][c] = hy4_ix_value(token_keys[(ulong(g) * 4 + s) * 128 + c]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device float *wv = value_w + ulong(c) * 128;
    const device float *wg = gate_w + ulong(c) * 128;
    float v[4], z[4];
    for (uint s = 0; s < 4; ++s) {
        float sv = 0.0f, sg = 0.0f;
        for (uint k = 0; k < 128; ++k) {
            sv = fma(wv[k], x[s][k], sv);
            sg = fma(wg[k], x[s][k], sg);
        }
        v[s] = sv;
        z[s] = sg + bias[s * 128 + c];
    }
    const float top = max(max(z[0], z[1]), max(z[2], z[3]));
    float w[4];
    for (uint s = 0; s < 4; ++s) w[s] = precise::exp(z[s] - top);
    const float denom = ((w[0] + w[1]) + w[2]) + w[3];
    float pooled = 0.0f;
    for (uint s = 0; s < 4; ++s) pooled += v[s] * (w[s] / denom);
    const float sq = simd_sum(pooled * pooled);
    if (lane == 0) partial[simd] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = ((partial[0] + partial[1]) + partial[2]) + partial[3];
    const float inv = precise::rsqrt(total / 128.0f + a.epsilon);
    group_keys[ulong(g) * 128 + c] = hy4_ix_round(norm_w[c] * (pooled * inv));
}
