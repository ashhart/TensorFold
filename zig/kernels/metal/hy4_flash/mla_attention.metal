// Absorbed MLA using physical BF16 KV slots and rank-local attention sinks.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint lat = 512, key = 576, chunk_keys = 512, heads_per_group = 4;

struct Hy4AttendShape {
    uint rows, heads, key_stride, chunks;
};

inline float hy4_float(ushort word) { return as_type<float>(uint(word) << 16); }

inline ushort hy4_bf16(float x) {
    const uint word = as_type<uint>(x);
    if (isnan(x)) return ushort((word >> 16) | 0x40u);
    return ushort((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}

kernel void tf_hy4_attend_chunks(
    const device ushort *query [[buffer(0)]],
    const device ushort *cache [[buffer(1)]],
    const device uint *slots [[buffer(2)]],
    const device uint *counts [[buffer(3)]],
    device float *scratch [[buffer(4)]],
    constant Hy4AttendShape &shape [[buffer(5)]],
    constant float &scale [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group.z, head = group.x * heads_per_group + simd, chunk = group.y;
    if (head >= shape.heads) return;
    const ulong row_head = ulong(row) * shape.heads + head;
    const device ushort *q = query + row_head * key;
    const ulong plane = ulong(shape.rows) * shape.heads;
    device float *out = scratch + (ulong(chunk) * plane + row_head) * (lat + 2);
    const uint start = chunk * chunk_keys, end = min(start + chunk_keys, counts[row]);
    const uint latent_col = 16 * lane, rope_col = lat + 2 * lane;
    float ql[16], accum[16] = {}, top = -INFINITY, denom = 0.0f;
    for (uint i = 0; i < 16; ++i) ql[i] = hy4_float(q[latent_col + i]);
    const float q0 = hy4_float(q[rope_col]), q1 = hy4_float(q[rope_col + 1]);
    const device uint *list = slots + ulong(row) * shape.key_stride;
    for (uint j = start; j < end; ++j) {
        const device ushort *k = cache + ulong(list[j]) * key;
        float kv[16], dot = 0.0f;
        for (uint i = 0; i < 16; ++i) {
            kv[i] = hy4_float(k[latent_col + i]);
            dot = fma(ql[i], kv[i], dot);
        }
        dot = fma(q0, hy4_float(k[rope_col]), dot);
        dot = fma(q1, hy4_float(k[rope_col + 1]), dot);
        for (ushort d = 16; d > 0; d >>= 1) dot += simd_shuffle_xor(dot, d);
        const float score = dot * scale, next = max(top, score);
        const float old = precise::exp(top - next), p = precise::exp(score - next);
        denom = fma(denom, old, p);
        for (uint i = 0; i < 16; ++i) accum[i] = fma(p, kv[i], accum[i] * old);
        top = next;
    }
    if (lane == 0) {
        out[lat] = top;
        out[lat + 1] = denom;
    }
    for (uint i = 0; i < 16; ++i) out[latent_col + i] = accum[i];
}

kernel void tf_hy4_attend_merge(
    const device float *scratch [[buffer(0)]],
    const device float *sinks [[buffer(1)]],
    device ushort *output [[buffer(2)]],
    constant Hy4AttendShape &shape [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const ulong head = group.x, row_head = ulong(group.y) * shape.heads + head;
    const ulong step = ulong(shape.rows) * shape.heads * (lat + 2);
    const device float *base = scratch + row_head * (lat + 2);
    const uint column = lane * 4;
    float top = sinks[head];
    for (uint i = 0; i < shape.chunks; ++i) top = max(top, base[i * step + lat]);
    float denom = precise::exp(sinks[head] - top), acc[4] = {};
    for (uint i = 0; i < shape.chunks; ++i) {
        const device float *part = base + i * step;
        const float p = precise::exp(part[lat] - top);
        denom = fma(part[lat + 1], p, denom);
        for (uint j = 0; j < 4; ++j) acc[j] = fma(part[column + j], p, acc[j]);
    }
    for (uint j = 0; j < 4; ++j) output[row_head * lat + column + j] = hy4_bf16(precise::divide(acc[j], denom));
}
