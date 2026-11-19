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

// Chunked attention in two steps. tf_hy4_attend_scores computes every retained key's score with
// tf_hy4_attend_chunks's lane split, fma order and butterfly (so each score's bits are the same), 64 keys a
// simdgroup and four heads a threadgroup, all keys at once. tf_hy4_attend_scored then runs each chunk's online
// softmax over those scores in the original key order, the same recurrence step for step, and writes the same
// partials for tf_hy4_attend_merge. Scores: [row][head][key_stride] floats.
kernel void tf_hy4_attend_scores(
    const device ushort *query [[buffer(0)]],
    const device ushort *cache [[buffer(1)]],
    const device uint *slots [[buffer(2)]],
    const device uint *counts [[buffer(3)]],
    device float *scores [[buffer(4)]],
    constant Hy4AttendShape &shape [[buffer(5)]],
    constant float &scale [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group.z, head = group.x * heads_per_group + simd;
    if (head >= shape.heads) return;
    const ulong row_head = ulong(row) * shape.heads + head;
    const device ushort *q = query + row_head * key;
    const uint start = group.y * 64, end = min(start + 64, counts[row]);
    if (start >= end) return;
    const uint latent_col = 16 * lane, rope_col = lat + 2 * lane;
    float ql[16];
    for (uint i = 0; i < 16; ++i) ql[i] = hy4_float(q[latent_col + i]);
    const float q0 = hy4_float(q[rope_col]), q1 = hy4_float(q[rope_col + 1]);
    const device uint *list = slots + ulong(row) * shape.key_stride;
    device float *out = scores + row_head * shape.key_stride;
    for (uint j = start; j < end; ++j) {
        const device ushort *k = cache + ulong(list[j]) * key;
        float dot = 0.0f;
        for (uint i = 0; i < 16; ++i) dot = fma(ql[i], hy4_float(k[latent_col + i]), dot);
        dot = fma(q0, hy4_float(k[rope_col]), dot);
        dot = fma(q1, hy4_float(k[rope_col + 1]), dot);
        for (ushort d = 16; d > 0; d >>= 1) dot += simd_shuffle_xor(dot, d);
        if (lane == 0) out[j] = dot * scale;
    }
}

kernel void tf_hy4_attend_scored(
    const device float *scores [[buffer(0)]],
    const device ushort *cache [[buffer(1)]],
    const device uint *slots [[buffer(2)]],
    const device uint *counts [[buffer(3)]],
    device float *scratch [[buffer(4)]],
    constant Hy4AttendShape &shape [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    // The threadgroup's four heads share one row's keys: all 128 threads load 16 keys' latent rows into threadgroup
    // memory together (one memory round trip a tile, not one a key), then each head steps through the tile.
    threadgroup uint4 tile[16][64];
    const uint row = group.z, head = group.x * heads_per_group + simd, chunk = group.y;
    const bool live = head < shape.heads;
    const ulong row_head = ulong(row) * shape.heads + head;
    const ulong plane = ulong(shape.rows) * shape.heads;
    const uint start = chunk * chunk_keys, end = min(start + chunk_keys, counts[row]);
    const uint latent_col = 16 * lane;
    float accum[16] = {}, top = -INFINITY, denom = 0.0f;
    const device uint *list = slots + ulong(row) * shape.key_stride;
    const device float *score_row = scores + (live ? row_head : 0) * shape.key_stride;
    // 32 keys of scores a step (lane t: key j + t): running maxima by a prefix max across lanes (max is exact, so
    // the same values as the key-by-key scan), each key's exponentials by the same formula on the same operands.
    for (uint j = start; j < end; j += 32) {
        const uint n = min(32u, end - j);
        const float score = lane < n ? score_row[j + lane] : -INFINITY;
        float m = score;
        for (ushort d = 1; d < 32; d <<= 1) {
            const float o = simd_shuffle_up(m, d);
            if (lane >= d) m = max(m, o);
        }
        const float next = max(top, m);
        float before = simd_shuffle_up(next, 1);
        if (lane == 0) before = top;
        const float old = precise::exp(before - next), p = precise::exp(score - next);
        for (uint h = 0; h < n; h += 16) {
            const uint tn = min(16u, n - h);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = tid; e < tn * 64; e += 128) {
                const uint t = e / 64, part = e % 64;
                tile[t][part] = *((const device uint4 *)(cache + ulong(list[j + h + t]) * key) + part);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (live) {
                for (uint t = 0; t < tn; ++t) {
                    const threadgroup ushort *kv_row = (const threadgroup ushort *)&tile[t][0];
                    const float pt = simd_shuffle(p, ushort(h + t)), ot = simd_shuffle(old, ushort(h + t));
                    denom = fma(denom, ot, pt);
                    for (uint i = 0; i < 16; ++i) accum[i] = fma(pt, hy4_float(kv_row[latent_col + i]), accum[i] * ot);
                }
            }
        }
        top = simd_shuffle(next, ushort(n - 1));
    }
    if (!live) return;
    device float *out = scratch + (ulong(chunk) * plane + row_head) * (lat + 2);
    if (lane == 0) {
        out[lat] = top;
        out[lat + 1] = denom;
    }
    for (uint i = 0; i < 16; ++i) out[latent_col + i] = accum[i];
}
