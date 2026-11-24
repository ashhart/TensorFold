#include <metal_stdlib>
using namespace metal;

struct Sam3Rotary { uint rows, heads, head_width, rotated_rows, positions, side; float scale; };
struct Sam3RotarySplit { uint rows, width, head_width, positions; };

// Pair j of a head turns by the x angle for j < head_width/4 and by the y angle above; tables hold (cos, sin).
[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_rotary_table(device float *table [[buffer(0)]], constant Sam3Rotary &p [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    const uint pairs = p.head_width / 2u, quarter = p.head_width / 4u;
    if (i >= p.side * p.side * pairs) return;
    const uint pair = i % pairs, position = i / pairs;
    const float u = pair < quarter ? float(position % p.side) : float(position / p.side);
    const float exponent = -float(pair % quarter) / float(quarter);
    const float theta = u * p.scale * fast::pow(10000.0f, exponent);
    table[2u * i] = fast::cos(theta);
    table[2u * i + 1u] = fast::sin(theta);
}

inline float4 sam3_rotary_quad(float4 v, float4 cs) {
    return float4(
        v.x * cs.x + v.y * (-cs.y),
        v.y * cs.x + v.x * cs.y,
        v.z * cs.z + v.w * (-cs.w),
        v.w * cs.z + v.z * cs.w);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_rotate(const device float *x [[buffer(0)]], const device float *table [[buffer(1)]], device float *y [[buffer(2)]],
                 constant Sam3Rotary &p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    const uint quads = (p.heads * p.head_width) >> 2u;
    if (i >= p.rows * quads) return;
    const device float4 *x4 = (const device float4 *)x;
    device float4 *y4 = (device float4 *)y;
    const uint row = i / quads;
    if (row >= p.rotated_rows) {
        y4[i] = x4[i];
        return;
    }
    const uint head_quads = p.head_width >> 2u;
    const uint lane = (i % quads) % head_quads;
    const uint at = (row % p.positions) * head_quads + lane;
    const float4 cs = ((const device float4 *)table)[at];
    y4[i] = sam3_rotary_quad(x4[i], cs);
}

template <typename T4, bool FLAG = false>
inline void sam3_rotary_split(const device float *x, const device float *table, device T4 *q, device T4 *k, device T4 *v,
                              device atomic_uint *flags, constant Sam3RotarySplit &p, uint i) {
    const uint quads = p.width >> 2u;
    if (i >= p.rows * quads) return;
    const device float4 *x4 = (const device float4 *)x;
    const uint row = i / quads, col = i % quads, head_quads = p.head_width >> 2u;
    const uint a = row * 3u * quads + col, b = a + quads;
    const uint at = (row % p.positions) * head_quads + (col % head_quads);
    const float4 cs = ((const device float4 *)table)[at];
    const float4 qv = sam3_rotary_quad(x4[a], cs);
    const float4 kv = sam3_rotary_quad(x4[b], cs);
    const float4 vv = x4[a + 2u * quads];
    q[i] = T4(qv);
    k[i] = T4(kv);
    v[i] = T4(vv);
    if constexpr (FLAG) {
        const uint4 qb = as_type<uint4>(qv) & 0x7fffffffu;
        const uint4 kb = as_type<uint4>(kv) & 0x7fffffffu;
        const uint4 vb = as_type<uint4>(vv) & 0x7fffffffu;
        const bool qk_bad = any((qb != 0u) & ((qb < 0x2d800000u) | (qb > 0x51800000u))) ||
                            any((kb != 0u) & ((kb < 0x2d800000u) | (kb > 0x51800000u)));
        const bool v_bad = any((vb != 0u) & ((vb < 0x2d800000u) | (vb > 0x51800000u)));
        uint invalid = 0;
        if (qk_bad || (v_bad && any(vb >= 0x7f800000u))) invalid = 1u;
        else if (v_bad) invalid = 2u;
        invalid = simd_or(invalid);
        if (invalid && simd_is_first()) atomic_fetch_or_explicit(flags, invalid, memory_order_relaxed);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_rotate_split_f32(const device float *x [[buffer(0)]], const device float *table [[buffer(1)]],
                           device float *q [[buffer(2)]], device float *k [[buffer(3)]], device float *v [[buffer(4)]],
                           constant Sam3RotarySplit &p [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    sam3_rotary_split<float4, false>(x, table, (device float4 *)q, (device float4 *)k, (device float4 *)v, nullptr, p, i);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_rotate_split_f32_flag(const device float *x [[buffer(0)]], const device float *table [[buffer(1)]],
                                device float *q [[buffer(2)]], device float *k [[buffer(3)]], device float *v [[buffer(4)]],
                                device atomic_uint *flags [[buffer(5)]], constant Sam3RotarySplit &p [[buffer(6)]],
                                uint i [[thread_position_in_grid]]) {
    sam3_rotary_split<float4, true>(x, table, (device float4 *)q, (device float4 *)k, (device float4 *)v, flags, p, i);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_rotate_split_f16(const device float *x [[buffer(0)]], const device float *table [[buffer(1)]],
                           device half *q [[buffer(2)]], device half *k [[buffer(3)]], device half *v [[buffer(4)]],
                           constant Sam3RotarySplit &p [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    sam3_rotary_split<half4, false>(x, table, (device half4 *)q, (device half4 *)k, (device half4 *)v, nullptr, p, i);
}
