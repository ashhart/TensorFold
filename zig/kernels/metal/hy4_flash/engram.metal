// Hy4-flash's file-backed rows enter this pass as BF16. Only the trained Engram branch simulates FP8.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

constant constexpr uint heads = 16, head_width = 96, streams = 4, threads = 128;
constant constexpr float rms_epsilon = 0x1.0p-23f;

inline float bf(float x) { return float(bfloat(x)); }

// E4M3FN rounds ties to even, includes subnormals and saturates at 448.
inline float hy4_e4m3(float x) {
    const float a = fabs(x);
    if (a == 0.0f) return x;
    const float step = exp2(max(floor(log2(a)), -6.0f) - 3.0f);
    return copysign(min(448.0f, rint(a / step) * step), x);
}

kernel void tf_hy4_engram_lookup(
    const device bfloat *raw [[buffer(0)]],
    const device bfloat *gamma [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant uint &fp8 [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint head = group.x;
    const ulong start = (ulong(group.y) * heads + head) * head_width;
    float x[3], maximum = 0.0f;
    for (uint i = 0; i < 3; ++i) {
        x[i] = float(raw[start + lane + i * 32]);
        maximum = max(maximum, fabs(x[i]));
    }
    if (fp8 != 0) {
        const float scale = bf(precise::divide(max(simd_max(maximum), bf(1e-12f)), 448.0f));
        for (uint i = 0; i < 3; ++i) {
            const float code = hy4_e4m3(clamp(bf(precise::divide(x[i], scale)), -448.0f, 448.0f));
            const float quantized = bf(code * scale);
            x[i] = bf(x[i] + bf(quantized - x[i]));
        }
    }
    const float sum = simd_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    const float inv = precise::rsqrt(sum / float(head_width) + rms_epsilon);
    for (uint i = 0; i < 3; ++i) {
        const uint col = lane + 32 * i;
        out[start + col] = bfloat((x[i] * inv) * float(gamma[head * head_width + col]));
    }
}

struct Hy4EngramArgs { uint rows, hidden; };

kernel void tf_hy4_engram_apply(
    const device bfloat *state [[buffer(0)]],
    const device bfloat *keys [[buffer(1)]],
    const device bfloat *key_bias [[buffer(2)]],
    const device bfloat *key_gain [[buffer(3)]],
    const device bfloat *query_gain [[buffer(4)]],
    const device bfloat *delta_gain [[buffer(5)]],
    const device bfloat *value [[buffer(6)]],
    const device bfloat *value_bias [[buffer(7)]],
    const device bfloat *value_gain [[buffer(8)]],
    const device uint *media_mask [[buffer(9)]],
    device bfloat *out [[buffer(10)]],
    constant Hy4EngramArgs &a [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint stream = group.x, row = group.y, h = a.hidden;
    const ulong base = (ulong(row) * streams + stream) * h;
    if (media_mask[row] != 0) {
        for (uint j = t; j < h; j += threads) out[base + j] = state[base + j];
        return;
    }
    const ulong kbase = (ulong(stream) * a.rows + row) * h, vbase = ulong(row) * h;
    const uint gain = stream * h;
    float sq_key = 0.0f, sq_query = 0.0f, sq_value = 0.0f;
    for (uint j = t; j < h; j += threads) {
        const float k = bf(float(keys[kbase + j]) + float(key_bias[gain + j]));
        const float q = float(state[base + j]);
        const float v = bf(float(value[vbase + j]) + float(value_bias[j]));
        sq_key = fma(k, k, sq_key);
        sq_query = fma(q, q, sq_query);
        sq_value = fma(v, v, sq_value);
    }
    const float key_sum = simd_sum(sq_key), query_sum = simd_sum(sq_query), value_sum = simd_sum(sq_value);
    threadgroup float partial[3][4];
    if (lane == 0) { partial[0][sg] = key_sum; partial[1][sg] = query_sum; partial[2][sg] = value_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float norm[3];
    for (uint i = 0; i < 3; ++i) {
        const float sum = (partial[i][0] + partial[i][2]) + (partial[i][1] + partial[i][3]);
        norm[i] = precise::rsqrt(sum / float(h) + rms_epsilon);
    }
    float dot = 0.0f;
    for (uint j = t; j < h; j += threads) {
        const float k = bf(float(keys[kbase + j]) + float(key_bias[gain + j]));
        const float q = float(state[base + j]);
        const float key = bf((k * norm[0]) * float(key_gain[gain + j]));
        const float query = bf((q * norm[1]) * float(query_gain[gain + j]));
        dot += bf(key * query);
    }
    const float sum = simd_sum(dot);
    threadgroup float parts[4], gate;
    if (lane == 0) parts[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        const float scaled = bf(bf((parts[0] + parts[2]) + (parts[1] + parts[3])) * precise::rsqrt(float(h)));
        const float x = scaled == 0.0f ? 0.0f : copysign(bf(precise::sqrt(max(fabs(scaled), bf(1e-6f)))), scaled);
        gate = bf(precise::divide(1.0f, 1.0f + precise::exp(-x)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = t; j < h; j += threads) {
        const float v = bf(float(value[vbase + j]) + float(value_bias[j]));
        const float signal = bf(bf(v * norm[2]) * float(value_gain[j]));
        const float delta = bf(bf(gate * signal) + bf(float(delta_gain[gain + j]) * signal));
        out[base + j] = bfloat(float(state[base + j]) + delta);
    }
}
