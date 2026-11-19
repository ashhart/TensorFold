// Hy4-flash reads independent stream gates in FP32 and stores BF16 hidden rows.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct Hy4IhcShape {
    uint rows, width, gates;
    float eps, gate_eps, post_magnitude;
};

inline float ihc_simd_sum(float score) {
    for (ushort s = 16; s > 0; s >>= 1) score += simd_shuffle_xor(score, s);
    return score;
}

kernel void tf_hy4_ihc_collapse(
    const device bfloat *hidden [[buffer(0)]],
    const device float *project [[buffer(1)]],
    const device float *bias [[buffer(2)]],
    const device float *scale [[buffer(3)]],
    device bfloat *collapsed [[buffer(4)]],
    device float *post [[buffer(5)]],
    constant Hy4IhcShape &a [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[9][8], pre[4];
    const uint wide = 4 * a.width;
    const device bfloat *x = hidden + ulong(row) * wide;
    float sq = 0.0f, logits[8] = {};
    for (uint i = t; i < wide; i += 256) {
        const float v = float(x[i]);
        sq = fma(v, v, sq);
        for (uint j = 0; j < a.gates; ++j) logits[j] = fma(v, project[j * wide + i], logits[j]);
    }
    sq = ihc_simd_sum(sq);
    if (lane == 0) partial[8][simd] = sq;
    for (uint j = 0; j < a.gates; ++j) {
        const float score = ihc_simd_sum(logits[j]);
        if (lane == 0) partial[j][simd] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < a.gates) {
        float sum = 0.0f, dot = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            sum += partial[8][i];
            dot += partial[t][i];
        }
        const float mix = dot * precise::rsqrt(sum / float(wide) + a.eps);
        const float gate = 1.0f / (1.0f + precise::exp(-(mix * scale[t / 4] + bias[t])));
        if (t < 4) pre[t] = gate + a.gate_eps;
        else post[ulong(row) * 4 + t - 4] = a.post_magnitude * gate + a.gate_eps;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < a.width; i += 256) {
        float v = 0.0f;
        for (uint stream = 0; stream < 4; ++stream) v += pre[stream] * float(x[stream * a.width + i]);
        collapsed[ulong(row) * a.width + i] = bfloat(v);
    }
}

kernel void tf_hy4_ihc_add(
    const device bfloat *residual [[buffer(0)]],
    const device bfloat *branch [[buffer(1)]],
    const device float *post [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant Hy4IhcShape &a [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
    const ulong full = ulong(tid.y) * 4 * a.width + tid.x;
    const ulong local = ulong(tid.y) * a.width + tid.x % a.width;
    output[full] = bfloat(float(residual[full]) + post[ulong(tid.y) * 4 + tid.x / a.width] * float(branch[local]));
}

// The fusion is attached only to slot zero of every DFly2 block.
kernel void tf_hy4_draft_fuse(
    device bfloat *embedded [[buffer(0)]],
    const device bfloat *delta [[buffer(1)]],
    const device bfloat *gain [[buffer(2)]],
    constant uint2 &shape [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
    const ulong anchor = ulong(tid.y) * shape.x * shape.y + tid.x;
    const bfloat update = bfloat(float(delta[ulong(tid.y) * shape.x + tid.x]) * float(gain[tid.x]));
    embedded[anchor] = bfloat(float(embedded[anchor]) + float(update));
}

kernel void tf_hy4_draft_residual(
    const device bfloat *base [[buffer(0)]],
    const device bfloat *branch [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant uint2 &shape [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
    const ulong o = ulong(tid.y) * shape.x + tid.x;
    const ulong i = ulong(tid.y) * shape.y + tid.x;
    out[o] = bfloat(float(base[i]) + float(branch[o]));
}
