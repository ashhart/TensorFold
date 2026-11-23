// Routed rows over group-64 affine experts, with checkpoint SwiGLU rounding.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct Hy4ExpertShape {
    uint width, inner, first, experts, pairs, rows;
    float swiglu_limit;
};

template<uint Bits>
inline float affine_expert_dot(
    const device bfloat *input, const device uint *words, const device bfloat *scale,
    const device bfloat *bias, uint width, uint lane) {
    constexpr uint mask = (1u << Bits) - 1;
    float acc = 0.0f;
    for (uint group = 0; group < width / 64; ++group) {
        const float s = float(scale[group]), b = float(bias[group]);
        for (uint i = lane; i < 64; i += 32) {
            const uint index = group * 64 + i, bit = index * Bits;
            const float code = float((words[bit / 32] >> (bit % 32)) & mask);
            const float w = float(bfloat(fma(code, s, b)));
            acc = fma(float(input[index]), w, acc);
        }
    }
    for (ushort d = 16; d > 0; d >>= 1) acc += simd_shuffle_xor(acc, d);
    return acc;
}

template<uint Bits, bool Up, typename Output>
inline void affine_expert_rows(
    const device bfloat *x, const device int *ids, const device int *row_ids,
    const device uint *words, const device bfloat *scale, const device bfloat *bias,
    device Output *result, constant Hy4ExpertShape &s, uint2 group, uint simd, uint lane) {
    const uint pair = group.y, n = group.x * 4 + simd;
    const uint width = Up ? s.width : s.inner, size = Up ? s.inner : s.width;
    if (n >= size) return;
    const ulong out = ulong(pair) * size + n;
    const int id = ids[pair], row = Up ? row_ids[pair] : int(pair);
    if (id < 0 || uint(id) < s.first || uint(id) - s.first >= s.experts ||
        row < 0 || (Up && uint(row) >= s.rows)) {
        if (lane == 0) result[out] = Output(0);
        return;
    }
    const device bfloat *input = x + ulong(row) * width;
    const ulong expert_row = ulong(uint(id) - s.first) * (Up ? 2 * size : size) + n;
    const uint packed_width = width * Bits / 32, groups = width / 64;
    const ulong wc = expert_row * packed_width, gc = expert_row * groups;
    const float first = affine_expert_dot<Bits>(input, words + wc, scale + gc, bias + gc, width, lane);
    if (Up) {
        const ulong wu = wc + ulong(size) * packed_width, gu = gc + ulong(size) * groups;
        const float second = affine_expert_dot<Bits>(input, words + wu, scale + gu, bias + gu, width, lane);
        if (lane == 0) {
            float gate = float(bfloat(first)), value = float(bfloat(second));
            if (s.swiglu_limit > 0) {
                gate = min(gate, s.swiglu_limit);
                value = clamp(value, -s.swiglu_limit, s.swiglu_limit);
            }
            const float a = float(bfloat(precise::divide(gate, 1.0f + precise::exp(-gate))));
            result[out] = Output(a * value);
        }
    } else if (lane == 0) {
        result[out] = Output(first);
    }
}

#define TF_HY4_EXPERTS(bits, direction, up, out_type)                                                       \
kernel void tf_affine_experts_##direction##_##bits(                                                        \
    const device bfloat *x [[buffer(0)]], const device int *ids [[buffer(1)]],                               \
    const device int *rows [[buffer(2)]], const device uint *words [[buffer(3)]],                            \
    const device bfloat *scale [[buffer(4)]], const device bfloat *bias [[buffer(5)]],                       \
    device out_type *output [[buffer(6)]], constant Hy4ExpertShape &shape [[buffer(7)]],                     \
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],              \
    uint lane [[thread_index_in_simdgroup]]) {                                                              \
    affine_expert_rows<bits, up>(x, ids, rows, words, scale, bias, output, shape, group, simd, lane);         \
}

TF_HY4_EXPERTS(4, up, true, bfloat)
TF_HY4_EXPERTS(4, down, false, float)
TF_HY4_EXPERTS(8, up, true, bfloat)
TF_HY4_EXPERTS(8, down, false, float)

// All expert contributions remain FP32 until the weighted latent sum is ready for the next projection.
kernel void tf_hy4_routed_sum(
    const device float *parts [[buffer(0)]],
    const device float *weights [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant uint2 &shape [[buffer(3)]],
    uint2 at [[thread_position_in_grid]]) {
    const uint width = shape.x, picks = shape.y;
    const ulong row = ulong(at.y) * picks;
    float sum = 0.0f;
    for (uint slot = 0; slot < picks; ++slot) {
        sum = fma(weights[row + slot], parts[(row + slot) * width + at.x], sum);
    }
    out[ulong(at.y) * width + at.x] = bfloat(sum);
}
