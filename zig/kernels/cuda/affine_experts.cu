// CUDA grouped FFN rows over the checkpoint's 4- or 8-bit MLX affine experts (group size 64).
#include <cuda_bf16.h>
#include <stdint.h>
#include "common/reduce.cuh"

struct AffineExpertShape {
    uint32_t width, inner, first, experts, pairs, rows;
    float swiglu_limit;
};

template<int Bits>
__device__ float dot(const __nv_bfloat16 *input, const uint32_t *words, const __nv_bfloat16 *scale,
                     const __nv_bfloat16 *bias, uint32_t width, uint32_t lane) {
    constexpr uint32_t mask = (1 << Bits) - 1;
    float acc = 0.0f;
    for (uint32_t group = 0; group < width / 64; ++group) {
        const float s = __bfloat162float(scale[group]), b = __bfloat162float(bias[group]);
        for (uint32_t i = lane; i < 64; i += 32) {
            const uint32_t index = group * 64 + i;
            const uint32_t bit = index * Bits;
            const float code = float((words[bit / 32] >> (bit % 32)) & mask);
            const float w = __bfloat162float(__float2bfloat16_rn(__fmaf_rn(code, s, b)));
            acc = __fmaf_rn(__bfloat162float(input[index]), w, acc);
        }
    }
    return tf_common::warp_sum(acc);
}

template<int Bits, bool Up>
__device__ void expert_rows(const __nv_bfloat16 *x, const int32_t *ids, const int32_t *source,
                            const uint32_t *words, const __nv_bfloat16 *scale, const __nv_bfloat16 *bias,
                            void *output, AffineExpertShape a) {
    const uint32_t pair = blockIdx.y, lane = threadIdx.x % 32;
    const uint32_t n = 4 * blockIdx.x + threadIdx.x / 32;
    const uint32_t width = Up ? a.width : a.inner, size = Up ? a.inner : a.width;
    if (n >= size) return;
    const uint64_t out = uint64_t(pair) * size + n;
    const int32_t id = ids[pair];
    const int32_t row = Up ? source[pair] : pair;
    if (id < 0 || uint32_t(id) < a.first || uint32_t(id) - a.first >= a.experts ||
        row < 0 || (Up && uint32_t(row) >= a.rows)) {
        if (lane == 0) {
            if constexpr (Up) reinterpret_cast<__nv_bfloat16 *>(output)[out] = __float2bfloat16_rn(0.0f);
            else reinterpret_cast<float *>(output)[out] = 0.0f;
        }
        return;
    }

    const __nv_bfloat16 *input = x + uint64_t(row) * width;
    const uint64_t expert_row = uint64_t(uint32_t(id) - a.first) * (Up ? 2 * size : size) + n;
    const uint32_t packed_width = width * Bits / 32, groups = width / 64;
    const uint64_t wc = expert_row * packed_width, gc = expert_row * groups;
    const float first = dot<Bits>(input, words + wc, scale + gc, bias + gc, width, lane);
    if constexpr (Up) {
        const uint64_t wu = wc + uint64_t(size) * packed_width, gu = gc + uint64_t(size) * groups;
        const float second = dot<Bits>(input, words + wu, scale + gu, bias + gu, width, lane);
        if (lane == 0) {
            float gate = __bfloat162float(__float2bfloat16_rn(first));
            float up = __bfloat162float(__float2bfloat16_rn(second));
            if (a.swiglu_limit > 0) {
                gate = fminf(gate, a.swiglu_limit);
                up = fminf(a.swiglu_limit, fmaxf(-a.swiglu_limit, up));
            }
            const float activated = __bfloat162float(__float2bfloat16_rn(gate / (1.0f + expf(-gate))));
            reinterpret_cast<__nv_bfloat16 *>(output)[out] = __float2bfloat16_rn(activated * up);
        }
    } else if (lane == 0) {
        reinterpret_cast<float *>(output)[out] = first;
    }
}

#define TF_AFFINE_EXPERTS(bits, suffix, up)                                                              \
extern "C" __global__ void __launch_bounds__(128) tf_affine_experts_##suffix##_##bits(                    \
    const __nv_bfloat16 *x, const int32_t *ids, const int32_t *rows, const uint32_t *words,                \
    const __nv_bfloat16 *scale, const __nv_bfloat16 *bias, void *out, AffineExpertShape shape) {           \
    expert_rows<bits, up>(x, ids, rows, words, scale, bias, out, shape);                                  \
}

TF_AFFINE_EXPERTS(4, up, true)
TF_AFFINE_EXPERTS(4, down, false)
TF_AFFINE_EXPERTS(8, up, true)
TF_AFFINE_EXPERTS(8, down, false)
