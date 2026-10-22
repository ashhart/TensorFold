// GLM's draft quantizers the Python family ran in torch at load (experts.quantize, qmm.quantize4); x / k = x * (1/k).

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace tf_glm_quant {

// E2M1's magnitudes in code order; experts.py's argmin keeps the first of equal distances (midpoints round down).
__device__ __constant__ float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

__device__ __forceinline__ void widen8(const uint4& u, float* x) {
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = __bfloat162float(h[j]);
}

// argmin over |v| - E2M1 (first minimum), plus 8 when v < 0 (-0.0 is not).
__device__ __forceinline__ unsigned e2m1(float v) {
    const float a = fabsf(v);
    unsigned best = 0;
    float d = fabsf(a - kE2M1[0]);
#pragma unroll
    for (unsigned i = 1; i < 8; ++i) {
        const float di = fabsf(a - kE2M1[i]);
        if (di < d) d = di, best = i;
    }
    return best + (v < 0.0f ? 8u : 0u);
}

}  // namespace tf_glm_quant

using tf_glm_quant::e2m1;
using tf_glm_quant::widen8;

// experts._quantize's per-expert scale, a block per expert: g[e] = clamp_min(amax |w[e]| * fp32(1 / 2688), 1e-30).
extern "C" __global__ void tf_glm_nvfp4_amax(
        const __nv_bfloat16* __restrict__ w, long long per, float* __restrict__ g) {
    const uint4* src = reinterpret_cast<const uint4*>(w + blockIdx.x * per);
    float m = 0.0f;
    for (long long i = threadIdx.x; i < per / 8; i += blockDim.x) {
        float x[8];
        widen8(src[i], x);
#pragma unroll
        for (int j = 0; j < 8; ++j) m = fmaxf(m, fabsf(x[j]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    __shared__ float part[32];
    if (threadIdx.x % 32 == 0) part[threadIdx.x / 32] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned i = 1; i < blockDim.x / 32; ++i) m = fmaxf(m, part[i]);
        g[blockIdx.x] = fmaxf(m * (1.0f / 2688.0f), (float)1e-30);
    }
}

// experts._quantize's blocks, a thread per 16 inputs: s = e4m3(min(amax * fp32(1 / 6) / g, 448)), codes and scales.
extern "C" __global__ void tf_glm_nvfp4_quantize(const __nv_bfloat16* __restrict__ w, const float* __restrict__ g,
                                                 long long per, long long total, uint8_t* __restrict__ codes,
                                                 uint8_t* __restrict__ scales) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float ge = g[i / per];
    const uint4* src = reinterpret_cast<const uint4*>(w) + 2 * i;
    float x[16];
    widen8(src[0], x);
    widen8(src[1], x + 8);
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) amax = fmaxf(amax, fabsf(x[j]));
    const float pre = fminf(__fdiv_rn(amax * (1.0f / 6.0f), ge), 448.0f);
    __nv_fp8_e4m3 s;
    s.__x = __nv_cvt_float_to_fp8(pre, __NV_SATFINITE, __NV_E4M3);
    const float step = static_cast<float>(s) * ge;
    const float den = fmaxf(step, (float)1e-30);
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float v = step > 0.0f ? __fdiv_rn(x[j], den) : 0.0f;
        const uint32_t c = e2m1(v);
        if (j < 8) lo |= c << (4 * j);
        else hi |= c << (4 * (j - 8));
    }
    reinterpret_cast<uint2*>(codes)[i] = make_uint2(lo, hi);
    scales[i] = s.__x;
}

// qmm.quantize4 before make_q4's tiling, a thread per 64-group: bf16 scale (hi - lo) * fp32(1 / 15) >= 1e-8, bias lo.
extern "C" __global__ void tf_glm_quantize4(const __nv_bfloat16* __restrict__ w, long long groups,
                                            int32_t* __restrict__ words, __nv_bfloat16* __restrict__ scales,
                                            __nv_bfloat16* __restrict__ biases) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= groups) return;
    const uint4* src = reinterpret_cast<const uint4*>(w) + 8 * i;
    float x[64];
#pragma unroll
    for (int t = 0; t < 8; ++t) widen8(src[t], x + 8 * t);
    float lo = x[0], hi = x[0];
#pragma unroll
    for (int j = 1; j < 64; ++j) lo = fminf(lo, x[j]), hi = fmaxf(hi, x[j]);
    // +0/-0 tie: amin keeps the zero at the largest bit-reversed position of 64; hi's sign never matters.
    if (lo == 0.0f) {
        int best = -1;
#pragma unroll
        for (int j = 0; j < 64; ++j) {
            const int r = static_cast<int>(__brev(static_cast<unsigned>(j)) >> 26);
            if (x[j] == 0.0f && r > best) best = r, lo = x[j];
        }
    }
    const __nv_bfloat16 sb = __float2bfloat16_rn(fmaxf((hi - lo) * (1.0f / 15.0f), (float)1e-8));
    const __nv_bfloat16 bb = __float2bfloat16_rn(lo);
    const float scale = __bfloat162float(sb), bias = __bfloat162float(bb);
    uint32_t out[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        uint32_t word = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float q = fminf(fmaxf(rintf(__fdiv_rn(x[8 * t + j] - bias, scale)), 0.0f), 15.0f);
            word |= static_cast<uint32_t>(q) << (4 * j);
        }
        out[t] = word;
    }
    uint4* dst = reinterpret_cast<uint4*>(words) + 2 * i;
    dst[0] = make_uint4(out[0], out[1], out[2], out[3]);
    dst[1] = make_uint4(out[4], out[5], out[6], out[7]);
    scales[i] = sb;
    biases[i] = bb;
}
