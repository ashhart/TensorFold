// Inkling 2 Mini's routed experts on HIP: w2 down decoded in registers into fp32 chains, and the activation's factors.
#include "inkling2/experts.hpp"

namespace ink2e {

// OCP E2M1 (FP4): sign, 2 exponent bits, 1 mantissa bit; code 8 is -0.
__device__ __forceinline__ float e2m1(uint32_t n) {
    const uint32_t s = (n & 8u) ? 0x80000000u : 0u;
    const uint32_t e = (n >> 1) & 3u, m = n & 1u;
    if (e == 0u) return f32(s | (m ? 0x3f000000u : 0u));
    return f32(s | ((e + 126u) << 23) | (m << 22));
}

// OCP E4M3FN, its subnormals exact; 0x7f and 0xff are NaN.
__device__ __forceinline__ float e4m3(uint32_t n) {
    const uint32_t s = (n & 0x80u) ? 0x80000000u : 0u;
    const uint32_t e = (n >> 3) & 15u, m = n & 7u;
    if (e == 15u && m == 7u) return f32(0x7fc00000u);
    if (e == 0u) return f32(s | __float_as_uint(static_cast<float>(m) * 0x1.0p-9f));
    return f32(s | ((e + 120u) << 23) | (m << 20));
}

// NVFP4's outer factor: bf16(amax) / bf16(denominator), one fp32 division.
__device__ __forceinline__ float phiOf(const float* amax, const float* den, int expert) {
    return roundbf(amax[expert]) / roundbf(den[0]);
}

// One f32 chain over a row's 16-input NVFP4 groups: w = bf16(e2m1 * e4m3 * phi), then sum + w * x in input order.
__device__ __forceinline__ float dotNv(const uint8_t* codes, const uint8_t* scales, const float* x, float phi, int k) {
    float sum = 0.0f;
    for (int c0 = 0; c0 < k; c0 += 16) {
        const uint2 q = *reinterpret_cast<const uint2*>(codes + c0 / 2);
        const float sc = e4m3(scales[c0 / 16]);
        const uint32_t word[2] = {q.x, q.y};
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint32_t code = (word[j / 8] >> (4 * (j % 8))) & 15u;  // low nibble: the even input
            const float w = roundbf(e2m1(code) * sc * phi);
            sum = sum + w * x[c0 + j];
        }
    }
    return sum;
}

// One f32 chain over a row's 32-input MXFP8 groups: w = bf16(e4m3 * 2^(scale - 127)), so scale 255 is no overflow.
__device__ __forceinline__ float dotMx(const uint8_t* codes, const uint8_t* scales, const float* x, int k) {
    float sum = 0.0f;
    for (int c0 = 0; c0 < k; c0 += 32) {
        const int ex = static_cast<int>(scales[c0 / 32]) - 127;
        const uint4 a = *reinterpret_cast<const uint4*>(codes + c0);
        const uint4 b = *reinterpret_cast<const uint4*>(codes + c0 + 16);
        const uint32_t word[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const float w = roundbf(ldexpf(e4m3((word[j / 4] >> (8 * (j % 4))) & 255u), ex));
            sum = sum + w * x[c0 + j];
        }
    }
    return sum;
}

// The block's input row (bf16) as fp32 in LDS; every thread takes part before any returns.
__device__ __forceinline__ void stage(float* xs, const uint16_t* row, int k) {
    for (int c = threadIdx.x; c < k; c += kThreads) xs[c] = bf(row[c]);
    __syncthreads();
}

}  // namespace ink2e

using namespace ink2e;

// Down: block (x, route) holds 64 outputs of the route's own activation row; one bf16 round each.
extern "C" __global__ void __launch_bounds__(kThreads)
    tf_ink2_nvfp4_down(const uint16_t* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
                       const float* __restrict__ amax, const float* __restrict__ den, const int* __restrict__ ids,
                       uint16_t* __restrict__ y, int k, int n, int experts) {
    __shared__ float xs[kMaxK];
    const int route = blockIdx.y, col = blockIdx.x * kThreads + threadIdx.x;
    stage(xs, x + static_cast<size_t>(route) * k, k);
    if (col >= n) return;
    const int e = ids[route];
    uint16_t* out = y + static_cast<size_t>(route) * n + col;
    if (e < 0 || e >= experts) {
        *out = kNan;
        return;
    }
    const size_t row = static_cast<size_t>(e) * n + col;
    *out = tobf(dotNv(w + row * (k / 2), s + row * (k / 16), xs, phiOf(amax, den, e), k));
}

extern "C" __global__ void __launch_bounds__(kThreads)
    tf_ink2_mxfp8_down(const uint16_t* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
                       const int* __restrict__ ids, uint16_t* __restrict__ y, int k, int n, int experts) {
    __shared__ float xs[kMaxK];
    const int route = blockIdx.y, col = blockIdx.x * kThreads + threadIdx.x;
    stage(xs, x + static_cast<size_t>(route) * k, k);
    if (col >= n) return;
    const int e = ids[route];
    uint16_t* out = y + static_cast<size_t>(route) * n + col;
    if (e < 0 || e >= experts) {
        *out = kNan;
        return;
    }
    const size_t row = static_cast<size_t>(e) * n + col;
    *out = tobf(dotMx(w + row * k, s + row * (k / 32), xs, k));
}

// The contract's proof: both activation factors at each given bf16 input; `native` uses the device's tanhf and expf.
extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_act_factors(const uint16_t* __restrict__ in, float* __restrict__ gate, float* __restrict__ up, int count,
                        int native) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= count) return;
    const float v = bf(in[i]);
    if (native) {
        gate[i] = 4.0f * tanhf(v / 4.0f) * (1.0f / (1.0f + expf(-v)));
        up[i] = 25.0f * tanhf(v / 25.0f);
    } else {
        gate[i] = gateFactor(v);
        up[i] = upFactor(v);
    }
}
