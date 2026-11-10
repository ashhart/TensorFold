#pragma once

// ModelOpt FP8's product: y[m, n] = scale[n] * sum over k of x[m, k] * e4m3(code[n, k]). Codes are stored one a byte,
// the scale is one f32 a projection, and an E4M3 code turns into the exact f32 it names, so the sum is f32 and the
// scale multiplies it once. One kernel for every row count, as the NVFP4 product has.

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace tf {
namespace rocm {

struct ModelFp8 {
    const void* x;        // (m, k) row major, bf16
    const void* codes;    // (n, k) row major, E4M3 one a byte
    const float* scale;   // (n): one a projection, or one a row
    float* out;           // (m, n)
    int m, n, k;
    int per_row;          // 1 when the scale table is one a row
};

// An E4M3 code as the f32 it names: exact, since three mantissa bits fit twenty-three. 0x7f and 0xff are NaN and are
// refused at read, so every byte here is a finite value.
__device__ __forceinline__ float fp8_level(unsigned c) {
    const unsigned e = (c >> 3) & 0xfu, m = c & 0x7u;
    const float v = e == 0 ? static_cast<float>(m) * 0.125f * __uint_as_float((127u - 6u) << 23)
                           : (1.0f + static_cast<float>(m) * 0.125f) * __uint_as_float((e + 120u) << 23);
    return (c & 0x80u) != 0 ? -v : v;
}

__device__ __forceinline__ void fp8_product(ModelFp8 a) {
    const long long total = static_cast<long long>(a.m) * a.n;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride) {
        const int row = static_cast<int>(i / a.n);
        const int col = static_cast<int>(i % a.n);
        const unsigned char* w = static_cast<const unsigned char*>(a.codes) + static_cast<long long>(col) * a.k;
        const hip_bfloat16* x = static_cast<const hip_bfloat16*>(a.x) + static_cast<long long>(row) * a.k;
        float acc = 0.f;
        for (int j = 0; j < a.k; ++j) acc = fmaf(static_cast<float>(x[j]), fp8_level(w[j]), acc);
        a.out[i] = acc * a.scale[a.per_row != 0 ? col : 0];
    }
}

}  // namespace rocm
}  // namespace tf

// This path serves bf16 activations only, as the NVFP4 product does.
extern "C" __global__ void __launch_bounds__(256) tf_fp8_gemm(tf::rocm::ModelFp8 a) {
    tf::rocm::fp8_product(a);
}
