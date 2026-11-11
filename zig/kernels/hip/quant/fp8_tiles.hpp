#pragma once

// ModelOpt FP8's product: an E4M3 code a byte, one f32 scale a projection, the sum in f32 and the scale once.

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace tf {
namespace rocm {

struct ModelFp8 {
    const void* x;        // (m, k) row major, bf16
    const void* codes;    // (n, k) row major, E4M3 one a byte
    const float* scale;   // (1): the projection's one scale
    float* out;           // (m, n)
    int m, n, k;
};

// An E4M3 code as the f32 it names, exactly: 0x7f and 0xff are NaN and refused at read.
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
        a.out[i] = acc * a.scale[0];
    }
}

}  // namespace rocm
}  // namespace tf

// This path serves bf16 activations only, as the NVFP4 product does.
extern "C" __global__ void __launch_bounds__(256) tf_fp8_gemm(tf::rocm::ModelFp8 a) {
    tf::rocm::fp8_product(a);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // PROBE: what the device reads of the arguments
        a.out[0] = static_cast<float>(a.m);
        a.out[1] = static_cast<float>(a.n);
        a.out[2] = static_cast<float>(a.k);
        a.out[3] = static_cast<float>(sizeof(tf::rocm::ModelFp8));
    }
}
