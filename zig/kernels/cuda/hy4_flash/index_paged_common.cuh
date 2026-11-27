// Shared primitives, Fault enum, and RoPE table kernel body for hy4_index.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

#include "common/cp_async.cuh"
#include "common/glue_math.cuh"
#include "common/reduce.cuh"

namespace tf_hy4_idx {

typedef __nv_bfloat16 bf16;

// Planted faults a gate must catch: NaN dropped, mutating input, skipping tail, fp16 sums, or higher-id ties.
enum Fault { kNone = 0, kBadNan = 1, kBadRepeat = 2, kBadTail = 3, kBadSums = 4, kBadTies = 5 };

constexpr int kDim = 128;
constexpr int kHeads = 16;
constexpr int kSlots = 4;
constexpr int kRopePairs = 32;
constexpr int kSmallMaxJobs = 48;
constexpr int kBatBk = 32;
constexpr int kPool1CtaSmem = (16 * 4 * kDim + 2 * 64 * kBatBk) * sizeof(float);
constexpr int kPoolPersistSmem = (64 * kDim + 2 * 128 * 16 + 2 * 32) * sizeof(float);

__host__ __device__ __forceinline__ int mrope_plane(int i) { return (i < 30 && (i % 3) != 0) ? i % 3 : 0; }

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

__device__ __forceinline__ uint32_t bf_pack(float lo, float hi) {
    return static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(lo))) |
           (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(hi))) << 16);
}

template <int F>
__device__ __forceinline__ float load_f(float v) {
    return (F == kBadNan && v != v) ? 0.0f : v;
}

template <int F>
__device__ __forceinline__ float step_fma(float a, float b, float acc) {
    const float s = __fmaf_rn(a, b, acc);
    return F == kBadSums ? __half2float(__float2half_rn(s)) : s;
}

template <int F>
__device__ __forceinline__ float add_f(float a, float b) {
    const float s = __fadd_rn(a, b);
    return F == kBadSums ? __half2float(__float2half_rn(s)) : s;
}

template <int F>
__device__ __forceinline__ float nmax(float a, float b) {
    if constexpr (F == kBadNan) return fmaxf(a, b);
    return (a != a || a > b) ? a : b;
}

// kBadRepeat: block 0 thread 0 bumps the first word of a buffer by one ULP on every launch.
template <int F, typename T>
__device__ __forceinline__ void bump(const T* p) {
    if (F != kBadRepeat || blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 || threadIdx.x != 0) return;
    if constexpr (sizeof(T) == 2) {
        unsigned short* w = const_cast<unsigned short*>(reinterpret_cast<const unsigned short*>(p));
        *w = static_cast<unsigned short>(*w + 1);
    } else {
        unsigned* w = const_cast<unsigned*>(reinterpret_cast<const unsigned*>(p));
        *w = *w + 1;
    }
}

__device__ __forceinline__ void rope_pair(float x0, float x1, float c, float s, float& y0, float& y1) {
    y0 = __fadd_rn(__fmul_rn(x0, c), __fmul_rn(-x1, s));
    y1 = __fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s));
}

template <int F>
__device__ __forceinline__ void rope_table_body(float2* __restrict__ CS, int first, int npos,
                                                const float* __restrict__ INV, const int* __restrict__ POS3, int ld3,
                                                int sec_h, int sec_w, int round_bf16) {
    bump<F>(INV);
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= static_cast<int64_t>(npos) * kRopePairs) return;
    const int i = static_cast<int>(e / kRopePairs), j = static_cast<int>(e % kRopePairs);
    if constexpr (F == kBadTail) {
        if (j == kRopePairs - 1) return;
    }
    int pos;
    if (POS3 != nullptr) {
        const int axis = (j % 3 == 1 && j < 3 * sec_h) ? 1 : (j % 3 == 2 && j < 3 * sec_w) ? 2 : 0;
        pos = __ldg(POS3 + static_cast<size_t>(axis) * ld3 + i);
    } else {
        pos = first + i;
    }
    const float inv = load_f<F>(__ldg(INV + j));
    float s, c;
    sincosf(__fmul_rn(static_cast<float>(pos), inv), &s, &c);
    if (round_bf16) {
        c = tf_glue::rbf(c);
        s = tf_glue::rbf(s);
    }
    CS[static_cast<size_t>(first) * kRopePairs + e] = make_float2(c, s);
}

}  // namespace tf_hy4_idx
