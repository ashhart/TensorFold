// norms.cu's 16-byte bf16 rows and add_rms_h with its fused MoE combine, shared with contract_controls.cu.

#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

// bf16 rows with 16-byte loads (width % 8, at most 8 * VEC_MOST * VNT, 16 B aligned): a short straight line.
namespace tf_norms {

constexpr int VNT = 256;
constexpr int VEC_MOST = 4; // 8,192 wide

__device__ __forceinline__ void unpack8(const uint4& u, float* f) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __bfloat1622float2(h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

__device__ __forceinline__ uint4 pack8(const float* f) {
    uint4 u;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    return u;
}

__device__ __forceinline__ void load8f(const float* p, float* f) {
    const float4 a = reinterpret_cast<const float4*>(p)[0];
    const float4 b = reinterpret_cast<const float4*>(p)[1];
    f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w, f[4] = b.x, f[5] = b.y, f[6] = b.z, f[7] = b.w;
}

__device__ __forceinline__ void store8f(float* p, const float* f) {
    reinterpret_cast<float4*>(p)[0] = make_float4(f[0], f[1], f[2], f[3]);
    reinterpret_cast<float4*>(p)[1] = make_float4(f[4], f[5], f[6], f[7]);
}

// 1 / sqrt(sum / width + eps), IEEE division and square root, over a VNT-thread block: warp trees, then the 8 warps'.
template <bool ONCE = false>
__device__ __forceinline__ float inv_rms_v(float part, int width, float eps) {
    __shared__ float warps[VNT / 32];
    for (int m = 16; m > 0; m >>= 1) part = __fadd_rn(part, __shfl_xor_sync(0xFFFFFFFFu, part, m));
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = part;
    __syncthreads();
    // every warp folds the warps' sums in the same tree, so each holds the same total: no broadcast
    float s = (threadIdx.x & 31) < VNT / 32 ? warps[threadIdx.x & 31] : 0.f;
    for (int m = 4; m > 0; m >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, m));
    s = __shfl_sync(0xFFFFFFFFu, s, 0);
    const float r = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(s, static_cast<float>(width)), eps)));
    if constexpr (!ONCE) __syncthreads(); // `warps` is reused by a second norm in the same block
    return r;
}

}  // namespace tf_norms

// Rows that hand on a bf16 residual and fp32 sums over each 64 outputs (the next affine matmul's group sums).
namespace tf_norms {

__device__ __forceinline__ void load8w(const float* p, float* f) { load8f(p, f); }
__device__ __forceinline__ void load8w(const __nv_bfloat16* p, float* f) { unpack8(*reinterpret_cast<const uint4*>(p), f); }

// A thread's 8 stored outputs summed in order, then the group's 8 lanes in a fixed tree (every lane of the warp calls).
__device__ __forceinline__ float group_sum8(const uint4& packed, bool live) {
    float f[8];
    unpack8(packed, f);
    float s = 0.f;
    if (live)
#pragma unroll
        for (int i = 0; i < 8; ++i) s = __fadd_rn(s, f[i]);
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 4));
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 2));
    return __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 1));
}

// Columns d..d+7's combine: w[k] * ys[k] fmas in slot order from 0, rounded to bf16 (SK: 0 fp32, 2 bf16; KEEP: not).
template <int SK, bool KEEP = false>
__device__ __forceinline__ void slot_sum8(const void* __restrict__ ys, const float* __restrict__ wt, int slots, int64_t row,
                                          int width, int d, float* rv) {
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;
    for (int k = 0; k < slots; ++k) {
        const float wk = wt[row * slots + k];
        const int64_t at = (row * slots + k) * width + d;
        float yv[8];
        if constexpr (SK == 0) load8f(static_cast<const float*>(ys) + at, yv);
        else unpack8(*reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(ys) + at), yv);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = __fmaf_rn(wk, yv[i], acc[i]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) rv[i] = KEEP ? acc[i] : __bfloat162float(__float2bfloat16_rn(acc[i]));
}

// h = bf16(x + r) (r null: h = x; SK >= 0: r is slot_sum8 of ys, wt), y = bf16(rms(h) * w), xs y's 64-value sums.
template <typename WT, int SK = -1, bool KEEP = false>
__device__ __forceinline__ void add_rms_h(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ r,
                                          const WT* __restrict__ w, __nv_bfloat16* __restrict__ h,
                                          __nv_bfloat16* __restrict__ y, float* __restrict__ xs, int width, float eps,
                                          const void* __restrict__ ys = nullptr, const float* __restrict__ wt = nullptr,
                                          int slots = 0) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * width;
    const int chunks = width / 8;
    float v[VEC_MOST][8], wv[VEC_MOST][8];
    uint4 xp[VEC_MOST], rp[VEC_MOST];
#pragma unroll
    for (int j = 0; j < VEC_MOST; ++j) { // every load issued before any is used
        const int c = threadIdx.x + j * VNT;
        if (c < chunks) {
            xp[j] = reinterpret_cast<const uint4*>(x + at)[c];
            if (SK < 0 && r != nullptr) rp[j] = reinterpret_cast<const uint4*>(r + at)[c];
            load8w(w + 8 * c, wv[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < VEC_MOST; ++j) {
        const int c = threadIdx.x + j * VNT;
        if (c < chunks) {
            unpack8(xp[j], v[j]);
            if (SK >= 0 || r != nullptr) {
                float rv[8];
                if constexpr (SK >= 0) slot_sum8<SK, KEEP>(ys, wt, slots, blockIdx.x, width, 8 * c, rv);
                else unpack8(rp[j], rv);
#pragma unroll
                for (int i = 0; i < 8; ++i) v[j][i] = __fadd_rn(v[j][i], rv[i]);
                const uint4 hp = pack8(v[j]);
                reinterpret_cast<uint4*>(h + at)[c] = hp;
                unpack8(hp, v[j]); // the norm reads h as stored
            }
        }
    }
    float part = 0.f;
#pragma unroll
    for (int j = 0; j < VEC_MOST; ++j)
        if (threadIdx.x + j * VNT < chunks)
#pragma unroll
            for (int i = 0; i < 8; ++i) part = __fmaf_rn(v[j][i], v[j][i], part);
    const float inv = inv_rms_v<true>(part, width, eps);
#pragma unroll
    for (int j = 0; j < VEC_MOST; ++j) {
        const int c = threadIdx.x + j * VNT;
        const bool live = c < chunks;
        float o[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = __fmul_rn(__fmul_rn(v[j][i], inv), wv[j][i]);
        const uint4 yp = pack8(o);
        if (live) reinterpret_cast<uint4*>(y + at)[c] = yp;
        if (xs != nullptr && j * VNT < chunks) { // whole warps of whole groups: width % 64 == 0
            const float s = group_sum8(yp, live);
            if (live && (c & 7) == 0) xs[static_cast<int64_t>(blockIdx.x) * (width / 64) + c / 8] = s;
        }
    }
}

}  // namespace tf_norms
