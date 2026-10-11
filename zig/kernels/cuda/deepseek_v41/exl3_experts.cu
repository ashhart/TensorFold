// Copy of src/tensorfold/cuda/exl3/experts.cu (blob 8ce66ca83bce): sass_check.sh

// EXL3 routed experts, a width per expert: fixed-order splits, slots and butterflies, no atomics, so bit-exact.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>  // what the cut torch includes provided: the fixed-width integers

#include "experts_grouped.cuh"

namespace tf_exl3_experts {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

// Grouping in one block: distinct experts (< E) in id order, members row * 32 + slot in row order, -1 after the last.
constexpr int GROUP_THREADS = 1024;
constexpr int GROUP_PER_THREAD = 4;

__global__ void __launch_bounds__(GROUP_THREADS) group_kernel(const int* __restrict__ pick, int* __restrict__ uids,
                                                              int* __restrict__ ucount, int* __restrict__ members,
                                                              int R, int slots, int E, int maxm) {
    extern __shared__ int sh_pick[];
    __shared__ int warp_tot[GROUP_THREADS / 32];
    const int n = R * slots;
    for (int i = threadIdx.x; i < n; i += GROUP_THREADS) sh_pick[i] = pick[i];
    __syncthreads();
    int cnt[GROUP_PER_THREAD];
    int used = 0;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        int c = 0;
        if (e < E)
            for (int i = 0; i < n; ++i) c += sh_pick[i] == e;
        cnt[q] = c;
        used += c > 0;
    }
    // exclusive scan of `used` over threads
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inc = used;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
    }
    if (lane == 31) warp_tot[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        int v = warp_tot[lane];
        int s = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int x = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += x;
        }
        warp_tot[lane] = s - v;                                   // exclusive per warp
        if (lane == 31) ucount[0] = s;
    }
    __syncthreads();
    int place = warp_tot[warp] + inc - used;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        if (cnt[q] == 0) continue;
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        uids[place] = e;
        int j = 0;
        for (int i = 0; i < n && j < maxm; ++i)
            if (sh_pick[i] == e) members[place * maxm + j++] = (i / slots) * 32 + (i % slots);
        for (; j < maxm; ++j) members[place * maxm + j] = -1;
        ++place;
    }
}

// Walsh-Hadamard of 128 values, 4 a lane, fixed butterfly order (strides 1, 2 in registers, 4..64 across lanes).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }

// Program (member row, 128-block of K, matrix): Xh = fp16((x * suh) @ H) for gate and up of every routed slot.
template <typename TIN>
__global__ void rot_in_kernel(const TIN* __restrict__ x, int x_stride, const int* __restrict__ pick,
                              const half* __restrict__ suh0, const half* __restrict__ suh1, half* __restrict__ out0,
                              half* __restrict__ out1, int K, int slots, int E) {
    const int p = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    const int row = p / slots;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const TIN* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = to_f<TIN>(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// Program (row, 128-block of width): ordered splits, rotate, * svh, SwiGLU (0: bf16 roundings, 1: fp32), act -> Xd.
__global__ void gateup_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                       const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                       const half* __restrict__ suh_d, half* __restrict__ xd, int P, int N, int SK,
                                       int E, float limit, int act_mode) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += Z[((size_t)(0 * SK + s) * P + p) * N + n + j];
            su += Z[((size_t)(1 * SK + s) * P + p) * N + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float act;
        if (act_mode == 0) {
            float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
            float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit),
                             limit);
            act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {
            float gg = fminf(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]), limit);
            float uu = fminf(fmaxf(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]), -limit), limit);
            act = gg / (1.f + expf(-gg)) * uu;
        }
        v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (member row, 128-block of the model width): Y = (splits summed in order) @ H * svh_d, fp32.
__global__ void down_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                     const half* __restrict__ svh_d, float* __restrict__ y, int P, int D, int SK,
                                     int E) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < SK; ++k) s += Z[((size_t)k * P + p) * D + n + j];
        v[j] = s;
    }
    fwht128(v, lane);
    float* o = y + (size_t)p * D + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
}

// out[r][d] = sum over slots in order of wts[r][k] * y[r * slots + k][d] (fp32, fma chain from 0).
__global__ void combine_kernel(const float* __restrict__ y, const float* __restrict__ wts, float* __restrict__ out,
                               int D, int slots) {
    const int r = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.f;
    for (int k = 0; k < slots; ++k) acc = fmaf(wts[r * slots + k], y[((size_t)r * slots + k) * D + d], acc);
    out[(size_t)r * D + d] = acc;
}

// down_epilogue_kernel then combine_kernel in one launch, the same arithmetic in the same order (the same bits).
__device__ __forceinline__ void store_out(float* o, float v) { *o = v; }
__device__ __forceinline__ void store_out(__nv_bfloat16* o, float v) { *o = __float2bfloat16_rn(v); }

// OutT bf16: the fp32 sum rounded to nearest even as it is stored (same bits as fp32 out then to bfloat16)
template <typename OutT>
__global__ void down_combine_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                    const half* __restrict__ svh_d, float* __restrict__ y,
                                    const float* __restrict__ wts, OutT* __restrict__ out, int P, int D, int SK,
                                    int E, int slots) {
    __shared__ float4 part[32][32];                 // [slot][lane]: the slot's 4 outputs of the lane
    const int r = blockIdx.x, blk = blockIdx.y;
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 128 + 4 * lane;
    const int p = r * slots + k;
    const int e = pick[p];
    float o[4];
    if (e >= 0 && e < E) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int q = 0; q < SK; ++q) s += Z[((size_t)q * P + p) * D + n + j];
            v[j] = s;
        }
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
            y[(size_t)p * D + n + j] = o[j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = y[(size_t)p * D + n + j];
    }
    part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    if (k != 0) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < slots; ++q) {
        const float w = wts[r * slots + q];
        const float4 u = part[q][lane];
        acc[0] = fmaf(w, u.x, acc[0]);
        acc[1] = fmaf(w, u.y, acc[1]);
        acc[2] = fmaf(w, u.z, acc[2]);
        acc[3] = fmaf(w, u.w, acc[3]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) store_out(out + (size_t)r * D + n + j, acc[j]);
}

}  // namespace tf_exl3_experts

// upstream grouped_kernel at mul1 for every (nt, warps, pf) x TF_RANGES, and dequant_kernel<2, K2> (tests)
#define TF_EXL3X_INST(NT, W, PF, LO, HI) template __global__ void tf_exl3x::grouped_kernel<2, NT, W, PF, LO, HI>( \
    const half*, const half*, const int64_t*, const int64_t*, const int*, const int*, const int*, const int*, \
    const int*, float*, int, int, int, int, int, int);
TF_EXL3X_INST(8, 4, 1, 8, 8)
TF_EXL3X_INST(8, 4, 1, 2, 10)
TF_EXL3X_INST(8, 4, 1, 2, 16)
TF_EXL3X_INST(8, 4, 2, 8, 8)
TF_EXL3X_INST(8, 4, 2, 2, 10)
TF_EXL3X_INST(8, 4, 2, 2, 16)
TF_EXL3X_INST(4, 4, 2, 8, 8)
TF_EXL3X_INST(4, 4, 2, 2, 10)
TF_EXL3X_INST(4, 4, 2, 2, 16)
template __global__ void tf_exl3x::dequant_kernel<2, 2>(const uint32_t*, half*, int, int);
template __global__ void tf_exl3x::dequant_kernel<2, 3>(const uint32_t*, half*, int, int);
template __global__ void tf_exl3x::dequant_kernel<2, 4>(const uint32_t*, half*, int, int);
template __global__ void tf_exl3x::dequant_kernel<2, 5>(const uint32_t*, half*, int, int);
template __global__ void tf_exl3x::dequant_kernel<2, 6>(const uint32_t*, half*, int, int);
template __global__ void tf_exl3x::dequant_kernel<2, 7>(const uint32_t*, half*, int, int);
template __global__ void tf_exl3x::dequant_kernel<2, 8>(const uint32_t*, half*, int, int);
template __global__ void tf_exl3x::dequant_kernel<2, 9>(const uint32_t*, half*, int, int);
template __global__ void tf_exl3x::dequant_kernel<2, 10>(const uint32_t*, half*, int, int);
template __global__ void tf_exl3x::dequant_kernel<2, 11>(const uint32_t*, half*, int, int);
template __global__ void tf_exl3x::dequant_kernel<2, 12>(const uint32_t*, half*, int, int);
template __global__ void tf_exl3x::dequant_kernel<2, 13>(const uint32_t*, half*, int, int);
template __global__ void tf_exl3x::dequant_kernel<2, 14>(const uint32_t*, half*, int, int);
template __global__ void tf_exl3x::dequant_kernel<2, 15>(const uint32_t*, half*, int, int);
template __global__ void tf_exl3x::dequant_kernel<2, 16>(const uint32_t*, half*, int, int);
// down_combine_kernel at both outputs the binding takes (fp32, bf16: TF_DSV41_MOE_BF16)
template __global__ void tf_exl3_experts::down_combine_kernel<float>(const float*, const int*, const half*, float*, \
    const float*, float*, int, int, int, int, int);
template __global__ void tf_exl3_experts::down_combine_kernel<__nv_bfloat16>(const float*, const int*, const half*, float*, \
    const float*, __nv_bfloat16*, int, int, int, int, int);
// rot_in_kernel for both inputs the binding takes
template __global__ void tf_exl3_experts::rot_in_kernel<__nv_bfloat16>(const __nv_bfloat16*, int, const int*, const half*, const half*, half*, \
    half*, int, int, int);
template __global__ void tf_exl3_experts::rot_in_kernel<half>(const half*, int, const int*, const half*, const half*, half*, \
    half*, int, int, int);
