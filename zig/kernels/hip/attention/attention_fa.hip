// Flash Attention 2 prefill: a block owns 16 query rows and walks the keys in order with an online softmax.

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <type_traits>

namespace tf {
namespace rocm {

constexpr int kFaThreads = 256;
constexpr int kFaBc = 32;
constexpr int kFaMaxD = 256;

// The query stays fp32 and the keys widen in the product: dot2 would round the query.
template <typename Cache>
using fa_q_t = float;

__device__ __forceinline__ __half fa_store(__half, float x) { return __float2half(x); }
__device__ __forceinline__ hip_bfloat16 fa_store(hip_bfloat16, float x) { return hip_bfloat16(x); }
__device__ __forceinline__ float fa_store(float, float x) { return x; }

template <typename T>
__host__ __device__ constexpr int fa_ld() {
    // Pad so 32 keys at one column sit in 32 different banks. The row itself stays contiguous.
    return sizeof(T) == 4 ? 257 : 258;
}

template <int D>
__device__ __forceinline__ float fa_dot_half(const __half* q, const __half* k) {
    const __half2* qq = reinterpret_cast<const __half2*>(q);
    const __half2* kk = reinterpret_cast<const __half2*>(k);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll 8
    for (int p = 0; p < D / 2; p += 4) {
        a0 = __builtin_amdgcn_fdot2(qq[p], kk[p], a0, false);
        a1 = __builtin_amdgcn_fdot2(qq[p + 1], kk[p + 1], a1, false);
        a2 = __builtin_amdgcn_fdot2(qq[p + 2], kk[p + 2], a2, false);
        a3 = __builtin_amdgcn_fdot2(qq[p + 3], kk[p + 3], a3, false);
    }
    return (a0 + a1) + (a2 + a3);
}

__device__ __forceinline__ float fa_dot_half(const __half* q, const __half* k, int d) {
    const __half2* qq = reinterpret_cast<const __half2*>(q);
    const __half2* kk = reinterpret_cast<const __half2*>(k);
    int pairs = d >> 1;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    int p = 0;
    for (; p + 4 <= pairs; p += 4) {
        a0 = __builtin_amdgcn_fdot2(qq[p], kk[p], a0, false);
        a1 = __builtin_amdgcn_fdot2(qq[p + 1], kk[p + 1], a1, false);
        a2 = __builtin_amdgcn_fdot2(qq[p + 2], kk[p + 2], a2, false);
        a3 = __builtin_amdgcn_fdot2(qq[p + 3], kk[p + 3], a3, false);
    }
    float acc = (a0 + a1) + (a2 + a3);
    for (; p < pairs; ++p) acc = __builtin_amdgcn_fdot2(qq[p], kk[p], acc, false);
    return acc;
}

template <int D>
__device__ __forceinline__ void fa_dot_bf2(const float* q0, const float* q1, const hip_bfloat16* k, float& acc0,
                                           float& acc1) {
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
#pragma unroll 8
    for (int dim = 0; dim < D; dim += 4) {
        float k0 = static_cast<float>(k[dim]);
        float k1 = static_cast<float>(k[dim + 1]);
        float k2 = static_cast<float>(k[dim + 2]);
        float k3 = static_cast<float>(k[dim + 3]);
        a0 = fmaf(q0[dim], k0, a0);
        b0 = fmaf(q1[dim], k0, b0);
        a1 = fmaf(q0[dim + 1], k1, a1);
        b1 = fmaf(q1[dim + 1], k1, b1);
        a2 = fmaf(q0[dim + 2], k2, a2);
        b2 = fmaf(q1[dim + 2], k2, b2);
        a3 = fmaf(q0[dim + 3], k3, a3);
        b3 = fmaf(q1[dim + 3], k3, b3);
    }
    acc0 = (a0 + a1) + (a2 + a3);
    acc1 = (b0 + b1) + (b2 + b3);
}

template <int D>
__device__ __forceinline__ float fa_dot_bf(const float* q, const hip_bfloat16* k) {
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll 8
    for (int dim = 0; dim < D; dim += 4) {
        a0 = fmaf(q[dim], static_cast<float>(k[dim]), a0);
        a1 = fmaf(q[dim + 1], static_cast<float>(k[dim + 1]), a1);
        a2 = fmaf(q[dim + 2], static_cast<float>(k[dim + 2]), a2);
        a3 = fmaf(q[dim + 3], static_cast<float>(k[dim + 3]), a3);
    }
    return (a0 + a1) + (a2 + a3);
}

__device__ __forceinline__ float fa_dot_bf(const float* q, const hip_bfloat16* k, int d) {
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    int dim = 0;
    for (; dim + 4 <= d; dim += 4) {
        a0 = fmaf(q[dim], static_cast<float>(k[dim]), a0);
        a1 = fmaf(q[dim + 1], static_cast<float>(k[dim + 1]), a1);
        a2 = fmaf(q[dim + 2], static_cast<float>(k[dim + 2]), a2);
        a3 = fmaf(q[dim + 3], static_cast<float>(k[dim + 3]), a3);
    }
    float acc = (a0 + a1) + (a2 + a3);
    for (; dim < d; ++dim) acc = fmaf(q[dim], static_cast<float>(k[dim]), acc);
    return acc;
}

template <int D>
__device__ __forceinline__ float fa_dot_f(const float* q, const float* k) {
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll 8
    for (int dim = 0; dim < D; dim += 4) {
        a0 = fmaf(q[dim], k[dim], a0);
        a1 = fmaf(q[dim + 1], k[dim + 1], a1);
        a2 = fmaf(q[dim + 2], k[dim + 2], a2);
        a3 = fmaf(q[dim + 3], k[dim + 3], a3);
    }
    return (a0 + a1) + (a2 + a3);
}

__device__ __forceinline__ float fa_dot_f(const float* q, const float* k, int d) {
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    int dim = 0;
    for (; dim + 4 <= d; dim += 4) {
        a0 = fmaf(q[dim], k[dim], a0);
        a1 = fmaf(q[dim + 1], k[dim + 1], a1);
        a2 = fmaf(q[dim + 2], k[dim + 2], a2);
        a3 = fmaf(q[dim + 3], k[dim + 3], a3);
    }
    float acc = (a0 + a1) + (a2 + a3);
    for (; dim < d; ++dim) acc = fmaf(q[dim], k[dim], acc);
    return acc;
}

template <int DSpec>
__device__ __forceinline__ float fa_qk(const __half* q, const __half* k, int d) {
    if constexpr (DSpec == 256) return fa_dot_half<256>(q, k);
    return fa_dot_half(q, k, d);
}

template <int DSpec>
__device__ __forceinline__ float fa_qk(const float* q, const hip_bfloat16* k, int d) {
    if constexpr (DSpec == 256) return fa_dot_bf<256>(q, k);
    return fa_dot_bf(q, k, d);
}

template <int DSpec>
__device__ __forceinline__ float fa_qk(const float* q, const float* k, int d) {
    if constexpr (DSpec == 256) return fa_dot_f<256>(q, k);
    return fa_dot_f(q, k, d);
}

template <int Bc>
__device__ __forceinline__ void fa_pair_half(const __half* q, const __half* k0, const __half* k1, int d, float& s0,
                                             float& s1) {
    const __half2* qq = reinterpret_cast<const __half2*>(q);
    const __half2* a = reinterpret_cast<const __half2*>(k0);
    const __half2* b = reinterpret_cast<const __half2*>(k1);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
    int pairs = d >> 1;
    int p = 0;
#pragma unroll 8
    for (; p + 4 <= pairs; p += 4) {
        __half2 q0 = qq[p];
        __half2 q1 = qq[p + 1];
        __half2 q2 = qq[p + 2];
        __half2 q3 = qq[p + 3];
        a0 = __builtin_amdgcn_fdot2(q0, a[p], a0, false);
        a1 = __builtin_amdgcn_fdot2(q1, a[p + 1], a1, false);
        a2 = __builtin_amdgcn_fdot2(q2, a[p + 2], a2, false);
        a3 = __builtin_amdgcn_fdot2(q3, a[p + 3], a3, false);
        if constexpr (Bc == 64) {
            b0 = __builtin_amdgcn_fdot2(q0, b[p], b0, false);
            b1 = __builtin_amdgcn_fdot2(q1, b[p + 1], b1, false);
            b2 = __builtin_amdgcn_fdot2(q2, b[p + 2], b2, false);
            b3 = __builtin_amdgcn_fdot2(q3, b[p + 3], b3, false);
        }
    }
    s0 = (a0 + a1) + (a2 + a3);
    s1 = (b0 + b1) + (b2 + b3);
    for (; p < pairs; ++p) {
        __half2 qv = qq[p];
        s0 = __builtin_amdgcn_fdot2(qv, a[p], s0, false);
        if constexpr (Bc == 64) s1 = __builtin_amdgcn_fdot2(qv, b[p], s1, false);
    }
}

template <int Bc>
__device__ __forceinline__ void fa_pair_hf(const float* q, const __half* k0, const __half* k1, int d, float& s0,
                                           float& s1) {
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
    int dim = 0;
#pragma unroll 8
    for (; dim + 4 <= d; dim += 4) {
        float q0 = q[dim];
        float q1 = q[dim + 1];
        float q2 = q[dim + 2];
        float q3 = q[dim + 3];
        a0 = fmaf(q0, __half2float(k0[dim]), a0);
        a1 = fmaf(q1, __half2float(k0[dim + 1]), a1);
        a2 = fmaf(q2, __half2float(k0[dim + 2]), a2);
        a3 = fmaf(q3, __half2float(k0[dim + 3]), a3);
        if constexpr (Bc == 64) {
            b0 = fmaf(q0, __half2float(k1[dim]), b0);
            b1 = fmaf(q1, __half2float(k1[dim + 1]), b1);
            b2 = fmaf(q2, __half2float(k1[dim + 2]), b2);
            b3 = fmaf(q3, __half2float(k1[dim + 3]), b3);
        }
    }
    s0 = (a0 + a1) + (a2 + a3);
    s1 = (b0 + b1) + (b2 + b3);
    for (; dim < d; ++dim) {
        s0 = fmaf(q[dim], __half2float(k0[dim]), s0);
        if constexpr (Bc == 64) s1 = fmaf(q[dim], __half2float(k1[dim]), s1);
    }
}

template <int Bc>
__device__ __forceinline__ void fa_pair_bf(const float* q, const hip_bfloat16* k0, const hip_bfloat16* k1, int d,
                                           float& s0, float& s1) {
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
    int dim = 0;
#pragma unroll 8
    for (; dim + 4 <= d; dim += 4) {
        float q0 = q[dim];
        float q1 = q[dim + 1];
        float q2 = q[dim + 2];
        float q3 = q[dim + 3];
        a0 = fmaf(q0, static_cast<float>(k0[dim]), a0);
        a1 = fmaf(q1, static_cast<float>(k0[dim + 1]), a1);
        a2 = fmaf(q2, static_cast<float>(k0[dim + 2]), a2);
        a3 = fmaf(q3, static_cast<float>(k0[dim + 3]), a3);
        if constexpr (Bc == 64) {
            b0 = fmaf(q0, static_cast<float>(k1[dim]), b0);
            b1 = fmaf(q1, static_cast<float>(k1[dim + 1]), b1);
            b2 = fmaf(q2, static_cast<float>(k1[dim + 2]), b2);
            b3 = fmaf(q3, static_cast<float>(k1[dim + 3]), b3);
        }
    }
    s0 = (a0 + a1) + (a2 + a3);
    s1 = (b0 + b1) + (b2 + b3);
    for (; dim < d; ++dim) {
        s0 = fmaf(q[dim], static_cast<float>(k0[dim]), s0);
        if constexpr (Bc == 64) s1 = fmaf(q[dim], static_cast<float>(k1[dim]), s1);
    }
}

template <int Bc>
__device__ __forceinline__ void fa_pair_f(const float* q, const float* k0, const float* k1, int d, float& s0,
                                          float& s1) {
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
    int dim = 0;
#pragma unroll 8
    for (; dim + 4 <= d; dim += 4) {
        float q0 = q[dim];
        float q1 = q[dim + 1];
        float q2 = q[dim + 2];
        float q3 = q[dim + 3];
        a0 = fmaf(q0, k0[dim], a0);
        a1 = fmaf(q1, k0[dim + 1], a1);
        a2 = fmaf(q2, k0[dim + 2], a2);
        a3 = fmaf(q3, k0[dim + 3], a3);
        if constexpr (Bc == 64) {
            b0 = fmaf(q0, k1[dim], b0);
            b1 = fmaf(q1, k1[dim + 1], b1);
            b2 = fmaf(q2, k1[dim + 2], b2);
            b3 = fmaf(q3, k1[dim + 3], b3);
        }
    }
    s0 = (a0 + a1) + (a2 + a3);
    s1 = (b0 + b1) + (b2 + b3);
    for (; dim < d; ++dim) {
        s0 = fmaf(q[dim], k0[dim], s0);
        if constexpr (Bc == 64) s1 = fmaf(q[dim], k1[dim], s1);
    }
}

template <typename Cache, int Bc>
__device__ void fa_load_tile(Cache* dst, const Cache* src, int k0, int span, int d, long long stride, int tid) {
    constexpr int ld = fa_ld<Cache>();
    const bool vec8 = sizeof(Cache) == 2 && (d % 8) == 0 && (stride % 8) == 0 &&
                      (reinterpret_cast<uintptr_t>(src) & 15u) == 0;
    if (vec8) {
        int chunks = d >> 3;
        for (int e = tid; e < Bc * chunks; e += kFaThreads) {
            int key = e / chunks;
            int chunk = e - key * chunks;
            unsigned bits0 = 0, bits1 = 0, bits2 = 0, bits3 = 0;
            if (k0 + key < span) {
                const unsigned* g = reinterpret_cast<const unsigned*>(
                    src + (static_cast<long long>(k0 + key) * stride) + chunk * 8);
                bits0 = g[0];
                bits1 = g[1];
                bits2 = g[2];
                bits3 = g[3];
            }
            unsigned* slot = reinterpret_cast<unsigned*>(dst + key * ld + chunk * 8);
            slot[0] = bits0;
            slot[1] = bits1;
            slot[2] = bits2;
            slot[3] = bits3;
        }
        return;
    }
    for (int e = tid; e < Bc * d; e += kFaThreads) {
        int key = e / d;
        int dim = e - key * d;
        float raw = 0.f;
        if (k0 + key < span) raw = static_cast<float>(src[(static_cast<long long>(k0 + key) * stride) + dim]);
        dst[key * ld + dim] = fa_store(Cache{}, raw);
    }
}

template <typename Cache, int Bc>
__global__ void __launch_bounds__(kFaThreads, 1)
fa_prefill(const float* __restrict__ q, const Cache* __restrict__ k, const Cache* __restrict__ v,
           float* __restrict__ out, int qlen, int span, int heads, int kv_heads, int d, float scale, int q_pos0,
           long long k_sb, long long k_sh, long long k_ss, long long v_sb, long long v_sh, long long v_ss) {
    constexpr int kBr = 16;
    constexpr int kWarps = kFaThreads / 32;
    constexpr int kQLd = fa_ld<fa_q_t<Cache>>();
    constexpr int kKLd = fa_ld<Cache>();
    __shared__ alignas(16) fa_q_t<Cache> sQ[kBr * kQLd];
    __shared__ alignas(16) Cache sK[Bc * kKLd];
    __shared__ float sP[kBr * Bc];
    __shared__ float sM[kBr];
    __shared__ float sL[kBr];
    __shared__ float sA[kBr];

    int q0 = blockIdx.x * kBr;
    int head = blockIdx.y;
    int batch = blockIdx.z;
    if (q0 >= qlen) return;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int q_hi = qlen - q0;
    if (q_hi > kBr) q_hi = kBr;
    int max_vis = q_pos0 + q0 + q_hi;
    if (max_vis > span) max_vis = span;
    int kv = head / (heads / kv_heads);

    const float* qrows = q + ((static_cast<long long>(batch) * heads + head) * qlen + q0) * d;
    float* orows = out + ((static_cast<long long>(batch) * heads + head) * qlen + q0) * d;
    const Cache* kbase = k + static_cast<long long>(batch) * k_sb + static_cast<long long>(kv) * k_sh;
    const Cache* vbase = v + static_cast<long long>(batch) * v_sb + static_cast<long long>(kv) * v_sh;

    for (int e = tid; e < kBr * d; e += kFaThreads) {
        int row = e / d;
        int dim = e - row * d;
        sQ[row * kQLd + dim] = row < q_hi ? qrows[row * d + dim] : 0.f;
    }
    float oreg[kBr];
#pragma unroll
    for (int row = 0; row < kBr; ++row) oreg[row] = 0.f;
    if (tid < kBr) {
        sM[tid] = -INFINITY;
        sL[tid] = 0.f;
        sA[tid] = 1.f;
    }
    __syncthreads();

    for (int k0 = 0; k0 < max_vis; k0 += Bc) {
        fa_load_tile<Cache, Bc>(sK, kbase, k0, span, d, k_ss, tid);
        __syncthreads();
#pragma unroll 1
        for (int iter = 0; iter < kBr / kWarps; ++iter) {
            int row = warp + iter * kWarps;
            float s0 = 0.f, s1 = 0.f;
            if (row < q_hi) {
                const Cache* key0 = sK + lane * kKLd;
                const Cache* key1 = Bc == 64 ? sK + (lane + 32) * kKLd : key0;
                if constexpr (std::is_same_v<Cache, __half>)
                    fa_pair_hf<Bc>(sQ + row * kQLd, key0, key1, d, s0, s1);
                else if constexpr (std::is_same_v<Cache, hip_bfloat16>)
                    fa_pair_bf<Bc>(sQ + row * kQLd, key0, key1, d, s0, s1);
                else
                    fa_pair_f<Bc>(sQ + row * kQLd, key0, key1, d, s0, s1);
                s0 *= scale;
                s1 *= scale;
                int vis = q_pos0 + q0 + row + 1;
                if (vis > span) vis = span;
                if (k0 + lane >= vis) s0 = -INFINITY;
                if (k0 + lane + 32 >= vis) s1 = -INFINITY;
            }
            if (row < kBr) {
                sP[row * Bc + lane] = s0;
                if constexpr (Bc == 64) sP[row * Bc + lane + 32] = s1;
            }
        }
        __syncthreads();
        if (tid < kBr) {
            int row = tid;
            if (row >= q_hi) {
                sA[row] = 1.f;
                for (int key = 0; key < Bc; ++key) sP[row * Bc + key] = 0.f;
            } else {
                float tile_max = -INFINITY;
                bool any = false;
                for (int key = 0; key < Bc; ++key) {
                    float score = sP[row * Bc + key];
                    if (score != -INFINITY) {
                        tile_max = fmaxf(tile_max, score);
                        any = true;
                    }
                }
                if (!any) {
                    sA[row] = 1.f;
                    for (int key = 0; key < Bc; ++key) sP[row * Bc + key] = 0.f;
                } else {
                    float next_m = fmaxf(sM[row], tile_max);
                    float alpha = sM[row] == -INFINITY ? 0.f : expf(sM[row] - next_m);
                    float sum = 0.f;
                    for (int key = 0; key < Bc; ++key) {
                        float score = sP[row * Bc + key];
                        float prob = score == -INFINITY ? 0.f : expf(score - next_m);
                        sP[row * Bc + key] = prob;
                        sum += prob;
                    }
                    sL[row] = fmaf(alpha, sL[row], sum);
                    sM[row] = next_m;
                    sA[row] = alpha;
                }
            }
        }
        __syncthreads();
        fa_load_tile<Cache, Bc>(sK, vbase, k0, span, d, v_ss, tid);
        __syncthreads();
        if (tid < d) {
            float pv[kBr];
#pragma unroll
            for (int row = 0; row < kBr; ++row) pv[row] = 0.f;
            for (int key = 0; key < Bc; ++key) {
                float vv = static_cast<float>(sK[key * kKLd + tid]);
#pragma unroll
                for (int row = 0; row < kBr; ++row) pv[row] = fmaf(sP[row * Bc + key], vv, pv[row]);
            }
#pragma unroll
            for (int row = 0; row < kBr; ++row) oreg[row] = fmaf(sA[row], oreg[row], pv[row]);
        }
        __syncthreads();
    }

    if (tid < d) {
        for (int row = 0; row < q_hi; ++row) {
            float inv = sL[row] == 0.f ? 0.f : 1.f / sL[row];
            orows[row * d + tid] = oreg[row] * inv;
        }
    }
}

}  // namespace rocm
}  // namespace tf
