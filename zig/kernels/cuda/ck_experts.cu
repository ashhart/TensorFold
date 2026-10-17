// NVFP4 experts in the checkpoint's own math: NVFP4 rows times NVFP4 weights on the block-scaled FP4 mma (sm_121a).

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "mma4.cuh"
#include "nvfp4q.cuh"

namespace tf_ck4 {

__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

__device__ __forceinline__ void unpack8(const uint4 u, float (&f)[8]) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 v = __bfloat1622float2(p[i]);
        f[2 * i] = v.x;
        f[2 * i + 1] = v.y;
    }
}

}  // namespace tf_ck4

// Row r's 16-input block b: an e4m3 scale e4m3(g * amax / 6) at scales[r][b], codes e2m1(x * g / scale) at codes[r].
extern "C" __global__ void __launch_bounds__(128)
    tf_ck_quant4(const __nv_bfloat16* __restrict__ x, int ldx, int M, int K, float g, uint8_t* __restrict__ codes,
                 uint8_t* __restrict__ scales) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;  // one thread a block of 16
    if (i >= static_cast<long long>(M) * (K / 16)) return;
    const int row = static_cast<int>(i / (K / 16)), blk = static_cast<int>(i % (K / 16));
    const uint4* src = reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * ldx + blk * 16);
    float f[16];
    tf_ck4::unpack8(src[0], *reinterpret_cast<float(*)[8]>(f));
    tf_ck4::unpack8(src[1], *reinterpret_cast<float(*)[8]>(f + 8));
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(f[i]));
    const nvfp4q::Scale sc = nvfp4q::block_scale(amax, g);
    uint32_t w[2] = {0u, 0u};
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i / 8] |= nvfp4q::e2m1(f[i] * sc.mul) << (4 * (i % 8));
    *reinterpret_cast<uint2*>(codes + static_cast<size_t>(row) * (K / 2) + blk * 8) = make_uint2(w[0], w[1]);
    scales[static_cast<size_t>(row) * (K / 16) + blk] = static_cast<uint8_t>(sc.sf8);
}

// Experts' bytes [E][n][ks/2] (inputs k0 ..) to words [E][n/64][k/64][8][32][2]: lane (g, t) of tile j, column 8j + g.
extern "C" __global__ void __launch_bounds__(256)
    tf_ck_pack4(const uint8_t* __restrict__ src, int n, int ks, int k0, int k, uint32_t* __restrict__ dst) {
    const int tile = blockIdx.x, step = blockIdx.y, e = blockIdx.z, j = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = tile * 64 + j * 8 + (lane >> 2), t = lane & 3, KG = k / 64;
    const uint8_t* row = src + (static_cast<size_t>(e) * n + col) * (ks / 2) + (k0 + step * 64) / 2;
    uint32_t* out = dst + (((static_cast<size_t>(e) * (n / 64) + tile) * KG + step) * 8 + j) * 64 + lane * 2;
    out[0] = *reinterpret_cast<const uint32_t*>(row + 4 * t);
    out[1] = *reinterpret_cast<const uint32_t*>(row + 16 + 4 * t);
}

// Experts' e4m3 scales [E][n][ks/16] (inputs k0 ..) to [E][n/64][k/64][64][4]: a 64-column tile's step together.
extern "C" __global__ void __launch_bounds__(256)
    tf_ck_pack4_scales(const uint8_t* __restrict__ src, int n, int ks, int k0, int k, uint8_t* __restrict__ dst,
                       long long total) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int q = i % 4, c = (i / 4) % 64, KG = k / 64;
    const long long tg = i / 256;
    const int g = tg % KG;
    const long long te = tg / KG;
    const int tile = te % (n / 64);
    const long long e = te / (n / 64);
    dst[i] = src[(e * n + tile * 64 + c) * (ks / 16) + (k0 + g * 64) / 16 + q];
}

// A warp: a plan item's pairs (16 at most) by 64 columns; EPI 1 relu^2 of bf16(alpha_e sum) to bf16, EPI 0 fp32 sums.
template <int EPI>
__device__ __forceinline__ void grouped(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ xs, int slots,
                                        const uint32_t* __restrict__ w, const uint8_t* __restrict__ ws,
                                        const float* __restrict__ alpha, int K, int N, const int* __restrict__ items,
                                        const int* __restrict__ counts, const int* __restrict__ members,
                                        void* __restrict__ out) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, NB = N / 64, KG = K / 64;
    const int units = __ldg(counts) * NB;
    for (int unit = blockIdx.x * 4 + (threadIdx.x >> 5); unit < units; unit += gridDim.x * 4) {
        const int it = unit / NB, cb = unit - it * NB;
        const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
        const bool v0 = g < cnt, v1 = g + 8 < cnt;
        const int p0 = v0 ? __ldg(members + first + g) : 0, p1 = v1 ? __ldg(members + first + g + 8) : 0;
        const int x0 = slots ? p0 / slots : p0, x1 = slots ? p1 / slots : p1;
        const uint8_t* c0 = codes + static_cast<size_t>(x0) * (K / 2);
        const uint8_t* c1 = codes + static_cast<size_t>(x1) * (K / 2);
        const uint8_t* s0 = xs + static_cast<size_t>(x0) * (K / 16);
        const uint8_t* s1 = xs + static_cast<size_t>(x1) * (K / 16);
        const size_t tile = (static_cast<size_t>(e) * NB + cb) * KG;
        float acc[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
        for (int kg = 0; kg < KG; ++kg) {
            uint32_t a[4];  // rows g and g + 8, inputs 8t .. and 32 + 8t .. of this step (rows past cnt: zeros)
            a[0] = v0 ? __ldg(reinterpret_cast<const uint32_t*>(c0 + kg * 32 + 4 * t)) : 0u;
            a[1] = v1 ? __ldg(reinterpret_cast<const uint32_t*>(c1 + kg * 32 + 4 * t)) : 0u;
            a[2] = v0 ? __ldg(reinterpret_cast<const uint32_t*>(c0 + kg * 32 + 16 + 4 * t)) : 0u;
            a[3] = v1 ? __ldg(reinterpret_cast<const uint32_t*>(c1 + kg * 32 + 16 + 4 * t)) : 0u;
            const bool hi = t & 1;  // lanes t 0 and 1 give rows g and g + 8's four block scales; t 2, 3 are not read
            const bool vr = hi ? v1 : v0;
            const uint32_t sa = vr ? __ldg(reinterpret_cast<const uint32_t*>((hi ? s1 : s0) + kg * 4)) : 0u;
            const uint2* wb = reinterpret_cast<const uint2*>(w) + (tile + kg) * 256;
            const uint32_t* sw = reinterpret_cast<const uint32_t*>(ws) + (tile + kg) * 64;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint2 b = __ldg(wb + j * 32 + lane);
                mma4::mma_fp4(acc[j], a, b.x, b.y, sa, __ldg(sw + j * 8 + g));
            }
        }
        const float al = __ldg(alpha + e);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            if (!(h ? v1 : v0)) continue;
            const size_t row = static_cast<size_t>(h ? p1 : p0) * N;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = cb * 64 + j * 8 + 2 * t;
                const float a0 = acc[j][2 * h] * al, a1 = acc[j][2 * h + 1] * al;
                if constexpr (EPI == 0) {
                    *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + row + col) = make_float2(a0, a1);
                } else {
                    const float u0 = fmaxf(tf_ck4::bf(a0), 0.0f), u1 = fmaxf(tf_ck4::bf(a1), 0.0f);
                    *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + row + col) =
                        __floats2bfloat162_rn(u0 * u0, u1 * u1);
                }
            }
        }
    }
}

#define TF_CK4(NAME, EPI)                                                                                              \
    extern "C" __global__ void __launch_bounds__(128)                                                                  \
        NAME(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ xs, int slots,                            \
             const uint32_t* __restrict__ w, const uint8_t* __restrict__ ws, const float* __restrict__ alpha, int K,  \
             int N, const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,   \
             void* __restrict__ out) {                                                                                 \
        grouped<EPI>(codes, xs, slots, w, ws, alpha, K, N, items, counts, members, out);                              \
    }
TF_CK4(tf_ck4_up_relu2, 1)
TF_CK4(tf_ck4_down_f32, 0)
