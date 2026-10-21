// FP8 in a compressed-tensors export's math: rows to e4m3 under a scale a token, weights e4m3 under a scale a column.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "mma4.cuh"
#include "qmm_frag.cuh"

using namespace qmm_frag;
using namespace mma4;

namespace tf_ck8t {

constexpr int STAGES = 4, ROW = 64, W64 = 64 * 64;

__device__ __forceinline__ void unpack8(const uint4 u, float (&f)[8]) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 v = __bfloat1622float2(p[i]);
        f[2 * i] = v.x;
        f[2 * i + 1] = v.y;
    }
}

// x e4m3 [M, K] (quant8t's) times w [npad/64, K/64, 8, 32, 2, 8] (the FP8 pack), K whole: one mma chain an output.
template <int BM, int WM, bool SKIP>
__device__ __forceinline__ void lane(const uint8_t* __restrict__ x, const float* __restrict__ rs,
                                     const uint8_t* __restrict__ w, const float* __restrict__ cs,
                                     __nv_bfloat16* __restrict__ out, int M, int N, int K, int group) {
    constexpr int BN = 64, WN = 4, THREADS = WM * WN * 32, CH = ROW / 16;
    constexpr int MT = BM / WM / 16, NT = BN / WN / 8, X = BM * ROW, STAGE = (X + W64 + 127) / 128 * 128;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int r0 = warp / WN * (BM / WM), c0 = warp % WN * (BN / WN);
    const int KG = K / 64;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y;
    auto stage = [&](int s) { return buf + s * STAGE; };
    auto load = [&](int s, int kg) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * CH; c += THREADS) {
            const int r = c / CH, ch = c % CH;
            const bool in = m0 + r < M;
            cp16z(p + r * ROW + chunk<A8>(r, ch) * 16, x + static_cast<size_t>(in ? m0 + r : 0) * K + kg * ROW + ch * 16,
                  in);
        }
        unsigned char* pw = p + X;
        for (int c = tid; c < W64 / 16; c += THREADS) cp16z(pw + c * 16, w + (static_cast<size_t>(n0 / 64) * KG + kg) * W64 + c * 16, true);
    };
    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KG) load(s, s);
        commit();
    }
    const int q = lane >> 3, rr = lane & 7;
    const int live = SKIP ? min(MT, max(0, (M - m0 - r0 + 15) / 16)) : MT;
    for (int it = 0; it < KG; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < KG) load(next % STAGES, next);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const unsigned char* pw = p + X;
        uint4 b[NT];
#pragma unroll
        for (int j = 0; j < NT; ++j) b[j] = reinterpret_cast<const uint4*>(pw)[(c0 / 8 + j) * 32 + lane];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (i >= live) break;
                uint32_t a[4];
                const int r = r0 + i * 16 + rr + (q & 1) * 8;
                ldmatrix4(a, p + r * ROW + chunk<A8>(r, 2 * h + (q >> 1)) * 16);
#pragma unroll
                for (int j = 0; j < NT; ++j) mma_fp8(acc[i][j], a, h ? b[j].z : b[j].x, h ? b[j].w : b[j].y);
            }
        }
    }
    wait<0>();
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int col = n0 + c0 + j * 8 + t * 2;
            if (col >= N) continue;
            const float s0 = __ldg(cs + col), s1 = __ldg(cs + col + 1);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + r0 + i * 16 + g + h * 8;
                if (row >= M) continue;
                const float sr = __ldg(rs + row);  // (sum * row scale) * column scale, two fp32 roundings
                *reinterpret_cast<__nv_bfloat162*>(out + static_cast<size_t>(row) * N + col) =
                    __floats2bfloat162_rn(acc[i][j][2 * h] * sr * s0, acc[i][j][2 * h + 1] * sr * s1);
            }
        }
}

}  // namespace tf_ck8t

#define TF_CK8T(NAME, BM, WM, SKIP)                                                                                    \
    extern "C" __global__ void __launch_bounds__(WM * 128)                                                            \
        NAME(const uint8_t* __restrict__ x, const float* __restrict__ rs, const uint8_t* __restrict__ w,            \
             const float* __restrict__ cs, __nv_bfloat16* __restrict__ out, int M, int N, int K, int group) {        \
        tf_ck8t::lane<BM, WM, SKIP>(x, rs, w, cs, out, M, N, K, group);                                               \
    }
// Rows up to 16, up to 64, and prompt rows: each output the same chain, so the same bits at every row count.
TF_CK8T(tf_ck8t_16, 16, 1, false)
TF_CK8T(tf_ck8t_64, 64, 1, true)
TF_CK8T(tf_ck8t_128, 128, 2, true)

// A block a row: the row's amax, its scale amax / 448 (rs), e4m3(x / scale) saturating, in quant8's fragment order.
extern "C" __global__ void __launch_bounds__(128)
    tf_ck_quant8t(const __nv_bfloat16* __restrict__ x, int ldx, int K, uint8_t* __restrict__ out,
                  float* __restrict__ rs) {
    __shared__ float red[4];
    const int row = blockIdx.x, tid = threadIdx.x;
    const __nv_bfloat16* src = x + static_cast<size_t>(row) * ldx;
    float amax = 0.0f;
    for (int i = tid; i < K; i += 128) amax = fmaxf(amax, fabsf(__bfloat162float(src[i])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((tid & 31) == 0) red[tid >> 5] = amax;
    __syncthreads();
    amax = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    const float scale = __fdiv_rn(amax, 448.0f);
    if (tid == 0) rs[row] = scale;
    for (int grp = tid; grp < K / 16; grp += 128) {
        const uint4* s16 = reinterpret_cast<const uint4*>(src + grp * 16);
        float f[16];
        tf_ck8t::unpack8(s16[0], *reinterpret_cast<float(*)[8]>(f));
        tf_ck8t::unpack8(s16[1], *reinterpret_cast<float(*)[8]>(f + 8));
        uint32_t wd[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            uint32_t packed = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k = 2 * q + (j % 2) + 8 * (j / 2);
                const float v = scale > 0.0f ? __fdiv_rn(f[k], scale) : 0.0f;
                packed |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3)) << (8 * j);
            }
            wd[q] = packed;
        }
        *reinterpret_cast<uint4*>(out + static_cast<size_t>(row) * K + grp * 16) = make_uint4(wd[0], wd[1], wd[2], wd[3]);
    }
}
