// Ours: qmm_prefill's prompt GEMM over AffineN's tiles (affine8.cu's layouts) for 2, 3, 5, 6 and 8-bit codes.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "affine_codes.cuh"
#include "qmm_frag.cuh"

namespace tf_affine8 {

using namespace qmm_frag;

// qmm_prefill's 128 by 128 GEMM: weights rounded once to bf16 (s * q + b), one fp32 chain over K, as for 4 bits.
template <int BITS>
struct Prefill {
    static constexpr int GS = 64, BM = 128, BN = 128, WM = 2, WN = 4, STAGES = 3;
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;
    static constexpr int NT = BN / WN / 8;
    static constexpr int ROW = GS * 2;
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;
    static constexpr int TILE_BYTES = 64 * GS * BITS / 8;  // one stored 64-column tile's group block
    static constexpr int W = BN / 64 * TILE_BYTES;
    static constexpr int S = BN * 2;
    static constexpr int STAGE = X + W + 2 * S;
    static constexpr int SMEM = STAGES * STAGE;
};

__device__ __forceinline__ uint32_t fma2(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

template <int BITS>
__device__ __forceinline__ void prefill(const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ w,
                                        const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases,
                                        __nv_bfloat16* __restrict__ out, int M, int N, int K, int npad, int ldx, int group) {
    using T = Prefill<BITS>;
    constexpr int GS = T::GS, BN = T::BN, BM = T::BM, WN = T::WN, WM = T::WM, NT = T::NT, MT = T::MT, STAGES = T::STAGES;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8, m0 + r < M);
        }
        unsigned char* pw = p + T::X;
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (T::TILE_BYTES / 16), off = c % (T::TILE_BYTES / 16);
            const size_t tile = static_cast<size_t>(n0 / 64 + t) * KG + g;
            if (n0 + t * 64 < npad) cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + tile * T::TILE_BYTES + off * 16);
        }
        unsigned char* ps = pw + T::W;
        for (int c = tid; c < 2 * (T::S / 16); c += T::THREADS) {
            const int which = c / (T::S / 16), off = c % (T::S / 16);
            const __nv_bfloat16* src = (which ? biases : scales) + static_cast<size_t>(g) * npad + n0 + off * 8;
            if (n0 + off * 8 < npad) cp16(ps + which * T::S + off * 16, src);
        }
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
    // this warp's n8 tiles are 4 wn to 4 wn + 3 of the block: stored tile wn / 2, its n8 tiles (wn % 2) * 4 + j
    const int t64 = wn / 2, sub = (wn % 2) * 4;
    for (int it = 0; it < KG; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < KG) load(next % STAGES, next);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X) + t64 * (T::TILE_BYTES / 4);
        const uint16_t* ps = reinterpret_cast<const uint16_t*>(p + T::X + T::W);
        constexpr int SETS = BITS == 8 ? NT : 2;  // a lane's word sets: one an n8 tile (8-bit), one a decode warp's pair
        constexpr int LW = BITS == 8 ? GS / 16 : BITS;
        uint32_t lw[SETS][LW], sv[NT], bv[NT];
#pragma unroll
        for (int q = 0; q < SETS; ++q)
#pragma unroll
            for (int v = 0; v < LW; ++v) lw[q][v] = pw[(((BITS == 8 ? sub : sub / 2) + q) * 32 + lane) * LW + v];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int col = wn * (BN / WN) + j * 8 + (lane >> 2);
            sv[j] = ps[col] * 0x10001u;
            bv[j] = ps[BN + col] * 0x10001u;
        }
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                uint32_t q0, q1;
                if constexpr (BITS == 8) {
                    q0 = byte_pair(lw[j][kt], 0);
                    q1 = byte_pair(lw[j][kt], 16);
                } else {
                    q0 = pair_at<BITS>(lw[j / 2], ((j % 2) * 4 + kt) * 2);
                    q1 = pair_at<BITS>(lw[j / 2], ((j % 2) * 4 + kt) * 2 + 1);
                }
                const uint32_t b0 = fma2(q0, sv[j], bv[j]);
                const uint32_t b1 = fma2(q1, sv[j], bv[j]);
#pragma unroll
                for (int i = 0; i < MT; ++i) mma(acc[i][j], a[i], b0, b1);
            }
        }
    }
    wait<0>();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                __nv_bfloat16* dst = out + static_cast<size_t>(row) * N + col;
                if (col + 1 < N && (N & 1) == 0) {
                    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                } else {
                    if (col < N) dst[0] = __float2bfloat16_rn(v0);
                    if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                }
            }
        }
}

}  // namespace tf_affine8

// The prompt GEMM a width: shared bytes 50688 + 3072 * BITS.
#define TF_AFFINE_PREFILL(B)                                                                                                    \
    static_assert(tf_affine8::Prefill<B>::SMEM == 50688 + 3072 * B, "prefill smem");                                           \
    extern "C" __global__ void __launch_bounds__(256)                                                                           \
        tf_affine##B##_prefill(const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ w,                             \
                               const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases,              \
                               __nv_bfloat16* __restrict__ out, int M, int N, int K, int npad, int ldx, int group) {            \
        tf_affine8::prefill<B>(x, w, scales, biases, out, M, N, K, npad, ldx, group);                                           \
    }

TF_AFFINE_PREFILL(2)
TF_AFFINE_PREFILL(3)
TF_AFFINE_PREFILL(5)
TF_AFFINE_PREFILL(6)
TF_AFFINE_PREFILL(8)
#undef TF_AFFINE_PREFILL
