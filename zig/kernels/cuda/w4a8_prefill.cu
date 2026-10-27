// Ours: prompt matmuls with int8 activations against MLX affine 4-bit weights (W4A8) on the int8 mma.
#include <cuda_bf16.h>
#include <stdint.h>

#include "common/qmm_frag.cuh"
#include "w4a8_quantize.cuh"
#include "w4a8_store.cuh"

namespace tf_w4a8 {
using namespace qmm_frag;

template <int BM, int BN, int WM, int WN, int STAGES>
struct Tile8 {
    static constexpr int GS = 64, THREADS = WM * WN * 32, MT = BM / WM / 16, NT = BN / WN / 8;
    static constexpr int X = BM * GS;               // int8 inputs a stage (64-byte rows)
    static constexpr int XF = BM * 4 * 2;           // their scales and sums
    static constexpr int W = BN * GS / 2;
    static constexpr int S = BN * 2;
    static constexpr int STAGE = X + XF + W + 2 * S;
    static constexpr int SMEM = STAGES * STAGE;
};

// Nibbles at s, s + 16, s + 4, s + 20 of w as four int8 bytes in that order (prmt: a.b0, a.b2, b.b0, b.b2).
__device__ __forceinline__ uint32_t bytes4(uint32_t w, int s) {
    const uint32_t t0 = (w >> s) & 0x000F000Fu, t1 = (w >> (s + 4)) & 0x000F000Fu;
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, 0x6420;\n" : "=r"(r) : "r"(t0), "r"(t1));
    return r;
}

__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// x through MLX affine 4-bit weights (qmm_prefill's tiles): a group's s a (q . xq), exact on the int8 mma, + b (sum x).
template <int BM, int BN, int WM, int WN, int STAGES>
__device__ __forceinline__ void prefill8_body(const uint8_t* __restrict__ xq, const float* __restrict__ xa,
        const float* __restrict__ xs, const uint32_t* __restrict__ w, const __nv_bfloat16* __restrict__ scales,
        const __nv_bfloat16* __restrict__ biases, __nv_bfloat16* __restrict__ out, int M, int N, int K, int npad, int group) {
    using T = Tile8<BM, BN, WM, WN, STAGES>;
    constexpr int GS = 64;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y;
    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * 4; c += T::THREADS) {   // 4 chunks of 16 bytes a 64-byte row
            const int r = c >> 2, ch = c & 3, row = min(m0 + r, M - 1);
            cp16z(p + r * GS + swz<4>(r, ch) * 16, xq + (size_t)row * K + g * GS + ch * 16, m0 + r < M);
        }
        float* pf = reinterpret_cast<float*>(p + T::X);
        for (int c = tid; c < 2 * BM / 4; c += T::THREADS) {  // scales then sums, 4 rows a 16-byte copy
            const int which = c / (BM / 4), r = (c % (BM / 4)) * 4;
            const float* src = which ? xs : xa;
            // rows are not contiguous per group: copy 4 bytes each
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = min(m0 + r + e, M - 1);
                cp4(pf + which * BM + r + e, src + (size_t)row * KG + g);
            }
        }
        unsigned char* pw = p + T::X + T::XF;
        constexpr int TILE_BYTES = 64 * GS / 2;
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            const size_t tile = (size_t)(n0 / 64 + t) * KG + g;
            if (n0 + t * 64 < npad) cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + tile * TILE_BYTES + off * 16);
        }
        unsigned char* ps = pw + T::W;
        for (int c = tid; c < 2 * (T::S / 16); c += T::THREADS) {
            const int which = c / (T::S / 16), off = c % (T::S / 16);
            const __nv_bfloat16* src = (which ? biases : scales) + (size_t)g * npad + n0 + off * 8;
            if (n0 + off * 8 < npad) cp16(ps + which * T::S + off * 16, src);
        }
    };
    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KG) load(s, s);
        commit();
    }
    for (int it = 0; it < KG; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < KG) load(next % STAGES, next);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const float* pf = reinterpret_cast<const float*>(p + T::X);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X + T::XF);
        const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::XF + T::W);
        int iacc[T::MT][T::NT][4];
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j) iacc[i][j][0] = iacc[i][j][1] = iacc[i][j][2] = iacc[i][j][3] = 0x4B400000;
#pragma unroll
        for (int kt = 0; kt < 2; ++kt) {                       // two 32-input steps a group
            uint32_t a[T::MT][4];
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * GS + swz<4>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const uint32_t word = pw[((wn * T::NT + j) * 32 + lane) * 2 + kt];
                const uint32_t b0 = bytes4(word, 0), b1 = bytes4(word, 8);
#pragma unroll
                for (int i = 0; i < T::MT; ++i) mma_s8(iacc[i][j], a[i], b0, b1);
            }
        }
        // accumulators start at the float 12582912: one fma drops it and applies a (rc rounds: a dot unit at most)
        float ra[T::MT][2], rs[T::MT][2], rc[T::MT][2];
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                ra[i][h] = pf[r];
                rs[i][h] = pf[BM + r];
                rc[i][h] = -12582912.0f * ra[i][h];
            }
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int c = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
            const __nv_bfloat162 sp = *reinterpret_cast<const __nv_bfloat162*>(ps + c), bp = *reinterpret_cast<const __nv_bfloat162*>(ps + BN + c);
            const float s0 = __low2float(sp), s1 = __high2float(sp), b0 = __low2float(bp), b1 = __high2float(bp);
#pragma unroll
            for (int i = 0; i < T::MT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float d0 = fmaf(__int_as_float(iacc[i][j][2 * h]), ra[i][h], rc[i][h]);
                    const float d1 = fmaf(__int_as_float(iacc[i][j][2 * h + 1]), ra[i][h], rc[i][h]);
                    acc[i][j][2 * h] = fmaf(b0, rs[i][h], fmaf(s0, d0, acc[i][j][2 * h]));
                    acc[i][j][2 * h + 1] = fmaf(b1, rs[i][h], fmaf(s1, d1, acc[i][j][2 * h + 1]));
                }
        }
    }
    wait<0>();
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M || col >= N) continue;
                __nv_bfloat16* dst = out + (size_t)row * N + col;
                tf_w4a8_store(dst, N - col, acc[i][j][2 * h], acc[i][j][2 * h + 1]);
            }
        }
}

}  // namespace tf_w4a8

extern "C" __global__ void __launch_bounds__(256) tf_w4a8_prefill(const uint8_t* __restrict__ xq, const float* __restrict__ xa,
        const float* __restrict__ xs, const uint32_t* __restrict__ w, const __nv_bfloat16* __restrict__ scales,
        const __nv_bfloat16* __restrict__ biases, __nv_bfloat16* __restrict__ out, int M, int N, int K, int npad, int group) {
    tf_w4a8::prefill8_body<128, 128, 2, 4, 4>(xq, xa, xs, w, scales, biases, out, M, N, K, npad, group);
}
