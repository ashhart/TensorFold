// Dense NVFP4 x NVFP4 (checkpoint math): quant4's rows times tf_ck_pack4's tiles, one K-ordered mma chain an output.

#include <cuda_bf16.h>
#include <stdint.h>

#include "mma4.cuh"
#include "qmm_frag.cuh"

namespace tf_ckd {

constexpr int TJ = 4, U = 4;                     // decode: a warp's 4 n8 tiles (32 columns), 4 steps read ahead
constexpr int PM = 128, PN = 128, PSTAGES = 4;  // prompt: a block's rows and columns, staged 64-input steps

struct Step {
    uint32_t a[4], sa;
    uint2 b[TJ];
    uint32_t s[TJ];
};

struct PromptStage {
    uint4 a[PM][2];  // 32-byte rows, chunks swizzled for ldmatrix
    uint32_t sa[PM];
    uint2 b[2][256];  // two 64-column weight tiles as packed
    uint32_t sb[2][64];
};

}  // namespace tf_ckd

// A warp a unit of 16 rows by 32 columns; codes [M][K/2] and scales [M][K/16] (quant4), out [M][N] bf16.
extern "C" __global__ void __launch_bounds__(128)
    tf_ck4_dense(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ xs, const uint32_t* __restrict__ w,
                 const uint8_t* __restrict__ ws, float al, int M, int K, int N, __nv_bfloat16* __restrict__ out) {
    using namespace tf_ckd;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, NG = N / (8 * TJ), KG = K / 64;
    const int units = (M + 15) / 16 * NG;
    for (int unit = blockIdx.x * 4 + (threadIdx.x >> 5); unit < units; unit += gridDim.x * 4) {
        const int mb = unit / NG, n0 = (unit - mb * NG) * 8 * TJ;
        const int p0 = mb * 16 + g, p1 = p0 + 8;
        const bool v0 = p0 < M, v1 = p1 < M;
        const uint8_t* c0 = codes + static_cast<size_t>(v0 ? p0 : 0) * (K / 2);
        const uint8_t* c1 = codes + static_cast<size_t>(v1 ? p1 : 0) * (K / 2);
        const bool hi = t & 1;  // lanes t 0 and 1 give rows g and g + 8's four block scales; t 2, 3 are not read
        const bool vr = hi ? v1 : v0;
        const uint8_t* sr = xs + static_cast<size_t>(vr ? (hi ? p1 : p0) : 0) * (K / 16);
        const size_t tile = static_cast<size_t>(n0 / 64) * KG;  // the 64-column tile holding n0
        const uint2* wb = reinterpret_cast<const uint2*>(w) + tile * 256 + n0 % 64 / 8 * 32 + lane;
        const uint32_t* sw = reinterpret_cast<const uint32_t*>(ws) + tile * 64 + n0 % 64 + g;
        auto fetch = [&](int kg, Step& f) {  // rows past M: zeros
            f.a[0] = v0 ? __ldg(reinterpret_cast<const uint32_t*>(c0 + kg * 32 + 4 * t)) : 0u;
            f.a[1] = v1 ? __ldg(reinterpret_cast<const uint32_t*>(c1 + kg * 32 + 4 * t)) : 0u;
            f.a[2] = v0 ? __ldg(reinterpret_cast<const uint32_t*>(c0 + kg * 32 + 16 + 4 * t)) : 0u;
            f.a[3] = v1 ? __ldg(reinterpret_cast<const uint32_t*>(c1 + kg * 32 + 16 + 4 * t)) : 0u;
            f.sa = vr ? __ldg(reinterpret_cast<const uint32_t*>(sr + kg * 4)) : 0u;
#pragma unroll
            for (int j = 0; j < TJ; ++j) {
                f.b[j] = __ldg(wb + kg * 256 + j * 32);
                f.s[j] = __ldg(sw + kg * 64 + j * 8);
            }
        };
        float acc[TJ][4];
#pragma unroll
        for (int j = 0; j < TJ; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
        Step st[U];
#pragma unroll
        for (int u = 0; u < U - 1; ++u)
            if (u < KG) fetch(u, st[u]);
        for (int k0 = 0; k0 < KG; k0 += U) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int kg = k0 + u;
                if (kg + U - 1 < KG) fetch(kg + U - 1, st[(u + U - 1) % U]);
                if (kg < KG)
#pragma unroll
                    for (int j = 0; j < TJ; ++j)
                        mma4::mma_fp4(acc[j], st[u].a, st[u].b[j].x, st[u].b[j].y, st[u].sa, st[u].s[j]);
            }
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            if (!(h ? v1 : v0)) continue;
            const size_t row = static_cast<size_t>(h ? p1 : p0) * N;
#pragma unroll
            for (int j = 0; j < TJ; ++j) {
                const int col = n0 + j * 8 + 2 * t;
                *reinterpret_cast<__nv_bfloat162*>(out + row + col) =
                    __floats2bfloat162_rn(acc[j][2 * h] * al, acc[j][2 * h + 1] * al);
            }
        }
    }
}

// A block a 128 x 128 unit, `group` row tiles a band, steps staged in shared memory; tf_ck4_dense's bits.
extern "C" __global__ void __launch_bounds__(256)
    tf_ck4_dense_prompt(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ xs,
                        const uint32_t* __restrict__ w, const uint8_t* __restrict__ ws, float al, int M, int K, int N,
                        __nv_bfloat16* __restrict__ out, int group) {
    using namespace tf_ckd;
    __shared__ __align__(128) PromptStage st[PSTAGES];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int wm = warp >> 2, wn = warp & 3;
    const int NB = N / 64, NT = (NB + 1) / 2, KG = K / 64, units = (M + PM - 1) / PM * NT;
    const int arow = tid >> 1, half = tid & 1, bt = tid >> 7, bq = tid & 127;  // loader roles
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lchunk = lane >> 4;   // ldmatrix rows and chunk
    for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
        // `group` row tiles a band, each band's column tiles in turn: a weight tile read once a band, not a row tile
        const int2 at = qmm_frag::tile_of(unit, M, NT * PN, PM, PN, group);
        const int ct = at.y / PN, m0 = at.x, cnt = min(PM, M - m0);
        const size_t xr = m0 + min(arow, cnt - 1);  // rows past M read the last row, unused
        const uint8_t* a_src = codes + xr * (K / 2) + half * 16;
        const uint8_t* sa_src = xs + xr * (K / 16);
        const size_t tile = min(ct * 2 + bt, NB - 1);  // past N: the last tile, unread
        const uint4* b_src = reinterpret_cast<const uint4*>(w) + tile * KG * 128 + bq;
        const uint4* sb_src = reinterpret_cast<const uint4*>(ws) + tile * KG * 16 + bq;
        auto load = [&](int stage, int kg) {
            tf_ckd::PromptStage& p = st[stage];
            qmm_frag::cp16(&p.a[arow][mma4::chunk<mma4::A4>(arow, half)], a_src + kg * 32);
            if (!half) qmm_frag::cp4(&p.sa[arow], sa_src + kg * 4);
            qmm_frag::cp16(reinterpret_cast<uint4*>(p.b[bt]) + bq, b_src + kg * 128);
            if (bq < 16) qmm_frag::cp16(reinterpret_cast<uint4*>(p.sb[bt]) + bq, sb_src + kg * 16);
        };
#pragma unroll
        for (int s = 0; s < PSTAGES - 1; ++s) {
            if (s < KG) load(s, s);
            qmm_frag::commit();
        }
        const bool cols = ct * PN + wn * 32 < N;  // a warp's 32 columns are all in or all past N
        float acc[4][4][4] = {};
        for (int kg = 0; kg < KG; ++kg) {
            qmm_frag::wait<PSTAGES - 2>();
            __syncthreads();
            if (kg + PSTAGES - 1 < KG) load((kg + PSTAGES - 1) % PSTAGES, kg + PSTAGES - 1);
            qmm_frag::commit();
            if (!cols) continue;
            const tf_ckd::PromptStage& p = st[kg % PSTAGES];
            uint2 b[4];
            uint32_t sb[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int nt = (wn & 1) * 4 + j;
                b[j] = p.b[wn >> 1][nt * 32 + lane];
                sb[j] = p.sb[wn >> 1][nt * 8 + g];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r0 = wm * 64 + i * 16;
                if (r0 >= cnt) break;
                uint32_t a[4];
                const int r = r0 + lrow;
                qmm_frag::ldmatrix4(a, &p.a[r][mma4::chunk<mma4::A4>(r, lchunk)]);
                const uint32_t sa = p.sa[r0 + g + (t & 1) * 8];
#pragma unroll
                for (int j = 0; j < 4; ++j) mma4::mma_fp4(acc[i][j], a, b[j].x, b[j].y, sa, sb[j]);
            }
        }
        qmm_frag::wait<0>();
        __syncthreads();  // every warp is past the last stage before the next unit loads into it
        if (!cols) continue;
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = wm * 64 + i * 16 + g + h * 8;
                if (r >= cnt) continue;
                const size_t dst = static_cast<size_t>(m0 + r) * N;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col = ct * PN + wn * 32 + j * 8 + 2 * t;
                    *reinterpret_cast<__nv_bfloat162*>(out + dst + col) =
                        __floats2bfloat162_rn(acc[i][j][2 * h] * al, acc[i][j][2 * h + 1] * al);
                }
            }
    }
}
