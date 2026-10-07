// Tile-2 BF16 matmul retains K-slice order within each column CTA.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "qmm_frag.cuh"

namespace tf_lane_gemv {

using namespace qmm_frag;

struct Part {
    const uint32_t* w;
    const __nv_bfloat16* scales;
    const __nv_bfloat16* biases;
    void* out;
    int n, npad, sk, tiles;
};

__device__ __forceinline__ uint32_t pairm(uint32_t w, int s, uint32_t mask) {
    const uint32_t t = ((w >> s) & mask) | 0x43004300u;
    uint32_t r;
    asm("sub.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(t), "r"(0x43004300u));
    return r;
}

// Column CTAs retain group_kernel chains and K-slice order.
template <int GS, int BN, int STAGES>
__global__ void __launch_bounds__(128) gemv_kernel(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,
                                                   const __grid_constant__ Part P, int M, int K, int ldx) {
    using T = LaneTile<GS, 16, BN, 1, 4, STAGES>;
    static_assert(T::MT == 1 && T::THREADS == 128, "one m16 row tile, four column warps");
    constexpr int NT = T::NT;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5;
    const int KG = K / GS, per = KG / P.sk;
    const int mine = P.tiles > (int)blockIdx.x ? (P.tiles - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
    const int items = mine * KG;
    auto stage = [&](int s) { return buf + s * T::STAGE; };
    const int rows = min(16, M);
    constexpr int TILE_BYTES = 64 * GS / 2;
    constexpr int XC = (16 * T::CHUNKS + T::THREADS - 1) / T::THREADS, WC = (T::W / 16 + T::THREADS - 1) / T::THREADS;
    constexpr int SC = (2 * T::S / 16 + T::THREADS - 1) / T::THREADS;
    const __nv_bfloat16* xsrc[XC];
    size_t woff[WC];
    int soff[SC], xdst[XC], wdst[WC], sdst[SC];
    bool xok[XC], wok[WC], sok[SC], sbias[SC];
#pragma unroll
    for (int j = 0; j < XC; ++j) {
        const int c = tid + j * T::THREADS, r = c / T::CHUNKS, ch = c % T::CHUNKS;
        xok[j] = c < 16 * T::CHUNKS && r < rows;
        xsrc[j] = x + static_cast<size_t>(min(r, rows - 1)) * ldx + ch * 8;
        xdst[j] = r * T::ROW + swz<T::CHUNKS>(r, ch) * 16;
    }
#pragma unroll
    for (int j = 0; j < WC; ++j) {
        const int c = tid + j * T::THREADS, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
        wok[j] = c < T::W / 16;
        woff[j] = static_cast<size_t>(t) * KG * TILE_BYTES + off * 16;
        wdst[j] = T::X + c * 16;
    }
#pragma unroll
    for (int j = 0; j < SC; ++j) {
        const int c = tid + j * T::THREADS, which = c / (T::S / 16), off = c % (T::S / 16);
        sok[j] = c < 2 * (T::S / 16);
        sbias[j] = which;
        soff[j] = off * 8;
        sdst[j] = T::X + T::W + which * T::S + off * 16;
    }
    const bool xsok = tid < rows;
    const float* xssrc = xs + static_cast<size_t>(min(tid, rows - 1)) * KG;
    auto tile_of_item = [&](int i) { return (int)blockIdx.x + (i / KG) * (int)gridDim.x; };
    auto load_x = [&](int s, int i) {
        const int g = i % KG;
        unsigned char* p = stage(s);
#pragma unroll
        for (int j = 0; j < XC; ++j)
            if (xok[j]) cp16(p + xdst[j], xsrc[j] + g * GS);
        if (xsok) cp4(p + T::X + T::W + 2 * T::S + tid * 4, xssrc + g);
    };
    auto load_w = [&](int s, int i) {
        const int g = i % KG, n0 = tile_of_item(i) * BN;
        unsigned char* p = stage(s);
        const unsigned char* wb = reinterpret_cast<const unsigned char*>(P.w) +
                                  static_cast<size_t>(n0 / 64) * KG * TILE_BYTES + (n0 % 64) * GS / 2;
#pragma unroll
        for (int j = 0; j < WC; ++j)
            if (wok[j]) cp16(p + wdst[j], wb + woff[j] + static_cast<size_t>(g) * TILE_BYTES);
#pragma unroll
        for (int j = 0; j < SC; ++j)
            if (sok[j]) cp16(p + sdst[j], (sbias[j] ? P.biases : P.scales) + n0 + soff[j] + static_cast<size_t>(g) * P.npad);
    };
    uint32_t mask;
    asm volatile("mov.b32 %0, 0x000F000F;\n" : "=r"(mask));

    float acc[NT][4], tot[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = tot[j][e] = 0.0f;
    for (int c = tid; c < 16 * T::CHUNKS; c += T::THREADS)
        if (c / T::CHUNKS >= rows)
#pragma unroll
            for (int s = 0; s < STAGES; ++s)
                *reinterpret_cast<uint4*>(stage(s) + c / T::CHUNKS * T::ROW + c % T::CHUNKS * 16) = uint4{};
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < items) load_w(s, s);
        commit();
    }
    grid_wait();
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < items) load_x(s, s);
        commit();
    }
    grid_launch();
    const int N = P.n;
    for (int it = 0; it < items; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < items) {
            load_x(next % STAGES, next);
            load_w(next % STAGES, next);
        }
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::W);
        const float* px = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S);
        uint32_t words[NT][GS / 32];
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[((wn * NT + j) * 32 + lane) * (GS / 32) + v];
        float d[NT][4];
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[4];
            const int r = (lane & 7) + ((lane >> 3) & 1) * 8, ch = kt * 2 + (lane >> 4);
            ldmatrix4(a, p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint32_t b0 = pairm(words[j][kt / 2], (kt & 1) * 8, mask);
                const uint32_t b1 = pairm(words[j][kt / 2], (kt & 1) * 8 + 4, mask);
                if (kt == 0) mma0(d[j], a, b0, b1);
                else mma(d[j], a, b0, b1);
            }
        }
        const int row = lane >> 2;
        const float xv[2] = {px[row], px[row + 8]};
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int col = wn * (BN / 4) + j * 8 + (lane & 3) * 2;
            const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + col);
            const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(ps + BN + col);
            const float sv[2] = {__low2float(s2), __high2float(s2)};
            const float bv[2] = {__low2float(b2), __high2float(b2)};
#pragma unroll
            for (int e = 0; e < 4; ++e)
                acc[j][e] = __fmaf_rn(xv[e >> 1], bv[e & 1], __fmaf_rn(d[j][e], sv[e & 1], acc[j][e]));
        }
        const int g = it % KG;
        if ((g + 1) % per != 0) continue;
        const bool first = g + 1 == per;
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                tot[j][e] = first ? acc[j][e] : tot[j][e] + acc[j][e];
                acc[j][e] = 0.0f;
            }
        if (g != KG - 1) continue;
        const int n0 = tile_of_item(it) * BN;
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = n0 + wn * (BN / 4) + j * 8 + (lane & 3) * 2, r = row + h * 8;
                if (r >= M) continue;
                auto* dst = reinterpret_cast<__nv_bfloat16*>(P.out) + static_cast<size_t>(r) * N + col;
                if (col + 1 < N && (N & 1) == 0) {
                    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(tot[j][2 * h], tot[j][2 * h + 1]);
                } else {
                    if (col < N) dst[0] = __float2bfloat16_rn(tot[j][2 * h]);
                    if (col + 1 < N) dst[1] = __float2bfloat16_rn(tot[j][2 * h + 1]);
                }
            }
    }
}

}  // namespace tf_lane_gemv

template __global__ void tf_lane_gemv::gemv_kernel<64, 64, 8>(const __nv_bfloat16*, const float*,
                                                             const __grid_constant__ tf_lane_gemv::Part, int, int, int);
