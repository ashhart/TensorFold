// #952's hyper-connection read-out (fn_hc_down / fn_hc_upmix bits) on prompt tiles: factored groups, norm in smem.
#include "qmm_frag.cuh"
#include "glue_math.cuh"

namespace hcf {
using namespace qmm_frag;
using tf_glue::bf;
using tf_glue::rbf;

template <int BM, int BN, int WM, int WN, int STAGES, int GPS>
struct Tile {
    static constexpr int GS = 32;
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;
    static constexpr int NT = BN / WN / 8;
    static constexpr int ROW = GS * 2;
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int XG = BM * ROW;
    static constexpr int WG = BN * 16;
    static constexpr int SG = BN * 2;
    static constexpr int X = GPS * XG;
    static constexpr int W = GPS * WG;
    static constexpr int S = GPS * SG;
    static constexpr int XS = GPS * BM * 4;
    static constexpr int SC = GPS * GS * 4;            // the norm's scale for the stage's groups
    static constexpr int STAGE = X + W + 2 * S + XS + SC;
    static constexpr int RINV = BM * 4 * 4;
    static constexpr int SMEM = STAGES * STAGE + RINV;
};

// fn_hc's nibbles(w, 2t): codes 2t (low half) and 2t + 1 (high half) as exact bf16 integers.
__device__ __forceinline__ uint32_t nib2(uint32_t w, int t) {
    const uint32_t v = w >> (8 * t);
    const uint32_t q = (v & 0xFu) | ((v << 12) & 0xF0000u) | 0x43004300u;
    uint32_t r;
#if __CUDA_ARCH__ >= 900
    asm("sub.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(q), "r"(0x43004300u));
#else
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(q), "r"(0x3F803F80u), "r"(0xC300C300u));  // qmm_frag::pair's form
#endif
    return r;
}

template <int BM, int BN, int WM, int WN, int STAGES, int GPS, bool NORM, bool FOLD, bool OUT16, int MINB = 1>
__global__ void __launch_bounds__(WM * WN * 32, MINB) hcf_kernel(
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const uint4* __restrict__ w,
        const __nv_bfloat16* __restrict__ sc, const __nv_bfloat16* __restrict__ bi, void* __restrict__ out,
        int M, int N, int K, int SK, int ldx, int group, const float* __restrict__ pss, const float* __restrict__ scale,
        float eps, int D, int NC, int SS) {
    using T = Tile<BM, BN, WM, WN, STAGES, GPS>;
    constexpr int GS = T::GS;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, t4 = lane & 3;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, cut = KG / SK, npad = (N + 63) / 64 * 64;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y;
    const int stages_n = KG / GPS;
    float* rinv_s = reinterpret_cast<float*>(buf + STAGES * T::STAGE);

    if constexpr (NORM) {
        for (int i = tid; i < BM * SS; i += T::THREADS) {
            const int r = min(m0 + i / SS, M - 1), st = i % SS;
            float total = 0.0f;
            for (int c = 0; c < NC; ++c) total = __fadd_rn(total, pss[(static_cast<long long>(r) * NC + c) * SS + st]);
            rinv_s[i] = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(total, static_cast<float>(D)), eps)));
        }
    }

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int gfirst) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int gi = 0; gi < GPS; ++gi) {
            const int g = gfirst + gi;
            unsigned char* px = p + gi * T::XG;
            for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
                const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
                const int row = min(m0 + r, M - 1);
                cp16z(px + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8, m0 + r < M);
            }
            unsigned char* pw = p + T::X + gi * T::WG;
            for (int c = tid; c < BN; c += T::THREADS) {
                const int col = n0 + c;
                if (col < npad) cp16(pw + c * 16, w + (static_cast<size_t>(col / 64) * KG + g) * 64 + col % 64);
            }
            unsigned char* ps = p + T::X + T::W + gi * T::SG;
            for (int c = tid; c < BN; c += T::THREADS) {   // c < BN / 2: scale pairs, the rest bias pairs
                const int which = c / (BN / 2), off = (c % (BN / 2)) * 2, col = n0 + off;
                uint32_t* dst = reinterpret_cast<uint32_t*>(ps + which * T::S + off * 2);
                if (col < N) cp4(dst, (which ? bi : sc) + static_cast<size_t>(g) * N + col);
                else *dst = 0u;
            }
            if constexpr (NORM) {
                float* psc = reinterpret_cast<float*>(p + T::X + T::W + 2 * T::S + T::XS) + gi * GS;
                for (int c = tid; c < GS / 4; c += T::THREADS) cp16(psc + c * 4, scale + g * GS + c * 4);
            } else {
                float* pxs = reinterpret_cast<float*>(p + T::X + T::W + 2 * T::S) + gi * BM;
                for (int r = tid; r < BM; r += T::THREADS) cp4(pxs + r, xs + static_cast<size_t>(min(m0 + r, M - 1)) * KG + g);
            }
        }
    };

    float acc[T::MT][T::NT][4];
    float tot[FOLD ? T::MT : 1][FOLD ? T::NT : 1][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < stages_n) load(s, s * GPS);
        commit();
    }
    for (int it = 0; it < stages_n; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < stages_n) load(next % STAGES, next * GPS);
        commit();
        unsigned char* p = stage(it % STAGES);
        if constexpr (NORM) {
            // x = bf16(h * rinv[row, stream] * scale[k]) as fn_hc_down rounds it, each row's 32-group sum in order
            float* pxs = reinterpret_cast<float*>(p + T::X + T::W + 2 * T::S);
            for (int c = tid; c < GPS * BM; c += T::THREADS) {
                const int gi = c / BM, r = c % BM, k0 = (it * GPS + gi) * GS;
                unsigned char* px = p + gi * T::XG + r * T::ROW;
                const float rv = rinv_s[r * SS + k0 / D];
                float sum = 0.0f;
#pragma unroll
                for (int ch = 0; ch < T::CHUNKS; ++ch) {
                    uint4* cell = reinterpret_cast<uint4*>(px + swz<T::CHUNKS>(r, ch) * 16);
                    const uint4 raw = *cell;
                    const __nv_bfloat16* hv = reinterpret_cast<const __nv_bfloat16*>(&raw);
                    const float* psc = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S + T::XS) + gi * GS + ch * 8;
                    const float4 sa = *reinterpret_cast<const float4*>(psc);
                    const float4 sb = *reinterpret_cast<const float4*>(psc + 4);
                    const float sv[8] = {sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w};
                    __nv_bfloat16 o8[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        o8[i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(bf(hv[i]), rv), sv[i]));
                        sum = __fadd_rn(sum, bf(o8[i]));
                    }
                    *cell = *reinterpret_cast<const uint4*>(o8);
                }
                pxs[gi * BM + r] = sum;
            }
            __syncthreads();
        }
#pragma unroll
        for (int gi = 0; gi < GPS; ++gi) {
            const unsigned char* px = p + gi * T::XG;
            const uint4* pw = reinterpret_cast<const uint4*>(p + T::X + gi * T::WG);
            const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::W + gi * T::SG);
            const float* pxs = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S) + gi * BM;
            uint32_t a[2][T::MT][4];
#pragma unroll
            for (int kt = 0; kt < 2; ++kt)
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int ch = kt * 2 + (lane >> 4);
                    ldmatrix4(a[kt][i], px + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
                }
            float xi[T::MT][2];
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int row = wm * (BM / WM) + i * 16 + g8;
                xi[i][0] = pxs[row];
                xi[i][1] = pxs[row + 8];
            }
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const int cw = wn * (BN / WN) + j * 8;
                const uint4 words = pw[cw + g8];
                const uint32_t q0 = nib2(words.x, t4), q1 = nib2(words.y, t4), q2 = nib2(words.z, t4), q3 = nib2(words.w, t4);
                const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + cw + 2 * t4);
                const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(ps + T::S / 2 + cw + 2 * t4);
                const float s0 = __low2float(s2), s1 = __high2float(s2), b0 = __low2float(b2), b1 = __high2float(b2);
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    float dd[4];
                    mma0(dd, a[0][i], q0, q1);
                    mma(dd, a[1][i], q2, q3);
                    acc[i][j][0] = __fmaf_rn(xi[i][0], b0, __fmaf_rn(dd[0], s0, acc[i][j][0]));
                    acc[i][j][1] = __fmaf_rn(xi[i][0], b1, __fmaf_rn(dd[1], s1, acc[i][j][1]));
                    acc[i][j][2] = __fmaf_rn(xi[i][1], b0, __fmaf_rn(dd[2], s0, acc[i][j][2]));
                    acc[i][j][3] = __fmaf_rn(xi[i][1], b1, __fmaf_rn(dd[3], s1, acc[i][j][3]));
                }
            }
            if constexpr (FOLD) {
                const int gl = it * GPS + gi;
                if ((gl + 1) % cut == 0) {
#pragma unroll
                    for (int i = 0; i < T::MT; ++i)
#pragma unroll
                        for (int j = 0; j < T::NT; ++j)
#pragma unroll
                            for (int e = 0; e < 4; ++e) {
                                tot[i][j][e] = gl + 1 == cut ? acc[i][j][e] : __fadd_rn(tot[i][j][e], acc[i][j][e]);
                                acc[i][j][e] = 0.0f;
                            }
                }
            }
        }
    }
    wait<0>();
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + 2 * t4;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + g8 + h * 8;
                if (row >= M) continue;
                const float v0 = FOLD ? tot[i][j][2 * h] : acc[i][j][2 * h];
                const float v1 = FOLD ? tot[i][j][2 * h + 1] : acc[i][j][2 * h + 1];
                if constexpr (OUT16) {
                    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
                    if (col + 1 < N) *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                    else if (col < N) dst[0] = __float2bfloat16_rn(v0);
                } else {
                    float* dst = reinterpret_cast<float*>(out) + static_cast<size_t>(row) * N + col;
                    if (col < N) dst[0] = v0;
                    if (col + 1 < N) dst[1] = v1;
                }
            }
        }
}

}  // namespace hcf

// HC down: 32 x 192 tiles, the streams normed in shared memory, the 32 K slices folded in slice order (fp32 out).
template __global__ void hcf::hcf_kernel<32, 192, 2, 4, 4, 2, true, true, false, 2>(const __nv_bfloat16*, const float*,
    const uint4*, const __nv_bfloat16*, const __nv_bfloat16*, void*, int, int, int, int, int, int, const float*, const float*,
    float, int, int, int);
// HC up: 64 x 128 tiles, bf16 out (the up rows fn_hc_upmix mixes).
template __global__ void hcf::hcf_kernel<64, 128, 2, 4, 3, 2, false, false, true, 2>(const __nv_bfloat16*, const float*,
    const uint4*, const __nv_bfloat16*, const __nv_bfloat16*, void*, int, int, int, int, int, int, const float*, const float*,
    float, int, int, int);
