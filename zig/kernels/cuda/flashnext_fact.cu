// Prompt GEMM on #952's factored arithmetic (qmm_kernel's bits): 2 groups a stage, mma per group then fp32 fmas.
#include "qmm_frag.cuh"

namespace fact {
using namespace qmm_frag;

template <int BM, int BN, int WM, int WN, int STAGES, int GPS>
struct Tile {
    static constexpr int GS = 32;
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;
    static constexpr int NT = BN / WN / 8;
    static constexpr int ROW = GS * 2;
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int XG = BM * ROW;
    static constexpr int WG = BN * GS / 2;
    static constexpr int SG = BN * 2;
    static constexpr int X = GPS * XG;
    static constexpr int W = GPS * WG;
    static constexpr int S = GPS * SG;
    static constexpr int XS = GPS * BM * 4;
    static constexpr int STAGE = X + W + 2 * S + XS;
    static constexpr int SMEM = STAGES * STAGE;
};

template <int BM, int BN, int WM, int WN, int STAGES, int GPS, bool DBUF, int MINB, bool FOLD = false, bool OUT16 = false, bool BLOCKED = false>
__global__ void __launch_bounds__(WM * WN * 32, MINB) fact_kernel(
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const uint32_t* __restrict__ w,
        const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases,
        void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad, int ldx, int group) {
    using T = Tile<BM, BN, WM, WN, STAGES, GPS>;
    constexpr int GS = T::GS;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, cut = KG / SK, per = FOLD ? KG : cut, slice = FOLD ? 0 : blockIdx.z, g0 = slice * per;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y;
    const int stages_n = per / GPS;

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
                cp16z(px + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8,
                      m0 + r < M);
            }
            unsigned char* pw = p + T::X + gi * T::WG;
            constexpr int TILE_BYTES = 64 * GS / 2;
            for (int c = tid; c < T::WG / 16; c += T::THREADS) {
                const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
                const size_t tile = static_cast<size_t>(n0 / 64 + t) * KG + g;
                if (n0 + t * 64 < npad) cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + tile * TILE_BYTES + off * 16);
            }
            unsigned char* ps = p + T::X + T::W + gi * T::SG;
            for (int c = tid; c < 2 * (T::SG / 16); c += T::THREADS) {
                const int which = c / (T::SG / 16), off = c % (T::SG / 16);
                const __nv_bfloat16* src = (which ? biases : scales) + static_cast<size_t>(g) * npad + n0 + off * 8;
                if (n0 + off * 8 < npad) cp16(ps + which * T::S + off * 16, src);
            }
            float* pxs = reinterpret_cast<float*>(p + T::X + T::W + 2 * T::S) + gi * BM;
            for (int r = tid; r < BM; r += T::THREADS) cp4(pxs + r, xs + static_cast<size_t>(min(m0 + r, M - 1)) * KG + g);
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
    float d[2][T::MT][T::NT][4], sv[2][T::NT][2], bv[2][T::NT][2], xv[2][T::MT][2];

    auto epilogue = [&](int b) {
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int i = 0; i < T::MT; ++i)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[i][j][e] = __fmaf_rn(xv[b][i][e >> 1], bv[b][j][e & 1], __fmaf_rn(d[b][i][j][e], sv[b][j][e & 1], acc[i][j][e]));
    };
    // a slice ends after group gl (0-based, in this CTA): fold its total in slice order and restart from zero
    auto slice_end = [&](int gl) {
        if constexpr (FOLD) {
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
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < stages_n) load(s, g0 + s * GPS);
        commit();
    }
    for (int it = 0; it < stages_n; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < stages_n) load(next % STAGES, g0 + next * GPS);
        commit();
        const unsigned char* p = stage(it % STAGES);
#pragma unroll
        for (int gi = 0; gi < GPS; ++gi) {
            constexpr int dummy = 0;
            (void)dummy;
            const int b = gi & 1;
            const unsigned char* px = p + gi * T::XG;
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X + gi * T::WG);
            const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::W + gi * T::SG);
            const float* pxs = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S) + gi * BM;
            uint32_t words[T::NT];
#pragma unroll
            for (int j = 0; j < T::NT; ++j) words[j] = pw[(wn * T::NT + j) * 32 + lane];
            if constexpr (BLOCKED) {
                // each (i, j) block's two k16 steps, then its epilogue: d lives a few registers, the bits are the same
                uint32_t a[2][T::MT][4];
#pragma unroll
                for (int kt = 0; kt < 2; ++kt)
#pragma unroll
                    for (int i = 0; i < T::MT; ++i) {
                        const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                        const int ch = kt * 2 + (lane >> 4);
                        ldmatrix4(a[kt][i], px + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
                    }
                float sj[T::NT][2], bj[T::NT][2], xi[T::MT][2];
#pragma unroll
                for (int j = 0; j < T::NT; ++j) {
                    const int col = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
                    const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + col);
                    const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(ps + T::S / 2 + col);
                    sj[j][0] = __low2float(s2); sj[j][1] = __high2float(s2);
                    bj[j][0] = __low2float(b2); bj[j][1] = __high2float(b2);
                }
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    const int row = wm * (BM / WM) + i * 16 + (lane >> 2);
                    xi[i][0] = pxs[row];
                    xi[i][1] = pxs[row + 8];
                }
#pragma unroll
                for (int j = 0; j < T::NT; ++j) {
                    const uint32_t q0 = pair(words[j], 0), q1 = pair(words[j], 4), q2 = pair(words[j], 8), q3 = pair(words[j], 12);
#pragma unroll
                    for (int i = 0; i < T::MT; ++i) {
                        float dd[4];
                        mma0(dd, a[0][i], q0, q1);
                        mma(dd, a[1][i], q2, q3);
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            acc[i][j][e] = __fmaf_rn(xi[i][e >> 1], bj[j][e & 1], __fmaf_rn(dd[e], sj[j][e & 1], acc[i][j][e]));
                    }
                }
                slice_end(it * GPS + gi);
                continue;
            }
#pragma unroll
            for (int kt = 0; kt < GS / 16; ++kt) {
                uint32_t a[T::MT][4];
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int ch = kt * 2 + (lane >> 4);
                    ldmatrix4(a[i], px + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
                }
#pragma unroll
                for (int j = 0; j < T::NT; ++j) {
                    const uint32_t b0 = pair(words[j], (kt & 1) * 8);
                    const uint32_t b1 = pair(words[j], (kt & 1) * 8 + 4);
#pragma unroll
                    for (int i = 0; i < T::MT; ++i) {
                        if (kt == 0) mma0(d[b][i][j], a[i], b0, b1);
                        else mma(d[b][i][j], a[i], b0, b1);
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const int col = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
                const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + col);
                const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(ps + T::S / 2 + col);
                sv[b][j][0] = __low2float(s2); sv[b][j][1] = __high2float(s2);
                bv[b][j][0] = __low2float(b2); bv[b][j][1] = __high2float(b2);
            }
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int row = wm * (BM / WM) + i * 16 + (lane >> 2);
                xv[b][i][0] = pxs[row];
                xv[b][i][1] = pxs[row + 8];
            }
            const int gl = it * GPS + gi;
            if constexpr (DBUF) {
                if (gl > 0) {
                    epilogue(b ^ 1);
                    slice_end(gl - 1);
                }
            } else {
                epilogue(b);
                slice_end(gl);
            }
        }
    }
    if constexpr (DBUF && !BLOCKED) {
        if (stages_n > 0) {
            epilogue((GPS - 1) & 1);
            slice_end(stages_n * GPS - 1);
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
                if (row >= M) continue;
                const float v0 = FOLD ? tot[i][j][2 * h] : acc[i][j][2 * h];
                const float v1 = FOLD ? tot[i][j][2 * h + 1] : acc[i][j][2 * h + 1];
                if (OUT16 && !FOLD && SK == 1) {
                    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
                    if (col + 1 < N && (N & 1) == 0)
                        *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                    else {
                        if (col < N) dst[0] = __float2bfloat16_rn(v0);
                        if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                    }
                    continue;
                }
                float* dst = (SK > 1 && !FOLD) ? part + (static_cast<size_t>(slice) * M + row) * N + col
                                                : (FOLD ? part : reinterpret_cast<float*>(out)) + static_cast<size_t>(row) * N + col;
                if (col < N) dst[0] = v0;
                if (col + 1 < N) dst[1] = v1;
            }
        }
}

}  // namespace fact

// The prompt form the batched pass launches from 64 rows: 64 x 128 on 2 x 4 warps, bf16 out (one slice) or fp32 slices.
template __global__ void fact::fact_kernel<64, 128, 2, 4, 3, 2, false, 2, false, true, false>(const __nv_bfloat16*,
    const float*, const uint32_t*, const __nv_bfloat16*, const __nv_bfloat16*, void*, float*, int, int, int, int, int, int, int);
// The folded form from 64 rows: every K slice in the CTA, their totals added in slice order (one fp32 slice).
template __global__ void fact::fact_kernel<64, 128, 2, 4, 3, 2, false, 2, true, true, true>(const __nv_bfloat16*,
    const float*, const uint32_t*, const __nv_bfloat16*, const __nv_bfloat16*, void*, float*, int, int, int, int, int, int, int);
