// NVFP4 experts in the checkpoint's own math: NVFP4 rows times NVFP4 weights on the block-scaled FP4 mma (sm_121a).

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "mma4.cuh"
#include "nvfp4q.cuh"
#include "common/qmm_frag.cuh"

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

// One 64-input step a warp reads: rows g and g + 8's codes and block scales, then each n8 tile's weights and scales.
template <int TJ>
struct Step {
    uint32_t a[4], sa;
    uint2 b[TJ];
    uint32_t s[TJ];
};

// A warp: a plan item's pairs (16 at most) by TJ n8 tiles, U steps read ahead; EPI 1 relu^2 to bf16, EPI 0 fp32 sums.
template <int EPI, int TJ, int U>
__device__ __forceinline__ void grouped(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ xs, int slots,
                                        const uint32_t* __restrict__ w, const uint8_t* __restrict__ ws,
                                        const float* __restrict__ alpha, int K, int N, const int* __restrict__ items,
                                        const int* __restrict__ counts, const int* __restrict__ members,
                                        void* __restrict__ out) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, NG = N / (8 * TJ), KG = K / 64;
    const int units = __ldg(counts) * NG;
    for (int unit = blockIdx.x * 4 + (threadIdx.x >> 5); unit < units; unit += gridDim.x * 4) {
        const int it = unit / NG, n0 = (unit - it * NG) * 8 * TJ;
        const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
        const bool v0 = g < cnt, v1 = g + 8 < cnt;
        const int p0 = v0 ? __ldg(members + first + g) : 0, p1 = v1 ? __ldg(members + first + g + 8) : 0;
        const int x0 = slots ? p0 / slots : p0, x1 = slots ? p1 / slots : p1;
        const uint8_t* c0 = codes + static_cast<size_t>(x0) * (K / 2);
        const uint8_t* c1 = codes + static_cast<size_t>(x1) * (K / 2);
        const bool hi = t & 1;  // lanes t 0 and 1 give rows g and g + 8's four block scales; t 2, 3 are not read
        const bool vr = hi ? v1 : v0;
        const uint8_t* sr = xs + static_cast<size_t>(hi ? x1 : x0) * (K / 16);
        const size_t tile = (static_cast<size_t>(e) * (N / 64) + n0 / 64) * KG;  // the 64-column tile holding n0
        const uint2* wb = reinterpret_cast<const uint2*>(w) + tile * 256 + n0 % 64 / 8 * 32 + lane;
        const uint32_t* sw = reinterpret_cast<const uint32_t*>(ws) + tile * 64 + n0 % 64 + g;
        auto fetch = [&](int kg, Step<TJ>& f) {  // rows past cnt: zeros
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
        Step<TJ> st[U];
#pragma unroll
        for (int u = 0; u < U - 1; ++u)
            if (u < KG) fetch(u, st[u]);
        for (int k0 = 0; k0 < KG; k0 += U) {  // a column's sum takes its steps in order whatever TJ and U are
            // step k0 + u's slot is u: indices known at compile time keep st in registers
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
        const float al = __ldg(alpha + e);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            if (!(h ? v1 : v0)) continue;
            const size_t row = static_cast<size_t>(h ? p1 : p0) * N;
#pragma unroll
            for (int j = 0; j < TJ; ++j) {
                const int col = n0 + j * 8 + 2 * t;
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

#define TF_CK4(NAME, EPI, TJ, U)                                                                                       \
    extern "C" __global__ void __launch_bounds__(128)                                                                  \
        NAME(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ xs, int slots,                            \
             const uint32_t* __restrict__ w, const uint8_t* __restrict__ ws, const float* __restrict__ alpha, int K,  \
             int N, const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,   \
             void* __restrict__ out) {                                                                                 \
        grouped<EPI, TJ, U>(codes, xs, slots, w, ws, alpha, K, N, items, counts, members, out);                       \
    }
// Under 128 token rows: 64 columns a warp, a step at a time; decode-sized: 32 a warp, 4 steps in flight. Same bits.
TF_CK4(tf_ck4_up_relu2, 1, 8, 1)
TF_CK4(tf_ck4_down_f32, 0, 8, 1)
TF_CK4(tf_ck4_up_relu2_few, 1, 4, 4)
TF_CK4(tf_ck4_down_f32_few, 0, 4, 4)

namespace tf_ck4 {

constexpr int PM = 128, PN = 128, PSTAGES = 4;  // a prompt CTA's rows (one plan item), columns, staged K steps

// One staged 64-input step: the item's rows (chunks swizzled for ldmatrix), two 64-column weight tiles, their scales.
struct PromptStage {
    uint4 a[PM][2];  // 32-byte rows
    uint32_t sa[PM];
    uint2 b[2][256];  // as packed: fragment order
    uint32_t sb[2][64];
};

// fp32 sums; relu^2 rows quantized as tf_ck_quant4 does; or an item's shares summed as tf_super_combine sums them.
enum PromptEpi : int { F32 = 0, RELU2_NVFP4 = 1, SHARE_SUM = 2 };

}  // namespace tf_ck4

// A block: a plan item (up to 128 pairs of one expert) by 128 columns; each output keeps the decode warp's MMA chain.
template <int EPI>
__device__ __forceinline__ void grouped_prompt(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ xs,
                                               int slots, const uint32_t* __restrict__ w,
                                               const uint8_t* __restrict__ ws, const float* __restrict__ alpha, int K,
                                               int N, const int* __restrict__ items, const int* __restrict__ counts,
                                               const int* __restrict__ members, void* __restrict__ out,
                                               uint8_t* __restrict__ out_scales, float g_out, int shares, int experts) {
    using namespace tf_ck4;
    __shared__ __align__(128) PromptStage st[PSTAGES];
    __shared__ __align__(16) uint2 row_codes[PM][PN / 16];  // RELU2_NVFP4: the tile's rows, written out whole
    __shared__ uint8_t row_scales[PM][PN / 16];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int wm = warp >> 2, wn = warp & 3;
    const int NB = N / 64, NT = (NB + 1) / 2, KG = K / 64, steps = shares * KG, units = __ldg(counts) / shares * NT;
    const size_t share_w = static_cast<size_t>(experts) * NB * KG;  // share s: pair + s, expert s * experts + e
    // loader roles: A row tid / 2 (16-byte half tid % 2) and its scales; weight tile tid / 128, 16 bytes tid % 128
    const int arow = tid >> 1, half = tid & 1, bt = tid >> 7, bq = tid & 127;
    // ldmatrix rows: lane l gives row (l % 8) + 8 ((l / 8) % 2) of chunk l / 16
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lchunk = lane >> 4;
    // the staged unit: its item and the sources its loads read
    int u_e = 0, u_first = 0, u_cnt = 0, u_ct = 0;
    const uint8_t *a_src = nullptr, *sa_src = nullptr;
    const uint4 *b_src = nullptr, *sb_src = nullptr;
    auto load = [&](int stage, int step) {
        const int s = step / KG, kg = step - s * KG;
        PromptStage& p = st[stage];
        qmm_frag::cp16(&p.a[arow][mma4::chunk<mma4::A4>(arow, half)], a_src + s * (K / 2) + kg * 32);
        if (!half) qmm_frag::cp4(&p.sa[arow], sa_src + s * (K / 16) + kg * 4);
        qmm_frag::cp16(reinterpret_cast<uint4*>(p.b[bt]) + bq, b_src + (s * share_w + kg) * 128);
        if (bq < 16) qmm_frag::cp16(reinterpret_cast<uint4*>(p.sb[bt]) + bq, sb_src + (s * share_w + kg) * 16);
    };
    // a unit's item and sources, and its first steps in flight (always PSTAGES - 1 commit groups)
    auto stage_unit = [&](int unit) {
        const int it = unit / NT;
        u_ct = unit % NT;
        u_e = __ldg(items + 3 * it), u_first = __ldg(items + 3 * it + 1), u_cnt = __ldg(items + 3 * it + 2);
        const int pair = arow < u_cnt ? __ldg(members + u_first + arow) : 0;  // rows past the item read row 0, unused
        const size_t xr = slots ? pair / slots : pair;
        a_src = codes + xr * (K / 2) + half * 16;
        sa_src = xs + xr * (K / 16);
        const size_t tile = static_cast<size_t>(u_e) * NB + min(u_ct * 2 + bt, NB - 1);  // past N: the last, unread
        b_src = reinterpret_cast<const uint4*>(w) + tile * KG * 128 + bq;
        sb_src = reinterpret_cast<const uint4*>(ws) + tile * KG * 16 + bq;
#pragma unroll
        for (int s = 0; s < PSTAGES - 1; ++s) {
            if (s < steps) load(s, s);
            qmm_frag::commit();
        }
    };
    if (blockIdx.x < units) stage_unit(blockIdx.x);
    for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
        const int e = u_e, first = u_first, cnt = u_cnt, ct = u_ct;
        const bool cols = ct * PN + wn * 32 < N;  // a warp's 32 columns are all in or all past N
        float acc[4][4][4] = {}, sum[4][4][4];
        auto fold = [&](int s) {  // share s's sums times its expert's alpha, added in share order
            const float al = __ldg(alpha + s * experts + e);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const float v = acc[i][j][c] * al;
                        sum[i][j][c] = s ? __fadd_rn(sum[i][j][c], v) : v;
                        acc[i][j][c] = 0.0f;
                    }
        };
        for (int step = 0; step < steps; ++step) {
            qmm_frag::wait<PSTAGES - 2>();
            __syncthreads();
            if (step + PSTAGES - 1 < steps) load((step + PSTAGES - 1) % PSTAGES, step + PSTAGES - 1);
            qmm_frag::commit();
            if (EPI == SHARE_SUM && step % KG == 0 && step > 0) fold(step / KG - 1);
            if (!cols) continue;
            const PromptStage& p = st[step % PSTAGES];
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
        fold(shares - 1);
        __syncthreads();  // every warp is past the last stage: the next unit's first steps load under this epilogue
        if (unit + gridDim.x < units) stage_unit(unit + gridDim.x);
        if constexpr (EPI == RELU2_NVFP4) {
            if (cols) {  // tf_ck_quant4's blocks of 16 inputs: two n8 tiles across a quad, four values a lane
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int bk = 0; bk < 2; ++bk) {
                            float f[4], amax = 0.0f;
#pragma unroll
                            for (int c = 0; c < 4; ++c) {
                                const float u = fmaxf(tf_ck4::bf(sum[i][2 * bk + c / 2][2 * h + c % 2]), 0.0f);
                                f[c] = tf_ck4::bf(u * u);
                                amax = fmaxf(amax, fabsf(f[c]));
                            }
                            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
                            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
                            const nvfp4q::Scale sc = nvfp4q::block_scale(amax, g_out);
                            // lane t holds inputs 2t, 2t + 1 of each 8: byte t of the block's two code words
                            uint32_t lo = (nvfp4q::e2m1(f[0] * sc.mul) | nvfp4q::e2m1(f[1] * sc.mul) << 4) << (8 * t);
                            uint32_t hi = (nvfp4q::e2m1(f[2] * sc.mul) | nvfp4q::e2m1(f[3] * sc.mul) << 4) << (8 * t);
#pragma unroll
                            for (int o = 1; o < 4; o <<= 1) {
                                lo |= __shfl_xor_sync(0xffffffffu, lo, o);
                                hi |= __shfl_xor_sync(0xffffffffu, hi, o);
                            }
                            const int r = wm * 64 + i * 16 + g + h * 8;
                            if (t == 0) {
                                row_codes[r][wn * 2 + bk] = make_uint2(lo, hi);
                                row_scales[r][wn * 2 + bk] = static_cast<uint8_t>(sc.sf8);
                            }
                        }
            }
            __syncthreads();  // the staged rows whole (their next writes wait on the next unit's step barriers)
            const int r = tid >> 1, part = tid & 1;  // two threads a row: 32 code bytes each, the odd one the scales
            if (r >= cnt) continue;
            const size_t row = __ldg(members + first + r);
            uint4* to = reinterpret_cast<uint4*>(static_cast<uint8_t*>(out) + row * (N / 2) + ct * (PN / 2));
            const uint4* from = reinterpret_cast<const uint4*>(row_codes[r]);
            const int chunks = min(PN / 32, (N - ct * PN) / 32);  // 16-byte chunks of the row inside N
#pragma unroll
            for (int c = 2 * part; c < 2 * part + 2; ++c)
                if (c < chunks) to[c] = from[c];
            if (part)
                for (int b = 0; b < min(PN, N - ct * PN) / 16; ++b)
                    out_scales[row * (N / 16) + ct * (PN / 16) + b] = row_scales[r][b];
        } else {
            if (!cols) continue;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = wm * 64 + i * 16 + g + h * 8;
                    if (r >= cnt) continue;
                    const size_t dst = static_cast<size_t>(__ldg(members + first + r)) / shares * N;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int col = ct * PN + wn * 32 + j * 8 + 2 * t;
                        const float v0 = sum[i][j][2 * h], v1 = sum[i][j][2 * h + 1];
                        if constexpr (EPI == F32)
                            *reinterpret_cast<float2*>(static_cast<float*>(out) + dst + col) = make_float2(v0, v1);
                        else
                            *reinterpret_cast<__nv_bfloat162*>(static_cast<__nv_bfloat16*>(out) + dst + col) =
                                __floats2bfloat162_rn(v0, v1);
                    }
                }
            }
        }
    }
}

// Prompt up: relu^2 rows as tf_ck_quant4's NVFP4 rows under global scale g (codes at out, e4m3 scales at out_scales).
extern "C" __global__ void __launch_bounds__(256)
    tf_ck4_up_nvfp4_prompt(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ xs, int slots,
                           const uint32_t* __restrict__ w, const uint8_t* __restrict__ ws,
                           const float* __restrict__ alpha, int K, int N, const int* __restrict__ items,
                           const int* __restrict__ counts, const int* __restrict__ members, uint8_t* __restrict__ out,
                           uint8_t* __restrict__ out_scales, float g) {
    grouped_prompt<tf_ck4::RELU2_NVFP4>(codes, xs, slots, w, ws, alpha, K, N, items, counts, members, out, out_scales,
                                        g, 1, 0);
}

extern "C" __global__ void __launch_bounds__(256)
    tf_ck4_down_f32_prompt(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ xs, int slots,
                           const uint32_t* __restrict__ w, const uint8_t* __restrict__ ws,
                           const float* __restrict__ alpha, int K, int N, const int* __restrict__ items,
                           const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out) {
    grouped_prompt<tf_ck4::F32>(codes, xs, slots, w, ws, alpha, K, N, items, counts, members, out, nullptr, 0.0f, 1,
                                0);
}

// Prompt down over `shares` x `experts` share experts: each first-share item's pairs summed over shares, bf16 a row.
extern "C" __global__ void __launch_bounds__(256)
    tf_ck4_down_sum_prompt(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ xs, int slots,
                           const uint32_t* __restrict__ w, const uint8_t* __restrict__ ws,
                           const float* __restrict__ alpha, int K, int N, const int* __restrict__ items,
                           const int* __restrict__ counts, const int* __restrict__ members,
                           __nv_bfloat16* __restrict__ out, int shares, int experts) {
    grouped_prompt<tf_ck4::SHARE_SUM>(codes, xs, slots, w, ws, alpha, K, N, items, counts, members, out, nullptr, 0.0f,
                                      shares, experts);
}
