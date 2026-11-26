// Hy4-flash iHC boundary kernels (`Args`, helpers, and 1-launch ticket `Arm A`; see `hy4_ihc_coop_prefill.cuh`).
#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "../common/cp_async.cuh"
#include "../common/glue_math.cuh"
#include "../common/reduce.cuh"
#include "../common/vec.cuh"

namespace hy4ihc {

typedef __nv_bfloat16 bf16;

// IHC_TRACE (debug builds only): thread 0 of each CTA stamps %globaltimer at the phase edges.
#ifdef IHC_TRACE
__device__ unsigned long long g_ihc_trace[128 * 16];
__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long v;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
    return v;
}
#define IHC_TR(i) \
    if (threadIdx.x == 0) g_ihc_trace[(blockIdx.y * gridDim.x + blockIdx.x) * 16 + (i)] = gtimer()
#else
#define IHC_TR(i)
#endif

constexpr int H = 3072;        // hidden
constexpr int NS = 4;          // iHC streams
constexpr int NSH = NS * H;    // flat width 12,288
constexpr int NT = 256;        // threads a CTA
constexpr int NCH = 24;        // hy4_hc_mix chunks of 512 flat inputs
constexpr int NU = 48;         // warp-units (cj, w)
constexpr int KV = 9;          // partial values a (row, chunk, warp): mixes 0..7, sum of squares at 8
constexpr int MAXR = 16;       // decode rows (fused kernels)
constexpr int MAXRPG = 4;      // rows a row group (ticket arm)
constexpr int MAXUPC = 4;      // units a CTA (ticket arm)

// One launch's arguments (passed by value as a __grid_constant__ struct; the Zig side mirrors this layout).
struct Args {
    const float* G;          // gathered sublayer partials [world][R][H] fp32 (rank stride gs floats)
    long long gs;            // rank stride in floats (R * H for a dense gather)
    int world;               // 1, 2, 4 ...
    int R;                   // rows
    bf16* X;                 // streams [R][4][H] bf16, updated in place (POST)
    const float* post_prev;  // [R][4] post coefficients of the sublayer just run (POST)
    const bf16* post_nw;     // sandwich norm weight [H] (post_proj_norm / combine_output_norm / dense norm)
    const float* fn;         // next site's fn [M][12288] fp32
    const float* base;       // [M]
    const float* scale;      // [2] (head: [1])
    const bf16* next_nw;     // next sublayer's input norm weight [H] (head: model.norm)
    bf16* A;                 // next sublayer input [R][H] bf16
    float* post_next;        // [R][4] next sublayer's post coefficients (M == 8); may alias post_prev
    float* wp;               // scratch [R][24][8][9] warp partials of the mixes
    int* cnt;                // ticket counters [16] (zero at rest; self-cleaning)
    float* ssy;              // coop scratch [R][8]
    float* sso;              // coop scratch [R][8]
    float* yb;               // coop scratch [R][H] fp32 (post-MLP input when round_in == 0)
    bf16* ob;                // coop scratch [R][H]
    float* dbg_tot;          // optional [R][9]: folded mixes and sum of squares; null in production
    int rpg;                 // ticket arm: rows per row group (grid.y = ceil(R / rpg)); grid.x = 48 / upc
    float eps, hc_eps, mag;
    int round_in;            // 1: round summed G to bf16 before sandwich RMSNorm (MLA); 0: keep fp32 (MLP)
    int pad;
};

using tf_common::pack8;
using tf_common::unpack8;
using tf_common::warp_sum;
using tf_glue::rbf;
using tf_glue::tf_sigmoid;

__device__ __forceinline__ bf16 tb(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ float bfv(bf16 v) { return __bfloat162float(v); }

__device__ __forceinline__ float rinv_of(float ss, int n, float eps) {  // exact IEEE 1 / sqrt(ss / n + eps)
    return tf_common::rms_scale(ss, n, eps);
}
__device__ __forceinline__ float pre_coef(float tot, float rinv, float scale0, float base, float hc_eps) {
    const float mix = __fmul_rn(tot, rinv);
    return __fadd_rn(tf_sigmoid(__fadd_rn(__fmul_rn(mix, scale0), base)), hc_eps);
}
__device__ __forceinline__ float post_coef(float tot, float rinv, float scale1, float base, float mag, float hc_eps) {
    const float mix = __fmul_rn(tot, rinv);
    return __fadd_rn(__fmul_rn(mag, tf_sigmoid(__fadd_rn(__fmul_rn(mix, scale1), base))), hc_eps);
}

using tf_common::cp16;
using tf_common::cp16z;
using tf_common::cp4;
using tf_common::cp_commit;
using tf_common::cp_wait;
__device__ __forceinline__ uint4 ldcg4(const void* p) { return __ldcg(reinterpret_cast<const uint4*>(p)); }

// Warp reduce-scatter: `K = 2^LOGK` values per lane, returning the xor-butterfly warp sum in `lane >> (5 - LOGK)`.
template <int LOGK>
__device__ __forceinline__ float rs_butterfly(float (&v)[1 << LOGK]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int lv = 0; lv < LOGK; ++lv) {
        const int half = (1 << LOGK) >> (lv + 1);
        const int o = 16 >> lv;
        const bool hi = (lane & o) != 0;
#pragma unroll
        for (int i = 0; i < half; ++i) {
            const float send = hi ? v[i] : v[i + half];
            const float keep = hi ? v[i + half] : v[i];
            v[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
    }
    float x = v[0];
#pragma unroll
    for (int o = 16 >> LOGK; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

// The mix values of one (row, stream s, unit) task for lane l: v[m] (m < M), v[8] = sum of squares; others 0.
template <int M>
__device__ __forceinline__ void mix_chain(const float* fv, float x0, float x1,
                                          float (&v)[16]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int m = 0; m < 8; ++m)
        v[m] = (m < M) ? __fmaf_rn(fv[m * 256 + 32 + lane], x1, __fmaf_rn(fv[m * 256 + lane], x0, 0.f)) : 0.f;
    v[8] = __fmaf_rn(x1, x1, __fmaf_rn(x0, x0, 0.f));
#pragma unroll
    for (int k = 9; k < 16; ++k) v[k] = 0.f;
}

// Mix warp partials `wp[R][9][24][8]`: 8 warp partials per `(row, chunk)` sit in 32 contiguous bytes.
__device__ __forceinline__ size_t wp_idx(int r, int k, int j, int w) {
    return ((static_cast<size_t>(r) * KV + k) * NCH + j) * 8 + w;
}
__device__ __forceinline__ float chunk_sum(const float4 p, const float4 q) {
    float c = 0.f;
    c = __fadd_rn(c, p.x);
    c = __fadd_rn(c, p.y);
    c = __fadd_rn(c, p.z);
    c = __fadd_rn(c, p.w);
    c = __fadd_rn(c, q.x);
    c = __fadd_rn(c, q.y);
    c = __fadd_rn(c, q.z);
    return __fadd_rn(c, q.w);
}
// pre (is_post false) or post (true): the hy4_hc_apply expression; mag * x is skipped for pre (x * 1 == x)
__device__ __forceinline__ float hc_coef(float tot, float rinv, float sc, float bs, bool is_post, float mag, float hc_eps) {
    const float sg = tf_sigmoid(__fadd_rn(__fmul_rn(__fmul_rn(tot, rinv), sc), bs));
    return __fadd_rn(is_post ? __fmul_rn(mag, sg) : sg, hc_eps);
}

// One warp executes two `(row, stream, unit)` mix tasks via interleaved reduce-scatters.
template <int M>
__device__ __forceinline__ void mix_pair(const float* fv0, const float* xv0, float* dst0, bool has1, const float* fv1,
                                         const float* xv1, float* dst1) {
    const int lane = threadIdx.x & 31;
    float v0[16], v1[16];
    mix_chain<M>(fv0, xv0[lane], xv0[32 + lane], v0);
    if (has1) mix_chain<M>(fv1, xv1[lane], xv1[32 + lane], v1);
    else
#pragma unroll
        for (int k = 0; k < 16; ++k) v1[k] = 0.f;
    const float x0 = rs_butterfly<4>(v0);  // lane holds value lane >> 1
    const float x1 = rs_butterfly<4>(v1);
    const int k = lane >> 1;
    if (!(lane & 1) && (k < M || k == 8)) {
        dst0[static_cast<size_t>(k) * NCH * 8] = x0;
        if (has1) dst1[static_cast<size_t>(k) * NCH * 8] = x1;
    }
}

// Arm A: 1-launch ticket kernel (`grid (48 / upc, ceil(R / rpg))`, self-cleaning atomic ticket per row group).
__host__ __device__ constexpr int fin_os_offset() { return 3840; }  // bytes: csum | tots | coef | red2 | rinvo
__host__ __device__ constexpr int ticket_pre_bytes(int M) { return M > 0 ? H * 2 + 64 : 0; }  // next_nw | base | scale
__host__ __device__ inline int ticket_main_bytes(int upc, int rpg, int M) {
    return (upc * M * 256 + rpg * 4 * upc * 64 + rpg * upc * 64) * 4 + rpg * 4 * upc * 64 * 2 + upc * 64 * 2 + rpg * 16;
}
__host__ __device__ inline int ticket_smem_bytes(int upc, int rpg, int M) {
    const int main_b = ticket_main_bytes(upc, rpg, M);
    const int fin_b = M > 0 ? fin_os_offset() + rpg * H * 2 : 0;
    return ticket_pre_bytes(M) + (main_b > fin_b ? main_b : fin_b);
}

// o = rbf(sum_s cf[s] X'[s]) of 8 columns (hy4_hc_apply's sum, stream order from 0), stored as bf16.
__device__ __forceinline__ void collapse8(const float* cf, const uint4 (&xv)[4], bf16* dst) {
    float x[4][8];
#pragma unroll
    for (int s = 0; s < 4; ++s) unpack8(xv[s], x[s]);
    float o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        float v = 0.f;
#pragma unroll
        for (int s = 0; s < 4; ++s) v = __fadd_rn(v, __fmul_rn(cf[s], x[s][k]));
        o[k] = v;
    }
    *reinterpret_cast<uint4*>(dst) = pack8(o);
}

template <bool POST, int M>
__device__ __forceinline__ void ticket_body(const Args& a) {
    extern __shared__ __align__(16) unsigned char dsm[];
    __shared__ float red[MAXRPG * 8];
    __shared__ float rinv_s[MAXRPG];
    __shared__ int last;
    const int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    const int upc = NU / gridDim.x;
    const int u0 = blockIdx.x * upc;
    const int rpg = a.rpg;
    const int rbeg = blockIdx.y * rpg;
    const int nr = min(rpg, a.R - rbeg);
    if (nr <= 0) return;  // never with a well-formed grid
    IHC_TR(0);
    bf16* nnw_s = reinterpret_cast<bf16*>(dsm);                 // [H]            (M > 0)
    float* bs_s = reinterpret_cast<float*>(dsm + H * 2);        // base [8] | scale [2]
    unsigned char* mainp = dsm + ticket_pre_bytes(M);
    float* fs = reinterpret_cast<float*>(mainp);                // [upc][M][4][2][32]
    float* xs = fs + upc * M * 256;                             // [rpg][4][upc][2][32]
    float* ys = xs + rpg * 4 * upc * 64;                        // [rpg][upc][2][32]
    bf16* xpre = reinterpret_cast<bf16*>(ys + rpg * upc * 64);  // [rpg][4][upc][2][32]
    bf16* pnw_s = xpre + rpg * 4 * upc * 64;                    // [upc][2][32]
    float* pp_s = reinterpret_cast<float*>(pnw_s + upc * 64);   // [rpg][4]

    // ---- entry: every load that does not depend on another, in one cp.async group ----
    if constexpr (M > 0) {
        const int nv = upc * M * 4 * 2 * 8;
        for (int i = t; i < nv; i += NT) {
            const int q = i & 7, h = (i >> 3) & 1, s = (i >> 4) & 3;
            const int rest = i >> 6, m = rest % M, ul = rest / M;
            const int u = u0 + ul, cj = u >> 3, w = u & 7;
            cp16(fs + ((((ul * M + m) * 4 + s) * 2 + h) * 32) + q * 4,
                 a.fn + static_cast<size_t>(m) * NSH + s * H + cj * 512 + h * 256 + w * 32 + q * 4);
        }
        for (int i = t; i < H / 8; i += NT) cp16(nnw_s + i * 8, a.next_nw + i * 8);
        if (t < M) cp4(bs_s + t, a.base + t);
        else if (t >= 32 && t < 32 + (M == 8 ? 2 : 1)) cp4(bs_s + 8 + (t - 32), a.scale + (t - 32));
    }
    {
        const int nx = nr * 4 * upc * 8;  // 16-byte pieces (r, s, ul, h, quad)
        for (int i = t; i < nx; i += NT) {
            const int quad = i & 3, h = (i >> 2) & 1;
            const int rest = i >> 3, ul = rest % upc, rs = rest / upc, s = rs & 3, r = rs >> 2;
            const int u = u0 + ul, cj = u >> 3, w = u & 7;
            cp16(xpre + (((r * 4 + s) * upc + ul) * 2 + h) * 32 + quad * 8,
                 a.X + (static_cast<size_t>(rbeg + r) * NS + s) * H + cj * 512 + h * 256 + w * 32 + quad * 8);
        }
    }
    if constexpr (POST) {
        for (int i = t; i < upc * 8; i += NT) {
            const int quad = i & 3, h = (i >> 2) & 1, ul = i >> 3;
            const int u = u0 + ul, cj = u >> 3, w = u & 7;
            cp16(pnw_s + (ul * 2 + h) * 32 + quad * 8, a.post_nw + cj * 512 + h * 256 + w * 32 + quad * 8);
        }
        if (t < nr) cp16(pp_s + t * 4, a.post_prev + static_cast<size_t>(rbeg + t) * 4);
    }
    cp_commit();

    if constexpr (POST) {
        // Phase 1: `y` and `sum(y^2)` (`c = t + 256*j`) in rank order across all rows of the group.
        float y[MAXRPG][12];
#pragma unroll
        for (int r = 0; r < MAXRPG; ++r)
#pragma unroll
            for (int j = 0; j < 12; ++j)
                y[r][j] = r < nr ? __ldg(a.G + static_cast<size_t>(rbeg + r) * H + t + 256 * j) : 0.f;
        for (int k = 1; k < a.world; ++k) {
            const float* gk = a.G + static_cast<size_t>(k) * a.gs + static_cast<size_t>(rbeg) * H + t;
#pragma unroll
            for (int r = 0; r < MAXRPG; ++r)
                if (r < nr)
#pragma unroll
                    for (int j = 0; j < 12; ++j) y[r][j] = __fadd_rn(y[r][j], __ldg(gk + r * H + 256 * j));
        }
        float ss[MAXRPG];
#pragma unroll
        for (int r = 0; r < MAXRPG; ++r) {
            ss[r] = 0.f;
#pragma unroll
            for (int j = 0; j < 12; ++j) {
                const float yv = a.round_in ? rbf(y[r][j]) : y[r][j];
                ss[r] = __fmaf_rn(yv, yv, ss[r]);
                const int ul = (j >> 1) * 8 + wid - u0;  // column t + 256 j belongs to unit (j / 2, wid)
                if (r < nr && ul >= 0 && ul < upc) ys[((r * upc + ul) * 2 + (j & 1)) * 32 + lane] = yv;
            }
        }
        const float x = rs_butterfly<2>(ss);  // lane holds row lane >> 3
        if ((lane & 7) == 0) red[(lane >> 3) * 8 + wid] = x;
        IHC_TR(1);
        cp_wait<0>();
        __syncthreads();
        IHC_TR(2);
        if (t < nr) {
            float tot = 0.f;
#pragma unroll
            for (int w = 0; w < 8; ++w) tot = __fadd_rn(tot, red[t * 8 + w]);
            rinv_s[t] = rinv_of(tot, H, a.eps);
        }
    } else {
        cp_wait<0>();
    }
    __syncthreads();

    // ---- phase 2: X' of the own columns (8 a thread) ----
    {
        const int ntask = nr * 4 * upc * 8;
        for (int q = t; q < ntask; q += NT) {
            const int quad = q & 3, h = (q >> 2) & 1;
            const int rest = q >> 3, ul = rest % upc, rs = rest / upc, s = rs & 3, r = rs >> 2;
            const int off = (((r * 4 + s) * upc + ul) * 2 + h) * 32 + quad * 8;
            float x[8];
            unpack8(*reinterpret_cast<const uint4*>(xpre + off), x);
            if constexpr (POST) {
                const int u = u0 + ul, cj = u >> 3, w = u & 7;
                const int c = cj * 512 + h * 256 + w * 32 + quad * 8;
                const float rinv = rinv_s[r];
                const float p = pp_s[r * 4 + s];
                float wn[8];
                unpack8(*reinterpret_cast<const uint4*>(pnw_s + (ul * 2 + h) * 32 + quad * 8), wn);
                const float* yv = ys + ((r * upc + ul) * 2 + h) * 32 + quad * 8;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const float z = rbf(__fmul_rn(__fmul_rn(yv[k], rinv), wn[k]));
                    x[k] = rbf(__fadd_rn(__fmul_rn(p, z), x[k]));
                }
                *reinterpret_cast<uint4*>(a.X + (static_cast<size_t>(rbeg + r) * NS + s) * H + c) = pack8(x);
            }
            if constexpr (M > 0) {
                float* xd = xs + off;
                reinterpret_cast<float4*>(xd)[0] = make_float4(x[0], x[1], x[2], x[3]);
                reinterpret_cast<float4*>(xd)[1] = make_float4(x[4], x[5], x[6], x[7]);
            }
        }
    }
    IHC_TR(3);
    if constexpr (M == 0) return;
    else {
        __syncthreads();
        // ---- phase 3: mix warp partials, two (row, stream, unit) tasks a warp at a time ----
        const int ntm = 4 * upc * nr;
        for (int p = wid; p < ntm; p += 16) {
            const int p1 = p + 8;
            const bool has1 = p1 < ntm;
            const int s0 = p & 3, ul0 = (p >> 2) % upc, r0 = (p >> 2) / upc;
            const int s1 = p1 & 3, ul1 = (p1 >> 2) % upc, r1 = has1 ? (p1 >> 2) / upc : r0;
            const int u0g = u0 + ul0, u1g = u0 + ul1;
            mix_pair<M>(fs + ((ul0 * M) * 4 + s0) * 64, xs + (((r0 * 4 + s0) * upc + ul0) * 2) * 32,
                        a.wp + wp_idx(rbeg + r0, 0, s0 * 6 + (u0g >> 3), u0g & 7), has1,
                        fs + ((ul1 * M) * 4 + s1) * 64, xs + (((r1 * 4 + s1) * upc + ul1) * 2) * 32,
                        a.wp + wp_idx(rbeg + r1, 0, s1 * 6 + (u1g >> 3), u1g & 7));
        }
        // ---- the ticket ----
        IHC_TR(4);
        __threadfence();
        __syncthreads();
        IHC_TR(5);
        if (t == 0) last = atomicAdd(a.cnt + blockIdx.y, 1) == static_cast<int>(gridDim.x) - 1;
        __syncthreads();
        IHC_TR(6);
        if (!last) return;
        __threadfence();
        IHC_TR(7);

        // ---- finish ----
        float* csum = reinterpret_cast<float*>(mainp);  // [MAXRPG][9][24]   (aliases the phase buffers)
        float* tots = csum + MAXRPG * 9 * NCH;          // [MAXRPG][9]
        float* coef = tots + MAXRPG * 9;                // [MAXRPG][4]
        float* red2 = coef + MAXRPG * 4;                // [MAXRPG][8]
        float* rinvo = red2 + MAXRPG * 8;               // [MAXRPG]
        bf16* os = reinterpret_cast<bf16*>(mainp + fin_os_offset());  // [rpg][H]
        constexpr int KVN = M + 1;                      // folded values: mixes 0..M-1, then the sum of squares
        const int nitem = nr * KVN * NCH;               // <= 864: up to 4 chunk sums a thread
        float4 wa[4], wb[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = t + NT * i;
            if (q < nitem) {
                const int j = q % NCH, rest = q / NCH, kk = rest % KVN, r = rest / KVN;
                const float4* src = reinterpret_cast<const float4*>(a.wp + wp_idx(rbeg + r, kk < M ? kk : 8, j, 0));
                wa[i] = __ldcg(src);
                wb[i] = __ldcg(src + 1);
            }
        }
        // the group's X' (every CTA wrote a slice; L2): <= 6 (row, 8-column) tasks a thread, in flight through the fold
        const int ncol = nr * (H / 8);
        uint4 xr[6][4];
#pragma unroll
        for (int i = 0; i < 6; ++i) {
            const int q = t + NT * i;
            if (q < ncol) {
                const bf16* xp = a.X + static_cast<size_t>(rbeg + q / (H / 8)) * NSH + (q % (H / 8)) * 8;
#pragma unroll
                for (int s = 0; s < 4; ++s) xr[i][s] = ldcg4(xp + s * H);
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = t + NT * i;
            if (q < nitem) {
                const int j = q % NCH, rest = q / NCH, kk = rest % KVN, r = rest / KVN;
                csum[(r * 9 + kk) * NCH + j] = chunk_sum(wa[i], wb[i]);
            }
        }
        __syncthreads();
        if (t == 0) atomicExch(a.cnt + blockIdx.y, 0);  // self-cleaning: reset after wp/csum loads complete
        IHC_TR(8);
        if (t < nr * KVN) {
            const int r = t / KVN, kk = t % KVN;
            float tot = 0.f;
#pragma unroll
            for (int j = 0; j < NCH; ++j) tot = __fadd_rn(tot, csum[(r * 9 + kk) * NCH + j]);
            tots[r * 9 + kk] = tot;
        }
        __syncthreads();
        IHC_TR(9);
        constexpr int NE = M == 8 ? 8 : 4;  // coefficients a row: pre 0..3, post 4..7
        if (t < nr * NE) {
            const int r = t / NE, e = t % NE;
            const float* tt = tots + r * 9;
            const float rinv = rinv_of(tt[M], NSH, a.eps);
            const float cf = hc_coef(tt[e], rinv, bs_s[e < 4 ? 8 : 9], bs_s[e], e >= 4, a.mag, a.hc_eps);
            if (e < 4) coef[r * 4 + e] = cf;
            else a.post_next[static_cast<size_t>(rbeg + r) * 4 + (e - 4)] = cf;
        }
        if (a.dbg_tot != nullptr && t < nr * 9) {
            const int r = t / 9, k = t % 9;
            a.dbg_tot[static_cast<size_t>(rbeg + r) * 9 + k] = k < M ? tots[r * 9 + k] : (k == 8 ? tots[r * 9 + M] : 0.f);
        }
        __syncthreads();
        IHC_TR(10);
#pragma unroll
        for (int i = 0; i < 6; ++i) {
            const int q = t + NT * i;
            if (q < ncol) collapse8(coef + (q / (H / 8)) * 4, xr[i], os + q * 8);
        }
        __syncthreads();
        float sso[MAXRPG];
#pragma unroll
        for (int r = 0; r < MAXRPG; ++r) {
            sso[r] = 0.f;
            if (r < nr) {
#pragma unroll
                for (int j = 0; j < 12; ++j) {
                    const float o = bfv(os[r * H + t + 256 * j]);
                    sso[r] = __fmaf_rn(o, o, sso[r]);
                }
            }
        }
        const float x = rs_butterfly<2>(sso);
        if ((lane & 7) == 0) red2[(lane >> 3) * 8 + wid] = x;
        IHC_TR(11);
        __syncthreads();
        if (t < nr) {
            float tot = 0.f;
#pragma unroll
            for (int w = 0; w < 8; ++w) tot = __fadd_rn(tot, red2[t * 8 + w]);
            rinvo[t] = rinv_of(tot, H, a.eps);
        }
        __syncthreads();
        IHC_TR(12);
        for (int q = t; q < ncol; q += NT) {
            const int r = q / (H / 8), c = (q % (H / 8)) * 8;
            float o[8], wn[8];
            unpack8(*reinterpret_cast<const uint4*>(os + q * 8), o);
            unpack8(*reinterpret_cast<const uint4*>(nnw_s + c), wn);
            const float rinv = rinvo[r];
#pragma unroll
            for (int k = 0; k < 8; ++k) o[k] = __fmul_rn(__fmul_rn(o[k], rinv), wn[k]);
            *reinterpret_cast<uint4*>(a.A + static_cast<size_t>(rbeg + r) * H + c) = pack8(o);
        }
        IHC_TR(13);
    }
}

// Prefill rank sum + iHC post, one row per CTA (`float4`/`uint4`, bit-equal to `tf_rank_sum` + `hy4_hc_post`).
__device__ __forceinline__ void post_vec_body(const Args& a) {
    __shared__ __align__(16) float ysm[H];
    __shared__ float red[8];
    const int t = threadIdx.x, lane = t & 31, wid = t >> 5, r = blockIdx.x;
    if (r >= a.R) return;
    const float* g0 = a.G + static_cast<size_t>(r) * H;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int q = t + NT * i;
        float4 v = __ldg(reinterpret_cast<const float4*>(g0) + q);
        for (int k = 1; k < a.world; ++k) {
            const float4 u = __ldg(reinterpret_cast<const float4*>(g0 + static_cast<size_t>(k) * a.gs) + q);
            v.x = __fadd_rn(v.x, u.x);
            v.y = __fadd_rn(v.y, u.y);
            v.z = __fadd_rn(v.z, u.z);
            v.w = __fadd_rn(v.w, u.w);
        }
        reinterpret_cast<float4*>(ysm)[q] =
            a.round_in ? make_float4(rbf(v.x), rbf(v.y), rbf(v.z), rbf(v.w)) : v;
    }
    __syncthreads();
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 12; ++j) {
        const float y = ysm[t + 256 * j];
        ss = __fmaf_rn(y, y, ss);
    }
    ss = warp_sum(ss);
    if (lane == 0) red[wid] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) tot = __fadd_rn(tot, red[w]);
    const float rinv = rinv_of(tot, H, a.eps);
    float p[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) p[s] = a.post_prev[r * 4 + s];
    bf16* xr = a.X + static_cast<size_t>(r) * NSH;
    for (int cq = t; cq < H / 8; cq += NT) {
        const int c = cq * 8;
        float wn[8], z[8];
        unpack8(__ldg(reinterpret_cast<const uint4*>(a.post_nw + c)), wn);
        const float4 y0 = reinterpret_cast<const float4*>(ysm + c)[0], y1 = reinterpret_cast<const float4*>(ysm + c)[1];
        const float yv[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
#pragma unroll
        for (int k = 0; k < 8; ++k) z[k] = rbf(__fmul_rn(__fmul_rn(yv[k], rinv), wn[k]));
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            uint4* xp = reinterpret_cast<uint4*>(xr + s * H + c);
            float x[8];
            unpack8(*xp, x);
#pragma unroll
            for (int k = 0; k < 8; ++k) x[k] = __fadd_rn(__fmul_rn(p[s], z[k]), x[k]);
            *xp = pack8(x);
        }
    }
}
}  // namespace hy4ihc

#include "ihc_fused_coop_prefill.cuh"
