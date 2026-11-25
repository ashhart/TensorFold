// Hy4-flash 4:1 compressed-indexer group pooling (GrgaPalcic & popay97, 2026; 1-CTA sm_80+ & 4-CTA sm_120+).
#pragma once

#include <cooperative_groups.h>

#include "index_paged_common.cuh"

namespace tf_hy4_idx {
static_assert(0x47503937u == (('G' << 24) | ('P' << 16) | ('9' << 8) | '7'), "gp97:idx-pool4");

// 4-slot softmax-gated pool of one channel: gt = g + ape, max, e = exp(gt - max), sum, sum_j v_j * (e_j / sum).
template <int F>
__device__ __forceinline__ float pool4(const float (&v)[4], const float (&g)[4], const float (&ape)[4]) {
    float gt[4], mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        gt[j] = add_f<F>(g[j], ape[j]);
        mx = nmax<F>(mx, gt[j]);
    }
    float ex[4], sum = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        ex[j] = tf_glue::tf_exp(__fsub_rn(gt[j], mx));
        sum = add_f<F>(sum, ex[j]);
    }
    float pooled = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) pooled = add_f<F>(pooled, __fmul_rn(v[j], __fdiv_rn(ex[j], sum)));
    return pooled;
}

__device__ __forceinline__ float rinv_of(float ss, float eps) {
    return __frsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(kDim)), eps));
}

__device__ __forceinline__ float rope_half(float self, float other, float2 cs, bool odd) {
    return odd ? __fadd_rn(__fmul_rn(self, cs.x), __fmul_rn(other, cs.y))
               : __fadd_rn(__fmul_rn(self, cs.x), __fmul_rn(-other, cs.y));
}

__device__ __forceinline__ int job_stream(const int* __restrict__ JS, int job) {
    return JS != nullptr ? __ldg(JS + job) : 0;
}

__device__ __forceinline__ int job_group(const int* __restrict__ JG, int g0, int job) {
    return JG != nullptr ? __ldg(JG + job) : (g0 + job);
}

__device__ __forceinline__ size_t gk_offset(const uint32_t* __restrict__ PT, int pt_stride, int g_base, int s_, int g,
                                            int gpool) {
    if (PT != nullptr) {
        const unsigned ug = static_cast<unsigned>(g_base + g);
        return static_cast<size_t>(__ldg(PT + static_cast<size_t>(s_) * pt_stride + (ug >> 6))) * 64u + (ug & 63u);
    }
    return static_cast<size_t>(s_) * gpool + g;
}

__device__ __forceinline__ const bf16* tk_row(const int* __restrict__ JS, const int* __restrict__ JG, int g0, int job,
                                              int slot, const bf16* __restrict__ TKC, int ctx) {
    const int s_ = job_stream(JS, job), g = job_group(JG, g0, job);
    return TKC + (static_cast<size_t>(s_) * ctx + 4 * static_cast<size_t>(g) + slot) * kDim;
}

// 32-channel square sum in the baseline's xor-butterfly order for register-tiled octet lanes cg + 8i (i = 0..3).
template <int F>
__device__ __forceinline__ float block_sq_sum_octet(const float (&pl)[4]) {
    const float a0 = add_f<F>(__fmul_rn(pl[0], pl[0]), __fmul_rn(pl[2], pl[2]));
    const float a1 = add_f<F>(__fmul_rn(pl[1], pl[1]), __fmul_rn(pl[3], pl[3]));
    float s = add_f<F>(a0, a1);
    s = add_f<F>(s, __shfl_xor_sync(0xffffffffu, s, 4));
    s = add_f<F>(s, __shfl_xor_sync(0xffffffffu, s, 2));
    s = add_f<F>(s, __shfl_xor_sync(0xffffffffu, s, 1));
    return s;
}

template <int F, int NK4, int XSTRIDE, int WSTRIDE>
__device__ __forceinline__ void mma_rows4(float (&av)[4][4], float (&ag)[4][4], const float* __restrict__ xrow, int xk4,
                                          int xsw, const float* __restrict__ wbase, int wk4, int cg_, bool last_k) {
#pragma unroll
    for (int k4 = 0; k4 < NK4; ++k4) {
        float4 wv[4], wg[4], x[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            wv[i] = *reinterpret_cast<const float4*>(wbase + (cg_ + 8 * i) * WSTRIDE + (((wk4 + k4) ^ cg_) << 2));
            wg[i] = *reinterpret_cast<const float4*>(wbase + (32 + cg_ + 8 * i) * WSTRIDE + (((wk4 + k4) ^ cg_) << 2));
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
            x[j] = *reinterpret_cast<const float4*>(xrow + j * XSTRIDE + (((xk4 + k4) ^ xsw) << 2));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                av[j][i] = step_fma<F>(wv[i].x, x[j].x, av[j][i]);
                ag[j][i] = step_fma<F>(wg[i].x, x[j].x, ag[j][i]);
                av[j][i] = step_fma<F>(wv[i].y, x[j].y, av[j][i]);
                ag[j][i] = step_fma<F>(wg[i].y, x[j].y, ag[j][i]);
                av[j][i] = step_fma<F>(wv[i].z, x[j].z, av[j][i]);
                ag[j][i] = step_fma<F>(wg[i].z, x[j].z, ag[j][i]);
                if (!(F == kBadTail && last_k && k4 == NK4 - 1)) {
                    av[j][i] = step_fma<F>(wv[i].w, x[j].w, av[j][i]);
                    ag[j][i] = step_fma<F>(wg[i].w, x[j].w, ag[j][i]);
                }
            }
        }
    }
}

__device__ __forceinline__ void load_ape(const float* __restrict__ APE, int q, int cg_, float (&ape)[4][4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) ape[i][j] = __ldg(APE + j * kDim + 32 * q + cg_ + 8 * i);
}

__device__ __forceinline__ void load_rt(const float2* __restrict__ RT, int q, int cg_, size_t rt_row, float2 (&cs)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) cs[i] = __ldg(RT + rt_row * kRopePairs + ((32 * q + cg_ + 8 * i - 64) >> 1));
}

template <int F>
__device__ __forceinline__ void pool_tile(const float (&av)[4][4], const float (&ag)[4][4], const float (&ape)[4][4],
                                          float (&pl)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float v4[4] = {av[0][i], av[1][i], av[2][i], av[3][i]};
        const float g4[4] = {ag[0][i], ag[1][i], ag[2][i], ag[3][i]};
        pl[i] = pool4<F>(v4, g4, ape[i]);
    }
}

template <int F>
__device__ __forceinline__ void store_tile(const float (&pl)[4], int q, int cg_, float rinv, const float (&knw)[4],
                                           const float2 (&cs)[4], bool live, bf16* __restrict__ dst) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = 32 * q + cg_ + 8 * i;
        float y = tf_glue::rbf(__fmul_rn(knw[i], __fmul_rn(pl[i], rinv)));
        if (q < 2) {
            if (F == kBadNan && y != y) y = 0.0f;
            if (live) dst[c] = __float2bfloat16_rn(y);
        } else {
            const float other = __shfl_xor_sync(0xffffffffu, y, 1);
            float out_v = rope_half(y, other, cs[i], (cg_ & 1) != 0);
            if (F == kBadNan && out_v != out_v) out_v = 0.0f;
            if (live) dst[c] = __float2bfloat16_rn(out_v);
        }
    }
}

template <int F>
__device__ __forceinline__ void pool_1cta_body(const int* __restrict__ JS, const int* __restrict__ JG, int g0,
                                               int njobs, const bf16* __restrict__ TKC, int ctx,
                                               const float* __restrict__ WV, const float* __restrict__ WG,
                                               const float* __restrict__ APE, const float* __restrict__ KNW,
                                               const float2* __restrict__ RT, int rt_stream_stride, int rt_group_stride,
                                               float eps, bf16* __restrict__ GKC,
                                               const uint32_t* __restrict__ PT = nullptr, int pt_stride = 0,
                                               int g_base = 0) {
    constexpr int BJ = 16, NSTAGE = 2, NT = BJ * 8, ROWS = BJ * 4, BK = kBatBk, KCH = kDim / BK, CHF = 64 * BK,
                  NCHUNK = 4 * KCH;
    extern __shared__ __align__(16) float smem[];
    float *Xs = smem, *Wr = smem + ROWS * kDim;
    const int tid = threadIdx.x, cg_ = tid & 7, rg = tid >> 3, job0 = blockIdx.x * BJ;
    if (njobs > 0) bump<F>(tk_row(JS, JG, g0, 0, 0, TKC, ctx));
    auto issue_w = [&](int ch) {
        const int q = ch / KCH, kc = ch % KCH;
        float* dst = Wr + (ch % NSTAGE) * CHF;
#pragma unroll
        for (int e = tid; e < 64 * (BK / 4); e += NT) {
            const int w = e / (BK / 4), u = e % (BK / 4);
            const float* src = (w < 32 ? WV + static_cast<size_t>(32 * q + w) * kDim
                                       : WG + static_cast<size_t>(32 * q + w - 32) * kDim) +
                               kc * BK + 4 * u;
            tf_common::cp16(dst + w * BK + ((u ^ (w & 7)) << 2), src);
        }
    };
    issue_w(0);
    tf_common::cp_commit();
#pragma unroll
    for (int e = tid; e < ROWS * 16; e += NT) {
        const int r = e >> 4, u = e & 15, job = job0 + (r >> 2);
        uint4 w = make_uint4(0u, 0u, 0u, 0u);
        if (job < njobs) w = __ldg(reinterpret_cast<const uint4*>(tk_row(JS, JG, g0, job, r & 3, TKC, ctx)) + u);
        const int sw = (r >> 2) & 7;
        *reinterpret_cast<float4*>(Xs + r * kDim + (((2 * u) ^ sw) << 2)) =
            make_float4(bf_lo(w.x), bf_hi(w.x), bf_lo(w.y), bf_hi(w.y));
        *reinterpret_cast<float4*>(Xs + r * kDim + (((2 * u + 1) ^ sw) << 2)) =
            make_float4(bf_lo(w.z), bf_hi(w.z), bf_lo(w.w), bf_hi(w.w));
    }
    float av[4][4] = {}, ag[4][4] = {}, P[4][4] = {}, S[4] = {}, ape[4][4];
    load_ape(APE, 0, cg_, ape);
    const float* xrow = Xs + 4 * rg * kDim;
    const int xsw = rg & 7;
#pragma unroll 1
    for (int ch = 0; ch < NCHUNK; ++ch) {
        tf_common::cp_wait<0>();
        __syncthreads();
        if (ch + 1 < NCHUNK) issue_w(ch + 1);
        tf_common::cp_commit();
        const int kc = ch % KCH;
        mma_rows4<F, BK / 4, kDim, BK>(av, ag, xrow, kc * (BK / 4), xsw, Wr + (ch % NSTAGE) * CHF, 0, cg_,
                                       kc == KCH - 1);
        if (kc == KCH - 1) {
            const int q = ch / KCH;
            float pl[4];
            pool_tile<F>(av, ag, ape, pl);
            if (q + 1 < 4) load_ape(APE, q + 1, cg_, ape);
            const float s = block_sq_sum_octet<F>(pl);
#pragma unroll
            for (int qq = 0; qq < 4; ++qq)
                if (qq == q) {
                    S[qq] = s;
#pragma unroll
                    for (int i = 0; i < 4; ++i) P[qq][i] = pl[i];
                }
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) av[j][i] = ag[j][i] = 0.f;
        }
    }
    tf_common::cp_wait<0>();
    float t = 0.f;
#pragma unroll
    for (int q = 0; q < 4; ++q) t = add_f<F>(t, S[q]);
    const float rinv = rinv_of(t, eps);
    const int job = job0 + rg;
    const bool live = job < njobs;
    const int s_ = live ? job_stream(JS, job) : 0, g = live ? job_group(JG, g0, job) : 0;
    bf16* dst = GKC + gk_offset(PT, pt_stride, g_base, s_, g, ctx / 4) * kDim;
    float knw[4][4];
    float2 cs[4][4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            knw[q][i] = __ldg(KNW + 32 * q + cg_ + 8 * i);
            cs[q][i] = make_float2(0.f, 0.f);
        }
    if (live) {
        const size_t rt_row = static_cast<size_t>(s_) * rt_stream_stride + static_cast<size_t>(g) * rt_group_stride;
        load_rt(RT, 2, cg_, rt_row, cs[2]);
        load_rt(RT, 3, cg_, rt_row, cs[3]);
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) store_tile<F>(P[q], q, cg_, rinv, knw[q], cs[q], live, dst);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200

__device__ __forceinline__ void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }

__device__ __forceinline__ void cp_wait_dyn(int n) {
    switch (n) {
        case 0: tf_common::cp_wait<0>(); break;
        case 1: tf_common::cp_wait<1>(); break;
        case 2: tf_common::cp_wait<2>(); break;
        default: tf_common::cp_wait<3>(); break;
    }
}

template <int F>
__device__ __forceinline__ void pool_small_body(const int* __restrict__ JS, const int* __restrict__ JG, int g0,
                                                int njobs, const bf16* __restrict__ TKC, int ctx,
                                                const float* __restrict__ WV, const float* __restrict__ WG,
                                                const float* __restrict__ APE, const float* __restrict__ KNW,
                                                const float2* __restrict__ RT, int rt_stream_stride,
                                                int rt_group_stride, float eps, bf16* __restrict__ GKC,
                                                const uint32_t* __restrict__ PT = nullptr, int pt_stride = 0,
                                                int g_base = 0) {
    __shared__ __align__(16) float Ws[64 * kDim];
    __shared__ __align__(16) float Xs[4 * kDim];
    __shared__ float Ex[2][4][32];
    __shared__ float Sx[1];
    namespace cg = cooperative_groups;
    cg::cluster_group cl = cg::this_cluster();
    const int q = static_cast<int>(cl.block_rank()), job = blockIdx.x / 4, tid = threadIdx.x;
    if (njobs > 0) bump<F>(tk_row(JS, JG, g0, 0, 0, TKC, ctx));
#pragma unroll
    for (int kq = 0; kq < 4; ++kq) {
#pragma unroll
        for (int e = tid; e < 64 * 8; e += 128) {
            const int w = e >> 3, u = 8 * kq + (e & 7);
            const float* src = (w < 32 ? WV + static_cast<size_t>(32 * q + w) * kDim
                                       : WG + static_cast<size_t>(32 * q + w - 32) * kDim) +
                               4 * u;
            tf_common::cp16(Ws + w * kDim + ((u ^ (w & 7)) << 2), src);
        }
        tf_common::cp_commit();
    }
    const int c = tid & 31, j = tid >> 5, sw = c & 7, w = tid >> 5, ch = 32 * q + c;
    float ape[4], knw = 0.f;
    float2 cs = make_float2(0.f, 0.f);
    int s_ = 0, g_ = 0;
    const bool live = w == 0 && job < njobs;
    if (w == 0) {
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) ape[jj] = __ldg(APE + jj * kDim + ch);
        knw = __ldg(KNW + ch);
        if (live) {
            s_ = job_stream(JS, job);
            g_ = job_group(JG, g0, job);
            if (q >= 2) {
                const size_t rt_row =
                    static_cast<size_t>(s_) * rt_stream_stride + static_cast<size_t>(g_) * rt_group_stride;
                cs = __ldg(RT + rt_row * kRopePairs + ((ch - 64) >> 1));
            }
        }
    }
    for (int e = tid; e < 64; e += 128) {
        const int r = e >> 4, u = e & 15;
        uint4 x = make_uint4(0u, 0u, 0u, 0u);
        if (job < njobs) x = __ldg(reinterpret_cast<const uint4*>(tk_row(JS, JG, g0, job, r, TKC, ctx)) + u);
        *reinterpret_cast<float4*>(Xs + r * kDim + 8 * u) = make_float4(bf_lo(x.x), bf_hi(x.x), bf_lo(x.y), bf_hi(x.y));
        *reinterpret_cast<float4*>(Xs + r * kDim + 8 * u + 4) =
            make_float4(bf_lo(x.z), bf_hi(x.z), bf_lo(x.w), bf_hi(x.w));
    }
    const float *wvr = Ws + c * kDim, *wgr = Ws + (32 + c) * kDim;
    float v = 0.f, g = 0.f;
#pragma unroll
    for (int kq = 0; kq < 4; ++kq) {
        cp_wait_dyn(3 - kq);
        __syncthreads();
#pragma unroll
        for (int k4 = 8 * kq; k4 < 8 * kq + 8; ++k4) {
            const float4 a = *reinterpret_cast<const float4*>(wvr + ((k4 ^ sw) << 2));
            const float4 b = *reinterpret_cast<const float4*>(wgr + ((k4 ^ sw) << 2));
            const float4 x = *reinterpret_cast<const float4*>(Xs + j * kDim + 4 * k4);
            v = step_fma<F>(a.x, x.x, v);
            g = step_fma<F>(b.x, x.x, g);
            v = step_fma<F>(a.y, x.y, v);
            g = step_fma<F>(b.y, x.y, g);
            v = step_fma<F>(a.z, x.z, v);
            g = step_fma<F>(b.z, x.z, g);
            if (!(F == kBadTail && kq == 3 && k4 == 31)) {
                v = step_fma<F>(a.w, x.w, v);
                g = step_fma<F>(b.w, x.w, g);
            }
        }
    }
    Ex[0][j][c] = v;
    Ex[1][j][c] = g;
    __syncthreads();
    float pooled = 0.f;
    if (w == 0) {
        float v4[4], g4[4];
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            v4[jj] = Ex[0][jj][c];
            g4[jj] = Ex[1][jj][c];
        }
        pooled = pool4<F>(v4, g4, ape);
        float s = __fmul_rn(pooled, pooled);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s = add_f<F>(s, __shfl_xor_sync(0xffffffffu, s, o));
        if (c == 0) Sx[0] = s;
    }
    cl.sync();
    if (w == 0) {
        float t = 0.f;
#pragma unroll
        for (int r = 0; r < 4; ++r) t = add_f<F>(t, cl.map_shared_rank(Sx, r)[0]);
        const float rinv = rinv_of(t, eps);
        float y = tf_glue::rbf(__fmul_rn(knw, __fmul_rn(pooled, rinv)));
        const float other = __shfl_xor_sync(0xffffffffu, y, 1);
        float out_v = q < 2 ? y : rope_half(y, other, cs, (c & 1) != 0);
        if (F == kBadNan && out_v != out_v) out_v = 0.0f;
        if (live) GKC[gk_offset(PT, pt_stride, g_base, s_, g_, ctx / 4) * kDim + ch] = __float2bfloat16_rn(out_v);
    }
    cl.sync();
}

template <int F, int NK4, int XSTRIDE, int WSTRIDE>
__device__ __forceinline__ void mma_tile(float (&av)[4][4], float (&ag)[4][4], const float* __restrict__ xrow, int xsw,
                                         const float* __restrict__ wbase, int wk4, int cg_, bool last_k) {
#pragma unroll 1
    for (int k4 = 0; k4 < NK4; ++k4) {
        float4 x[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) x[r] = *reinterpret_cast<const float4*>(xrow + r * XSTRIDE + ((k4 ^ xsw) << 2));
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 w =
                *reinterpret_cast<const float4*>(wbase + (cg_ + 8 * i) * WSTRIDE + (((wk4 + k4) ^ cg_) << 2));
            float (&a)[4][4] = i < 4 ? av : ag;
            const int ii = i & 3;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                a[r][ii] = step_fma<F>(w.x, x[r].x, a[r][ii]);
                a[r][ii] = step_fma<F>(w.y, x[r].y, a[r][ii]);
                a[r][ii] = step_fma<F>(w.z, x[r].z, a[r][ii]);
                if (!(F == kBadTail && last_k && k4 == NK4 - 1)) a[r][ii] = step_fma<F>(w.w, x[r].w, a[r][ii]);
            }
        }
    }
}

template <int F>
__device__ __forceinline__ void pool_persist_body(const int* __restrict__ JS, const int* __restrict__ JG, int g0,
                                                  int njobs, const bf16* __restrict__ TKC, int ctx,
                                                  const float* __restrict__ WV, const float* __restrict__ WG,
                                                  const float* __restrict__ APE, const float* __restrict__ KNW,
                                                  const float2* __restrict__ RT, int rt_stream_stride,
                                                  int rt_group_stride, float eps, bf16* __restrict__ GKC,
                                                  const uint32_t* __restrict__ PT = nullptr, int pt_stride = 0,
                                                  int g_base = 0) {
    constexpr int NT = 256, BK = 16, BJ = NT / 8, ROWS = 4 * BJ, KCH = kDim / BK, XCH = ROWS * BK,
                  NU = ROWS * (BK / 8) / NT, SWM = BK / 4 - 1;
    extern __shared__ __align__(16) float smem[];
    float *Ws = smem, *Xr = smem + 64 * kDim, *Sx = Xr + 2 * XCH;
    namespace cg = cooperative_groups;
    cg::cluster_group cl = cg::this_cluster();
    const int q = static_cast<int>(cl.block_rank()), cid = blockIdx.x / 4, ncl = gridDim.x / 4;
    const int nblk = (njobs + BJ - 1) / BJ, tid = threadIdx.x, cg_ = tid & 7, rg = tid >> 3;
    if (njobs > 0) bump<F>(tk_row(JS, JG, g0, 0, 0, TKC, ctx));
#pragma unroll
    for (int e = tid; e < 64 * (kDim / 4); e += NT) {
        const int w = e >> 5, u = e & 31;
        const float* src = (w < 32 ? WV + static_cast<size_t>(32 * q + w) * kDim
                                   : WG + static_cast<size_t>(32 * q + w - 32) * kDim) +
                           4 * u;
        tf_common::cp16(Ws + w * kDim + ((u ^ (w & 7)) << 2), src);
    }
    tf_common::cp_commit();
    const int nchunk = nblk > cid ? ((nblk - cid + ncl - 1) / ncl) * KCH : 0;
    uint4 pre[NU];
    auto fetch = [&](int t) {
        const int blk = cid + (t / KCH) * ncl, kc = t % KCH;
#pragma unroll
        for (int h = 0; h < NU; ++h) {
            const int e = tid + h * NT, r = e / (BK / 8), u8 = e % (BK / 8), job = blk * BJ + (r >> 2);
            pre[h] = make_uint4(0u, 0u, 0u, 0u);
            if (job < njobs)
                pre[h] = __ldg(reinterpret_cast<const uint4*>(tk_row(JS, JG, g0, job, r & 3, TKC, ctx) + kc * BK) + u8);
        }
    };
    auto put = [&](int t) {
        float* dst = Xr + (t & 1) * XCH;
#pragma unroll
        for (int h = 0; h < NU; ++h) {
            const int e = tid + h * NT, r = e / (BK / 8), u8 = e % (BK / 8), sw = (r / 4) & SWM;
            const uint4 w = pre[h];
            *reinterpret_cast<float4*>(dst + r * BK + (((2 * u8) ^ sw) << 2)) =
                make_float4(bf_lo(w.x), bf_hi(w.x), bf_lo(w.y), bf_hi(w.y));
            *reinterpret_cast<float4*>(dst + r * BK + (((2 * u8 + 1) ^ sw) << 2)) =
                make_float4(bf_lo(w.z), bf_hi(w.z), bf_lo(w.w), bf_hi(w.w));
        }
    };
    float av[4][4] = {}, ag[4][4] = {};
    bool pend = false, plive = false;
    int pbuf = 0, ps = 0, pg = 0;
    float ppl[4];
    float2 pcs[4];
    auto finish = [&]() {
        cluster_wait();
        float knw[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) knw[i] = __ldg(KNW + 32 * q + cg_ + 8 * i);
        float tt = 0.f;
#pragma unroll
        for (int r = 0; r < 4; ++r) tt = add_f<F>(tt, cl.map_shared_rank(Sx, r)[pbuf * BJ + rg]);
        store_tile<F>(ppl, q, cg_, rinv_of(tt, eps), knw, pcs, plive,
                      GKC + (plive ? gk_offset(PT, pt_stride, g_base, ps, pg, ctx / 4) : 0u) * kDim);
        pend = false;
    };
    if (nchunk > 0) {
        fetch(0);
        put(0);
    }
    tf_common::cp_wait<0>();
    __syncthreads();
    const int xsw = rg & SWM;
#pragma unroll 1
    for (int t = 0; t < nchunk; ++t) {
        if (t + 1 < nchunk) fetch(t + 1);
        const int kc = t % KCH;
        mma_tile<F, BK / 4, BK, kDim>(av, ag, Xr + (t & 1) * XCH + 4 * rg * BK, xsw, Ws, kc * (BK / 4), cg_,
                                      kc == KCH - 1);
        if (t + 1 < nchunk) put(t + 1);
        if (pend) finish();
        if (kc == KCH - 1) {
            const int lb = t / KCH, blk = cid + lb * ncl;
            float ape[4][4];
            load_ape(APE, q, cg_, ape);
            pbuf = lb & 1;
            pool_tile<F>(av, ag, ape, ppl);
            const float s = block_sq_sum_octet<F>(ppl);
            if (cg_ == 0) Sx[pbuf * BJ + rg] = s;
            cluster_arrive();
            const int job = blk * BJ + rg;
            plive = job < njobs;
            ps = 0;
            pg = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) pcs[i] = make_float2(0.f, 0.f);
            if (plive) {
                ps = job_stream(JS, job);
                pg = job_group(JG, g0, job);
                if (q >= 2)
                    load_rt(RT, q, cg_,
                            static_cast<size_t>(ps) * rt_stream_stride + static_cast<size_t>(pg) * rt_group_stride,
                            pcs);
            }
            pend = true;
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int i = 0; i < 4; ++i) av[r][i] = ag[r][i] = 0.f;
        }
        __syncthreads();
    }
    if (pend) finish();
    cl.sync();
}

#else

template <int F>
__device__ __forceinline__ void pool_small_body(const int*, const int*, int, int, const bf16*, int, const float*,
                                                const float*, const float*, const float*, const float2*, int, int,
                                                float, bf16*, const uint32_t* = nullptr, int = 0, int = 0) {
    __trap();
}

template <int F>
__device__ __forceinline__ void pool_persist_body(const int*, const int*, int, int, const bf16*, int, const float*,
                                                  const float*, const float*, const float*, const float2*, int, int,
                                                  float, bf16*, const uint32_t* = nullptr, int = 0, int = 0) {
    __trap();
}

#endif

}  // namespace tf_hy4_idx
