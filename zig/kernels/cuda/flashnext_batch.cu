// The small device steps of Flash Next's batched prompt pass (cuda_batch.zig), built without FMA contraction.

#include <cuda_bf16.h>
#include <stdint.h>
#include "common/reduce.cuh"

// out[i] = the sum of the sk slices' part[s][i] in order, rounded once (qmm.cu's reduce_kernel): bf16 if out_bf16.
extern "C" __global__ void fn_sum_slices(const float* __restrict__ part, void* __restrict__ out, int out_bf16,
                                         long long n, int sk) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = part[i];
    for (int s = 1; s < sk; ++s) acc = acc + part[static_cast<long long>(s) * n + i];
    if (out_bf16) reinterpret_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(acc);
    else reinterpret_cast<float*>(out)[i] = acc;
}

// Shifts the last-`taps`-rows window [taps, c] in place past `rows`, n rows `stride` apart (conv history, n-gram tail).
static __device__ __forceinline__ void shift_window(__nv_bfloat16* __restrict__ win, const __nv_bfloat16* __restrict__ rows,
                                                   long long stride, int n, int taps, int c) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= c) return;
    __nv_bfloat16 next[16];
    for (int j = 0; j < taps; ++j) {
        const int src = j + n;                       // row src of [window (taps rows); new rows]
        next[j] = src < taps ? win[static_cast<long long>(src) * c + ch]
                             : rows[static_cast<long long>(src - taps) * stride + ch];
    }
    for (int j = 0; j < taps; ++j) win[static_cast<long long>(j) * c + ch] = next[j];
}

extern "C" __global__ void fn_shift_window(__nv_bfloat16* __restrict__ win, const __nv_bfloat16* __restrict__ rows,
                                           long long stride, int n, int taps, int c) {
    shift_window(win, rows, stride, n, taps, c);
}

// ---- the DeltaNet chain over a chunk: flashnext_gdn.cu's chain cut into front, rows and norm, with its bits ----

namespace fn_gdn {
constexpr int NK = 16, NV = 48, DK = 128, DV = 128, TAPS = 4;
constexpr int C = 2 * NK * DK + NV * DV;
constexpr int PW = C + NV * DV + 2 * NV;

static __device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

using tf_common::warp_sum;

static __device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

static __device__ __forceinline__ float softplusf_(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

static __device__ __forceinline__ float conv_act(const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs,
                                                 const __nv_bfloat16* __restrict__ cw, int r, int c) {
    float acc = 0.0f;
#pragma unroll
    for (int tap = 0; tap < TAPS; ++tap) {
        const int at = r + tap;
        const float x = at < TAPS - 1 ? __bfloat162float(cs[at * C + c])
                                      : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * PW + c]);
        acc = acc + __bfloat162float(cw[c * TAPS + tap]) * x;
    }
    return bf(acc / (1.0f + expf(-acc)));
}
}  // namespace fn_gdn

// A layer's pending-row log: log_rows rows of k [NK, DK], v [NV, DV], decay and beta [NV, 2], fp32; ctl first | n << 16
namespace fn_late {
using namespace fn_gdn;
static __device__ __forceinline__ const float* vs(const float* log, int log_rows) { return log + (size_t)log_rows * NK * DK; }
static __device__ __forceinline__ const float* gbs(const float* log, int log_rows) { return vs(log, log_rows) + (size_t)log_rows * NV * DV; }
}  // namespace fn_late

// Program (row, unit), 256 threads: key head u's normed q and k (u < NK), else a value head's v, decay g and beta.
template <bool LOG>  // LOG: k, v and the gates go into the log's row after the pending ones
static __device__ __forceinline__ void gdn_front(
        const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs, const __nv_bfloat16* __restrict__ cw,
        const float* __restrict__ a_log, const float* __restrict__ dt_bias, float* __restrict__ qn,
        float* __restrict__ kn, float* __restrict__ v, float* __restrict__ gb, const unsigned* __restrict__ ctl, int log_rows) {
    using namespace fn_gdn;
    const int r = blockIdx.x, u = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int kr = LOG ? (int)(((*ctl & 0xffffu) + (*ctl >> 16) + (unsigned)r) % (unsigned)log_rows) : r;
    if (u < NK) {
        __shared__ float x[2][DK];
        const int hk = u;
        x[t / DK][t % DK] = conv_act(P, cs, cw, r, t < DK ? hk * DK + t : NK * DK + hk * DK + (t - DK));
        __syncthreads();
        if (warp < 2) {
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v4[i] = x[warp][lane * 4 + i];
                ss = ss + v4[i] * v4[i];
            }
            ss = warp_sum(ss);
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            if (warp == 0) inv = inv * (1.0f / sqrtf((float)DK));
            float* dst = warp == 0 ? qn + ((size_t)r * NK + hk) * DK : kn + ((size_t)kr * NK + hk) * DK;
#pragma unroll
            for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = v4[i] * inv;
        }
    } else {
        const int hv = u - NK;
        if (t < DV) v[(size_t)kr * NV * DV + hv * DV + t] = conv_act(P, cs, cw, r, 2 * NK * DK + hv * DV + t);
        else if (t == DV) {
            const float b = __bfloat162float(P[(size_t)r * PW + C + NV * DV + hv]);
            const float a = __bfloat162float(P[(size_t)r * PW + C + NV * DV + NV + hv]);
            gb[((size_t)kr * NV + hv) * 2] = expf(-expf(a_log[hv]) * softplusf_(a + dt_bias[hv]));
            gb[((size_t)kr * NV + hv) * 2 + 1] = bf(sigmoidf_(b));
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) fn_gdn_front(
        const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs, const __nv_bfloat16* __restrict__ cw,
        const float* __restrict__ a_log, const float* __restrict__ dt_bias, float* __restrict__ qn,
        float* __restrict__ kn, float* __restrict__ v, float* __restrict__ gb) {
    gdn_front<false>(P, cs, cw, a_log, dt_bias, qn, kn, v, gb, nullptr, 1);
}

// fn_gdn_front with the window's k, v and gates into the stream's log after its pending rows (q as fn_gdn_front's).
extern "C" __global__ void __launch_bounds__(256) fn_gdn_front_log(
        const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs, const __nv_bfloat16* __restrict__ cw,
        const float* __restrict__ a_log, const float* __restrict__ dt_bias, float* __restrict__ qn,
        float* __restrict__ log, const unsigned* __restrict__ ctl, int log_rows) {
    using namespace fn_gdn;
    float* vl = log + (size_t)log_rows * NK * DK;
    gdn_front<true>(P, cs, cw, a_log, dt_bias, qn, log, vl, vl + (size_t)log_rows * NV * DV, ctl, log_rows);
}

// The chain through the chunk from state_in to state_out (in place, or apart for a rollback): a warp 4 state rows.
static __device__ __forceinline__ void gdn_rows(
        const float* __restrict__ qn, const float* __restrict__ kn, const float* __restrict__ v,
        const float* __restrict__ gb, const float* state_in, float* state_out, int rows, float* __restrict__ ys_out) {
    using namespace fn_gdn;
    const int hv = blockIdx.x, hk = hv / (NV / NK), warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.y * 16 + warp * 4;
    const size_t sbase = (size_t)hv * DV * DK;
    float s[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(row0 + j) * DK + lane * 4 + i];
    float kk[4], qq[4], vv[4], g = 0.0f, beta = 0.0f;
    auto load = [&](int r, float (&k_)[4], float (&q_)[4], float (&v_)[4], float& g_, float& b_) {
        const float4 k4 = *reinterpret_cast<const float4*>(kn + ((size_t)r * NK + hk) * DK + lane * 4);
        const float4 q4 = *reinterpret_cast<const float4*>(qn + ((size_t)r * NK + hk) * DK + lane * 4);
        const float4 v4 = *reinterpret_cast<const float4*>(v + (size_t)r * NV * DV + hv * DV + row0);
        k_[0] = k4.x; k_[1] = k4.y; k_[2] = k4.z; k_[3] = k4.w;
        q_[0] = q4.x; q_[1] = q4.y; q_[2] = q4.z; q_[3] = q4.w;
        v_[0] = v4.x; v_[1] = v4.y; v_[2] = v4.z; v_[3] = v4.w;
        g_ = gb[((size_t)r * NV + hv) * 2];
        b_ = gb[((size_t)r * NV + hv) * 2 + 1];
    };
    if (rows > 0) load(0, kk, qq, vv, g, beta);
    for (int r = 0; r < rows; ++r) {
        float nk[4], nq[4], nv[4], ng = 0.0f, nb = 0.0f;
        if (r + 1 < rows) load(r + 1, nk, nq, nv, ng, nb);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float kv = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[j][i] = s[j][i] * g;
                kv = kv + s[j][i] * kk[i];
            }
            kv = warp_sum(kv);
            const float delta = (vv[j] - kv) * beta;
#pragma unroll
            for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
        }
        float o4[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
            o4[j] = bf(warp_sum(o));
        }
        if (lane == 0) *reinterpret_cast<float4*>(ys_out + ((size_t)r * NV + hv) * DV + row0) = make_float4(o4[0], o4[1], o4[2], o4[3]);
        if (r + 1 < rows) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                kk[i] = nk[i];
                qq[i] = nq[i];
                vv[i] = nv[i];
            }
            g = ng;
            beta = nb;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(row0 + j) * DK + lane * 4 + i] = s[j][i];
}

extern "C" __global__ void __launch_bounds__(128) fn_gdn_rows(
        const float* __restrict__ qn, const float* __restrict__ kn, const float* __restrict__ v,
        const float* __restrict__ gb, float* state, int rows, float* __restrict__ ys_out) {
    gdn_rows(qn, kn, v, gb, state, state, rows, ys_out);
}

extern "C" __global__ void __launch_bounds__(128) fn_gdn_rows_to(
        const float* __restrict__ qn, const float* __restrict__ kn, const float* __restrict__ v,
        const float* __restrict__ gb, const float* state_in, float* state_out, int rows, float* __restrict__ ys_out) {
    gdn_rows(qn, kn, v, gb, state_in, state_out, rows, ys_out);
}

// A window's chain after its stream's pending rows: those first (state only), then its rows read out; fn_gdn_rows' sums
static __device__ __forceinline__ void gdn_rows_late(
        const float* __restrict__ qn, const float* __restrict__ log, const unsigned* __restrict__ ctl, int log_rows,
        int store_above, float* state_in, float* state_out, int rows, float* __restrict__ ys_out) {
    using namespace fn_gdn;
    const int hv = blockIdx.x, hk = hv / (NV / NK), warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.y * 16 + warp * 4;
    const size_t sbase = (size_t)hv * DV * DK;
    const float* vl = fn_late::vs(log, log_rows);
    const float* gl = fn_late::gbs(log, log_rows);
    const unsigned first = *ctl & 0xffffu, np = *ctl >> 16;
    float s[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(row0 + j) * DK + lane * 4 + i];
    const unsigned total = np + (unsigned)rows;
    for (unsigned p = 0; p < total; ++p) {
        if (p == np && (int)np > store_above) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) state_in[sbase + (size_t)(row0 + j) * DK + lane * 4 + i] = s[j][i];
        }
        const size_t lr = (first + p) % (unsigned)log_rows;
        const float4 k4 = *reinterpret_cast<const float4*>(log + (lr * NK + hk) * DK + lane * 4);
        const float4 v4 = *reinterpret_cast<const float4*>(vl + lr * NV * DV + hv * DV + row0);
        const float kk[4] = {k4.x, k4.y, k4.z, k4.w}, vv[4] = {v4.x, v4.y, v4.z, v4.w};
        const float g = gl[(lr * NV + hv) * 2], beta = gl[(lr * NV + hv) * 2 + 1];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float kv = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[j][i] = s[j][i] * g;
                kv = kv + s[j][i] * kk[i];
            }
            kv = warp_sum(kv);
            const float delta = (vv[j] - kv) * beta;
#pragma unroll
            for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
        }
        if (p < np) continue;
        const int r = (int)(p - np);
        const float4 q4 = *reinterpret_cast<const float4*>(qn + ((size_t)r * NK + hk) * DK + lane * 4);
        const float qq[4] = {q4.x, q4.y, q4.z, q4.w};
        float o4[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
            o4[j] = bf(warp_sum(o));
        }
        if (lane == 0) *reinterpret_cast<float4*>(ys_out + ((size_t)r * NV + hv) * DV + row0) = make_float4(o4[0], o4[1], o4[2], o4[3]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(row0 + j) * DK + lane * 4 + i] = s[j][i];
}

extern "C" __global__ void __launch_bounds__(128) fn_gdn_rows_late(
        const float* __restrict__ qn, const float* __restrict__ log, const unsigned* __restrict__ ctl, int log_rows,
        int store_above, float* state_in, float* state_out, int rows, float* __restrict__ ys_out) {
    gdn_rows_late(qn, log, ctl, log_rows, store_above, state_in, state_out, rows, ys_out);
}

// A rolled-back lane's conv windows and n-gram tail in one launch: each window is its copy moved on by `keep` rows.
namespace fn_roll {
constexpr int MAXJ = 40;
struct Job {
    __nv_bfloat16* win;
    const __nv_bfloat16* hist;
    const __nv_bfloat16* rows;
    long long stride;
    int taps, c;
};
struct Jobs {
    Job j[MAXJ];
    int n;
};
}  // namespace fn_roll

extern "C" __global__ void fn_late_roll(const fn_roll::Jobs t, int keep) {
    const fn_roll::Job& j = t.j[blockIdx.y];
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= j.c) return;
    for (int r = 0; r < j.taps; ++r) {
        const int src = r + keep;  // row src of [window (taps rows); the pass's rows]
        j.win[(long long)r * j.c + ch] = src < j.taps ? j.hist[(long long)src * j.c + ch] : j.rows[(long long)(src - j.taps) * j.stride + ch];
    }
}

extern "C" __global__ void __launch_bounds__(128) fn_gdn_norm(
        const float* __restrict__ ys_in, const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ norm_w,
        float eps, __nv_bfloat16* __restrict__ out, float* __restrict__ xs) {
    using namespace fn_gdn;
    const int r = blockIdx.x, hv = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float ys[DV];
    __shared__ float rinv;
    ys[t] = ys_in[((size_t)r * NV + hv) * DV + t];
    __syncthreads();
    if (warp == 0) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float y = ys[lane * 4 + i];
            ss = ss + y * y;
        }
        ss = warp_sum(ss);
        if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
    }
    __syncthreads();
    const float yn = bf(bf(ys[t] * rinv) * __bfloat162float(norm_w[t]));
    const float z = __bfloat162float(P[(size_t)r * PW + C + hv * DV + t]);
    const float o = bf(yn * sigmoidf_(z));
    out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(o);
    const float gs = warp_sum(o);
    if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + warp] = gs;
}

// ---- sparse attention (QSA): each row's blocks ----------------------------------------------------------------------

// attention._select's order-preserving key of a block score: negative floats flipped, positive ones topped.
__device__ __forceinline__ uint32_t fn_qsa_key(float v) {
    const uint32_t bits = __float_as_uint(v);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// An inclusive block-wide sum of one int a thread (blockDim.x a multiple of 32, at most 1024); *total gets the sum.
__device__ __forceinline__ int fn_block_scan(int v, int* warp_sums, int* total) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    for (int o = 1; o < 32; o <<= 1) {
        const int n = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += n;
    }
    if (lane == 31) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        int w = lane < warps ? warp_sums[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            const int n = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += n;
        }
        if (lane < warps) warp_sums[lane] = w;
    }
    __syncthreads();
    if (warp > 0) v += warp_sums[warp - 1];
    *total = warp_sums[warps - 1];
    __syncthreads();
    return v;
}

// A radix pass's cut by one warp (8 bins a lane): the highest bin whose suffix sum reaches need, and how many it gives.
__device__ __forceinline__ void fn_qsa_bin_cut(const int* hist, int need, uint32_t cut, int shift, uint32_t* cut_s, int* need_s) {
    const int lane = threadIdx.x & 31;
    int c[8], local = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        c[j] = hist[lane * 8 + j];
        local += c[j];
    }
    int above = local;  // the bins of this lane and every lane above it
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_down_sync(0xffffffffu, above, o);
        if (lane + o < 32) above += v;
    }
    above -= local;
    int best = -1, rest = 0, s = above;  // s = S[d + 1] as d walks down the lane's bins
#pragma unroll
    for (int j = 7; j >= 0; --j) {
        const int d = lane * 8 + j;
        if (best < 0 && d >= 1 && s + c[j] >= need) {
            best = d;
            rest = need - s;
        }
        s += c[j];
    }
    const int top_d = __reduce_max_sync(0xffffffffu, best);
    if (top_d < 0) {  // no bin from 1 up reaches it: bin 0, need - S[1]
        const int s1 = __shfl_sync(0xffffffffu, s - c[0], 0);
        if (lane == 0) {
            *need_s = need - s1;
            *cut_s = cut;
        }
    } else if (best == top_d) {
        *need_s = rest;
        *cut_s = cut | (static_cast<uint32_t>(top_d) << shift);
    }
}

// attention._select and _select_tiles, a row a block: a row past `top` complete blocks lists the best, then its tail.
extern "C" __global__ void __launch_bounds__(1024) fn_qsa_select(const float* __restrict__ sc,
                                                                 const int* __restrict__ pos0, int* __restrict__ ids,
                                                                 int* __restrict__ nkr, int* __restrict__ spr, int nb,
                                                                 int idw, int top, int ratio) {
    __shared__ int hist[256];
    __shared__ int warp_sums[32];
    __shared__ uint32_t cut_s;
    __shared__ int need_s;
    const int r = blockIdx.x;
    const int end = *pos0 + r + 1;
    const int complete = end / ratio;
    if (complete <= top) {
        if (threadIdx.x == 0) {
            nkr[r] = end;
            spr[r] = 0;
        }
        return;
    }
    const float* row = sc + static_cast<long long>(r) * nb;
    int* out = ids + static_cast<long long>(r) * idw;
    uint32_t cut = 0;
    int need = top;
    const int lane = threadIdx.x & 31;
    for (int p = 0; p < 4; ++p) {
        const int shift = 24 - 8 * p;
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        constexpr int U = 8;  // a thread's keys in flight before their adds (one L2 wait a key was the pass's time)
        for (int b0 = 0; b0 < complete; b0 += U * blockDim.x) {
            float v[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int b = b0 + u * blockDim.x + threadIdx.x;
                v[u] = b < complete ? row[b] : 0.0f;
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int b = b0 + u * blockDim.x + threadIdx.x;
                const uint32_t k = fn_qsa_key(v[u]);
                if (b < complete && (p == 0 || (k >> (shift + 8)) == (cut >> (shift + 8)))) atomicAdd(&hist[(k >> shift) & 0xFF], 1);
            }
        }
        __syncthreads();
        if (threadIdx.x < 32) fn_qsa_bin_cut(hist, need, cut, shift, &cut_s, &need_s);
        __syncthreads();
        cut = cut_s;
        need = need_s;
        __syncthreads();
    }
    // keys above the cut, then the first `need` equal ones in block order: a warp a run, ballots and two block scans
    const int warps = static_cast<int>(blockDim.x) >> 5, warp = threadIdx.x >> 5;
    const int steps = (complete + 32 * warps - 1) / (32 * warps);
    const int w0 = warp * steps * 32;
    const unsigned below = (1u << lane) - 1u;
    int na = 0, ne = 0;
    for (int st = 0; st < steps; ++st) {
        const int b = w0 + st * 32 + lane;
        const uint32_t k = b < complete ? fn_qsa_key(row[b]) : 0u;
        na += __popc(__ballot_sync(0xffffffffu, b < complete && k > cut));
        ne += __popc(__ballot_sync(0xffffffffu, b < complete && k == cut));
    }
    int total = 0;
    const int eq_before = __shfl_sync(0xffffffffu, fn_block_scan(lane == 0 ? ne : 0, warp_sums, &total), 0) - ne;
    const int take = max(0, min(ne, need - eq_before));
    const int mine = na + take;
    int place = __shfl_sync(0xffffffffu, fn_block_scan(lane == 0 ? mine : 0, warp_sums, &total), 0) - mine;
    int eq_seen = 0;
    for (int st = 0; st < steps; ++st) {
        const int b = w0 + st * 32 + lane;
        const uint32_t k = b < complete ? fn_qsa_key(row[b]) : 0u;
        const unsigned eq = __ballot_sync(0xffffffffu, b < complete && k == cut);
        const bool chosen = b < complete && (k > cut || (k == cut && eq_seen + __popc(eq & below) < take));
        const unsigned ch = __ballot_sync(0xffffffffu, chosen);
        if (chosen) {
            const int at = place + __popc(ch & below);
            for (int j = 0; j < ratio; ++j) out[at * ratio + j] = b * ratio + j;
        }
        place += __popc(ch);
        eq_seen += __popc(eq);
    }
    if (threadIdx.x < ratio) {
        const int tail = ratio * complete + threadIdx.x;
        if (tail < end) out[top * ratio + threadIdx.x] = tail;
    }
    if (threadIdx.x == 0) {
        nkr[r] = top * ratio + end - ratio * complete;
        spr[r] = 1;
    }
}

// Training or a live pocket requests the ordered FP32 expert sum.
extern "C" __global__ void fn_moe_sum(const float* __restrict__ y, const float* __restrict__ wts, float* __restrict__ out,
                                      int d, int slots) {
    const int r = blockIdx.x, col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= d) return;
    float acc = 0.0f;
    for (int k = 0; k < slots; ++k)
        acc = __fmaf_rn(y[(static_cast<long long>(r) * slots + k) * d + col], wts[r * slots + k], acc);
    out[static_cast<long long>(r) * d + col] = acc;
}

// dst [rows, width] (rows dst_pitch apart) = the columns src [rows, width] (src_pitch apart), 16-bit values.
extern "C" __global__ void fn_copy_cols(const uint16_t* __restrict__ src, long long src_pitch, uint16_t* __restrict__ dst,
                                        long long dst_pitch, int width) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (col < width) dst[r * dst_pitch + col] = src[r * src_pitch + col];
}

#include "flashnext_gdn_tpr.cuh"

// The prompt chain on two threads a state row (gdn_tpr): fn_gdn_rows_to's bits, state_in == state_out in place.
extern "C" __global__ void __launch_bounds__(256) fn_gdn_chain_tpr(const float* qn, const float* kn, const float* v,
                                                                 const float* gb, const float* state_in, float* state_out,
                                                                 int rows, float* ys_out) {
    gdn_tpr::chain<2, 16>(qn, kn, v, gb, state_in, state_out, rows, ys_out);
}

// The prompt GDN block in one kernel a value head (front, chain, gated norm; their bits), no fp32 rows through memory.
#include "flashnext_gdn_fused.cuh"
