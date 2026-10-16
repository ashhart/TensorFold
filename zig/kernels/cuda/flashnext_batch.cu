// The small device steps of Flash Next's batched prompt pass (cuda_batch.zig), built without FMA contraction.

#include <cuda_bf16.h>
#include <stdint.h>

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
extern "C" __global__ void fn_shift_window(__nv_bfloat16* __restrict__ win, const __nv_bfloat16* __restrict__ rows,
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

// ---- the DeltaNet chain over a chunk: flashnext_gdn.cu's chain cut into front, rows and norm, with its bits ----

namespace fn_gdn {
constexpr int NK = 16, NV = 48, DK = 128, DV = 128, TAPS = 4;
constexpr int C = 2 * NK * DK + NV * DV;
constexpr int PW = C + NV * DV + 2 * NV;

static __device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

static __device__ __forceinline__ float warp_sum(float x) {
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

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

// Program (row, unit), 256 threads: key head u's normed q and k (u < NK), else a value head's v, decay g and beta.
extern "C" __global__ void __launch_bounds__(256) fn_gdn_front(
        const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs, const __nv_bfloat16* __restrict__ cw,
        const float* __restrict__ a_log, const float* __restrict__ dt_bias, float* __restrict__ qn,
        float* __restrict__ kn, float* __restrict__ v, float* __restrict__ gb) {
    using namespace fn_gdn;
    const int r = blockIdx.x, u = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
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
            float* dst = (warp == 0 ? qn : kn) + ((size_t)r * NK + hk) * DK;
#pragma unroll
            for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = v4[i] * inv;
        }
    } else {
        const int hv = u - NK;
        if (t < DV) v[(size_t)r * NV * DV + hv * DV + t] = conv_act(P, cs, cw, r, 2 * NK * DK + hv * DV + t);
        else if (t == DV) {
            const float b = __bfloat162float(P[(size_t)r * PW + C + NV * DV + hv]);
            const float a = __bfloat162float(P[(size_t)r * PW + C + NV * DV + NV + hv]);
            gb[((size_t)r * NV + hv) * 2] = expf(-expf(a_log[hv]) * softplusf_(a + dt_bias[hv]));
            gb[((size_t)r * NV + hv) * 2 + 1] = bf(sigmoidf_(b));
        }
    }
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
    for (int p = 0; p < 4; ++p) {
        const int shift = 24 - 8 * p;
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int b = threadIdx.x; b < complete; b += blockDim.x) {
            const uint32_t k = fn_qsa_key(row[b]);
            if (p == 0 || (k >> (shift + 8)) == (cut >> (shift + 8))) atomicAdd(&hist[(k >> shift) & 0xFF], 1);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int above = 0;            // keys under the prefix with a larger byte
            int d = 255;
            for (; d > 0; --d) {
                if (above + hist[d] >= need) break;
                above += hist[d];
            }
            need_s = need - above;
            cut_s = cut | (static_cast<uint32_t>(d) << shift);
        }
        __syncthreads();
        cut = cut_s;
        need = need_s;
        __syncthreads();
    }
    // every key above the cut, then the first `need` keys equal to it, placed in block order
    int seen = 0;
    int placed = 0;
    for (int t0 = 0; t0 < complete && placed < top; t0 += blockDim.x) {
        const int b = t0 + threadIdx.x;
        const bool ok = b < complete;
        const uint32_t k = ok ? fn_qsa_key(row[b]) : 0u;
        const bool above = ok && k > cut;
        const bool equal = ok && k == cut;
        int eq_total = 0;
        const int rank = seen + fn_block_scan(equal ? 1 : 0, warp_sums, &eq_total);
        const bool chosen = above || (equal && rank <= need);
        int ch_total = 0;
        const int place = placed + fn_block_scan(chosen ? 1 : 0, warp_sums, &ch_total) - 1;
        if (chosen)
            for (int j = 0; j < ratio; ++j) out[place * ratio + j] = b * ratio + j;
        seen += eq_total;
        placed += ch_total;
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
