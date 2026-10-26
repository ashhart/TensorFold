// Muse Spark's glue (norms, gates, router, expert sums, head), exllamav3's roundings; built with the glue flags.

#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>

namespace {

constexpr int kMaxPer = 16;  // values a thread holds: rows up to 1024 * 16 wide

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets the total.
__device__ float block_sum(float v) {
    __shared__ float part[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float s = lane < nw ? part[lane] : 0.f;
        s = warp_sum(s);
        if (lane == 0) total = s;
    }
    __syncthreads();
    return total;
}

// block_sum over a cluster of 8 blocks of 128: the same warp sums read back in warp order, block_sum's bits.
__device__ __forceinline__ float cluster_sum(float v, float* part) {
    namespace cg = cooperative_groups;
    cg::cluster_group cl = cg::this_cluster();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) part[warp] = v;
    cl.sync();
    const float* rp = cl.map_shared_rank(part, lane >> 2);
    return warp_sum(rp[lane & 3]);
}

}  // namespace

// x = rms(embed[ids].float()) (Embedding then the weightless embed_norm, fp32 out). A block a row.
extern "C" __global__ void muse_embed_norm(const uint16_t* __restrict__ table, const int64_t* __restrict__ ids,
                                           float* __restrict__ out, int D, float eps) {
    const int64_t r = blockIdx.x;
    const uint16_t* row = table + ids[r] * (int64_t)D;
    float v[kMaxPer];
    float ss = 0.f;
    int n = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x, ++n) {
        v[n] = __uint_as_float(((uint32_t)row[d]) << 16);
        ss += v[n] * v[n];
    }
    const float rs = rsqrtf(block_sum(ss) / (float)D + eps);
    n = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x, ++n) out[r * D + d] = v[n] * rs;
}

// out = half(x * rsqrt(mean(x^2) + eps) * w); parts P: x holds the ranks' gathered column slices.
extern "C" __global__ void muse_rms(const void* __restrict__ x, int x_half, const float* __restrict__ w,
                                    __half* __restrict__ out, int D, float eps, int parts) {
    const int64_t r = blockIdx.x, rows = gridDim.x;
    float v[kMaxPer];
    float ss = 0.f;
    int n = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x, ++n) {
        const int64_t at = parts ? ((int64_t)(d / parts) * rows + r) * parts + d % parts : r * D + d;
        v[n] = x_half ? __half2float(((const __half*)x)[at]) : ((const float*)x)[at];
        ss += v[n] * v[n];
    }
    const float rs = rsqrtf(block_sum(ss) / (float)D + eps);
    n = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x, ++n) {
        float y = v[n] * rs;
        if (w) y = y * w[d];
        out[r * D + d] = __float2half_rn(y);
    }
}

// Sublayer end + next norm: res = res * alpha + rms(y) * beta; x = half(rms(res) * w_next), xf unrounded.
extern "C" __global__ void muse_gate_norm(float* __restrict__ res, const float* __restrict__ y,
                                          const float* __restrict__ y2, const float* __restrict__ scale,
                                          int round_bf16, float post_eps, const float* __restrict__ ab,
                                          const float* __restrict__ w_next, float next_eps, __half* __restrict__ x,
                                          float* __restrict__ xf, int D) {
    const int64_t r = blockIdx.x;
    float v[kMaxPer];
    float ss = 0.f;
    int n = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x, ++n) {
        float t = y[r * D + d];
        if (y2) t = t + y2[r * D + d];
        if (scale) {
            t = t * scale[d];
            if (round_bf16) t = bf16r(t);
        }
        v[n] = t;
        ss += t * t;
    }
    const float rs = rsqrtf(block_sum(ss) / (float)D + post_eps);
    float ss2 = 0.f;
    n = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x, ++n) {
        const float yn = v[n] * rs;
        const float h = res[r * D + d] * ab[d] + yn * ab[D + d];
        res[r * D + d] = h;
        v[n] = h;
        ss2 += h * h;
    }
    const float rs2 = rsqrtf(block_sum(ss2) / (float)D + next_eps);
    n = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x, ++n) {
        const float o = v[n] * rs2 * w_next[d];
        x[r * D + d] = __float2half_rn(o);
        if (xf) xf[r * D + d] = o;
    }
}

// muse_gate_norm on a cluster of 8 blocks of 128 threads a row: each thread's columns and sums, its bits.
extern "C" __global__ void __cluster_dims__(8, 1, 1) __launch_bounds__(128)
    muse_gate_norm_c(float* __restrict__ res, const float* __restrict__ y, const float* __restrict__ y2,
                     const float* __restrict__ scale, int round_bf16, float post_eps, const float* __restrict__ ab,
                     const float* __restrict__ w_next, float next_eps, __half* __restrict__ x,
                     float* __restrict__ xf, int D) {
    __shared__ float part[2][4];
    const int64_t r = blockIdx.x >> 3;
    const int t0 = cooperative_groups::this_cluster().block_rank() * 128 + threadIdx.x;
    float v[kMaxPer];
    float ss = 0.f;
    int n = 0;
    for (int d = t0; d < D; d += 1024, ++n) {
        float t = y[r * D + d];
        if (y2) t = t + y2[r * D + d];
        if (scale) {
            t = t * scale[d];
            if (round_bf16) t = bf16r(t);
        }
        v[n] = t;
        ss += t * t;
    }
    const float rs = rsqrtf(cluster_sum(ss, part[0]) / (float)D + post_eps);
    float ss2 = 0.f;
    n = 0;
    for (int d = t0; d < D; d += 1024, ++n) {
        const float yn = v[n] * rs;
        const float h = res[r * D + d] * ab[d] + yn * ab[D + d];
        res[r * D + d] = h;
        v[n] = h;
        ss2 += h * h;
    }
    const float rs2 = rsqrtf(cluster_sum(ss2, part[1]) / (float)D + next_eps);
    n = 0;
    for (int d = t0; d < D; d += 1024, ++n) {
        const float o = v[n] * rs2 * w_next[d];
        x[r * D + d] = __float2half_rn(o);
        if (xf) xf[r * D + d] = o;
    }
    cooperative_groups::this_cluster().sync();  // no block leaves while another reads its partials
}

// Router logits in fp32 (x . W[e], W bf16 or fp32): a warp an expert, a lane's columns d = lane + 32 j in order.
extern "C" __global__ void muse_router(const float* __restrict__ x, const void* __restrict__ W, int w_bf16,
                                       float* __restrict__ out, int D, int E) {
    const int r = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * (blockDim.x >> 5) + warp;
    if (e >= E) return;
    const float* xr = x + (int64_t)r * D;
    float acc = 0.f;
    if (w_bf16) {
        const uint16_t* wr = (const uint16_t*)W + (int64_t)e * D;
        int d = lane;
        for (; d + 224 < D; d += 256) {
            uint16_t wv[8];
            float xv[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                wv[j] = __ldg(wr + d + 32 * j);
                xv[j] = xr[d + 32 * j];
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) acc += xv[j] * __uint_as_float(((uint32_t)wv[j]) << 16);
        }
        for (; d < D; d += 32) acc += xr[d] * __uint_as_float(((uint32_t)wr[d]) << 16);
    } else {
        const float* wr = (const float*)W + (int64_t)e * D;
        for (int d = lane; d < D; d += 32) acc += xr[d] * wr[d];
    }
    acc = warp_sum(acc);
    if (lane == 0) out[(int64_t)r * E + e] = acc;
}

// s = sigmoid(logits * mult), top K of s + bias, weights s / (sum + 1e-15) * scale, picks in expert-id order.
extern "C" __global__ void muse_topk(const float* __restrict__ logits, const float* __restrict__ bias, int E, int K,
                                     float mult, float scale, int32_t* __restrict__ pick, float* __restrict__ wts) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    __shared__ int sel[32];
    __shared__ float sc[1024];
    const int r = blockIdx.x, t = threadIdx.x;
    float c = -CUDART_INF_F;
    if (t < E) {
        const float s = 1.f / (1.f + expf(-(logits[(int64_t)r * E + t] * mult)));
        sc[t] = s;
        c = s + bias[t];
    }
    for (int k = 0; k < K; ++k) {
        bv[t] = c;
        bi[t] = t;
        __syncthreads();
        for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
            if (t < o) {
                const float v = bv[t + o];
                const int i = bi[t + o];
                if (v > bv[t] || (v == bv[t] && i < bi[t])) {
                    bv[t] = v;
                    bi[t] = i;
                }
            }
            __syncthreads();
        }
        if (t == 0) sel[k] = bi[0];
        __syncthreads();
        if (t == sel[k]) c = -CUDART_INF_F;
        __syncthreads();
    }
    if (t == 0) {
        float sum = 0.f;
        for (int k = 0; k < K; ++k) sum += sc[sel[k]];
        const float den = sum + 1e-15f;
        int ord[32];
        for (int k = 0; k < K; ++k) ord[k] = sel[k];
        for (int i = 1; i < K; ++i) {
            const int v = ord[i];
            int j = i - 1;
            while (j >= 0 && ord[j] > v) {
                ord[j + 1] = ord[j];
                --j;
            }
            ord[j + 1] = v;
        }
        for (int k = 0; k < K; ++k) {
            pick[r * K + k] = ord[k];
            wts[r * K + k] = (sc[ord[k]] / den) * scale;
        }
    }
}

// Routed sum: per slot norm(bf16(y * (1 + w))) * weight, added in slot order; out = half(acc) columns.
extern "C" __global__ void muse_expert_sum(const float* __restrict__ y, const float* __restrict__ y2,
                                           const float* __restrict__ wts, const float* __restrict__ scale, float eps,
                                           __half* __restrict__ out, int L, int K, int out_lo, int out_w) {
    const int64_t r = blockIdx.x;
    float acc[kMaxPer];
    float v[kMaxPer];
    int n = 0;
    for (int d = threadIdx.x; d < L; d += blockDim.x, ++n) acc[n] = 0.f;
    for (int k = 0; k < K; ++k) {
        const int64_t p = r * K + k;
        float ss = 0.f;
        n = 0;
        for (int d = threadIdx.x; d < L; d += blockDim.x, ++n) {
            float t = y[p * L + d];
            if (y2) t = t + y2[p * L + d];
            t = bf16r(t * scale[d]);
            v[n] = t;
            ss += t * t;
        }
        const float rs = rsqrtf(block_sum(ss) / (float)L + eps);
        const float wk = wts[p];
        n = 0;
        for (int d = threadIdx.x; d < L; d += blockDim.x, ++n) acc[n] = acc[n] + (v[n] * rs) * wk;
    }
    n = 0;
    for (int d = threadIdx.x; d < L; d += blockDim.x, ++n)
        if (d >= out_lo && d < out_lo + out_w) out[r * out_w + d - out_lo] = __float2half_rn(acc[n]);
}

// muse_expert_sum on a cluster of 8 blocks of 128 threads a row, its bits (as muse_gate_norm_c); K <= 16.
extern "C" __global__ void __cluster_dims__(8, 1, 1) __launch_bounds__(128)
    muse_expert_sum_c(const float* __restrict__ y, const float* __restrict__ y2, const float* __restrict__ wts,
                      const float* __restrict__ scale, float eps, __half* __restrict__ out, int L, int K, int out_lo,
                      int out_w) {
    __shared__ float part[16][4];
    const int64_t r = blockIdx.x >> 3;
    const int t0 = cooperative_groups::this_cluster().block_rank() * 128 + threadIdx.x;
    float acc[kMaxPer];
    float v[kMaxPer];
    int n = 0;
    for (int d = t0; d < L; d += 1024, ++n) acc[n] = 0.f;
    for (int k = 0; k < K; ++k) {
        const int64_t p = r * K + k;
        float ss = 0.f;
        n = 0;
        for (int d = t0; d < L; d += 1024, ++n) {
            float t = y[p * L + d];
            if (y2) t = t + y2[p * L + d];
            t = bf16r(t * scale[d]);
            v[n] = t;
            ss += t * t;
        }
        const float rs = rsqrtf(cluster_sum(ss, part[k]) / (float)L + eps);
        const float wk = wts[p];
        n = 0;
        for (int d = t0; d < L; d += 1024, ++n) acc[n] = acc[n] + (v[n] * rs) * wk;
    }
    n = 0;
    for (int d = t0; d < L; d += 1024, ++n)
        if (d >= out_lo && d < out_lo + out_w) out[r * out_w + d - out_lo] = __float2half_rn(acc[n]);
    cooperative_groups::this_cluster().sync();
}

// The head: x = x * mult, then cap * tanh(x / cap) when cap > 0 (Linear pre_scale, ext.softcap), in place.
extern "C" __global__ void muse_softcap(float* __restrict__ x, int64_t n, float mult, float cap) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        float v = x[i] * mult;
        if (cap > 0.f) v = tanhf(v / cap) * cap;
        x[i] = v;
    }
}

// Argmax over the first `vocab` columns of both ranks' gathered head slices: the first maximum, NaN counts.
extern "C" __global__ void muse_argmax(const float* __restrict__ recv, int32_t* __restrict__ out, int w0, int w1,
                                       int64_t part, int vocab) {
    __shared__ float bv[256];
    __shared__ int bi[256];
    const int r = blockIdx.x, tid = threadIdx.x;
    float best = -CUDART_INF_F;
    int idx = 0x7fffffff;
    for (int c = tid; c < vocab; c += 256) {
        const float v = c < w0 ? recv[(int64_t)r * w0 + c] : recv[part + (int64_t)r * w1 + (c - w0)];
        const bool nan_v = v != v, nan_b = best != best;
        if (idx == 0x7fffffff || (nan_v && !nan_b) || (!nan_b && v > best)) {
            best = v;
            idx = c;
        }
    }
    bv[tid] = best;
    bi[tid] = idx;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (tid < o) {
            const float v = bv[tid + o], b = bv[tid];
            const int i = bi[tid + o];
            const bool nv = v != v, nb = b != b;
            const bool take = i != 0x7fffffff && (bi[tid] == 0x7fffffff || (nv && !nb) ||
                                                  (nv == nb && (nv ? i < bi[tid] : (v > b || (v == b && i < bi[tid])))));
            if (take) {
                bv[tid] = v;
                bi[tid] = i;
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[r] = bi[0];
}

// RoPE tables (NeoX, full head): cos / sin [context, dim / 2] of p * theta ** (-2j / dim), in fp64 then fp32.
extern "C" __global__ void muse_rope_tables(float* __restrict__ cos_out, float* __restrict__ sin_out, int context,
                                            double theta, int dim) {
    const int half = dim / 2;
    const int64_t n = (int64_t)context * half;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const int j = (int)(i % half);
        const int64_t p = i / half;
        const double inv = 1.0 / pow(theta, (double)(2 * j) / (double)dim);
        const double ang = (double)p * inv;
        cos_out[i] = __double2float_rn(cos(ang));
        sin_out[i] = __double2float_rn(sin(ang));
    }
}
