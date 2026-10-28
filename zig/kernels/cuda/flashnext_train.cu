// Flash Next's own input gradients for Living Weights: the hyper-connections, and the pocket's SwiGLU neurons.

// The pocket sits beside the shared expert, behind its gate; fp32 gradients from bf16 kept activations, as train.cu.

#include <cuda_bf16.h>
#include <stdint.h>
#include "common/reduce.cuh"
#include "common/act.cuh"

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float f32(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 b16(float v) { return __float2bfloat16(v); }

using tf_common::warp_sum;
using tf_common::block_sum_tree;

using tf_common::sigmoidf_;
using tf_common::silu;

using tf_common::silu_grad;

// ---- the pocket -----------------------------------------------------------------------------------------------------

#define POCKET_MAX 128

// Neuron j's dot products on row y: z = (gate_j . y, up_j . y) and, when g is set, e = down_j . g.
__device__ __forceinline__ void pocket_dots(const bf16* y, const float* g, int d, const float* gate, const float* up,
                                            const float* down, int j, float& zg, float& zu, float& e) {
    const int lane = threadIdx.x & 31;
    const float* gj = gate + (size_t)j * d;
    const float* uj = up + (size_t)j * d;
    const float* dj = down + (size_t)j * d;
    float a = 0, b = 0, c = 0;
    for (int i = lane; i < d; i += 32) {
        const float v = f32(y[i]);
        a += gj[i] * v;
        b += uj[i] * v;
        if (g) c += dj[i] * g[i];
    }
    zg = warp_sum(a), zu = warp_sum(b), e = g ? warp_sum(c) : 0.0f;
}

// Neuron j's scale on row r: the row's own (rs[r], a pass's sequence scale) from neuron `from` on, else 1.
__device__ __forceinline__ float pocket_scale(const float* rs, int from, int j, int r) {
    return (rs != nullptr && j >= from) ? rs[r] : 1.0f;
}

// out[r] (+)= s_r sum_j f_j silu(gate_j . y_r) (up_j . y_r) down_j, s_r the shared gate: the reference the pair equals.
extern "C" __global__ void __launch_bounds__(256) tf_fn_pocket_rows(const bf16* y, int d, const float* gate, const float* up,
                                                                   const float* down, int n, const float* wts, int slots, int at,
                                                                   void* out, int out_f32, const float* rs, int from) {
    __shared__ float act[POCKET_MAX];
    const int r = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const bf16* yr = y + (size_t)r * d;
    const float s = wts ? wts[(size_t)r * slots + at] : 1.0f;
    for (int j = w; j < n; j += 8) {
        float zg, zu, e;
        pocket_dots(yr, nullptr, d, gate, up, down, j, zg, zu, e);
        if (lane == 0) act[j] = s * pocket_scale(rs, from, j, r) * silu(zg) * zu;
    }
    __syncthreads();
    for (int k = threadIdx.x; k < d; k += 256) {
        float v = 0;
        for (int j = 0; j < n; j++) v += act[j] * down[(size_t)j * d + k];
        if (out_f32) {
            static_cast<float*>(out)[(size_t)r * d + k] += v;
        } else {
            bf16* o = static_cast<bf16*>(out) + (size_t)r * d + k;
            *o = b16(f32(*o) + v);
        }
    }
}

// tf_fn_pocket_rows' activations into act [rows, POCKET_MAX]: a warp NPW neurons by R rows, pocket_dots' order.
template <int NPW, int R, int U>
__device__ __forceinline__ void pocket_in(const bf16* y, int d, const float* gate, const float* up, int n,
                                          const float* wts, int slots, int at, float* act, int rows, const float* rs,
                                          int from) {
    const int lane = threadIdx.x & 31;
    const int j0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * NPW;
    if (j0 >= n) return;
    const int r0 = blockIdx.y * R;
    const int tile = min(R, rows - r0);
    const float* gj[NPW];
    const float* uj[NPW];
    const bf16* yr[R];
#pragma unroll
    for (int q = 0; q < NPW; ++q) {
        gj[q] = gate + (size_t)min(j0 + q, n - 1) * d;
        uj[q] = up + (size_t)min(j0 + q, n - 1) * d;
    }
#pragma unroll
    for (int t = 0; t < R; ++t) yr[t] = y + (size_t)min(r0 + t, rows - 1) * d;
    float a[NPW][R], b[NPW][R];
#pragma unroll
    for (int q = 0; q < NPW; ++q)
#pragma unroll
        for (int t = 0; t < R; ++t) a[q][t] = 0.0f, b[q][t] = 0.0f;
#pragma unroll U
    for (int i = lane; i < d; i += 32) {
        float g[NPW], u[NPW], x[R];
#pragma unroll
        for (int q = 0; q < NPW; ++q) g[q] = gj[q][i], u[q] = uj[q][i];
#pragma unroll
        for (int t = 0; t < R; ++t) x[t] = f32(yr[t][i]);
#pragma unroll
        for (int q = 0; q < NPW; ++q)
#pragma unroll
            for (int t = 0; t < R; ++t) a[q][t] += g[q] * x[t], b[q][t] += u[q] * x[t];
    }
#pragma unroll
    for (int q = 0; q < NPW; ++q) {
#pragma unroll
        for (int t = 0; t < R; ++t) {
            const float zg = warp_sum(a[q][t]), zu = warp_sum(b[q][t]);
            const int j = j0 + q, r = r0 + t;
            if (lane == 0 && j < n && t < tile) {
                const float s = wts ? wts[(size_t)r * slots + at] : 1.0f;
                act[(size_t)r * POCKET_MAX + j] = s * pocket_scale(rs, from, j, r) * silu(zg) * zu;
            }
        }
    }
}

// out (bf16, or fp32 if out_f32) [rows, d] += act down: a thread an output over R rows, neurons in order, B at a time.
template <int R, int B>
__device__ __forceinline__ void pocket_out(const float* act, const float* down, int n, int d, void* out, int out_f32,
                                           int rows) {
    if (n == 0) return;  // no neuron in use: the output keeps its bits
    __shared__ float a[R * POCKET_MAX];
    const int r0 = blockIdx.y * R;
    const int tile = min(R, rows - r0);
    for (int i = threadIdx.x; i < tile * POCKET_MAX; i += blockDim.x)
        if (i % POCKET_MAX < n) a[i] = act[(size_t)r0 * POCKET_MAX + i];
    __syncthreads();
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= d) return;
    float s[R];
#pragma unroll
    for (int t = 0; t < R; ++t) s[t] = 0.0f;
    for (int j0 = 0; j0 < n; j0 += B) {
        float w[B];
#pragma unroll
        for (int c = 0; c < B; ++c) w[c] = j0 + c < n ? down[(size_t)(j0 + c) * d + k] : 0.0f;
#pragma unroll
        for (int t = 0; t < R; ++t) {
            if (t < tile) {
#pragma unroll
                for (int c = 0; c < B; ++c)
                    if (j0 + c < n) s[t] += a[t * POCKET_MAX + j0 + c] * w[c];
            }
        }
    }
#pragma unroll
    for (int t = 0; t < R; ++t) {
        if (t < tile) {
            const size_t o = (size_t)(r0 + t) * d + k;
            if (out_f32) {
                static_cast<float*>(out)[o] += s[t];
            } else {
                bf16* p = static_cast<bf16*>(out) + o;
                *p = b16(f32(*p) + s[t]);
            }
        }
    }
}

// tf_fn_pocket_rows' first half: a warp a neuron, a block row a row; grid (n / 4 up, rows), block 128.
extern "C" __global__ void __launch_bounds__(128) tf_fn_pocket_in(const bf16* y, int d, const float* gate,
                                                                 const float* up, int n, const float* wts, int slots,
                                                                 int at, float* act, int rows, const float* rs,
                                                                 int from) {
    pocket_in<1, 1, 4>(y, d, gate, up, n, wts, slots, at, act, rows, rs, from);
}

// The same for many rows: a warp two neurons by four rows; grid (n / 16 up, rows / 4 up), block 256.
extern "C" __global__ void __launch_bounds__(256) tf_fn_pocket_in_rows(const bf16* y, int d, const float* gate,
                                                                      const float* up, int n, const float* wts,
                                                                      int slots, int at, float* act, int rows,
                                                                      const float* rs, int from) {
    pocket_in<2, 4, 2>(y, d, gate, up, n, wts, slots, at, act, rows, rs, from);
}

// tf_fn_pocket_rows' second half: a thread an output, a block row a row; grid (d / 128 up, rows), block 128.
extern "C" __global__ void __launch_bounds__(128) tf_fn_pocket_out(const float* act, const float* down, int n, int d,
                                                                  void* out, int out_f32, int rows) {
    pocket_out<1, 16>(act, down, n, d, out, out_f32, rows);
}

// The same for many rows: a thread an output over four rows; grid (d / 128 up, rows / 4 up), block 128.
extern "C" __global__ void __launch_bounds__(128) tf_fn_pocket_out_rows(const float* act, const float* down, int n,
                                                                       int d, void* out, int out_f32, int rows) {
    pocket_out<4, 16>(act, down, n, d, out, out_f32, rows);
}

// The pocket undone on row r from g: dx[r] += its input gradient, and the shared slot's a[r slots + at] += its part.
extern "C" __global__ void __launch_bounds__(256) tf_fn_pocket_back(const bf16* y, int d, const float* gate, const float* up,
                                                                   const float* down, int n, const float* wts, int slots, int at,
                                                                   const float* g, float* dx, float* a, const float* rs, int from) {
    __shared__ float cg[POCKET_MAX], cu[POCKET_MAX], ca[POCKET_MAX];
    const int r = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const bf16* yr = y + (size_t)r * d;
    const float* gr = g + (size_t)r * d;
    const float s = wts[(size_t)r * slots + at];
    for (int j = w; j < n; j += 8) {
        float zg, zu, e;
        pocket_dots(yr, gr, d, gate, up, down, j, zg, zu, e);
        if (lane == 0) {
            const float f = pocket_scale(rs, from, j, r);
            cg[j] = s * f * e * silu_grad(zg) * zu;
            cu[j] = s * f * e * silu(zg);
            ca[j] = f * e * silu(zg) * zu;
        }
    }
    __syncthreads();
    for (int k = threadIdx.x; k < d; k += 256) {
        float v = 0;
        for (int j = 0; j < n; j++) v += cg[j] * gate[(size_t)j * d + k] + cu[j] * up[(size_t)j * d + k];
        dx[(size_t)r * d + k] += v;
    }
    if (threadIdx.x == 0) {
        float v = 0;
        for (int j = 0; j < n; j++) v += ca[j];
        a[(size_t)r * slots + at] += v;
    }
}

// grad[j] += w s act_j push[row_j]: neuron j's output vector's gradient at its row rows[j] (none when negative).
extern "C" __global__ void __launch_bounds__(256) tf_fn_pocket_grad(const bf16* y, int d, const float* gate, const float* up,
                                                                   int n, const float* wts, int slots, int at, const float* push,
                                                                   const int* rows, float w, float* grad) {
    __shared__ float scratch[32];
    const int j = blockIdx.x, t = threadIdx.x, r = rows[j];
    if (j >= n || r < 0) return;
    const bf16* yr = y + (size_t)r * d;
    float a = 0, b = 0;
    for (int i = t; i < d; i += 256) {
        const float v = f32(yr[i]);
        a += gate[(size_t)j * d + i] * v;
        b += up[(size_t)j * d + i] * v;
    }
    a = block_sum_tree(a, scratch, 8);
    b = block_sum_tree(b, scratch, 8);
    const float k = w * wts[(size_t)r * slots + at] * silu(a) * b;
    for (int i = t; i < d; i += 256) grad[(size_t)j * d + i] += k * push[(size_t)r * d + i];
}

// ---- the hyper-connections ------------------------------------------------------------------------------------------

// A block's write-back h' = h + bf16(branch inject_s) undone: dbranch = sum_s g_s inject_s, dinject_s = g_s . branch.
extern "C" __global__ void __launch_bounds__(256) tf_fn_hc_branch_back(const float* g, const bf16* inject, const bf16* branch,
                                                                      float* dbranch, float* dinject, int d, int streams) {
    __shared__ float scratch[32];
    const int r = blockIdx.x, t = threadIdx.x;
    const float* gr = g + (size_t)r * streams * d;
    float inj[8], part[8];
    for (int s = 0; s < streams; s++) inj[s] = f32(inject[r * streams + s]), part[s] = 0;
    for (int i = t; i < d; i += 256) {
        const float b = f32(branch[(size_t)r * d + i]);
        float v = 0;
        for (int s = 0; s < streams; s++) {
            const float gs = gr[(size_t)s * d + i];
            v += gs * inj[s];
            part[s] += gs * b;
        }
        dbranch[(size_t)r * d + i] = v;
    }
    for (int s = 0; s < streams; s++) {
        const float v = block_sum_tree(part[s], scratch, 8);
        if (t == 0) dinject[r * streams + s] = v;
    }
}

// The read-out's mix = mean_s sigmoid(up_s) normed_s undone from dmix: dnormed and dup [rows, S d].
extern "C" __global__ void tf_fn_hc_mix_back(const float* dmix, const bf16* up, const bf16* normed, float* dnormed, float* dup,
                                             int d, int streams) {
    const int i = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y;
    if (i >= d) return;
    const float g = dmix[(size_t)r * d + i] / (float)streams;
    for (int s = 0; s < streams; s++) {
        const size_t at = ((size_t)r * streams + s) * d + i;
        const float sg = sigmoidf_(f32(up[at]));
        dnormed[at] = g * sg;
        dup[at] = g * f32(normed[at]) * sg * (1.0f - sg);
    }
}

// The down projection's activations act = silu(v / S) and inject = 2 sigmoid(v / S) undone into dv.
extern "C" __global__ void __launch_bounds__(256) tf_fn_hc_act_back(const float* dact, const float* v, const float* dinject,
                                                                   float* dv, int low, int streams, int nv) {
    const int r = blockIdx.x;
    const float inv = 1.0f / (float)streams;
    for (int i = threadIdx.x; i < nv; i += 256) {
        const float x = v[(size_t)r * nv + i] * inv;
        float g = 0;
        if (i < low) {
            g = dact[(size_t)r * low + i] * silu_grad(x) * inv;
        } else if (dinject) {
            const float sg = sigmoidf_(x);
            g = dinject[r * streams + i - low] * 2.0f * sg * (1.0f - sg) * inv;
        }
        dv[(size_t)r * nv + i] = g;
    }
}

// g[r, s] += the stream's RMS norm (scale fp32 1 + w) undone from dnormed.
extern "C" __global__ void __launch_bounds__(256) tf_fn_hc_norm_back(const bf16* h, const float* scale, const float* dnormed,
                                                                    float* g, int d, int streams, float eps) {
    __shared__ float scratch[32];
    const int s = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
    const size_t base = ((size_t)r * streams + s) * d;
    const float* w = scale + (size_t)s * d;
    float sq = 0, dot = 0;
    for (int i = t; i < d; i += 256) {
        const float v = f32(h[base + i]);
        sq += v * v;
        dot += w[i] * dnormed[base + i] * v;
    }
    sq = block_sum_tree(sq, scratch, 8);
    dot = block_sum_tree(dot, scratch, 8);
    const float inv = rsqrtf(sq / (float)d + eps), k = inv * inv * inv * dot / (float)d;
    for (int i = t; i < d; i += 256) g[base + i] += inv * w[i] * dnormed[base + i] - k * f32(h[base + i]);
}
