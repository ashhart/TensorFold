// Living Weights' gated attention backward.

#include <cuda_bf16.h>
#include <stdint.h>
#include "common/reduce.cuh"

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float f32(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 b16(float v) { return __float2bfloat16(v); }

using tf_common::warp_sum;
using tf_common::warp_max;
using tf_common::block_sum_tree;
using tf_common::block_max_tree;

__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }

// ---- the attention layer --------------------------------------------------------------------------------------------

// Attention's shape (train.cu's Heads): rows, heads, kv heads, head dim, q|k|v row width, scale, prefix keys.
struct Heads {
    int rows, heads, kv_heads, dim, qkv;
    float scale;
    int prefix;
};

// The rotate-half RoPE angle of pair i at position pos.
__device__ __forceinline__ void rope_cs(int pos, const float* inv, int i, float& c, float& s) {
    const float ang = (float)pos * inv[i];
    c = cosf(ang), s = sinf(ang);
}

// The attention's prepared heads as glue.attn_prep writes them, into qkv [rows, (heads + 2 kv) dim].
extern "C" __global__ void __launch_bounds__(256) tf_attn_prep(const bf16* pq, const bf16* pk, const bf16* pv,
                                                                 const float* qscale, const float* kscale, const float* inv,
                                                                 int pos0, bf16* qkv, int heads, int kv_heads, int dim, int half,
                                                                 float eps) {
    __shared__ float xn[256], scratch[32];
    const int hd = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
    const int qw = heads * dim, kw = kv_heads * dim, width = qw + 2 * kw;
    if (hd >= heads + kv_heads) {
        const int k = hd - heads - kv_heads;
        qkv[(size_t)r * width + qw + kw + k * dim + d] = pv[(size_t)r * kw + k * dim + d];
        return;
    }
    const bool q = hd < heads;
    const float x = q ? f32(pq[(size_t)r * 2 * qw + hd * 2 * dim + d]) : f32(pk[(size_t)r * kw + (hd - heads) * dim + d]);
    const float* w = q ? qscale : kscale;
    const float rinv = 1.0f / sqrtf(block_sum_tree(x * x, scratch, dim / 32) / (float)dim + eps);
    xn[d] = f32(b16(x * rinv * w[d]));
    __syncthreads();
    float out = xn[d];
    if (d < 2 * half) {
        float c, s;
        rope_cs(pos0 + r, inv, d % half, c, s);
        out = d < half ? xn[d] * c - xn[d + half] * s : xn[d - half] * s + xn[d] * c;
    }
    qkv[(size_t)r * width + (q ? hd * dim : qw + (hd - heads) * dim) + d] = b16(out);
}

#define ATTN_KEYS 2048

// tf_train_attn_q for up to ATTN_KEYS keys: one query row i (position prefix + i) of head h.
extern "C" __global__ void __launch_bounds__(256) tf_attn_q(const bf16* qkv, const float* dout, float* probs, float* dscores,
                                                              float* dq, Heads a, const bf16* pk, const bf16* pv) {
    __shared__ float qi[256], gi[256], sc[ATTN_KEYS], dp[ATTN_KEYS], scratch[32];
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, h = blockIdx.x, i = blockIdx.y;
    const int kv = h / (a.heads / a.kv_heads), qw = a.heads * a.dim, kw = a.kv_heads * a.dim;
    const int keys = a.prefix + a.rows, at = a.prefix + i;
    const bf16* k = qkv + qw;
    const bf16* v = qkv + qw + kw;
    for (int d = t; d < a.dim; d += 256) {
        qi[d] = f32(qkv[(size_t)i * a.qkv + h * a.dim + d]);
        gi[d] = dout[(size_t)i * qw + h * a.dim + d];
    }
    __syncthreads();
    for (int j = w; j <= at; j += 8) {
        const bf16* kr = (j < a.prefix ? pk + (size_t)j * kw : k + (size_t)(j - a.prefix) * a.qkv) + kv * a.dim;
        const bf16* vr = (j < a.prefix ? pv + (size_t)j * kw : v + (size_t)(j - a.prefix) * a.qkv) + kv * a.dim;
        float s = 0, g = 0;
        for (int d = lane; d < a.dim; d += 32) {
            s += qi[d] * f32(kr[d]);
            g += gi[d] * f32(vr[d]);
        }
        s = warp_sum(s);
        g = warp_sum(g);
        if (lane == 0) sc[j] = s * a.scale, dp[j] = g;
    }
    __syncthreads();
    float top = -INFINITY;
    for (int j = t; j <= at; j += 256) top = fmaxf(top, sc[j]);
    top = block_max_tree(top, scratch, 8);
    float total = 0;
    for (int j = t; j <= at; j += 256) {
        const float e = expf(sc[j] - top);
        sc[j] = e;
        total += e;
    }
    total = block_sum_tree(total, scratch, 8);
    float mix = 0;
    for (int j = t; j <= at; j += 256) {
        sc[j] /= total;
        mix += sc[j] * dp[j];
    }
    mix = block_sum_tree(mix, scratch, 8);
    const size_t row = ((size_t)h * a.rows + i) * keys;
    for (int j = t; j < keys; j += 256) {
        const bool live = j <= at;
        const float p = live ? sc[j] : 0.0f, g = live ? p * (dp[j] - mix) : 0.0f;
        if (live) dp[j] = g;
        probs[row + j] = p;
        dscores[row + j] = g;
    }
    __syncthreads();
    for (int d = t; d < a.dim; d += 256) {
        float acc = 0;
        for (int j = 0; j <= at; j++) {
            const bf16* kr = j < a.prefix ? pk + (size_t)j * kw : k + (size_t)(j - a.prefix) * a.qkv;
            acc += dp[j] * f32(kr[kv * a.dim + d]);
        }
        dq[(size_t)i * qw + h * a.dim + d] = acc * a.scale;
    }
}

// The output gate gated = o sigmoid(gate) undone from dgated [rows, heads dim].
extern "C" __global__ void tf_attn_gate_back(const float* dgated, const bf16* gated, const bf16* pq, float* dout, float* dpq,
                                                int heads, int dim) {
    const int c = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y, qw = heads * dim;
    if (c >= qw) return;
    const int h = c / dim, d = c % dim;
    const size_t at = (size_t)r * 2 * qw + h * 2 * dim + dim + d;
    const float sg = sigm(f32(pq[at])), g = dgated[(size_t)r * qw + c];
    dout[(size_t)r * qw + c] = g * sg;
    dpq[at] = g * f32(gated[(size_t)r * qw + c]) * (1.0f - sg);
}

// tf_attn_prep undone for the q and k heads: dq [rows, heads dim] and dk [rows.
extern "C" __global__ void __launch_bounds__(256) tf_attn_prep_back(const bf16* pq, const bf16* pk, const float* qscale,
                                                                      const float* kscale, const float* inv, int pos0,
                                                                      const float* dq, const float* dk, float* dpq, float* dpk,
                                                                      int heads, int kv_heads, int dim, int half, float eps) {
    __shared__ float go[256], scratch[32];
    const int hd = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
    const int qw = heads * dim, kw = kv_heads * dim;
    const bool q = hd < heads;
    const float x = q ? f32(pq[(size_t)r * 2 * qw + hd * 2 * dim + d]) : f32(pk[(size_t)r * kw + (hd - heads) * dim + d]);
    go[d] = q ? dq[(size_t)r * qw + hd * dim + d] : dk[(size_t)r * kw + (hd - heads) * dim + d];
    __syncthreads();
    float gn = go[d];
    if (d < 2 * half) {
        float c, s;
        rope_cs(pos0 + r, inv, d % half, c, s);
        gn = d < half ? go[d] * c + go[d + half] * s : go[d] * c - go[d - half] * s;
    }
    const float w = (q ? qscale : kscale)[d];
    const float sq = block_sum_tree(x * x, scratch, dim / 32), dot = block_sum_tree(w * gn * x, scratch, dim / 32);
    const float rinv = 1.0f / sqrtf(sq / (float)dim + eps);
    const float g = rinv * w * gn - rinv * rinv * rinv * x * dot / (float)dim;
    if (q)
        dpq[(size_t)r * 2 * qw + hd * 2 * dim + d] = g;
    else
        dpk[(size_t)r * kw + (hd - heads) * dim + d] = g;
}


// ---- shared by the learners' products and nudges ------------------------------------------------------------------

// C [M, N] (rows ldc apart) (+)= the split products' partials [splits, M, N], summed in split order.
extern "C" __global__ void tf_sum_parts(const float* parts, int splits, int M, int N, float* C, int ldc, int accumulate) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)M * N) return;
    float s = 0;
    for (int z = 0; z < splits; z++) s += parts[(size_t)z * M * N + i];
    float* out = C + (size_t)(i / N) * ldc + i % N;
    *out = accumulate ? *out + s : s;
}

// y[i] += x[i] for n floats: a nudge onto its row of a layer's output.
extern "C" __global__ void tf_axpy(float* y, const float* x, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) y[i] += x[i];
}
