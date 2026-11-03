// Owned learner algebra with generated Metal bindings; regenerate with tools/zig/metal_learning_sources.py.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bf16;
struct Heads {
    int rows, heads, kv_heads, dim, qkv;
    float scale;
    int prefix;
};

#define ATTN_KEYS 2048
namespace tf_metal_learn {
struct Context {
  uint3 threadIdx, blockIdx, blockDim;
  threadgroup float* shared;
  inline float tf_log1p(float x) { return metal::log(1.0f + x); }
  inline float warp_sum(float v) {
    for (ushort mask = 16; mask > 0; mask >>= 1) v += simd_shuffle_xor(v, mask);
    return v;
  }
  inline float warp_max(float v) {
    for (ushort mask = 16; mask > 0; mask >>= 1) v = max(v, simd_shuffle_xor(v, mask));
    return v;
  }
  inline float block_sum_tree(float v, threadgroup float* scratch, int warps) {
    const uint lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) scratch[warp] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = warp == 0 && lane < uint(warps) ? scratch[lane] : 0.0f;
    if (warp == 0) total = warp_sum(total);
    if (threadIdx.x == 0) scratch[0] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
  }
  inline float block_max_tree(float v, threadgroup float* scratch, int warps) {
    const uint lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) scratch[warp] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = warp == 0 && lane < uint(warps) ? scratch[lane] : -INFINITY;
    if (warp == 0) total = warp_max(total);
    if (threadIdx.x == 0) scratch[0] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
  }
inline float f32(bf16 v) { return float(v); }
inline bf16 b16(float v) { return bfloat(v); }

inline float sigmoidf_(float x) { return 1.0f / (1.0f + precise::exp(-x)); }

inline void rope_cs(int pos, const device float* inv, int i, thread float& c, thread float& s) {
    const float ang = (float)pos * inv[i];
    c = precise::cos(ang), s = precise::sin(ang);
}
  void tf_attn_q(
    const device bf16* qkv, const device float* dout, device float* probs, device float* dscores, device float* dq, Heads a, const device bf16* pk,
    const device bf16* pv) {

    threadgroup float* qi = shared + 0;
    threadgroup float* gi = shared + 256;
    threadgroup float* sc = shared + 512;
    threadgroup float* dp = shared + 2560;
    threadgroup float* scratch = shared + 4608;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, h = blockIdx.x, i = blockIdx.y;
    const int kv = h / (a.heads / a.kv_heads), qw = a.heads * a.dim, kw = a.kv_heads * a.dim;
    const int keys = a.prefix + a.rows, at = a.prefix + i;
    const device bf16* k = qkv + qw;
    const device bf16* v = qkv + qw + kw;
    for (int d = t; d < a.dim; d += 256) {
        qi[d] = f32(qkv[(size_t)i * a.qkv + h * a.dim + d]);
        gi[d] = dout[(size_t)i * qw + h * a.dim + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = w; j <= at; j += 8) {
        const device bf16* kr = (j < a.prefix ? pk + (size_t)j * kw : k + (size_t)(j - a.prefix) * a.qkv) + kv * a.dim;
        const device bf16* vr = (j < a.prefix ? pv + (size_t)j * kw : v + (size_t)(j - a.prefix) * a.qkv) + kv * a.dim;
        float s = 0, g = 0;
        for (int d = lane; d < a.dim; d += 32) {
            s += qi[d] * f32(kr[d]);
            g += gi[d] * f32(vr[d]);
        }
        s = warp_sum(s);
        g = warp_sum(g);
        if (lane == 0) sc[j] = s * a.scale, dp[j] = g;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float top = -INFINITY;
    for (int j = t; j <= at; j += 256) top = max(top, sc[j]);
    top = block_max_tree(top, scratch, 8);
    float total = 0;
    for (int j = t; j <= at; j += 256) {
        const float e = precise::exp(sc[j] - top);
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = t; d < a.dim; d += 256) {
        float acc = 0;
        for (int j = 0; j <= at; j++) {
            const device bf16* kr = j < a.prefix ? pk + (size_t)j * kw : k + (size_t)(j - a.prefix) * a.qkv;
            acc += dp[j] * f32(kr[kv * a.dim + d]);
        }
        dq[(size_t)i * qw + h * a.dim + d] = acc * a.scale;
    }

  }
  void tf_attn_gate_back(
    const device float* dgated, const device bf16* gated, const device bf16* pq, device float* dout, device float* dpq, int heads, int dim) {

    const int c = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y, qw = heads * dim;
    if (c >= qw) return;
    const int h = c / dim, d = c % dim;
    const size_t at = (size_t)r * 2 * qw + h * 2 * dim + dim + d;
    const float sg = sigmoidf_(f32(pq[at])), g = dgated[(size_t)r * qw + c];
    dout[(size_t)r * qw + c] = g * sg;
    dpq[at] = 0.0f;

  }
  void tf_attn_prep_back(
    const device bf16* pq, const device bf16* pk, const device float* qscale, const device float* kscale, const device float* inv, int pos0,
    const device float* dq, const device float* dk, device float* dpq, device float* dpk, int heads, int kv_heads, int dim, int rope_half, float eps) {

    threadgroup float* go = shared + 0;
    threadgroup float* scratch = shared + 256;
    const int hd = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
    const int qw = heads * dim, kw = kv_heads * dim;
    const bool q = hd < heads;
    const float x = q ? f32(pq[(size_t)r * 2 * qw + hd * 2 * dim + d]) : f32(pk[(size_t)r * kw + (hd - heads) * dim + d]);
    go[d] = q ? dq[(size_t)r * qw + hd * dim + d] : dk[(size_t)r * kw + (hd - heads) * dim + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gn = go[d];
    if (d < 2 * rope_half) {
        float c, s;
        rope_cs(pos0 + r, inv, d % rope_half, c, s);
        gn = d < rope_half ? go[d] * c + go[d + rope_half] * s : go[d] * c - go[d - rope_half] * s;
    }
    const float w = (q ? qscale : kscale)[d];
    const float sq = block_sum_tree(x * x, scratch, dim / 32), dot = block_sum_tree(w * gn * x, scratch, dim / 32);
    const float rinv = 1.0f / precise::sqrt(sq / (float)dim + eps);
    const float g = rinv * w * gn - rinv * rinv * rinv * x * dot / (float)dim;
    if (q)
        dpq[(size_t)r * 2 * qw + hd * 2 * dim + d] = g;
    else
        dpk[(size_t)r * kw + (hd - heads) * dim + d] = g;

  }
};
}
