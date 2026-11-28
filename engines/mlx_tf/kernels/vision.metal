// Clean-room Vision Transformer (ViT) Metal 3.1+ kernels for:
//   - Qwen3.5-VL / Qwen3.8-VL (3D-collapsed patch embed, 2D bilinear pos_embed, 2D Vision RoPE,
//     LayerNorm + bidirectional MHA + GELU MLP, 2x2 spatial patch merger)
//   - Gemma 3-VL SigLIP (2D patch embed + learned pos_embed, LayerNorm + bidirectional MHA +
//     GELU MLP, post_layernorm, 2D AvgPool + Gemma3RMSNorm + multimodal linear projector)
#include <metal_stdlib>
using namespace metal;

#define UNROLL _Pragma("clang loop unroll(full)")

inline float fast_erf(float x) {
  const float sign = (x < 0.0f) ? -1.0f : 1.0f;
  const float a = fabs(x);
  if (a >= 6.0f) return sign;
  const float t = 1.0f / (1.0f + 0.3275911f * a);
  const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t) * exp(-a * a);
  return sign * y;
}

inline float gelu_exact(float x) {
  return 0.5f * x * (1.0f + fast_erf(x * 0.7071067811865475f));
}

inline float gelu_tanh(float x) {
  if (x >= 10.0f) return x;
  if (x <= -10.0f) return 0.0f;
  const float inner = clamp(0.7978845608028654f * (x + 0.044715f * x * x * x), -15.0f, 15.0f);
  return 0.5f * x * (1.0f + tanh(inner));
}

struct VitLnParams {
  uint rows;
  uint dim;
  float eps;
};

// Exact FP32 LayerNorm: out = ((x - mean) * rsqrt(var + eps)) * weight + bias -> bf16.
// Grid: threadgroups (rows, 1, 1) of 256 threads (8 simdgroups).
kernel void vit_layer_norm(const device bfloat* x [[buffer(0)]],
                           const device bfloat* weight [[buffer(1)]],
                           const device bfloat* bias [[buffer(2)]],
                           device bfloat* out [[buffer(3)]],
                           constant VitLnParams& p [[buffer(4)]],
                           uint row [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
  if (row >= p.rows) return;
  const device bfloat* xr = x + (size_t)row * p.dim;
  device bfloat* yr = out + (size_t)row * p.dim;

  float sum = 0.0f;
  for (uint d = tid; d < p.dim; d += 256) sum += float(xr[d]);
  sum = simd_sum(sum);
  threadgroup float sg_buf[8];
  if (lane == 0) sg_buf[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float tot = simd_sum(lane < 8 ? sg_buf[lane] : 0.0f);
    if (lane == 0) sg_buf[0] = tot / float(p.dim);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mean = sg_buf[0];

  float var = 0.0f;
  for (uint d = tid; d < p.dim; d += 256) {
    const float diff = float(xr[d]) - mean;
    var = fma(diff, diff, var);
  }
  var = simd_sum(var);
  if (lane == 0) sg_buf[sg] = var;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float tot = simd_sum(lane < 8 ? sg_buf[lane] : 0.0f);
    if (lane == 0) sg_buf[0] = rsqrt(tot / float(p.dim) + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sg_buf[0];

  for (uint d = tid; d < p.dim; d += 256) {
    const float n = (float(xr[d]) - mean) * inv;
    yr[d] = bfloat(fma(n, float(weight[d]), float(bias[d])));
  }
}

struct VitLinearParams {
  uint rows;
  uint N;
  uint K;
  uint has_bias;
  uint act;     // 0 = none, 1 = gelu_tanh, 2 = gelu_exact
  uint add_res; // 1 = Y += out (residual add in-place)
};

// BF16 4-row weight-reusing Linear + optional bias + optional GELU + optional residual add (arbitrary K, N).
// Grid: threadgroups ((N + 3) / 4, (rows + 3) / 4, 1) of 128 threads (4 simdgroups = 4 output cols x 4 rows).
kernel void vit_linear_act(const device bfloat* X [[buffer(0)]],
                           const device bfloat* W [[buffer(1)]],
                           const device bfloat* bias [[buffer(2)]],
                           device bfloat* Y [[buffer(3)]],
                           constant VitLinearParams& p [[buffer(4)]],
                           uint2 grp [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
  const uint n = grp.x * 4 + sg;
  const uint m0 = grp.y * 4;
  if (n >= p.N || m0 >= p.rows) return;

  const uint nc = min(4u, p.rows - m0);
  const device bfloat* wr = W + (size_t)n * p.K;
  const device bfloat* x0 = X + (size_t)(m0 + 0u) * p.K;
  const device bfloat* x1 = X + (size_t)(m0 + (nc > 1u ? 1u : 0u)) * p.K;
  const device bfloat* x2 = X + (size_t)(m0 + (nc > 2u ? 2u : 0u)) * p.K;
  const device bfloat* x3 = X + (size_t)(m0 + (nc > 3u ? 3u : 0u)) * p.K;

  float4 acc = float4(0.0f);
  const uint k4 = p.K >> 2;
  const device vec<bfloat, 4>* w4 = (const device vec<bfloat, 4>*)wr;
  if ((p.K & 3u) == 0u) {
    const device vec<bfloat, 4>* a0 = (const device vec<bfloat, 4>*)x0;
    const device vec<bfloat, 4>* a1 = (const device vec<bfloat, 4>*)x1;
    const device vec<bfloat, 4>* a2 = (const device vec<bfloat, 4>*)x2;
    const device vec<bfloat, 4>* a3 = (const device vec<bfloat, 4>*)x3;
    for (uint i = lane; i < k4; i += 32) {
      const float4 wv = float4(w4[i]);
      acc.x += dot(wv, float4(a0[i]));
      if (nc > 1u) acc.y += dot(wv, float4(a1[i]));
      if (nc > 2u) acc.z += dot(wv, float4(a2[i]));
      if (nc > 3u) acc.w += dot(wv, float4(a3[i]));
    }
  } else {
    for (uint k = lane; k < p.K; k += 32) {
      const float wv = float(wr[k]);
      acc.x = fma(wv, float(x0[k]), acc.x);
      if (nc > 1u) acc.y = fma(wv, float(x1[k]), acc.y);
      if (nc > 2u) acc.z = fma(wv, float(x2[k]), acc.z);
      if (nc > 3u) acc.w = fma(wv, float(x3[k]), acc.w);
    }
  }
  acc.x = simd_sum(acc.x);
  if (nc > 1u) acc.y = simd_sum(acc.y);
  if (nc > 2u) acc.z = simd_sum(acc.z);
  if (nc > 3u) acc.w = simd_sum(acc.w);

  if (lane == 0) {
    const float bval = (p.has_bias != 0) ? float(bias[n]) : 0.0f;
    for (uint r = 0u; r < nc; ++r) {
      float val = acc[r] + bval;
      if (p.act == 1) val = gelu_tanh(val);
      else if (p.act == 2) val = gelu_exact(val);
      const size_t y_idx = (size_t)(m0 + r) * p.N + n;
      if (p.add_res != 0) val = float(bfloat(val)) + float(Y[y_idx]);
      Y[y_idx] = bfloat(val);
    }
  }
}

struct VitAddParams {
  uint count;
};

kernel void vit_add_bf16(device bfloat* dst [[buffer(0)]],
                         const device bfloat* src [[buffer(1)]],
                         constant VitAddParams& p [[buffer(2)]],
                         uint tid [[thread_position_in_grid]]) {
  if (tid >= p.count) return;
  dst[tid] = bfloat(float(dst[tid]) + float(src[tid]));
}

struct VitRopeParams {
  uint seq_len;
  uint num_heads;
  uint head_dim;
};

// Apply 2D Vision RoPE (rotate_half) in-place to Q and K inside fused QKV [seq_len, 3 * num_heads * head_dim].
// Grid: threadgroups (num_heads, seq_len, 1) of 32 threads.
kernel void vit_apply_rope_qk(device bfloat* qkv [[buffer(0)]],
                              const device float* cos_tbl [[buffer(1)]],
                              const device float* sin_tbl [[buffer(2)]],
                              constant VitRopeParams& p [[buffer(3)]],
                              uint2 grp [[threadgroup_position_in_grid]],
                              ushort lane [[thread_index_in_simdgroup]]) {
  const uint head = grp.x;
  const uint row = grp.y;
  if (head >= p.num_heads || row >= p.seq_len) return;

  const uint dim = p.num_heads * p.head_dim;
  const uint half_d = p.head_dim >> 1;
  device bfloat* q_ptr = qkv + (size_t)row * (3 * dim) + (size_t)head * p.head_dim;
  device bfloat* k_ptr = q_ptr + dim;
  const device float* c_ptr = cos_tbl + (size_t)row * half_d;
  const device float* s_ptr = sin_tbl + (size_t)row * half_d;

  for (uint i = lane; i < half_d; i += 32) {
    const float c = c_ptr[i];
    const float s = s_ptr[i];
    const float q0 = float(q_ptr[i]), q1 = float(q_ptr[i + half_d]);
    q_ptr[i] = bfloat(q0 * c - q1 * s);
    q_ptr[i + half_d] = bfloat(q1 * c + q0 * s);
    const float k0 = float(k_ptr[i]), k1 = float(k_ptr[i + half_d]);
    k_ptr[i] = bfloat(k0 * c - k1 * s);
    k_ptr[i + half_d] = bfloat(k1 * c + k0 * s);
  }
}

struct VitAttnParams {
  uint seq_len;
  uint num_heads;
  uint head_dim;
  float scale;
};

// Bidirectional Multi-Head Self-Attention over fused QKV [seq_len, 3 * dim] -> out [seq_len, dim].
// Supports arbitrary head_dim <= 256 (including hd=64 for Qwen3.5-0.8B/2B and hd=72 for Qwen3.5-27B / SigLIP!).
// Grid: threadgroups (num_heads, seq_len, 1) of 128 threads (4 simdgroups).
kernel void vit_bidir_attn(const device bfloat* qkv [[buffer(0)]],
                           device bfloat* out [[buffer(1)]],
                           constant VitAttnParams& p [[buffer(2)]],
                           uint2 grp [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
  const uint head = grp.x;
  const uint row = grp.y;
  if (head >= p.num_heads || row >= p.seq_len) return;

  const uint dim = p.num_heads * p.head_dim;
  const uint hd = p.head_dim;
  const uint blocks = (hd + 31u) >> 5; // <= 8
  const size_t stride3 = (size_t)3 * dim;
  const device bfloat* q_ptr = qkv + (size_t)row * stride3 + (size_t)head * hd;

  float q[8];
  float acc[8];
  for (uint b = 0; b < blocks; b++) {
    const uint d = b * 32 + lane;
    q[b] = (d < hd) ? (float(q_ptr[d]) * p.scale) : 0.0f;
    acc[b] = 0.0f;
  }

  float m_val = -INFINITY;
  float l_val = 0.0f;
  for (uint t = sg; t < p.seq_len; t += 4) {
    const device bfloat* k_ptr = qkv + (size_t)t * stride3 + dim + (size_t)head * hd;
    float dot_p = 0.0f;
    for (uint b = 0; b < blocks; b++) {
      const uint d = b * 32 + lane;
      if (d < hd) dot_p = fma(q[b], float(k_ptr[d]), dot_p);
    }
    const float score = simd_sum(dot_p);
    const float new_m = max(m_val, score);
    const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
    const float prob = exp(score - new_m);
    l_val = fma(l_val, alpha, prob);
    m_val = new_m;

    const device bfloat* v_ptr = k_ptr + dim;
    for (uint b = 0; b < blocks; b++) {
      const uint d = b * 32 + lane;
      if (d < hd) acc[b] = fma(acc[b], alpha, prob * float(v_ptr[d]));
    }
  }

  threadgroup float tg_m[4];
  threadgroup float tg_l[4];
  threadgroup float tg_acc[4][256];
  if (lane == 0) {
    tg_m[sg] = m_val;
    tg_l[sg] = l_val;
  }
  for (uint b = 0; b < blocks; b++) {
    const uint d = b * 32 + lane;
    if (d < hd) tg_acc[sg][d] = acc[b];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;

  const float gm = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
  float w[4];
  float gl = 0.0f;
  UNROLL for (uint s = 0; s < 4; s++) {
    w[s] = (tg_l[s] > 0.0f) ? exp(tg_m[s] - gm) : 0.0f;
    gl += tg_l[s] * w[s];
  }
  const float inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;

  device bfloat* dst = out + (size_t)row * dim + (size_t)head * hd;
  for (uint b = 0; b < blocks; b++) {
    const uint d = b * 32 + lane;
    if (d < hd) {
      float o = 0.0f;
      UNROLL for (uint s = 0; s < 4; s++) o = fma(tg_acc[s][d], w[s], o);
      dst[d] = bfloat(o * inv_l);
    }
  }
}

struct VitPoolNormParams {
  uint in_grid;
  uint out_grid;
  uint k_pool;
  uint dim;
  float eps;
};

// Gemma 3-VL MultiModalProjector: AvgPool2d(k_pool, k_pool) + Gemma3RMSNorm (weight pre-shifted +1).
// Grid: threadgroups (out_grid * out_grid, 1, 1) of 256 threads.
kernel void vit_gemma3_pool_rmsnorm(const device bfloat* x [[buffer(0)]],
                                    const device bfloat* norm_w_plus1 [[buffer(1)]],
                                    device bfloat* out [[buffer(2)]],
                                    constant VitPoolNormParams& p [[buffer(3)]],
                                    uint out_idx [[threadgroup_position_in_grid]],
                                    ushort tid [[thread_index_in_threadgroup]],
                                    ushort sg [[simdgroup_index_in_threadgroup]],
                                    ushort lane [[thread_index_in_simdgroup]]) {
  const uint n_out = p.out_grid * p.out_grid;
  if (out_idx >= n_out) return;
  const uint oy = out_idx / p.out_grid;
  const uint ox = out_idx % p.out_grid;
  const float inv_area = 1.0f / float(p.k_pool * p.k_pool);

  float sq = 0.0f;
  for (uint d = tid; d < p.dim; d += 256) {
    float s = 0.0f;
    for (uint dy = 0; dy < p.k_pool; dy++) {
      for (uint dx = 0; dx < p.k_pool; dx++) {
        const uint iy = oy * p.k_pool + dy;
        const uint ix = ox * p.k_pool + dx;
        s += float(x[(size_t)(iy * p.in_grid + ix) * p.dim + d]);
      }
    }
    const float avg = s * inv_area;
    out[(size_t)out_idx * p.dim + d] = bfloat(avg);
    const float r = float(bfloat(avg));
    sq = fma(r, r, sq);
  }
  sq = simd_sum(sq);
  threadgroup float sg_sq[8];
  if (lane == 0) sg_sq[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float tot = simd_sum(lane < 8 ? sg_sq[lane] : 0.0f);
    if (lane == 0) sg_sq[0] = rsqrt(tot / float(p.dim) + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sg_sq[0];

  device bfloat* dst = out + (size_t)out_idx * p.dim;
  for (uint d = tid; d < p.dim; d += 256) {
    dst[d] = bfloat(float(dst[d]) * inv * float(norm_w_plus1[d]));
  }
}
