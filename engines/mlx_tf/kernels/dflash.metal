// Clean-room fused Metal kernels for DFlash2 block-diffusion speculative drafter.
//
// Fuses dynamic 2-tap convolutions with residual adds, RMSNorm, and per-64-group x-sums
// so the entire 5-layer DFlash2 draft frame executes in ~70 dispatches.
#include <metal_stdlib>
using namespace metal;

#define UNROLL _Pragma("clang loop unroll(full)")

constant constexpr uint DHD = 128; // DFlash2 head_dim = 128

struct RowSumsParams {
  uint rows;
  uint width;       // multiple of 64
  uint sums_stride; // multiple of 16
};

// Compute per-64-group sums of a bf16 matrix [rows, width] into sums[width/64, sums_stride].
// Grid: threadgroups (width / 64, rows) of 32 threads.
kernel void dflash_row_sums(const device bfloat* x [[buffer(0)]],
                            device float* sums [[buffer(1)]],
                            constant RowSumsParams& p [[buffer(2)]],
                            uint2 grp [[threadgroup_position_in_grid]],
                            ushort lane [[thread_index_in_simdgroup]]) {
  const uint g = grp.x;
  const uint row = grp.y;
  if (row >= p.rows) return;
  const device bfloat* xr = x + (size_t)row * p.width + g * 64;
  const float s = simd_sum(float(xr[lane]) + float(xr[lane + 32]));
  if (lane == 0) sums[(size_t)g * p.sums_stride + row] = s;
}

struct DynConvParams {
  uint rows;
  uint width;       // 5120 (multiple of 64)
  uint conv_group;  // 16
  uint block;       // 16
  uint branch;      // 0 = prepare (pre-sublayer), 1 = finish (post-sublayer)
  uint sums_stride;
  float eps;
};

// Branch 0: Fused 2-tap grouped dynamic causal conv + per-64-group x-sums.
// Grid: threadgroups (width / 64, rows) of 32 threads (1 simdgroup per 64-elem group/row).
kernel void dflash_dyn_conv_sums(const device bfloat* x [[buffer(0)]],
                                 const device bfloat* dyn [[buffer(1)]],
                                 const device bfloat* base_k [[buffer(2)]],
                                 device bfloat* out [[buffer(3)]],
                                 device float* sums [[buffer(4)]],
                                 constant DynConvParams& p [[buffer(5)]],
                                 uint2 grp [[threadgroup_position_in_grid]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
  const uint g64 = grp.x;
  const uint row = grp.y;
  if (row >= p.rows) return;

  const uint n_cgroups = p.width / p.conv_group;
  const uint dyn_stride = 4 * n_cgroups;
  const uint has_prev = (row % p.block != 0) ? 1u : 0u;
  const device bfloat* dyn_row = dyn + (size_t)row * dyn_stride;
  const device bfloat* bk0 = base_k + (size_t)(p.branch * 2 + 0) * p.width;
  const device bfloat* bk1 = base_k + (size_t)(p.branch * 2 + 1) * p.width;
  const device bfloat* d0_ptr = dyn_row + (p.branch * 2 + 0) * n_cgroups;
  const device bfloat* d1_ptr = dyn_row + (p.branch * 2 + 1) * n_cgroups;

  const uint i0 = g64 * 64 + lane;
  const uint i1 = i0 + 32;
  const uint cg0 = i0 / p.conv_group;
  const uint cg1 = i1 / p.conv_group;

  const float x0_cur = float(x[(size_t)row * p.width + i0]);
  const float x0_prv = has_prev ? float(x[(size_t)(row - 1) * p.width + i0]) : 0.0f;
  const float x1_cur = float(x[(size_t)row * p.width + i1]);
  const float x1_prv = has_prev ? float(x[(size_t)(row - 1) * p.width + i1]) : 0.0f;

  const float y0_f = (float(bk0[i0]) + float(d0_ptr[cg0])) * x0_cur + (float(bk1[i0]) + float(d1_ptr[cg0])) * x0_prv;
  const float y1_f = (float(bk0[i1]) + float(d0_ptr[cg1])) * x1_cur + (float(bk1[i1]) + float(d1_ptr[cg1])) * x1_prv;
  const bfloat y0 = bfloat(y0_f);
  const bfloat y1 = bfloat(y1_f);

  device bfloat* dst = out + (size_t)row * p.width;
  dst[i0] = y0;
  dst[i1] = y1;

  const float gs = simd_sum(float(y0) + float(y1));
  if (lane == 0) sums[(size_t)g64 * p.sums_stride + row] = gs;
}

// Branch 1: Fused 2-tap dynamic causal conv + residual add (`h += conv(x)`) + RMSNorm + group sums.
// 1 threadgroup (256 threads = 8 simdgroups) per row.
kernel void dflash_dyn_conv_add_rms_sums(const device bfloat* x [[buffer(0)]],
                                         const device bfloat* dyn [[buffer(1)]],
                                         const device bfloat* base_k [[buffer(2)]],
                                         device bfloat* h [[buffer(3)]],
                                         const device bfloat* norm_w [[buffer(4)]],
                                         device bfloat* normed [[buffer(5)]],
                                         device float* sums [[buffer(6)]],
                                         constant DynConvParams& p [[buffer(7)]],
                                         uint row [[threadgroup_position_in_grid]],
                                         ushort sg [[simdgroup_index_in_threadgroup]],
                                         ushort lane [[thread_index_in_simdgroup]]) {
  if (row >= p.rows) return;

  const uint groups = p.width >> 6;
  const uint n_cgroups = p.width / p.conv_group;
  const uint dyn_stride = 4 * n_cgroups;
  const uint has_prev = (row % p.block != 0) ? 1u : 0u;
  const device bfloat* dyn_row = dyn + (size_t)row * dyn_stride;
  const device bfloat* bk0 = base_k + (size_t)(p.branch * 2 + 0) * p.width;
  const device bfloat* bk1 = base_k + (size_t)(p.branch * 2 + 1) * p.width;
  const device bfloat* d0_ptr = dyn_row + (p.branch * 2 + 0) * n_cgroups;
  const device bfloat* d1_ptr = dyn_row + (p.branch * 2 + 1) * n_cgroups;

  device bfloat* hr = h + (size_t)row * p.width;
  device bfloat* nr = normed + (size_t)row * p.width;

  float sq = 0.0f;
  for (uint g = sg; g < groups; g += 8) {
    const uint i0 = g * 64 + lane;
    const uint i1 = i0 + 32;
    const uint cg0 = i0 / p.conv_group;
    const uint cg1 = i1 / p.conv_group;

    const float x0_cur = float(x[(size_t)row * p.width + i0]);
    const float x0_prv = has_prev ? float(x[(size_t)(row - 1) * p.width + i0]) : 0.0f;
    const float x1_cur = float(x[(size_t)row * p.width + i1]);
    const float x1_prv = has_prev ? float(x[(size_t)(row - 1) * p.width + i1]) : 0.0f;

    const float c0 = (float(bk0[i0]) + float(d0_ptr[cg0])) * x0_cur + (float(bk1[i0]) + float(d1_ptr[cg0])) * x0_prv;
    const float c1 = (float(bk0[i1]) + float(d0_ptr[cg1])) * x1_cur + (float(bk1[i1]) + float(d1_ptr[cg1])) * x1_prv;

    const bfloat v0 = bfloat(float(hr[i0]) + float(bfloat(c0)));
    const bfloat v1 = bfloat(float(hr[i1]) + float(bfloat(c1)));
    hr[i0] = v0;
    hr[i1] = v1;

    const float f0 = float(v0);
    const float f1 = float(v1);
    sq = fma(f0, f0, fma(f1, f1, sq));
  }
  sq = simd_sum(sq);

  threadgroup float sg_sq[8];
  if (lane == 0) sg_sq[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0) {
    float v = lane < 8 ? sg_sq[lane] : 0.0f;
    float total_sq = simd_sum(v);
    if (lane == 0) sg_sq[0] = rsqrt(total_sq / float(p.width) + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sg_sq[0];

  for (uint g = sg; g < groups; g += 8) {
    const uint i0 = g * 64 + lane;
    const uint i1 = i0 + 32;
    const bfloat y0 = bfloat(float(bfloat(float(hr[i0]) * inv)) * float(norm_w[i0]));
    const bfloat y1 = bfloat(float(bfloat(float(hr[i1]) * inv)) * float(norm_w[i1]));
    nr[i0] = y0;
    nr[i1] = y1;
    const float gs = simd_sum(float(y0) + float(y1));
    if (lane == 0) sums[(size_t)g * p.sums_stride + row] = gs;
  }
}

struct DflashPrepParams {
  uint rows;
  uint num_heads;    // 32
  uint num_kv_heads; // 8
  uint qkv_stride;   // 6144 for block (Q+K+V) or 2048 for ctx (K+V)
  uint is_ctx;       // 1 = context K/V only (into sliding ring), 0 = block Q/K/V
  uint base_pos;     // logical position of row 0
  uint window;       // 2048
  float eps;
};

// Fused Q/K RMSNorm + full 128-dim split-half RoPE + KV store for DFlash2 (head_dim = 128).
// Lane l holds pairs (l, l+64) and (l+32, l+96) -> zero shuffles for 128-dim RoPE!
// Grid: threadgroups (is_ctx ? num_kv_heads : (num_heads + num_kv_heads), rows) of 32 threads.
kernel void dflash_qk_rope_store(const device bfloat* qkv_buf [[buffer(0)]],
                                 const device bfloat* q_norm [[buffer(1)]],
                                 const device bfloat* k_norm [[buffer(2)]],
                                 const device float* inv_freq [[buffer(3)]], // [64] f32
                                 device bfloat* q_out [[buffer(4)]],
                                 device bfloat* k_out [[buffer(5)]],
                                 device bfloat* v_out [[buffer(6)]],
                                 constant DflashPrepParams& p [[buffer(7)]],
                                 uint2 grp [[threadgroup_position_in_grid]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
  const uint idx = grp.x;
  const uint row = grp.y;
  if (row >= p.rows) return;

  const float pos = float(p.base_pos + row);
  const float a0 = pos * inv_freq[lane];
  const float a1 = pos * inv_freq[lane + 32];
  const float c0 = cos(a0), s0 = sin(a0);
  const float c1 = cos(a1), s1 = sin(a1);

  const device bfloat* row_ptr = qkv_buf + (size_t)row * p.qkv_stride;

  if (p.is_ctx == 0 && idx < p.num_heads) {
    const device bfloat* src = row_ptr + (size_t)idx * DHD;
    float v0 = float(src[lane]);
    float v1 = float(src[lane + 32]);
    float v2 = float(src[lane + 64]);
    float v3 = float(src[lane + 96]);
    const float sq = simd_sum(fma(v0, v0, fma(v1, v1, fma(v2, v2, v3 * v3))));
    const float inv = rsqrt(sq / float(DHD) + p.eps);

    v0 = float(bfloat(v0 * inv)) * float(q_norm[lane]);
    v1 = float(bfloat(v1 * inv)) * float(q_norm[lane + 32]);
    v2 = float(bfloat(v2 * inv)) * float(q_norm[lane + 64]);
    v3 = float(bfloat(v3 * inv)) * float(q_norm[lane + 96]);

    device bfloat* dst = q_out + (size_t)(row * p.num_heads + idx) * DHD;
    dst[lane]      = bfloat(v0 * c0 - v2 * s0);
    dst[lane + 64] = bfloat(v2 * c0 + v0 * s0);
    dst[lane + 32] = bfloat(v1 * c1 - v3 * s1);
    dst[lane + 96] = bfloat(v3 * c1 + v1 * s1);
    return;
  }

  const uint kh = (p.is_ctx != 0) ? idx : (idx - p.num_heads);
  const size_t k_off = (p.is_ctx != 0) ? (size_t)kh * DHD : ((size_t)p.num_heads * DHD + (size_t)kh * DHD);
  const size_t v_off = k_off + (size_t)p.num_kv_heads * DHD;
  const device bfloat* ks = row_ptr + k_off;
  const device bfloat* vs = row_ptr + v_off;

  float k0 = float(ks[lane]);
  float k1 = float(ks[lane + 32]);
  float k2 = float(ks[lane + 64]);
  float k3 = float(ks[lane + 96]);
  const float sq = simd_sum(fma(k0, k0, fma(k1, k1, fma(k2, k2, k3 * k3))));
  const float inv = rsqrt(sq / float(DHD) + p.eps);

  k0 = float(bfloat(k0 * inv)) * float(k_norm[lane]);
  k1 = float(bfloat(k1 * inv)) * float(k_norm[lane + 32]);
  k2 = float(bfloat(k2 * inv)) * float(k_norm[lane + 64]);
  k3 = float(bfloat(k3 * inv)) * float(k_norm[lane + 96]);

  const uint dst_slot = (p.is_ctx != 0) ? ((p.base_pos + row) % p.window) : row;
  const size_t dst_base = (size_t)(dst_slot * p.num_kv_heads + kh) * DHD;
  device bfloat* kd = k_out + dst_base;
  device bfloat* vd = v_out + dst_base;

  kd[lane]      = bfloat(k0 * c0 - k2 * s0);
  kd[lane + 64] = bfloat(k2 * c0 + k0 * s0);
  kd[lane + 32] = bfloat(k1 * c1 - k3 * s1);
  kd[lane + 96] = bfloat(k3 * c1 + k1 * s1);

  vd[lane]      = vs[lane];
  vd[lane + 32] = vs[lane + 32];
  vd[lane + 64] = vs[lane + 64];
  vd[lane + 96] = vs[lane + 96];
}

struct DflashAttnParams {
  uint block_rows;   // 16
  uint trained_block;// 8
  uint num_heads;    // 32
  uint num_kv_heads; // 8
  uint ctx_begin;    // oldest valid logical position in ctx ring
  uint ctx_end;      // logical position of block row 0 (ctx holds [ctx_begin, ctx_end))
  uint window;       // 2048
  uint sums_stride;
  float scale;       // 1 / sqrt(128)
};

// Fused DFlash2 sliding-window + block GQA FlashAttention + o_proj group-64 sums.
// 4 simdgroups (128 threads) per (head, q_row).
// Grid: threadgroups (num_heads, block_rows) of 128 threads.
kernel void dflash_attn_sums(const device bfloat* q_block [[buffer(0)]],
                             const device bfloat* k_ctx [[buffer(1)]],
                             const device bfloat* v_ctx [[buffer(2)]],
                             const device bfloat* k_block [[buffer(3)]],
                             const device bfloat* v_block [[buffer(4)]],
                             device bfloat* out [[buffer(5)]],
                             device float* sums [[buffer(6)]],
                             constant DflashAttnParams& p [[buffer(7)]],
                             uint2 grp [[threadgroup_position_in_grid]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
  const uint head = grp.x;
  const uint row = grp.y;
  if (head >= p.num_heads || row >= p.block_rows) return;

  const uint kv_head = head / (p.num_heads / p.num_kv_heads);
  const device bfloat* qp = q_block + (size_t)(row * p.num_heads + head) * DHD;
  const float q0 = float(qp[lane]) * p.scale;
  const float q1 = float(qp[lane + 32]) * p.scale;
  const float q2 = float(qp[lane + 64]) * p.scale;
  const float q3 = float(qp[lane + 96]) * p.scale;

  // Determine valid sliding-window context range for logical position (ctx_end + row)
  const uint q_pos = p.ctx_end + row;
  const uint win_min = (q_pos + 1 > p.window) ? (q_pos + 1 - p.window) : 0u;
  const uint c_start = max(p.ctx_begin, win_min);
  const uint c_count = (p.ctx_end > c_start) ? (p.ctx_end - c_start) : 0u;
  // Block slots < trained_block (8) only attend within 0..7; slots >= 8 attend to 0..block_rows-1
  const uint b_limit = (row < p.trained_block) ? min(p.trained_block, p.block_rows) : p.block_rows;
  const uint total_k = c_count + b_limit;

  float m_val = -INFINITY;
  float l_val = 0.0f;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

  const size_t kv_stride = (size_t)p.num_kv_heads * DHD;
  const size_t kh_off = (size_t)kv_head * DHD;

  for (uint idx = sg; idx < total_k; idx += 4) {
    const device bfloat* kp;
    const device bfloat* vp;
    if (idx < c_count) {
      const uint slot = (c_start + idx) % p.window;
      kp = k_ctx + (size_t)slot * kv_stride + kh_off;
      vp = v_ctx + (size_t)slot * kv_stride + kh_off;
    } else {
      const uint b_slot = idx - c_count;
      kp = k_block + (size_t)b_slot * kv_stride + kh_off;
      vp = v_block + (size_t)b_slot * kv_stride + kh_off;
    }
    const float dot_p = fma(q0, float(kp[lane]), fma(q1, float(kp[lane + 32]), fma(q2, float(kp[lane + 64]), q3 * float(kp[lane + 96]))));
    const float score = simd_sum(dot_p);
    const float new_m = max(m_val, score);
    const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
    const float prob = exp(score - new_m);
    l_val = fma(l_val, alpha, prob);
    m_val = new_m;

    a0 = fma(a0, alpha, prob * float(vp[lane]));
    a1 = fma(a1, alpha, prob * float(vp[lane + 32]));
    a2 = fma(a2, alpha, prob * float(vp[lane + 64]));
    a3 = fma(a3, alpha, prob * float(vp[lane + 96]));
  }

  threadgroup float tg_m[4];
  threadgroup float tg_l[4];
  threadgroup float tg_acc[4][DHD];

  if (lane == 0) {
    tg_m[sg] = m_val;
    tg_l[sg] = l_val;
  }
  tg_acc[sg][lane] = a0;
  tg_acc[sg][lane + 32] = a1;
  tg_acc[sg][lane + 64] = a2;
  tg_acc[sg][lane + 96] = a3;
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

  float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
  UNROLL for (uint s = 0; s < 4; s++) {
    o0 = fma(tg_acc[s][lane], w[s], o0);
    o1 = fma(tg_acc[s][lane + 32], w[s], o1);
    o2 = fma(tg_acc[s][lane + 64], w[s], o2);
    o3 = fma(tg_acc[s][lane + 96], w[s], o3);
  }
  const bfloat y0 = bfloat(o0 * inv_l);
  const bfloat y1 = bfloat(o1 * inv_l);
  const bfloat y2 = bfloat(o2 * inv_l);
  const bfloat y3 = bfloat(o3 * inv_l);

  device bfloat* op = out + (size_t)(row * p.num_heads + head) * DHD;
  op[lane] = y0;
  op[lane + 32] = y1;
  op[lane + 64] = y2;
  op[lane + 96] = y3;

  // Group 0 = [0..63] (y0, y1); Group 1 = [64..127] (y2, y3)
  const float gs0 = simd_sum(float(y0) + float(y1));
  const float gs1 = simd_sum(float(y2) + float(y3));
  if (lane == 0) {
    sums[(size_t)(head * 2) * p.sums_stride + row] = gs0;
    sums[(size_t)(head * 2 + 1) * p.sums_stride + row] = gs1;
  }
}

struct Bf16LinearParams {
  uint rows;
  uint n; // 256
  uint k; // 5120
};

// Small BF16 linear projection for candidate_selector.hidden_projection [256, 5120] -> f32 [rows, 256].
// Grid: threadgroups (n / 8, rows) of 256 threads (8 simdgroups, 1 col per simdgroup).
kernel void dflash_selector_proj(const device bfloat* x [[buffer(0)]],
                                 const device bfloat* w [[buffer(1)]],
                                 device float* y [[buffer(2)]],
                                 constant Bf16LinearParams& p [[buffer(3)]],
                                 uint2 grp [[threadgroup_position_in_grid]],
                                 ushort sg [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
  const uint col = grp.x * 8 + sg;
  const uint row = grp.y;
  if (col >= p.n || row >= p.rows) return;
  const device bfloat* xr = x + (size_t)row * p.k;
  const device bfloat* wr = w + (size_t)col * p.k;
  float acc = 0.0f;
  for (uint i = lane; i < p.k; i += 32) {
    acc = fma(float(xr[i]), float(wr[i]), acc);
  }
  acc = simd_sum(acc);
  if (lane == 0) y[(size_t)row * p.n + col] = float(bfloat(acc));
}

struct TopkEntry {
  uint token;
  float score;
};

struct TopkParams {
  uint rows;
  uint vocab; // trimmed head_width (98592)
  uint spans; // 16
};

inline bool topkBetter(float a_s, uint a_t, float b_s, uint b_t) {
  return (a_s > b_s) || (a_s == b_s && a_t < b_t);
}

// Stage 1: Divide trimmed vocab into 16 spans per row; each TG extracts top-16 within its span.
// Grid: threadgroups (16, rows) of 256 threads.
kernel void dflash_topk16_span(const device bfloat* logits [[buffer(0)]],
                               device TopkEntry* partials [[buffer(1)]], // [rows][16][16]
                               constant TopkParams& p [[buffer(2)]],
                               uint2 grp [[threadgroup_position_in_grid]],
                               ushort tid [[thread_index_in_threadgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
  const uint span = grp.x;
  const uint row = grp.y;
  if (span >= p.spans || row >= p.rows) return;

  const uint span_len = (p.vocab + p.spans - 1) / p.spans;
  const uint start = span * span_len;
  const uint end = min(p.vocab, start + span_len);
  const device bfloat* lr = logits + (size_t)row * p.vocab;

  // Maintain per-thread top-2 so even densely clustered peaks never miss top-16
  // Actually iterative TG argmax x16 over 256 threads is super fast & exact (16 iterations over 24 items/thread)!
  threadgroup float tg_v[8];
  threadgroup uint tg_i[8];
  threadgroup uint picked[16];

  for (uint k = 0; k < 16; k++) {
    float bv = -INFINITY;
    uint bi = 0xFFFFFFFFu;
    for (uint idx = start + tid; idx < end; idx += 256) {
      bool skip = false;
      for (uint j = 0; j < k; j++) {
        if (picked[j] == idx) { skip = true; break; }
      }
      if (skip) continue;
      const float v = float(lr[idx]);
      if (isfinite(v) && topkBetter(v, idx, bv, bi)) {
        bv = v;
        bi = idx;
      }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
      const float ov = simd_shuffle_down(bv, off);
      const uint oi = simd_shuffle_down(bi, off);
      if (topkBetter(ov, oi, bv, bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { tg_v[sg] = bv; tg_i[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float v = (lane < 8) ? tg_v[lane] : -INFINITY;
      uint idx = (lane < 8) ? tg_i[lane] : 0xFFFFFFFFu;
      for (ushort off = 4; off > 0; off >>= 1) {
        const float ov = simd_shuffle_down(v, off);
        const uint oi = simd_shuffle_down(idx, off);
        if (topkBetter(ov, oi, v, idx)) { v = ov; idx = oi; }
      }
      if (lane == 0) {
        picked[k] = idx;
        partials[((size_t)row * p.spans + span) * 16 + k] = TopkEntry{ idx, v };
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Stage 2: Merge 16 * 16 = 256 span candidates into final top-16 per row.
// Grid: threadgroups (rows, 1) of 256 threads.
kernel void dflash_topk16_merge(const device TopkEntry* partials [[buffer(0)]],
                                device TopkEntry* out [[buffer(1)]], // [rows][16]
                                constant TopkParams& p [[buffer(2)]],
                                uint row [[threadgroup_position_in_grid]],
                                ushort tid [[thread_index_in_threadgroup]],
                                ushort sg [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]]) {
  if (row >= p.rows) return;
  const device TopkEntry* src = partials + (size_t)row * 256;

  threadgroup float vals[256];
  threadgroup uint toks[256];
  threadgroup float tg_v[8];
  threadgroup uint tg_i[8];

  vals[tid] = src[tid].score;
  toks[tid] = src[tid].token;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint k = 0; k < 16; k++) {
    float bv = vals[tid];
    uint bi = toks[tid];
    uint bslot = tid;

    for (ushort off = 16; off > 0; off >>= 1) {
      const float ov = simd_shuffle_down(bv, off);
      const uint oi = simd_shuffle_down(bi, off);
      const uint os = simd_shuffle_down(bslot, off);
      if (topkBetter(ov, oi, bv, bi)) { bv = ov; bi = oi; bslot = os; }
    }
    if (lane == 0) { tg_v[sg] = bv; tg_i[sg] = bslot; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
      uint slot = (lane < 8) ? tg_i[lane] : 0;
      float v = (lane < 8) ? tg_v[lane] : -INFINITY;
      uint tok = (lane < 8) ? toks[slot] : 0xFFFFFFFFu;
      for (ushort off = 4; off > 0; off >>= 1) {
        const float ov = simd_shuffle_down(v, off);
        const uint ot = simd_shuffle_down(tok, off);
        const uint os = simd_shuffle_down(slot, off);
        if (topkBetter(ov, ot, v, tok)) { v = ov; tok = ot; slot = os; }
      }
      if (lane == 0) {
        out[(size_t)row * 16 + k] = TopkEntry{ tok, v };
        vals[slot] = -INFINITY;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
