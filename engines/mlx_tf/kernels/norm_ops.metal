// Clean-room fused embedding, RMSNorm + group-sums, Residual-Add + RMSNorm + group-sums,
// and GDN per-head RMSNorm + SiLU gate + group-sums kernels.
#include <metal_stdlib>
using namespace metal;

#define UNROLL _Pragma("clang loop unroll(full)")

struct EmbedParams {
  uint rows;
  uint hidden;
};

// Dequantize 4-bit affine group-64 embeddings for `rows` tokens.
// Grid: threads (hidden / 8, rows) — each thread unpacks 8 elements (one uint32 word).
kernel void embed_lookup(const device uint* tokens [[buffer(0)]],
                         const device uint* codes [[buffer(1)]],
                         const device bfloat* scales [[buffer(2)]],
                         const device bfloat* biases [[buffer(3)]],
                         device bfloat* out [[buffer(4)]],
                         constant EmbedParams& p [[buffer(5)]],
                         uint2 tid [[thread_position_in_grid]]) {
  const uint word_idx = tid.x;
  const uint row = tid.y;
  const uint words_per_row = p.hidden >> 3;
  if (word_idx >= words_per_row || row >= p.rows) return;

  const uint tok = tokens[row];
  const uint g = word_idx >> 3; // 8 words = 64 elements = 1 quant group
  const uint groups = p.hidden >> 6;
  const uint packed = codes[(size_t)tok * words_per_row + word_idx];
  const float scale = float(scales[(size_t)tok * groups + g]);
  const float bias = float(biases[(size_t)tok * groups + g]);

  device bfloat* dst = out + (size_t)row * p.hidden + word_idx * 8;
  UNROLL for (uint i = 0; i < 8; i++) {
    const uint c = (packed >> (i * 4)) & 0xFu;
    dst[i] = bfloat(fma(float(c), scale, bias));
  }
}

struct NormParams {
  uint rows;
  uint hidden;
  uint sums_stride;
  float eps;
};

// Fused RMSNorm + per-64-group x-sums (`SUMS[g * sums_stride + row]`).
// 1 threadgroup (256 threads = 8 simdgroups) per row.
kernel void rms_norm_sums(const device bfloat* x [[buffer(0)]],
                          const device bfloat* weight [[buffer(1)]],
                          device bfloat* out [[buffer(2)]],
                          device float* sums [[buffer(3)]],
                          constant NormParams& p [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
  if (row >= p.rows) return;
  const device bfloat* xr = x + (size_t)row * p.hidden;
  device bfloat* yr = out + (size_t)row * p.hidden;
  const uint groups = p.hidden >> 6;

  float sq = 0.0f;
  for (uint g = sg; g < groups; g += 8) {
    const uint base = g * 64 + lane;
    const float v0 = float(xr[base]);
    const float v1 = float(xr[base + 32]);
    sq = fma(v0, v0, fma(v1, v1, sq));
  }
  sq = simd_sum(sq);
  threadgroup float sg_sq[8];
  if (lane == 0) sg_sq[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float total_sq = 0.0f;
  if (sg == 0) {
    float v = lane < 8 ? sg_sq[lane] : 0.0f;
    total_sq = simd_sum(v);
    if (lane == 0) sg_sq[0] = rsqrt(total_sq / float(p.hidden) + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sg_sq[0];

  for (uint g = sg; g < groups; g += 8) {
    const uint base = g * 64 + lane;
    const bfloat y0 = bfloat(float(xr[base]) * inv * float(weight[base]));
    const bfloat y1 = bfloat(float(xr[base + 32]) * inv * float(weight[base + 32]));
    yr[base] = y0;
    yr[base + 32] = y1;
    const float gs = simd_sum(float(y0) + float(y1));
    if (lane == 0) sums[(size_t)g * p.sums_stride + row] = gs;
  }
}

// Fused in-place Residual Add (`h += delta`) + RMSNorm + per-64-group x-sums.
// Single-pass register-resident path for hidden <= 8192 eliminates reloading `h` from memory!
kernel void add_rms_norm_sums(device bfloat* h [[buffer(0)]],
                              const device bfloat* delta [[buffer(1)]],
                              const device bfloat* weight [[buffer(2)]],
                              device bfloat* out [[buffer(3)]],
                              device float* sums [[buffer(4)]],
                              constant NormParams& p [[buffer(5)]],
                              uint row [[threadgroup_position_in_grid]],
                              ushort tid [[thread_index_in_threadgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
  if (row >= p.rows) return;
  device bfloat* hr = h + (size_t)row * p.hidden;
  const device bfloat* dr = delta + (size_t)row * p.hidden;
  device bfloat* yr = out + (size_t)row * p.hidden;
  const uint groups = p.hidden >> 6;

  // Fast register-resident single-pass path (covers hidden <= 8192: 0.8B/2B/27B/70B!)
  if (groups <= 128) {
    float2 reg[16]; // at most 128/8 = 16 groups per simdgroup, 2 elements per lane per group
    float sq = 0.0f;
    uint k = 0;
    for (uint g = sg; g < groups; g += 8, k++) {
      const uint base = g * 64 + lane;
      const bfloat v0 = bfloat(float(hr[base]) + float(dr[base]));
      const bfloat v1 = bfloat(float(hr[base + 32]) + float(dr[base + 32]));
      hr[base] = v0;
      hr[base + 32] = v1;
      const float f0 = float(v0);
      const float f1 = float(v1);
      reg[k] = float2(f0, f1);
      sq = fma(f0, f0, fma(f1, f1, sq));
    }
    sq = simd_sum(sq);
    threadgroup float sg_sq[8];
    if (lane == 0) sg_sq[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
      float v = lane < 8 ? sg_sq[lane] : 0.0f;
      float total_sq = simd_sum(v);
      if (lane == 0) sg_sq[0] = rsqrt(total_sq / float(p.hidden) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = sg_sq[0];

    k = 0;
    for (uint g = sg; g < groups; g += 8, k++) {
      const uint base = g * 64 + lane;
      const bfloat y0 = bfloat(reg[k].x * inv * float(weight[base]));
      const bfloat y1 = bfloat(reg[k].y * inv * float(weight[base + 32]));
      yr[base] = y0;
      yr[base + 32] = y1;
      if (p.sums_stride != 0) {
        const float gs = simd_sum(float(y0) + float(y1));
        if (lane == 0) sums[(size_t)g * p.sums_stride + row] = gs;
      }
    }
    return;
  }

  float sq = 0.0f;
  for (uint g = sg; g < groups; g += 8) {
    const uint base = g * 64 + lane;
    const bfloat v0 = bfloat(float(hr[base]) + float(dr[base]));
    const bfloat v1 = bfloat(float(hr[base + 32]) + float(dr[base + 32]));
    hr[base] = v0;
    hr[base + 32] = v1;
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
    if (lane == 0) sg_sq[0] = rsqrt(total_sq / float(p.hidden) + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sg_sq[0];

  for (uint g = sg; g < groups; g += 8) {
    const uint base = g * 64 + lane;
    const bfloat y0 = bfloat(float(hr[base]) * inv * float(weight[base]));
    const bfloat y1 = bfloat(float(hr[base + 32]) * inv * float(weight[base + 32]));
    yr[base] = y0;
    yr[base + 32] = y1;
    if (p.sums_stride != 0) {
      const float gs = simd_sum(float(y0) + float(y1));
      if (lane == 0) sums[(size_t)g * p.sums_stride + row] = gs;
    }
  }
}

struct GdnNormParams {
  uint rows;
  uint nv;
  uint dv;             // 128
  uint in_proj_stride; // stride of fused in_proj_all buffer
  uint z_offset;       // starting column of Z inside in_proj_all
  uint sums_stride;
  float eps;
};

// Fused GDN per-head RMSNorm (DV=128) + SiLU(z) output gate + out_proj group sums.
// Grid: threadgroups (nv, rows) of 32 threads (1 simdgroup per head/row, zero TG barriers).
kernel void gdn_norm_gate_sums(const device bfloat* scan_out [[buffer(0)]],
                               const device bfloat* in_proj_all [[buffer(1)]],
                               const device bfloat* norm_w [[buffer(2)]],
                               device bfloat* out [[buffer(3)]],
                               device float* sums [[buffer(4)]],
                               constant GdnNormParams& p [[buffer(5)]],
                               uint2 grp [[threadgroup_position_in_grid]],
                               ushort lane [[thread_index_in_simdgroup]]) {
  const uint head = grp.x;
  const uint row = grp.y;
  if (head >= p.nv || row >= p.rows) return;

  // Each lane holds 2 elements from group 0 ([0..63]) and 2 elements from group 1 ([64..127]).
  const uint base = (row * p.nv + head) * 128;
  const uint z_base = row * p.in_proj_stride + p.z_offset + head * 128;

  const float x0 = float(scan_out[base + lane]);
  const float x1 = float(scan_out[base + 32 + lane]);
  const float x2 = float(scan_out[base + 64 + lane]);
  const float x3 = float(scan_out[base + 96 + lane]);

  const float sq = simd_sum(fma(x0, x0, fma(x1, x1, fma(x2, x2, x3 * x3))));
  const float inv = rsqrt(sq / 128.0f + p.eps);

  const float z0 = float(in_proj_all[z_base + lane]);
  const float z1 = float(in_proj_all[z_base + 32 + lane]);
  const float z2 = float(in_proj_all[z_base + 64 + lane]);
  const float z3 = float(in_proj_all[z_base + 96 + lane]);

  const float s0 = z0 / (1.0f + exp(-z0));
  const float s1 = z1 / (1.0f + exp(-z1));
  const float s2 = z2 / (1.0f + exp(-z2));
  const float s3 = z3 / (1.0f + exp(-z3));

  const bfloat y0 = bfloat((x0 * inv * float(norm_w[lane])) * s0);
  const bfloat y1 = bfloat((x1 * inv * float(norm_w[32 + lane])) * s1);
  const bfloat y2 = bfloat((x2 * inv * float(norm_w[64 + lane])) * s2);
  const bfloat y3 = bfloat((x3 * inv * float(norm_w[96 + lane])) * s3);

  out[base + lane] = y0;
  out[base + 32 + lane] = y1;
  out[base + 64 + lane] = y2;
  out[base + 96 + lane] = y3;

  if (p.sums_stride != 0) {
    const float gs0 = simd_sum(float(y0) + float(y1));
    const float gs1 = simd_sum(float(y2) + float(y3));
    if (lane == 0) {
      sums[(size_t)(head * 2) * p.sums_stride + row] = gs0;
      sums[(size_t)(head * 2 + 1) * p.sums_stride + row] = gs1;
    }
  }
}

// Ultra-low-latency 1-row RMSNorm (used on rows == 1 after residual add is fused into qmv_1).
// Single-simdgroup ZERO-barrier path for hidden <= 2048 (0.8B/1.5B/2B), vectorized bfloat4 loads!
// Grid: threadgroups (1, 1, 1) of 32 threads (hidden <= 2048) or 128 threads (hidden > 2048).
kernel void rms_norm_1_sg(const device vec<bfloat, 4>* x [[buffer(0)]],
                          const device vec<bfloat, 4>* weight [[buffer(1)]],
                          device vec<bfloat, 4>* out [[buffer(2)]],
                          constant NormParams& p [[buffer(3)]],
                          ushort lane [[thread_index_in_simdgroup]]) {
  const uint n4 = p.hidden >> 2;
  float4 reg[16]; // up to 32 * 16 * 4 = 2048 elements in registers!
  float sq = 0.0f;
  uint k = 0;
  for (uint i = lane; i < n4; i += 32, k++) {
    const float4 v = float4(x[i]);
    reg[k] = v;
    sq = fma(v[0], v[0], fma(v[1], v[1], fma(v[2], v[2], fma(v[3], v[3], sq))));
  }
  const float inv = rsqrt(simd_sum(sq) / float(p.hidden) + p.eps);
  k = 0;
  for (uint i = lane; i < n4; i += 32, k++) {
    const float4 w = float4(weight[i]);
    out[i] = vec<bfloat, 4>(reg[k] * inv * w);
  }
}

kernel void rms_norm_1_wide(const device vec<bfloat, 4>* x [[buffer(0)]],
                            const device vec<bfloat, 4>* weight [[buffer(1)]],
                            device vec<bfloat, 4>* out [[buffer(2)]],
                            constant NormParams& p [[buffer(3)]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
  const uint n4 = p.hidden >> 2;
  float4 reg[16]; // up to 128 * 16 * 4 = 8192 elements in registers!
  float sq = 0.0f;
  uint k = 0;
  for (uint i = tid; i < n4; i += 128, k++) {
    const float4 v = float4(x[i]);
    reg[k] = v;
    sq = fma(v[0], v[0], fma(v[1], v[1], fma(v[2], v[2], fma(v[3], v[3], sq))));
  }
  sq = simd_sum(sq);
  threadgroup float sg_sq[4];
  if (lane == 0) sg_sq[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total_sq = (sg_sq[0] + sg_sq[1]) + (sg_sq[2] + sg_sq[3]);
  const float inv = rsqrt(total_sq / float(p.hidden) + p.eps);
  k = 0;
  for (uint i = tid; i < n4; i += 128, k++) {
    const float4 w = float4(weight[i]);
    out[i] = vec<bfloat, 4>(reg[k] * inv * w);
  }
}

struct ScaleParams {
  uint count;
  float scale;
};

// In-place embedding scaling (`x *= scale`, used for Gemma 2/3/4 `sqrt(hidden_size)`).
kernel void scale_bf16(device bfloat* x [[buffer(0)]],
                       constant ScaleParams& p [[buffer(1)]],
                       uint tid [[thread_position_in_grid]]) {
  if (tid >= p.count) return;
  x[tid] = bfloat(float(x[tid]) * p.scale);
}

struct PostNormResParams {
  uint rows;
  uint hidden;
  uint sums_stride;
  uint has_scalar;
  float eps;
};

// Fused Gemma 2/3/4 residual block transition:
//   h_new = (h + RMSNorm(delta) * w_post) * (has_scalar ? scalar[0] : 1.0)
//   out   = RMSNorm(h_new) * w_next   (+ optional group-64 sums)
// Grid: threadgroups (rows, 1, 1) of 256 threads (8 simdgroups).
kernel void norm_res_norm_sums(device bfloat* h [[buffer(0)]],
                               const device bfloat* delta [[buffer(1)]],
                               const device bfloat* w_post [[buffer(2)]],
                               const device bfloat* layer_scalar [[buffer(3)]],
                               const device bfloat* w_next [[buffer(4)]],
                               device bfloat* out [[buffer(5)]],
                               device float* sums [[buffer(6)]],
                               constant PostNormResParams& p [[buffer(7)]],
                               uint row [[threadgroup_position_in_grid]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
  if (row >= p.rows) return;
  device bfloat* hr = h + (size_t)row * p.hidden;
  const device bfloat* dr = delta + (size_t)row * p.hidden;
  device bfloat* yr = out + (size_t)row * p.hidden;
  const uint groups = p.hidden >> 6;
  const float sc = (p.has_scalar != 0) ? float(layer_scalar[0]) : 1.0f;

  // 1. RMSNorm(delta) denominator
  float sq1 = 0.0f;
  for (uint g = sg; g < groups; g += 8) {
    const uint base = g * 64 + lane;
    const float d0 = float(dr[base]);
    const float d1 = float(dr[base + 32]);
    sq1 = fma(d0, d0, fma(d1, d1, sq1));
  }
  sq1 = simd_sum(sq1);
  threadgroup float sg_sq[8];
  if (lane == 0) sg_sq[sg] = sq1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float tot = simd_sum(lane < 8 ? sg_sq[lane] : 0.0f);
    if (lane == 0) sg_sq[0] = rsqrt(tot / float(p.hidden) + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv1 = sg_sq[0];

  // 2. h = (h + RMSNorm(delta) * w_post) * sc, and accumulate sq2 for RMSNorm(h)
  float sq2 = 0.0f;
  for (uint g = sg; g < groups; g += 8) {
    const uint base = g * 64 + lane;
    const float n0 = float(dr[base]) * inv1 * float(w_post[base]);
    const float n1 = float(dr[base + 32]) * inv1 * float(w_post[base + 32]);
    const bfloat h0 = bfloat((float(hr[base]) + n0) * sc);
    const bfloat h1 = bfloat((float(hr[base + 32]) + n1) * sc);
    hr[base] = h0;
    hr[base + 32] = h1;
    const float f0 = float(h0);
    const float f1 = float(h1);
    sq2 = fma(f0, f0, fma(f1, f1, sq2));
  }
  sq2 = simd_sum(sq2);
  if (lane == 0) sg_sq[sg] = sq2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float tot = simd_sum(lane < 8 ? sg_sq[lane] : 0.0f);
    if (lane == 0) sg_sq[0] = rsqrt(tot / float(p.hidden) + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv2 = sg_sq[0];

  // 3. out = RMSNorm(h) * w_next + group-64 sums
  for (uint g = sg; g < groups; g += 8) {
    const uint base = g * 64 + lane;
    const bfloat y0 = bfloat(float(hr[base]) * inv2 * float(w_next[base]));
    const bfloat y1 = bfloat(float(hr[base + 32]) * inv2 * float(w_next[base + 32]));
    yr[base] = y0;
    yr[base + 32] = y1;
    if (p.sums_stride != 0) {
      const float gs = simd_sum(float(y0) + float(y1));
      if (lane == 0) sums[(size_t)g * p.sums_stride + row] = gs;
    }
  }
}

// ============================================================================
// LiquidAI LFM2 / LFM2.5 ShortConv (Gated Causal Depthwise Conv1d + Group Sums)
// ============================================================================

struct LfmConvParams {
  uint rows;
  uint hidden;
  uint l_cache;     // conv kernel length (e.g. 3 or 4, max 8)
  uint sums_stride; // 0 to skip sums
  uint has_in_bias;
  uint has_conv_bias;
  uint commit_hist;
};

// Grid: threadgroups (hidden / 64, 1, 1) of 32 threads (1 simdgroup = 64 channels, 2 ch/lane).
// Computes: (B, C, x) = split(BCx); Bx = B * x; conv_out = depthwise_conv1d(cat(hist, Bx), conv_w);
//           y = C * conv_out; writes y -> out, optional group-64 sums -> sums, and updates hist.
kernel void lfm2_short_conv_sums(const device bfloat* bcx [[buffer(0)]],
                                 const device bfloat* in_bias [[buffer(1)]],
                                 const device bfloat* conv_w [[buffer(2)]],
                                 const device bfloat* conv_bias [[buffer(3)]],
                                 device bfloat* hist [[buffer(4)]],
                                 device bfloat* out [[buffer(5)]],
                                 device float* sums [[buffer(6)]],
                                 constant LfmConvParams& p [[buffer(7)]],
                                 uint g [[threadgroup_position_in_grid]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
  const uint H = p.hidden;
  const uint L = p.l_cache;
  const uint H_hist = L - 1;
  const uint c0 = g * 64 + lane;
  const uint c1 = c0 + 32;
  if (c0 >= H) return;

  float w0[8], w1[8];
  for (uint k = 0; k < L; k++) {
    w0[k] = float(conv_w[(size_t)c0 * L + k]);
    w1[k] = float(conv_w[(size_t)c1 * L + k]);
  }
  const float cb0 = (p.has_conv_bias != 0) ? float(conv_bias[c0]) : 0.0f;
  const float cb1 = (p.has_conv_bias != 0) ? float(conv_bias[c1]) : 0.0f;
  const float ib_b0 = (p.has_in_bias != 0) ? float(in_bias[c0]) : 0.0f;
  const float ib_c0 = (p.has_in_bias != 0) ? float(in_bias[H + c0]) : 0.0f;
  const float ib_x0 = (p.has_in_bias != 0) ? float(in_bias[2 * H + c0]) : 0.0f;
  const float ib_b1 = (p.has_in_bias != 0) ? float(in_bias[c1]) : 0.0f;
  const float ib_c1 = (p.has_in_bias != 0) ? float(in_bias[H + c1]) : 0.0f;
  const float ib_x1 = (p.has_in_bias != 0) ? float(in_bias[2 * H + c1]) : 0.0f;

  float ring0[8], ring1[8];
  for (uint k = 0; k < H_hist; k++) {
    ring0[k] = float(hist[(size_t)k * H + c0]);
    ring1[k] = float(hist[(size_t)k * H + c1]);
  }

  const size_t stride3 = (size_t)3 * H;
  for (uint r = 0; r < p.rows; r++) {
    const device bfloat* row_bcx = bcx + (size_t)r * stride3;
    const float b0 = float(row_bcx[c0]) + ib_b0;
    const float cv0 = float(row_bcx[H + c0]) + ib_c0;
    const float x0 = float(row_bcx[2 * H + c0]) + ib_x0;
    const float b1 = float(row_bcx[c1]) + ib_b1;
    const float cv1 = float(row_bcx[H + c1]) + ib_c1;
    const float x1 = float(row_bcx[2 * H + c1]) + ib_x1;

    ring0[H_hist] = float(bfloat(b0 * x0));
    ring1[H_hist] = float(bfloat(b1 * x1));

    float acc0 = cb0, acc1 = cb1;
    for (uint k = 0; k < L; k++) {
      acc0 = fma(ring0[k], w0[k], acc0);
      acc1 = fma(ring1[k], w1[k], acc1);
    }
    const bfloat y0 = bfloat(cv0 * float(bfloat(acc0)));
    const bfloat y1 = bfloat(cv1 * float(bfloat(acc1)));

    device bfloat* dst = out + (size_t)r * H;
    dst[c0] = y0;
    dst[c1] = y1;
    if (p.sums_stride != 0) {
      const float gs = simd_sum(float(y0) + float(y1));
      if (lane == 0) sums[(size_t)g * p.sums_stride + r] = gs;
    }

    for (uint k = 0; k < H_hist; k++) {
      ring0[k] = ring0[k + 1];
      ring1[k] = ring1[k + 1];
    }
  }

  if (p.commit_hist != 0) {
    for (uint k = 0; k < H_hist; k++) {
      hist[(size_t)k * H + c0] = bfloat(ring0[k]);
      hist[(size_t)k * H + c1] = bfloat(ring1[k]);
    }
  }
}

struct AddBiasParams {
  uint rows;
  uint cols;
};

// Optional row-wise bias addition (used when LFM2 conv_bias == true on conv.out_proj).
kernel void add_bias_bf16(device bfloat* x [[buffer(0)]],
                          const device bfloat* bias [[buffer(1)]],
                          constant AddBiasParams& p [[buffer(2)]],
                          uint2 tid [[thread_position_in_grid]]) {
  if (tid.x >= p.cols || tid.y >= p.rows) return;
  const size_t idx = (size_t)tid.y * p.cols + tid.x;
  x[idx] = bfloat(float(x[idx]) + float(bias[tid.x]));
}

struct SpliceParams {
  uint rows;
  uint hidden;
};

// Multimodal VLM embedding splice: overwrites h[row, :] with vis_embs[splice_idx[row], :] where splice_idx[row] >= 0.
// Dispatched AFTER embed_lookup and optional Gemma embed_scale so vision tokens stay unscaled.
kernel void splice_vision_embs(device bfloat* h [[buffer(0)]],
                               const device bfloat* vis_embs [[buffer(1)]],
                               const device int* splice_idx [[buffer(2)]],
                               constant SpliceParams& p [[buffer(3)]],
                               uint2 tid [[thread_position_in_grid]]) {
  if (tid.x >= p.hidden || tid.y >= p.rows) return;
  const int vi = splice_idx[tid.y];
  if (vi < 0) return;
  h[(size_t)tid.y * p.hidden + tid.x] = vis_embs[(size_t)vi * p.hidden + tid.x];
}
