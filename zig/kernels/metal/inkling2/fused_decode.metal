// Inkling 2 fused decode kernels: conv+store+residual, q/rel/kv prep, dual loop mix, and core restart.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

constant constexpr uint I2_FUSED_TG = 256;
constant constexpr uint I2_FUSED_SGS = 8;
constant constexpr uint I2_FUSED_CONV_TAPS = 4;
constant constexpr uint I2_FUSED_HEAD_LANE_ELEMS = 8;
constant constexpr uint I2_FUSED_LP_MAX_SOURCES = 9;
constant constexpr uint I2_FUSED_LP_MAX_QUERIES = 4;

inline float i2_fused_max(float a, float b) {
  if (isnan(a) || isnan(b)) return NAN;
  return a > b ? a : b;
}

inline float i2_fused_rsqrt_mean(float sum, uint n, float eps) {
  return metal::precise::rsqrt(sum / float(n) + eps);
}

inline float i2_fused_clip_factor(float sum_squares, float threshold) {
  const float f = threshold / (metal::precise::sqrt(sum_squares) + 1e-6f);
  return isnan(f) ? f : min(f, 1.0f);
}

inline float i2_fused_conv_in(const device bfloat* x, const device bfloat* hist, uint channels, uint pos0, uint ring,
                              uint t, uint lag, uint c) {
  if (t >= lag) return float(x[ulong(t - lag) * channels + c]);
  const long position = long(pos0) + long(t) - long(lag);
  if (position < 0) return 0.0f;
  return float(hist[ulong(position % long(ring)) * channels + c]);
}

// Fusion 1A: post-attention convStatic + convStore + residualAttn (+ optional block copy), 1 <= rows <= ring - 3.
struct I2ConvResAttnArgs {
  uint rows, channels, pos0, ring;
  uint has_block, save_block, pad0, pad1;
  float divisor;
};

[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_conv_store_residual_attn(
    const device bfloat* x [[buffer(0)]], device bfloat* hist [[buffer(1)]], const device bfloat* w [[buffer(2)]],
    const device bfloat* hin [[buffer(3)]], device bfloat* block_dst [[buffer(4)]], device float* out [[buffer(5)]],
    constant I2ConvResAttnArgs& a [[buffer(6)]], uint2 at [[thread_position_in_grid]]) {
  const uint c = at.x, t = at.y;
  if (c >= a.channels || t >= a.rows) return;
  const ulong idx = ulong(t) * a.channels + c;
  const bfloat cur_bf = x[idx];
  float sum = 0.0f;
  for (uint j = 0; j < I2_FUSED_CONV_TAPS; ++j) {
    const float tap = i2_fused_conv_in(x, hist, a.channels, a.pos0, a.ring, t, I2_FUSED_CONV_TAPS - 1 - j, c);
    sum = fma(float(w[ulong(c) * I2_FUSED_CONV_TAPS + j]), tap, sum);
  }
  const float conv_val = float(bfloat((sum + float(cur_bf)) / a.divisor));
  hist[ulong((a.pos0 + t) % a.ring) * a.channels + c] = cur_bf;
  if (a.has_block != 0) {
    out[idx] = float(hin[idx]) + conv_val;
  } else {
    out[idx] = conv_val;
    if (a.save_block != 0) block_dst[idx] = hin[idx];
  }
}

// Fusion 1B: post-MLP convStatic + convStore + residualMlp, valid for 1 <= rows <= ring - 3.
struct I2ConvResMlpArgs {
  uint rows, channels, pos0, ring;
  float divisor;
};

[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_conv_store_residual_mlp(
    const device bfloat* x [[buffer(0)]], device bfloat* hist [[buffer(1)]], const device bfloat* w [[buffer(2)]],
    const device float* h32 [[buffer(3)]], device bfloat* out [[buffer(4)]],
    constant I2ConvResMlpArgs& a [[buffer(5)]], uint2 at [[thread_position_in_grid]]) {
  const uint c = at.x, t = at.y;
  if (c >= a.channels || t >= a.rows) return;
  const ulong idx = ulong(t) * a.channels + c;
  const bfloat cur_bf = x[idx];
  float sum = 0.0f;
  for (uint j = 0; j < I2_FUSED_CONV_TAPS; ++j) {
    const float tap = i2_fused_conv_in(x, hist, a.channels, a.pos0, a.ring, t, I2_FUSED_CONV_TAPS - 1 - j, c);
    sum = fma(float(w[ulong(c) * I2_FUSED_CONV_TAPS + j]), tap, sum);
  }
  const float conv_val = float(bfloat((sum + float(cur_bf)) / a.divisor));
  hist[ulong((a.pos0 + t) % a.ring) * a.channels + c] = cur_bf;
  out[idx] = bfloat(h32[idx] + conv_val);
}

// Fusion 2: qPrep+relPrep (tg.x < q_heads) and K/V conv+store+kvPrep (tg.x >= q_heads), grid (q_heads+kv_heads, rows).
struct I2FusedQkvArgs {
  uint rows, q_heads, kv_heads, dim;
  uint rel_dim, pos0, conv_ring, kv_ring;
  uint log_scaled, floor_positions, pad0, pad1;
  float qk_threshold, rel_threshold, alpha, eps;
  float conv_divisor;
};

inline void i2_fused_clip_head(device bfloat* x, const device bfloat* scale, uint head, uint row, uint heads, uint dim,
                               float threshold, uint log_scaled, float tau, uint lane) {
  const ulong base = (ulong(row) * heads + head) * dim;
  float v[I2_FUSED_HEAD_LANE_ELEMS];
  float ss = 0.0f;
  for (uint e = 0; e < I2_FUSED_HEAD_LANE_ELEMS; ++e) {
    const uint i = lane + 32 * e;
    v[e] = 0.0f;
    if (i < dim) {
      v[e] = float(x[base + i]) * float(scale[ulong(head) * dim + i]);
      ss = fma(v[e], v[e], ss);
    }
  }
  const float factor = i2_fused_clip_factor(simd_sum(ss), threshold);
  for (uint e = 0; e < I2_FUSED_HEAD_LANE_ELEMS; ++e) {
    const uint i = lane + 32 * e;
    if (i >= dim) continue;
    float r = float(bfloat(v[e] * factor));
    if (log_scaled != 0) r = float(bfloat(r * tau));
    x[base + i] = bfloat(r);
  }
}

inline float i2_fused_tau(constant I2FusedQkvArgs& a, uint row) {
  if (a.log_scaled == 0) return 1.0f;
  const float ratio = max(float(a.pos0 + row + 1) / float(a.floor_positions), 1.0f);
  return 1.0f + a.alpha * metal::precise::log(ratio);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void inkling2_fused_q_rel_prep(
    device bfloat* q [[buffer(0)]], const device bfloat* q_scale [[buffer(1)]], device bfloat* rel [[buffer(2)]],
    const device bfloat* rel_scale [[buffer(3)]], constant I2FusedQkvArgs& a [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, row = tg.y;
  const float tau = i2_fused_tau(a, row);
  i2_fused_clip_head(q, q_scale, head, row, a.q_heads, a.dim, a.qk_threshold, a.log_scaled, tau, lane);
  i2_fused_clip_head(rel, rel_scale, head, row, a.q_heads, a.rel_dim, a.rel_threshold, a.log_scaled, tau, lane);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void inkling2_fused_qkv_prep_static(
    device bfloat* q [[buffer(0)]], const device bfloat* q_scale [[buffer(1)]], device bfloat* rel [[buffer(2)]],
    const device bfloat* rel_scale [[buffer(3)]], const device bfloat* kraw [[buffer(4)]],
    const device bfloat* vraw [[buffer(5)]], device bfloat* khist [[buffer(6)]], device bfloat* vhist [[buffer(7)]],
    const device bfloat* kw [[buffer(8)]], const device bfloat* vw [[buffer(9)]], device bfloat* kcache [[buffer(10)]],
    device bfloat* vcache [[buffer(11)]], constant I2FusedQkvArgs& a [[buffer(12)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg.y;
  if (tg.x < a.q_heads) {
    const uint head = tg.x;
    const float tau = i2_fused_tau(a, row);
    i2_fused_clip_head(q, q_scale, head, row, a.q_heads, a.dim, a.qk_threshold, a.log_scaled, tau, lane);
    i2_fused_clip_head(rel, rel_scale, head, row, a.q_heads, a.rel_dim, a.rel_threshold, a.log_scaled, tau, lane);
    return;
  }
  const uint kv_head = tg.x - a.q_heads;
  const uint channels = a.kv_heads * a.dim;
  const ulong hist_slot = ulong((a.pos0 + row) % a.conv_ring) * channels;
  float kv[I2_FUSED_HEAD_LANE_ELEMS], vv[I2_FUSED_HEAD_LANE_ELEMS];
  float kss = 0.0f, vss = 0.0f;
  for (uint e = 0; e < I2_FUSED_HEAD_LANE_ELEMS; ++e) {
    const uint i = lane + 32 * e;
    kv[e] = 0.0f;
    vv[e] = 0.0f;
    if (i < a.dim) {
      const uint c = kv_head * a.dim + i;
      const ulong xidx = ulong(row) * channels + c;
      const bfloat k_cur = kraw[xidx];
      const bfloat v_cur = vraw[xidx];
      float ksum = 0.0f, vsum = 0.0f;
      for (uint j = 0; j < I2_FUSED_CONV_TAPS; ++j) {
        const uint lag = I2_FUSED_CONV_TAPS - 1 - j;
        const float kt = i2_fused_conv_in(kraw, khist, channels, a.pos0, a.conv_ring, row, lag, c);
        const float vt = i2_fused_conv_in(vraw, vhist, channels, a.pos0, a.conv_ring, row, lag, c);
        ksum = fma(float(kw[ulong(c) * I2_FUSED_CONV_TAPS + j]), kt, ksum);
        vsum = fma(float(vw[ulong(c) * I2_FUSED_CONV_TAPS + j]), vt, vsum);
      }
      kv[e] = float(bfloat((ksum + float(k_cur)) / a.conv_divisor));
      vv[e] = float(bfloat((vsum + float(v_cur)) / a.conv_divisor));
      khist[hist_slot + c] = k_cur;
      vhist[hist_slot + c] = v_cur;
      kss = fma(kv[e], kv[e], kss);
      vss = fma(vv[e], vv[e], vss);
    }
  }
  if (a.kv_ring != 0 && row + a.kv_ring < a.rows) return;
  const float kfactor = i2_fused_clip_factor(simd_sum(kss), a.qk_threshold);
  const float vinv = i2_fused_rsqrt_mean(simd_sum(vss), a.dim, a.eps);
  const ulong position = ulong(a.pos0) + row;
  const ulong slot = a.kv_ring == 0 ? position : position % a.kv_ring;
  const ulong dst = (slot * a.kv_heads + kv_head) * a.dim;
  for (uint e = 0; e < I2_FUSED_HEAD_LANE_ELEMS; ++e) {
    const uint i = lane + 32 * e;
    if (i >= a.dim) continue;
    kcache[dst + i] = bfloat(kv[e] * kfactor);
    vcache[dst + i] = bfloat(vv[e] * vinv);
  }
}

[[kernel, max_total_threads_per_threadgroup(32)]] void inkling2_fused_qkv_prep_dynamic(
    device bfloat* q [[buffer(0)]], const device bfloat* q_scale [[buffer(1)]], device bfloat* rel [[buffer(2)]],
    const device bfloat* rel_scale [[buffer(3)]], const device bfloat* kraw [[buffer(4)]],
    const device bfloat* vraw [[buffer(5)]], device bfloat* khist [[buffer(6)]], device bfloat* vhist [[buffer(7)]],
    const device bfloat* kw [[buffer(8)]], const device bfloat* vw [[buffer(9)]], device bfloat* kcache [[buffer(10)]],
    device bfloat* vcache [[buffer(11)]], const device bfloat* dynk [[buffer(12)]],
    const device bfloat* dynv [[buffer(13)]], constant I2FusedQkvArgs& a [[buffer(14)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg.y;
  if (tg.x < a.q_heads) {
    const uint head = tg.x;
    const float tau = i2_fused_tau(a, row);
    i2_fused_clip_head(q, q_scale, head, row, a.q_heads, a.dim, a.qk_threshold, a.log_scaled, tau, lane);
    i2_fused_clip_head(rel, rel_scale, head, row, a.q_heads, a.rel_dim, a.rel_threshold, a.log_scaled, tau, lane);
    return;
  }
  const uint kv_head = tg.x - a.q_heads;
  const uint channels = a.kv_heads * a.dim;
  const ulong hist_slot = ulong((a.pos0 + row) % a.conv_ring) * channels;
  float kv[I2_FUSED_HEAD_LANE_ELEMS], vv[I2_FUSED_HEAD_LANE_ELEMS];
  float kss = 0.0f, vss = 0.0f;
  for (uint e = 0; e < I2_FUSED_HEAD_LANE_ELEMS; ++e) {
    const uint i = lane + 32 * e;
    kv[e] = 0.0f;
    vv[e] = 0.0f;
    if (i < a.dim) {
      const uint c = kv_head * a.dim + i;
      const ulong xidx = ulong(row) * channels + c;
      const bfloat k_cur = kraw[xidx];
      const bfloat v_cur = vraw[xidx];
      float ksum = 0.0f, vsum = 0.0f;
      for (uint lag = 0; lag < I2_FUSED_CONV_TAPS; ++lag) {
        const ulong didx = ulong(row) * a.kv_heads * I2_FUSED_CONV_TAPS + kv_head * I2_FUSED_CONV_TAPS + lag;
        const float kw_val = float(kw[ulong(lag) * channels + c]) + float(dynk[didx]);
        const float vw_val = float(vw[ulong(lag) * channels + c]) + float(dynv[didx]);
        const float kt = i2_fused_conv_in(kraw, khist, channels, a.pos0, a.conv_ring, row, lag, c);
        const float vt = i2_fused_conv_in(vraw, vhist, channels, a.pos0, a.conv_ring, row, lag, c);
        ksum = fma(kw_val, kt, ksum);
        vsum = fma(vw_val, vt, vsum);
      }
      kv[e] = float(bfloat((ksum + float(k_cur)) / a.conv_divisor));
      vv[e] = float(bfloat((vsum + float(v_cur)) / a.conv_divisor));
      khist[hist_slot + c] = k_cur;
      vhist[hist_slot + c] = v_cur;
      kss = fma(kv[e], kv[e], kss);
      vss = fma(vv[e], vv[e], vss);
    }
  }
  if (a.kv_ring != 0 && row + a.kv_ring < a.rows) return;
  const float kfactor = i2_fused_clip_factor(simd_sum(kss), a.qk_threshold);
  const float vinv = i2_fused_rsqrt_mean(simd_sum(vss), a.dim, a.eps);
  const ulong position = ulong(a.pos0) + row;
  const ulong slot = a.kv_ring == 0 ? position : position % a.kv_ring;
  const ulong dst = (slot * a.kv_heads + kv_head) * a.dim;
  for (uint e = 0; e < I2_FUSED_HEAD_LANE_ELEMS; ++e) {
    const uint i = lane + 32 * e;
    if (i >= a.dim) continue;
    kcache[dst + i] = bfloat(kv[e] * kfactor);
    vcache[dst + i] = bfloat(vv[e] * vinv);
  }
}

// Fusion 3A: dual loop_mix (messages + gated attn mix) in one dispatch over grid = (rows, 2, 1).
struct I2DualLoopMixArgs {
  uint hidden, block_stride, out_stride, partial_pitch;
  uint blocks0, queries0, blocks1, queries1;
  float eps_unweighted, scale;
};

inline float i2_fused_lp_source(const device bfloat* blocks, const device bfloat* partial, uint nblocks,
                                uint block_stride, uint partial_pitch, uint hidden, uint s, uint row, uint i) {
  if (s < nblocks) return float(blocks[ulong(s) * block_stride + ulong(row) * hidden + i]);
  return float(partial[ulong(row) * partial_pitch + i]);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_fused_dual_loop_mix(
    const device bfloat* blocks0 [[buffer(0)]], const device bfloat* partial0 [[buffer(1)]],
    const device float* q0_0 [[buffer(2)]], const device float* q0_1 [[buffer(3)]],
    const device float* q0_2 [[buffer(4)]], const device float* q0_3 [[buffer(5)]], device float* out0 [[buffer(6)]],
    const device bfloat* blocks1 [[buffer(7)]], const device bfloat* partial1 [[buffer(8)]],
    const device float* q1_0 [[buffer(9)]], const device float* q1_1 [[buffer(10)]],
    const device float* q1_2 [[buffer(11)]], device float* out1 [[buffer(12)]],
    constant I2DualLoopMixArgs& a [[buffer(13)]], uint2 grid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[I2_FUSED_SGS][I2_FUSED_LP_MAX_QUERIES * I2_FUSED_LP_MAX_SOURCES];
  threadgroup float inv_tg[I2_FUSED_LP_MAX_SOURCES];

  const uint row = grid.x;
  const bool second = grid.y != 0;
  const device bfloat* blocks = second ? blocks1 : blocks0;
  const device bfloat* partial = second ? partial1 : partial0;
  const device float* q0 = second ? q1_0 : q0_0;
  const device float* q1 = second ? q1_1 : q0_1;
  const device float* q2 = second ? q1_2 : q0_2;
  const device float* q3 = second ? q1_0 : q0_3;
  device float* out = second ? out1 : out0;
  const uint nblocks = second ? a.blocks1 : a.blocks0;
  const uint nqueries = second ? a.queries1 : a.queries0;
  const uint S = nblocks + 1;
  const uint H = a.hidden;

  float ss[I2_FUSED_LP_MAX_SOURCES];
  for (uint s = 0; s < I2_FUSED_LP_MAX_SOURCES; ++s) {
    ss[s] = 0.0f;
    if (s >= S) continue;
    float acc = 0.0f;
    for (uint i = tid; i < H; i += I2_FUSED_TG) {
      const float v = i2_fused_lp_source(blocks, partial, nblocks, a.block_stride, a.partial_pitch, H, s, row, i);
      acc = fma(v, v, acc);
    }
    ss[s] = simd_sum(acc);
  }
  if (lane == 0)
    for (uint s = 0; s < S; ++s) part[sg][s] = ss[s];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < S) {
    float t = part[0][tid];
    for (uint g = 1; g < I2_FUSED_SGS; ++g) t += part[g][tid];
    inv_tg[tid] = i2_fused_rsqrt_mean(t, H, a.eps_unweighted);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float dots[I2_FUSED_LP_MAX_QUERIES][I2_FUSED_LP_MAX_SOURCES];
  for (uint q = 0; q < I2_FUSED_LP_MAX_QUERIES; ++q) {
    const device float* qv = q == 0 ? q0 : (q == 1 ? q1 : (q == 2 ? q2 : q3));
    for (uint s = 0; s < I2_FUSED_LP_MAX_SOURCES; ++s) {
      dots[q][s] = 0.0f;
      if (q >= nqueries || s >= S) continue;
      float acc = 0.0f;
      for (uint i = tid; i < H; i += I2_FUSED_TG) {
        const float v = i2_fused_lp_source(blocks, partial, nblocks, a.block_stride, a.partial_pitch, H, s, row, i);
        acc = fma(qv[i], v, acc);
      }
      dots[q][s] = simd_sum(acc);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0)
    for (uint q = 0; q < nqueries; ++q)
      for (uint s = 0; s < S; ++s) part[sg][q * I2_FUSED_LP_MAX_SOURCES + s] = dots[q][s];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float coef[I2_FUSED_LP_MAX_QUERIES][I2_FUSED_LP_MAX_SOURCES];
  for (uint q = 0; q < I2_FUSED_LP_MAX_QUERIES; ++q) {
    for (uint s = 0; s < I2_FUSED_LP_MAX_SOURCES; ++s) coef[q][s] = 0.0f;
    if (q >= nqueries) continue;
    float score[I2_FUSED_LP_MAX_SOURCES];
    float m = -INFINITY;
    for (uint s = 0; s < S; ++s) {
      float t = part[0][q * I2_FUSED_LP_MAX_SOURCES + s];
      for (uint g = 1; g < I2_FUSED_SGS; ++g) t += part[g][q * I2_FUSED_LP_MAX_SOURCES + s];
      score[s] = (t * inv_tg[s]) * a.scale;
      m = i2_fused_max(m, score[s]);
    }
    float l = 0.0f;
    for (uint s = 0; s < S; ++s) {
      score[s] = score[s] == -INFINITY ? 0.0f : metal::precise::exp(score[s] - m);
      l += score[s];
    }
    for (uint s = 0; s < S; ++s) coef[q][s] = (score[s] / l) * inv_tg[s];
  }

  for (uint q = 0; q < I2_FUSED_LP_MAX_QUERIES; ++q) {
    if (q >= nqueries) break;
    device float* dst = out + ulong(q) * a.out_stride + ulong(row) * H;
    for (uint i = tid; i < H; i += I2_FUSED_TG) {
      float u = 0.0f;
      for (uint s = 0; s < S; ++s) {
        const float v = i2_fused_lp_source(blocks, partial, nblocks, a.block_stride, a.partial_pitch, H, s, row, i);
        u = fma(coef[q][s], v, u);
      }
      dst[i] = u;
    }
  }
}

// Fusion 3B: enter() core restart bank + partial reset in one dispatch.
struct I2FusedEnterArgs {
  uint hidden, rows, bank_rows, pad;
};

[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_fused_enter(
    const device bfloat* src_blocks [[buffer(0)]], device bfloat* bank [[buffer(1)]],
    const device bfloat* entry_block [[buffer(2)]], device bfloat* hid [[buffer(3)]],
    device bfloat* bank_h [[buffer(4)]], constant I2FusedEnterArgs& a [[buffer(5)]],
    uint2 at [[thread_position_in_grid]]) {
  const uint c = at.x, r = at.y;
  if (c >= a.hidden) return;
  if (r < a.rows) {
    const ulong idx = ulong(r) * a.hidden + c;
    const bfloat prev = hid[idx];
    bank_h[idx] = prev;
    hid[idx] = entry_block[idx];
  } else {
    const uint br = r - a.rows;
    if (br < a.bank_rows) {
      const ulong idx = ulong(br) * a.hidden + c;
      bank[idx] = src_blocks[idx];
    }
  }
}
