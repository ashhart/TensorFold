#include <metal_stdlib>
#pragma METAL fp contract(off)
using namespace metal;

// Inkling 2 Mini glue. Every reduction is FP32; bfloat stores round nearest-even.
constant float INKLING_EPS = 1e-6f;
constant float INKLING_AR_SCALE = 0x1.555556p-6f; // binary32(1 / 48)

inline float ink_sigmoid(float x) {
  const float e = precise::exp(-fabs(x));
  return x < 0.0f ? precise::divide(e, 1.0f + e) : precise::divide(1.0f, 1.0f + e);
}
inline float ink_cap(float x, float c) {
  return c * precise::tanh(precise::divide(x, c));
}
inline float ink_value(const device bfloat* blocks, const device float* partial,
                       uint source, uint sources, uint d) {
  return source + 1 == sources ? partial[d] : float(blocks[size_t(source) * 2304 + d]);
}

// Two BF16 pairs of w and x multiplied and added to sum in column order.
inline float ink_pair(float sum, uint w, uint x) {
  sum += as_type<float>(w << 16u) * as_type<float>(x << 16u);
  return sum + as_type<float>(w & 0xffff0000u) * as_type<float>(x & 0xffff0000u);
}
// One thread's dot over k columns, k a multiple of 32 and both rows 16-byte aligned: the serial chain, loads batched.
inline float ink_chain(const device bfloat* w, const device bfloat* x, uint k) {
  float sum = 0.0f;
  for (uint d = 0; d < k; d += 32) {
    uint4 a[4], b[4];
    for (uint i = 0; i < 4; ++i) {
      a[i] = *(const device uint4*)(w + d + 8 * i);
      b[i] = *(const device uint4*)(x + d + 8 * i);
    }
    for (uint i = 0; i < 4; ++i) {
      sum = ink_pair(sum, a[i].x, b[i].x);
      sum = ink_pair(sum, a[i].y, b[i].y);
      sum = ink_pair(sum, a[i].z, b[i].z);
      sum = ink_pair(sum, a[i].w, b[i].w);
    }
  }
  return sum;
}

struct InkEmbedArgs { uint rows; };
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_embed_norm(
    const device uint* ids [[buffer(0)]], const device bfloat* table [[buffer(1)]],
    const device bfloat* weight [[buffer(2)]], device bfloat* out [[buffer(3)]],
    constant InkEmbedArgs& a [[buffer(4)]], uint t [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows) return;
  const device bfloat* x = table + size_t(ids[row]) * 2304;
  float part = 0.0f;
  for (uint d = t; d < 2304; d += 256) { const float v = float(x[d]); part += v * v; }
  threadgroup float red[8], inverse;
  const float reduced = simd_sum(part);
  if (lane == 0) red[t / 32] = reduced;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float total = 0.0f;
    for (uint j = 0; j < 8; ++j) total += red[j];
    inverse = precise::rsqrt(precise::divide(total, 2304.0f) + INKLING_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = t; d < 2304; d += 256)
    out[size_t(row) * 2304 + d] = bfloat(float(weight[d]) * (float(x[d]) * inverse));
}

struct InkARArgs { uint sources, queries, rows, blocks_stride, partial_stride, out_stride; };
struct InkCachedARArgs { uint sources, queries; };
struct InkBlockInvRmsArgs { uint rows, slot; };
inline float ink_partial_inv_rms(float ss, uint t, uint lane, threadgroup float* red) {
  const float reduced = simd_sum(ss);
  if (lane == 0) red[t / 32] = reduced;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t != 0) return 0.0f;
  float total = 0.0f;
  for (uint j = 0; j < 8; ++j) total += red[j];
  return precise::rsqrt(precise::divide(total, 2304.0f) + INKLING_EPS);
}
inline void ink_ar_softmax(uint sources, uint t, threadgroup float* dots, threadgroup float* weights) {
  if (t == 0) {
    float peak = -INFINITY;
    for (uint s = 0; s < sources; ++s) {
      float dot = 0.0f;
      for (uint j = 0; j < 8; ++j) dot += dots[s * 8 + j];
      weights[s] = dot * INKLING_AR_SCALE;
      peak = max(peak, weights[s]);
    }
    float den = 0.0f;
    for (uint s = 0; s < sources; ++s) { weights[s] = precise::exp(weights[s] - peak); den += weights[s]; }
    for (uint s = 0; s < sources; ++s) weights[s] = precise::divide(weights[s], den);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
inline float ink_reg_value(
    const device bfloat* blocks, const thread float* partial_regs,
    const thread bfloat* override_bf, uint override_slot,
    uint s, uint sources, uint k, uint d) {
  if (s + 1 == sources) return partial_regs[k];
  if (s == override_slot) return float(override_bf[k]);
  return float(blocks[size_t(s) * 2304 + d]);
}
inline void ink_ar_finish(
    const device bfloat* blocks, const device float* partial,
    const device float* queries, device float* out,
    uint sources, uint q, uint t, uint lane,
    threadgroup float* dots, threadgroup float* means, threadgroup float* weights) {
  for (uint s = 0; s < sources; ++s) {
    float dot = 0.0f;
    for (uint d = t; d < 2304; d += 256) {
      const float z = ink_value(blocks, partial, s, sources, d) * means[s];
      dot += queries[size_t(q) * 2304 + d] * z;
    }
    const float reduced = simd_sum(dot);
    if (lane == 0) dots[s * 8 + t / 32] = reduced;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ink_ar_softmax(sources, t, dots, weights);
  for (uint d = t; d < 2304; d += 256) {
    float acc = 0.0f;
    for (uint s = 0; s < sources; ++s)
      acc += weights[s] * (ink_value(blocks, partial, s, sources, d) * means[s]);
    out[size_t(q) * 2304 + d] = acc;
  }
}
inline void ink_ar_finish_regs(
    const device bfloat* blocks, const thread float* partial_regs,
    const thread bfloat* override_bf, uint override_slot,
    const device float* query, device float* out,
    uint sources, uint t, uint lane,
    threadgroup float* dots, threadgroup float* means, threadgroup float* weights) {
  for (uint s = 0; s < sources; ++s) {
    float dot = 0.0f;
    for (uint k = 0; k < 9; ++k) {
      const uint d = t + k * 256;
      const float z = ink_reg_value(blocks, partial_regs, override_bf, override_slot, s, sources, k, d) * means[s];
      dot += query[d] * z;
    }
    const float reduced = simd_sum(dot);
    if (lane == 0) dots[s * 8 + t / 32] = reduced;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ink_ar_softmax(sources, t, dots, weights);
  for (uint k = 0; k < 9; ++k) {
    const uint d = t + k * 256;
    float acc = 0.0f;
    for (uint s = 0; s < sources; ++s)
      acc += weights[s] * (ink_reg_value(blocks, partial_regs, override_bf, override_slot, s, sources, k, d) * means[s]);
    out[d] = acc;
  }
}
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_ar(
    const device bfloat* blocks_base [[buffer(0)]], const device float* partial_base [[buffer(1)]],
    const device float* queries [[buffer(2)]], device float* out_base [[buffer(3)]],
    constant InkARArgs& a [[buffer(4)]], uint t [[thread_index_in_threadgroup]],
    uint2 at [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint q = at.x, row = at.y;
  if (q >= a.queries || row >= a.rows || a.sources == 0 || a.sources > 5) return;
  // Row `row`'s blocks, partial sums and output start that row's stride past the first row's.
  const device bfloat* blocks = blocks_base + size_t(row) * a.blocks_stride;
  const device float* partial = partial_base + size_t(row) * a.partial_stride;
  device float* out = out_base + size_t(row) * a.out_stride;
  threadgroup float red[2 * 5 * 8], means[5], weights[5];
  for (uint s = 0; s < a.sources; ++s) {
    float ss = 0.0f;
    for (uint d = t; d < 2304; d += 256) {
      const float x = ink_value(blocks, partial, s, a.sources, d);
      ss += x * x;
    }
    const float reduced = simd_sum(ss);
    if (lane == 0) red[s * 8 + t / 32] = reduced;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) for (uint s = 0; s < a.sources; ++s) {
    float ss = 0.0f;
    for (uint j = 0; j < 8; ++j) ss += red[s * 8 + j];
    means[s] = precise::rsqrt(precise::divide(ss, 2304.0f) + INKLING_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ink_ar_finish(blocks, partial, queries, out, a.sources, q, t, lane, red + 40, means, weights);
}
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_block_inv_rms(
    const device bfloat* blocks [[buffer(0)]], device float* inv_rms [[buffer(1)]],
    constant InkBlockInvRmsArgs& a [[buffer(2)]], uint t [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows || a.slot >= 4) return;
  const device bfloat* x = blocks + (size_t(row) * 4 + a.slot) * 2304;
  float ss = 0.0f;
  for (uint d = t; d < 2304; d += 256) {
    const float v = float(x[d]);
    ss += v * v;
  }
  threadgroup float red[8];
  const float inv = ink_partial_inv_rms(ss, t, lane, red);
  if (t == 0) inv_rms[size_t(row) * 4 + a.slot] = inv;
}
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_cached_ar(
    const device bfloat* blocks [[buffer(0)]], const device float* partial [[buffer(1)]],
    const device float* block_inv_rms [[buffer(2)]], const device float* queries [[buffer(3)]],
    device float* out [[buffer(4)]], constant InkCachedARArgs& a [[buffer(5)]],
    uint t [[thread_position_in_threadgroup]], uint q [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (q >= a.queries || a.sources == 0 || a.sources > 5) return;
  threadgroup float red[8], dots[5 * 8], means[5], weights[5];
  if (t + 1 < a.sources) means[t] = block_inv_rms[t];
  float ss = 0.0f;
  for (uint d = t; d < 2304; d += 256) {
    const float x = partial[d];
    ss += x * x;
  }
  const float inv = ink_partial_inv_rms(ss, t, lane, red);
  if (t == 0) means[a.sources - 1] = inv;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ink_ar_finish(blocks, partial, queries, out, a.sources, q, t, lane, dots, means, weights);
}
struct InkAttnPartialArArgs { uint rows, sources, boundary; };
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_attn_partial_ar(
    const device bfloat* p [[buffer(0)]], const device bfloat* attn_conv [[buffer(1)]],
    device bfloat* blocks [[buffer(2)]], device float* block_inv_rms [[buffer(3)]],
    device float* partial [[buffer(4)]], const device float* query [[buffer(5)]],
    device float* out [[buffer(6)]], constant InkAttnPartialArArgs& a [[buffer(7)]],
    uint t [[thread_position_in_threadgroup]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows || a.sources == 0 || a.sources > 5 || a.boundary > 4) return;
  const device bfloat* p_row = p + size_t(row) * 2304;
  const device bfloat* x_row = attn_conv + size_t(row) * 2304;
  device bfloat* blocks_row = blocks + size_t(row) * 4 * 2304;
  device float* inv_row = block_inv_rms + size_t(row) * 4;
  device float* partial_row = partial + size_t(row) * 2304;
  device float* out_row = out + size_t(row) * 2304;
  float partial_regs[9];
  bfloat p_bf[9];
  float part_ss = 0.0f;
  float block_ss = 0.0f;
  for (uint k = 0; k < 9; ++k) {
    const uint d = t + k * 256;
    const float x = float(x_row[d]);
    float val;
    if (a.boundary) {
      const bfloat pb = p_row[d];
      p_bf[k] = pb;
      blocks_row[size_t(a.boundary - 1) * 2304 + d] = pb;
      const float pv = float(pb);
      block_ss += pv * pv;
      val = x;
    } else {
      val = float(p_row[d]) + x;
    }
    partial_row[d] = val;
    partial_regs[k] = val;
    part_ss += val * val;
  }
  threadgroup float red[8], block_red[8], dots[5 * 8], means[5], weights[5];
  if (t + 1 < a.sources && !(a.boundary != 0 && t == a.boundary - 1)) means[t] = inv_row[t];
  const float part_inv = ink_partial_inv_rms(part_ss, t, lane, red);
  float block_inv = 0.0f;
  if (a.boundary) block_inv = ink_partial_inv_rms(block_ss, t, lane, block_red);
  if (t == 0) {
    if (a.boundary) {
      inv_row[a.boundary - 1] = block_inv;
      if (a.boundary < a.sources) means[a.boundary - 1] = block_inv;
    }
    means[a.sources - 1] = part_inv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint override_slot = a.boundary ? (a.boundary - 1) : 5;
  ink_ar_finish_regs(blocks_row, partial_regs, p_bf, override_slot, query, out_row, a.sources, t, lane, dots, means, weights);
}
struct InkMlpPartialArArgs { uint rows, sources, queries; };
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_mlp_partial_ar(
    const device float* partial_in [[buffer(0)]], const device bfloat* mlp_conv [[buffer(1)]],
    const device bfloat* blocks [[buffer(2)]], const device float* block_inv_rms [[buffer(3)]],
    const device float* queries [[buffer(4)]], device bfloat* p_out [[buffer(5)]],
    device float* partial_out [[buffer(6)]], device float* ar_out [[buffer(7)]],
    constant InkMlpPartialArArgs& a [[buffer(8)]], uint t [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows || a.sources == 0 || a.sources > 5 || a.queries == 0 || a.queries > 3) return;
  const device float* pin_row = partial_in + size_t(row) * 2304;
  const device bfloat* conv_row = mlp_conv + size_t(row) * 2304;
  const device bfloat* blocks_row = blocks + size_t(row) * 4 * 2304;
  const device float* inv_row = block_inv_rms + size_t(row) * 4;
  device bfloat* p_row = p_out + size_t(row) * 2304;
  device float* pout_row = partial_out + size_t(row) * 2304;
  float partial_regs[9];
  float part_ss = 0.0f;
  for (uint k = 0; k < 9; ++k) {
    const uint d = t + k * 256;
    const bfloat next_p = bfloat(pin_row[d] + float(conv_row[d]));
    const float widened = float(next_p);
    p_row[d] = next_p;
    pout_row[d] = widened;
    partial_regs[k] = widened;
    part_ss += widened * widened;
  }
  threadgroup float red[8], dots[5 * 8], means[5], weights[5];
  if (t + 1 < a.sources) means[t] = inv_row[t];
  const float inv = ink_partial_inv_rms(part_ss, t, lane, red);
  if (t == 0) means[a.sources - 1] = inv;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bfloat dummy[1] = { bfloat(0.0f) };
  for (uint q = 0; q < a.queries; ++q) {
    const device float* query = queries + size_t(q) * 2304;
    device float* out_row = ar_out + (size_t(q) * a.rows + row) * 2304;
    ink_ar_finish_regs(blocks_row, partial_regs, dummy, 5, query, out_row, a.sources, t, lane, dots, means, weights);
    if (q + 1 < a.queries) threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

struct InkConvArgs { uint rows, channels; };
[[max_total_threads_per_threadgroup(32)]]
kernel void tf_inkling2_conv_static(
    const device bfloat* taps [[buffer(0)]], const device bfloat* weight [[buffer(1)]],
    device bfloat* out [[buffer(2)]], constant InkConvArgs& a [[buffer(3)]],
    uint2 at [[thread_position_in_grid]]) {
  const uint c = at.x, r = at.y;
  if (c >= a.channels || r >= a.rows) return;
  float sum = float(taps[size_t(r + 3) * a.channels + c]);
  for (uint tap = 0; tap < 4; ++tap)
    sum += float(weight[c * 4 + tap]) * float(taps[size_t(r + tap) * a.channels + c]);
  out[size_t(r) * a.channels + c] = bfloat(precise::divide(sum, precise::sqrt(5.0f)));
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_conv_dynamic(
    const device bfloat* taps [[buffer(0)]], const device bfloat* input [[buffer(1)]],
    const device bfloat* static_w [[buffer(2)]], const device bfloat* projection [[buffer(3)]],
    device bfloat* out [[buffer(4)]], constant InkConvArgs& a [[buffer(5)]],
    uint t [[thread_position_in_threadgroup]], uint r [[threadgroup_position_in_grid]]) {
  if (r >= a.rows) return;
  threadgroup float projected[32];
  if (t < 32) {
    projected[t] = float(bfloat(ink_chain(projection + size_t(t) * 2304, input + size_t(r) * 2304, 2304)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = t; c < a.channels; c += 256) {
    float sum = float(taps[size_t(r + 3) * a.channels + c]);
    for (uint lag = 0; lag < 4; ++lag) {
      const float w = float(static_w[lag * a.channels + c]) + projected[4 * (c / 128) + lag];
      sum += w * float(taps[size_t(r + 3 - lag) * a.channels + c]);
    }
    out[size_t(r) * a.channels + c] = bfloat(precise::divide(sum, 3.0f));
  }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void tf_inkling2_conv_static_inplace(
    const device bfloat* history [[buffer(0)]], const device bfloat* raw [[buffer(1)]],
    const device bfloat* weight [[buffer(2)]], device bfloat* out [[buffer(3)]],
    constant InkConvArgs& a [[buffer(4)]], uint2 at [[thread_position_in_grid]]) {
  const uint c = at.x, r = at.y;
  if (c >= a.channels || r >= a.rows) return;
  float sum = float(raw[size_t(r) * a.channels + c]);
  for (uint tap = 0; tap < 4; ++tap) {
    const uint idx = r + tap;
    const bfloat x = idx < 3 ? history[size_t(idx) * a.channels + c] : raw[size_t(idx - 3) * a.channels + c];
    sum += float(weight[c * 4 + tap]) * float(x);
  }
  out[size_t(r) * a.channels + c] = bfloat(precise::divide(sum, precise::sqrt(5.0f)));
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_conv_dynamic_inplace(
    const device bfloat* history [[buffer(0)]], const device bfloat* raw [[buffer(1)]],
    const device bfloat* input [[buffer(2)]], const device bfloat* static_w [[buffer(3)]],
    const device bfloat* projection [[buffer(4)]], device bfloat* out [[buffer(5)]],
    constant InkConvArgs& a [[buffer(6)]], uint t [[thread_position_in_threadgroup]],
    uint r [[threadgroup_position_in_grid]]) {
  if (r >= a.rows) return;
  threadgroup float projected[32];
  if (t < 32) {
    projected[t] = float(bfloat(ink_chain(projection + size_t(t) * 2304, input + size_t(r) * 2304, 2304)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = t; c < a.channels; c += 256) {
    float sum = float(raw[size_t(r) * a.channels + c]);
    for (uint lag = 0; lag < 4; ++lag) {
      const float w = float(static_w[lag * a.channels + c]) + projected[4 * (c / 128) + lag];
      const uint idx = r + 3 - lag;
      const bfloat x = idx < 3 ? history[size_t(idx) * a.channels + c] : raw[size_t(idx - 3) * a.channels + c];
      sum += w * float(x);
    }
    out[size_t(r) * a.channels + c] = bfloat(precise::divide(sum, 3.0f));
  }
}

struct InkRouteArgs { uint rows; };
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_router(
    const device bfloat* x [[buffer(0)]], const device bfloat* weight [[buffer(1)]],
    const device float* bias [[buffer(2)]], device uint* ids [[buffer(3)]],
    device float* weights [[buffer(4)]], constant InkRouteArgs& a [[buffer(5)]],
    uint t [[thread_position_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
  if (row >= a.rows) return;
  threadgroup float scores[128], key[128], chosen[8];
  threadgroup uint picked[8], selected[128], any_nan;
  if (t < 128) {
    scores[t] = ink_sigmoid(ink_chain(weight + size_t(t) * 2304, x + size_t(row) * 2304, 2304));
    key[t] = scores[t] + bias[t];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Ranks 0 to 7 by (key descending, index ascending) are the serial scan's picks; a NaN key runs the scan.
  if (t < 128) {
    uint rank = 0;
    bool nan = false;
    for (uint f = 0; f < 128; ++f) {
      nan = nan || isnan(key[f]);
      rank += (key[f] > key[t] || (key[f] == key[t] && f < t)) ? 1u : 0u;
    }
    selected[t] = rank < 8 ? 1u : 0u;
    // Every thread scanned all 128 keys; thread 0 alone stores the flag, so the barrier publishes one write.
    if (t == 0) any_nan = nan ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (any_nan != 0) {
    if (t == 0) {
      bool used[128] = {};
      for (uint j = 0; j < 8; ++j) {
        uint best = 128;
        for (uint e = 0; e < 128; ++e) if (!used[e]) {
          if (best == 128 || key[e] > key[best]) best = e;
        }
        used[best] = true; picked[j] = best; chosen[j] = scores[best];
      }
      // Combining, including the denominator, is in ascending expert index.
      for (uint j = 1; j < 8; ++j) {
        uint id = picked[j]; float score = chosen[j]; uint k = j;
        while (k > 0 && picked[k - 1] > id) { picked[k] = picked[k - 1]; chosen[k] = chosen[k - 1]; --k; }
        picked[k] = id; chosen[k] = score;
      }
    }
  } else if (t < 128 && selected[t] != 0) {
    // Combining, including the denominator, is in ascending expert index.
    uint at = 0;
    for (uint f = 0; f < t; ++f) at += selected[f];
    picked[at] = t; chosen[at] = scores[t];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 8) {
    float den = 1e-20f;
    for (uint j = 0; j < 8; ++j) den += chosen[j];
    ids[row * 8 + t] = picked[t];
    weights[row * 8 + t] = precise::divide(chosen[t], den) * 8.0f;
  }
}

struct InkActArgs { uint rows, width; };
[[max_total_threads_per_threadgroup(32)]]
kernel void tf_inkling2_swiglu(
    const device bfloat* gu [[buffer(0)]], device bfloat* out [[buffer(1)]],
    constant InkActArgs& a [[buffer(2)]], uint2 at [[thread_position_in_grid]]) {
  const uint j = at.x, r = at.y;
  if (j >= a.width || r >= a.rows) return;
  const float g = float(gu[(size_t(r) * a.width + j) * 2]);
  const float u = float(gu[(size_t(r) * a.width + j) * 2 + 1]);
  out[size_t(r) * a.width + j] = bfloat(ink_cap(g, 4.0f) * ink_sigmoid(g) * ink_cap(u, 25.0f));
}

struct InkLogitArgs { uint rows, vocab; };
[[max_total_threads_per_threadgroup(32)]]
kernel void tf_inkling2_logits(
    const device bfloat* h [[buffer(0)]], const device bfloat* unembed [[buffer(1)]],
    device float* out [[buffer(2)]], constant InkLogitArgs& a [[buffer(3)]],
    uint2 at [[thread_position_in_grid]]) {
  const uint v = at.x, r = at.y;
  if (v >= a.vocab || r >= a.rows) return;
  const float dot = ink_chain(unembed + size_t(v) * 2304, h + size_t(r) * 2304, 2304);
  // A9(a): unembedding rounds to bf16 before it widens for softcap30.
  out[size_t(r) * a.vocab + v] = ink_cap(float(bfloat(dot)), 30.0f);
}

// Coalesced multi-row logits: serial k=0..2303 summation matching ink_chain bit for bit.
template <uint SG, uint RPSG>
inline void ink_logits_coalesced(
    const device bfloat* h, const device bfloat* unembed, device float* out,
    constant InkLogitArgs& a, uint tid, uint sg, uint lane, uint2 tg,
    threadgroup uint (&sh_w)[32][33], threadgroup float2 (&sh_h)[SG * RPSG][32]) {
  constexpr uint TILE_R = SG * RPSG;
  const uint v0 = tg.x * 32u;
  if (v0 >= a.vocab) return;
  const uint r_start = tg.y * TILE_R;
  const uint r_end = min(r_start + TILE_R, a.rows);
  if (r_start >= r_end) return;
  const uint sub = tid >> 3u, col4 = (tid & 7u) << 2u, v = v0 + lane;
  constexpr uint ROW_STEP = (SG * 4u) < 32u ? (SG * 4u) : 32u;
  float acc[RPSG];
  for (uint rr = 0u; rr < RPSG; ++rr) acc[rr] = 0.0f;
  const uint row_base = r_start + sg * RPSG;
  const bool sg_active = row_base < r_end;
  for (uint k_step = 0u; k_step < 36u; ++k_step) {
    const uint d_bf = k_step * 64u;
    for (uint b = 0u; b < 32u; b += ROW_STEP) {
      const uint vr = b + sub;
      if (vr < 32u) {
        const uint4 wv = (v0 + vr < a.vocab)
            ? *(const device uint4*)(unembed + size_t(v0 + vr) * 2304u + d_bf + (col4 << 1u))
            : uint4(0u);
        sh_w[vr][col4 + 0u] = wv.x;
        sh_w[vr][col4 + 1u] = wv.y;
        sh_w[vr][col4 + 2u] = wv.z;
        sh_w[vr][col4 + 3u] = wv.w;
      }
    }
    for (uint rr = 0u; rr < RPSG; ++rr) {
      const uint r = sg * RPSG + rr;
      const uint hx = (r_start + r < r_end)
          ? *(const device uint*)(h + size_t(r_start + r) * 2304u + d_bf + (lane << 1u))
          : 0u;
      sh_h[r][lane] = float2(as_type<float>(hx << 16u), as_type<float>(hx & 0xffff0000u));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_active) {
      for (uint j = 0u; j < 32u; ++j) {
        const uint w = sh_w[lane][j];
        const float w0 = as_type<float>(w << 16u);
        const float w1 = as_type<float>(w & 0xffff0000u);
        for (uint rr = 0u; rr < RPSG; ++rr) {
          const float2 hv = sh_h[sg * RPSG + rr][j];
          acc[rr] = (acc[rr] + w0 * hv.x) + w1 * hv.y;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (v < a.vocab && sg_active) {
    for (uint rr = 0u; rr < RPSG; ++rr) {
      const uint row = row_base + rr;
      if (row < r_end) out[size_t(row) * a.vocab + v] = ink_cap(float(bfloat(acc[rr])), 30.0f);
    }
  }
}
[[max_total_threads_per_threadgroup(128)]]
kernel void tf_inkling2_coalesced_logits_r4(
    const device bfloat* h [[buffer(0)]], const device bfloat* unembed [[buffer(1)]],
    device float* out [[buffer(2)]], constant InkLogitArgs& a [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup uint sh_w[32][33];
  threadgroup float2 sh_h[4][32];
  ink_logits_coalesced<4, 1>(h, unembed, out, a, tid, sg, lane, tg, sh_w, sh_h);
}
[[max_total_threads_per_threadgroup(128)]]
kernel void tf_inkling2_coalesced_logits_r8(
    const device bfloat* h [[buffer(0)]], const device bfloat* unembed [[buffer(1)]],
    device float* out [[buffer(2)]], constant InkLogitArgs& a [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup uint sh_w[32][33];
  threadgroup float2 sh_h[8][32];
  ink_logits_coalesced<4, 2>(h, unembed, out, a, tid, sg, lane, tg, sh_w, sh_h);
}
[[max_total_threads_per_threadgroup(128)]]
kernel void tf_inkling2_coalesced_logits_r16(
    const device bfloat* h [[buffer(0)]], const device bfloat* unembed [[buffer(1)]],
    device float* out [[buffer(2)]], constant InkLogitArgs& a [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup uint sh_w[32][33];
  threadgroup float2 sh_h[16][32];
  ink_logits_coalesced<4, 4>(h, unembed, out, a, tid, sg, lane, tg, sh_w, sh_h);
}
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_coalesced_logits_r32(
    const device bfloat* h [[buffer(0)]], const device bfloat* unembed [[buffer(1)]],
    device float* out [[buffer(2)]], constant InkLogitArgs& a [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup uint sh_w[32][33];
  threadgroup float2 sh_h[32][32];
  ink_logits_coalesced<8, 4>(h, unembed, out, a, tid, sg, lane, tg, sh_w, sh_h);
}
struct InkExpertArgs { uint rows, channels; };
[[max_total_threads_per_threadgroup(32)]]
kernel void tf_inkling2_expert_pick(
    const device bfloat* experts [[buffer(0)]], const device uint* ids [[buffer(1)]],
    device bfloat* out [[buffer(2)]], constant InkExpertArgs& a [[buffer(3)]],
    uint3 at [[thread_position_in_grid]]) {
  const uint j = at.x, d = at.y, r = at.z;
  if (j >= 8 || d >= a.channels || r >= a.rows) return;
  out[(size_t(r) * 8 + j) * a.channels + d] = experts[(size_t(r) * 128 + ids[r * 8 + j]) * a.channels + d];
}

[[max_total_threads_per_threadgroup(32)]]
kernel void tf_inkling2_expert_combine(
    const device bfloat* picked [[buffer(0)]], const device float* weights [[buffer(1)]],
    device bfloat* out [[buffer(2)]], constant InkExpertArgs& a [[buffer(3)]],
    uint2 at [[thread_position_in_grid]]) {
  const uint d = at.x, r = at.y;
  if (d >= a.channels || r >= a.rows) return;
  float acc = 0.0f;
  for (uint j = 0; j < 8; ++j)
    acc += weights[r * 8 + j] * float(picked[(size_t(r) * 8 + j) * a.channels + d]);
  out[size_t(r) * a.channels + d] = bfloat(acc);
}
