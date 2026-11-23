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

struct InkARArgs { uint sources, queries, rows; };
// Four consecutive values of a source: a completed block's BF16, or the running partial's FP32 for the last source.
inline float4 ink_values(const device bfloat* blocks, const device float* partial, uint source, uint sources, uint d) {
  if (source + 1 == sources) return *(const device float4*)(partial + d);
  return float4(*(const device bfloat4*)(blocks + size_t(source) * 2304 + d));
}

// Threadgroup (q, row): row's sources mixed by query q into out[q * rows + row], four values a thread.
[[max_total_threads_per_threadgroup(576)]]
kernel void tf_inkling2_ar(
    const device bfloat* all_blocks [[buffer(0)]], const device float* all_partial [[buffer(1)]],
    const device float* queries [[buffer(2)]], device float* all_out [[buffer(3)]],
    constant InkARArgs& a [[buffer(4)]], uint t [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint q = tg.x, row = tg.y, d = 4u * t;
  if (q >= a.queries || row >= a.rows || a.sources == 0 || a.sources > 5) return;
  const device bfloat* blocks = all_blocks + size_t(row) * 4 * 2304;
  const device float* partial = all_partial + size_t(row) * 2304;
  device float* out = all_out + (size_t(q) * a.rows + row) * 2304;
  // Each of the 18 simdgroups folds the partial sums itself, so no thread waits on another's result.
  threadgroup float red[5 * 18], dots[5 * 18];
  float4 x[5];
  for (uint s = 0; s < 5; ++s) {
    x[s] = s < a.sources ? ink_values(blocks, partial, s, a.sources, d) : float4(0.0f);
    const float4 sq = x[s] * x[s];
    const float reduced = simd_sum((sq.x + sq.y) + (sq.z + sq.w));
    if (lane == 0) red[s * 18 + sg] = reduced;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float4 query = *(const device float4*)(queries + size_t(q) * 2304 + d);
  float means[5];
  for (uint s = 0; s < 5; ++s) {
    const float ss = simd_sum(lane < 18u ? red[s * 18 + lane] : 0.0f);
    means[s] = precise::rsqrt(precise::divide(ss, 2304.0f) + INKLING_EPS);
    const float4 z = query * (x[s] * means[s]);
    const float reduced = simd_sum((z.x + z.y) + (z.z + z.w));
    if (lane == 0) dots[s * 18 + sg] = reduced;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float weights[5], peak = -INFINITY;
  for (uint s = 0; s < 5; ++s) {
    const float dot = simd_sum(lane < 18u ? dots[s * 18 + lane] : 0.0f);
    weights[s] = dot * INKLING_AR_SCALE;
    if (s < a.sources) peak = max(peak, weights[s]);
  }
  float den = 0.0f;
  for (uint s = 0; s < a.sources; ++s) { weights[s] = precise::exp(weights[s] - peak); den += weights[s]; }
  float4 acc = float4(0.0f);
  for (uint s = 0; s < a.sources; ++s) acc += precise::divide(weights[s], den) * (x[s] * means[s]);
  *(device float4*)(out + d) = acc;
}

struct InkConvArgs { uint rows, channels; };
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
  if (t < 32u) {
    const device bfloat* proj_row = projection + size_t(t) * 2304u;
    const device bfloat* in_row = input + size_t(r) * 2304u;
    float dot = 0.0f;
    for (uint d = 0u; d < 2304u; d += 4u) {
      const float4 p = float4(*(const device bfloat4*)(proj_row + d));
      const float4 x = float4(*(const device bfloat4*)(in_row + d));
      dot += p.x * x.x;
      dot += p.y * x.y;
      dot += p.z * x.z;
      dot += p.w * x.w;
    }
    projected[t] = float(bfloat(dot));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = t; c < a.channels; c += 256u) {
    float sum = float(taps[size_t(r + 3) * a.channels + c]);
    for (uint lag = 0u; lag < 4u; ++lag) {
      const float w = float(static_w[lag * a.channels + c]) + projected[4u * (c / 128u) + lag];
      sum += w * float(taps[size_t(r + 3 - lag) * a.channels + c]);
    }
    out[size_t(r) * a.channels + c] = bfloat(precise::divide(sum, 3.0f));
  }
}

struct InkRouteArgs { uint rows; };
inline uint ink_order_key(float v) {
  if (v == 0.0f) v = 0.0f;
  const uint b = as_type<uint>(v);
  return (b & 0x80000000u) != 0u ? ~b : (b | 0x80000000u);
}

inline void ink_select_top8(
    const thread float* sc, const device float* bias, device uint* ids, device float* weights,
    threadgroup float* scores, uint row, uint lane) {
  float key[4];
  bool lane_nan = false;
  for (uint k = 0u; k < 4u; ++k) {
    const uint e = lane + 32u * k;
    key[k] = sc[k] + bias[e];
    lane_nan = lane_nan || isnan(key[k]);
  }
  uint picked[8];
  float chosen[8];
  if (!simd_any(lane_nan)) {
    uint okey[4];
    for (uint k = 0u; k < 4u; ++k) okey[k] = ink_order_key(key[k]);
    uint taken = 0u;
    for (uint r = 0u; r < 8u; ++r) {
      uint best = 0u, at = 0xFFFFFFFFu;
      for (uint k = 0u; k < 4u; ++k) {
        if (((taken >> k) & 1u) == 0u && okey[k] > best) {
          best = okey[k];
          at = lane + 32u * k;
        }
      }
      const uint top = simd_max(best);
      const uint winner = simd_min(best == top ? at : 0xFFFFFFFFu);
      float s = 0.0f;
      if ((winner & 31u) == lane) {
        const uint k = winner >> 5u;
        taken |= 1u << k;
        s = sc[k];
      }
      picked[r] = winner;
      chosen[r] = simd_shuffle(s, ushort(winner & 31u));
    }
  } else {
    for (uint k = 0u; k < 4u; ++k) scores[lane + 32u * k] = sc[k];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
      bool used[128] = {};
      for (uint j = 0u; j < 8u; ++j) {
        uint best = 128u;
        for (uint e = 0u; e < 128u; ++e) if (!used[e]) {
          const float ke = scores[e] + bias[e];
          const float kb = best == 128u ? 0.0f : scores[best] + bias[best];
          if (best == 128u || ke > kb) best = e;
        }
        used[best] = true;
        picked[j] = best;
        chosen[j] = scores[best];
      }
    }
  }
  if (lane == 0u) {
    for (uint j = 1u; j < 8u; ++j) {
      const uint id = picked[j];
      const float score = chosen[j];
      uint k = j;
      while (k > 0u && picked[k - 1u] > id) {
        picked[k] = picked[k - 1u];
        chosen[k] = chosen[k - 1u];
        --k;
      }
      picked[k] = id;
      chosen[k] = score;
    }
    float den = 1e-20f;
    for (uint j = 0u; j < 8u; ++j) den += chosen[j];
    for (uint j = 0u; j < 8u; ++j) {
      ids[row * 8u + j] = picked[j];
      weights[row * 8u + j] = precise::divide(chosen[j], den) * 8.0f;
    }
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_route_scores(
    const device bfloat* x [[buffer(0)]], const device bfloat* weight [[buffer(1)]],
    device float* dots [[buffer(2)]], constant InkRouteArgs& a [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint e = tg.x * 8u + sg, row = tg.y;
  if (e >= 128u || row >= a.rows) return;
  const device bfloat* w = weight + size_t(e) * 2304u;
  const device bfloat* v = x + size_t(row) * 2304u;
  float acc = 0.0f;
  for (uint i = 8u * lane; i < 2304u; i += 256u) {
    const float4 w0 = float4(*(const device bfloat4*)(w + i)), w1 = float4(*(const device bfloat4*)(w + i + 4u));
    const float4 v0 = float4(*(const device bfloat4*)(v + i)), v1 = float4(*(const device bfloat4*)(v + i + 4u));
    acc += dot(w0, v0) + dot(w1, v1);
  }
  const float sum = simd_sum(acc);
  if (lane == 0u) dots[row * 128u + e] = sum;
}

[[max_total_threads_per_threadgroup(32)]]
kernel void tf_inkling2_route_select(
    const device float* dots [[buffer(0)]], const device float* bias [[buffer(1)]],
    device uint* ids [[buffer(2)]], device float* weights [[buffer(3)]],
    constant InkRouteArgs& a [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]], uint row [[threadgroup_position_in_grid]]) {
  if (row >= a.rows) return;
  threadgroup float scores[128];
  float sc[4];
  for (uint k = 0u; k < 4u; ++k) sc[k] = ink_sigmoid(dots[row * 128u + lane + 32u * k]);
  ink_select_top8(sc, bias, ids, weights, scores, row, lane);
}

// Fused router: sigmoid(weight . x), top-8 of score + bias (lowest expert on tie), ascending IDs, 512 B TG.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_router(
    const device bfloat* x [[buffer(0)]], const device bfloat* weight [[buffer(1)]],
    const device float* bias [[buffer(2)]], device uint* ids [[buffer(3)]],
    device float* weights [[buffer(4)]], constant InkRouteArgs& a [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows) return;
  threadgroup float scores[128];
  const device bfloat* v = x + size_t(row) * 2304u;
  for (uint p = 0u; p < 16u; p += 4u) {
    const uint e0 = sg + 8u * p, e1 = e0 + 8u, e2 = e0 + 16u, e3 = e0 + 24u;
    const device bfloat* w0 = weight + size_t(e0) * 2304u;
    const device bfloat* w1 = weight + size_t(e1) * 2304u;
    const device bfloat* w2 = weight + size_t(e2) * 2304u;
    const device bfloat* w3 = weight + size_t(e3) * 2304u;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint i = 8u * lane; i < 2304u; i += 256u) {
      const float4 v0 = float4(*(const device bfloat4*)(v + i));
      const float4 v1 = float4(*(const device bfloat4*)(v + i + 4u));
      acc0 += dot(float4(*(const device bfloat4*)(w0 + i)), v0) + dot(float4(*(const device bfloat4*)(w0 + i + 4u)), v1);
      acc1 += dot(float4(*(const device bfloat4*)(w1 + i)), v0) + dot(float4(*(const device bfloat4*)(w1 + i + 4u)), v1);
      acc2 += dot(float4(*(const device bfloat4*)(w2 + i)), v0) + dot(float4(*(const device bfloat4*)(w2 + i + 4u)), v1);
      acc3 += dot(float4(*(const device bfloat4*)(w3 + i)), v0) + dot(float4(*(const device bfloat4*)(w3 + i + 4u)), v1);
    }
    const float s0 = simd_sum(acc0), s1 = simd_sum(acc1), s2 = simd_sum(acc2), s3 = simd_sum(acc3);
    if (lane == 0u) {
      scores[e0] = ink_sigmoid(s0);
      scores[e1] = ink_sigmoid(s1);
      scores[e2] = ink_sigmoid(s2);
      scores[e3] = ink_sigmoid(s3);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0u) return;
  float sc[4];
  for (uint k = 0u; k < 4u; ++k) sc[k] = scores[lane + 32u * k];
  ink_select_top8(sc, bias, ids, weights, scores, row, lane);
}

struct InkActArgs { uint rows, width; };
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
kernel void tf_inkling2_logits(
    const device bfloat* h [[buffer(0)]], const device bfloat* unembed [[buffer(1)]],
    device float* out [[buffer(2)]], constant InkLogitArgs& a [[buffer(3)]],
    uint2 at [[thread_position_in_grid]]) {
  // A thread takes vocabulary row v for up to eight rows of h, so a window reads the row once; a dot adds in d order.
  const uint v = at.x, r0 = 8u * at.y;
  if (v >= a.vocab || r0 >= a.rows) return;
  const uint n = min(8u, a.rows - r0);
  const device bfloat* w = unembed + size_t(v) * 2304;
  const device bfloat* x = h + size_t(r0) * 2304;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint d = 0; d < 2304; d += 4) {
    const float4 wv = float4(*(const device bfloat4*)(w + d));
    for (uint r = 0; r < n; ++r) {
      const float4 xv = float4(*(const device bfloat4*)(x + size_t(r) * 2304 + d));
      float sum = acc[r];
      sum += wv.x * xv.x; sum += wv.y * xv.y; sum += wv.z * xv.z; sum += wv.w * xv.w;
      acc[r] = sum;
    }
  }
  for (uint r = 0; r < n; ++r) {
    const float dot = acc[r];
    // A9(a): unembedding rounds to bf16 before it widens for softcap30.
    out[size_t(r0 + r) * a.vocab + v] = ink_cap(float(bfloat(dot)), 30.0f);
  }
}

struct InkExpertArgs { uint rows, channels; };
kernel void tf_inkling2_expert_pick(
    const device bfloat* experts [[buffer(0)]], const device uint* ids [[buffer(1)]],
    device bfloat* out [[buffer(2)]], constant InkExpertArgs& a [[buffer(3)]],
    uint3 at [[thread_position_in_grid]]) {
  const uint j = at.x, d = at.y, r = at.z;
  if (j >= 8 || d >= a.channels || r >= a.rows) return;
  out[(size_t(r) * 8 + j) * a.channels + d] = experts[(size_t(r) * 128 + ids[r * 8 + j]) * a.channels + d];
}

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
