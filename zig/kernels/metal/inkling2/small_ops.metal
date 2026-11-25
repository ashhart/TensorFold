#include <metal_stdlib>
#pragma METAL fp contract(off)
using namespace metal;

// Inkling 2 Small (H=4608, E=384, 8 blocks/taps) Metal glue and runtime ops; FP32 serial reductions.
constant float INK2_SMALL_EPS = 1e-6f;
constant float INK2_SMALL_AR_SCALE = 0x1.e2b7dep-7f; // binary32(1 / sqrt(4608))
constant uint INK2_SMALL_H = 4608u;
constant uint INK2_SMALL_EXPERTS = 384u;
constant uint INK2_SMALL_BLOCKS = 8u;
constant uint INK2_SMALL_TAPS = 8u;

inline float ink2s_sigmoid(float x) {
  const float e = precise::exp(-fabs(x));
  return x < 0.0f ? precise::divide(e, 1.0f + e) : precise::divide(1.0f, 1.0f + e);
}

inline float ink2s_cap(float x, float c) {
  return c * precise::tanh(precise::divide(x, c));
}

inline float ink2s_value(const device bfloat* blocks, const device float* partial,
                         uint source, uint sources, uint d) {
  return source + 1u == sources ? partial[d] : float(blocks[size_t(source) * INK2_SMALL_H + d]);
}

// Two BF16 pairs of w and x multiplied and added to sum in column order.
inline float ink2s_pair(float sum, uint w, uint x) {
  sum += as_type<float>(w << 16u) * as_type<float>(x << 16u);
  return sum + as_type<float>(w & 0xffff0000u) * as_type<float>(x & 0xffff0000u);
}

// Serial FP32 dot over k columns (multiple of 32, 16-byte aligned) using batched 16-byte loads.
inline float ink2s_chain(const device bfloat* w, const device bfloat* x, uint k) {
  float sum = 0.0f;
  for (uint d = 0u; d < k; d += 32u) {
    uint4 a[4], b[4];
    for (uint i = 0u; i < 4u; ++i) {
      a[i] = *(const device uint4*)(w + d + 8u * i);
      b[i] = *(const device uint4*)(x + d + 8u * i);
    }
    for (uint i = 0u; i < 4u; ++i) {
      sum = ink2s_pair(sum, a[i].x, b[i].x);
      sum = ink2s_pair(sum, a[i].y, b[i].y);
      sum = ink2s_pair(sum, a[i].z, b[i].z);
      sum = ink2s_pair(sum, a[i].w, b[i].w);
    }
  }
  return sum;
}

struct InkSmallEmbedArgs { uint rows; };
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_small_embed_norm(
    const device uint* ids [[buffer(0)]], const device bfloat* table [[buffer(1)]],
    const device bfloat* weight [[buffer(2)]], device bfloat* out [[buffer(3)]],
    constant InkSmallEmbedArgs& a [[buffer(4)]], uint t [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows) return;
  const device bfloat* x = table + size_t(ids[row]) * INK2_SMALL_H;
  float part = 0.0f;
  for (uint d = t; d < INK2_SMALL_H; d += 256u) { const float v = float(x[d]); part += v * v; }
  threadgroup float red[8], inverse;
  const float reduced = simd_sum(part);
  if (lane == 0u) red[t / 32u] = reduced;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0u) {
    float total = 0.0f;
    for (uint j = 0u; j < 8u; ++j) total += red[j];
    inverse = precise::rsqrt(precise::divide(total, float(INK2_SMALL_H)) + INK2_SMALL_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = t; d < INK2_SMALL_H; d += 256u)
    out[size_t(row) * INK2_SMALL_H + d] = bfloat(float(weight[d]) * (float(x[d]) * inverse));
}

struct InkSmallARArgs { uint sources, queries, rows, blocks_stride, partial_stride, out_stride; };
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_small_ar(
    const device bfloat* blocks_base [[buffer(0)]], const device float* partial_base [[buffer(1)]],
    const device float* queries [[buffer(2)]], device float* out_base [[buffer(3)]],
    constant InkSmallARArgs& a [[buffer(4)]], uint t [[thread_index_in_threadgroup]],
    uint2 at [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint q = at.x, row = at.y;
  if (q >= a.queries || row >= a.rows || a.sources == 0u || a.sources > 9u) return;
  const device bfloat* blocks = blocks_base + size_t(row) * a.blocks_stride;
  const device float* partial = partial_base + size_t(row) * a.partial_stride;
  device float* out = out_base + size_t(row) * a.out_stride;
  threadgroup float red[2 * 9 * 8], means[9], weights[9];
  for (uint s = 0u; s < a.sources; ++s) {
    float ss = 0.0f;
    for (uint d = t; d < INK2_SMALL_H; d += 256u) {
      const float x = ink2s_value(blocks, partial, s, a.sources, d);
      ss += x * x;
    }
    const float reduced = simd_sum(ss);
    if (lane == 0u) red[s * 8u + t / 32u] = reduced;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0u) for (uint s = 0u; s < a.sources; ++s) {
    float ss = 0.0f;
    for (uint j = 0u; j < 8u; ++j) ss += red[s * 8u + j];
    means[s] = precise::rsqrt(precise::divide(ss, float(INK2_SMALL_H)) + INK2_SMALL_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float* dots = red + 72;
  for (uint s = 0u; s < a.sources; ++s) {
    float dot = 0.0f;
    for (uint d = t; d < INK2_SMALL_H; d += 256u) {
      const float z = ink2s_value(blocks, partial, s, a.sources, d) * means[s];
      dot += queries[size_t(q) * INK2_SMALL_H + d] * z;
    }
    const float reduced = simd_sum(dot);
    if (lane == 0u) dots[s * 8u + t / 32u] = reduced;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0u) {
    float peak = -INFINITY;
    for (uint s = 0u; s < a.sources; ++s) {
      float dot = 0.0f;
      for (uint j = 0u; j < 8u; ++j) dot += dots[s * 8u + j];
      weights[s] = dot * INK2_SMALL_AR_SCALE;
      peak = max(peak, weights[s]);
    }
    float den = 0.0f;
    for (uint s = 0u; s < a.sources; ++s) { weights[s] = precise::exp(weights[s] - peak); den += weights[s]; }
    for (uint s = 0u; s < a.sources; ++s) weights[s] = precise::divide(weights[s], den);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = t; d < INK2_SMALL_H; d += 256u) {
    float acc = 0.0f;
    for (uint s = 0u; s < a.sources; ++s)
      acc += weights[s] * (ink2s_value(blocks, partial, s, a.sources, d) * means[s]);
    out[size_t(q) * INK2_SMALL_H + d] = acc;
  }
}

struct InkSmallConvArgs { uint rows, channels; };
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_small_conv_dynamic(
    const device bfloat* taps [[buffer(0)]], const device bfloat* input [[buffer(1)]],
    const device bfloat* static_w [[buffer(2)]], const device bfloat* projection [[buffer(3)]],
    device bfloat* out [[buffer(4)]], constant InkSmallConvArgs& a [[buffer(5)]],
    uint t [[thread_position_in_threadgroup]], uint r [[threadgroup_position_in_grid]]) {
  if (r >= a.rows) return;
  threadgroup float projected[32];
  if (t < 32u) {
    projected[t] = float(bfloat(ink2s_chain(projection + size_t(t) * INK2_SMALL_H,
                                            input + size_t(r) * INK2_SMALL_H, INK2_SMALL_H)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = t; c < a.channels; c += 256u) {
    float sum = float(taps[size_t(r + 3u) * a.channels + c]);
    for (uint lag = 0u; lag < 4u; ++lag) {
      const float w = float(static_w[lag * a.channels + c]) + projected[4u * (c / 128u) + lag];
      sum += w * float(taps[size_t(r + 3u - lag) * a.channels + c]);
    }
    out[size_t(r) * a.channels + c] = bfloat(precise::divide(sum, 3.0f));
  }
}

struct InkSmallRouteArgs { uint rows; };
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_small_router(
    const device bfloat* x [[buffer(0)]], const device bfloat* weight [[buffer(1)]],
    const device float* bias [[buffer(2)]], device uint* ids [[buffer(3)]],
    device float* weights [[buffer(4)]], constant InkSmallRouteArgs& a [[buffer(5)]],
    uint t [[thread_position_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
  if (row >= a.rows) return;
  threadgroup float scores[INK2_SMALL_EXPERTS], key[INK2_SMALL_EXPERTS], chosen[8];
  threadgroup uint picked[8], selected[INK2_SMALL_EXPERTS], any_nan;
  for (uint e = t; e < INK2_SMALL_EXPERTS; e += 256u) {
    scores[e] = ink2s_sigmoid(ink2s_chain(weight + size_t(e) * INK2_SMALL_H,
                                          x + size_t(row) * INK2_SMALL_H, INK2_SMALL_H));
    key[e] = scores[e] + bias[e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Ranks 0 to 7 by (key descending, index ascending) match the serial scan; any NaN key runs the scan.
  bool saw_nan = false;
  for (uint e = t; e < INK2_SMALL_EXPERTS; e += 256u) {
    uint rank = 0u;
    for (uint f = 0u; f < INK2_SMALL_EXPERTS; ++f) {
      saw_nan = saw_nan || isnan(key[f]);
      rank += (key[f] > key[e] || (key[f] == key[e] && f < e)) ? 1u : 0u;
    }
    selected[e] = rank < 8u ? 1u : 0u;
  }
  if (t == 0u) any_nan = saw_nan ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (any_nan != 0u) {
    if (t == 0u) {
      bool used[INK2_SMALL_EXPERTS] = {};
      for (uint j = 0u; j < 8u; ++j) {
        uint best = INK2_SMALL_EXPERTS;
        for (uint e = 0u; e < INK2_SMALL_EXPERTS; ++e) if (!used[e]) {
          if (best == INK2_SMALL_EXPERTS || key[e] > key[best]) best = e;
        }
        used[best] = true; picked[j] = best; chosen[j] = scores[best];
      }
      for (uint j = 1u; j < 8u; ++j) {
        uint id = picked[j]; float score = chosen[j]; uint k = j;
        while (k > 0u && picked[k - 1u] > id) { picked[k] = picked[k - 1u]; chosen[k] = chosen[k - 1u]; --k; }
        picked[k] = id; chosen[k] = score;
      }
    }
  } else {
    for (uint e = t; e < INK2_SMALL_EXPERTS; e += 256u) if (selected[e] != 0u) {
      uint at = 0u;
      for (uint f = 0u; f < e; ++f) at += selected[f];
      picked[at] = e; chosen[at] = scores[e];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 8u) {
    float den = 1e-20f;
    for (uint j = 0u; j < 8u; ++j) den += chosen[j];
    ids[row * 8u + t] = picked[t];
    weights[row * 8u + t] = precise::divide(chosen[t], den) * 8.0f;
  }
}

struct InkSmallLogitArgs { uint rows, vocab; };
[[max_total_threads_per_threadgroup(32)]]
kernel void tf_ink2_small_logits(
    const device bfloat* h [[buffer(0)]], const device bfloat* unembed [[buffer(1)]],
    device float* out [[buffer(2)]], constant InkSmallLogitArgs& a [[buffer(3)]],
    uint2 at [[thread_position_in_grid]]) {
  const uint v = at.x, r = at.y;
  if (v >= a.vocab || r >= a.rows) return;
  const float dot = ink2s_chain(unembed + size_t(v) * INK2_SMALL_H,
                                h + size_t(r) * INK2_SMALL_H, INK2_SMALL_H);
  out[size_t(r) * a.vocab + v] = ink2s_cap(float(bfloat(dot)), 30.0f);
}

struct InkSmallRuntimeArgs { uint count, boundary; };
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_small_attention_partial(
    const device bfloat* p [[buffer(0)]], const device bfloat* x [[buffer(1)]],
    device bfloat* blocks [[buffer(2)]], device float* t [[buffer(3)]],
    constant InkSmallRuntimeArgs& a [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const uint row = i / INK2_SMALL_H, d = i % INK2_SMALL_H;
  t[i] = a.boundary ? float(x[i]) : float(p[i]) + float(x[i]);
  if (a.boundary) blocks[size_t(row) * INK2_SMALL_BLOCKS * INK2_SMALL_H + (a.boundary - 1u) * INK2_SMALL_H + d] = p[i];
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_small_capture_tap(
    const device bfloat* x [[buffer(0)]], device bfloat* y [[buffer(1)]],
    constant InkSmallRuntimeArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const uint row = i / INK2_SMALL_H, d = i % INK2_SMALL_H;
  y[size_t(row) * INK2_SMALL_TAPS * INK2_SMALL_H + a.boundary * INK2_SMALL_H + d] = x[i];
}
