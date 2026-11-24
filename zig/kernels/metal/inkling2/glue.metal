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

struct InkARArgs { uint sources, queries, rows, blocks_stride, partial_stride, out_stride; };
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
  threadgroup float* dots = red + 40;
  for (uint s = 0; s < a.sources; ++s) {
    float dot = 0.0f;
    for (uint d = t; d < 2304; d += 256) {
      const float z = ink_value(blocks, partial, s, a.sources, d) * means[s];
      dot += queries[size_t(q) * 2304 + d] * z;
    }
    const float reduced = simd_sum(dot);
    if (lane == 0) dots[s * 8 + t / 32] = reduced;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float peak = -INFINITY;
    for (uint s = 0; s < a.sources; ++s) {
      float dot = 0.0f;
      for (uint j = 0; j < 8; ++j) dot += dots[s * 8 + j];
      weights[s] = dot * INKLING_AR_SCALE;
      peak = max(peak, weights[s]);
    }
    float den = 0.0f;
    for (uint s = 0; s < a.sources; ++s) { weights[s] = precise::exp(weights[s] - peak); den += weights[s]; }
    for (uint s = 0; s < a.sources; ++s) weights[s] = precise::divide(weights[s], den);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = t; d < 2304; d += 256) {
    float acc = 0.0f;
    for (uint s = 0; s < a.sources; ++s)
      acc += weights[s] * (ink_value(blocks, partial, s, a.sources, d) * means[s]);
    out[size_t(q) * 2304 + d] = acc;
  }
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

kernel void tf_inkling2_conv_dynamic(
    const device bfloat* taps [[buffer(0)]], const device bfloat* input [[buffer(1)]],
    const device bfloat* static_w [[buffer(2)]], const device bfloat* projection [[buffer(3)]],
    device bfloat* out [[buffer(4)]], constant InkConvArgs& a [[buffer(5)]],
    uint t [[thread_position_in_threadgroup]], uint r [[threadgroup_position_in_grid]]) {
  if (r >= a.rows) return;
  threadgroup float projected[32];
  if (t < 32) {
    float dot = 0.0f;
    for (uint d = 0; d < 2304; ++d)
      dot += float(projection[size_t(t) * 2304 + d]) * float(input[size_t(r) * 2304 + d]);
    projected[t] = float(bfloat(dot));
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

struct InkRouteArgs { uint rows; };
kernel void tf_inkling2_router(
    const device bfloat* x [[buffer(0)]], const device bfloat* weight [[buffer(1)]],
    const device float* bias [[buffer(2)]], device uint* ids [[buffer(3)]],
    device float* weights [[buffer(4)]], constant InkRouteArgs& a [[buffer(5)]],
    uint t [[thread_position_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
  if (row >= a.rows) return;
  threadgroup float scores[128], chosen[8];
  threadgroup uint picked[8];
  if (t < 128) {
    float dot = 0.0f;
    for (uint d = 0; d < 2304; ++d)
      dot += float(weight[size_t(t) * 2304 + d]) * float(x[size_t(row) * 2304 + d]);
    scores[t] = ink_sigmoid(dot);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    bool used[128] = {};
    for (uint j = 0; j < 8; ++j) {
      uint best = 128;
      for (uint e = 0; e < 128; ++e) if (!used[e]) {
        if (best == 128 || scores[e] + bias[e] > scores[best] + bias[best]) best = e;
      }
      used[best] = true; picked[j] = best; chosen[j] = scores[best];
    }
    // Combining, including the denominator, is in ascending expert index.
    for (uint j = 1; j < 8; ++j) {
      uint id = picked[j]; float score = chosen[j]; uint k = j;
      while (k > 0 && picked[k - 1] > id) { picked[k] = picked[k - 1]; chosen[k] = chosen[k - 1]; --k; }
      picked[k] = id; chosen[k] = score;
    }
    float den = 1e-20f;
    for (uint j = 0; j < 8; ++j) den += chosen[j];
    for (uint j = 0; j < 8; ++j) {
      ids[row * 8 + j] = picked[j];
      weights[row * 8 + j] = precise::divide(chosen[j], den) * 8.0f;
    }
  }
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
  const uint v = at.x, r = at.y;
  if (v >= a.vocab || r >= a.rows) return;
  float dot = 0.0f;
  for (uint d = 0; d < 2304; ++d)
    dot += float(unembed[size_t(v) * 2304 + d]) * float(h[size_t(r) * 2304 + d]);
  // A9(a): unembedding rounds to bf16 before it widens for softcap30.
  out[size_t(r) * a.vocab + v] = ink_cap(float(bfloat(dot)), 30.0f);
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
