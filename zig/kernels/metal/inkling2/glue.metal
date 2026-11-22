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
template <typename P>
inline float ink_value(const device bfloat* blocks, const device P* partial,
                       uint source, uint sources, uint d) {
  return source + 1 == sources ? float(partial[d]) : float(blocks[size_t(source) * 2304 + d]);
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
// Threadgroup (q, row): row's sources mixed by query q into out[q * rows + row]; a row has four block slots.
template <typename P>
inline void ink2_ar_t(
    const device bfloat* all_blocks, const device P* all_partial,
    const device float* queries, device float* all_out,
    constant InkARArgs& a, uint t, uint2 tg, uint lane,
    threadgroup float* red, threadgroup float* means, threadgroup float* weights) {
  const uint q = tg.x, row = tg.y;
  if (q >= a.queries || row >= a.rows || a.sources == 0 || a.sources > 5) return;
  const device bfloat* blocks = all_blocks + size_t(row) * 4 * 2304;
  const device P* partial = all_partial + size_t(row) * 2304;
  device float* out = all_out + (size_t(q) * a.rows + row) * 2304;
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
    out[d] = acc;
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_ar(
    const device bfloat* all_blocks [[buffer(0)]], const device float* all_partial [[buffer(1)]],
    const device float* queries [[buffer(2)]], device float* all_out [[buffer(3)]],
    constant InkARArgs& a [[buffer(4)]], uint t [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[2 * 5 * 8], means[5], weights[5];
  ink2_ar_t<float>(all_blocks, all_partial, queries, all_out, a, t, tg, lane, red, means, weights);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_ar_bf16(
    const device bfloat* all_blocks [[buffer(0)]], const device bfloat* all_partial [[buffer(1)]],
    const device float* queries [[buffer(2)]], device float* all_out [[buffer(3)]],
    constant InkARArgs& a [[buffer(4)]], uint t [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[2 * 5 * 8], means[5], weights[5];
  ink2_ar_t<bfloat>(all_blocks, all_partial, queries, all_out, a, t, tg, lane, red, means, weights);
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

struct InkConvDynSeg { ulong taps, input, out; uint row0, rows; };
struct InkConvDynSegsArgs { uint n_segs, total_rows, channels; };

inline void ink2_conv_dynamic_t(
    const device bfloat* taps, const device bfloat* input,
    const device bfloat* static_w, const device bfloat* projection,
    device bfloat* out, uint r, uint channels, uint t) {
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
  const float proj_t = float(bfloat(dot));
  for (uint b128 = 0u, g = 0u; b128 < channels; b128 += 128u, g += 4u) {
    const float pv[4] = {
        simd_shuffle(proj_t, ushort(g)),
        simd_shuffle(proj_t, ushort(g + 1u)),
        simd_shuffle(proj_t, ushort(g + 2u)),
        simd_shuffle(proj_t, ushort(g + 3u)),
    };
    const uint end = min(b128 + 128u, channels);
    for (uint c = b128 + t; c < end; c += 32u) {
      float sum = float(taps[size_t(r + 3) * channels + c]);
      for (uint lag = 0u; lag < 4u; ++lag) {
        const float w = float(static_w[lag * channels + c]) + pv[lag];
        sum += w * float(taps[size_t(r + 3 - lag) * channels + c]);
      }
      out[size_t(r) * channels + c] = bfloat(precise::divide(sum, 3.0f));
    }
  }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void tf_inkling2_conv_dynamic(
    const device bfloat* taps [[buffer(0)]], const device bfloat* input [[buffer(1)]],
    const device bfloat* static_w [[buffer(2)]], const device bfloat* projection [[buffer(3)]],
    device bfloat* out [[buffer(4)]], constant InkConvArgs& a [[buffer(5)]],
    uint t [[thread_position_in_threadgroup]], uint r [[threadgroup_position_in_grid]]) {
  if (r >= a.rows) return;
  ink2_conv_dynamic_t(taps, input, static_w, projection, out, r, a.channels, t);
}

[[max_total_threads_per_threadgroup(32)]]
kernel void tf_inkling2_conv_dynamic_segs(
    const device InkConvDynSeg* segs [[buffer(0)]], const device bfloat* static_w [[buffer(1)]],
    const device bfloat* projection [[buffer(2)]], constant InkConvDynSegsArgs& a [[buffer(3)]],
    uint t [[thread_position_in_threadgroup]], uint global_r [[threadgroup_position_in_grid]]) {
  if (global_r >= a.total_rows) return;
  uint si = 0;
  while (si + 1u < a.n_segs && global_r >= segs[si].row0 + segs[si].rows) ++si;
  const device InkConvDynSeg& seg = segs[si];
  const uint r = global_r - seg.row0;
  ink2_conv_dynamic_t(
      (const device bfloat*)seg.taps, (const device bfloat*)seg.input,
      static_w, projection, (device bfloat*)seg.out, r, a.channels, t);
}

struct InkRouteArgs { uint rows; };
inline uint ink_order_key(float v) {
  if (v == 0.0f) v = 0.0f;
  const uint b = as_type<uint>(v);
  return (b & 0x80000000u) != 0u ? ~b : (b | 0x80000000u);
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
  float sc[4], key[4];
  bool lane_nan = false;
  for (uint k = 0u; k < 4u; ++k) {
    const uint e = lane + 32u * k;
    sc[k] = scores[e];
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
  } else if (lane == 0u) {
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

struct InkLogitArgs { uint rows, vocab, flags, part_stride; };
inline bool ink_argmax_better(float a, uint ia, float b, uint ib) {
  if (ia == 0xffffffffu) return false;
  if (ib == 0xffffffffu) return true;
  const bool na = isnan(a), nb = isnan(b);
  if (na || nb) {
    if (na && ia == 0u) return !(nb && ib == 0u);
    if (nb && ib == 0u) return false;
    if (na != nb) return !na;
    return ia < ib;
  }
  return a > b || (a == b && ia < ib);
}
// Threadgroup (x, y) handles 8 vocab rows x up to 8 token rows; a simdgroup per vocab row shares unembed reads.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_logits(
    const device bfloat* h [[buffer(0)]], const device bfloat* unembed [[buffer(1)]],
    device float* out [[buffer(2)]], device float* part_vals [[buffer(3)]],
    device uint* part_ids [[buffer(4)]], constant InkLogitArgs& a [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tg_vals[64];
  const uint v = tg.x * 8u + sg;
  const uint r0 = tg.y * 8u;
  if (r0 >= a.rows) return;
  const uint rn = min(8u, a.rows - r0);
  float y_row[8];
  if (v < a.vocab) {
    const device bfloat* w = unembed + size_t(v) * 2304u;
    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint i = 8u * lane; i < 2304u; i += 256u) {
      const float4 w0 = float4(*(const device bfloat4*)(w + i)), w1 = float4(*(const device bfloat4*)(w + i + 4u));
      for (uint r = 0u; r < rn; ++r) {
        const device bfloat* hr = h + size_t(r0 + r) * 2304u + i;
        const float4 x0 = float4(*(const device bfloat4*)hr), x1 = float4(*(const device bfloat4*)(hr + 4u));
        acc[r] += dot(w0, x0) + dot(w1, x1);
      }
    }
    for (uint r = 0u; r < rn; ++r) {
      const float dot = simd_sum(acc[r]);
      // A9(a): unembedding rounds to bf16 before it widens for softcap30.
      if (lane == 0u) {
        const float y = ink_cap(float(bfloat(dot)), 30.0f);
        y_row[r] = y;
        if ((a.flags & 1u) != 0u) out[size_t(r0 + r) * a.vocab + v] = y;
      }
    }
  }
  if ((a.flags & 2u) != 0u) {
    if (v < a.vocab && lane == 0u) {
      for (uint r = 0u; r < rn; ++r) tg_vals[r * 8u + sg] = y_row[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < rn) {
      const uint id0 = tg.x * 8u + lane;
      uint bid = (lane < 8u && id0 < a.vocab) ? id0 : 0xffffffffu;
      float bv = (bid != 0xffffffffu) ? tg_vals[sg * 8u + lane] : 0.0f;
      for (ushort d = 4u; d > 0u; d >>= 1u) {
        const float ov = simd_shuffle_xor(bv, d);
        const uint oid = simd_shuffle_xor(bid, d);
        if (ink_argmax_better(ov, oid, bv, bid)) {
          bv = ov;
          bid = oid;
        }
      }
      if (lane == 0u) {
        const size_t idx = size_t(r0 + sg) * a.part_stride + tg.x;
        part_vals[idx] = bv;
        part_ids[idx] = bid;
      }
    }
  }
}

struct InkArgmaxArgs { uint rows, parts; };
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_reduce_argmax(
    const device float* part_vals [[buffer(0)]], const device uint* part_ids [[buffer(1)]],
    device uint* out_ids [[buffer(2)]], constant InkArgmaxArgs& a [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sg_vals[8];
  threadgroup uint sg_ids[8];
  if (row >= a.rows) return;
  const size_t base = size_t(row) * a.parts;
  float bv = 0.0f;
  uint bid = 0xffffffffu;
  for (uint i = t; i < a.parts; i += 256u) {
    const float v = part_vals[base + i];
    const uint id = part_ids[base + i];
    if (ink_argmax_better(v, id, bv, bid)) {
      bv = v;
      bid = id;
    }
  }
  for (ushort d = 16u; d > 0u; d >>= 1u) {
    const float ov = simd_shuffle_xor(bv, d);
    const uint oid = simd_shuffle_xor(bid, d);
    if (ink_argmax_better(ov, oid, bv, bid)) {
      bv = ov;
      bid = oid;
    }
  }
  if (lane == 0u) {
    sg_vals[sg] = bv;
    sg_ids[sg] = bid;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float fv = (lane < 8u) ? sg_vals[lane] : 0.0f;
    uint fid = (lane < 8u) ? sg_ids[lane] : 0xffffffffu;
    for (ushort d = 4u; d > 0u; d >>= 1u) {
      const float ov = simd_shuffle_xor(fv, d);
      const uint oid = simd_shuffle_xor(fid, d);
      if (ink_argmax_better(ov, oid, fv, fid)) {
        fv = ov;
        fid = oid;
      }
    }
    if (lane == 0u) out_ids[row] = fid;
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
