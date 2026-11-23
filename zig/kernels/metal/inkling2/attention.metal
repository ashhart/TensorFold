//! Inkling 2 Mini's scalar attention preparation, chunk decode and ordered sink/gate merge.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

constant constexpr uint D = 128, REL = 16, EXTENT = 1024;
constant constexpr float LOG2E = 1.4426950408889634f;

struct InkSeg {
  int row0, n, p0, pad;
  long off;
};
struct InkPrepArgs {
  uint heads, kv_heads, ring, global;
  float qk_clip, rel_clip, floor_pos, alpha, eps;
};
struct InkAttnArgs {
  uint heads, kv_heads, ring, window, chunk, c0, chunks;
  float scale;
};
struct InkMergeArgs {
  uint heads, window, chunk, c0, chunks, subs;
};
struct InkSlotMeta {
  uint ring, chunk, pad0, pad1;
};
struct InkPrepSlotsArgs {
  uint heads, kv_heads, global, n_slots;
  uint rings[8];
  float qk_clip, rel_clip, floor_pos, alpha, eps;
};
struct InkAttnSlotsArgs {
  uint heads, kv_heads, window, c0, chunks, n_slots;
  float scale;
  uint pad;
  InkSlotMeta slots[8];
};

inline float ink_warp_sum(float v) {
  for (uint o = 16; o; o >>= 1) v += simd_shuffle_xor(v, ushort(o));
  return v;
}
inline float ink_clip_factor(float ss, float threshold) {
  return fmin(threshold / (precise::sqrt(ss) + 1e-6f), 1.0f);
}
inline float ink_tau(uint pos, uint global, float floor_pos, float alpha) {
  return global ? 1.0f + alpha * precise::log(max(float(pos + 1) / floor_pos, 1.0f)) : 1.0f;
}
inline void ink_prep_qrel(uint row, uint unit, uint lane, uint heads, uint global, float qk_clip, float rel_clip,
    float tau, const device bfloat* q_raw, const device bfloat* rel_raw, const device bfloat* q_scale,
    const device bfloat* rel_scale, device bfloat* q_out, device bfloat* rel_out) {
  const size_t rh = size_t(row) * heads + unit;
  float x[4];
  for (uint i = 0; i < 4; ++i) x[i] = float(q_raw[rh * D + lane * 4 + i]) * float(q_scale[unit * D + lane * 4 + i]);
  const float ss = ((x[0] * x[0] + x[1] * x[1]) + x[2] * x[2]) + x[3] * x[3];
  const float f = ink_clip_factor(ink_warp_sum(ss), qk_clip);
  for (uint i = 0; i < 4; ++i) {
    x[i] *= f;
    const bfloat once = bfloat(x[i]);
    q_out[rh * D + lane * 4 + i] = global ? bfloat(float(once) * tau) : once;
  }
  const float r = lane < REL ? float(rel_raw[rh * REL + lane]) * float(rel_scale[unit * REL + lane]) : 0.0f;
  const float rf = ink_clip_factor(ink_warp_sum(r * r), rel_clip);
  if (lane < REL) {
    const bfloat once = bfloat(r * rf);
    rel_out[rh * REL + lane] = global ? bfloat(float(once) * tau) : once;
  }
}
inline void ink_prep_kv(const device bfloat* src, device bfloat* dst, bool key, float qk_clip, float eps, uint lane) {
  float x[4];
  for (uint i = 0; i < 4; ++i) x[i] = float(src[lane * 4 + i]);
  const float ss = ((x[0] * x[0] + x[1] * x[1]) + x[2] * x[2]) + x[3] * x[3];
  const float scale = key ? ink_clip_factor(ink_warp_sum(ss), qk_clip)
                          : precise::rsqrt(ink_warp_sum(ss) / float(D) + eps);
  for (uint i = 0; i < 4; ++i) dst[lane * 4 + i] = bfloat(x[i] * scale);
}

// One 32-thread group handles a row and head; independent q/k/v and relative rows keep their first BF16 rounding.
[[max_total_threads_per_threadgroup(32)]]
kernel void tf_ink2_prep(const device bfloat* q_raw [[buffer(0)]], const device bfloat* k_raw [[buffer(1)]],
    const device bfloat* v_raw [[buffer(2)]], const device bfloat* rel_raw [[buffer(3)]],
    const device bfloat* q_scale [[buffer(4)]], const device bfloat* rel_scale [[buffer(5)]],
    const device int* rpos [[buffer(6)]], const device int* rseg [[buffer(7)]],
    const device InkSeg* segs [[buffer(8)]], device bfloat* q_out [[buffer(9)]],
    device bfloat* rel_out [[buffer(10)]], device bfloat* kc [[buffer(11)]], device bfloat* vc [[buffer(12)]],
    constant InkPrepArgs& a [[buffer(13)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, unit = tg.y, pos = uint(rpos[row]);
  const float tau = a.global ? 1.0f + a.alpha * precise::log(max(float(pos + 1) / a.floor_pos, 1.0f)) : 1.0f;
  if (unit < a.heads) {
    ink_prep_qrel(row, unit, lane, a.heads, a.global, a.qk_clip, a.rel_clip, tau, q_raw, rel_raw, q_scale, rel_scale,
                  q_out, rel_out);
    return;
  }
  const bool key = unit < a.heads + a.kv_heads;
  const uint h = key ? unit - a.heads : unit - a.heads - a.kv_heads;
  const device bfloat* src = (key ? k_raw : v_raw) + (size_t(row) * a.kv_heads + h) * D;
  const InkSeg s = segs[rseg[row]];
  device bfloat* dst = (key ? kc : vc) + s.off + (size_t(h) * a.ring + pos % a.ring) * D;
  ink_prep_kv(src, dst, key, a.qk_clip, a.eps, lane);
}

// Batches up to 8 streams in one launch; rseg[row] selects the slot's ring and KV cache.
[[max_total_threads_per_threadgroup(32)]]
kernel void tf_ink2_prep_slots(const device bfloat* q_raw [[buffer(0)]], const device bfloat* k_raw [[buffer(1)]],
    const device bfloat* v_raw [[buffer(2)]], const device bfloat* rel_raw [[buffer(3)]],
    const device bfloat* q_scale [[buffer(4)]], const device bfloat* rel_scale [[buffer(5)]],
    const device int* rpos [[buffer(6)]], const device int* rseg [[buffer(7)]],
    device bfloat* q_out [[buffer(8)]], device bfloat* rel_out [[buffer(9)]],
    device bfloat* kc0 [[buffer(10)]], device bfloat* kc1 [[buffer(11)]],
    device bfloat* kc2 [[buffer(12)]], device bfloat* kc3 [[buffer(13)]],
    device bfloat* kc4 [[buffer(14)]], device bfloat* kc5 [[buffer(15)]],
    device bfloat* kc6 [[buffer(16)]], device bfloat* kc7 [[buffer(17)]],
    device bfloat* vc0 [[buffer(18)]], device bfloat* vc1 [[buffer(19)]],
    device bfloat* vc2 [[buffer(20)]], device bfloat* vc3 [[buffer(21)]],
    device bfloat* vc4 [[buffer(22)]], device bfloat* vc5 [[buffer(23)]],
    device bfloat* vc6 [[buffer(24)]], device bfloat* vc7 [[buffer(25)]],
    constant InkPrepSlotsArgs& a [[buffer(26)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, unit = tg.y, pos = uint(rpos[row]);
  const float tau_val = ink_tau(pos, a.global, a.floor_pos, a.alpha);
  if (unit < a.heads) {
    ink_prep_qrel(row, unit, lane, a.heads, a.global, a.qk_clip, a.rel_clip, tau_val, q_raw, rel_raw, q_scale,
                  rel_scale, q_out, rel_out);
    return;
  }
  const bool key = unit < a.heads + a.kv_heads;
  const uint h = key ? unit - a.heads : unit - a.heads - a.kv_heads;
  const device bfloat* kv_src = (key ? k_raw : v_raw) + (size_t(row) * a.kv_heads + h) * D;
  device bfloat* kc_arr[8] = {kc0, kc1, kc2, kc3, kc4, kc5, kc6, kc7};
  device bfloat* vc_arr[8] = {vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7};
  const uint slot = uint(rseg[row]);
  if (slot >= min(a.n_slots, 8u)) return;
  const uint ring = a.rings[slot];
  device bfloat* dst = (key ? kc_arr[slot] : vc_arr[slot]) + (size_t(h) * ring + pos % ring) * D;
  ink_prep_kv(kv_src, dst, key, a.qk_clip, a.eps, lane);
}

// The bf16 relative-bias table, one fp32 accumulation in projection tap order.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_rel_table(const device bfloat* rel [[buffer(0)]], const device bfloat* proj [[buffer(1)]],
    device bfloat* tab [[buffer(2)]], uint rh [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  float r[REL];
  for (uint j = 0; j < REL; ++j) r[j] = float(rel[rh * REL + j]);
  for (uint d = t; d < EXTENT; d += 256) {
    float acc = 0.0f;
    for (uint j = 0; j < REL; ++j) acc += r[j] * float(proj[j * EXTENT + d]);
    tab[rh * EXTENT + d] = bfloat(acc);
  }
}

inline float ink_attn_dot(float4 qv, const device bfloat* kc, size_t at, uint lane) {
  float4 kv = float4(*(const device bfloat4*)(kc + at + lane * 4));
  return ink_warp_sum(dot(qv, kv));
}
inline float ink_attn_bias(const device bfloat* tab, size_t prow, uint pos, uint kp) {
  const uint distance = pos - kp;
  return distance < EXTENT ? float(tab[prow * EXTENT + distance]) : 0.0f;
}
inline void ink_attn_step(float score, const device bfloat* vc_at, uint lane, thread float& top,
    thread float& total, thread float* acc) {
  const float next = max(top, score), a0 = precise::exp2(top - next), b0 = precise::exp2(score - next);
  const bfloat p = bfloat(b0);
  const float pf = float(p);
  const float4 vv = float4(*(const device bfloat4*)(vc_at + lane * 4));
  total = total * a0 + pf;
  acc[0] = acc[0] * a0 + pf * vv[0];
  acc[1] = acc[1] * a0 + pf * vv[1];
  acc[2] = acc[2] * a0 + pf * vv[2];
  acc[3] = acc[3] * a0 + pf * vv[3];
  top = next;
}
inline void ink_attn_store(device float* part_o, device float2* part_ml, size_t slot, float top, float total,
    const thread float* acc, uint lane) {
  if (lane == 0) part_ml[slot] = float2(top, total);
  for (uint i = 0; i < 4; ++i) part_o[slot * D + lane * 4 + i] = acc[i];
}

// A (row, head, key chunk) a simdgroup. Every P is bf16 before both its denominator add and P.V product.
[[max_total_threads_per_threadgroup(32)]]
kernel void tf_ink2_attn_scalar(const device bfloat* q [[buffer(0)]], const device bfloat* kc [[buffer(1)]],
    const device bfloat* vc [[buffer(2)]], const device bfloat* tab [[buffer(3)]],
    const device InkSeg* segs [[buffer(4)]], device float* part_o [[buffer(5)]], device float2* part_ml [[buffer(6)]],
    const device int* rpos [[buffer(7)]], const device int* rseg [[buffer(8)]], constant InkAttnArgs& a [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint rh = tg.x, row = rh / a.heads, h = rh % a.heads, group = a.heads / a.kv_heads, kvh = h / group;
  const uint pos = uint(rpos[row]);
  const uint lo = a.window && pos + 1 > a.window ? pos + 1 - a.window : 0;
  const uint first = a.window ? lo / a.chunk : a.c0;
  const uint c = first + tg.y;
  if (c > pos / a.chunk || (a.window && c < lo / a.chunk)) return;
  const uint from = max(lo, c * a.chunk), to = min(pos + 1, (c + 1) * a.chunk);
  const InkSeg s = segs[rseg[row]];
  const size_t head = s.off + size_t(kvh) * a.ring * D;
  const size_t prow = rh;
  const float4 qv = float4(*(const device bfloat4*)(q + prow * D + lane * 4));
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float top = -INFINITY, total = 0.0f;
  for (uint kp = from; kp < to; ++kp) {
    const size_t at = head + size_t(kp % a.ring) * D;
    const float dot_score = ink_attn_dot(qv, kc, at, lane);
    const float bias = ink_attn_bias(tab, prow, pos, kp);
    const float score = (dot_score * a.scale + bias) * LOG2E;
    ink_attn_step(score, vc + at, lane, top, total, acc);
  }
  const size_t slot = size_t(prow) * a.chunks + tg.y;
  ink_attn_store(part_o, part_ml, slot, top, total, acc, lane);
}

inline void ink_attn_step_v(float score, float4 vv, thread float& top, thread float& total, thread float* acc) {
  if (score <= top && isfinite(top)) {
    const float pf = float(bfloat(precise::exp2(score - top)));
    total += pf;
    acc[0] += pf * vv[0];
    acc[1] += pf * vv[1];
    acc[2] += pf * vv[2];
    acc[3] += pf * vv[3];
    return;
  }
  const float next = max(top, score), a0 = precise::exp2(top - next), b0 = precise::exp2(score - next);
  const float pf = float(bfloat(b0));
  total = total * a0 + pf;
  acc[0] = acc[0] * a0 + pf * vv[0];
  acc[1] = acc[1] * a0 + pf * vv[1];
  acc[2] = acc[2] * a0 + pf * vv[2];
  acc[3] = acc[3] * a0 + pf * vv[3];
  top = next;
}

inline void ink_attn_reduce8(threadgroup float2* s_ml, threadgroup float* s_o, device float* part_o,
    device float2* part_ml, size_t slot, float top, float tot, const thread float* acc, uint sg, uint lane) {
  if (lane == 0u) s_ml[sg] = float2(top, tot);
  for (uint i = 0u; i < 4u; ++i) s_o[sg * D + lane * 4u + i] = acc[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float m = -INFINITY;
    for (uint o = 0u; o < 8u; ++o) {
      const float2 ml = s_ml[o];
      m = (isnan(m) || isnan(ml.x) || isnan(ml.y)) ? NAN : max(m, ml.x);
    }
    float sum_tot = 0.0f;
    float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (m != -INFINITY) {
      for (uint o = 0u; o < 8u; ++o) {
        const float2 ml = s_ml[o];
        if (ml.x != -INFINITY || isnan(m)) {
          const float w = precise::exp2(ml.x - m);
          sum_tot += ml.y * w;
          for (uint i = 0u; i < 4u; ++i) out[i] += s_o[o * D + lane * 4u + i] * w;
        }
      }
    }
    ink_attn_store(part_o, part_ml, slot, m, sum_tot, out, lane);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// GQA 3:1 KV-sharing decode/verify kernel (3 Q-heads x 1 row, or x 2 rows on global verify), bit-identical to scalar.
[[max_total_threads_per_threadgroup(32)]]
kernel void tf_ink2_attn_gqa(const device bfloat* q [[buffer(0)]], const device bfloat* kc [[buffer(1)]],
    const device bfloat* vc [[buffer(2)]], const device bfloat* tab [[buffer(3)]],
    const device InkSeg* segs [[buffer(4)]], device float* part_o [[buffer(5)]], device float2* part_ml [[buffer(6)]],
    const device int* rpos [[buffer(7)]], const device int* rseg [[buffer(8)]], constant InkAttnArgs& a [[buffer(9)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint kvh = tg.x % a.kv_heads;
  const uint rb = tg.x / a.kv_heads;
  const bool pair_rows = (a.window == 0u && a.c0 > 1u);
  const uint rows_total = a.c0;
  const uint r0 = pair_rows ? 2u * rb : rb;
  if (r0 >= rows_total) return;
  const bool has_r1 = pair_rows && (r0 + 1u < rows_total);
  const uint r1 = has_r1 ? (r0 + 1u) : r0;
  const uint pos0 = uint(rpos[r0]);
  const uint pos1 = has_r1 ? uint(rpos[r1]) : pos0;
  const uint lo0 = a.window && pos0 + 1u > a.window ? pos0 + 1u - a.window : 0u;
  const uint first = a.window ? lo0 / a.chunk : 0u;
  const uint c = first + tg.y;
  const uint max_pos = has_r1 ? max(pos0, pos1) : pos0;
  if (c > max_pos / a.chunk || (a.window && c < lo0 / a.chunk)) return;
  const uint from = max(lo0, c * a.chunk);
  const uint to_max = min(max_pos + 1u, (c + 1u) * a.chunk);
  const InkSeg s = segs[rseg[r0]];
  const size_t head = s.off + size_t(kvh) * a.ring * D;
  const size_t h0 = size_t(kvh) * 3u;
  const size_t prow0 = size_t(r0) * a.heads + h0;
  const float4 q00 = float4(*(const device bfloat4*)(q + (prow0 + 0u) * D + lane * 4u));
  const float4 q01 = float4(*(const device bfloat4*)(q + (prow0 + 1u) * D + lane * 4u));
  const float4 q02 = float4(*(const device bfloat4*)(q + (prow0 + 2u) * D + lane * 4u));
  float acc0[3][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  float top0[3] = {-INFINITY, -INFINITY, -INFINITY}, tot0[3] = {0.0f, 0.0f, 0.0f};
  if (!has_r1) {
    for (uint kp = from; kp < to_max; ++kp) {
      const size_t at = head + size_t(kp % a.ring) * D;
      const float4 kv = float4(*(const device bfloat4*)(kc + at + lane * 4u));
      const float4 vv = float4(*(const device bfloat4*)(vc + at + lane * 4u));
      const float d0 = ink_warp_sum(dot(q00, kv));
      const float d1 = ink_warp_sum(dot(q01, kv));
      const float d2 = ink_warp_sum(dot(q02, kv));
      ink_attn_step_v((d0 * a.scale + ink_attn_bias(tab, prow0 + 0u, pos0, kp)) * LOG2E, vv, top0[0], tot0[0], acc0[0]);
      ink_attn_step_v((d1 * a.scale + ink_attn_bias(tab, prow0 + 1u, pos0, kp)) * LOG2E, vv, top0[1], tot0[1], acc0[1]);
      ink_attn_step_v((d2 * a.scale + ink_attn_bias(tab, prow0 + 2u, pos0, kp)) * LOG2E, vv, top0[2], tot0[2], acc0[2]);
    }
    for (uint g = 0u; g < 3u; ++g)
      ink_attn_store(part_o, part_ml, (prow0 + g) * a.chunks + tg.y, top0[g], tot0[g], acc0[g], lane);
    return;
  }
  const size_t prow1 = size_t(r1) * a.heads + h0;
  const float4 q10 = float4(*(const device bfloat4*)(q + (prow1 + 0u) * D + lane * 4u));
  const float4 q11 = float4(*(const device bfloat4*)(q + (prow1 + 1u) * D + lane * 4u));
  const float4 q12 = float4(*(const device bfloat4*)(q + (prow1 + 2u) * D + lane * 4u));
  float acc1[3][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  float top1[3] = {-INFINITY, -INFINITY, -INFINITY}, tot1[3] = {0.0f, 0.0f, 0.0f};
  for (uint kp = from; kp < to_max; ++kp) {
    const size_t at = head + size_t(kp % a.ring) * D;
    const float4 kv = float4(*(const device bfloat4*)(kc + at + lane * 4u));
    const float4 vv = float4(*(const device bfloat4*)(vc + at + lane * 4u));
    if (kp <= pos0) {
      const float d0 = ink_warp_sum(dot(q00, kv));
      const float d1 = ink_warp_sum(dot(q01, kv));
      const float d2 = ink_warp_sum(dot(q02, kv));
      ink_attn_step_v((d0 * a.scale + ink_attn_bias(tab, prow0 + 0u, pos0, kp)) * LOG2E, vv, top0[0], tot0[0], acc0[0]);
      ink_attn_step_v((d1 * a.scale + ink_attn_bias(tab, prow0 + 1u, pos0, kp)) * LOG2E, vv, top0[1], tot0[1], acc0[1]);
      ink_attn_step_v((d2 * a.scale + ink_attn_bias(tab, prow0 + 2u, pos0, kp)) * LOG2E, vv, top0[2], tot0[2], acc0[2]);
    }
    if (kp <= pos1) {
      const float d0 = ink_warp_sum(dot(q10, kv));
      const float d1 = ink_warp_sum(dot(q11, kv));
      const float d2 = ink_warp_sum(dot(q12, kv));
      ink_attn_step_v((d0 * a.scale + ink_attn_bias(tab, prow1 + 0u, pos1, kp)) * LOG2E, vv, top1[0], tot1[0], acc1[0]);
      ink_attn_step_v((d1 * a.scale + ink_attn_bias(tab, prow1 + 1u, pos1, kp)) * LOG2E, vv, top1[1], tot1[1], acc1[1]);
      ink_attn_step_v((d2 * a.scale + ink_attn_bias(tab, prow1 + 2u, pos1, kp)) * LOG2E, vv, top1[2], tot1[2], acc1[2]);
    }
  }
  if (c <= pos0 / a.chunk) {
    for (uint g = 0u; g < 3u; ++g)
      ink_attn_store(part_o, part_ml, (prow0 + g) * a.chunks + tg.y, top0[g], tot0[g], acc0[g], lane);
  }
  if (c <= pos1 / a.chunk) {
    for (uint g = 0u; g < 3u; ++g)
      ink_attn_store(part_o, part_ml, (prow1 + g) * a.chunks + tg.y, top1[g], tot1[g], acc1[g], lane);
  }
}

// 8-simdgroup split-K global attention with 3:1 GQA and 2-row verify KV sharing (4160 B TG scratch).
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_attn_split(const device bfloat* q [[buffer(0)]], const device bfloat* kc [[buffer(1)]],
    const device bfloat* vc [[buffer(2)]], const device bfloat* tab [[buffer(3)]],
    const device InkSeg* segs [[buffer(4)]], device float* part_o [[buffer(5)]], device float2* part_ml [[buffer(6)]],
    const device int* rpos [[buffer(7)]], const device int* rseg [[buffer(8)]], constant InkAttnArgs& a [[buffer(9)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float2 s_ml[8];
  threadgroup float s_o[8 * D];
  const uint kvh = tg.x % a.kv_heads;
  const uint rb = tg.x / a.kv_heads;
  const uint rows_total = a.c0;
  const bool pair_rows = rows_total > 1u;
  const uint r0 = pair_rows ? 2u * rb : rb;
  if (r0 >= rows_total) return;
  const bool has_r1 = pair_rows && (r0 + 1u < rows_total);
  const uint r1 = has_r1 ? (r0 + 1u) : r0;
  const uint pos0 = uint(rpos[r0]);
  const uint pos1 = has_r1 ? uint(rpos[r1]) : pos0;
  const uint c = tg.y;
  const uint max_pos = has_r1 ? max(pos0, pos1) : pos0;
  if (c > max_pos / a.chunk) return;
  const uint from = c * a.chunk;
  const uint to_max = min(max_pos + 1u, (c + 1u) * a.chunk);
  const uint n_keys = to_max > from ? to_max - from : 0u;
  const uint step = (n_keys + 7u) >> 3u;
  const uint s_from = from + sg * step;
  const uint s_to = min(s_from + step, to_max);
  const InkSeg s = segs[rseg[r0]];
  const size_t head = s.off + size_t(kvh) * a.ring * D;
  const size_t h0 = size_t(kvh) * 3u;
  const size_t prow0 = size_t(r0) * a.heads + h0;
  const float4 q00 = float4(*(const device bfloat4*)(q + (prow0 + 0u) * D + lane * 4u));
  const float4 q01 = float4(*(const device bfloat4*)(q + (prow0 + 1u) * D + lane * 4u));
  const float4 q02 = float4(*(const device bfloat4*)(q + (prow0 + 2u) * D + lane * 4u));
  float acc0[3][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  float top0[3] = {-INFINITY, -INFINITY, -INFINITY}, tot0[3] = {0.0f, 0.0f, 0.0f};
  if (!has_r1) {
    for (uint kp = s_from; kp < s_to; ++kp) {
      const size_t at = head + size_t(kp % a.ring) * D;
      const float4 kv = float4(*(const device bfloat4*)(kc + at + lane * 4u));
      const float4 vv = float4(*(const device bfloat4*)(vc + at + lane * 4u));
      const float d0 = ink_warp_sum(dot(q00, kv));
      const float d1 = ink_warp_sum(dot(q01, kv));
      const float d2 = ink_warp_sum(dot(q02, kv));
      ink_attn_step_v((d0 * a.scale + ink_attn_bias(tab, prow0 + 0u, pos0, kp)) * LOG2E, vv, top0[0], tot0[0], acc0[0]);
      ink_attn_step_v((d1 * a.scale + ink_attn_bias(tab, prow0 + 1u, pos0, kp)) * LOG2E, vv, top0[1], tot0[1], acc0[1]);
      ink_attn_step_v((d2 * a.scale + ink_attn_bias(tab, prow0 + 2u, pos0, kp)) * LOG2E, vv, top0[2], tot0[2], acc0[2]);
    }
    for (uint g = 0u; g < 3u; ++g)
      ink_attn_reduce8(s_ml, s_o, part_o, part_ml, (prow0 + g) * a.chunks + tg.y, top0[g], tot0[g], acc0[g], sg, lane);
    return;
  }
  const size_t prow1 = size_t(r1) * a.heads + h0;
  const float4 q10 = float4(*(const device bfloat4*)(q + (prow1 + 0u) * D + lane * 4u));
  const float4 q11 = float4(*(const device bfloat4*)(q + (prow1 + 1u) * D + lane * 4u));
  const float4 q12 = float4(*(const device bfloat4*)(q + (prow1 + 2u) * D + lane * 4u));
  float acc1[3][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  float top1[3] = {-INFINITY, -INFINITY, -INFINITY}, tot1[3] = {0.0f, 0.0f, 0.0f};
  for (uint kp = s_from; kp < s_to; ++kp) {
    const size_t at = head + size_t(kp % a.ring) * D;
    const float4 kv = float4(*(const device bfloat4*)(kc + at + lane * 4u));
    const float4 vv = float4(*(const device bfloat4*)(vc + at + lane * 4u));
    if (kp <= pos0) {
      const float d0 = ink_warp_sum(dot(q00, kv));
      const float d1 = ink_warp_sum(dot(q01, kv));
      const float d2 = ink_warp_sum(dot(q02, kv));
      ink_attn_step_v((d0 * a.scale + ink_attn_bias(tab, prow0 + 0u, pos0, kp)) * LOG2E, vv, top0[0], tot0[0], acc0[0]);
      ink_attn_step_v((d1 * a.scale + ink_attn_bias(tab, prow0 + 1u, pos0, kp)) * LOG2E, vv, top0[1], tot0[1], acc0[1]);
      ink_attn_step_v((d2 * a.scale + ink_attn_bias(tab, prow0 + 2u, pos0, kp)) * LOG2E, vv, top0[2], tot0[2], acc0[2]);
    }
    if (kp <= pos1) {
      const float d0 = ink_warp_sum(dot(q10, kv));
      const float d1 = ink_warp_sum(dot(q11, kv));
      const float d2 = ink_warp_sum(dot(q12, kv));
      ink_attn_step_v((d0 * a.scale + ink_attn_bias(tab, prow1 + 0u, pos1, kp)) * LOG2E, vv, top1[0], tot1[0], acc1[0]);
      ink_attn_step_v((d1 * a.scale + ink_attn_bias(tab, prow1 + 1u, pos1, kp)) * LOG2E, vv, top1[1], tot1[1], acc1[1]);
      ink_attn_step_v((d2 * a.scale + ink_attn_bias(tab, prow1 + 2u, pos1, kp)) * LOG2E, vv, top1[2], tot1[2], acc1[2]);
    }
  }
  if (c <= pos0 / a.chunk) {
    for (uint g = 0u; g < 3u; ++g)
      ink_attn_reduce8(s_ml, s_o, part_o, part_ml, (prow0 + g) * a.chunks + tg.y, top0[g], tot0[g], acc0[g], sg, lane);
  }
  if (c <= pos1 / a.chunk) {
    for (uint g = 0u; g < 3u; ++g)
      ink_attn_reduce8(s_ml, s_o, part_o, part_ml, (prow1 + g) * a.chunks + tg.y, top1[g], tot1[g], acc1[g], sg, lane);
  }
}

// Batches up to 8 streams in one launch; rseg[row] selects the slot's ring, chunk, and KV cache.
[[max_total_threads_per_threadgroup(32)]]
kernel void tf_ink2_attn_slots(const device bfloat* q [[buffer(0)]], const device bfloat* tab [[buffer(1)]],
    device float* part_o [[buffer(2)]], device float2* part_ml [[buffer(3)]],
    const device int* rpos [[buffer(4)]], const device int* rseg [[buffer(5)]],
    const device bfloat* kc0 [[buffer(6)]], const device bfloat* kc1 [[buffer(7)]],
    const device bfloat* kc2 [[buffer(8)]], const device bfloat* kc3 [[buffer(9)]],
    const device bfloat* kc4 [[buffer(10)]], const device bfloat* kc5 [[buffer(11)]],
    const device bfloat* kc6 [[buffer(12)]], const device bfloat* kc7 [[buffer(13)]],
    const device bfloat* vc0 [[buffer(14)]], const device bfloat* vc1 [[buffer(15)]],
    const device bfloat* vc2 [[buffer(16)]], const device bfloat* vc3 [[buffer(17)]],
    const device bfloat* vc4 [[buffer(18)]], const device bfloat* vc5 [[buffer(19)]],
    const device bfloat* vc6 [[buffer(20)]], const device bfloat* vc7 [[buffer(21)]],
    constant InkAttnSlotsArgs& a [[buffer(22)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint rh = tg.x, row = rh / a.heads, h = rh % a.heads, group = a.heads / a.kv_heads, kvh = h / group;
  const uint slot = uint(rseg[row]);
  if (slot >= min(a.n_slots, 8u)) return;
  const uint ring = a.slots[slot].ring, chunk = a.slots[slot].chunk;
  const uint pos = uint(rpos[row]);
  const uint lo = a.window && pos + 1 > a.window ? pos + 1 - a.window : 0;
  const uint first = a.window ? lo / chunk : a.c0;
  const uint c = first + tg.y;
  if (c > pos / chunk || (a.window && c < lo / chunk)) return;
  const uint from = max(lo, c * chunk), to = min(pos + 1, (c + 1) * chunk);
  const device bfloat* kc_arr[8] = {kc0, kc1, kc2, kc3, kc4, kc5, kc6, kc7};
  const device bfloat* vc_arr[8] = {vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7};
  const device bfloat* kc_s = kc_arr[slot];
  const device bfloat* vc_s = vc_arr[slot];
  const size_t head = size_t(kvh) * ring * D;
  const size_t prow = rh;
  const float4 qv = float4(*(const device bfloat4*)(q + prow * D + lane * 4));
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float top = -INFINITY, total = 0.0f;
  for (uint kp = from; kp < to; ++kp) {
    const size_t at = head + size_t(kp % ring) * D;
    const float dot_score = ink_attn_dot(qv, kc_s, at, lane);
    const float bias = ink_attn_bias(tab, prow, pos, kp);
    const float sc = (dot_score * a.scale + bias) * LOG2E;
    ink_attn_step(sc, vc_s + at, lane, top, total, acc);
  }
  ink_attn_store(part_o, part_ml, size_t(prow) * a.chunks + tg.y, top, total, acc, lane);
}

// Fold chunks and any sub-chunks in order, add the sink logit, apply the gate, round once to bf16.
[[max_total_threads_per_threadgroup(128)]]
kernel void tf_ink2_merge(const device float* part_o [[buffer(0)]], const device float2* part_ml [[buffer(1)]],
    const device bfloat* sink [[buffer(2)]], const device bfloat* gate [[buffer(3)]], const device int* rpos [[buffer(4)]],
    device bfloat* out [[buffer(5)]], constant InkMergeArgs& a [[buffer(6)]], uint rh [[threadgroup_position_in_grid]],
    uint d [[thread_index_in_threadgroup]]) {
  const uint row = rh / a.heads, pos = uint(rpos[row]);
  const uint lo = a.window && pos + 1 > a.window ? pos + 1 - a.window : 0;
  float mm = -INFINITY, ll = 0.0f, o = 0.0f;
  for (uint c = lo / a.chunk; c <= pos / a.chunk; ++c) {
    float cm, cl, co;
    if (a.subs > 1) {
      float sm = -INFINITY, sl = 0.0f, so = 0.0f;
      const uint last = min(a.subs - 1, (pos - c * a.chunk) / 1024);
      for (uint u = 0; u <= last; ++u) {
        const size_t at = size_t(rh) * a.chunks + (c - a.c0) * a.subs + u;
        const float2 ml = part_ml[at];
        const float mn = max(sm, ml.x);
        const float x = precise::exp2(sm - mn), y = precise::exp2(ml.x - mn);
        sl = sl * x + ml.y * y;
        so = so * x + part_o[at * D + d] * y;
        sm = mn;
      }
      cm = sm; cl = sl; co = so;
    } else {
      const size_t at = size_t(rh) * a.chunks + c - (a.window ? lo / a.chunk : a.c0);
      const float2 ml = part_ml[at];
      cm = ml.x; cl = ml.y; co = part_o[at * D + d];
    }
    const float mn = max(mm, cm), x = precise::exp2(mm - mn), y = precise::exp2(cm - mn);
    ll = ll * x + cl * y;
    o = o * x + co * y;
    mm = mn;
  }
  const float sk = float(sink[rh]) * LOG2E, mf = max(mm, sk), a0 = precise::exp2(mm - mf);
  const float y = o * a0 / (ll * a0 + precise::exp2(sk - mf));
  const float gv = float(gate[rh * D + d]);
  out[rh * D + d] = bfloat(y * (1.0f / (1.0f + precise::exp(-gv))));
}
