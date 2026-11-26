//! Inkling 2 Mini's scalar attention preparation, chunk decode and ordered sink/gate merge.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

constant constexpr uint D = 128, REL = 16, EXTENT = 1024, DECODE_TK = 32, DECODE_RB = 2;
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
  uint rows, pad;
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
  uint rows;
  InkSlotMeta slots[8];
};
struct InkMergeArgs {
  uint heads, window, chunk, c0, chunks, subs;
};

inline float ink_warp_sum(float v) {
  for (uint o = 16; o; o >>= 1) v += simd_shuffle_xor(v, ushort(o));
  return v;
}
inline float ink_clip_factor(float ss, float threshold) {
  return fmin(threshold / (precise::sqrt(ss) + 1e-6f), 1.0f);
}

inline void ink_prep_impl(const device bfloat* q_raw, const device bfloat* k_raw, const device bfloat* v_raw,
    const device bfloat* rel_raw, const device bfloat* q_scale, const device bfloat* rel_scale,
    const device int* rpos, const device int* rseg, const device InkSeg* segs,
    device bfloat* q_out, device bfloat* rel_out, device bfloat* kc, device bfloat* vc,
    constant InkPrepArgs& a, uint2 tg, uint lane, bool tree) {
  const uint row = tg.x, unit = tg.y, pos = uint(rpos[row]);
  const float tau = a.global ? 1.0f + a.alpha * precise::log(max(float(pos + 1) / a.floor_pos, 1.0f)) : 1.0f;
  float x[4];
  if (unit < a.heads) {
    const size_t rh = size_t(row) * a.heads + unit;
    for (uint i = 0; i < 4; ++i) x[i] = float(q_raw[rh * D + lane * 4 + i]) * float(q_scale[unit * D + lane * 4 + i]);
    float ss = ((x[0] * x[0] + x[1] * x[1]) + x[2] * x[2]) + x[3] * x[3];
    const float f = ink_clip_factor(ink_warp_sum(ss), a.qk_clip);
    for (uint i = 0; i < 4; ++i) {
      x[i] *= f;
      const bfloat once = bfloat(x[i]);
      q_out[rh * D + lane * 4 + i] = a.global ? bfloat(float(once) * tau) : once;
    }
    const float r = lane < REL ? float(rel_raw[rh * REL + lane]) * float(rel_scale[unit * REL + lane]) : 0.0f;
    const float rf = ink_clip_factor(ink_warp_sum(r * r), a.rel_clip);
    if (lane < REL) {
      const bfloat once = bfloat(r * rf);
      rel_out[rh * REL + lane] = a.global ? bfloat(float(once) * tau) : once;
    }
    return;
  }
  const bool key = unit < a.heads + a.kv_heads;
  const uint h = key ? unit - a.heads : unit - a.heads - a.kv_heads;
  const device bfloat* src = (key ? k_raw : v_raw) + (size_t(row) * a.kv_heads + h) * D;
  for (uint i = 0; i < 4; ++i) x[i] = float(src[lane * 4 + i]);
  float scale;
  if (key) {
    const float ss = ((x[0] * x[0] + x[1] * x[1]) + x[2] * x[2]) + x[3] * x[3];
    scale = ink_clip_factor(ink_warp_sum(ss), a.qk_clip);
  } else {
    const float ss = ((x[0] * x[0] + x[1] * x[1]) + x[2] * x[2]) + x[3] * x[3];
    scale = precise::rsqrt(ink_warp_sum(ss) / float(D) + a.eps);
  }
  const InkSeg s = segs[rseg[row]];
  const uint kv_pos = tree ? uint(s.p0) + row - uint(s.row0) : pos;
  device bfloat* dst = (key ? kc : vc) + s.off + (size_t(h) * a.ring + kv_pos % a.ring) * D;
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
  ink_prep_impl(q_raw, k_raw, v_raw, rel_raw, q_scale, rel_scale, rpos, rseg, segs, q_out, rel_out, kc, vc, a, tg, lane, false);
}

[[max_total_threads_per_threadgroup(32)]]
kernel void tf_ink2_prep_tree(const device bfloat* q_raw [[buffer(0)]], const device bfloat* k_raw [[buffer(1)]],
    const device bfloat* v_raw [[buffer(2)]], const device bfloat* rel_raw [[buffer(3)]],
    const device bfloat* q_scale [[buffer(4)]], const device bfloat* rel_scale [[buffer(5)]],
    const device int* rpos [[buffer(6)]], const device int* rseg [[buffer(7)]],
    const device InkSeg* segs [[buffer(8)]], device bfloat* q_out [[buffer(9)]],
    device bfloat* rel_out [[buffer(10)]], device bfloat* kc [[buffer(11)]], device bfloat* vc [[buffer(12)]],
    constant InkPrepArgs& a [[buffer(13)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  ink_prep_impl(q_raw, k_raw, v_raw, rel_raw, q_scale, rel_scale, rpos, rseg, segs, q_out, rel_out, kc, vc, a, tg, lane, true);
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

inline void ink_attn_scalar_impl(const device bfloat* q, const device bfloat* kc, const device bfloat* vc,
    const device bfloat* rel, const device bfloat* proj, const device InkSeg* segs, device float* part_o,
    device float2* part_ml, const device int* rpos, const device int* rseg, const device uint* tree_paths,
    constant InkAttnArgs& a, uint2 tg, uint lane, bool tree) {
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
  const bool has_rel = from < to && (pos - (to - 1)) < EXTENT;
  const float r_lane = (has_rel && lane < REL) ? float(rel[prow * REL + lane]) : 0.0f;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float top = -INFINITY, total = 0.0f;
  for (uint k0 = from; k0 < to; k0 += 32) {
    const uint tile_to = min(k0 + 32, to);
    const bool tile_bias = has_rel && (pos - (tile_to - 1)) < EXTENT;
    float bias_lane = 0.0f;
    if (tile_bias) {
      const uint kp_lane = k0 + lane;
      const bool need_bias = kp_lane < tile_to && (pos - kp_lane) < EXTENT;
      const uint dist_safe = need_bias ? (pos - kp_lane) : 0;
      float acc_b = 0.0f;
      for (uint j = 0; j < REL; ++j) {
        const float r_j = simd_shuffle(r_lane, ushort(j));
        acc_b += r_j * (need_bias ? float(proj[j * EXTENT + dist_safe]) : 0.0f);
      }
      bias_lane = need_bias ? float(bfloat(acc_b)) : 0.0f;
    }
    for (uint kp = k0; kp < tile_to; ++kp) {
      const uint phys = (tree && kp >= uint(s.p0)) ? uint(s.p0) + tree_paths[row * 16 + (kp - uint(s.p0))] : kp;
      const size_t at = head + size_t(phys % a.ring) * D;
      const float4 kv = float4(*(const device bfloat4*)(kc + at + lane * 4));
      const float dot_score = ink_warp_sum(dot(qv, kv));
      const float bias = tile_bias ? simd_shuffle(bias_lane, ushort(kp - k0)) : 0.0f;
      const float score = (dot_score * a.scale + bias) * LOG2E;
      const float next = max(top, score), a0 = precise::exp2(top - next), b0 = precise::exp2(score - next);
      const bfloat p = bfloat(b0);
      total = total * a0 + float(p);
      for (uint i = 0; i < 4; ++i) acc[i] = acc[i] * a0 + float(p) * float(vc[at + lane * 4 + i]);
      top = next;
    }
  }
  const size_t slot = size_t(prow) * a.chunks + tg.y;
  if (lane == 0) part_ml[slot] = float2(top, total);
  for (uint i = 0; i < 4; ++i) part_o[slot * D + lane * 4 + i] = acc[i];
}

inline float ink_attn_bias(const device bfloat* rel, const device bfloat* proj, size_t prow, uint pos, uint kp, uint lane) {
  const uint distance = pos - kp;
  if (distance >= EXTENT) return 0.0f;
  const float r_lane = lane < REL ? float(rel[prow * REL + lane]) : 0.0f;
  float acc = 0.0f;
  for (uint j = 0; j < REL; ++j) {
    const float r_j = simd_shuffle(r_lane, ushort(j));
    acc += r_j * float(proj[j * EXTENT + distance]);
  }
  return float(bfloat(acc));
}
inline float ink_attn_score(float dot_score, float scale, float bias, uint kp, uint from) {
  (void)kp;
  (void)from;
  const float score = (dot_score * scale + bias) * LOG2E;
  return score;
}
inline void ink_attn_step_v(float score, float4 vv, thread float& top, thread float& total, thread float4& acc) {
  const float next = max(top, score), a0 = precise::exp2(top - next), b0 = precise::exp2(score - next);
  const bfloat p = bfloat(b0);
  total = total * a0 + float(p);
  acc = acc * a0 + float(p) * vv;
  top = next;
}
inline void ink_attn_store_v(device float* part_o, device float2* part_ml, size_t slot, float top, float total,
    float4 acc, uint lane) {
  if (lane == 0) part_ml[slot] = float2(top, total);
  *(device float4*)(part_o + slot * D + lane * 4) = acc;
}

inline void ink_attn_pair(const device bfloat* q, const device bfloat* kc, const device bfloat* vc,
    const device bfloat* rel, const device bfloat* proj, device float* part_o, device float2* part_ml,
    threadgroup bfloat* ks, threadgroup bfloat* vs, const thread uint* act_rows,
    const thread uint* act_pos, const thread uint* act_from, const thread uint* act_to,
    uint n_act, uint heads, uint kv_heads, uint kvh, uint ring, uint tg_c, uint chunks,
    float scale, uint tg_from, uint tg_to, uint sg, uint lane, bool more_passes) {
  const uint group = heads / kv_heads, coh = sg / 3, g = sg % 3, h = kvh * group + g;
  const bool active = (coh < n_act) && (g < group);
  const size_t head = size_t(kvh) * ring * D;
  const uint pos = coh < n_act ? act_pos[coh] : 0;
  const uint from = coh < n_act ? act_from[coh] : 1;
  const uint to = coh < n_act ? act_to[coh] : 0;
  const size_t prow = active ? size_t(act_rows[coh]) * heads + h : 0;
  const float4 qf = active ? float4(*(const device bfloat4*)(q + prow * D + lane * 4)) : float4(0.0f);
  float4 acc = float4(0.0f);
  float top = -INFINITY, total = 0.0f;
  for (uint kt = tg_from; kt < tg_to; kt += DECODE_TK) {
    const uint tile_n = min(DECODE_TK, tg_to - kt);
    const uint k_mod = kt % ring;
    for (uint k = sg; k < tile_n; k += 6) {
      const uint k_off = k_mod + k;
      const uint phys = k_off < ring ? k_off : k_off % ring;
      const size_t at = head + size_t(phys) * D + lane * 4;
      *(threadgroup bfloat4*)(ks + k * D + lane * 4) = *(const device bfloat4*)(kc + at);
      *(threadgroup bfloat4*)(vs + k * D + lane * 4) = *(const device bfloat4*)(vc + at);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      uint k = 0;
      if (kt >= from && kt + tile_n <= to) {
        for (; k + 4 <= tile_n; k += 4) {
          const float4 k0 = float4(*(const threadgroup bfloat4*)(ks + (k + 0) * D + lane * 4));
          const float4 k1 = float4(*(const threadgroup bfloat4*)(ks + (k + 1) * D + lane * 4));
          const float4 k2 = float4(*(const threadgroup bfloat4*)(ks + (k + 2) * D + lane * 4));
          const float4 k3 = float4(*(const threadgroup bfloat4*)(ks + (k + 3) * D + lane * 4));
          const float d0 = ink_warp_sum(dot(qf, k0)), d1 = ink_warp_sum(dot(qf, k1));
          const float d2 = ink_warp_sum(dot(qf, k2)), d3 = ink_warp_sum(dot(qf, k3));
          const float s0 = ink_attn_score(d0, scale, ink_attn_bias(rel, proj, prow, pos, kt + k + 0, lane), kt + k + 0, from);
          const float s1 = ink_attn_score(d1, scale, ink_attn_bias(rel, proj, prow, pos, kt + k + 1, lane), kt + k + 1, from);
          const float s2 = ink_attn_score(d2, scale, ink_attn_bias(rel, proj, prow, pos, kt + k + 2, lane), kt + k + 2, from);
          const float s3 = ink_attn_score(d3, scale, ink_attn_bias(rel, proj, prow, pos, kt + k + 3, lane), kt + k + 3, from);
          ink_attn_step_v(s0, float4(*(const threadgroup bfloat4*)(vs + (k + 0) * D + lane * 4)), top, total, acc);
          ink_attn_step_v(s1, float4(*(const threadgroup bfloat4*)(vs + (k + 1) * D + lane * 4)), top, total, acc);
          ink_attn_step_v(s2, float4(*(const threadgroup bfloat4*)(vs + (k + 2) * D + lane * 4)), top, total, acc);
          ink_attn_step_v(s3, float4(*(const threadgroup bfloat4*)(vs + (k + 3) * D + lane * 4)), top, total, acc);
        }
      }
      for (; k < tile_n; ++k) {
        const uint kp = kt + k;
        if (kp >= from && kp < to) {
          const float4 kv = float4(*(const threadgroup bfloat4*)(ks + k * D + lane * 4));
          const float4 vv = float4(*(const threadgroup bfloat4*)(vs + k * D + lane * 4));
          const float dot_score = ink_warp_sum(dot(qf, kv));
          const float bias = ink_attn_bias(rel, proj, prow, pos, kp, lane);
          const float score = ink_attn_score(dot_score, scale, bias, kp, from);
          ink_attn_step_v(score, vv, top, total, acc);
        }
      }
    }
    if (kt + DECODE_TK < tg_to || more_passes) threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (active) ink_attn_store_v(part_o, part_ml, prow * chunks + tg_c, top, total, acc, lane);
}

inline void ink_attn_group(const device bfloat* q, const device bfloat* kc, const device bfloat* vc,
    const device bfloat* rel, const device bfloat* proj, device float* part_o, device float2* part_ml,
    threadgroup bfloat* ks, threadgroup bfloat* vs, const thread uint* rows, const thread uint* rpos,
    uint n_rows, uint heads, uint kv_heads, uint kvh, uint ring, uint window, uint arg_chunk,
    uint c0, uint chunks, float scale, uint tg_c, uint sg, uint lane) {
  uint r_pos[DECODE_RB], r_from[DECODE_RB], r_to[DECODE_RB], r_chunk[DECODE_RB];
  uint pending = 0;
  for (uint i = 0; i < n_rows; ++i) {
    const uint pos = rpos[i];
    const uint chunk = arg_chunk;
    const uint lo = window && pos + 1 > window ? pos + 1 - window : 0;
    const uint first = window ? lo / chunk : c0;
    const uint c = first + tg_c;
    if (c <= pos / chunk && !(window && c < lo / chunk)) {
      r_pos[i] = pos;
      r_from[i] = max(lo, c * chunk);
      r_to[i] = min(pos + 1, (c + 1) * chunk);
      r_chunk[i] = chunk;
      pending |= (1u << i);
    }
  }
  while (pending) {
    const uint r0 = ctz(pending);
    const uint base_chunk = r_chunk[r0], base_from = r_from[r0], base_to = r_to[r0];
    uint pass_mask = 0;
    uint tg_from = base_from, tg_to = base_to;
    for (uint m = pending; m; ) {
      const uint i = ctz(m);
      m &= m - 1;
      const bool same = window ? (r_from[i] < base_to + DECODE_TK && r_to[i] + DECODE_TK > base_from)
                               : (r_chunk[i] == base_chunk);
      if (same) {
        pass_mask |= (1u << i);
        tg_from = min(tg_from, r_from[i]);
        tg_to = max(tg_to, r_to[i]);
      }
    }
    pending &= ~pass_mask;
    uint act_rows[DECODE_RB], act_pos[DECODE_RB], act_from[DECODE_RB], act_to[DECODE_RB];
    uint n_act = 0;
    for (uint m = pass_mask; m; ) {
      const uint i = ctz(m);
      m &= m - 1;
      act_rows[n_act] = rows[i];
      act_pos[n_act] = r_pos[i];
      act_from[n_act] = r_from[i];
      act_to[n_act] = r_to[i];
      n_act++;
    }
    ink_attn_pair(q, kc, vc, rel, proj, part_o, part_ml, ks, vs, act_rows, act_pos, act_from, act_to, n_act,
                  heads, kv_heads, kvh, ring, tg_c, chunks, scale, tg_from, tg_to, sg, lane, pending != 0);
  }
}

// Two 3-simdgroup cohorts share staged (K, V) across 3 GQA heads and up to 2 rows per chunk.
[[max_total_threads_per_threadgroup(192)]]
kernel void tf_ink2_attn_scalar(const device bfloat* q [[buffer(0)]], const device bfloat* kc [[buffer(1)]],
    const device bfloat* vc [[buffer(2)]], const device bfloat* rel [[buffer(3)]],
    const device bfloat* proj [[buffer(4)]], const device InkSeg* segs [[buffer(5)]],
    device float* part_o [[buffer(6)]], device float2* part_ml [[buffer(7)]],
    const device int* rpos [[buffer(8)]], const device int* rseg [[buffer(9)]],
    constant InkAttnArgs& a [[buffer(10)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat4 ks_buf[DECODE_TK * (D / 4)], vs_buf[DECODE_TK * (D / 4)];
  const uint kvh = tg.x, row0 = tg.z * DECODE_RB;
  if (row0 >= a.rows) return;
  const uint count = min(DECODE_RB, a.rows - row0);
  uint seg_pending = (1u << count) - 1u;
  while (seg_pending) {
    const uint first = ctz(seg_pending);
    const int seg_id = rseg[row0 + first];
    uint rows[DECODE_RB], pos[DECODE_RB];
    uint n = 0;
    for (uint m = seg_pending; m; ) {
      const uint i = ctz(m);
      m &= m - 1;
      if (rseg[row0 + i] == seg_id) {
        seg_pending &= ~(1u << i);
        rows[n] = row0 + i;
        pos[n] = uint(rpos[row0 + i]);
        n++;
      }
    }
    const InkSeg s = segs[seg_id];
    ink_attn_group(q, kc + s.off, vc + s.off, rel, proj, part_o, part_ml, (threadgroup bfloat*)ks_buf,
                   (threadgroup bfloat*)vs_buf, rows, pos, n, a.heads, a.kv_heads, kvh, a.ring, a.window, a.chunk,
                   a.c0, a.chunks, a.scale, tg.y, sg, lane);
    if (seg_pending) threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Batches up to 8 stream slots; each 192-thread TG shares staged (K, V) across 3 GQA heads and 2 slot rows.
[[max_total_threads_per_threadgroup(192)]]
kernel void tf_ink2_attn_slots(const device bfloat* q [[buffer(0)]], const device bfloat* rel [[buffer(1)]],
    const device bfloat* proj [[buffer(2)]], device float* part_o [[buffer(3)]], device float2* part_ml [[buffer(4)]],
    const device int* rpos [[buffer(5)]], const device int* rseg [[buffer(6)]],
    const device bfloat* kc0 [[buffer(7)]], const device bfloat* kc1 [[buffer(8)]],
    const device bfloat* kc2 [[buffer(9)]], const device bfloat* kc3 [[buffer(10)]],
    const device bfloat* kc4 [[buffer(11)]], const device bfloat* kc5 [[buffer(12)]],
    const device bfloat* kc6 [[buffer(13)]], const device bfloat* kc7 [[buffer(14)]],
    const device bfloat* vc0 [[buffer(15)]], const device bfloat* vc1 [[buffer(16)]],
    const device bfloat* vc2 [[buffer(17)]], const device bfloat* vc3 [[buffer(18)]],
    const device bfloat* vc4 [[buffer(19)]], const device bfloat* vc5 [[buffer(20)]],
    const device bfloat* vc6 [[buffer(21)]], const device bfloat* vc7 [[buffer(22)]],
    constant InkAttnSlotsArgs& a [[buffer(23)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat4 ks_buf[DECODE_TK * (D / 4)], vs_buf[DECODE_TK * (D / 4)];
  const uint slot = tg.x / a.kv_heads, kvh = tg.x % a.kv_heads;
  if (slot >= min(a.n_slots, 8u)) return;
  uint rows[DECODE_RB], pos[DECODE_RB];
  uint n = 0, seen = 0;
  const uint skip = tg.z * DECODE_RB;
  for (uint r = 0; r < a.rows && n < DECODE_RB; ++r) {
    if (uint(rseg[r]) == slot) {
      if (seen >= skip) {
        rows[n] = r;
        pos[n] = uint(rpos[r]);
        n++;
      }
      seen++;
    }
  }
  if (n == 0) return;
  const device bfloat* kc_arr[8] = {kc0, kc1, kc2, kc3, kc4, kc5, kc6, kc7};
  const device bfloat* vc_arr[8] = {vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7};
  const device bfloat* kc_s = kc_arr[slot];
  const device bfloat* vc_s = vc_arr[slot];
  ink_attn_group(q, kc_s, vc_s, rel, proj, part_o, part_ml, (threadgroup bfloat*)ks_buf,
                 (threadgroup bfloat*)vs_buf, rows, pos, n, a.heads, a.kv_heads, kvh, a.slots[slot].ring,
                 a.window, a.slots[slot].chunk, a.c0, a.chunks, a.scale, tg.y, sg, lane);
}

[[max_total_threads_per_threadgroup(32)]]
kernel void tf_ink2_attn_scalar_tree(const device bfloat* q [[buffer(0)]], const device bfloat* kc [[buffer(1)]],
    const device bfloat* vc [[buffer(2)]], const device bfloat* rel [[buffer(3)]],
    const device bfloat* proj [[buffer(4)]], const device InkSeg* segs [[buffer(5)]],
    device float* part_o [[buffer(6)]], device float2* part_ml [[buffer(7)]],
    const device int* rpos [[buffer(8)]], const device int* rseg [[buffer(9)]],
    const device uint* tree_paths [[buffer(10)]], constant InkAttnArgs& a [[buffer(11)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  ink_attn_scalar_impl(q, kc, vc, rel, proj, segs, part_o, part_ml, rpos, rseg, tree_paths, a, tg, lane, true);
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
