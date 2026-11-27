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
struct InkBatchedSeg {
  device bfloat* k_cache;
  device bfloat* v_cache;
  int row0, n, p0;
  uint ring;
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

// Multi-segment prep: writes each segment's K/V through its own GPU pointer and ring in one launch.
[[max_total_threads_per_threadgroup(32)]]
kernel void tf_ink2_prep_batched(const device bfloat* q_raw [[buffer(0)]], const device bfloat* k_raw [[buffer(1)]],
    const device bfloat* v_raw [[buffer(2)]], const device bfloat* rel_raw [[buffer(3)]],
    const device bfloat* q_scale [[buffer(4)]], const device bfloat* rel_scale [[buffer(5)]],
    const device int* rpos [[buffer(6)]], const device int* rseg [[buffer(7)]],
    constant InkBatchedSeg* segs [[buffer(8)]], device bfloat* q_out [[buffer(9)]],
    device bfloat* rel_out [[buffer(10)]], constant InkPrepArgs& a [[buffer(11)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, unit = tg.y, pos = uint(rpos[row]);
  const float t_scale = a.global ? 1.0f + a.alpha * precise::log(max(float(pos + 1) / a.floor_pos, 1.0f)) : 1.0f;
  float x[4];
  if (unit < a.heads) {
    const size_t rh = size_t(row) * a.heads + unit;
    for (uint i = 0; i < 4; ++i) x[i] = float(q_raw[rh * D + lane * 4 + i]) * float(q_scale[unit * D + lane * 4 + i]);
    const float ss = ((x[0] * x[0] + x[1] * x[1]) + x[2] * x[2]) + x[3] * x[3];
    const float f = ink_clip_factor(ink_warp_sum(ss), a.qk_clip);
    for (uint i = 0; i < 4; ++i) {
      x[i] *= f;
      const bfloat once = bfloat(x[i]);
      q_out[rh * D + lane * 4 + i] = a.global ? bfloat(float(once) * t_scale) : once;
    }
    const float r = lane < REL ? float(rel_raw[rh * REL + lane]) * float(rel_scale[unit * REL + lane]) : 0.0f;
    const float rf = ink_clip_factor(ink_warp_sum(r * r), a.rel_clip);
    if (lane < REL) {
      const bfloat once = bfloat(r * rf);
      rel_out[rh * REL + lane] = a.global ? bfloat(float(once) * t_scale) : once;
    }
    return;
  }
  const bool is_k = unit < a.heads + a.kv_heads;
  const uint h = is_k ? unit - a.heads : unit - a.heads - a.kv_heads;
  const device bfloat* kv_src = (is_k ? k_raw : v_raw) + (size_t(row) * a.kv_heads + h) * D;
  for (uint i = 0; i < 4; ++i) x[i] = float(kv_src[lane * 4 + i]);
  const float ss = ((x[0] * x[0] + x[1] * x[1]) + x[2] * x[2]) + x[3] * x[3];
  const float scale = is_k ? ink_clip_factor(ink_warp_sum(ss), a.qk_clip) : precise::rsqrt(ink_warp_sum(ss) / float(D) + a.eps);
  const InkBatchedSeg bs = segs[rseg[row]];
  device bfloat* dst = (is_k ? bs.k_cache : bs.v_cache) + (size_t(h) * bs.ring + pos % bs.ring) * D;
  for (uint i = 0; i < 4; ++i) dst[lane * 4 + i] = bfloat(x[i] * scale);
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

// A (row, head, key chunk) a simdgroup; relative biases project on the fly in tap order and round to bf16.
[[max_total_threads_per_threadgroup(32)]]
kernel void tf_ink2_attn_scalar(const device bfloat* q [[buffer(0)]], const device bfloat* kc [[buffer(1)]],
    const device bfloat* vc [[buffer(2)]], const device bfloat* rel [[buffer(3)]],
    const device bfloat* proj [[buffer(4)]], const device InkSeg* segs [[buffer(5)]],
    device float* part_o [[buffer(6)]], device float2* part_ml [[buffer(7)]],
    const device int* rpos [[buffer(8)]], const device int* rseg [[buffer(9)]],
    constant InkAttnArgs& a [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  ink_attn_scalar_impl(q, kc, vc, rel, proj, segs, part_o, part_ml, rpos, rseg, nullptr, a, tg, lane, false);
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

// GQA 3:1 exact 0-ULP decode core: one simdgroup processes all 3 query heads sharing (row, kvh, chunk).
inline void ink_attn_gqa3_core(const device bfloat* q, const device bfloat* kc, const device bfloat* vc,
    const device bfloat* rel, const device bfloat* proj, device float* part_o, device float2* part_ml,
    uint pos, uint ring, uint row, uint kvh, uint chunk_idx, constant InkAttnArgs& a, uint lane) {
  const uint lo = a.window && pos + 1 > a.window ? pos + 1 - a.window : 0;
  const uint first = a.window ? lo / a.chunk : a.c0;
  const uint c = first + chunk_idx;
  if (c > pos / a.chunk || (a.window && c < lo / a.chunk)) return;
  const uint from = max(lo, c * a.chunk), to = min(pos + 1, (c + 1) * a.chunk);
  const size_t head = size_t(kvh) * ring * D;
  const size_t prow0 = size_t(row) * a.heads + size_t(kvh) * 3;
  float4 qv[3];
  float r_lane[3];
  const bool has_rel = from < to && (pos - (to - 1)) < EXTENT;
  for (uint g = 0; g < 3; ++g) {
    qv[g] = float4(*(const device bfloat4*)(q + (prow0 + g) * D + lane * 4));
    r_lane[g] = (has_rel && lane < REL) ? float(rel[(prow0 + g) * REL + lane]) : 0.0f;
  }
  float acc[3][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
  float top[3] = {-INFINITY, -INFINITY, -INFINITY};
  float total[3] = {0.0f, 0.0f, 0.0f};
  for (uint k0 = from; k0 < to; k0 += 32) {
    const uint tile_to = min(k0 + 32, to);
    const bool tile_bias = has_rel && (pos - (tile_to - 1)) < EXTENT;
    float bias_lane[3] = {0.0f, 0.0f, 0.0f};
    if (tile_bias) {
      const uint kp_lane = k0 + lane;
      const bool need_bias = kp_lane < tile_to && (pos - kp_lane) < EXTENT;
      const uint dist_safe = need_bias ? (pos - kp_lane) : 0;
      float acc_b[3] = {0.0f, 0.0f, 0.0f};
      for (uint j = 0; j < REL; ++j) {
        const float pj = need_bias ? float(proj[j * EXTENT + dist_safe]) : 0.0f;
        for (uint g = 0; g < 3; ++g) {
          const float r_jg = simd_shuffle(r_lane[g], ushort(j));
          acc_b[g] += r_jg * pj;
        }
      }
      for (uint g = 0; g < 3; ++g) bias_lane[g] = need_bias ? float(bfloat(acc_b[g])) : 0.0f;
    }
    for (uint kp = k0; kp < tile_to; ++kp) {
      const size_t at = head + size_t(kp % ring) * D;
      const float4 kv = float4(*(const device bfloat4*)(kc + at + lane * 4));
      const float4 vv = float4(*(const device bfloat4*)(vc + at + lane * 4));
      for (uint g = 0; g < 3; ++g) {
        const float dot_score = ink_warp_sum(dot(qv[g], kv));
        const float bias = tile_bias ? simd_shuffle(bias_lane[g], ushort(kp - k0)) : 0.0f;
        const float score_g = (dot_score * a.scale + bias) * LOG2E;
        const float next = max(top[g], score_g), a0 = precise::exp2(top[g] - next), b0 = precise::exp2(score_g - next);
        const bfloat p = bfloat(b0);
        total[g] = total[g] * a0 + float(p);
        for (uint i = 0; i < 4; ++i) acc[g][i] = acc[g][i] * a0 + float(p) * vv[i];
        top[g] = next;
      }
    }
  }
  for (uint g = 0; g < 3; ++g) {
    const size_t slot = (prow0 + g) * a.chunks + chunk_idx;
    if (lane == 0) part_ml[slot] = float2(top[g], total[g]);
    for (uint i = 0; i < 4; ++i) part_o[slot * D + lane * 4 + i] = acc[g][i];
  }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void tf_ink2_attn_scalar_gqa(const device bfloat* q [[buffer(0)]], const device bfloat* kc [[buffer(1)]],
    const device bfloat* vc [[buffer(2)]], const device bfloat* rel [[buffer(3)]],
    const device bfloat* proj [[buffer(4)]], const device InkSeg* segs [[buffer(5)]],
    device float* part_o [[buffer(6)]], device float2* part_ml [[buffer(7)]],
    const device int* rpos [[buffer(8)]], const device int* rseg [[buffer(9)]],
    constant InkAttnArgs& a [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint rkvh = tg.x, row = rkvh / a.kv_heads, kvh = rkvh % a.kv_heads;
  const InkSeg s = segs[rseg[row]];
  ink_attn_gqa3_core(q, kc + s.off, vc + s.off, rel, proj, part_o, part_ml, uint(rpos[row]), a.ring, row, kvh, tg.y, a, lane);
}

[[max_total_threads_per_threadgroup(32)]]
kernel void tf_ink2_attn_scalar_gqa_batched(const device bfloat* q [[buffer(0)]], const device bfloat* rel [[buffer(1)]],
    const device bfloat* proj [[buffer(2)]], constant InkBatchedSeg* segs [[buffer(3)]],
    device float* part_o [[buffer(4)]], device float2* part_ml [[buffer(5)]],
    const device int* rpos [[buffer(6)]], const device int* rseg [[buffer(7)]],
    constant InkAttnArgs& a [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint rkvh = tg.x, row = rkvh / a.kv_heads, kvh = rkvh % a.kv_heads;
  const InkBatchedSeg bseg = segs[rseg[row]];
  ink_attn_gqa3_core(q, bseg.k_cache, bseg.v_cache, rel, proj, part_o, part_ml, uint(rpos[row]), bseg.ring, row, kvh, tg.y, a, lane);
}

constant constexpr uint SLIDE_TG_ROWS = 2, SLIDE_G = 3, SLIDE_TILE_K = 32, SLIDE_K_STEP = 8;

inline uint ink_slide_lo(uint window, uint pos) {
  return window && pos + 1 > window ? pos + 1 - window : 0;
}

inline void ink_slide_step(float dot_score, float bias, float scale, float4 vv, uint kp, uint from,
    thread float& top, thread float& total, thread float (&acc)[4]) {
  (void)kp;
  (void)from;
  const float logit = (dot_score * scale + bias) * LOG2E;
  const float next = max(top, logit), a0 = precise::exp2(top - next), b0 = precise::exp2(logit - next);
  const bfloat sp = bfloat(b0);
  total = total * a0 + float(sp);
  for (uint i = 0; i < 4; ++i) acc[i] = acc[i] * a0 + float(sp) * vv[i];
  top = next;
}

// 2 rows x 3 GQA heads = 6 active simdgroups per 256-thread group; all 8 simdgroups stage 32-key tiles.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_attn_sliding_prompt(const device bfloat* q [[buffer(0)]], const device bfloat* kc [[buffer(1)]],
    const device bfloat* vc [[buffer(2)]], const device bfloat* rel [[buffer(3)]],
    const device bfloat* proj [[buffer(4)]], const device InkSeg* segs [[buffer(5)]],
    device float* part_o [[buffer(6)]], device float2* part_ml [[buffer(7)]],
    const device int* rpos [[buffer(8)]], const device int* rseg [[buffer(9)]],
    constant InkAttnArgs& a [[buffer(10)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  (void)rseg;
  threadgroup bfloat4 ks[SLIDE_TILE_K * (D / 4)], vs[SLIDE_TILE_K * (D / 4)];
  if (a.heads != a.kv_heads * SLIDE_G) return;
  const InkSeg s = segs[0];
  const uint row0 = tg.x * SLIDE_TG_ROWS, kvh = tg.y, c = a.c0 + tg.z;
  const uint local_r = sg / SLIDE_G, g = sg % SLIDE_G, row = row0 + local_r;
  uint tg_from = 0xffffffffu, tg_to = 0;
  for (uint r = 0; r < SLIDE_TG_ROWS; ++r) {
    const uint rr = row0 + r;
    if (rr >= uint(s.n)) break;
    const uint rp = uint(rpos[rr]), rlo = ink_slide_lo(a.window, rp);
    const uint rfirst = a.window ? rlo / a.chunk : a.c0;
    if (c >= rfirst && c <= rp / a.chunk) {
      tg_from = min(tg_from, max(rlo, c * a.chunk));
      tg_to = max(tg_to, min(rp + 1, (c + 1) * a.chunk));
    }
  }
  if (tg_from >= tg_to) return;
  const bool row_live = sg < SLIDE_TG_ROWS * SLIDE_G && row < uint(s.n);
  const uint pos = row_live ? uint(rpos[row]) : 0;
  const uint lo = row_live ? ink_slide_lo(a.window, pos) : 0;
  const uint row_first = a.window ? lo / a.chunk : a.c0;
  const bool chunk_live = row_live && c >= row_first && c <= pos / a.chunk;
  const uint from = chunk_live ? max(lo, c * a.chunk) : 1;
  const uint to = chunk_live ? min(pos + 1, (c + 1) * a.chunk) : 0;
  const size_t head = s.off + size_t(kvh) * a.ring * D;
  const size_t prow = size_t(row) * a.heads + kvh * SLIDE_G + g;
  const float4 qv = chunk_live ? float4(*(const device bfloat4*)(q + prow * D + lane * 4)) : float4(0);
  const bool has_rel = chunk_live && from < to && (pos - (to - 1)) < EXTENT;
  const float r_lane = (has_rel && lane < REL) ? float(rel[prow * REL + lane]) : 0.0f;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float top = -INFINITY, total = 0.0f;
  const uint kt_start = c * a.chunk + ((tg_from - c * a.chunk) / SLIDE_TILE_K) * SLIDE_TILE_K;
  for (uint kt = kt_start; kt < tg_to; kt += SLIDE_TILE_K) {
    for (uint iter = 0; iter < SLIDE_TILE_K / 8; ++iter) {
      const uint key = sg + iter * 8, kp_load = kt + key;
      if (kp_load >= tg_from && kp_load < tg_to) {
        const size_t at = head + size_t(kp_load % a.ring) * D;
        ks[key * (D / 4) + lane] = *(const device bfloat4*)(kc + at + lane * 4);
        vs[key * (D / 4) + lane] = *(const device bfloat4*)(vc + at + lane * 4);
      } else {
        ks[key * (D / 4) + lane] = bfloat4(0);
        vs[key * (D / 4) + lane] = bfloat4(0);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint kp_start = chunk_live ? max(kt, from) : 1;
    const uint kp_end = chunk_live ? min(kt + SLIDE_TILE_K, to) : 0;
    if (kp_start < kp_end) {
      const bool tile_bias = has_rel && (pos - (kp_end - 1)) < EXTENT;
      float bias_lane = 0.0f;
      if (tile_bias) {
        const uint kp_lane = kt + lane;
        const bool need_bias = kp_lane >= kp_start && kp_lane < kp_end && (pos - kp_lane) < EXTENT;
        const uint dist_safe = need_bias ? (pos - kp_lane) : 0;
        float acc_bias = 0.0f;
        for (uint j = 0; j < REL; ++j) {
          const float r_j = simd_shuffle(r_lane, ushort(j));
          const float pv = need_bias ? float(proj[j * EXTENT + dist_safe]) : 0.0f;
          acc_bias += r_j * pv;
        }
        bias_lane = need_bias ? float(bfloat(acc_bias)) : 0.0f;
      }
      uint kp = kp_start;
      for (; kp + SLIDE_K_STEP <= kp_end; kp += SLIDE_K_STEP) {
        const uint k0 = kp - kt;
        float ds[SLIDE_K_STEP];
        for (uint k = 0; k < SLIDE_K_STEP; ++k) ds[k] = dot(qv, float4(ks[(k0 + k) * (D / 4) + lane]));
        for (uint k = 0; k < SLIDE_K_STEP; ++k) ds[k] = ink_warp_sum(ds[k]);
        for (uint k = 0; k < SLIDE_K_STEP; ++k) {
          const float bias = tile_bias ? simd_shuffle(bias_lane, ushort(k0 + k)) : 0.0f;
          const float4 vv = float4(vs[(k0 + k) * (D / 4) + lane]);
          ink_slide_step(ds[k], bias, a.scale, vv, kp + k, from, top, total, acc);
        }
      }
      for (; kp < kp_end; ++kp) {
        const uint k_local = kp - kt;
        const float ds = ink_warp_sum(dot(qv, float4(ks[k_local * (D / 4) + lane])));
        const float bias = tile_bias ? simd_shuffle(bias_lane, ushort(k_local)) : 0.0f;
        const float4 vv = float4(vs[k_local * (D / 4) + lane]);
        ink_slide_step(ds, bias, a.scale, vv, kp, from, top, total, acc);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!chunk_live) return;
  const size_t slot = prow * a.chunks + size_t(c - row_first);
  if (lane == 0) part_ml[slot] = float2(top, total);
  for (uint i = 0; i < 4; ++i) part_o[slot * D + lane * 4 + i] = acc[i];
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
