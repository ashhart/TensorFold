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

inline float ink_warp_sum(float v) {
  for (uint o = 16; o; o >>= 1) v += simd_shuffle_xor(v, ushort(o));
  return v;
}
inline float ink_clip_factor(float ss, float threshold) {
  return fmin(threshold / (precise::sqrt(ss) + 1e-6f), 1.0f);
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
  device bfloat* dst = (key ? kc : vc) + s.off + (size_t(h) * a.ring + pos % a.ring) * D;
  for (uint i = 0; i < 4; ++i) dst[lane * 4 + i] = bfloat(x[i] * scale);
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
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float top = -INFINITY, total = 0.0f;
  for (uint kp = from; kp < to; ++kp) {
    const size_t at = head + size_t(kp % a.ring) * D;
    float4 qv = float4(*(const device bfloat4*)(q + prow * D + lane * 4));
    float4 kv = float4(*(const device bfloat4*)(kc + at + lane * 4));
    const float dot_score = ink_warp_sum(dot(qv, kv));
    const uint distance = pos - kp;
    const float bias = distance < EXTENT ? float(tab[prow * EXTENT + distance]) : 0.0f;
    const float score = (dot_score * a.scale + bias) * LOG2E;
    const float next = max(top, score), a0 = precise::exp2(top - next), b0 = precise::exp2(score - next);
    const bfloat p = bfloat(b0);
    total = total * a0 + float(p);
    for (uint i = 0; i < 4; ++i) acc[i] = acc[i] * a0 + float(p) * float(vc[at + lane * 4 + i]);
    top = next;
  }
  const size_t slot = size_t(prow) * a.chunks + tg.y;
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
