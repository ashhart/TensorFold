// music3 AR attention: the body's q / k norm and rope, split-key attention and its merge, the depth decoder's.
#include "m3_common.h"

struct RopeArgs {
  uint t, ctx;
  float eps;
};

// Body q / k: a head's RMS norm, then rotate-half rope; q to q_out, k and v to the caches. grid (rows, 48).
[[max_total_threads_per_threadgroup(128)]] kernel void m3_qk_rope(device const bf16* qkv [[buffer(0)]], device const bf16* qn [[buffer(1)]],
                       device const bf16* kn [[buffer(2)]], device const float* inv_freq [[buffer(3)]],
                       device const int* pos [[buffer(4)]], device bf16* q_out [[buffer(5)]], device bf16* kc [[buffer(6)]],
                       device bf16* vc [[buffer(7)]], constant RopeArgs& a [[buffer(8)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sh[32];
  threadgroup float v_sh[128];
  const uint r = tg.x, h = tg.y;
  const uint b = r / a.t, p = uint(*pos) + r % a.t;
  device const bf16* row = qkv + (ulong)r * 6144;
  if (h >= 40) {  // v
    const uint g = h - 40;
    vc[(((ulong)b * 8 + g) * a.ctx + p) * 128 + d] = row[4096 + 1024 + g * 128 + d];
    return;
  }
  const bool isq = h < 32;
  device const bf16* src = isq ? row + h * 128 : row + 4096 + (h - 32) * 128;
  const float x = b2f(src[d]);
  const float ss = group_sum(x * x, sh, sg, lane, 4);
  const float rs = 1.0f / precise::sqrt(ss / 128.0f + a.eps);
  const float nx = rb(x * rs * b2f((isq ? qn : kn)[d]));
  v_sh[d] = nx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float ang = float(p) * inv_freq[d & 63];
  const float c = precise::cos(ang), s = precise::sin(ang);
  const float other = d < 64 ? -v_sh[d + 64] : v_sh[d - 64];
  const bf16 o = f2b(nx * c + other * s);
  if (isq)
    q_out[((ulong)r * 32 + h) * 128 + d] = o;
  else
    kc[(((ulong)b * 8 + (h - 32)) * a.ctx + p) * 128 + d] = o;
}

struct AttnArgs {
  uint t, ctx, splits;
  float scale;
};

// Split-key attention: a group a (kv head, row, split), simdgroups the 4 q heads; partials ml / acc a split.
[[max_total_threads_per_threadgroup(128)]] kernel void m3_attn_part(device const bf16* q [[buffer(0)]], device const bf16* kc [[buffer(1)]],
                         device const bf16* vc [[buffer(2)]], device const int* pos [[buffer(3)]],
                         device float* ml [[buffer(4)]], device float* acc [[buffer(5)]],
                         constant AttnArgs& a [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                         uint w [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint g = tg.x, r = tg.y, sp = tg.z;
  const uint h = g * 4 + w, b = r / a.t;
  const uint L = uint(*pos) + r % a.t + 1;
  const uint chunk = (L + a.splits - 1) / a.splits;
  const uint k0 = sp * chunk, k1 = min(L, k0 + chunk);
  const float4 qv = bf4(*(device const uint2*)(q + ((ulong)r * 32 + h) * 128 + lane * 4)) * a.scale;
  device const bf16* kb = kc + ((ulong)b * 8 + g) * a.ctx * 128 + lane * 4;
  device const bf16* vb = vc + ((ulong)b * 8 + g) * a.ctx * 128 + lane * 4;
  float m = -INFINITY, l = 0.0f;
  float4 o = 0.0f;
  uint j = k0;
  for (; j + 4 <= k1; j += 4) {  // four keys a step: one reduction of their four partial dots
    float4 kd;
    for (int e = 0; e < 4; ++e) kd[e] = dot(qv, bf4(*(device const uint2*)(kb + (ulong)(j + e) * 128)));
    const float4 s = simd_sum(kd);
    const float mn = max(m, max(max(s.x, s.y), max(s.z, s.w)));
    const float cor = precise::exp(m - mn);
    const float4 p = precise::exp(s - mn);
    l = l * cor + ((p.x + p.y) + (p.z + p.w));
    o = o * cor;
    for (int e = 0; e < 4; ++e) o += p[e] * bf4(*(device const uint2*)(vb + (ulong)(j + e) * 128));
    m = mn;
  }
  for (; j < k1; ++j) {
    const float s = simd_sum(dot(qv, bf4(*(device const uint2*)(kb + (ulong)j * 128))));
    const float4 vv = bf4(*(device const uint2*)(vb + (ulong)j * 128));
    const float mn = max(m, s);
    const float cor = precise::exp(m - mn), e = precise::exp(s - mn);
    l = l * cor + e;
    o = o * cor + e * vv;
    m = mn;
  }
  const ulong at = ((ulong)r * 32 + h) * a.splits + sp;
  if (lane == 0) {
    ml[at * 2] = m;
    ml[at * 2 + 1] = l;
  }
  *(device float4*)(acc + at * 128 + lane * 4) = o;
}

// The S partials of a (row, head) into out [rows, 32 * 128] bf16. grid (rows, 32), 128 threads.
[[max_total_threads_per_threadgroup(128)]] kernel void m3_attn_merge(device const float* ml [[buffer(0)]], device const float* acc [[buffer(1)]],
                          device bf16* out [[buffer(2)]], constant uint& S [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
  const ulong base = ((ulong)tg.x * 32 + tg.y) * S;
  float m = -INFINITY;
  for (uint s = 0; s < S; ++s) m = max(m, ml[(base + s) * 2]);
  float l = 0.0f, o = 0.0f;
  for (uint s = 0; s < S; ++s) {
    const float ms = ml[(base + s) * 2];
    if (ms == -INFINITY) continue;
    const float c = precise::exp(ms - m);
    l += ml[(base + s) * 2 + 1] * c;
    o += acc[(base + s) * 128 + d] * c;
  }
  out[((ulong)tg.x * 32 + tg.y) * 128 + d] = f2b(o / l);
}

struct DepthArgs {
  uint p0, B;
  float scale;
};

// Depth rows' k and v into the caches [b][16][8][256]: pass row r is (position p0 + r / B, batch r % B).
[[max_total_threads_per_threadgroup(256)]] kernel void m3_depth_kv(device const bf16* qkv [[buffer(0)]], device bf16* kc [[buffer(1)]],
                        device bf16* vc [[buffer(2)]], constant DepthArgs& a [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
  const uint r = tg.x, h = tg.y;
  const uint p = a.p0 + r / a.B, b = r % a.B;
  device const bf16* row = qkv + (ulong)r * 12288;
  const ulong o = (((ulong)b * 16 + h) * 8 + p) * 256 + d;
  kc[o] = row[4096 + h * 256 + d];
  vc[o] = row[8192 + h * 256 + d];
}

// Causal attention of the depth rows over positions <= p (at most 8), out [rows, 4096]. grid (rows, 16), 256 threads.
[[max_total_threads_per_threadgroup(256)]] kernel void m3_depth_attn(device const bf16* qkv [[buffer(0)]], device const bf16* kc [[buffer(1)]],
                          device const bf16* vc [[buffer(2)]], device bf16* out [[buffer(3)]],
                          constant DepthArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint d [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sh[32];
  float pr[8];
  const uint r = tg.x, h = tg.y;
  const uint p = a.p0 + r / a.B, b = r % a.B;
  const float qd = b2f(qkv[(ulong)r * 12288 + h * 256 + d]);
  device const bf16* kb = kc + ((ulong)b * 16 + h) * 8 * 256;
  device const bf16* vb = vc + ((ulong)b * 16 + h) * 8 * 256;
  float m = -INFINITY;
  for (uint j = 0; j <= p; ++j) {
    pr[j] = group_sum(qd * b2f(kb[j * 256 + d]), sh, sg, lane, 8) * a.scale;
    m = max(m, pr[j]);
  }
  float l = 0.0f, o = 0.0f;
  for (uint j = 0; j <= p; ++j) {
    const float e = precise::exp(pr[j] - m);
    l += e;
    o = fma(e, b2f(vb[j * 256 + d]), o);
  }
  out[(ulong)r * 4096 + h * 256 + d] = f2b(o / l);
}
