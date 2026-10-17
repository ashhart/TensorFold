// Drama 3 gated full attention (backbone and tower): 16 query heads, 2 KV heads, head dim 256, rotary on 64 dims.
#include "d3_common.h"

struct RopeArgs {
  uint ps, q_off, q_head_stride, k_off, v_off, interleaved;
  float theta, eps;
};

// q / k zero-centred RMSNorm, rotary (NeoX or interleaved); q to qout, k / v to cache row pos; grid (rows, 16 + 2).
[[max_total_threads_per_threadgroup(256)]] kernel void d3_qk_rope(device const bf16* proj [[buffer(0)]], device const bf16* qn_w [[buffer(1)]],
                       device const bf16* kn_w [[buffer(2)]], device const int* pos0 [[buffer(3)]],
                       device bf16* qout [[buffer(4)]], device bf16* kc [[buffer(5)]], device bf16* vc [[buffer(6)]],
                       constant RopeArgs& a [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint i [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  threadgroup float xn[256];
  const uint t = tg.x, hh = tg.y;
  const int pos = pos0[0] + int(t);
  device const bf16* row = proj + (ulong)t * a.ps;
  const bool isq = hh < 16;
  device const bf16* src = isq ? row + a.q_off + hh * a.q_head_stride : row + a.k_off + (hh - 16) * 256;
  const float x = b2f(src[i]);
  const float inv = rsqrt(group_sum(x * x, scratch, sg, lane, 8) / 256.0f + a.eps);
  const float wv = b2f((isq ? qn_w : kn_w)[i]);
  xn[i] = rb(x * inv * (1.0f + wv));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float outv = xn[i];
  if (i < 64) {
    uint j, partner;
    float sign;
    if (a.interleaved) {
      j = i >> 1;
      partner = i ^ 1;
      sign = (i & 1) ? 1.0f : -1.0f;
    } else {
      j = i & 31;
      partner = i < 32 ? i + 32 : i - 32;
      sign = i < 32 ? -1.0f : 1.0f;
    }
    const float inv_freq = 1.0f / precise::pow(a.theta, float(2 * j) / 64.0f);
    const float ang = float(pos) * inv_freq;
    outv = xn[i] * precise::cos(ang) + sign * xn[partner] * precise::sin(ang);
  }
  if (isq) {
    qout[((ulong)t * 16 + hh) * 256 + i] = f2b(outv);
  } else {
    const uint g = hh - 16;
    kc[((ulong)pos * 2 + g) * 256 + i] = f2b(outv);
    vc[((ulong)pos * 2 + g) * 256 + i] = row[a.v_off + g * 256 + i];
  }
}

struct AttnArgs {
  uint ps, gate_off, gate_head_stride;
};

// Row t over cache rows 0..pos0 + t, 8 simdgroups of keys, fp32 online softmax; out = bf16(a) * bf16(sigmoid(gate)).
[[max_total_threads_per_threadgroup(256)]] kernel void d3_attn(device const bf16* q [[buffer(0)]], device const bf16* kc [[buffer(1)]],
                    device const bf16* vc [[buffer(2)]], device const int* pos0 [[buffer(3)]],
                    device const bf16* proj [[buffer(4)]], device bf16* out [[buffer(5)]],
                    constant AttnArgs& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                    uint i [[thread_index_in_threadgroup]], uint w [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  const uint t = tg.x, h = tg.y, g = h >> 3;
  const int L = pos0[0] + int(t) + 1;
  const uint2 qa = ((device const uint2*)(q + ((ulong)t * 16 + h) * 256))[lane * 2];
  const uint2 qb = ((device const uint2*)(q + ((ulong)t * 16 + h) * 256))[lane * 2 + 1];
  const float4 q0 = bf4(qa), q1 = bf4(qb);
  float m = -INFINITY, l = 0.0f, acc[8];
  for (int j = 0; j < 8; ++j) acc[j] = 0.0f;
  for (int p = int(w); p < L; p += 8) {
    device const uint2* kr = (device const uint2*)(kc + ((ulong)p * 2 + g) * 256);
    device const uint2* vr = (device const uint2*)(vc + ((ulong)p * 2 + g) * 256);
    const float4 k0 = bf4(kr[lane * 2]), k1 = bf4(kr[lane * 2 + 1]);
    float d = q0.x * k0.x;
    d = fma(q0.y, k0.y, d);
    d = fma(q0.z, k0.z, d);
    d = fma(q0.w, k0.w, d);
    d = fma(q1.x, k1.x, d);
    d = fma(q1.y, k1.y, d);
    d = fma(q1.z, k1.z, d);
    d = fma(q1.w, k1.w, d);
    const float sc = simd_sum(d) * 0.0625f;
    const float4 v0 = bf4(vr[lane * 2]), v1 = bf4(vr[lane * 2 + 1]);
    const float nm = max(m, sc);
    const float corr = precise::exp(m - nm), e = precise::exp(sc - nm);
    l = l * corr + e;
    const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    for (int j = 0; j < 8; ++j) acc[j] = acc[j] * corr + e * vv[j];
    m = nm;
  }
  threadgroup float sm[8], sl[8], sacc[8][256];
  if (lane == 0) {
    sm[w] = m;
    sl[w] = l;
  }
  for (int j = 0; j < 8; ++j) sacc[w][lane * 8 + j] = acc[j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float M = -INFINITY;
  for (int k = 0; k < 8; ++k) M = max(M, sm[k]);
  float Ls = 0.0f, A = 0.0f;
  for (int k = 0; k < 8; ++k) {
    if (sm[k] == -INFINITY) continue;
    const float c = precise::exp(sm[k] - M);
    Ls += sl[k] * c;
    A += sacc[k][i] * c;
  }
  const float av = rb(A / Ls);
  const float gt = b2f(proj[(ulong)t * a.ps + a.gate_off + h * a.gate_head_stride + i]);
  out[((ulong)t * 16 + h) * 256 + i] = f2b(av * rb(d3_sigmoid(gt)));
}
