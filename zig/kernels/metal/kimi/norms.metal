// Row kernels around the layers: attention residuals fused with the residual add and Kimi's RMS norm, router, argmax.
#include "kimi_common.h"

constant constexpr uint TG = 256;
constant constexpr uint MAXE = 10;

// Threadgroup sum of one value a thread, in a fixed order: simd_sum, then the simdgroups' sums in index order.
inline float k3_tg_sum(float v, threadgroup float* part, uint sg, uint lane) {
  v = simd_sum(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float s = 0.0f;
  for (uint i = 0; i < TG / 32; ++i) s += part[i];
  return s;
}

struct ResArgs {
  uint D, rows, blocks, flags;
  float eps;
};

constant constexpr uint RES_DELTA = 1, RES_PREFIX = 2, RES_APPEND = 4;

// prefix = bf16(prefix + delta) (or delta), attention residual over [blocks, prefix], then KimiRMSNorm into out.
kernel void k3_res_norm(device bfloat* P [[buffer(0)]], device const bfloat* delta [[buffer(1)]],
                        device bfloat* B [[buffer(2)]], device const bfloat* w_norm [[buffer(3)]],
                        device const bfloat* w_proj [[buffer(4)]], device const bfloat* w_out [[buffer(5)]],
                        device bfloat* out [[buffer(6)]], constant ResArgs& a [[buffer(7)]],
                        uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[TG / 32];
  const uint D = a.D, nb = a.blocks;
  device bfloat* p = P + row * D;
  device const bfloat* d = delta + row * D;
  const uint plane = a.rows * D;
  float sq[MAXE], dot[MAXE];
  for (uint e = 0; e <= nb; ++e) sq[e] = dot[e] = 0.0f;
  for (uint i = t; i < D; i += TG) {
    float v = (a.flags & RES_PREFIX) ? float(p[i]) : 0.0f;
    if (a.flags & RES_DELTA) v = (a.flags & RES_PREFIX) ? float(bfloat(v + float(d[i]))) : float(d[i]);
    p[i] = bfloat(v);
    if (nb == 0) continue;
    const float sw = float(w_norm[i]) * float(w_proj[i]);
    for (uint e = 0; e < nb; ++e) {
      const float b = float(B[e * plane + row * D + i]);
      sq[e] = fma(b, b, sq[e]);
      dot[e] = fma(b, sw, dot[e]);
    }
    sq[nb] = fma(v, v, sq[nb]);
    dot[nb] = fma(v, sw, dot[nb]);
  }
  float prob[MAXE];
  if (nb > 0) {
    float m = -INFINITY;
    for (uint e = 0; e <= nb; ++e) {
      const float var = k3_tg_sum(sq[e], part, sg, lane) / float(D);
      prob[e] = k3_tg_sum(dot[e], part, sg, lane) * precise::rsqrt(var + a.eps);
      m = max(m, prob[e]);
    }
    float z = 0.0f;
    for (uint e = 0; e <= nb; ++e) {
      prob[e] = precise::exp(prob[e] - m);
      z += prob[e];
    }
    for (uint e = 0; e <= nb; ++e) prob[e] /= z;
  }
  float hs[(7168 + TG - 1) / TG];
  float ss = 0.0f;
  uint n = 0;
  for (uint i = t; i < D; i += TG, ++n) {
    float h = float(p[i]);
    if (nb > 0) {
      float acc = 0.0f;
      for (uint e = 0; e < nb; ++e) acc = fma(prob[e], float(B[e * plane + row * D + i]), acc);
      h = float(bfloat(fma(prob[nb], h, acc)));
    }
    if (a.flags & RES_APPEND) B[nb * plane + row * D + i] = p[i];
    hs[n] = h;
    ss = fma(h, h, ss);
  }
  const float r = precise::rsqrt(k3_tg_sum(ss, part, sg, lane) / float(D) + a.eps);
  n = 0;
  for (uint i = t; i < D; i += TG, ++n) out[row * D + i] = bfloat(float(w_out[i]) * float(bfloat(hs[n] * r)));
}

struct NormArgs {
  uint D, x_stride, y_stride;
  float eps;
};

// KimiRMSNorm on a row: bf16(w * bf16(x * rsqrt(mean(x^2) + eps))).
kernel void k3_rms_norm(device const bfloat* X [[buffer(0)]], device const bfloat* w [[buffer(1)]],
                        device bfloat* Y [[buffer(2)]], constant NormArgs& a [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[TG / 32];
  device const bfloat* x = X + row * a.x_stride;
  float ss = 0.0f;
  for (uint i = t; i < a.D; i += TG) ss = fma(float(x[i]), float(x[i]), ss);
  const float r = precise::rsqrt(k3_tg_sum(ss, part, sg, lane) / float(a.D) + a.eps);
  for (uint i = t; i < a.D; i += TG) Y[row * a.y_stride + i] = bfloat(float(w[i]) * float(bfloat(float(x[i]) * r)));
}

// out = bf16(a + b), four values a thread.
kernel void k3_add2(device const uint2* x [[buffer(0)]], device const uint2* y [[buffer(1)]],
                    device uint2* out [[buffer(2)]], constant uint& n4 [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= n4) return;
  const float4 s = k3_bf16x4(x[i]) + k3_bf16x4(y[i]);
  out[i] = uint2(as_type<uint>(bfloat2(s.xy)), as_type<uint>(bfloat2(s.zw)));
}

// Embedding rows by token id (bf16 copy, 8 bytes a thread).
kernel void k3_embed(device const uint* ids [[buffer(0)]], device const uint2* table [[buffer(1)]],
                     device uint2* out [[buffer(2)]], constant uint& D [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]]) {
  const uint q = D / 4;
  if (gid.x < q) out[gid.y * q + gid.x] = table[ids[gid.y] * q + gid.x];
}

struct RouteArgs {
  uint experts, topk;
};

// Sigmoid router: top-k of score + bias (ties to the lower id) in rank order, weights renormalised in rank order.
kernel void k3_router(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]],
                      device uint* ids [[buffer(2)]], device float* weights [[buffer(3)]],
                      constant RouteArgs& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                      uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint nt [[threads_per_threadgroup]]) {
  threadgroup float s_val[1024];
  threadgroup float best_v[32];
  threadgroup uint best_i[32];
  threadgroup uint chosen[32];
  for (uint e = t; e < a.experts; e += nt) s_val[e] = k3_sigmoid(logits[row * a.experts + e]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float c = t < a.experts ? s_val[t] + bias[t] : -INFINITY;
  for (uint k = 0; k < a.topk; ++k) {
    float v = c;
    uint i = t;
    for (ushort off = 16; off >= 1; off >>= 1) {
      const float ov = simd_shuffle_down(v, off);
      const uint oi = simd_shuffle_down(i, off);
      if (ov > v || (ov == v && oi < i)) v = ov, i = oi;
    }
    if (lane == 0) best_v[sg] = v, best_i[sg] = i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      float bv = best_v[0];
      uint bi = best_i[0];
      for (uint g = 1; g < (nt + 31) / 32; ++g)
        if (best_v[g] > bv || (best_v[g] == bv && best_i[g] < bi)) bv = best_v[g], bi = best_i[g];
      chosen[k] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == chosen[k]) c = -INFINITY;
  }
  if (t == 0) {
    float z = 0.0f;
    for (uint k = 0; k < a.topk; ++k) z += s_val[chosen[k]];
    z += 1e-20f;
    for (uint k = 0; k < a.topk; ++k) {
      ids[row * a.topk + k] = chosen[k];
      weights[row * a.topk + k] = s_val[chosen[k]] / z;
    }
  }
}

// Greedy token: the largest bf16 logit of a row, ties to the lower id.
kernel void k3_argmax(device const bfloat* logits [[buffer(0)]], device uint* out [[buffer(1)]],
                      constant uint& V [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                      uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float bv[TG / 32];
  threadgroup uint bi[TG / 32];
  float v = -INFINITY;
  uint i = 0;
  for (uint k = t; k < V; k += TG) {
    const float x = float(logits[row * V + k]);
    if (x > v) v = x, i = k;
  }
  for (ushort off = 16; off >= 1; off >>= 1) {
    const float ov = simd_shuffle_down(v, off);
    const uint oi = simd_shuffle_down(i, off);
    if (ov > v || (ov == v && oi < i)) v = ov, i = oi;
  }
  if (lane == 0) bv[sg] = v, bi[sg] = i;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    for (uint g = 1; g < TG / 32; ++g)
      if (bv[g] > v || (bv[g] == v && bi[g] < i)) v = bv[g], i = bi[g];
    out[row] = i;
  }
}
