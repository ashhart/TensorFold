// GLM-5.3's glue for lane rows in fp32: embeddings, residual adds with the next RMSNorm, SiLU times up.
#include <metal_stdlib>
using namespace metal;

struct GlueArgs { uint width, x_stride, y_stride, parts, part_stride, pad0, pad1, pad2; float eps, pad3, pad4, pad5; };
inline float glm_glue_lanes(float v) { for (ushort d = 16; d > 0; d >>= 1) v += simd_shuffle_xor(v, d); return v; }

// Row r of the hidden state from token ids[r]: E4M3 codes times the row's fp32 scale (the reference's decode).
kernel void glm_embed(const device uint* ids [[buffer(0)]], const device uchar* fp8 [[buffer(1)]],
                      const device float* scale [[buffer(2)]], constant GlueArgs& a [[buffer(3)]],
                      device float* h [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                      uint t [[thread_index_in_threadgroup]]) {
  const uint id = ids[r];
  const float s = scale[id];
  for (uint i = t; i < a.width; i += 256) {
    const uint code = fp8[ulong(id) * a.width + i];
    const float v = float(as_type<half>(ushort((code & 0x7Fu) << 7))) * 256.0f;
    h[ulong(r) * a.y_stride + i] = ((code & 0x80u) ? -v : v) * s;
  }
}

// Row r of the hidden state from token ids[r]: vectorized BF16 embedding table (uint2 -> float4).
kernel void glm_embed_bf16(const device uint* ids [[buffer(0)]], const device bfloat* table [[buffer(1)]],
                           constant GlueArgs& a [[buffer(2)]], device float* h [[buffer(3)]],
                           uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                           uint nth [[threads_per_threadgroup]]) {
  const uint id = ids[r];
  const device bfloat* row = table + ulong(id) * a.width;
  if (((a.width | a.y_stride) & 3u) == 0u) {
    const uint n4 = a.width >> 2;
    const device uint2* row2 = reinterpret_cast<const device uint2*>(row);
    device float4* dst4 = reinterpret_cast<device float4*>(h + ulong(r) * a.y_stride);
    for (uint i = t; i < n4; i += nth) {
      const uint2 q = row2[i];
      dst4[i] = float4(as_type<float>(q.x << 16), as_type<float>(q.x & 0xFFFF0000u),
                       as_type<float>(q.y << 16), as_type<float>(q.y & 0xFFFF0000u));
    }
    return;
  }
  for (uint i = t; i < a.width; i += nth) h[ulong(r) * a.y_stride + i] = float(row[i]);
}

// h[r] += the parts' sum (in part order), then y[r] = w * (h[r] / rms(h[r])); vectorized float4 + 1024t fast path!
kernel void glm_add_rms(device float* h [[buffer(0)]], const device float* add [[buffer(1)]],
                        const device bfloat* w [[buffer(2)]], constant GlueArgs& a [[buffer(3)]],
                        device float* y [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                        uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]], uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const uint nsgs = nth >> 5;
  if (((a.width | a.x_stride | a.y_stride | a.part_stride) & 3u) == 0u) {
    const uint n4 = a.width >> 2, xs4 = a.x_stride >> 2, ys4 = a.y_stride >> 2, ps4 = a.part_stride >> 2;
    device float4* x4 = reinterpret_cast<device float4*>(h) + ulong(r) * xs4;
    const device float4* a4 = reinterpret_cast<const device float4*>(add);
    const device uint2* w2 = reinterpret_cast<const device uint2*>(w);
    device float4* y4 = reinterpret_cast<device float4*>(y) + ulong(r) * ys4;
    float4 hv[2];
    float s = 0.0f;
    uint k = 0u;
    for (uint i = t; i < n4; i += nth) {
      float4 v = x4[i];
      if (a.parts > 0u) {
        float4 sum = a4[ulong(r) * xs4 + i];
        for (uint p = 1u; p < a.parts; ++p) sum += a4[ulong(p) * ps4 + ulong(r) * xs4 + i];
        v += sum;
        x4[i] = v;
      }
      if (k < 2u) hv[k] = v;
      ++k;
      s += dot(v, v);
    }
    s = glm_glue_lanes(s);
    if (lane == 0u) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0.0f;
    for (uint g = 0u; g < nsgs; ++g) var += part[g];
    const float rs = precise::rsqrt(var / float(a.width) + a.eps);
    k = 0u;
    for (uint i = t; i < n4; i += nth) {
      const uint2 wk = w2[i];
      const float4 wf = float4(as_type<float>(wk.x << 16), as_type<float>(wk.x & 0xFFFF0000u),
                               as_type<float>(wk.y << 16), as_type<float>(wk.y & 0xFFFF0000u));
      const float4 v = (k < 2u) ? hv[k] : x4[i];
      ++k;
      const float4 yv = wf * (v * rs);
      y4[i] = yv;
      if (a.pad0 != 0u) reinterpret_cast<device half4*>(y + ulong(a.pad0) * 4ul)[ulong(r) * ys4 + i] = half4(yv);
    }
    return;
  }
  device float* x = h + ulong(r) * a.x_stride;
  float s = 0.0f;
  for (uint i = t; i < a.width; i += nth) {
    float v = x[i];
    if (a.parts > 0u) {
      float sum = add[ulong(r) * a.x_stride + i];
      for (uint p = 1u; p < a.parts; ++p) sum += add[ulong(p) * a.part_stride + ulong(r) * a.x_stride + i];
      v += sum;
      x[i] = v;
    }
    s = fma(v, v, s);
  }
  s = glm_glue_lanes(s);
  if (lane == 0u) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (uint g = 0u; g < nsgs; ++g) var += part[g];
  const float rs = precise::rsqrt(var / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += nth) y[ulong(r) * a.y_stride + i] = float(w[i]) * (x[i] * rs);
}

// act[p][c] = silu(g) * u for gate at column c and up at column width + c of row p (x_stride apart).
kernel void glm_silu_mul(const device float* gu [[buffer(0)]], constant GlueArgs& a [[buffer(1)]],
                         device float* act [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]]) {
  const uint p = tg.y, c = tg.x * 256 + t;
  if (c >= a.width) return;
  const float g = gu[ulong(p) * a.x_stride + c], u = gu[ulong(p) * a.x_stride + a.width + c];
  const float v = precise::divide(g, 1.0f + precise::exp(-g)) * u;
  act[ulong(p) * a.y_stride + c] = v;
  if (a.pad0 != 0u) reinterpret_cast<device half*>(act + a.pad0)[ulong(p) * a.y_stride + c] = half(v);
}

// Row r of dst = row idx[r] of src, `a.x` 32-bit words a row (vectorized uint4 fast path!).
kernel void glm_gather(const device uint* src [[buffer(0)]], const device uint* idx [[buffer(1)]],
                       constant uint4& a [[buffer(2)]], device uint* dst [[buffer(3)]],
                       uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                       uint nth [[threads_per_threadgroup]]) {
  const device uint* s = src + ulong(idx[r]) * a.y;
  device uint* d = dst + ulong(r) * a.z;
  if (((a.x | a.y | a.z) & 3u) == 0u) {
    const uint n4 = a.x >> 2;
    const device uint4* s4 = reinterpret_cast<const device uint4*>(s);
    device uint4* d4 = reinterpret_cast<device uint4*>(d);
    for (uint i = t; i < n4; i += nth) d4[i] = s4[i];
    return;
  }
  for (uint i = t; i < a.x; i += nth) d[i] = s[i];
}

inline float4 glm_lm_bf16x4(uint a, uint b) {
  return float4(as_type<float>(a << 16), as_type<float>(a & 0xFFFF0000u),
                as_type<float>(b << 16), as_type<float>(b & 0xFFFF0000u));
}

// Fused exact BF16 lm_head + per-TG argmax for r=1 (Q=4 128t, 16 rows/TG, dims={V,K,lo,rows}).
kernel void glm_lm_bf16_argmax_1(const device float4* X4 [[buffer(0)]], const device uint4* W4 [[buffer(1)]],
                                 constant uint4& dims [[buffer(2)]], device uint* part [[buffer(3)]],
                                 uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float bv[4];
  threadgroup uint bi[4];
  const uint V = dims.x, K = dims.y, lo = dims.z, o0 = g * 16u + sg * 4u;
  float best = -INFINITY;
  uint bid = 0xFFFFFFFFu;
  if (o0 < V) {
    const uint row_u4 = K >> 3, c256 = K >> 8;
    const bool f4 = (o0 + 3u) < V;
    const device uint4* W0 = W4 + ulong(o0) * row_u4;
    const device uint4* W1 = W0 + row_u4, *W2 = W1 + row_u4, *W3 = W2 + row_u4;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    if (f4) {
      for (uint c = 0u; c < c256; ++c) {
        const uint xo = c * 64u + lane * 2u;
        const float4 x0 = X4[xo], x1 = X4[xo + 1u];
        const uint4 q0 = W0[c * 32u + lane], q1 = W1[c * 32u + lane], q2 = W2[c * 32u + lane], q3 = W3[c * 32u + lane];
        s0 += dot(glm_lm_bf16x4(q0.x, q0.y), x0) + dot(glm_lm_bf16x4(q0.z, q0.w), x1);
        s1 += dot(glm_lm_bf16x4(q1.x, q1.y), x0) + dot(glm_lm_bf16x4(q1.z, q1.w), x1);
        s2 += dot(glm_lm_bf16x4(q2.x, q2.y), x0) + dot(glm_lm_bf16x4(q2.z, q2.w), x1);
        s3 += dot(glm_lm_bf16x4(q3.x, q3.y), x0) + dot(glm_lm_bf16x4(q3.z, q3.w), x1);
      }
    } else {
      for (uint c = 0u; c < c256; ++c) {
        const uint xo = c * 64u + lane * 2u;
        const float4 x0 = X4[xo], x1 = X4[xo + 1u];
        const uint4 q0 = W0[c * 32u + lane];
        s0 += dot(glm_lm_bf16x4(q0.x, q0.y), x0) + dot(glm_lm_bf16x4(q0.z, q0.w), x1);
        if (o0 + 1u < V) { const uint4 q1 = W1[c * 32u + lane]; s1 += dot(glm_lm_bf16x4(q1.x, q1.y), x0) + dot(glm_lm_bf16x4(q1.z, q1.w), x1); }
        if (o0 + 2u < V) { const uint4 q2 = W2[c * 32u + lane]; s2 += dot(glm_lm_bf16x4(q2.x, q2.y), x0) + dot(glm_lm_bf16x4(q2.z, q2.w), x1); }
      }
    }
    const float v0 = glm_glue_lanes(s0), v1 = (o0 + 1u < V) ? glm_glue_lanes(s1) : -INFINITY;
    const float v2 = (o0 + 2u < V) ? glm_glue_lanes(s2) : -INFINITY, v3 = f4 ? glm_glue_lanes(s3) : -INFINITY;
    best = v0; bid = lo + o0;
    if (v1 > best) { best = v1; bid = lo + o0 + 1u; }
    if (v2 > best) { best = v2; bid = lo + o0 + 2u; }
    if (v3 > best) { best = v3; bid = lo + o0 + 3u; }
  }
  if (lane == 0u) { bv[sg] = best; bi[sg] = bid; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && lane == 0u) {
    float tb = bv[0]; uint ti = bi[0];
    for (uint k = 1u; k < 4u; ++k) if (bv[k] > tb || (bv[k] == tb && bi[k] < ti)) { tb = bv[k]; ti = bi[k]; }
    part[2u * g] = as_type<uint>(tb);
    part[2u * g + 1u] = ti;
  }
}

// Fused exact BF16 lm_head + per-TG argmax for r=2..16 (Q=2 256t, 16 rows/TG, part[row,g]=(val,id)).
kernel void glm_lm_bf16_argmax_n(const device float4* X4 [[buffer(0)]], const device uint4* W4 [[buffer(1)]],
                                 constant uint4& dims [[buffer(2)]], device uint* part [[buffer(3)]],
                                 uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]], uint ntg [[threadgroups_per_grid]]) {
  threadgroup float bv[16][8];
  threadgroup uint bi[16][8];
  const uint V = dims.x, K = dims.y, lo = dims.z, rows = min(dims.w, 16u);
  const uint o0 = g * 16u + sg * 2u, row_u4 = K >> 3, x_s4 = K >> 2, c256 = K >> 8;
  const bool a0 = o0 < V, a1 = (o0 + 1u) < V;
  if (rows <= 8u) {
    float s0[8], s1[8];
    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 8u; ++r) s0[r] = 0.0f, s1[r] = 0.0f;
    if (a0 && a1) {
      const device uint4* W0 = W4 + ulong(o0) * row_u4;
      const device uint4* W1 = W0 + row_u4;
      for (uint c = 0u; c < c256; ++c) {
        const uint xo = c * 64u + lane * 2u;
        const uint4 q0 = W0[c * 32u + lane], q1 = W1[c * 32u + lane];
        const float4 w00 = glm_lm_bf16x4(q0.x, q0.y), w01 = glm_lm_bf16x4(q0.z, q0.w);
        const float4 w10 = glm_lm_bf16x4(q1.x, q1.y), w11 = glm_lm_bf16x4(q1.z, q1.w);
        for (uint r = 0u; r < rows; ++r) {
          const ulong rx = ulong(r) * x_s4 + xo;
          const float4 x0 = X4[rx], x1 = X4[rx + 1u];
          s0[r] += dot(w00, x0) + dot(w01, x1);
          s1[r] += dot(w10, x0) + dot(w11, x1);
        }
      }
    } else if (a0) {
      const device uint4* W0 = W4 + ulong(o0) * row_u4;
      for (uint c = 0u; c < c256; ++c) {
        const uint xo = c * 64u + lane * 2u;
        const uint4 q0 = W0[c * 32u + lane];
        const float4 w00 = glm_lm_bf16x4(q0.x, q0.y), w01 = glm_lm_bf16x4(q0.z, q0.w);
        for (uint r = 0u; r < rows; ++r) {
          const ulong rx = ulong(r) * x_s4 + xo;
          s0[r] += dot(w00, X4[rx]) + dot(w01, X4[rx + 1u]);
        }
      }
    }
    for (uint r = 0u; r < rows; ++r) {
      float best = -INFINITY;
      uint bid = 0xFFFFFFFFu;
      if (a0) {
        best = glm_glue_lanes(s0[r]);
        bid = lo + o0;
        if (a1) {
          const float v1 = glm_glue_lanes(s1[r]);
          if (v1 > best) { best = v1; bid = lo + o0 + 1u; }
        }
      }
      if (lane == 0u) { bv[r][sg] = best; bi[r][sg] = bid; }
    }
  } else {
    if (lane == 0u) for (uint r = 0u; r < rows; ++r) { bv[r][sg] = -INFINITY; bi[r][sg] = 0xFFFFFFFFu; }
    for (uint q = 0u; q < 2u; ++q) {
      if (o0 + q >= V) break;
      float s[16];
      for (uint r = 0u; r < 16u; ++r) s[r] = 0.0f;
      const device uint4* Wq = W4 + ulong(o0 + q) * row_u4;
      for (uint c = 0u; c < c256; ++c) {
        const uint xo = c * 64u + lane * 2u;
        const uint4 qw = Wq[c * 32u + lane];
        const float4 w0 = glm_lm_bf16x4(qw.x, qw.y), w1 = glm_lm_bf16x4(qw.z, qw.w);
        for (uint r = 0u; r < rows; ++r) {
          const ulong rx = ulong(r) * x_s4 + xo;
          s[r] += dot(w0, X4[rx]) + dot(w1, X4[rx + 1u]);
        }
      }
      for (uint r = 0u; r < rows; ++r) {
        const float v = glm_glue_lanes(s[r]);
        if (lane == 0u && v > bv[r][sg]) { bv[r][sg] = v; bi[r][sg] = lo + o0 + q; }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && lane < rows) {
    float tb = bv[lane][0]; uint ti = bi[lane][0];
    for (uint k = 1u; k < 8u; ++k) if (bv[lane][k] > tb || (bv[lane][k] == tb && bi[lane][k] < ti)) { tb = bv[lane][k]; ti = bi[lane][k]; }
    const ulong dst = (ulong(lane) * ntg + g) * 2ul;
    part[dst] = as_type<uint>(tb);
    part[dst + 1ul] = ti;
  }
}

inline float glm_lm_q4_dot32(uint4 q, thread const half4 (&xv)[8]) {
  const uint w[4] = {q.x, q.y, q.z, q.w};
  half acc = 0.0h;
  #pragma clang loop unroll(full)
  for (uint k = 0u; k < 4u; ++k) {
    const uint v = w[k];
    const half2 a0 = as_type<half2>(((v & 0x000Fu) | ((v & 0x00F0u) << 12)) | 0x64006400u) - 1032.0h;
    const half2 a1 = as_type<half2>((((v >> 8) & 0x000Fu) | (((v >> 8) & 0x00F0u) << 12)) | 0x64006400u) - 1032.0h;
    const half2 b0 = as_type<half2>((((v >> 16) & 0x000Fu) | (((v >> 16) & 0x00F0u) << 12)) | 0x64006400u) - 1032.0h;
    const half2 b1 = as_type<half2>((((v >> 24) & 0x000Fu) | (((v >> 24) & 0x00F0u) << 12)) | 0x64006400u) - 1032.0h;
    acc += dot(half4(a0, a1), xv[2u * k]) + dot(half4(b0, b1), xv[2u * k + 1u]);
  }
  return float(acc);
}

inline void glm_top4_ins(thread float (&v)[4], thread uint (&id)[4], float nv, uint ni) {
  if (!(nv > v[3] || (nv == v[3] && ni < id[3]))) return;
  int p = 3;
  while (p > 0 && (nv > v[p - 1] || (nv == v[p - 1] && ni < id[p - 1]))) {
    v[p] = v[p - 1]; id[p] = id[p - 1]; --p;
  }
  v[p] = nv; id[p] = ni;
}

// Draft-only INT4 sym g64 lm_head + exact online softmax top-4 (4*(val,id), sum_exp) for r=1 (Q=2 256t, 652 GB/s!).
kernel void glm_lm_q4_argmax_1(const device float4* X4 [[buffer(0)]], const device uint4* W4 [[buffer(1)]],
                               const device half* S [[buffer(2)]], constant uint4& dims [[buffer(3)]],
                               device uint* part [[buffer(4)]], uint g [[threadgroup_position_in_grid]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint tid [[thread_index_in_threadgroup]]) {
  threadgroup half4 xh[256];
  threadgroup float sv[16];
  threadgroup uint si[16];
  const uint V = dims.x, K = dims.y, lo = dims.z, o0 = g * 16u + sg * 2u;
  const bool a0 = o0 < V, a1 = (o0 + 1u) < V;
  const uint row_u4 = K >> 5, c1024 = K >> 10, g_cols = K >> 6, sub_g = lane >> 1;
  const device uint4* W0 = W4 + ulong(o0) * row_u4;
  const device uint4* W1 = W0 + row_u4;
  const device half* S0 = S + ulong(o0) * g_cols;
  const device half* S1 = S0 + g_cols;
  float s0 = 0.0f, s1 = 0.0f;
  for (uint c = 0u; c < c1024; ++c) {
    xh[tid] = half4(X4[c * 256u + tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (a0) {
      half4 xv[8];
      #pragma clang loop unroll(full)
      for (uint k = 0u; k < 8u; ++k) xv[k] = xh[lane * 8u + k];
      const uint gi = c * 16u + sub_g;
      s0 = fma(glm_lm_q4_dot32(W0[c * 32u + lane], xv), float(S0[gi]), s0);
      if (a1) s1 = fma(glm_lm_q4_dot32(W1[c * 32u + lane], xv), float(S1[gi]), s1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float v0 = a0 ? glm_glue_lanes(s0) : -INFINITY;
  const float v1 = a1 ? glm_glue_lanes(s1) : -INFINITY;
  if (lane == 0u) {
    sv[2u * sg] = v0; si[2u * sg] = a0 ? (lo + o0) : 0xFFFFFFFFu;
    sv[2u * sg + 1u] = v1; si[2u * sg + 1u] = a1 ? (lo + o0 + 1u) : 0xFFFFFFFFu;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && lane == 0u) {
    float tv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    uint ti[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
    for (uint k = 0u; k < 16u; ++k) if (sv[k] > -INFINITY) glm_top4_ins(tv, ti, sv[k], si[k]);
    float tl = 0.0f;
    if (tv[0] > -INFINITY) for (uint k = 0u; k < 16u; ++k) if (sv[k] > -INFINITY) tl += exp(sv[k] - tv[0]);
    const ulong dst = ulong(g) * 9ul;
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) { part[dst + 2u * k] = as_type<uint>(tv[k]); part[dst + 2u * k + 1u] = ti[k]; }
    part[dst + 8u] = as_type<uint>(tl);
  }
}

// Batched r=2 INT4 sym g64 lm_head + top-4+sum_exp (loads W once for both branches -> 2x fewer DRAM reads!).
kernel void glm_lm_q4_argmax_2(const device float4* X4 [[buffer(0)]], const device uint4* W4 [[buffer(1)]],
                               const device half* S [[buffer(2)]], constant uint4& dims [[buffer(3)]],
                               device uint* part [[buffer(4)]], uint g [[threadgroup_position_in_grid]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint tid [[thread_index_in_threadgroup]]) {
  threadgroup half4 xh[2][256];
  threadgroup float sv[2][16];
  threadgroup uint si[2][16];
  const uint V = dims.x, K = dims.y, lo = dims.z, n_tgs = (V + 15u) >> 4, o0 = g * 16u + sg * 2u;
  const bool a0 = o0 < V, a1 = (o0 + 1u) < V;
  const uint row_u4 = K >> 5, c1024 = K >> 10, g_cols = K >> 6, sub_g = lane >> 1, k4 = K >> 2;
  const device uint4* W0 = W4 + ulong(o0) * row_u4, *W1 = W0 + row_u4;
  const device half* S0 = S + ulong(o0) * g_cols, *S1 = S0 + g_cols;
  float s0a = 0.0f, s1a = 0.0f, s0b = 0.0f, s1b = 0.0f;
  for (uint c = 0u; c < c1024; ++c) {
    xh[0][tid] = half4(X4[c * 256u + tid]);
    xh[1][tid] = half4(X4[k4 + c * 256u + tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (a0) {
      const uint gi = c * 16u + sub_g;
      const uint4 q0 = W0[c * 32u + lane], q1 = a1 ? W1[c * 32u + lane] : uint4(0u);
      const float sc0 = float(S0[gi]), sc1 = a1 ? float(S1[gi]) : 0.0f;
      half4 xa[8], xb[8];
      #pragma clang loop unroll(full)
      for (uint k = 0u; k < 8u; ++k) { xa[k] = xh[0][lane * 8u + k]; xb[k] = xh[1][lane * 8u + k]; }
      s0a = fma(glm_lm_q4_dot32(q0, xa), sc0, s0a);
      s0b = fma(glm_lm_q4_dot32(q0, xb), sc0, s0b);
      if (a1) {
        s1a = fma(glm_lm_q4_dot32(q1, xa), sc1, s1a);
        s1b = fma(glm_lm_q4_dot32(q1, xb), sc1, s1b);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float v0a = a0 ? glm_glue_lanes(s0a) : -INFINITY, v1a = a1 ? glm_glue_lanes(s1a) : -INFINITY;
  const float v0b = a0 ? glm_glue_lanes(s0b) : -INFINITY, v1b = a1 ? glm_glue_lanes(s1b) : -INFINITY;
  if (lane == 0u) {
    sv[0][2u * sg] = v0a; si[0][2u * sg] = a0 ? (lo + o0) : 0xFFFFFFFFu;
    sv[0][2u * sg + 1u] = v1a; si[0][2u * sg + 1u] = a1 ? (lo + o0 + 1u) : 0xFFFFFFFFu;
    sv[1][2u * sg] = v0b; si[1][2u * sg] = a0 ? (lo + o0) : 0xFFFFFFFFu;
    sv[1][2u * sg + 1u] = v1b; si[1][2u * sg + 1u] = a1 ? (lo + o0 + 1u) : 0xFFFFFFFFu;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg < 2u && lane == 0u) {
    float tv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    uint ti[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
    for (uint k = 0u; k < 16u; ++k) if (sv[sg][k] > -INFINITY) glm_top4_ins(tv, ti, sv[sg][k], si[sg][k]);
    float tl = 0.0f;
    if (tv[0] > -INFINITY) for (uint k = 0u; k < 16u; ++k) if (sv[sg][k] > -INFINITY) tl += exp(sv[sg][k] - tv[0]);
    const ulong dst = (ulong(sg) * n_tgs + g) * 9ul;
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) { part[dst + 2u * k] = as_type<uint>(tv[k]); part[dst + 2u * k + 1u] = ti[k]; }
    part[dst + 8u] = as_type<uint>(tl);
  }
}

// Final 1024-thread per-row reduction of per-TG (val, id) pairs directly into pair send slot `out[2*r..2*r+1]`.
kernel void glm_lm_argmax_reduce(const device uint* part [[buffer(0)]], constant uint& n_tgs [[buffer(1)]],
                                 device uint* out [[buffer(2)]], uint r [[threadgroup_position_in_grid]],
                                 uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  const device uint* rp = part + ulong(r) * n_tgs * 2ul;
  float best = -INFINITY;
  uint at = 0xFFFFFFFFu;
  for (uint i = t; i < n_tgs; i += 1024u) {
    const float v = as_type<float>(rp[2u * i]);
    const uint id = rp[2u * i + 1u];
    if (v > best || (v == best && id < at)) { best = v; at = id; }
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ob = simd_shuffle_xor(best, o);
    const uint oa = simd_shuffle_xor(at, o);
    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
  }
  threadgroup float vb[32];
  threadgroup uint va[32];
  if (lane == 0u) { vb[sg] = best; va[sg] = at; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0u) return;
  best = vb[lane];
  at = va[lane];
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ob = simd_shuffle_xor(best, o);
    const uint oa = simd_shuffle_xor(at, o);
    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
  }
  if (lane == 0u) { out[2u * r] = as_type<uint>(best); out[2u * r + 1u] = at; }
}

// Fast 256t (8-sg) per-row reduction of per-TG top-4 + sum_exp (9 uints/TG) into pair send slot out[9*r..9*r+8].
kernel void glm_lm_argmax_prob_reduce(const device uint* part [[buffer(0)]], constant uint& n_tgs [[buffer(1)]],
                                      device uint* out [[buffer(2)]], uint r [[threadgroup_position_in_grid]],
                                      uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
  const device uint* rp = part + ulong(r) * n_tgs * 9ul;
  float bv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}, lsum = 0.0f;
  uint bi[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
  for (uint i = t; i < n_tgs; i += 256u) {
    const float v0 = as_type<float>(rp[9u * i]);
    if (v0 > -INFINITY) {
      const float lv = as_type<float>(rp[9u * i + 8u]);
      if (v0 > bv[0]) lsum = (bv[0] > -INFINITY ? lsum * exp(bv[0] - v0) : 0.0f) + lv;
      else lsum += lv * exp(v0 - bv[0]);
      #pragma clang loop unroll(full)
      for (uint k = 0u; k < 4u; ++k) {
        const float vk = as_type<float>(rp[9u * i + 2u * k]);
        if (vk > -INFINITY) glm_top4_ins(bv, bi, vk, rp[9u * i + 2u * k + 1u]);
      }
    }
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    float ov[4]; uint oi[4];
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) { ov[k] = simd_shuffle_xor(bv[k], o); oi[k] = simd_shuffle_xor(bi[k], o); }
    const float ol = simd_shuffle_xor(lsum, o);
    if (ov[0] > -INFINITY) {
      if (ov[0] > bv[0]) lsum = (bv[0] > -INFINITY ? lsum * exp(bv[0] - ov[0]) : 0.0f) + ol;
      else lsum += ol * exp(ov[0] - bv[0]);
      #pragma clang loop unroll(full)
      for (uint k = 0u; k < 4u; ++k) if (ov[k] > -INFINITY) glm_top4_ins(bv, bi, ov[k], oi[k]);
    }
  }
  threadgroup float vb[8][4], vl[8];
  threadgroup uint va[8][4];
  if (lane == 0u) {
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) { vb[sg][k] = bv[k]; va[sg][k] = bi[k]; }
    vl[sg] = lsum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0u) return;
  #pragma clang loop unroll(full)
  for (uint k = 0u; k < 4u; ++k) { bv[k] = (lane < 8u) ? vb[lane][k] : -INFINITY; bi[k] = (lane < 8u) ? va[lane][k] : 0xFFFFFFFFu; }
  lsum = (lane < 8u) ? vl[lane] : 0.0f;
  for (ushort o = 4; o > 0; o >>= 1) {
    float ov[4]; uint oi[4];
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) { ov[k] = simd_shuffle_xor(bv[k], o); oi[k] = simd_shuffle_xor(bi[k], o); }
    const float ol = simd_shuffle_xor(lsum, o);
    if (ov[0] > -INFINITY) {
      if (ov[0] > bv[0]) lsum = (bv[0] > -INFINITY ? lsum * exp(bv[0] - ov[0]) : 0.0f) + ol;
      else lsum += ol * exp(ov[0] - bv[0]);
      #pragma clang loop unroll(full)
      for (uint k = 0u; k < 4u; ++k) if (ov[k] > -INFINITY) glm_top4_ins(bv, bi, ov[k], oi[k]);
    }
  }
  if (lane == 0u) {
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) { out[9u * r + 2u * k] = as_type<uint>(bv[k]); out[9u * r + 2u * k + 1u] = bi[k]; }
    out[9u * r + 8u] = as_type<uint>(lsum);
  }
}

struct CompactKvArgs {
  uint slot, base_pos, slot_keys, kept;
  uint has_idx, pad0, pad1, pad2;
  uint path[8];
};

// Compact accepted tree branch KV + index cache slots [base_pos + path[k]] -> [base_pos + k] (strictly increasing).
kernel void glm_mla_compact_kv(device uint4* cache [[buffer(0)]], device uint4* idx_cache [[buffer(1)]],
                               constant CompactKvArgs& a [[buffer(2)]], uint t [[thread_index_in_threadgroup]]) {
  const ulong k_base = (ulong(a.slot) * a.slot_keys + a.base_pos) * 72ul;
  const ulong i_base = (ulong(a.slot) * a.slot_keys + a.base_pos) * 16ul;
  for (uint k = 1u; k < a.kept; ++k) {
    const uint src = a.path[k];
    if (src != k) {
      if (t < 72u) cache[k_base + ulong(k) * 72ul + t] = cache[k_base + ulong(src) * 72ul + t];
      if (a.has_idx != 0u && t >= 72u && t < 88u) {
        const uint j = t - 72u;
        idx_cache[i_base + ulong(k) * 16ul + j] = idx_cache[i_base + ulong(src) * 16ul + j];
      }
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
}

struct MlaRowGlue { uint slot, pos, list, nkeys; };
struct MtpTreeAcceptArgs { uint rows, slot, base_pos, depth, i_topk, pad0, pad1, pad2; int parents[8]; };

// Single-TG GPU greedy tree accept matching lanes.accept.acceptPath: writes meta, compact_disp, d1_ids/idx, d_rows.
kernel void glm_mtp_tree_accept(const device uint* win_ids [[buffer(0)]], const device uint* win_picks [[buffer(1)]],
                                constant MtpTreeAcceptArgs& a [[buffer(2)]], device uint* meta [[buffer(3)]],
                                device uint* compact_disp [[buffer(4)]], device uint* d1_ids [[buffer(5)]],
                                device uint* d1_idx [[buffer(6)]], device uint* d2_idx [[buffer(7)]],
                                device MlaRowGlue* d_rows [[buffer(8)]], uint t [[thread_index_in_threadgroup]]) {
  if (t != 0u) return;
  uint path[8], cur = 0u, kept = 1u;
  path[0] = 0u;
  while (kept < a.rows) {
    const uint want = win_picks[cur];
    uint nxt = 0xFFFFFFFFu;
    for (uint r = cur + 1u; r < a.rows; ++r) {
      if (a.parents[r] == int(cur) && win_ids[r] == want) { nxt = r; break; }
    }
    if (nxt == 0xFFFFFFFFu) break;
    path[kept++] = nxt;
    cur = nxt;
  }
  uint need_c = 0u;
  for (uint k = 0u; k < kept; ++k) if (path[k] != k) need_c = 1u;
  meta[0] = kept; meta[1] = need_c;
  for (uint k = 0u; k < 8u; ++k) meta[2u + k] = (k < kept) ? path[k] : 0u;
  compact_disp[0] = need_c; compact_disp[1] = need_c; compact_disp[2] = need_c; compact_disp[3] = 0u;
  for (uint j = 0u; j < a.rows; ++j) {
    d1_ids[j] = (j < kept) ? win_picks[path[j]] : 0u;
    d1_idx[j] = (j < kept) ? path[j] : 0u;
  }
  d2_idx[0] = kept - 1u;
  const uint p0 = a.base_pos + kept - 1u;
  if (a.pad0 != 0u) {
    d_rows[0] = MlaRowGlue{a.slot, p0 + 1u, 0u, 0u}; d_rows[1] = MlaRowGlue{a.slot, p0 + 1u, a.i_topk, 0u};
    for (uint k = 3u; k <= a.depth && k < 8u; ++k) d_rows[k - 1u] = MlaRowGlue{a.slot, p0 + k - 1u, 0u, 0u};
  } else {
    for (uint k = 2u; k <= a.depth; ++k) d_rows[k - 2u] = MlaRowGlue{a.slot, p0 + k - 1u, 0u, 0u};
  }
}

// Indirect-dispatched KV + index compaction reading kept & path from meta (0 TGs when main branch wins!).
kernel void glm_mla_compact_kv_indirect(device uint4* cache [[buffer(0)]], device uint4* idx_cache [[buffer(1)]],
                                        const device uint* meta [[buffer(2)]], constant CompactKvArgs& a [[buffer(3)]],
                                        uint t [[thread_index_in_threadgroup]]) {
  const uint kept = meta[0];
  const ulong k_base = (ulong(a.slot) * a.slot_keys + a.base_pos) * 72ul;
  const ulong i_base = (ulong(a.slot) * a.slot_keys + a.base_pos) * 16ul;
  for (uint k = 1u; k < kept; ++k) {
    const uint src = meta[a.pad0 + k];
    if (src != k) {
      if (t < 72u) cache[k_base + ulong(k) * 72ul + t] = cache[k_base + ulong(src) * 72ul + t];
      if (a.has_idx != 0u && t >= 72u && t < 88u) {
        idx_cache[i_base + ulong(k) * 16ul + (t - 72u)] = idx_cache[i_base + ulong(src) * 16ul + (t - 72u)];
      }
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
}
