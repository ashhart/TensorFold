// EXLR8 dense matrices for prompt rows on the tensor units (after exlr8_dense.metal): weights exact in bf16, bf16 rows.
#include "../nax.h"

constant constexpr int TF_DR = 32;
constant constexpr int TF_DLD = 128 + 8;
constant constexpr int TF_DWLD = 16 + 8;

// Output o's eight weights at inputs i .. i + 7 as exact bf16 values; a format's per-group scale is left out.
inline void tf_dense_codes(const device uchar* W, TfDenseJob j, uint o, uint i, thread float (&w)[8]) {
  switch (j.format) {
    case TF_INT4: {
      const uint q = *reinterpret_cast<const device uint*>(W + j.w + ulong(o) * (j.in_total / 2) + i / 2);
      for (uint n = 0; n < 8; ++n) w[n] = float(int((q >> (4 * n)) & 15u) - 8);
      break;
    }
    case TF_INT4_IN: {
      const ulong g = ulong(i / 128) * j.out_total + o;
      const uint q = *reinterpret_cast<const device uint*>(W + j.w + g * 64 + (i % 128) / 2);
      for (uint n = 0; n < 8; ++n) w[n] = float(int((q >> (4 * n)) & 15u) - 8);
      break;
    }
    case TF_BF16: {
      const uint4 q = *reinterpret_cast<const device uint4*>(W + j.w + 2 * (ulong(o) * j.in_total + i));
      const uint u[4] = {q.x, q.y, q.z, q.w};
      for (uint n = 0; n < 4; ++n) w[2 * n] = as_type<float>(u[n] << 16), w[2 * n + 1] = as_type<float>(u[n] & 0xFFFF0000u);
      break;
    }
    default: {
      const uint2 q = *reinterpret_cast<const device uint2*>(W + j.w + ulong(o) * j.in_total + i);
      // MXFP8's power-of-two scale keeps an E4M3 value exact in bf16; FP8 block and row scales come after the sums
      const float s = (j.format == TF_MXFP8 ? as_type<float>(uint(W[j.s + ulong(o) * (j.in_total / 32) + i / 32]) << 23) : 1.0f) * 256.0f;
      const half2 a0 = as_type<half2>(((q.x & 0x007F007Fu) << 7) | ((q.x & 0x00800080u) << 8));
      const half2 b0 = as_type<half2>((((q.x >> 8) & 0x007F007Fu) << 7) | (((q.x >> 8) & 0x00800080u) << 8));
      const half2 a1 = as_type<half2>(((q.y & 0x007F007Fu) << 7) | ((q.y & 0x00800080u) << 8));
      const half2 b1 = as_type<half2>((((q.y >> 8) & 0x007F007Fu) << 7) | (((q.y >> 8) & 0x00800080u) << 8));
      w[0] = float(a0.x) * s; w[1] = float(b0.x) * s; w[2] = float(a0.y) * s; w[3] = float(b0.y) * s;
      w[4] = float(a1.x) * s; w[5] = float(b1.x) * s; w[6] = float(a1.y) * s; w[7] = float(b1.y) * s;
      break;
    }
  }
}

// The inputs a scale covers (the sums are scaled at each group's end) and the scale of output o's group at input i.
inline uint tf_dense_group(TfDenseJob j) {
  return j.format == TF_INT4 || j.format == TF_INT4_IN ? 32u : j.format == TF_FP8_BLOCK ? 128u : j.ins;
}
inline float tf_dense_scale(const device uchar* W, TfDenseJob j, uint o, uint i) {
  switch (j.format) {
    case TF_INT4: return float(*reinterpret_cast<const device half*>(W + j.s + 2 * (ulong(o) * (j.in_total / 32) + i / 32)));
    case TF_INT4_IN: {
      const ulong g = ulong(i / 128) * j.out_total + o;
      return float(*reinterpret_cast<const device half*>(W + j.s + 2 * (g * 4 + (i % 128) / 32)));
    }
    case TF_FP8_BLOCK: return reinterpret_cast<const device float*>(W + j.s)[ulong(o / 128) * ((j.in_total + 127) / 128) + i / 128];
    case TF_FP8_ROW: return reinterpret_cast<const device float*>(W + j.s)[o];
    default: return 1.0f;
  }
}

// groups[g] = (job, 32-output block); a threadgroup's two simdgroups take 16 outputs each for 32 rows (grid y).
kernel void tf_exlr8_dense_prompt(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
                                  const device TfDenseJob* jobs [[buffer(2)]], const device uint2* groups [[buffer(3)]],
                                  constant TfDenseRows& a [[buffer(4)]], device uchar* Y [[buffer(5)]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                  uint tid [[thread_index_in_threadgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat xs[TF_DR * TF_DLD];
  threadgroup bfloat ws[2 * 16 * TF_DWLD];
  const uint2 at = groups[tg.x];
  const TfDenseJob j = jobs[at.x];
  const uint row0 = TF_DR * tg.y, rows = min(uint(TF_DR), a.rows - row0), o0 = 32 * at.y + 16 * sg, G = tf_dense_group(j);
  const short2 home = tfp::frag_home(ushort(lane));
  threadgroup bfloat* w16 = ws + sg * 16 * TF_DWLD;
  tfp::frag<float> total[2] = {tfp::frag<float>(0.0f), tfp::frag<float>(0.0f)};
  tfp::frag<float> part[2] = {tfp::frag<float>(0.0f), tfp::frag<float>(0.0f)};
  for (uint c = 0; c < j.ins / 128; ++c) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < TF_DR * 128; i += 64) {
      const uint rr = i / 128;
      xs[rr * TF_DLD + i % 128] = rr < rows ? bfloat(X[ulong(row0 + rr) * a.x_stride + j.x_col + 128 * c + i % 128]) : bfloat(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 8; ++kk) {
      const uint o = lane / 2, i = j.in0 + 128 * c + 16 * kk + 8 * (lane % 2);
      float w[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      if (o0 + o < j.outs) tf_dense_codes(W, j, j.out0 + o0 + o, i, w);
      for (uint n = 0; n < 8; ++n) w16[o * TF_DWLD + 8 * (lane % 2) + n] = bfloat(w[n]);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      tfp::frag<bfloat> wt, x0, x1;
      tfp::frag_get(wt, w16, TF_DWLD, 0, 0, home);
      tfp::frag_get_t(x0, xs, TF_DLD, 0, 16 * kk, home);
      tfp::frag_get_t(x1, xs, TF_DLD, 16, 16 * kk, home);
      tfp::mma_16x32<false, true>(part[0], part[1], wt, x0, x1);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      const uint done = 128 * c + 16 * kk + 16;
      if (done % G == 0 || done == j.ins) {
        for (short e = 0; e < 8; ++e) {
          const uint oo = o0 + home.y + (e >> 2) * 8;
          const float s = oo < j.outs ? tf_dense_scale(W, j, j.out0 + oo, j.in0 + done - 1) : 0.0f;
          for (uint h = 0; h < 2; ++h) total[h][e] = fma(s, part[h][e], total[h][e]), part[h][e] = 0.0f;
        }
      }
    }
  }
  for (uint h = 0; h < 2; ++h)
    for (short e = 0; e < 8; ++e) {
      const uint oo = o0 + home.y + (e >> 2) * 8, r = 16 * h + home.x + TF_COL(e);
      if (oo >= j.outs || r >= rows) continue;
      const ulong y = ulong(row0 + r) * a.y_stride + j.y_col + oo;
      if (a.y_f32) reinterpret_cast<device float*>(Y)[y] = total[h][e];
      else reinterpret_cast<device bfloat*>(Y)[y] = bfloat(total[h][e]);
    }
}

#if defined(TF_SIMD_FRAGS) && defined(TF_GLM_M3_FAST)
inline half4 tf_dp_fp8x4_h(uint q) {
  const half2 a = as_type<half2>(((q & 0x007F007Fu) << 7) | ((q & 0x00800080u) << 8));
  const half2 b = as_type<half2>((((q >> 8) & 0x007F007Fu) << 7) | (((q >> 8) & 0x00800080u) << 8));
  return half4(a.x, b.x, a.y, b.y);
}
inline void tf_dp_fp8_store8_sc(uint2 q, float sc, threadgroup bfloat* dst) {
  const half4 h0 = tf_dp_fp8x4_h(q.x), h1 = tf_dp_fp8x4_h(q.y);
  dst[0] = bfloat(float(h0.x) * sc); dst[1] = bfloat(float(h0.y) * sc);
  dst[2] = bfloat(float(h0.z) * sc); dst[3] = bfloat(float(h0.w) * sc);
  dst[4] = bfloat(float(h1.x) * sc); dst[5] = bfloat(float(h1.y) * sc);
  dst[6] = bfloat(float(h1.z) * sc); dst[7] = bfloat(float(h1.w) * sc);
}

// High-throughput 64x64 half MMA (FP32 accum) prompt GEMM for rows >= 64: 2x2 sgs, coalesced half4 row-major xs!
kernel void tf_exlr8_dense_prompt_64x64(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
                                        const device TfDenseJob* jobs [[buffer(2)]], const device uint2* groups [[buffer(3)]],
                                        constant TfDenseRows& a [[buffer(4)]], device uchar* Y [[buffer(5)]],
                                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                        uint tid [[thread_index_in_threadgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr uint XLD = 32u + 8u, WLD = 32u + 8u;
  threadgroup half xs[64u * XLD];
  threadgroup half ws[64u * WLD];
  const uint2 at = groups[tg.x];
  const TfDenseJob j = jobs[at.x];
  const uint row0 = 64u * tg.y, rows = min(64u, a.rows - row0), n0 = 64u * at.y;
  const uint sg_n = (sg & 1u) * 32u, sg_m = (sg >> 1) * 32u;
  const bool is_bf16 = (j.format == TF_BF16), is_fp32 = (j.format == TF_FP32);
  const bool aligned4 = (((a.x_stride | j.x_col | j.in0) & 3u) == 0u);
  const uint sb_cols = (j.in_total + 127u) / 128u;
  const device float4* X4 = reinterpret_cast<const device float4*>(X);
  const uint x_s4 = a.x_stride >> 2, x_c4 = j.x_col >> 2;
  const uint oo = tid >> 1, kk16 = (tid & 1u) * 16u;
  const bool w_ok = (n0 + oo) < j.outs;
  const device float* sc_ptr = reinterpret_cast<const device float*>(W + j.s) + ulong((j.out0 + n0 + oo) >> 7) * sb_cols + (j.in0 >> 7);
  const ulong w_row_fp8 = j.w + ulong(j.out0 + n0 + oo) * j.in_total + j.in0;
  const ulong w_row_bf16 = j.w + 2ul * (ulong(j.out0 + n0 + oo) * j.in_total + j.in0);
  const ulong w_row_f32 = j.w + 4ul * (ulong(j.out0 + n0 + oo) * j.in_total + j.in0);

  simdgroup_matrix<float, 8, 8> acc[4][4];
  #pragma clang loop unroll(full)
  for (uint i = 0u; i < 4u; ++i)
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < 4u; ++m) acc[i][m] = simdgroup_matrix<float, 8, 8>(0.0f);

  for (uint c = 0u; c < j.ins / 128u; ++c) {
    const half sc_h = (!is_bf16 && !is_fp32) ? half(sc_ptr[c] * 256.0f) : half(1.0h);
    for (uint kh = 0u; kh < 4u; ++kh) {
      const uint kb = 128u * c + 32u * kh;
      if (aligned4) {
        #pragma clang loop unroll(full)
        for (uint p = 0u; p < 4u; ++p) {
          const uint idx = p * 128u + tid, rr = idx >> 3, cc4 = idx & 7u;
          const float4 v = (rr < rows) ? X4[ulong(row0 + rr) * x_s4 + x_c4 + (kb >> 2) + cc4] : float4(0.0f);
          *reinterpret_cast<threadgroup half4*>(xs + rr * XLD + cc4 * 4u) = half4(v);
        }
      } else {
        for (uint idx = tid; idx < 64u * 32u; idx += 128u) {
          const uint rr = idx >> 5, cc = idx & 31u;
          xs[rr * XLD + cc] = (rr < rows) ? half(X[ulong(row0 + rr) * a.x_stride + j.x_col + kb + cc]) : half(0.0h);
        }
      }
      threadgroup half4* wd4 = reinterpret_cast<threadgroup half4*>(ws + oo * WLD + kk16);
      if (w_ok) {
        if (is_bf16) {
          const device uint4* bp = reinterpret_cast<const device uint4*>(W + w_row_bf16 + 2ul * (kb + kk16));
          for (uint q = 0u; q < 2u; ++q) {
            const uint4 bv = bp[q];
            wd4[2u * q] = half4(float4(as_type<float>(bv.x << 16), as_type<float>(bv.x & 0xFFFF0000u),
                                       as_type<float>(bv.y << 16), as_type<float>(bv.y & 0xFFFF0000u)));
            wd4[2u * q + 1u] = half4(float4(as_type<float>(bv.z << 16), as_type<float>(bv.z & 0xFFFF0000u),
                                            as_type<float>(bv.w << 16), as_type<float>(bv.w & 0xFFFF0000u)));
          }
        } else if (is_fp32) {
          const device float4* fp = reinterpret_cast<const device float4*>(W + w_row_f32 + 4ul * (kb + kk16));
          wd4[0] = half4(fp[0]); wd4[1] = half4(fp[1]); wd4[2] = half4(fp[2]); wd4[3] = half4(fp[3]);
        } else {
          const uint4 qw = *reinterpret_cast<const device uint4*>(W + w_row_fp8 + kb + kk16);
          wd4[0] = tf_dp_fp8x4_h(qw.x) * sc_h; wd4[1] = tf_dp_fp8x4_h(qw.y) * sc_h;
          wd4[2] = tf_dp_fp8x4_h(qw.z) * sc_h; wd4[3] = tf_dp_fp8x4_h(qw.w) * sc_h;
        }
      } else {
        wd4[0] = half4(0.0h); wd4[1] = half4(0.0h); wd4[2] = half4(0.0h); wd4[3] = half4(0.0h);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint kk = 0u; kk < 32u; kk += 8u) {
        simdgroup_matrix<half, 8, 8> wa[4], xb[4];
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 4u; ++i) simdgroup_load(wa[i], ws + (sg_n + i * 8u) * WLD + kk, WLD, ulong2(0, 0), false);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < 4u; ++m) simdgroup_load(xb[m], xs + (sg_m + m * 8u) * XLD + kk, XLD, ulong2(0, 0), true);
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 4u; ++i)
          #pragma clang loop unroll(full)
          for (uint m = 0u; m < 4u; ++m) simdgroup_multiply_accumulate(acc[i][m], wa[i], xb[m], acc[i][m]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  const short2 home = tfp::frag_home(ushort(lane));
  #pragma clang loop unroll(full)
  for (uint i = 0u; i < 4u; ++i) {
    const uint out_o = n0 + sg_n + i * 8u + uint(home.y);
    if (out_o >= j.outs) continue;
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < 4u; ++m) {
      const uint r0 = sg_m + m * 8u + uint(home.x);
      if (r0 < rows) {
        const ulong y0 = ulong(row0 + r0) * a.y_stride + j.y_col + out_o;
        if (a.y_f32) reinterpret_cast<device float*>(Y)[y0] = acc[i][m].thread_elements()[0];
        else reinterpret_cast<device bfloat*>(Y)[y0] = bfloat(acc[i][m].thread_elements()[0]);
      }
      if (r0 + 1u < rows) {
        const ulong y1 = ulong(row0 + r0 + 1u) * a.y_stride + j.y_col + out_o;
        if (a.y_f32) reinterpret_cast<device float*>(Y)[y1] = acc[i][m].thread_elements()[1];
        else reinterpret_cast<device bfloat*>(Y)[y1] = bfloat(acc[i][m].thread_elements()[1]);
      }
    }
  }
}

// Fast 4-simdgroup (64 out x 32 rows / TG) transposed-xs MMA kernel for FP8_BLOCK & BF16 on M3/M4 (~20 TFLOPS).
kernel void tf_exlr8_dense_prompt_fast(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
                                       const device TfDenseJob* jobs [[buffer(2)]], const device uint2* groups [[buffer(3)]],
                                       constant TfDenseRows& a [[buffer(4)]], device uchar* Y [[buffer(5)]],
                                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                       uint tid [[thread_index_in_threadgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr uint XLD = 32u + 8u;
  constexpr uint WLD = 16u + 8u;
  threadgroup bfloat xs[64u * XLD];
  threadgroup bfloat ws[4u * 16u * WLD];
  const uint2 at = groups[tg.x];
  const TfDenseJob j = jobs[at.x];
  const uint row0 = 32u * tg.y, rows = min(32u, a.rows - row0);
  const uint o0 = 64u * at.y + 16u * sg;
  const bool sg_active = o0 < j.outs;
  const uint o_lane = lane >> 1, p_lane = 8u * (lane & 1u);
  const bool lane_row_ok = (o0 + o_lane) < j.outs;
  const bool is_bf16 = (j.format == TF_BF16);
  const bool is_fp32 = (j.format == TF_FP32);
  const bool aligned4 = (((a.x_stride | j.x_col | j.in0) & 3u) == 0u);
  const uint sb_cols = (j.in_total + 127u) / 128u;
  const uint out_sb = (j.out0 + o0) / 128u;
  const ulong w_row_fp8 = j.w + ulong(j.out0 + o0 + o_lane) * j.in_total + j.in0;
  const ulong w_row_bf16 = j.w + 2ul * (ulong(j.out0 + o0 + o_lane) * j.in_total + j.in0);
  const ulong w_row_f32 = j.w + 4ul * (ulong(j.out0 + o0 + o_lane) * j.in_total + j.in0);
  const device float* sc_ptr = reinterpret_cast<const device float*>(W + j.s) + ulong(out_sb) * sb_cols + (j.in0 / 128u);
  const device float4* X4 = reinterpret_cast<const device float4*>(X);
  const uint x_stride4 = a.x_stride >> 2;
  const uint x_col4 = j.x_col >> 2;
  threadgroup bfloat* w16 = ws + sg * 16u * WLD;

  simdgroup_matrix<float, 8, 8> acc[2][4];
  for (uint i = 0; i < 2u; ++i)
    for (uint m = 0; m < 4u; ++m) acc[i][m] = simdgroup_matrix<float, 8, 8>(0.0f);

  const uint n_blocks = j.ins / 128u;
  for (uint c = 0; c < n_blocks; ++c) {
    const float sc = (!is_bf16 && !is_fp32 && sg_active) ? (sc_ptr[c] * 256.0f) : 1.0f;
    for (uint kh = 0; kh < 2u; ++kh) {
      const uint k_base = 128u * c + 64u * kh;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (aligned4) {
        for (uint idx = tid; idx < 32u * 16u; idx += 128u) {
          const uint rr = idx >> 4, cc4 = idx & 15u;
          const float4 v = rr < rows ? X4[ulong(row0 + rr) * x_stride4 + x_col4 + (k_base >> 2) + cc4] : float4(0.0f);
          const uint k0 = cc4 * 4u;
          xs[(k0 + 0u) * XLD + rr] = bfloat(v.x);
          xs[(k0 + 1u) * XLD + rr] = bfloat(v.y);
          xs[(k0 + 2u) * XLD + rr] = bfloat(v.z);
          xs[(k0 + 3u) * XLD + rr] = bfloat(v.w);
        }
      } else {
        for (uint idx = tid; idx < 32u * 64u; idx += 128u) {
          const uint rr = idx >> 6, cc = idx & 63u;
          xs[cc * XLD + rr] = rr < rows ? bfloat(X[ulong(row0 + rr) * a.x_stride + j.x_col + k_base + cc]) : bfloat(0.0f);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg_active) {
        for (uint kk = 0; kk < 4u; ++kk) {
          threadgroup bfloat* wd = w16 + o_lane * WLD + p_lane;
          if (lane_row_ok) {
            if (is_bf16) {
              *reinterpret_cast<threadgroup uint4*>(wd) =
                  *reinterpret_cast<const device uint4*>(W + w_row_bf16 + 2ul * (k_base + 16u * kk + p_lane));
            } else if (is_fp32) {
              const device float4* fp = reinterpret_cast<const device float4*>(W + w_row_f32 + 4ul * (k_base + 16u * kk + p_lane));
              const float4 v0 = fp[0], v1 = fp[1];
              wd[0] = bfloat(v0.x); wd[1] = bfloat(v0.y); wd[2] = bfloat(v0.z); wd[3] = bfloat(v0.w);
              wd[4] = bfloat(v1.x); wd[5] = bfloat(v1.y); wd[6] = bfloat(v1.z); wd[7] = bfloat(v1.w);
            } else {
              const uint2 q = *reinterpret_cast<const device uint2*>(W + w_row_fp8 + k_base + 16u * kk + p_lane);
              tf_dp_fp8_store8_sc(q, sc, wd);
            }
          } else {
            *reinterpret_cast<threadgroup uint4*>(wd) = uint4(0u);
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          simdgroup_matrix<bfloat, 8, 8> a00, a01, a10, a11, b0, b1;
          simdgroup_load(a00, w16, WLD, ulong2(0, 0), false);
          simdgroup_load(a01, w16, WLD, ulong2(8, 0), false);
          simdgroup_load(a10, w16, WLD, ulong2(0, 8), false);
          simdgroup_load(a11, w16, WLD, ulong2(8, 8), false);
          for (uint m = 0; m < 4u; ++m) {
            simdgroup_load(b0, xs + (16u * kk) * XLD + m * 8u, XLD, ulong2(0, 0), false);
            simdgroup_load(b1, xs + (16u * kk + 8u) * XLD + m * 8u, XLD, ulong2(0, 0), false);
            simdgroup_multiply_accumulate(acc[0][m], a00, b0, acc[0][m]);
            simdgroup_multiply_accumulate(acc[1][m], a10, b0, acc[1][m]);
            simdgroup_multiply_accumulate(acc[0][m], a01, b1, acc[0][m]);
            simdgroup_multiply_accumulate(acc[1][m], a11, b1, acc[1][m]);
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
    }
  }
  if (!sg_active) return;
  const short2 home = tfp::frag_home(ushort(lane));
  for (uint i = 0; i < 2u; ++i) {
    const uint oo = o0 + i * 8u + uint(home.y);
    if (oo >= j.outs) continue;
    for (uint m = 0; m < 4u; ++m) {
      const uint r0 = m * 8u + uint(home.x);
      if (r0 < rows) {
        const ulong y0 = ulong(row0 + r0) * a.y_stride + j.y_col + oo;
        if (a.y_f32) reinterpret_cast<device float*>(Y)[y0] = acc[i][m].thread_elements()[0];
        else reinterpret_cast<device bfloat*>(Y)[y0] = bfloat(acc[i][m].thread_elements()[0]);
      }
      if (r0 + 1u < rows) {
        const ulong y1 = ulong(row0 + r0 + 1u) * a.y_stride + j.y_col + oo;
        if (a.y_f32) reinterpret_cast<device float*>(Y)[y1] = acc[i][m].thread_elements()[1];
        else reinterpret_cast<device bfloat*>(Y)[y1] = bfloat(acc[i][m].thread_elements()[1]);
      }
    }
  }
}
// Native BF16 MMA 32x64 prompt router GEMM (128t, 256 TGs for 2048x256): direct uint4 BF16 copy, -21% on route!
kernel void tf_exlr8_dense_prompt_bf16_32x64(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
                                             const device TfDenseJob* jobs [[buffer(2)]], const device uint2* groups [[buffer(3)]],
                                             constant TfDenseRows& a [[buffer(4)]], device uchar* Y [[buffer(5)]],
                                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                             uint tid [[thread_index_in_threadgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr uint XLD = 40u, WLD = 40u;
  threadgroup bfloat xs[64u * XLD], ws[32u * WLD];
  const uint2 at = groups[tg.x];
  const TfDenseJob j = jobs[at.x];
  const uint n0 = 32u * at.y, row0 = 64u * tg.y;
  const uint rows = (row0 < a.rows) ? min(64u, a.rows - row0) : 0u;
  if (rows == 0u || n0 >= j.outs) return;
  const uint sg_n = (sg >> 1) * 16u, sg_m = (sg & 1u) * 32u;
  const uint oo = tid >> 2, kk8 = (tid & 3u) * 8u;
  const bool w_ok = (n0 + oo) < j.outs, is_bf16 = (j.format == TF_BF16);
  const bool aligned4 = (((a.x_stride | j.x_col | j.in0) & 3u) == 0u);
  const device float4* X4 = reinterpret_cast<const device float4*>(X);
  const uint x_s4 = a.x_stride >> 2, x_c4 = j.x_col >> 2;
  const ulong w_row_bf16 = j.w + 2ul * (ulong(j.out0 + n0 + oo) * j.in_total + j.in0);
  const ulong w_row_f32 = j.w + 4ul * (ulong(j.out0 + n0 + oo) * j.in_total + j.in0);
  simdgroup_matrix<float, 8, 8> acc[2][4];
  #pragma clang loop unroll(full)
  for (uint i = 0u; i < 2u; ++i)
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < 4u; ++m) acc[i][m] = simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint kb = 0u; kb < j.ins; kb += 32u) {
    if (aligned4) {
      #pragma clang loop unroll(full)
      for (uint p = 0u; p < 4u; ++p) {
        const uint idx = p * 128u + tid, rr = idx >> 3, cc4 = idx & 7u;
        const float4 v = (rr < rows) ? X4[ulong(row0 + rr) * x_s4 + x_c4 + (kb >> 2) + cc4] : float4(0.0f);
        xs[rr * XLD + cc4 * 4u + 0u] = bfloat(v.x); xs[rr * XLD + cc4 * 4u + 1u] = bfloat(v.y);
        xs[rr * XLD + cc4 * 4u + 2u] = bfloat(v.z); xs[rr * XLD + cc4 * 4u + 3u] = bfloat(v.w);
      }
    } else {
      for (uint idx = tid; idx < 64u * 32u; idx += 128u) {
        const uint rr = idx >> 5, cc = idx & 31u;
        xs[rr * XLD + cc] = (rr < rows) ? bfloat(X[ulong(row0 + rr) * a.x_stride + j.x_col + kb + cc]) : bfloat(0.0f);
      }
    }
    threadgroup bfloat* wd = ws + oo * WLD + kk8;
    if (w_ok) {
      if (is_bf16) {
        *reinterpret_cast<threadgroup uint4*>(wd) = *reinterpret_cast<const device uint4*>(W + w_row_bf16 + 2ul * (kb + kk8));
      } else {
        const device float4* fp = reinterpret_cast<const device float4*>(W + w_row_f32 + 4ul * (kb + kk8));
        const float4 v0 = fp[0], v1 = fp[1];
        wd[0] = bfloat(v0.x); wd[1] = bfloat(v0.y); wd[2] = bfloat(v0.z); wd[3] = bfloat(v0.w);
        wd[4] = bfloat(v1.x); wd[5] = bfloat(v1.y); wd[6] = bfloat(v1.z); wd[7] = bfloat(v1.w);
      }
    } else {
      *reinterpret_cast<threadgroup uint4*>(wd) = uint4(0u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint kk = 0u; kk < 32u; kk += 8u) {
      simdgroup_matrix<bfloat, 8, 8> wa[2], xb[4];
      simdgroup_load(wa[0], ws + sg_n * WLD + kk, WLD, ulong2(0, 0), false);
      simdgroup_load(wa[1], ws + (sg_n + 8u) * WLD + kk, WLD, ulong2(0, 0), false);
      #pragma clang loop unroll(full)
      for (uint m = 0u; m < 4u; ++m) simdgroup_load(xb[m], xs + (sg_m + m * 8u) * XLD + kk, XLD, ulong2(0, 0), true);
      #pragma clang loop unroll(full)
      for (uint m = 0u; m < 4u; ++m) {
        simdgroup_multiply_accumulate(acc[0][m], wa[0], xb[m], acc[0][m]);
        simdgroup_multiply_accumulate(acc[1][m], wa[1], xb[m], acc[1][m]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const short2 home = tfp::frag_home(ushort(lane));
  #pragma clang loop unroll(full)
  for (uint i = 0u; i < 2u; ++i) {
    const uint out_o = n0 + sg_n + i * 8u + uint(home.y);
    if (out_o >= j.outs) continue;
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < 4u; ++m) {
      const uint r0 = sg_m + m * 8u + uint(home.x);
      if (r0 < rows) {
        const ulong y0 = ulong(row0 + r0) * a.y_stride + j.y_col + out_o;
        if (a.y_f32) reinterpret_cast<device float*>(Y)[y0] = acc[i][m].thread_elements()[0];
        else reinterpret_cast<device bfloat*>(Y)[y0] = bfloat(acc[i][m].thread_elements()[0]);
      }
      if (r0 + 1u < rows) {
        const ulong y1 = ulong(row0 + r0 + 1u) * a.y_stride + j.y_col + out_o;
        if (a.y_f32) reinterpret_cast<device float*>(Y)[y1] = acc[i][m].thread_elements()[1];
        else reinterpret_cast<device bfloat*>(Y)[y1] = bfloat(acc[i][m].thread_elements()[1]);
      }
    }
  }
}
#endif
