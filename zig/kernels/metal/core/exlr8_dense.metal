// EXLR8 dense matrices stored by row: int4 g32, MXFP8, FP8 block, FP8 row, bf16, fp32, decoded to the reference's fp32.
#include <metal_stdlib>
using namespace metal;

// Formats as the record numbers them; 12 is int4 g32 in the input-sharded layout (q4 [in/128, out, 64]).
constant constexpr uint TF_INT4 = 2, TF_MXFP8 = 3, TF_FP8_BLOCK = 4, TF_FP8_ROW = 5, TF_BF16 = 6, TF_FP32 = 7, TF_INT4_IN = 12;

// Rows [out0, out0 + outs) of one matrix over its inputs [in0, in0 + ins); w and s are byte offsets into W.
struct TfDenseJob {
  uint format, out0, outs, in0;
  uint ins, out_total, in_total, x_col;
  uint y_col, pad0, pad1, pad2;
  ulong w, s;
};
struct TfDenseRows {
  uint rows, x_stride, y_stride, y_f32;
};

// An E4M3 code's value: its 7 low bits as fp16 bits are the value times 2^-8 (normal and subnormal alike).
inline float tf_e4m3(uint code) {
  const float v = float(as_type<half>(ushort((code & 0x7Fu) << 7))) * 256.0f;
  const float s = (code & 0x80u) ? -v : v;
  return (code & 0x7Fu) == 0x7Fu ? NAN : s;
}

// Lane `lane`'s four weights of matrix row o at chunk inputs 4 lane .. 4 lane + 3, chunk c (128 inputs from in0).
inline void tf_dense_weights(const device uchar* W, TfDenseJob j, uint o, uint c, uint lane, thread float (&w)[4]) {
  const uint i = j.in0 + 128 * c + 4 * lane;
  switch (j.format) {
    case TF_INT4: {
      const ushort q = *reinterpret_cast<const device ushort*>(W + j.w + ulong(o) * (j.in_total / 2) + i / 2);
      const float s = float(*reinterpret_cast<const device half*>(W + j.s + 2 * (ulong(o) * (j.in_total / 32) + i / 32)));
      for (uint n = 0; n < 4; ++n) w[n] = float(int((q >> (4 * n)) & 15u) - 8) * s;
      break;
    }
    case TF_INT4_IN: {
      const ulong g = ulong(i / 128) * j.out_total + o;
      const ushort q = *reinterpret_cast<const device ushort*>(W + j.w + g * 64 + (i % 128) / 2);
      const float s = float(*reinterpret_cast<const device half*>(W + j.s + 2 * (g * 4 + (i % 128) / 32)));
      for (uint n = 0; n < 4; ++n) w[n] = float(int((q >> (4 * n)) & 15u) - 8) * s;
      break;
    }
    case TF_MXFP8: {
      const uint q = *reinterpret_cast<const device uint*>(W + j.w + ulong(o) * j.in_total + i);
      const uint e = W[j.s + ulong(o) * (j.in_total / 32) + i / 32];
      // 2^(e - 127) is exact; e outside [1, 254] would leave fp32's normal range and never occurs in a release.
      const float s = as_type<float>(e << 23);
      for (uint n = 0; n < 4; ++n) w[n] = tf_e4m3((q >> (8 * n)) & 0xFFu) * s;
      break;
    }
    case TF_FP8_BLOCK:
    case TF_FP8_ROW: {
      const uint q = *reinterpret_cast<const device uint*>(W + j.w + ulong(o) * j.in_total + i);
      const ulong at = j.format == TF_FP8_ROW ? ulong(o) : ulong(o / 128) * ((j.in_total + 127) / 128) + i / 128;
      const float s = reinterpret_cast<const device float*>(W + j.s)[at] * 256.0f;
      const half2 a = as_type<half2>(((q & 0x007F007Fu) << 7) | ((q & 0x00800080u) << 8));
      const half2 b = as_type<half2>((((q >> 8) & 0x007F007Fu) << 7) | (((q >> 8) & 0x00800080u) << 8));
      w[0] = float(a.x) * s; w[1] = float(b.x) * s; w[2] = float(a.y) * s; w[3] = float(b.y) * s;
      break;
    }
    case TF_BF16: {
      const uint2 q = *reinterpret_cast<const device uint2*>(W + j.w + 2 * (ulong(o) * j.in_total + i));
      w[0] = as_type<float>(q.x << 16); w[1] = as_type<float>(q.x & 0xFFFF0000u);
      w[2] = as_type<float>(q.y << 16); w[3] = as_type<float>(q.y & 0xFFFF0000u);
      break;
    }
    default: {
      const float4 q = *reinterpret_cast<const device float4*>(W + j.w + 4 * (ulong(o) * j.in_total + i));
      w[0] = q.x, w[1] = q.y, w[2] = q.z, w[3] = q.w;
      break;
    }
  }
}

// The lane's partial sums, added over the simdgroup in one fixed butterfly (the CUDA kernels pair lanes the same way).
inline float tf_dense_lanes(float v) {
  for (ushort d = 16; d > 0; d >>= 1) v += simd_shuffle_xor(v, d);
  return v;
}
inline half4 tf_fp8x4_h(uint q) {
  const half2 a = as_type<half2>(((q & 0x007F007Fu) << 7) | ((q & 0x00800080u) << 8));
  const half2 b = as_type<half2>((((q >> 8) & 0x007F007Fu) << 7) | (((q >> 8) & 0x00800080u) << 8));
  return half4(a.x, b.x, a.y, b.y);
}
inline uint tf_bf16x2(float a, float b) {
  return uint(as_type<ushort>(bfloat(a))) | (uint(as_type<ushort>(bfloat(b))) << 16);
}
inline float4 tf_bf16x4(uint a, uint b) {
  return float4(as_type<float>(a << 16), as_type<float>(a & 0xFFFF0000u),
                as_type<float>(b << 16), as_type<float>(b & 0xFFFF0000u));
}

// A threadgroup is 8 Q rows of one job (Q a simdgroup, the same sums whatever Q); inputs 128 at a time via xs.
template <uint R, uint Q>
inline void tf_dense_job(const device float* X, const device uchar* W, TfDenseJob j, uint block, constant TfDenseRows& a,
                         device uchar* Y, uint sg, uint lane, uint tid, threadgroup float* xs) {
  const uint rows = min(a.rows, R), o0 = 8 * Q * block + Q * sg;
  if ((j.in0 & 127u) == 0u && (j.in_total & 15u) == 0u && (j.w & 15ul) == 0ul && (j.x_col % 4u) == 0u &&
      (a.x_stride % 4u) == 0u && (j.ins % 512u) == 0u &&
      (j.format == TF_FP8_BLOCK || j.format == TF_BF16 || j.format == TF_FP32)) {
    const device float4* X4 = reinterpret_cast<const device float4*>(X + j.x_col);
    const ulong x_s4 = ulong(a.x_stride >> 2);
    constexpr uint CQ = (R > 8u && Q > 2u) ? 2u : Q;
    for (uint q0 = 0u; q0 < Q; q0 += CQ) {
      float acc[CQ][R];
      for (uint q = 0; q < CQ; ++q) for (uint r = 0; r < R; ++r) acc[q][r] = 0.0f;
      if (j.format == TF_FP8_BLOCK) {
        const uint c512 = j.ins / 512u, sb_cols = (j.in_total + 127u) / 128u, sb0 = (j.in0 >> 7) + (lane >> 3);
        const device float* S = reinterpret_cast<const device float*>(W + j.s);
        if (R == 1u) {
          for (uint c = 0; c < c512; ++c) {
            const uint x_off = c * 128u + lane * 4u, sb = sb0 + c * 4u;
            const half4 x0 = half4(X4[x_off]), x1 = half4(X4[x_off + 1u]), x2 = half4(X4[x_off + 2u]), x3 = half4(X4[x_off + 3u]);
            for (uint q = 0; q < CQ; ++q) {
              if (o0 + q0 + q >= j.outs) break;
              const uint o = j.out0 + o0 + q0 + q;
              const uint4 qw = *reinterpret_cast<const device uint4*>(W + j.w + ulong(o) * j.in_total + j.in0 + c * 512u + lane * 16u);
              const float s = S[ulong(o / 128u) * sb_cols + sb] * 256.0f;
              acc[q][0] = fma(float(dot(tf_fp8x4_h(qw.x), x0)) + float(dot(tf_fp8x4_h(qw.y), x1)) +
                              float(dot(tf_fp8x4_h(qw.z), x2)) + float(dot(tf_fp8x4_h(qw.w), x3)), s, acc[q][0]);
            }
          }
        } else {
          threadgroup half4* xh4 = reinterpret_cast<threadgroup half4*>(xs);
          for (uint c = 0; c < c512; ++c) {
            for (uint idx = tid; idx < rows * 128u; idx += 256u) {
              const uint rr = idx >> 7, col4 = idx & 127u;
              xh4[idx] = half4(X4[ulong(rr) * x_s4 + c * 128u + col4]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint sb = sb0 + c * 4u;
            for (uint q = 0; q < CQ; ++q) {
              if (o0 + q0 + q >= j.outs) break;
              const uint o = j.out0 + o0 + q0 + q;
              const uint4 qw = *reinterpret_cast<const device uint4*>(W + j.w + ulong(o) * j.in_total + j.in0 + c * 512u + lane * 16u);
              const float s = S[ulong(o / 128u) * sb_cols + sb] * 256.0f;
              const half4 w0 = tf_fp8x4_h(qw.x), w1 = tf_fp8x4_h(qw.y), w2 = tf_fp8x4_h(qw.z), w3 = tf_fp8x4_h(qw.w);
              for (uint r = 0; r < R; ++r) {
                if (r >= rows) break;
                const uint xb = r * 128u + lane * 4u;
                acc[q][r] = fma(float(dot(w0, xh4[xb])) + float(dot(w1, xh4[xb + 1u])) +
                                float(dot(w2, xh4[xb + 2u])) + float(dot(w3, xh4[xb + 3u])), s, acc[q][r]);
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }
        }
      } else if (j.format == TF_BF16) {
        const uint c256 = j.ins / 256u;
        if (R == 1u) {
          for (uint c = 0; c < c256; ++c) {
            const uint x_off = c * 64u + lane * 2u;
            const float4 x0 = X4[x_off], x1 = X4[x_off + 1u];
            for (uint q = 0; q < CQ; ++q) {
              if (o0 + q0 + q >= j.outs) break;
              const uint o = j.out0 + o0 + q0 + q;
              const uint4 qw = *reinterpret_cast<const device uint4*>(W + j.w + 2ul * (ulong(o) * j.in_total + j.in0 + c * 256u + lane * 8u));
              acc[q][0] += dot(tf_bf16x4(qw.x, qw.y), x0) + dot(tf_bf16x4(qw.z, qw.w), x1);
            }
          }
        } else {
          threadgroup float4* xf4 = reinterpret_cast<threadgroup float4*>(xs);
          for (uint c = 0; c < c256; ++c) {
            for (uint idx = tid; idx < rows * 64u; idx += 256u) {
              const uint rr = idx >> 6, col4 = idx & 63u;
              xf4[idx] = X4[ulong(rr) * x_s4 + c * 64u + col4];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint q = 0; q < CQ; ++q) {
              if (o0 + q0 + q >= j.outs) break;
              const uint o = j.out0 + o0 + q0 + q;
              const uint4 qw = *reinterpret_cast<const device uint4*>(W + j.w + 2ul * (ulong(o) * j.in_total + j.in0 + c * 256u + lane * 8u));
              const float4 w0 = tf_bf16x4(qw.x, qw.y), w1 = tf_bf16x4(qw.z, qw.w);
              for (uint r = 0; r < R; ++r) {
                if (r >= rows) break;
                const uint xb = r * 64u + lane * 2u;
                acc[q][r] += dot(w0, xf4[xb]) + dot(w1, xf4[xb + 1u]);
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }
        }
      } else {
        const uint c128 = j.ins / 128u;
        const ulong in_s4 = ulong(j.in_total >> 2), in0_4 = ulong(j.in0 >> 2);
        const device float4* W4 = reinterpret_cast<const device float4*>(W + j.w);
        for (uint c = 0; c < c128; ++c) {
          const uint x_off = c * 32u + lane;
          for (uint q = 0; q < CQ; ++q) {
            if (o0 + q0 + q >= j.outs) break;
            const float4 wv = W4[ulong(j.out0 + o0 + q0 + q) * in_s4 + in0_4 + x_off];
            for (uint r = 0; r < R; ++r) if (r < rows) acc[q][r] += dot(wv, X4[ulong(r) * x_s4 + x_off]);
          }
        }
      }
      for (uint q = 0; q < CQ; ++q) {
        if (o0 + q0 + q >= j.outs) break;
        for (uint r = 0; r < R; ++r) {
          const float v = tf_dense_lanes(acc[q][r]);
          if (lane != 0 || r >= rows) continue;
          const ulong at = ulong(r) * a.y_stride + j.y_col + o0 + q0 + q;
          if (a.y_f32) reinterpret_cast<device float*>(Y)[at] = v;
          else reinterpret_cast<device bfloat*>(Y)[at] = bfloat(v);
        }
      }
    }
    return;
  }
  float acc[Q][R];
  for (uint q = 0; q < Q; ++q) for (uint r = 0; r < R; ++r) acc[q][r] = 0.0f;
  for (uint c = 0; c < j.ins / 128; ++c) {
    for (uint i = tid; i < R * 128; i += 256)
      xs[i] = i / 128 < rows ? X[ulong(i / 128) * a.x_stride + j.x_col + 128 * c + i % 128] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = 0; q < Q; ++q) {
      if (o0 + q >= j.outs) break;
      float w[4];
      tf_dense_weights(W, j, j.out0 + o0 + q, c, lane, w);
      for (uint r = 0; r < R; ++r) {
        const float4 x = *reinterpret_cast<const threadgroup float4*>(xs + r * 128 + 4 * lane);
        acc[q][r] = fma(w[3], x.w, fma(w[2], x.z, fma(w[1], x.y, fma(w[0], x.x, acc[q][r]))));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint q = 0; q < Q; ++q) {
    if (o0 + q >= j.outs) break;
    for (uint r = 0; r < R; ++r) {
      const float v = tf_dense_lanes(acc[q][r]);
      if (lane != 0 || r >= rows) continue;
      const ulong at = ulong(r) * a.y_stride + j.y_col + o0 + q;
      if (a.y_f32) reinterpret_cast<device float*>(Y)[at] = v;
      else reinterpret_cast<device bfloat*>(Y)[at] = bfloat(v);
    }
  }
}

// One kernel a row capacity and rows a simdgroup; groups[g] = (job, 8 Q-row block of that job) for threadgroup g.
#define TF_EXLR8_DENSE(R, Q, NAME)                                                                                       \
  kernel void NAME(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],                            \
                   const device TfDenseJob* jobs [[buffer(2)]], const device uint2* groups [[buffer(3)]],               \
                   constant TfDenseRows& a [[buffer(4)]], device uchar* Y [[buffer(5)]],                                \
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                  \
                   uint tid [[thread_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {                 \
    threadgroup float xs[R * 256];                                                                                       \
    const uint2 at = groups[g];                                                                                        \
    tf_dense_job<R, Q>(X, W, jobs[at.x], at.y, a, Y, sg, lane, tid, xs);                                               \
  }
TF_EXLR8_DENSE(1, 4, tf_exlr8_dense_1)
TF_EXLR8_DENSE(2, 4, tf_exlr8_dense_2)
TF_EXLR8_DENSE(4, 4, tf_exlr8_dense_4)
TF_EXLR8_DENSE(8, 4, tf_exlr8_dense_8)
TF_EXLR8_DENSE(16, 4, tf_exlr8_dense_16)
TF_EXLR8_DENSE(1, 1, tf_exlr8_dense_narrow_1)
TF_EXLR8_DENSE(2, 1, tf_exlr8_dense_narrow_2)
TF_EXLR8_DENSE(4, 1, tf_exlr8_dense_narrow_4)

// Fast R=1 Q=2 (2 sgs = 64t, 4 outs/TG) direct-X zero-barrier GEMV: optimal on cold DRAM for x_g, q_b, wq_b!
kernel void tf_exlr8_dense_q2_1(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
                                const device TfDenseJob* jobs [[buffer(2)]], const device uint2* groups [[buffer(3)]],
                                constant TfDenseRows& a [[buffer(4)]], device uchar* Y [[buffer(5)]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                uint g [[threadgroup_position_in_grid]]) {
  const uint2 at = groups[g >> 1];
  const TfDenseJob j = jobs[at.x];
  const uint o0 = 8u * at.y + 4u * (g & 1u) + 2u * sg;
  if (o0 >= j.outs) return;
  const bool active1 = (o0 + 1u) < j.outs;
  float a0 = 0.0f, a1 = 0.0f;
  if ((j.in0 & 127u) == 0u && (j.x_col & 3u) == 0u && (j.ins % 512u) == 0u &&
      (j.format == TF_FP8_BLOCK || j.format == TF_BF16 || j.format == TF_FP32)) {
    const device float4* X4 = reinterpret_cast<const device float4*>(X + j.x_col);
    if (j.format == TF_FP8_BLOCK) {
      const uint c512 = j.ins / 512u, sb_cols = (j.in_total + 127u) / 128u, sub_sb = lane >> 3;
      const device float* S = reinterpret_cast<const device float*>(W + j.s);
      const uint og0 = j.out0 + o0;
      const ulong row0 = j.w + ulong(og0) * j.in_total + j.in0, row1 = row0 + j.in_total;
      const ulong sc0 = ulong(og0 >> 7) * sb_cols + (j.in0 >> 7), sc1 = ulong((og0 + 1u) >> 7) * sb_cols + (j.in0 >> 7);
      for (uint c = 0u; c < c512; ++c) {
        const uint xo = c * 128u + lane * 4u, sb = c * 4u + sub_sb;
        const half4 x0 = half4(X4[xo]), x1 = half4(X4[xo + 1u]), x2 = half4(X4[xo + 2u]), x3 = half4(X4[xo + 3u]);
        const float s0 = S[sc0 + sb] * 256.0f;
        const uint4 q0 = *reinterpret_cast<const device uint4*>(W + row0 + c * 512u + lane * 16u);
        a0 = fma(float(dot(tf_fp8x4_h(q0.x), x0)) + float(dot(tf_fp8x4_h(q0.y), x1)) +
                 float(dot(tf_fp8x4_h(q0.z), x2)) + float(dot(tf_fp8x4_h(q0.w), x3)), s0, a0);
        if (active1) {
          const float s1 = (sc1 == sc0) ? s0 : (S[sc1 + sb] * 256.0f);
          const uint4 q1 = *reinterpret_cast<const device uint4*>(W + row1 + c * 512u + lane * 16u);
          a1 = fma(float(dot(tf_fp8x4_h(q1.x), x0)) + float(dot(tf_fp8x4_h(q1.y), x1)) +
                   float(dot(tf_fp8x4_h(q1.z), x2)) + float(dot(tf_fp8x4_h(q1.w), x3)), s1, a1);
        }
      }
    } else if (j.format == TF_BF16) {
      const uint c256 = j.ins / 256u;
      const ulong r0 = j.w + 2ul * (ulong(j.out0 + o0) * j.in_total + j.in0), r1 = r0 + 2ul * j.in_total;
      for (uint c = 0u; c < c256; ++c) {
        const uint xo = c * 64u + lane * 2u;
        const float4 x0 = X4[xo], x1 = X4[xo + 1u];
        const uint4 q0 = *reinterpret_cast<const device uint4*>(W + r0 + 2ul * (c * 256u + lane * 8u));
        a0 += dot(tf_bf16x4(q0.x, q0.y), x0) + dot(tf_bf16x4(q0.z, q0.w), x1);
        if (active1) {
          const uint4 q1 = *reinterpret_cast<const device uint4*>(W + r1 + 2ul * (c * 256u + lane * 8u));
          a1 += dot(tf_bf16x4(q1.x, q1.y), x0) + dot(tf_bf16x4(q1.z, q1.w), x1);
        }
      }
    } else {
      const uint c128 = j.ins / 128u;
      const ulong in_s4 = ulong(j.in_total >> 2), in0_4 = ulong(j.in0 >> 2);
      const device float4* W4 = reinterpret_cast<const device float4*>(W + j.w);
      for (uint c = 0u; c < c128; ++c) {
        const uint xo = c * 32u + lane;
        const float4 xv = X4[xo];
        a0 += dot(W4[ulong(j.out0 + o0) * in_s4 + in0_4 + xo], xv);
        if (active1) a1 += dot(W4[ulong(j.out0 + o0 + 1u) * in_s4 + in0_4 + xo], xv);
      }
    }
  } else {
    for (uint c = 0u; c < j.ins / 128u; ++c) {
      const float4 xv = *reinterpret_cast<const device float4*>(X + j.x_col + 128u * c + 4u * lane);
      float w0[4]; tf_dense_weights(W, j, j.out0 + o0, c, lane, w0);
      a0 = fma(w0[3], xv.w, fma(w0[2], xv.z, fma(w0[1], xv.y, fma(w0[0], xv.x, a0))));
      if (active1) {
        float w1[4]; tf_dense_weights(W, j, j.out0 + o0 + 1u, c, lane, w1);
        a1 = fma(w1[3], xv.w, fma(w1[2], xv.z, fma(w1[1], xv.y, fma(w1[0], xv.x, a1))));
      }
    }
  }
  const float v0 = tf_dense_lanes(a0), v1 = active1 ? tf_dense_lanes(a1) : 0.0f;
  if (lane != 0u) return;
  const ulong y0 = ulong(j.y_col + o0);
  if (a.y_f32) {
    reinterpret_cast<device float*>(Y)[y0] = v0;
    if (active1) reinterpret_cast<device float*>(Y)[y0 + 1u] = v1;
  } else {
    reinterpret_cast<device bfloat*>(Y)[y0] = bfloat(v0);
    if (active1) reinterpret_cast<device bfloat*>(Y)[y0 + 1u] = bfloat(v1);
  }
}

// Fast R=2..6 (128t/256t, 8 or 16 outs/TG) spill-free GEMV w/ fused-X Q=2 reuse & optional peer BF16 pack.
template <uint R, uint Q, bool PACK, uint SGS>
inline void tf_dense_tg_impl(const device float* X, const device uchar* W, TfDenseJob j, uint block,
                             constant TfDenseRows& a, device uchar* Y, uint4 pack, device uint* send_base,
                             uint sg, uint lane, uint tid, threadgroup float* xs) {
  constexpr uint TH = SGS * 32u, OUTS_TG = Q * SGS;
  const uint rows = min(a.rows, R), o0 = OUTS_TG * block + Q * sg;
  const bool active0 = o0 < j.outs, active1 = (Q == 2u) && ((o0 + 1u) < j.outs);
  float a0[R], a1[R];
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < R; ++r) a0[r] = 0.0f, a1[r] = 0.0f;
  if ((j.in0 & 127u) == 0u && (j.x_col & 3u) == 0u && (a.x_stride & 3u) == 0u && (j.ins % 512u) == 0u &&
      (j.format == TF_FP8_BLOCK || j.format == TF_BF16 || j.format == TF_FP32)) {
    const device float4* X4 = reinterpret_cast<const device float4*>(X + j.x_col);
    const ulong x_s4 = ulong(a.x_stride >> 2);
    if (j.format == TF_FP8_BLOCK) {
      threadgroup half4* xh4 = reinterpret_cast<threadgroup half4*>(xs);
      const uint c512 = j.ins / 512u, sb_cols = (j.in_total + 127u) / 128u, sub_sb = lane >> 3;
      const device float* S = reinterpret_cast<const device float*>(W + j.s);
      const uint og0 = j.out0 + o0;
      const ulong row0 = j.w + ulong(og0) * j.in_total + j.in0, row1 = row0 + j.in_total;
      const ulong sc0 = ulong(og0 >> 7) * sb_cols + (j.in0 >> 7), sc1 = ulong((og0 + 1u) >> 7) * sb_cols + (j.in0 >> 7);
      for (uint c = 0u; c < c512; ++c) {
        for (uint idx = tid; idx < R * 128u; idx += TH) {
          const uint rr = idx >> 7, col4 = idx & 127u;
          xh4[idx] = (rr < rows) ? half4(X4[ulong(rr) * x_s4 + c * 128u + col4]) : half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active0) {
          const uint sb = c * 4u + sub_sb;
          const float s0 = S[sc0 + sb] * 256.0f;
          const uint4 q0 = *reinterpret_cast<const device uint4*>(W + row0 + c * 512u + lane * 16u);
          const half4 w00 = tf_fp8x4_h(q0.x), w01 = tf_fp8x4_h(q0.y), w02 = tf_fp8x4_h(q0.z), w03 = tf_fp8x4_h(q0.w);
          const float s1 = active1 ? ((sc1 == sc0) ? s0 : (S[sc1 + sb] * 256.0f)) : 0.0f;
          const uint4 q1 = active1 ? *reinterpret_cast<const device uint4*>(W + row1 + c * 512u + lane * 16u) : uint4(0u);
          const half4 w10 = tf_fp8x4_h(q1.x), w11 = tf_fp8x4_h(q1.y), w12 = tf_fp8x4_h(q1.z), w13 = tf_fp8x4_h(q1.w);
          #pragma clang loop unroll(full)
          for (uint r = 0u; r < R; ++r) {
            const uint xb = r * 128u + lane * 4u;
            const half4 x0 = xh4[xb], x1 = xh4[xb + 1u], x2 = xh4[xb + 2u], x3 = xh4[xb + 3u];
            a0[r] = fma(float(dot(w00, x0)) + float(dot(w01, x1)) + float(dot(w02, x2)) + float(dot(w03, x3)), s0, a0[r]);
            if (active1) a1[r] = fma(float(dot(w10, x0)) + float(dot(w11, x1)) + float(dot(w12, x2)) + float(dot(w13, x3)), s1, a1[r]);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    } else if (j.format == TF_BF16) {
      if (active0) {
        const ulong r0 = j.w + 2ul * (ulong(j.out0 + o0) * j.in_total + j.in0);
        const ulong r1 = r0 + 2ul * j.in_total;
        for (uint c = 0u; c < j.ins / 256u; ++c) {
          const uint off = 2u * (c * 256u + lane * 8u), xo = c * 64u + lane * 2u;
          const uint4 q0 = *reinterpret_cast<const device uint4*>(W + r0 + off);
          const float4 w00 = tf_bf16x4(q0.x, q0.y), w01 = tf_bf16x4(q0.z, q0.w);
          const uint4 q1 = active1 ? *reinterpret_cast<const device uint4*>(W + r1 + off) : uint4(0u);
          const float4 w10 = tf_bf16x4(q1.x, q1.y), w11 = tf_bf16x4(q1.z, q1.w);
          #pragma clang loop unroll(full)
          for (uint r = 0u; r < R; ++r) {
            const ulong rx = ulong(min(r, rows - 1u)) * x_s4 + xo;
            const float4 x0 = X4[rx], x1 = X4[rx + 1u];
            a0[r] += dot(w00, x0) + dot(w01, x1);
            if (active1) a1[r] += dot(w10, x0) + dot(w11, x1);
          }
        }
      }
    } else if (active0) {
      const ulong in_s4 = ulong(j.in_total >> 2), in0_4 = ulong(j.in0 >> 2);
      const device float4* W4 = reinterpret_cast<const device float4*>(W + j.w);
      for (uint c = 0u; c < j.ins / 128u; ++c) {
        const uint xo = c * 32u + lane;
        const float4 w0 = W4[ulong(j.out0 + o0) * in_s4 + in0_4 + xo];
        const float4 w1 = active1 ? W4[ulong(j.out0 + o0 + 1u) * in_s4 + in0_4 + xo] : float4(0.0f);
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < R; ++r) {
          const float4 xv = X4[ulong(min(r, rows - 1u)) * x_s4 + xo];
          a0[r] += dot(w0, xv);
          if (active1) a1[r] += dot(w1, xv);
        }
      }
    }
  } else {
    for (uint c = 0u; c < j.ins / 128u; ++c) {
      for (uint i = tid; i < R * 128u; i += TH)
        xs[i] = i / 128u < rows ? X[ulong(i / 128u) * a.x_stride + j.x_col + 128u * c + i % 128u] : 0.0f;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (active0) {
        float w0[4]; tf_dense_weights(W, j, j.out0 + o0, c, lane, w0);
        float w1[4]; if (active1) tf_dense_weights(W, j, j.out0 + o0 + 1u, c, lane, w1);
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < R; ++r) {
          const float4 x = *reinterpret_cast<const threadgroup float4*>(xs + r * 128u + 4u * lane);
          a0[r] = fma(w0[3], x.w, fma(w0[2], x.z, fma(w0[1], x.y, fma(w0[0], x.x, a0[r]))));
          if (active1) a1[r] = fma(w1[3], x.w, fma(w1[2], x.z, fma(w1[1], x.y, fma(w1[0], x.x, a1[r]))));
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  const ulong st_u1 = ulong(pack.z) * 2ul;
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < R; ++r) {
    if (r >= rows) continue;
    const float v0 = active0 ? tf_dense_lanes(a0[r]) : 0.0f;
    const float v1 = active1 ? tf_dense_lanes(a1[r]) : 0.0f;
    if (active0 && lane == 0u) {
      const ulong at0 = ulong(r) * a.y_stride + j.y_col + o0;
      if (a.y_f32) {
        reinterpret_cast<device float*>(Y)[at0] = v0;
        if (active1) reinterpret_cast<device float*>(Y)[at0 + 1u] = v1;
        if (PACK && Q == 2u && pack.y > 1u && active1) {
          const uint pk = tf_bf16x2(v0, v1);
          for (uint p = 0u; p < pack.y; ++p) if (p != pack.x) send_base[ulong(p) * st_u1 + (at0 >> 1)] = pk;
        }
      } else {
        reinterpret_cast<device bfloat*>(Y)[at0] = bfloat(v0);
        if (active1) reinterpret_cast<device bfloat*>(Y)[at0 + 1u] = bfloat(v1);
      }
    }
  }
}

#define TF_EXLR8_TG(R, Q, SGS, NAME)                                                                                     \
  kernel void NAME(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],                             \
                   const device TfDenseJob* jobs [[buffer(2)]], const device uint2* groups [[buffer(3)]],                \
                   constant TfDenseRows& a [[buffer(4)]], device uchar* Y [[buffer(5)]],                                 \
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                  \
                   uint tid [[thread_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {                  \
    threadgroup float xs[R * 256];                                                                                       \
    const uint2 at = groups[g];                                                                                          \
    tf_dense_tg_impl<R, Q, false, SGS>(X, W, jobs[at.x], at.y, a, Y, uint4(0u), nullptr, sg, lane, tid, xs);             \
  }
TF_EXLR8_TG(2, 2, 8, tf_exlr8_dense_tg2_2)
TF_EXLR8_TG(3, 2, 8, tf_exlr8_dense_tg2_3)
TF_EXLR8_TG(4, 1, 8, tf_exlr8_dense_tg2_4)
TF_EXLR8_TG(5, 1, 8, tf_exlr8_dense_tg2_5)
TF_EXLR8_TG(6, 1, 8, tf_exlr8_dense_tg2_6)
TF_EXLR8_TG(4, 2, 8, tf_exlr8_dense_tg2_q2_4)
TF_EXLR8_TG(5, 2, 8, tf_exlr8_dense_tg2_q2_5)
TF_EXLR8_TG(6, 2, 8, tf_exlr8_dense_tg2_q2_6)

#define TF_EXLR8_PACK(R, Q, SGS, NAME)                                                                                   \
  kernel void NAME(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],                             \
                   const device TfDenseJob* jobs [[buffer(2)]], const device uint2* groups [[buffer(3)]],                \
                   constant TfDenseRows& a [[buffer(4)]], device uchar* Y [[buffer(5)]],                                 \
                   constant uint4& pack [[buffer(6)]], device uint* send_base [[buffer(7)]],                             \
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                  \
                   uint tid [[thread_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {                  \
    threadgroup float xs[R * 256];                                                                                       \
    const uint2 at = groups[g];                                                                                          \
    tf_dense_tg_impl<R, Q, true, SGS>(X, W, jobs[at.x], at.y, a, Y, pack, send_base, sg, lane, tid, xs);                 \
  }
TF_EXLR8_PACK(2, 2, 8, tf_exlr8_dense_pack_bf16_2)
TF_EXLR8_PACK(3, 2, 8, tf_exlr8_dense_pack_bf16_3)
TF_EXLR8_PACK(4, 2, 8, tf_exlr8_dense_pack_bf16_4)
TF_EXLR8_PACK(5, 2, 8, tf_exlr8_dense_pack_bf16_5)
TF_EXLR8_PACK(6, 2, 8, tf_exlr8_dense_pack_bf16_6)

// Fast R=1 Q=4 (128t, 16 outs/TG via groups16) zero-barrier direct-X FP8 GEMV w/ float4 Y + uint2 peer BF16 pack!
kernel void tf_exlr8_dense_1_pack_bf16(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
                                       const device TfDenseJob* jobs [[buffer(2)]], const device uint2* groups [[buffer(3)]],
                                       constant TfDenseRows& a [[buffer(4)]], device float* Y [[buffer(5)]],
                                       constant uint4& pack [[buffer(6)]], device uint2* send_base [[buffer(7)]],
                                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                       uint g [[threadgroup_position_in_grid]]) {
  (void)a;
  const uint2 at = groups[g];
  const TfDenseJob j = jobs[at.x];
  const uint o0 = 16u * at.y + 4u * sg;
  if (o0 >= j.outs) return;
  const bool full4 = (o0 + 3u) < j.outs;
  const device float4* X4 = reinterpret_cast<const device float4*>(X + j.x_col);
  const uint c512 = j.ins / 512u, sb_cols = (j.in_total + 127u) / 128u, sub_sb = lane >> 3;
  const device float* S = reinterpret_cast<const device float*>(W + j.s);
  const uint og0 = j.out0 + o0;
  const ulong row0 = j.w + ulong(og0) * j.in_total + j.in0;
  const ulong sc0 = ulong(og0 >> 7) * sb_cols + (j.in0 >> 7);
  const ulong sc3 = ulong((og0 + 3u) >> 7) * sb_cols + (j.in0 >> 7);
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  for (uint c = 0u; c < c512; ++c) {
    const uint xo = c * 128u + lane * 4u, sb = c * 4u + sub_sb;
    const half4 x0 = half4(X4[xo]), x1 = half4(X4[xo + 1u]), x2 = half4(X4[xo + 2u]), x3 = half4(X4[xo + 3u]);
    const float s0 = S[sc0 + sb] * 256.0f;
    const uint4 q0 = *reinterpret_cast<const device uint4*>(W + row0 + c * 512u + lane * 16u);
    acc0 = fma(float(dot(tf_fp8x4_h(q0.x), x0)) + float(dot(tf_fp8x4_h(q0.y), x1)) +
               float(dot(tf_fp8x4_h(q0.z), x2)) + float(dot(tf_fp8x4_h(q0.w), x3)), s0, acc0);
    if (full4) {
      const float s1 = (sc3 == sc0) ? s0 : (S[ulong((og0 + 1u) >> 7) * sb_cols + (j.in0 >> 7) + sb] * 256.0f);
      const float s2 = (sc3 == sc0) ? s0 : (S[ulong((og0 + 2u) >> 7) * sb_cols + (j.in0 >> 7) + sb] * 256.0f);
      const float s3 = (sc3 == sc0) ? s0 : (S[sc3 + sb] * 256.0f);
      const uint4 q1 = *reinterpret_cast<const device uint4*>(W + row0 + j.in_total + c * 512u + lane * 16u);
      const uint4 q2 = *reinterpret_cast<const device uint4*>(W + row0 + 2ul * j.in_total + c * 512u + lane * 16u);
      const uint4 q3 = *reinterpret_cast<const device uint4*>(W + row0 + 3ul * j.in_total + c * 512u + lane * 16u);
      acc1 = fma(float(dot(tf_fp8x4_h(q1.x), x0)) + float(dot(tf_fp8x4_h(q1.y), x1)) +
                 float(dot(tf_fp8x4_h(q1.z), x2)) + float(dot(tf_fp8x4_h(q1.w), x3)), s1, acc1);
      acc2 = fma(float(dot(tf_fp8x4_h(q2.x), x0)) + float(dot(tf_fp8x4_h(q2.y), x1)) +
                 float(dot(tf_fp8x4_h(q2.z), x2)) + float(dot(tf_fp8x4_h(q2.w), x3)), s2, acc2);
      acc3 = fma(float(dot(tf_fp8x4_h(q3.x), x0)) + float(dot(tf_fp8x4_h(q3.y), x1)) +
                 float(dot(tf_fp8x4_h(q3.z), x2)) + float(dot(tf_fp8x4_h(q3.w), x3)), s3, acc3);
    }
  }
  const float v0 = tf_dense_lanes(acc0), v1 = tf_dense_lanes(acc1), v2 = tf_dense_lanes(acc2), v3 = tf_dense_lanes(acc3);
  if (lane != 0u) return;
  const ulong at0 = ulong(j.y_col + o0);
  if (full4) {
    *reinterpret_cast<device float4*>(Y + at0) = float4(v0, v1, v2, v3);
    if (pack.y > 1u) {
      const uint2 pk = uint2(tf_bf16x2(v0, v1), tf_bf16x2(v2, v3));
      const ulong idx2 = at0 >> 2;
      for (uint p = 0u; p < pack.y; ++p) if (p != pack.x) send_base[ulong(p) * pack.z + idx2] = pk;
    }
  } else {
    Y[at0] = v0;
  }
}

// Qualification: rows [out0, out0 + outs) of job 0's matrix over its inputs as fp32, row-major, ins wide.
kernel void tf_exlr8_dense_decode(const device uchar* W [[buffer(0)]], const device TfDenseJob* jobs [[buffer(1)]],
                                  device float* out [[buffer(2)]], uint lane [[thread_index_in_simdgroup]],
                                  uint2 tg [[threadgroup_position_in_grid]]) {
  const TfDenseJob j = jobs[0];
  const uint o = tg.y, c = tg.x;
  if (o >= j.outs) return;
  float w[4];
  tf_dense_weights(W, j, j.out0 + o, c, lane, w);
  for (uint n = 0; n < 4; ++n) out[ulong(o) * j.ins + 128 * c + 4 * lane + n] = w[n];
}
