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
      const float s = reinterpret_cast<const device float*>(W + j.s)[at];
      // one rounding, as the reference's E4M3 times its fp32 scale
      for (uint n = 0; n < 4; ++n) w[n] = tf_e4m3((q >> (8 * n)) & 0xFFu) * s;
      break;
    }
    case TF_BF16: {
      const uint2 q = *reinterpret_cast<const device uint2*>(W + j.w + 2 * (ulong(o) * j.in_total + i));
      w[0] = as_type<float>(q.x << 16);
      w[1] = as_type<float>(q.x & 0xFFFF0000u);
      w[2] = as_type<float>(q.y << 16);
      w[3] = as_type<float>(q.y & 0xFFFF0000u);
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

// A threadgroup is 8 Q rows of one job (Q a simdgroup, the same sums whatever Q); inputs 128 at a time via xs.
template <uint R, uint Q>
inline void tf_dense_job(const device float* X, const device uchar* W, TfDenseJob j, uint block, constant TfDenseRows& a,
                         device uchar* Y, uint sg, uint lane, uint tid, threadgroup float* xs) {
  const uint rows = min(a.rows, R), o0 = 8 * Q * block + Q * sg;
  float acc[Q][R];
  for (uint q = 0; q < Q; ++q)
    for (uint r = 0; r < R; ++r) acc[q][r] = 0.0f;
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
    threadgroup float xs[R * 128];                                                                                     \
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
