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
      const float s = j.format == TF_MXFP8 ? as_type<float>(uint(W[j.s + ulong(o) * (j.in_total / 32) + i / 32]) << 23) : 1.0f;
      for (uint n = 0; n < 8; ++n) w[n] = tf_e4m3(((n < 4 ? q.x : q.y) >> (8 * (n % 4))) & 0xFFu) * s;
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
