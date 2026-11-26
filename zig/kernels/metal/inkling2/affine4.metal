// Inkling 2 Mini MLX 4-bit affine group-64 Metal kernels: FP32 accumulation, no FMA contraction.
#include <metal_stdlib>
#pragma METAL fp contract(off)
using namespace metal;

constant uint INK2_Q4_HIDDEN = 2304u;
constant uint INK2_Q4_WORDS = 288u;
constant uint INK2_Q4_GROUPS = 36u;
constant float INK2_Q4_EPS = 1e-6f;
constant float INK2_Q4_CAP = 30.0f;

inline float ink2_q4_embed_val(const device uint* w_row, const device bfloat* s_row,
                               const device bfloat* b_row, uint d) {
  const uint q = (w_row[d >> 3u] >> ((d & 7u) * 4u)) & 0xfu;
  const uint g = d >> 6u;
  return float(bfloat(float(q) * float(s_row[g]) + float(b_row[g])));
}

inline float ink2_q4_acc_pair(float acc, float p0, float p1) {
  return (acc + p0) + p1;
}

inline float ink2_q4_softcap30(float dot) {
  const float v = float(bfloat(dot));
  return INK2_Q4_CAP * precise::tanh(precise::divide(v, INK2_Q4_CAP));
}

struct Ink2EmbedQ4Args { uint rows; };

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_embed_norm_q4(
    const device uint* ids [[buffer(0)]],
    const device uint* weight [[buffer(1)]],
    const device bfloat* scales [[buffer(2)]],
    const device bfloat* biases [[buffer(3)]],
    const device bfloat* norm_w [[buffer(4)]],
    device bfloat* out [[buffer(5)]],
    constant Ink2EmbedQ4Args& a [[buffer(6)]],
    uint t [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows) return;
  const size_t id = size_t(ids[row]);
  const device uint* w_row = weight + id * INK2_Q4_WORDS;
  const device bfloat* s_row = scales + id * INK2_Q4_GROUPS;
  const device bfloat* b_row = biases + id * INK2_Q4_GROUPS;
  float vals[9];
  float part = 0.0f;
  for (uint k = 0u; k < 9u; ++k) {
    const uint d = t + k * 256u;
    const float v = ink2_q4_embed_val(w_row, s_row, b_row, d);
    vals[k] = v;
    part += v * v;
  }
  threadgroup float red[8], inverse;
  const float reduced = simd_sum(part);
  if (lane == 0u) red[t >> 5u] = reduced;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0u) {
    float total = 0.0f;
    for (uint j = 0u; j < 8u; ++j) total += red[j];
    inverse = precise::rsqrt(precise::divide(total, float(INK2_Q4_HIDDEN)) + INK2_Q4_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 0u; k < 9u; ++k) {
    const uint d = t + k * 256u;
    out[size_t(row) * INK2_Q4_HIDDEN + d] = bfloat(float(norm_w[d]) * (vals[k] * inverse));
  }
}

struct Ink2DenseQ4Args { uint rows, n, k, ldy; };

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dense_q4(
    const device bfloat* x [[buffer(0)]],
    const device uint* weight [[buffer(1)]],
    const device bfloat* scales [[buffer(2)]],
    const device bfloat* biases [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    constant Ink2DenseQ4Args& a [[buffer(5)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  const uint col = tg.x * 8u + sg;
  const uint r0 = tg.y * 4u;
  if (col >= a.n || r0 >= a.rows || a.k == 0u || (a.k & 63u) != 0u) return;
  const uint groups = a.k >> 6u;
  const uint words_per_row = a.k >> 3u;
  const device uint* w_row = weight + size_t(col) * words_per_row;
  const device bfloat* s_row = scales + size_t(col) * groups;
  const device bfloat* b_row = biases + size_t(col) * groups;
  const uint word_in_group = lane >> 2u;
  const uint sh = (lane & 3u) * 8u;
  const uint active = min(4u, a.rows - r0);
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  for (uint g = 0u; g < groups; ++g) {
    const float sc = float(s_row[g]);
    const float bi = float(b_row[g]);
    const uint word = w_row[g * 8u + word_in_group];
    const float dw0 = float((word >> sh) & 0xfu) * sc + bi;
    const float dw1 = float((word >> (sh + 4u)) & 0xfu) * sc + bi;
    const uint k0 = g * 64u + lane * 2u;
    const device bfloat* x0 = x + size_t(r0) * a.k + k0;
    acc0 = ink2_q4_acc_pair(acc0, float(x0[0]) * dw0, float(x0[1]) * dw1);
    if (active > 1u) {
      const device bfloat* x1 = x + size_t(r0 + 1u) * a.k + k0;
      acc1 = ink2_q4_acc_pair(acc1, float(x1[0]) * dw0, float(x1[1]) * dw1);
    }
    if (active > 2u) {
      const device bfloat* x2 = x + size_t(r0 + 2u) * a.k + k0;
      acc2 = ink2_q4_acc_pair(acc2, float(x2[0]) * dw0, float(x2[1]) * dw1);
    }
    if (active > 3u) {
      const device bfloat* x3 = x + size_t(r0 + 3u) * a.k + k0;
      acc3 = ink2_q4_acc_pair(acc3, float(x3[0]) * dw0, float(x3[1]) * dw1);
    }
  }
  const float s0 = simd_sum(acc0);
  const float s1 = active > 1u ? simd_sum(acc1) : 0.0f;
  const float s2 = active > 2u ? simd_sum(acc2) : 0.0f;
  const float s3 = active > 3u ? simd_sum(acc3) : 0.0f;
  if (lane == 0u) {
    y[size_t(r0) * a.ldy + col] = bfloat(s0);
    if (active > 1u) y[size_t(r0 + 1u) * a.ldy + col] = bfloat(s1);
    if (active > 2u) y[size_t(r0 + 2u) * a.ldy + col] = bfloat(s2);
    if (active > 3u) y[size_t(r0 + 3u) * a.ldy + col] = bfloat(s3);
  }
}

struct Ink2LogitsQ4Args { uint rows, vocab; };

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_logits_q4(
    const device bfloat* h [[buffer(0)]],
    const device uint* weight [[buffer(1)]],
    const device bfloat* scales [[buffer(2)]],
    const device bfloat* biases [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant Ink2LogitsQ4Args& a [[buffer(5)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  const uint v = tg.x * 8u + sg;
  const uint r0 = tg.y * 4u;
  if (v >= a.vocab || r0 >= a.rows) return;
  const device uint* w_row = weight + size_t(v) * INK2_Q4_WORDS;
  const device bfloat* s_row = scales + size_t(v) * INK2_Q4_GROUPS;
  const device bfloat* b_row = biases + size_t(v) * INK2_Q4_GROUPS;
  const uint word_in_group = lane >> 2u;
  const uint sh = (lane & 3u) * 8u;
  const uint active = min(4u, a.rows - r0);
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  for (uint g = 0u; g < INK2_Q4_GROUPS; ++g) {
    const float sc = float(s_row[g]);
    const float bi = float(b_row[g]);
    const uint word = w_row[g * 8u + word_in_group];
    const float lw0 = float((word >> sh) & 0xfu) * sc + bi;
    const float lw1 = float((word >> (sh + 4u)) & 0xfu) * sc + bi;
    const uint k0 = g * 64u + lane * 2u;
    const device bfloat* h0 = h + size_t(r0) * INK2_Q4_HIDDEN + k0;
    acc0 = (acc0 + float(h0[0]) * lw0) + float(h0[1]) * lw1;
    if (active > 1u) {
      const device bfloat* h1 = h + size_t(r0 + 1u) * INK2_Q4_HIDDEN + k0;
      acc1 = (acc1 + float(h1[0]) * lw0) + float(h1[1]) * lw1;
    }
    if (active > 2u) {
      const device bfloat* h2 = h + size_t(r0 + 2u) * INK2_Q4_HIDDEN + k0;
      acc2 = (acc2 + float(h2[0]) * lw0) + float(h2[1]) * lw1;
    }
    if (active > 3u) {
      const device bfloat* h3 = h + size_t(r0 + 3u) * INK2_Q4_HIDDEN + k0;
      acc3 = (acc3 + float(h3[0]) * lw0) + float(h3[1]) * lw1;
    }
  }
  const float s0 = simd_sum(acc0);
  const float s1 = active > 1u ? simd_sum(acc1) : 0.0f;
  const float s2 = active > 2u ? simd_sum(acc2) : 0.0f;
  const float s3 = active > 3u ? simd_sum(acc3) : 0.0f;
  if (lane == 0u) {
    out[size_t(r0) * a.vocab + v] = ink2_q4_softcap30(s0);
    if (active > 1u) out[size_t(r0 + 1u) * a.vocab + v] = ink2_q4_softcap30(s1);
    if (active > 2u) out[size_t(r0 + 2u) * a.vocab + v] = ink2_q4_softcap30(s2);
    if (active > 3u) out[size_t(r0 + 3u) * a.vocab + v] = ink2_q4_softcap30(s3);
  }
}
