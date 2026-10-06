// Causal prefill attention on the M5 tensor unit, bit-identical to MLX 0.32.3's steel_attention (NAX, bq64 bk32 d128).
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;
#include "../nax.h"

namespace tfp {

// MLX's Limits<float>::finite_min, the score of a masked key.
constant constexpr float kMasked = -3.402823466e+38f;
constant constexpr int kAnyCols = 1 << 30;

// Fold one fragment's rows into r[2] (rows home.y, home.y + 8): pairs, then lanes 1 and 8 apart, then into r.
template <bool MAX>
inline void row_fold(thread const frag<float>& f, thread float (&r)[2]) {
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const short b = 4 * h;
    float t = MAX ? max(max(f[b], f[b + 1]), max(f[b + 2], f[b + 3])) : (f[b] + f[b + 1]) + (f[b + 2] + f[b + 3]);
    const float u = simd_shuffle_xor(t, ushort(1));
    t = MAX ? max(t, u) : t + u;
    const float w = simd_shuffle_xor(t, ushort(8));
    t = MAX ? max(t, w) : t + w;
    r[h] = MAX ? max(r[h], t) : r[h] + t;
  }
}

// One 64-query block of one head, online softmax over 32-key blocks; P holds the ATTN_PARAMS fields, F[0] the scale.
template <typename T, bool AQ, bool AK, bool CAUSAL>
inline void attention_nax(const device T* Q, const device T* K, const device T* V, device T* O, const device int* P,
                          const device float* F, uint3 tg, uint sg, uint lane) {
  constexpr int BQ = 64, BK = 32;
  const int qs = P[14], ks = P[17], vs = P[20], os = P[23];
  const int kv = int(tg.y) / P[4];
  Q += long(tg.z) * P[12] + long(tg.y) * P[13] + long(tg.x) * BQ * qs;
  K += long(tg.z) * P[15] + long(kv) * P[16];
  V += long(tg.z) * P[18] + long(kv) * P[19];
  O += long(tg.z) * P[21] + long(tg.y) * P[22] + long(tg.x) * BQ * os;
  const float scale2 = F[0] * 1.44269504089f;
  const short tm = 16 * short(sg);
  Q += tm * qs;
  const short2 home = frag_home(ushort(lane));

  frag<float> acc[8];               // 16 query rows x 128 values
  TF_UNROLL
  for (short i = 0; i < 8; i++) {
    acc[i] = frag<float>(0);
  }
  float top[2] = {kMasked, kMasked}, total[2] = {0.0f, 0.0f};

  int blocks = P[6], first_masked = P[6];
  if (CAUSAL) {
    blocks = min(P[6], ((int(tg.x) + 1) * BQ + P[11] + BK - 1) / BK);
    first_masked = max(0, int(tg.x) * BQ + P[11]) / BK;
  }
  const bool last_q = int(tg.x) == P[7];
  const short q_rows = short(P[9] - tm), k_rows = short(P[10]);

  for (int kb = 0; kb < blocks; kb++) {
    const bool last_k = kb == P[8];
    frag<float> s[2] = {frag<float>(0), frag<float>(0)};
#pragma clang loop unroll_count(4)
    for (short d = 0; d < 8; d++) {
      frag<T> q, k0, k1;
      if (!AQ && last_q) {
        frag_get_in(q, Q, qs, 0, 16 * d, home, q_rows, kAnyCols);
      } else {
        frag_get(q, Q, qs, 0, 16 * d, home);
      }
      if (!AK && last_k) {
        frag_get_t_in(k0, K, ks, 0, 16 * d, home, k_rows, kAnyCols);
        frag_get_t_in(k1, K, ks, 16, 16 * d, home, k_rows, kAnyCols);
      } else {
        frag_get_t(k0, K, ks, 0, 16 * d, home);
        frag_get_t(k1, K, ks, 16, 16 * d, home);
      }
      mma_16x32<false, true>(s[0], s[1], q, k0, k1);
    }
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        s[f][e] *= scale2;
      }
    }
    if (!AK && last_k) {
      TF_UNROLL
      for (short f = 0; f < 2; f++) {
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          s[f][e] = (16 * f + home.x + TF_COL(e)) < k_rows ? s[f][e] : kMasked;
        }
      }
    }
    if (CAUSAL && kb >= first_masked) {
      const int r0 = int(tg.x) * BQ + P[11] + tm + home.y, c0 = kb * BK + home.x;
      TF_UNROLL
      for (short f = 0; f < 2; f++) {
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          s[f][e] = (r0 + (e >> 2) * 8) < (c0 + 16 * f + TF_COL(e)) ? kMasked : s[f][e];
        }
      }
    }

    float top_new[2] = {top[0], top[1]};
    row_fold<true>(s[0], top_new);
    row_fold<true>(s[1], top_new);
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        s[f][e] = fast::exp2(s[f][e] - top_new[e >> 2]);
      }
    }
    float scale_old[2];
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      scale_old[h] = fast::exp2(top[h] - top_new[h]);
      top[h] = top_new[h];
      total[h] = total[h] * scale_old[h];
    }
    row_fold<false>(s[0], total);
    row_fold<false>(s[1], total);
    TF_UNROLL
    for (short i = 0; i < 8; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        acc[i][e] = acc[i][e] * scale_old[e >> 2];
      }
    }
    simdgroup_barrier(mem_flags::mem_none);

    TF_UNROLL
    for (short d = 0; d < 8; d += 2) {
      if (d == 4) {
        threadgroup_barrier(mem_flags::mem_none);
      }
      TF_UNROLL
      for (short k = 0; k < 2; k++) {
        frag<T> v0, v1;
        if (!AK && last_k) {
          frag_get_in(v0, V, vs, 16 * k, 16 * d, home, k_rows, kAnyCols);
          frag_get_in(v1, V, vs, 16 * k, 16 * d + 16, home, k_rows, kAnyCols);
        } else {
          frag_get(v0, V, vs, 16 * k, 16 * d, home);
          frag_get(v1, V, vs, 16 * k, 16 * d + 16, home);
        }
        mma_16x32<false, false>(acc[d], acc[d + 1], s[k], v0, v1);
      }
    }
    K += BK * ks;
    V += BK * vs;
  }

  threadgroup_barrier(mem_flags::mem_none);
  float inv[2];
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    inv[h] = 1.f / total[h];
  }
  TF_UNROLL
  for (short i = 0; i < 8; i++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      acc[i][e] = acc[i][e] * inv[e >> 2];
    }
  }
  O += tm * os;
  if (!AQ && last_q) {
    if (q_rows <= 0) {
      return;
    }
    TF_UNROLL
    for (short i = 0; i < 8; i++) {
      frag_put_in(acc[i], O, os, 0, 16 * i, home, q_rows, kAnyCols);
    }
  } else {
    TF_UNROLL
    for (short i = 0; i < 8; i++) {
      frag_put(acc[i], O, os, 0, 16 * i, home);
    }
  }
}

}  // namespace tfp
// ---- entry points: generated by tools/zig/check_prefill_ops.py --write ----
[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void custom_kernel_tf_attention_nax_bf16_n_n_t_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_float_bfloat16_t(
  const device bfloat16_t* Q [[buffer(0)]],
  const device bfloat16_t* K [[buffer(1)]],
  const device bfloat16_t* V [[buffer(2)]],
  const device int32_t* P [[buffer(3)]],
  const device float* F [[buffer(4)]],
  device bfloat16_t* O [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::attention_nax<bfloat16_t, false, false, true>(Q, K, V, O, P, F, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

}
[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void custom_kernel_tf_attention_nax_bf16_n_t_t_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_float_bfloat16_t(
  const device bfloat16_t* Q [[buffer(0)]],
  const device bfloat16_t* K [[buffer(1)]],
  const device bfloat16_t* V [[buffer(2)]],
  const device int32_t* P [[buffer(3)]],
  const device float* F [[buffer(4)]],
  device bfloat16_t* O [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::attention_nax<bfloat16_t, false, true, true>(Q, K, V, O, P, F, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

}
[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void custom_kernel_tf_attention_nax_bf16_t_n_t_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_float_bfloat16_t(
  const device bfloat16_t* Q [[buffer(0)]],
  const device bfloat16_t* K [[buffer(1)]],
  const device bfloat16_t* V [[buffer(2)]],
  const device int32_t* P [[buffer(3)]],
  const device float* F [[buffer(4)]],
  device bfloat16_t* O [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::attention_nax<bfloat16_t, true, false, true>(Q, K, V, O, P, F, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

}
[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void custom_kernel_tf_attention_nax_bf16_t_t_t_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_float_bfloat16_t(
  const device bfloat16_t* Q [[buffer(0)]],
  const device bfloat16_t* K [[buffer(1)]],
  const device bfloat16_t* V [[buffer(2)]],
  const device int32_t* P [[buffer(3)]],
  const device float* F [[buffer(4)]],
  device bfloat16_t* O [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::attention_nax<bfloat16_t, true, true, true>(Q, K, V, O, P, F, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

}
