// Idol's attention: tfp::attention_nax's online softmax at head width 64, every query row to every key row.

constant constexpr float kIdMasked = -3.402823466e+38f;
constant constexpr int kIdAnyCols = 1 << 30;

// Fold one fragment's rows into r[2] (rows home.y, home.y + 8): pairs, then lanes 1 and 8 apart, then into r.
template <bool MAX>
inline void id_fold(thread const tfp::frag<float>& f, thread float (&r)[2]) {
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

// One 64-query block of one head over 32-key blocks; P and F as Kernels.attention sets them (kernels.zig).
template <typename T, bool AQ, bool AK>
inline void id_attention(const device T* Q, const device T* K, const device T* V, device T* O, const device int* P,
                         const device float* F, uint3 tg, uint sg, uint lane) {
  constexpr int BQ = 64, BK = 32, D = 64;
  const int h = int(tg.y), lq = P[1], lk = P[2];
  Q += (long(h) * lq + long(tg.x) * BQ) * D;
  K += long(h) * lk * D;
  V += long(h) * lk * D;
  const int os = P[0] * D;
  O += long(tg.x) * BQ * os + h * D;
  const float scale2 = F[0] * 1.44269504089f;
  const short tm = 16 * short(sg);
  Q += tm * D;
  const short2 home = tfp::frag_home(ushort(lane));

  tfp::frag<float> acc[4];
  TF_UNROLL
  for (short i = 0; i < 4; i++) acc[i] = tfp::frag<float>(0);
  float top[2] = {kIdMasked, kIdMasked}, total[2] = {0.0f, 0.0f};
  const bool last_q = int(tg.x) == P[4];
  const short q_rows = short(P[6] - tm), k_rows = short(P[7]);

  for (int kb = 0; kb < P[3]; kb++) {
    const bool last_k = kb == P[5];
    tfp::frag<float> s[2] = {tfp::frag<float>(0), tfp::frag<float>(0)};
    TF_UNROLL
    for (short d = 0; d < 4; d++) {
      tfp::frag<T> q, k0, k1;
      if (!AQ && last_q) {
        tfp::frag_get_in(q, Q, D, 0, 16 * d, home, q_rows, kIdAnyCols);
      } else {
        tfp::frag_get(q, Q, D, 0, 16 * d, home);
      }
      if (!AK && last_k) {
        tfp::frag_get_t_in(k0, K, D, 0, 16 * d, home, k_rows, kIdAnyCols);
        tfp::frag_get_t_in(k1, K, D, 16, 16 * d, home, k_rows, kIdAnyCols);
      } else {
        tfp::frag_get_t(k0, K, D, 0, 16 * d, home);
        tfp::frag_get_t(k1, K, D, 16, 16 * d, home);
      }
      tfp::mma_16x32<false, true>(s[0], s[1], q, k0, k1);
    }
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) s[f][e] *= scale2;
    }
    if (!AK && last_k) {
      TF_UNROLL
      for (short f = 0; f < 2; f++) {
        TF_UNROLL
        for (short e = 0; e < 8; e++) s[f][e] = (16 * f + home.x + TF_COL(e)) < k_rows ? s[f][e] : kIdMasked;
      }
    }
    float top_new[2] = {top[0], top[1]};
    id_fold<true>(s[0], top_new);
    id_fold<true>(s[1], top_new);
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) s[f][e] = fast::exp2(s[f][e] - top_new[e >> 2]);
    }
    float scale_old[2];
    TF_UNROLL
    for (short r = 0; r < 2; r++) {
      scale_old[r] = fast::exp2(top[r] - top_new[r]);
      top[r] = top_new[r];
      total[r] = total[r] * scale_old[r];
    }
    id_fold<false>(s[0], total);
    id_fold<false>(s[1], total);
    TF_UNROLL
    for (short i = 0; i < 4; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * scale_old[e >> 2];
    }
    simdgroup_barrier(mem_flags::mem_none);
    TF_UNROLL
    for (short d = 0; d < 4; d += 2) {
      TF_UNROLL
      for (short k = 0; k < 2; k++) {
        tfp::frag<T> v0, v1;
        if (!AK && last_k) {
          tfp::frag_get_in(v0, V, D, 16 * k, 16 * d, home, k_rows, kIdAnyCols);
          tfp::frag_get_in(v1, V, D, 16 * k, 16 * d + 16, home, k_rows, kIdAnyCols);
        } else {
          tfp::frag_get(v0, V, D, 16 * k, 16 * d, home);
          tfp::frag_get(v1, V, D, 16 * k, 16 * d + 16, home);
        }
        tfp::mma_16x32<false, false>(acc[d], acc[d + 1], s[k], v0, v1);
      }
    }
    K += BK * D;
    V += BK * D;
  }

  threadgroup_barrier(mem_flags::mem_none);
  float inv[2];
  TF_UNROLL
  for (short r = 0; r < 2; r++) inv[r] = 1.f / total[r];
  TF_UNROLL
  for (short i = 0; i < 4; i++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * inv[e >> 2];
  }
  O += tm * os;
  if (!AQ && last_q) {
    if (q_rows <= 0) return;
    TF_UNROLL
    for (short i = 0; i < 4; i++) tfp::frag_put_in(acc[i], O, os, 0, 16 * i, home, q_rows, kIdAnyCols);
  } else {
    TF_UNROLL
    for (short i = 0; i < 4; i++) tfp::frag_put(acc[i], O, os, 0, 16 * i, home);
  }
}

#define ID_ATTENTION(NAME, T, AQ, AK)                                                                            \
  [[max_total_threads_per_threadgroup(128)]] [[kernel]] void NAME(                                             \
      const device T* Q [[buffer(0)]], const device T* K [[buffer(1)]], const device T* V [[buffer(2)]],        \
      const device int32_t* P [[buffer(3)]], const device float* F [[buffer(4)]], device T* O [[buffer(5)]],     \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                     \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                             \
    id_attention<T, AQ, AK>(Q, K, V, O, P, F, tg, sg, lane);                                                   \
  }
ID_ATTENTION(id_attention_00, half, false, false)
ID_ATTENTION(id_attention_01, half, false, true)
ID_ATTENTION(id_attention_10, half, true, false)
ID_ATTENTION(id_attention_11, half, true, true)
