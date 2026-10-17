// music3 DiT attention on the tensor units: heads of 64 from roped qkv rows, keys to each window's length.
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"

namespace m3 {
using namespace tfp;

constant constexpr float kMasked = -3.402823466e+38f;
constant constexpr int kAnyCols = 1 << 30;

// Fold one fragment's rows into r[2] (rows home.y, home.y + 8): pairs, then lanes 1 and 8 apart, then into r.
template <bool MAX>
inline void row_fold(thread const frag<float>& f, thread float (&r)[2]) {
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

}  // namespace m3

struct DitAttnArgs {
  uint Lseq;
  float scale;
};

// qkv [S, Lseq, 6144] (roped), lens [S] keys a row, out [S, Lseq, 2048]: 64 queries a group, 32 keys a block, exp2.
[[max_total_threads_per_threadgroup(128)]] kernel void m3_dit_attn(const device half* qkv [[buffer(0)]], const device int* lens [[buffer(1)]],
                        device half* out [[buffer(2)]], constant DitAttnArgs& a [[buffer(3)]],
                        uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  using namespace tfp;
  constexpr int BQ = 64, BK = 32, RS = 6144;
  const int L = lens[tg.z], Lseq = int(a.Lseq);
  const int q0 = int(tg.x) * BQ + 16 * int(sg);
  const device half* rows = qkv + (ulong)tg.z * Lseq * RS + tg.y * 64;
  const device half* Q = rows + (long)q0 * RS;
  const device half* K = rows + 2048;
  const device half* V = rows + 4096;
  const float scale2 = a.scale * 1.44269504089f;
  const short2 home = frag_home(ushort(lane));
  const int q_rows = Lseq - q0;
  frag<half> q[4];
  for (short d = 0; d < 4; d++) frag_get_in(q[d], Q, RS, 0, 16 * d, home, q_rows, m3::kAnyCols);
  frag<float> acc[4];
  for (short i = 0; i < 4; i++) acc[i] = frag<float>(0);
  float top[2] = {m3::kMasked, m3::kMasked}, total[2] = {0.0f, 0.0f};
  const int blocks = (L + BK - 1) / BK;
  for (int kb = 0; kb < blocks; kb++) {
    const int k_rows = L - kb * BK;
    const bool part = k_rows < BK;
    frag<float> s[2] = {frag<float>(0), frag<float>(0)};
    for (short d = 0; d < 4; d++) {
      frag<half> k0, k1;
      if (part) {
        frag_get_t_in(k0, K, RS, 0, 16 * d, home, k_rows, m3::kAnyCols);
        frag_get_t_in(k1, K, RS, 16, 16 * d, home, k_rows, m3::kAnyCols);
      } else {
        frag_get_t(k0, K, RS, 0, 16 * d, home);
        frag_get_t(k1, K, RS, 16, 16 * d, home);
      }
      mma_16x32<false, true>(s[0], s[1], q[d], k0, k1);
    }
    for (short f = 0; f < 2; f++)
      for (short e = 0; e < 8; e++) {
        const bool live = !part || (16 * f + home.x + TF_COL(e)) < k_rows;
        s[f][e] = live ? s[f][e] * scale2 : m3::kMasked;
      }
    float top_new[2] = {top[0], top[1]};
    m3::row_fold<true>(s[0], top_new);
    m3::row_fold<true>(s[1], top_new);
    for (short f = 0; f < 2; f++)
      for (short e = 0; e < 8; e++) s[f][e] = precise::exp2(s[f][e] - top_new[e >> 2]);
    float scale_old[2];
    for (short h = 0; h < 2; h++) {
      scale_old[h] = precise::exp2(top[h] - top_new[h]);
      top[h] = top_new[h];
      total[h] = total[h] * scale_old[h];
    }
    m3::row_fold<false>(s[0], total);
    m3::row_fold<false>(s[1], total);
    for (short i = 0; i < 4; i++)
      for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * scale_old[e >> 2];
    frag<half> p[2];
    for (short f = 0; f < 2; f++)
      for (short e = 0; e < 8; e++) p[f][e] = half(s[f][e]);  // probabilities in fp16 for P V, as on CUDA
    for (short d = 0; d < 4; d += 2) {
      for (short k = 0; k < 2; k++) {
        frag<half> v0, v1;
        if (part) {
          frag_get_in(v0, V, RS, 16 * k, 16 * d, home, k_rows, m3::kAnyCols);
          frag_get_in(v1, V, RS, 16 * k, 16 * d + 16, home, k_rows, m3::kAnyCols);
        } else {
          frag_get(v0, V, RS, 16 * k, 16 * d, home);
          frag_get(v1, V, RS, 16 * k, 16 * d + 16, home);
        }
        mma_16x32<false, false>(acc[d], acc[d + 1], p[k], v0, v1);
      }
    }
    K += BK * RS;
    V += BK * RS;
  }
  if (q_rows <= 0) return;
  const float inv[2] = {1.0f / total[0], 1.0f / total[1]};
  for (short i = 0; i < 4; i++)
    for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * inv[e >> 2];
  device half* O = out + ((ulong)tg.z * Lseq + q0) * 2048 + tg.y * 64;
  for (short i = 0; i < 4; i++) frag_put_in(acc[i], O, 2048, 0, 16 * i, home, q_rows, m3::kAnyCols);
}
