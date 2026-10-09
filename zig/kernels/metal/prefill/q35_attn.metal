// Qwen3.5 causal prompt-chunk attention on the tensor units (Flash Next's tf_attn256_nax at Qwen's layouts): 64 query rows
// a threadgroup, one query head, 32-key cache blocks, head size 256, fp32 online softmax. Queries [heads, rows, 256] as
// qwen35_queries writes them; keys and values the layer's cache [key heads, cap, 256]; out [rows, heads, 256] bf16 for
// qwen35_attention_gate. P: rows, keys (position + rows), position, cache rows, query heads, key heads, query rows in the buffer, first row.
#include "../nax.h"
using namespace tfp;

constant constexpr float q35_masked = -3.402823466e+38f;

// A fragment row's max (MAX) or sum over its 16 columns, folded into r for the lane's rows home.y and home.y + 8.
template <bool MAX>
inline void q35_fold(thread const frag<float>& f, thread float (&r)[2]) {
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

[[kernel]] void tf_q35_attn_nax(const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]], constant int* P [[buffer(3)]], constant float& scale [[buffer(4)]],
    device bfloat* O [[buffer(5)]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int BQ = 64, BK = 32, D = 256;
  const int qL = P[0], kL = P[1], qoff = P[2], cap = P[3], H = P[4], KH = P[5], T = P[6], base = P[7];
  const int h = int(tg.y), q0 = int(tg.x) * BQ;
  const short tm = 16 * short(sg);
  const int kv = h / (H / KH);
  const device bfloat* Qp = Q + (long(h) * T + base + q0 + tm) * D;
  const device bfloat* Kp = K + long(kv) * cap * D;
  const device bfloat* Vp = V + long(kv) * cap * D;
  const float scale2 = scale * 1.44269504089f;
  const short2 home = frag_home(ushort(lane));
  const int qrows = qL - (q0 + tm);  // this simdgroup's valid query rows (may be <= 0 in the last tile)
  frag<float> acc[D / 16];
  TF_UNROLL
  for (short i = 0; i < D / 16; i++) acc[i] = frag<float>(0);
  float top[2] = {q35_masked, q35_masked}, total[2] = {0.0f, 0.0f};
  const int last_row = qoff + min(q0 + BQ, qL) - 1;
  const int keys = min(kL, last_row + 1);
  const int blocks = (keys + BK - 1) / BK;
  for (int kb = 0; kb < blocks; kb++) {
    const int k0 = kb * BK;
    const bool whole = k0 + BK <= keys;
    frag<float> s[2] = {frag<float>(0), frag<float>(0)};
#pragma clang loop unroll_count(4)
    for (short d = 0; d < D / 16; d++) {
      frag<bfloat> q, ka, kb2;
      frag_get_in(q, Qp, D, 0, 16 * d, home, qrows, D);
      if (whole) {
        frag_get_t(ka, Kp + long(k0) * D, D, 0, 16 * d, home);
        frag_get_t(kb2, Kp + long(k0) * D, D, 16, 16 * d, home);
      } else {
        frag_get_t_in(ka, Kp + long(k0) * D, D, 0, 16 * d, home, keys - k0, D);
        frag_get_t_in(kb2, Kp + long(k0) * D, D, 16, 16 * d, home, keys - k0, D);
      }
      mma_16x32<false, true>(s[0], s[1], q, ka, kb2);
    }
    const int r0 = qoff + q0 + tm + home.y;
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int col = k0 + 16 * f + home.x + TF_COL(e), row = r0 + (e >> 2) * 8;
        s[f][e] = (col >= kL || col > row) ? q35_masked : s[f][e] * scale2;
      }
    }
    float top_new[2] = {top[0], top[1]};
    q35_fold<true>(s[0], top_new);
    q35_fold<true>(s[1], top_new);
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) s[f][e] = fast::exp2(s[f][e] - top_new[e >> 2]);
    }
    float scale_old[2];
    TF_UNROLL
    for (short hh = 0; hh < 2; hh++) {
      scale_old[hh] = fast::exp2(top[hh] - top_new[hh]);
      top[hh] = top_new[hh];
      total[hh] = total[hh] * scale_old[hh];
    }
    q35_fold<false>(s[0], total);
    q35_fold<false>(s[1], total);
    TF_UNROLL
    for (short i = 0; i < D / 16; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * scale_old[e >> 2];
    }
    TF_UNROLL
    for (short d = 0; d < D / 16; d += 2) {
      TF_UNROLL
      for (short k = 0; k < 2; k++) {
        frag<bfloat> v0, v1;
        if (whole) {
          frag_get(v0, Vp + long(k0) * D, D, 16 * k, 16 * d, home);
          frag_get(v1, Vp + long(k0) * D, D, 16 * k, 16 * d + 16, home);
        } else {
          frag_get_in(v0, Vp + long(k0) * D, D, 16 * k, 16 * d, home, keys - k0, D);
          frag_get_in(v1, Vp + long(k0) * D, D, 16 * k, 16 * d + 16, home, keys - k0, D);
        }
        mma_16x32<false, false>(acc[d], acc[d + 1], s[k], v0, v1);
      }
    }
  }
  const float inv[2] = {1.0f / total[0], 1.0f / total[1]};
  TF_UNROLL
  for (short i = 0; i < D / 16; i++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      const int row = q0 + tm + home.y + (e >> 2) * 8, col = 16 * i + home.x + TF_COL(e);
      if (row < qL) O[(long(row) * H + h) * D + col] = bfloat(acc[i][e] * inv[e >> 2]);
    }
  }
}
