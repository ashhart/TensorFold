// Flash Next prompt rows' sparse attention on the tensor units: the indexer's block scores for a chunk's rows, and
// each row's attention over its selected blocks and tail (or every key while the context is short).
#include "../nax.h"
using namespace tfp;

namespace tfa {

// A fragment row's max (MAX) or sum over the 32 columns of (f0 | f1), for the lane's rows home.y and home.y + 8.
template <bool MAX>
inline void row_fold(thread const frag<float>& f0, thread const frag<float>& f1, thread float (&r)[2]) {
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const short b = 4 * h;
    float t0 = MAX ? max(max(f0[b], f0[b + 1]), max(f0[b + 2], f0[b + 3])) : (f0[b] + f0[b + 1]) + (f0[b + 2] + f0[b + 3]);
    float t1 = MAX ? max(max(f1[b], f1[b + 1]), max(f1[b + 2], f1[b + 3])) : (f1[b] + f1[b + 1]) + (f1[b + 2] + f1[b + 3]);
    float t = MAX ? max(t0, t1) : t0 + t1;
    const float u = simd_shuffle_xor(t, ushort(1));
    t = MAX ? max(t, u) : t + u;
    const float w = simd_shuffle_xor(t, ushort(8));
    t = MAX ? max(t, w) : t + w;
    r[h] = MAX ? max(r[h], t) : r[h] + t;
  }
}

inline float bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }

}  // namespace tfa

// SC[r][b] = (relu(q_0 . k_b) + relu(q_1 . k_b) + relu(q_2 . k_b) + relu(q_3 . k_b)) / sqrt(128), fp32, for every row
// r < rows and block b < nb (the caller reads only each row's complete blocks). IQ [rows, 4, 128] bf16, POOLED [nb, 128]
// bf16. A threadgroup of 4 simdgroups: 32 rows x 64 blocks, a simdgroup 16 rows x 32 blocks. P: rows, nb.
[[kernel]] void tf_idx_scores_nax(const device bfloat* IQ [[buffer(0)]], const device bfloat* POOLED [[buffer(1)]],
    constant int2& P [[buffer(2)]], device float* SC [[buffer(3)]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int HI = 4, DI = 128;
  const int rows = P.x, nb = P.y;
  const int b0 = (int(tg.x) * 2 + int(sg % 2)) * 32, r0 = (int(tg.y) * 2 + int(sg / 2)) * 16;
  if (b0 >= nb || r0 >= rows) return;
  const short2 home = frag_home(ushort(lane));
  frag<float> tot[2] = {frag<float>(0), frag<float>(0)};
  TF_UNROLL
  for (short h = 0; h < HI; h++) {
    frag<float> s[2] = {frag<float>(0), frag<float>(0)};
    TF_UNROLL
    for (short d = 0; d < DI / 16; d++) {
      frag<bfloat> q, k0, k1;
      frag_get_in(q, IQ + h * DI, HI * DI, r0, 16 * d, home, rows, DI);
      frag_get_in(k0, POOLED, DI, b0, 16 * d, home, nb, DI);
      frag_get_in(k1, POOLED, DI, b0 + 16, 16 * d, home, nb, DI);
      mma_16x32<false, true>(s[0], s[1], q, k0, k1);
    }
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) tot[f][e] += metal::max(s[f][e], 0.0f);
    }
  }
  const float inv = 1.0f / metal::precise::sqrt(float(DI));
  TF_UNROLL
  for (short f = 0; f < 2; f++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) tot[f][e] = tot[f][e] * inv;
    frag_put_in(tot[f], SC + long(r0) * nb + b0 + 16 * f, nb, 0, 0, home, rows - r0, nb - b0 - 16 * f);
  }
}

// Row r's attention for key head g: the 12 query heads that read it are the 16-row operand (4 zero rows); the row's
// keys (NK[r]: with SPARSE[r] the ids IDS[r], its selected blocks then its tail; else keys 0 .. NK[r] - 1) in blocks
// of 32, each block's keys read once for the 12 heads. Simdgroup s owns dims 64 s .. 64 s + 63: its partial scores
// go through threadgroup memory and every simdgroup adds the four in order, so all four keep the same fp32 online
// softmax; it accumulates its dims of the output. The gate bf16(bf16(o) * bsig(gate)) on the way out.
// P: cache rows (the keys' row stride per head).
[[kernel]] void tf_sattn_nax(const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]], const device int* IDS [[buffer(3)]], const device int* NK [[buffer(4)]],
    const device int* SPARSE [[buffer(5)]], const device bfloat* GP [[buffer(6)]], const device float* F [[buffer(7)]],
    constant int& cap [[buffer(8)]], device bfloat* O [[buffer(9)]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int D = 256, H = 24, GQA = 12, BK = 32, KW = 2051, PW = 13952, DS = 64;
  constexpr float masked = -3.402823466e+38f;
  const int r = int(tg.x), g = int(tg.y);
  const int n = NK[r];
  const bool sparse = SPARSE[r] != 0;
  const device int* ids = IDS + long(r) * KW;
  const device bfloat* Kh = K + long(g) * cap * D + DS * int(sg);
  const device bfloat* Vh = V + long(g) * cap * D + DS * int(sg);
  const short2 home = frag_home(ushort(lane));
  threadgroup float part[4][16 * BK];
  // the queries of heads 12 g .. 12 g + 11, this simdgroup's 64 dims: 4 fragments
  frag<bfloat> q[DS / 16];
  const device bfloat* Qr = Q + (long(r) * H + GQA * g) * D + DS * int(sg);
  TF_UNROLL
  for (short d = 0; d < DS / 16; d++) frag_get_in(q[d], Qr, D, 0, 16 * d, home, GQA, DS);
  frag<float> acc[DS / 16];
  TF_UNROLL
  for (short i = 0; i < DS / 16; i++) acc[i] = frag<float>(0);
  float top[2] = {masked, masked}, total[2] = {0.0f, 0.0f};
  const float scale2 = F[0] * 1.44269504089f;
  const int blocks = (n + BK - 1) / BK;
  for (int kb = 0; kb < blocks; kb++) {
    const int j0 = kb * BK;
    // the lane's four keys of the block: fragment rows home.y and home.y + 8 of each 16-key half
    int key[4];
    TF_UNROLL
    for (short i = 0; i < 4; i++) {
      const int j = j0 + 16 * (i >> 1) + home.y + 8 * (i & 1);
      key[i] = j < n ? (sparse ? ids[j] : j) : 0;
    }
    // this simdgroup's dims of q . k for the 32 keys
    frag<float> s[2] = {frag<float>(0), frag<float>(0)};
    TF_UNROLL
    for (short d = 0; d < DS / 16; d++) {
      frag<bfloat> k0, k1;
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int c = 16 * d + home.x + (e & 3);
        k0[e] = Kh[long(key[(e >> 2)]) * D + c];
        k1[e] = Kh[long(key[2 + (e >> 2)]) * D + c];
      }
      mma_16x32<false, true>(s[0], s[1], q[d], k0, k1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // the previous block's partials are read
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) part[sg][(home.y + (e >> 2) * 8) * BK + 16 * f + home.x + (e & 3)] = s[f][e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int at = (home.y + (e >> 2) * 8) * BK + 16 * f + home.x + (e & 3);
        const float v = ((part[0][at] + part[1][at]) + part[2][at]) + part[3][at];
        const int j = j0 + 16 * f + home.x + (e & 3);
        s[f][e] = j < n ? v * scale2 : masked;
      }
    }
    float top_new[2] = {top[0], top[1]};
    tfa::row_fold<true>(s[0], s[1], top_new);
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) s[f][e] = fast::exp2(s[f][e] - top_new[e >> 2]);
    }
    float keep[2];
    TF_UNROLL
    for (short hh = 0; hh < 2; hh++) {
      keep[hh] = fast::exp2(top[hh] - top_new[hh]);
      top[hh] = top_new[hh];
      total[hh] = total[hh] * keep[hh];
    }
    tfa::row_fold<false>(s[0], s[1], total);
    TF_UNROLL
    for (short i = 0; i < DS / 16; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * keep[e >> 2];
    }
    // acc (16 x 64) += p (16 x 32 keys) * v (32 keys x 64 dims)
    TF_UNROLL
    for (short d = 0; d < DS / 16; d += 2) {
      TF_UNROLL
      for (short k = 0; k < 2; k++) {
        frag<bfloat> v0, v1;
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          const long row = long(key[2 * k + (e >> 2)]) * D;
          const int c = 16 * d + home.x + (e & 3);
          v0[e] = Vh[row + c];
          v1[e] = Vh[row + c + 16];
        }
        mma_16x32<false, false>(acc[d], acc[d + 1], s[k], v0, v1);
      }
    }
  }
  const float inv[2] = {1.0f / total[0], 1.0f / total[1]};
  TF_UNROLL
  for (short i = 0; i < DS / 16; i++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      const int head = home.y + (e >> 2) * 8, col = DS * int(sg) + 16 * i + home.x + (e & 3);
      if (head < GQA) {
        const int h = GQA * g + head;
        const float o = float(bfloat(acc[i][e] * inv[e >> 2]));
        const float gate = float(GP[long(r) * PW + h * 2 * D + D + col]);
        O[(long(r) * H + h) * D + col] = bfloat(o * tfa::bsig(gate));
      }
    }
  }
}
