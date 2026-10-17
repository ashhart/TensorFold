// Step-5's prompt attention on the tensor units: a KV group's 16 query heads are one operand over each row's keys.
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"
using namespace tfp;

// glm_sparse_nax.metal's layout at Step-5's shapes: 192 dims a head in three simdgroups of 64, keys 32 at a time.
constant constexpr int D = 192, GROUP = 16, BLOCK = 32, SLICE = 64, SG = 3;
constant constexpr float NO_SCORE = -3.4028234663852886e38f;

struct s5_sparse_args {
  uint rows, pos, heads, dim, span, ring, window, regions, groups, topk, first, norm;
  float eps, theta, scale;
};

#ifndef TF_SIMD_FRAGS
inline short t_key(short e, short2 home) { return home.y + (e >> 2) * 8; }
inline short t_dim(short e, short2 home) { return home.x + (e & 3); }
#else
inline short t_key(short e, short2 home) { return home.x + TF_COL(e); }
inline short t_dim(short e, short2 home) { return home.y + (e >> 2) * 8; }
#endif
inline short p_row(short e, short2 home) { return home.y + (e >> 2) * 8; }
inline short p_col(short e, short2 home) { return home.x + TF_COL(e); }

inline float row_max(float v) {
  v = max(v, simd_shuffle_xor(v, ushort(1)));
  return max(v, simd_shuffle_xor(v, ushort(8)));
}
inline float row_sum(float v) {
  v += simd_shuffle_xor(v, ushort(1));
  return v + simd_shuffle_xor(v, ushort(8));
}

// out [rows, heads, D] over each row's window (a ring) or picks and own region; exact products, fp32 sums.
[[kernel]] void s5_attend_nax(const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]], const device int* sel [[buffer(3)]], constant s5_sparse_args& a [[buffer(4)]],
    device bfloat* O [[buffer(5)]], uint s [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float partial[SG][32][16];
  threadgroup int keyrow[BLOCK];
  const int group = int(tg.x), row = int(tg.y), h0 = group * GROUP, d0 = int(s) * SLICE;
  const int kvh = int(a.groups), ld = kvh * D; // a cache row: every KV group's keys
  const short2 home = frag_home(ushort(lane));
  frag<bfloat> q[4];
  TF_UNROLL
  for (short t = 0; t < 4; t++) frag_get(q[t], Q + (long(row) * a.heads + h0) * D + d0, D, 0, 16 * t, home);
  frag<float> acc[2][2];
  TF_UNROLL
  for (short c = 0; c < 2; c++) {
    acc[c][0] = frag<float>(0);
    acc[c][1] = frag<float>(0);
  }
  float m[2] = {NO_SCORE, NO_SCORE}, l[2] = {0.0f, 0.0f}; // the running max and sum of this lane's two heads
  const uint p = a.pos + uint(row);
  uint lo = 0, own = 0;
  bool listed = false;
  if (a.window > 0) {
    lo = p + 1 > a.window ? p + 1 - a.window : 0;
  } else {
    own = p / 8 * 8;
    listed = p / 8 > a.topk;
  }
  const int count = int(listed ? a.topk * 8 + (p + 1 - own) : p + 1 - lo);
  const device int* picks = sel + (long(row) * kvh + group) * a.topk;
  const device bfloat* kg = K + group * D + d0;
  const device bfloat* vg = V + group * D + d0;
  for (int j0 = 0; j0 < count; j0 += BLOCK) {
    if (s == 0) {
      const int j = j0 + int(lane);
      int slot = -1;
      if (j < count) {
        const uint key = !listed ? lo + uint(j) : (uint(j) < a.topk * 8 ? uint(picks[j / 8]) * 8 + uint(j) % 8 : own + uint(j) - a.topk * 8);
        slot = int(a.ring ? key % a.ring : key);
      }
      keyrow[lane] = slot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    frag<float> lo_s = frag<float>(0), hi_s = frag<float>(0);
#ifndef TF_SIMD_FRAGS
    const device bfloat* kr[4];
    TF_UNROLL
    for (short i = 0; i < 4; i++) kr[i] = kg + long(max(keyrow[8 * i + home.y], 0)) * ld + home.x;
    TF_UNROLL
    for (short t = 0; t < 4; t++) {
      const frag<bfloat> b0 = frag<bfloat>(*(const device bfloat4*)(kr[0] + 16 * t), *(const device bfloat4*)(kr[1] + 16 * t));
      const frag<bfloat> b1 = frag<bfloat>(*(const device bfloat4*)(kr[2] + 16 * t), *(const device bfloat4*)(kr[3] + 16 * t));
      mma_16x32<false, true>(lo_s, hi_s, q[t], b0, b1);
    }
#else
    TF_UNROLL
    for (short t = 0; t < 4; t++) {
      frag<bfloat> b0, b1;
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int k0 = keyrow[t_key(e, home)], k1 = keyrow[16 + t_key(e, home)];
        const int d = 16 * t + t_dim(e, home);
        b0[e] = kg[long(max(k0, 0)) * ld + d];
        b1[e] = kg[long(max(k1, 0)) * ld + d];
      }
      mma_16x32<false, true>(lo_s, hi_s, q[t], b0, b1);
    }
#endif
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      partial[s][lane][e] = lo_s[e];
      partial[s][lane][8 + e] = hi_s[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc[16];
    TF_UNROLL
    for (short i = 0; i < 16; i++) {
      const float v = (partial[0][lane][i] + partial[1][lane][i]) + partial[2][lane][i];
      const int key = (i < 8 ? 0 : 16) + p_col(i & 7, home);
      sc[i] = keyrow[key] >= 0 ? v * a.scale : NO_SCORE;
    }
    float factor[2];
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      float bmax = NO_SCORE;
      TF_UNROLL
      for (short i = 0; i < 4; i++) bmax = max(bmax, max(sc[4 * h + i], sc[8 + 4 * h + i]));
      const float new_max = max(m[h], row_max(bmax));
      factor[h] = precise::exp(m[h] - new_max);
      m[h] = new_max;
    }
    frag<bfloat> p1[2], p2[2], p3[2]; // keys 0-15 and 16-31: the probabilities' three bf16 parts
    float bsum[2] = {0.0f, 0.0f};
    TF_UNROLL
    for (short i = 0; i < 16; i++) {
      const short e = i & 7, h = e >> 2, kb = i < 8 ? 0 : 1;
      const float pr = sc[i] > NO_SCORE ? precise::exp(sc[i] - m[h]) : 0.0f;
      bsum[h] += pr;
      const bfloat x = bfloat(pr);
      const float r1 = pr - float(x);
      const bfloat y = bfloat(r1);
      p1[kb][e] = x;
      p2[kb][e] = y;
      p3[kb][e] = bfloat(r1 - float(y));
    }
    TF_UNROLL
    for (short h = 0; h < 2; h++) l[h] = l[h] * factor[h] + row_sum(bsum[h]);
    TF_UNROLL
    for (short c = 0; c < 2; c++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        acc[c][0][e] *= factor[e >> 2];
        acc[c][1][e] *= factor[e >> 2];
      }
    }
    TF_UNROLL
    for (short kb = 0; kb < 2; kb++) {
#ifndef TF_SIMD_FRAGS
      const device bfloat* va = vg + long(max(keyrow[16 * kb + home.y], 0)) * ld + home.x;
      const device bfloat* vb = vg + long(max(keyrow[16 * kb + home.y + 8], 0)) * ld + home.x;
#endif
      TF_UNROLL
      for (short c = 0; c < 2; c++) {
        frag<bfloat> v0, v1;
#ifndef TF_SIMD_FRAGS
        v0 = frag<bfloat>(*(const device bfloat4*)(va + 32 * c), *(const device bfloat4*)(vb + 32 * c));
        v1 = frag<bfloat>(*(const device bfloat4*)(va + 32 * c + 16), *(const device bfloat4*)(vb + 32 * c + 16));
#else
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          const int k = max(keyrow[16 * kb + p_row(e, home)], 0);
          const int d = 32 * c + p_col(e, home);
          v0[e] = vg[long(k) * ld + d];
          v1[e] = vg[long(k) * ld + d + 16];
        }
#endif
        mma_16x32<false, false>(acc[c][0], acc[c][1], p1[kb], v0, v1);
        mma_16x32<false, false>(acc[c][0], acc[c][1], p2[kb], v0, v1);
        mma_16x32<false, false>(acc[c][0], acc[c][1], p3[kb], v0, v1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  device bfloat* obase = O + (long(row) * a.heads + h0) * D + d0;
  TF_UNROLL
  for (short c = 0; c < 2; c++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      frag<float> o = acc[c][j];
      TF_UNROLL
      for (short e = 0; e < 8; e++) o[e] = l[e >> 2] == 0.0f ? 0.0f : o[e] / l[e >> 2];
      frag_put(o, obase, D, 0, 32 * c + 16 * j, home);
    }
  }
}
