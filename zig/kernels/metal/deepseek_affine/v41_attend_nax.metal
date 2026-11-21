// DeepSeek-V4.1-Flash's prompt attention on the tensor units: exact bf16 products, fp32 sums, bf16 probability parts.
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"
using namespace tfp;

constant constexpr int HEADS = 64, HD = 512, GROUP = 16, BLOCK = 32, SLICE = HD / 4;
constant constexpr float NO_SCORE = -3.4028234663852886e38f;

struct AttnArgs {
  int first_pos;  // row r's position is first_pos + r
  int ring;       // the window ring's rows
  int window;
  int picks;      // compressed picks a row (0: window only)
  float scale;
  int window_end;
  int window_floor;
};

// An accumulator's element e: its column, a key (the M5 layout: this file is built only for GPUs with tensor units).
inline short p_col(short e, short2 home) { return home.x + TF_COL(e); }

// The max (or sum) of a head's values across the four lanes that hold its other keys.
inline float row_max(float v) {
  v = max(v, simd_shuffle_xor(v, ushort(1)));
  return max(v, simd_shuffle_xor(v, ushort(8)));
}
inline float row_sum(float v) {
  v += simd_shuffle_xor(v, ushort(1));
  return v + simd_shuffle_xor(v, ushort(8));
}

// A listed key's row: a compressed pick (k >= 0), a window ring row (k <= -2: row -k - 2), or none (-1, row 0, masked).
inline const device bfloat* key_at(const device bfloat* RING, const device bfloat* COMP, int k) {
  return k >= 0 ? COMP + long(k) * HD : RING + long(max(-k - 2, 0)) * HD;
}

// out [rows, 64, 512]: a threadgroup (4 simdgroups of 128 dims) a (16 heads, row), over its window then its picks.
[[kernel]] void ds_attend_nax(const device bfloat* Q [[buffer(0)]], const device bfloat* RING [[buffer(1)]],
                              const device bfloat* COMP [[buffer(2)]], const device int* PICKS [[buffer(3)]],
                              const device float* SINK [[buffer(4)]], device bfloat* O [[buffer(5)]],
                              constant AttnArgs& a [[buffer(6)]], uint s [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup float partial[4][32][16];  // each simdgroup's scores by lane (all simdgroups' lanes hold the same cells)
  threadgroup int keyrow[BLOCK];
  const int row = int(tg.y), h0 = int(tg.x) * GROUP, d0 = int(s) * SLICE;
  const int pos = a.first_pos + row, nwin = min(a.window, pos + 1), nkeys = nwin + a.picks;
  const device int* picks = PICKS + long(row) * a.picks;
  const short2 home = frag_home(ushort(lane));
  frag<bfloat> q[8];
  TF_UNROLL
  for (short t = 0; t < 8; t++) frag_get(q[t], Q + (long(row) * HEADS + h0) * HD + d0, HD, 0, 16 * t, home);
  frag<float> acc[4][2];
  TF_UNROLL
  for (short c = 0; c < 4; c++) acc[c][0] = acc[c][1] = frag<float>(0);
  float m[2] = {NO_SCORE, NO_SCORE}, l[2] = {0.0f, 0.0f};  // the running max and sum of this lane's two heads
  for (int j0 = 0; j0 < nkeys; j0 += BLOCK) {
    if (s == 0) {
      const int j = j0 + int(lane);
      int k = -1;
      if (j < nwin) {
        const int p = pos - nwin + 1 + j;
        if (p >= min(pos, a.window_floor)) k = -(p % a.ring) - 2;
      } else if (j < nkeys) k = max(picks[j - nwin], -1);
      keyrow[lane] = k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // scores over this simdgroup's dims
    frag<float> lo_s = frag<float>(0), hi_s = frag<float>(0);
    const device bfloat* kr[4];
    TF_UNROLL
    for (short i = 0; i < 4; i++) kr[i] = key_at(RING, COMP, keyrow[8 * i + home.y]) + d0 + home.x;
    TF_UNROLL
    for (short t = 0; t < 8; t++) {
      const frag<bfloat> b0 = frag<bfloat>(*(const device bfloat4*)(kr[0] + 16 * t), *(const device bfloat4*)(kr[1] + 16 * t));
      const frag<bfloat> b1 = frag<bfloat>(*(const device bfloat4*)(kr[2] + 16 * t), *(const device bfloat4*)(kr[3] + 16 * t));
      mma_16x32<false, true>(lo_s, hi_s, q[t], b0, b1);
    }
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      partial[s][lane][e] = lo_s[e];
      partial[s][lane][8 + e] = hi_s[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc[16];
    TF_UNROLL
    for (short i = 0; i < 16; i++) {
      const float v = ((partial[0][lane][i] + partial[1][lane][i]) + partial[2][lane][i]) + partial[3][lane][i];
      sc[i] = keyrow[(i < 8 ? 0 : 16) + p_col(i & 7, home)] != -1 ? v * a.scale : NO_SCORE;
    }
    // the online softmax a head at a time (elements 0-3 and 8-11 are this lane's first head, 4-7 and 12-15 its second)
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
    frag<bfloat> p1[2], p2[2], p3[2];  // keys 0-15 and 16-31: each probability as three bf16 parts (exact)
    float bsum[2] = {0.0f, 0.0f};
    TF_UNROLL
    for (short i = 0; i < 16; i++) {
      const short e = i & 7, h = e >> 2, kb = i < 8 ? 0 : 1;
      const float p = sc[i] > NO_SCORE ? precise::exp(sc[i] - m[h]) : 0.0f;
      bsum[h] += p;
      const bfloat b1 = bfloat(p);
      const float r1 = p - float(b1);
      const bfloat b2 = bfloat(r1);
      p1[kb][e] = b1;
      p2[kb][e] = b2;
      p3[kb][e] = bfloat(r1 - float(b2));
    }
    TF_UNROLL
    for (short h = 0; h < 2; h++) l[h] = l[h] * factor[h] + row_sum(bsum[h]);
    if (!simd_all(factor[0] == 1.0f && factor[1] == 1.0f))
      TF_UNROLL
      for (short c = 0; c < 4; c++) {
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          acc[c][0][e] *= factor[e >> 2];
          acc[c][1][e] *= factor[e >> 2];
        }
      }
    // values (the keys again): keys 0-15 then 16-31, each probability part in turn, into this simdgroup's dims
    TF_UNROLL
    for (short kb = 0; kb < 2; kb++) {
      const device bfloat* va = key_at(RING, COMP, keyrow[16 * kb + home.y]) + d0 + home.x;
      const device bfloat* vb = key_at(RING, COMP, keyrow[16 * kb + home.y + 8]) + d0 + home.x;
      TF_UNROLL
      for (short c = 0; c < 4; c++) {
        frag<bfloat> v0, v1;
        v0 = frag<bfloat>(*(const device bfloat4*)(va + 32 * c), *(const device bfloat4*)(vb + 32 * c));
        v1 = frag<bfloat>(*(const device bfloat4*)(va + 32 * c + 16), *(const device bfloat4*)(vb + 32 * c + 16));
        mma_16x32<false, false>(acc[c][0], acc[c][1], p1[kb], v0, v1);
        mma_16x32<false, false>(acc[c][0], acc[c][1], p2[kb], v0, v1);
        mma_16x32<false, false>(acc[c][0], acc[c][1], p3[kb], v0, v1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // the sink joins each head's denominator
  float den[2];
  TF_UNROLL
  for (short h = 0; h < 2; h++) den[h] = l[h] + precise::exp(SINK[h0 + home.y + 8 * h] - m[h]);
  device bfloat* obase = O + (long(row) * HEADS + h0) * HD + d0;
  TF_UNROLL
  for (short c = 0; c < 4; c++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      frag<float> o = acc[c][j];
      TF_UNROLL
      for (short e = 0; e < 8; e++) o[e] = o[e] / den[e >> 2];
      frag_put(o, obase, HD, 0, 32 * c + 16 * j, home);
    }
  }
}
