// GLM's sparse MLA for prompt chunks on the tensor units: exact bf16 products, fp32 sums, probabilities in three bf16 parts (exact).
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"
using namespace tfp;

constant constexpr int HEADS = 64, RANK = 512, GROUP = 16, BLOCK = 32, SLICE = RANK / 4;
constant constexpr float NO_SCORE = -3.4028234663852886e38f;

// A transposed right operand's element e: the row it comes from (here a key) and its column (a latent dim).
#ifndef TF_SIMD_FRAGS
inline short t_key(short e, short2 home) { return home.y + (e >> 2) * 8; }
inline short t_dim(short e, short2 home) { return home.x + (e & 3); }
#else
inline short t_key(short e, short2 home) { return home.x + TF_COL(e); }
inline short t_dim(short e, short2 home) { return home.y + (e >> 2) * 8; }
#endif
// A plain operand's or an accumulator's element e: its row (a head, or a key for the values) and column.
inline short p_row(short e, short2 home) { return home.y + (e >> 2) * 8; }
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

// out [rows, 64, 512] over each row's `width` listed keys (an entry outside [0, key_length) is no key); grid (4 head groups, rows).
[[kernel]] void glm_sparse_nax(const device bfloat* ql [[buffer(0)]], const device bfloat* keys [[buffer(1)]],
                               const device int32_t* indices [[buffer(2)]], constant float& scale [[buffer(3)]],
                               constant int4& meta [[buffer(4)]], device bfloat* out [[buffer(5)]],
                               uint s [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float partial[4][32][16]; // each simdgroup's scores, by lane (every simdgroup's lanes hold the same cells)
  threadgroup int keyrow[BLOCK];
  const int width = meta.x, key_length = meta.y;
  const int row = int(tg.y), h0 = int(tg.x) * GROUP, d0 = int(s) * SLICE;
  const short2 home = frag_home(ushort(lane));
  const device int32_t* list = indices + long(row) * width;
  // this simdgroup's 128 dims of the 16 heads' queries, as eight 16-dim fragments
  frag<bfloat> q[8];
  const device bfloat* qbase = ql + (long(row) * HEADS + h0) * RANK + d0;
  TF_UNROLL
  for (short t = 0; t < 8; t++) frag_get(q[t], qbase, RANK, 0, 16 * t, home);
  frag<float> acc[4][2];
  TF_UNROLL
  for (short c = 0; c < 4; c++) {
    acc[c][0] = frag<float>(0);
    acc[c][1] = frag<float>(0);
  }
  float m[2] = {NO_SCORE, NO_SCORE}, l[2] = {0.0f, 0.0f}; // the running max and sum of this lane's two heads
  for (int j0 = 0; j0 < width; j0 += BLOCK) {
    if (s == 0) {
      const int j = j0 + int(lane);
      const int k = j < width ? list[j] : -1;
      keyrow[lane] = k >= 0 && k < key_length ? k : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // scores: this simdgroup's dims
    frag<float> lo = frag<float>(0), hi = frag<float>(0);
    TF_UNROLL
    for (short t = 0; t < 8; t++) {
      frag<bfloat> b0, b1;
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int k0 = keyrow[t_key(e, home)], k1 = keyrow[16 + t_key(e, home)];
        const int d = d0 + 16 * t + t_dim(e, home);
        b0[e] = keys[long(max(k0, 0)) * RANK + d];
        b1[e] = keys[long(max(k1, 0)) * RANK + d];
      }
      mma_16x32<false, true>(lo, hi, q[t], b0, b1);
    }
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      partial[s][lane][e] = lo[e];
      partial[s][lane][8 + e] = hi[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc[16];
    TF_UNROLL
    for (short i = 0; i < 16; i++) {
      const float v = ((partial[0][lane][i] + partial[1][lane][i]) + partial[2][lane][i]) + partial[3][lane][i];
      const short e = i & 7;
      const int key = (i < 8 ? 0 : 16) + p_col(e, home);
      sc[i] = keyrow[key] >= 0 ? v * scale : NO_SCORE;
    }
    // the online softmax, a head at a time (elements 0-3 and 8-11 are this lane's first head, 4-7 and 12-15 its second)
    float factor[2];
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      float bmax = NO_SCORE;
      TF_UNROLL
      for (short i = 0; i < 4; i++) bmax = max(bmax, max(sc[4 * h + i], sc[8 + 4 * h + i]));
      const float new_max = max(m[h], row_max(bmax));
      factor[h] = fast::exp(m[h] - new_max);
      m[h] = new_max;
    }
    frag<bfloat> p1[2], p2[2], p3[2]; // keys 0-15 and 16-31: the probabilities' three bf16 parts
    float bsum[2] = {0.0f, 0.0f};
    TF_UNROLL
    for (short i = 0; i < 16; i++) {
      const short e = i & 7, h = e >> 2, kb = i < 8 ? 0 : 1;
      const float p = sc[i] > NO_SCORE ? fast::exp(sc[i] - m[h]) : 0.0f;
      bsum[h] += p;
      const bfloat a = bfloat(p);
      const float r1 = p - float(a);
      const bfloat b = bfloat(r1);
      p1[kb][e] = a;
      p2[kb][e] = b;
      p3[kb][e] = bfloat(r1 - float(b));
    }
    TF_UNROLL
    for (short h = 0; h < 2; h++) l[h] = l[h] * factor[h] + row_sum(bsum[h]);
    TF_UNROLL
    for (short c = 0; c < 4; c++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        acc[c][0][e] *= factor[e >> 2];
        acc[c][1][e] *= factor[e >> 2];
      }
    }
    // values: keys 0-15 then 16-31, each part in turn, into this simdgroup's 128 dims
    TF_UNROLL
    for (short kb = 0; kb < 2; kb++) {
      TF_UNROLL
      for (short c = 0; c < 4; c++) {
        frag<bfloat> v0, v1;
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          const int k = max(keyrow[16 * kb + p_row(e, home)], 0);
          const int d = d0 + 32 * c + p_col(e, home);
          v0[e] = keys[long(k) * RANK + d];
          v1[e] = keys[long(k) * RANK + d + 16];
        }
        mma_16x32<false, false>(acc[c][0], acc[c][1], p1[kb], v0, v1);
        mma_16x32<false, false>(acc[c][0], acc[c][1], p2[kb], v0, v1);
        mma_16x32<false, false>(acc[c][0], acc[c][1], p3[kb], v0, v1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  device bfloat* obase = out + (long(row) * HEADS + h0) * RANK + d0;
  TF_UNROLL
  for (short c = 0; c < 4; c++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      frag<float> o = acc[c][j];
      TF_UNROLL
      for (short e = 0; e < 8; e++) o[e] = l[e >> 2] == 0.0f ? 0.0f : o[e] / l[e >> 2];
      frag_put(o, obase, RANK, 0, 32 * c + 16 * j, home);
    }
  }
}
