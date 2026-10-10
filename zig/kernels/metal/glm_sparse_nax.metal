// GLM's sparse MLA for prompt chunks on the tensor units: exact bf16 products, fp32 sums, probabilities in three bf16 parts (exact).
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"
using namespace tfp;

#ifndef GLM_HEADS // TP2 builds one Mac's 32 heads
#define GLM_HEADS 64
#endif
constant constexpr int HEADS = GLM_HEADS, RANK = 512, GROUP = 16, BLOCK = 32, SLICE = RANK / 4;
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

// This simdgroup's 128 dims of 16 heads' queries as eight 16-dim fragments.
inline void sparse_queries(thread frag<bfloat> (&q)[8], const device bfloat* qbase, short2 home) {
  TF_UNROLL
  for (short t = 0; t < 8; t++) frag_get(q[t], qbase, RANK, 0, 16 * t, home);
}

// List entries [lo, hi) of one row (an entry outside [0, key_length) is no key): scores, the online softmax, values into `acc`.
inline void sparse_pass(thread frag<float> (&acc)[4][2], thread float (&m)[2], thread float (&l)[2], thread const frag<bfloat> (&q)[8],
                        const device bfloat* keys, const device int32_t* list, int lo, int hi, int key_length, float scale,
                        threadgroup float (&partial)[4][32][16], threadgroup int (&keyrow)[BLOCK], uint s, uint lane, short2 home, int d0) {
  for (int j0 = lo; j0 < hi; j0 += BLOCK) {
    if (s == 0) {
      const int j = j0 + int(lane);
      const int k = j < hi ? list[j] : -1;
      keyrow[lane] = k >= 0 && k < key_length ? k : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // scores: this simdgroup's dims
    frag<float> lo_s = frag<float>(0), hi_s = frag<float>(0);
#ifndef TF_SIMD_FRAGS
    // a lane's four dims of each of its four keys, 8 bytes a load
    const device bfloat* kr[4];
    TF_UNROLL
    for (short i = 0; i < 4; i++) kr[i] = keys + long(max(keyrow[8 * i + home.y], 0)) * RANK + d0 + home.x;
    TF_UNROLL
    for (short t = 0; t < 8; t++) {
      const frag<bfloat> b0 = frag<bfloat>(*(const device bfloat4*)(kr[0] + 16 * t), *(const device bfloat4*)(kr[1] + 16 * t));
      const frag<bfloat> b1 = frag<bfloat>(*(const device bfloat4*)(kr[2] + 16 * t), *(const device bfloat4*)(kr[3] + 16 * t));
      mma_16x32<false, true>(lo_s, hi_s, q[t], b0, b1);
    }
#else
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
    if (!simd_all(factor[0] == 1.0f && factor[1] == 1.0f)) // a factor of 1 leaves every value as it is
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
#ifndef TF_SIMD_FRAGS
      const device bfloat* va = keys + long(max(keyrow[16 * kb + home.y], 0)) * RANK + d0 + home.x;
      const device bfloat* vb = keys + long(max(keyrow[16 * kb + home.y + 8], 0)) * RANK + d0 + home.x;
#endif
      TF_UNROLL
      for (short c = 0; c < 4; c++) {
        frag<bfloat> v0, v1;
#ifndef TF_SIMD_FRAGS
        v0 = frag<bfloat>(*(const device bfloat4*)(va + 32 * c), *(const device bfloat4*)(vb + 32 * c));
        v1 = frag<bfloat>(*(const device bfloat4*)(va + 32 * c + 16), *(const device bfloat4*)(vb + 32 * c + 16));
#else
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          const int k = max(keyrow[16 * kb + p_row(e, home)], 0);
          const int d = d0 + 32 * c + p_col(e, home);
          v0[e] = keys[long(k) * RANK + d];
          v1[e] = keys[long(k) * RANK + d + 16];
        }
#endif
        mma_16x32<false, false>(acc[c][0], acc[c][1], p1[kb], v0, v1);
        mma_16x32<false, false>(acc[c][0], acc[c][1], p2[kb], v0, v1);
        mma_16x32<false, false>(acc[c][0], acc[c][1], p3[kb], v0, v1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

inline void sparse_clear(thread frag<float> (&acc)[4][2], thread float (&m)[2], thread float (&l)[2]) {
  TF_UNROLL
  for (short c = 0; c < 4; c++) {
    acc[c][0] = frag<float>(0);
    acc[c][1] = frag<float>(0);
  }
  m[0] = m[1] = NO_SCORE;
  l[0] = l[1] = 0.0f;
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
  frag<bfloat> q[8];
  sparse_queries(q, ql + (long(row) * HEADS + h0) * RANK + d0, home);
  frag<float> acc[4][2];
  float m[2], l[2]; // the running max and sum of this lane's two heads
  sparse_clear(acc, m, l);
  sparse_pass(acc, m, l, q, keys, indices + long(row) * width, 0, width, key_length, scale, partial, keyrow, s, lane, home, d0);
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

// Decode rows: block b's `span` list entries of each row, unnormalized, into po [rows, blocks, 64, 512] fp32 and its max and sum into pm, pl [rows, blocks, 64]; grid (4 head groups, blocks, rows).
[[kernel]] void glm_sparse_split(const device bfloat* ql [[buffer(0)]], const device bfloat* keys [[buffer(1)]],
                                 const device int32_t* indices [[buffer(2)]], constant float& scale [[buffer(3)]],
                                 constant int4& meta [[buffer(4)]], device float* po [[buffer(5)]], device float* pm [[buffer(6)]],
                                 device float* pl [[buffer(7)]], uint s [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]],
                                 uint3 grid [[threadgroups_per_grid]]) {
  threadgroup float partial[4][32][16];
  threadgroup int keyrow[BLOCK];
  const int width = meta.x, key_length = meta.y, span = meta.z;
  const int row = int(tg.z), b = int(tg.y), blocks = int(grid.y), h0 = int(tg.x) * GROUP, d0 = int(s) * SLICE;
  const short2 home = frag_home(ushort(lane));
  frag<bfloat> q[8];
  sparse_queries(q, ql + (long(row) * HEADS + h0) * RANK + d0, home);
  frag<float> acc[4][2];
  float m[2], l[2];
  sparse_clear(acc, m, l);
  const int lo = b * span;
  sparse_pass(acc, m, l, q, keys, indices + long(row) * width, lo, min(width, lo + span), key_length, scale, partial, keyrow, s, lane, home, d0);
  const long at = (long(row) * blocks + b) * HEADS + h0; // this block's first head
  TF_UNROLL
  for (short c = 0; c < 4; c++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) frag_put(acc[c][j], po + at * RANK + d0, RANK, 0, 32 * c + 16 * j, home);
  }
  if (s == 0 && home.x == 0) {
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      pm[at + home.y + 8 * h] = m[h];
      pl[at + home.y + 8 * h] = l[h];
    }
  }
}

// Decode rows: each head's blocks combined in block order, out [rows, 64, 512] bf16; grid (64 heads, rows), 128 threads of 4 dims.
[[kernel]] void glm_sparse_combine(const device float* po [[buffer(0)]], const device float* pm [[buffer(1)]],
                                   const device float* pl [[buffer(2)]], constant int& blocks [[buffer(3)]],
                                   device bfloat* out [[buffer(4)]], uint t [[thread_index_in_threadgroup]],
                                   uint2 tg [[threadgroup_position_in_grid]]) {
  const int head = int(tg.x), row = int(tg.y);
  const long first = long(row) * blocks * HEADS + head; // block 0's (row, head) entry; block b's is HEADS * b further
  float most = NO_SCORE;
  for (int b = 0; b < blocks; b++) most = max(most, pm[first + long(b) * HEADS]);
  float total = 0.0f;
  float4 o = float4(0.0f);
  for (int b = 0; b < blocks; b++) {
    const long at = first + long(b) * HEADS;
    const float f = fast::exp(pm[at] - most);
    total += pl[at] * f;
    o += *(const device float4*)(po + at * RANK + 4 * t) * f;
  }
  const float4 v = total == 0.0f ? float4(0.0f) : o / total;
  *(device bfloat4*)(out + (long(row) * HEADS + head) * RANK + 4 * t) = bfloat4(v);
}
