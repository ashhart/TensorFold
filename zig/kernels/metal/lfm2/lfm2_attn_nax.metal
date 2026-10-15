// LFM2 prompt attention on the tensor units: lfm2_attn_mma's cuts and merge, after lfm2.metal with nax.h inlined.
#include "../nax.h"
using namespace tfp;

constant constexpr uint NP = 72;  // bf16 a staged key or value row (64 and pad)

// A thread's share of a 32-key block among 128 threads: 8 dims of a key's K or V (past the cache end: its last).
inline void lfm2_nax_fetch(const device bfloat* kb, const device bfloat* vb, uint j0, uint cap, uint tid,
                           thread uint4* pre) {
  for (uint i = 0; i < 4; ++i) {
    const uint idx = tid + 128 * i, rem = idx % 256;
    const size_t at = size_t(min(j0 + rem / 8, cap - 1)) * HD + (rem % 8) * 8;
    pre[i] = *(const device uint4*)((idx < 256 ? kb : vb) + at);
  }
}

inline void lfm2_nax_stage(const thread uint4* pre, threadgroup bfloat* ks, threadgroup bfloat* vs, uint tid) {
  for (uint i = 0; i < 4; ++i) {
    const uint idx = tid + 128 * i, rem = idx % 256;
    *(threadgroup uint4*)((idx < 256 ? ks : vs) + (rem / 8) * NP + (rem % 8) * 8) = pre[i];
  }
}

// A row's max or sum over its 32 scores: a lane holds 8, the lanes 1 and 8 apart the rest.
inline float lfm2_nax_rowmax(float v) {
  v = max(v, simd_shuffle_xor(v, ushort(1)));
  return max(v, simd_shuffle_xor(v, ushort(8)));
}
inline float lfm2_nax_rowsum(float v) {
  v = v + simd_shuffle_xor(v, ushort(1));
  return v + simd_shuffle_xor(v, ushort(8));
}

// A threadgroup a (chunk, kv head, 16 rows): its four simdgroups are the kv head's query heads sharing staged K and V.
[[kernel]] void lfm2_attn_nax(const device bfloat* queries [[buffer(0)]], const device bfloat* keys [[buffer(1)]],
                              const device bfloat* values [[buffer(2)]], constant lfm2_attn_args& p [[buffer(3)]],
                              device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
                              device float* po [[buffer(6)]],
                              constant uint& rows [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat ks[2][KB * NP], vs[2][KB * NP];
  const uint chunk = tg.z, kh = tg.y, r0 = tg.x * 16, head = kh * (QH / KVH) + sg;
  const short2 home = frag_home(ushort(lane));
  const uint row0 = min(r0 + home.y, rows - 1), row1 = min(r0 + home.y + 8, rows - 1);  // this lane's rows
  const uint count0 = p.len + row0 + 1, count1 = p.len + row1 + 1;  // their visible keys
  const uint top = p.len + min(r0 + 15, rows - 1) + 1;  // the tile's most
  if (chunk * PER >= top) return;
  frag<bfloat> q[4];  // scaled once: the scale is a power of two (head_dim 64), so each product is exact
  for (int d = 0; d < 4; ++d) {
    const device bfloat* q0 = queries + (size_t(row0) * QH + head) * HD;
    const device bfloat* q1 = queries + (size_t(row1) * QH + head) * HD;
    frag_get_rows(q[d], q0, q1, 16 * d, home, true, true);
    for (short e = 0; e < 8; ++e) q[d][e] = bfloat(float(q[d][e]) * p.scale);
  }
  const device bfloat* kb = keys + size_t(kh) * p.cap * HD;
  const device bfloat* vb = values + size_t(kh) * p.cap * HD;
  float M[2] = {-INFINITY, -INFINITY}, L[2] = {0.0f, 0.0f};
  frag<float> O[4];
  for (int i = 0; i < 4; ++i) O[i] = frag<float>(0.0f);
  for (uint sub = 0; sub < SUBS; ++sub) {
    const uint first = chunk * PER + sub * SUB, last = min(top, first + SUB);
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    frag<float> o[4];
    for (int i = 0; i < 4; ++i) o[i] = frag<float>(0.0f);
    uint4 pre[4];
    lfm2_nax_fetch(kb, vb, first, p.cap, tid, pre);
    lfm2_nax_stage(pre, ks[0], vs[0], tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint buf = 0;
    for (uint j0 = first; j0 < last; j0 += KB, buf ^= 1) {
      const bool more = j0 + KB < last;
      if (more) lfm2_nax_fetch(kb, vb, j0 + KB, p.cap, tid, pre);
      const threadgroup bfloat* ksb = ks[buf];
      const threadgroup bfloat* vsb = vs[buf];
      frag<float> s[2];  // keys 0..15 and 16..31 of the block
      s[0] = frag<float>(0.0f);
      s[1] = frag<float>(0.0f);
      for (int d = 0; d < 4; ++d) {
        frag<bfloat> k0, k1;
        frag_get_tg(k0, ksb, NP, 0, 16 * d, home);
        frag_get_tg(k1, ksb, NP, 16, 16 * d, home);
        mma_16x32<false, true>(s[0], s[1], q[d], k0, k1);
      }
      // Scores masked by each row's visible keys (skipped where both rows see the block); e < 4 row0's, e >= 4 row1's.
      float bm[2] = {-INFINITY, -INFINITY};
      if (j0 + KB > count0) {
        for (int h = 0; h < 2; ++h)
          for (short e = 0; e < 8; ++e) {
            const uint key = j0 + 16 * h + home.x + TF_COL(e);
            if (key >= (e < 4 ? count0 : count1)) s[h][e] = -INFINITY;
          }
      }
      for (int h = 0; h < 2; ++h)
        for (short e = 0; e < 8; ++e) bm[e >> 2] = max(bm[e >> 2], s[h][e]);
      float alpha[2], m2[2];
      bool live[2];
      for (int r = 0; r < 2; ++r) {
        bm[r] = lfm2_nax_rowmax(bm[r]);
        live[r] = bm[r] != -INFINITY;  // per row: a block with no visible key leaves its state as it is
        m2[r] = live[r] ? max(m[r], bm[r]) : m[r];
        alpha[r] = live[r] ? metal::fast::exp(m[r] - m2[r]) : 1.0f;
      }
      frag<bfloat> pb[2];
      float bs[2] = {0.0f, 0.0f};
      for (int h = 0; h < 2; ++h)
        for (short e = 0; e < 8; ++e) {
          const int r = e >> 2;
          const float v = live[r] ? metal::fast::exp(s[h][e] - m2[r]) : 0.0f;
          pb[h][e] = bfloat(v);
          bs[r] += v;
        }
      for (int r = 0; r < 2; ++r) {
        l[r] = fma(l[r], alpha[r], lfm2_nax_rowsum(bs[r]));
        m[r] = m2[r];
      }
      for (int i = 0; i < 4; ++i)
        for (short e = 0; e < 8; ++e) o[i][e] *= alpha[e >> 2];
      for (int h = 0; h < 2; ++h) {
        frag<bfloat> v0, v1, v2, v3;
        frag_get_tg(v0, vsb, NP, 16 * h, 0, home);
        frag_get_tg(v1, vsb, NP, 16 * h, 16, home);
        frag_get_tg(v2, vsb, NP, 16 * h, 32, home);
        frag_get_tg(v3, vsb, NP, 16 * h, 48, home);
        mma_16x32<false, false>(o[0], o[1], pb[h], v0, v1);
        mma_16x32<false, false>(o[2], o[3], pb[h], v2, v3);
      }
      if (more) lfm2_nax_stage(pre, ks[buf ^ 1], vs[buf ^ 1], tid);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int r = 0; r < 2; ++r) {
      float alpha, beta;
      lfm2_merge(M[r], L[r], m[r], l[r], alpha, beta);
      for (int i = 0; i < 4; ++i)
        for (short e = 0; e < 4; ++e) O[i][4 * r + e] = fma(O[i][4 * r + e], alpha, o[i][4 * r + e] * beta);
    }
  }
  for (int r = 0; r < 2; ++r) {
    const uint row = r0 + home.y + 8 * r;
    if (row >= rows) return;
    const size_t at = (size_t(row) * QH + head) * (CHUNKS * p.subs) + chunk * p.subs;
    if (home.x == 0) {
      pm[at] = M[r];
      pl[at] = L[r];
    }
    for (int i = 0; i < 4; ++i)
      for (short e = 0; e < 4; ++e) po[at * HD + 16 * i + home.x + TF_COL(e)] = O[i][4 * r + e];
  }
}
