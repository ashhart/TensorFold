// DSV41's prompt sparse attention on the tensor units: GLM's fragment structure over the packed cache.
// Compiled by csa2.zig after kv.metal (the exact packed-cache decoders) and csa2.metal (Dsv41Attention);
// nax.h is inlined at load, so the same text also builds for GPUs without tensor units (per-element decode).
#include "../nax.h"
using namespace tfp;

#ifndef DSV41_HEADS
#define DSV41_HEADS 64
#endif
constant constexpr int HEADS = DSV41_HEADS, RANK = 512, GROUP = 16, BLOCK = 32, SLICE = RANK / 4;
constant constexpr float NO_SCORE = -INFINITY; // the scalar kernel's score for a missing key

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
inline short p_col(short e, short2 home) { return home.x + (e & 3); }

// The max (or sum) of a head's values across the four lanes that hold its other keys.
inline float row_max(float v) {
  v = max(v, simd_shuffle_xor(v, ushort(1)));
  return max(v, simd_shuffle_xor(v, ushort(8)));
}
inline float row_sum(float v) {
  v += simd_shuffle_xor(v, ushort(1));
  return v + simd_shuffle_xor(v, ushort(8));
}

// The scalar dsv41_attn_key's slot as one int: -1 for no key, bit 30 set with the row index for a compressed
// one, the bare position for a window one. The window and pick rules are csa2.metal's, byte for byte.
inline int dsv41_slot_at(constant Dsv41Attention &p, uint row, uint query, uint columns, uint slot,
                         device const int *indices) {
  if (slot < columns) {
    uint pos;
    if (query == 0) {
      pos = 0;
    } else if (query + 1 < p.window) {
      pos = (query + 1 + slot) % p.window;
    } else {
      uint begin = query + 1 > p.window ? query + 1 - p.window : 0;
      pos = begin + slot;
    }
    if (!p.chunk_all && pos > query) return -1;
    return int(pos);
  }
  if (!p.ratio) return -1;
  int pos = indices[row * p.index_stride + slot - columns];
  if (pos < 0 || uint(pos) >= min(p.compressed_rows, (query + 1) / p.ratio)) return -1;
  return int(0x40000000u | uint(pos));
}

// A slot descriptor's packed row and its kind (0: the window's e4 rows, 1: the compressed e2 rows).
inline device const uchar *dsv41_key_ptr(int enc, device const uchar *ring, device const uchar *chunk,
                                         device const uchar *main, constant Dsv41Attention &p, thread uint &kind) {
  if (enc & (1 << 30)) {
    kind = 1;
    return main + uint(enc & 0x3FFFFFFF) * p.main_stride;
  }
  kind = 0;
  const uint pos = uint(enc);
  return pos >= p.position ? chunk + (pos - p.position) * p.chunk_stride
                           : ring + (pos % p.window) * p.ring_stride;
}

// One dim's exact packed decode through a slot descriptor (dsv41_unpack with the slot's block).
inline bfloat dsv41_decode_at(int enc, device const uchar *ring, device const uchar *chunk,
                              device const uchar *main, uint d, constant Dsv41Attention &p) {
  if (enc < 0) return bfloat(0);
  uint kind;
  device const uchar *k = dsv41_key_ptr(enc, ring, chunk, main, p, kind);
  return bfloat(dsv41_unpack(k, d, p.width, kind != 0u ? p.main_block : 32u, kind));
}

// Aligned-when-possible small vector loads: the packed rows keep every dec4 address a multiple of 4 (or 2 for
// the nibble rows) in practice, but only the pointer says so for sure.
inline uchar4 dsv41_load4(device const uchar *k) {
  if (((unsigned long)k & 3u) == 0u) return *(const device uchar4 *)k;
  return uchar4(k[0], k[1], k[2], k[3]);
}
inline uchar2 dsv41_load2(device const uchar *k) {
  if (((unsigned long)k & 1u) == 0u) return *(const device uchar2 *)k;
  return uchar2(k[0], k[1]);
}

// Four consecutive dims of one key as the scalar decodes them: every value rounded to bf16 exactly, so the
// mma's bf16 products are the scalar's products. Missing keys decode to zero (their score is NO_SCORE anyway).
inline bfloat4 dsv41_dec4(int enc, device const uchar *ring, device const uchar *chunk, device const uchar *main,
                          uint d, constant Dsv41Attention &p) {
  if (enc < 0) return bfloat4(float4(0.0f));
  uint kind;
  device const uchar *k = dsv41_key_ptr(enc, ring, chunk, main, p, kind);
  if (kind == 0) {
    const float s = as_type<float>(uint(k[p.width + (d >> 5)]) << 23);
    const uchar4 c = dsv41_load4(k + d);
    bfloat4 v;
    TF_UNROLL
    for (short i = 0; i < 4; i++) v[i] = bfloat(dsv41_bf(dsv41_round(dsv41_e4(c[i]) * s)));
    return v;
  }
  const float s = dsv41_e4(k[p.width / 2 + (d >> 4)]);
  const uchar2 c = dsv41_load2(k + d / 2); // d is even on every dec4 call (d0, the strides and home.x are multiples of 4)
  bfloat4 v;
  v[0] = bfloat(dsv41_bf(dsv41_round(dsv41_e2(c[0] & 15) * s)));
  v[1] = bfloat(dsv41_bf(dsv41_round(dsv41_e2(c[0] >> 4) * s)));
  v[2] = bfloat(dsv41_bf(dsv41_round(dsv41_e2(c[1] & 15) * s)));
  v[3] = bfloat(dsv41_bf(dsv41_round(dsv41_e2(c[1] >> 4) * s)));
  return v;
}

// The scalar epilogue's store: each value rounded to bf16 with dsv41_round, never a float conversion.
inline void dsv41_frag_put(thread const frag<float> &f, device ushort *p, int ld, int r, int c, short2 home) {
  device ushort *q = p + (r + home.y) * ld + (c + home.x);
  TF_UNROLL
  for (short e = 0; e < 8; e++) q[(e >> 2) * 8 * ld + TF_COL(e)] = dsv41_round(f[e]);
}


// Dense KV: each visible key decoded once per layer, so attention reads bf16 instead of unpacking per row.
struct DenseKv { uint rows, width, stride, first_row, source; };
static_assert(sizeof(DenseKv) == 20, "DenseKv must mirror csa2.zig's DenseKv");
kernel void dsv41_dense_kv(const device uchar *src [[buffer(0)]], device ushort *dense [[buffer(1)]],
                           constant DenseKv &d [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= d.rows || gid.y * 4 >= d.width) return;
  const device uchar *k = src + ulong(gid.x) * d.stride;
  const uint dd = gid.y * 4;
  device ushort *o = dense + ulong(d.first_row + gid.x) * d.width + dd;
  bfloat4 v;
  if (d.source == 0) {
    const float s = as_type<float>(uint(k[d.width + (dd >> 5)]) << 23);
    const uchar4 c = *(const device uchar4 *)(k + dd);
    TF_UNROLL
    for (short i = 0; i < 4; i++) v[i] = bfloat(dsv41_bf(dsv41_round(dsv41_e4(c[i]) * s)));
  } else {
    const float s = dsv41_e4(k[d.width / 2 + (dd >> 4)]);
    const uchar2 c = *(const device uchar2 *)(k + dd / 2);
    v[0] = bfloat(dsv41_bf(dsv41_round(dsv41_e2(c[0] & 15) * s)));
    v[1] = bfloat(dsv41_bf(dsv41_round(dsv41_e2(c[0] >> 4) * s)));
    v[2] = bfloat(dsv41_bf(dsv41_round(dsv41_e2(c[1] & 15) * s)));
    v[3] = bfloat(dsv41_bf(dsv41_round(dsv41_e2(c[1] >> 4) * s)));
  }
  *(device bfloat4 *)o = v;
}

// The dense row a slot descriptor names: the window ring, this chunk, or the compressed picks.
inline ulong dsv41_dense_row(int enc, constant Dsv41Attention &p) {
  if (enc & (1 << 30)) return ulong(p.window) + ulong(p.chunk_rows) + uint(enc & 0x3FFFFFFF);
  const uint pos = uint(enc);
  return pos >= p.position ? ulong(p.window) + (pos - p.position) : ulong(pos % p.window);
}

inline bfloat4 dsv41_dense4(int enc, device const ushort *dense, uint d, constant Dsv41Attention &p) {
  if (enc < 0) return bfloat4(float4(0.0f));
  return *(const device bfloat4 *)(dense + dsv41_dense_row(enc, p) * p.width + d);
}

inline bfloat dsv41_dense_at(int enc, device const ushort *dense, uint d, constant Dsv41Attention &p) {
  if (enc < 0) return bfloat(0);
  return as_type<bfloat>(dense[dsv41_dense_row(enc, p) * p.width + d]);
}

// The same loads from a row offset a block staged once (see `keyoff`): no branchy resolution a fragment.
inline bfloat4 dsv41_dense4_off(int enc, uint off, device const ushort *dense, uint d) {
  if (enc < 0) return bfloat4(float4(0.0f));
  return *(const device bfloat4 *)(dense + ulong(off) + d);
}

inline bfloat dsv41_dense_at_off(int enc, uint off, device const ushort *dense, uint d) {
  if (enc < 0) return bfloat(0);
  return as_type<bfloat>(dense[ulong(off) + d]);
}

// A prompt chunk's rows: 16 heads at a time, 32 keys at a time, out [rows, heads, 512] bf16 over each row's
// window plus its compressed picks. grid (head groups, rows); one threadgroup of four simdgroups per row.
[[kernel]] void dsv41_sparse_nax(device const ushort *q [[buffer(0)]], device const uchar *ring [[buffer(1)]],
                                 device const uchar *chunk [[buffer(2)]], device const uchar *main [[buffer(3)]],
                                 device const int *indices [[buffer(4)]], device const float *sink [[buffer(5)]],
                                 device ushort *out [[buffer(6)]], constant Dsv41Attention &p [[buffer(7)]],
                                 device const ushort *dense [[buffer(8)]], device float *parts [[buffer(9)]],
                                 uint s [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                 uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float partial[4][32][16]; // each simdgroup's scores, by lane (every simdgroup's lanes hold the same cells)
  threadgroup int keyrow[BLOCK];
  threadgroup uint keyoff[BLOCK]; // the row's offset into `dense`, resolved once a key
  if (tg.x * GROUP >= p.heads || tg.y >= p.rows) return;
  const uint row = tg.y, query = p.chunk_all ? p.position : p.position + row;
  const uint columns = query == 0 ? 1u : p.window; // dsv41_window_columns
  const uint compressed = p.ratio ? min(p.topk, min(p.compressed_rows, (query + 1) / p.ratio)) : 0u;
  const uint count = columns + compressed;
  const int h0 = int(tg.x) * GROUP, d0 = int(s) * SLICE;
  const short2 home = frag_home(ushort(lane));
  // this simdgroup's 128 dims of its 16 heads' queries
  const device bfloat *qb = reinterpret_cast<device const bfloat *>(q + ulong(row) * p.q_stride + ulong(h0) * RANK + uint(d0));
  frag<bfloat> qf[8];
  TF_UNROLL
  for (short t = 0; t < 8; t++) frag_get(qf[t], qb, RANK, 0, 16 * t, home);
  frag<float> acc[4][2];
  float m[2], l[2];
  TF_UNROLL
  for (short c = 0; c < 4; c++) {
    acc[c][0] = frag<float>(0);
    acc[c][1] = frag<float>(0);
  }
  m[0] = m[1] = -1e30f;
  l[0] = l[1] = 0.0f;
  // A decode layer's whole key scan would run four threadgroups; a split takes a slice of it and the merge
  // kernel sums the online-softmax parts in split order.
  const uint split = tg.z, per = (count + p.splits - 1) / max(p.splits, 1u);
  const uint j_begin = split * per, j_end = min(count, j_begin + per);
  for (uint j0 = j_begin; j0 < j_end; j0 += BLOCK) {
    // One lane a key: the same values, a thirty-second of the work (every lane used to run all BLOCK steps).
    // The row offset a key's loads need is resolved here, once, beside the slot -- it used to be recomputed by
    // every load of every fragment (4,480 times a block).
    if (s == 0 && lane < uint(BLOCK)) {
      const int enc = j0 + lane < count ? dsv41_slot_at(p, row, query, columns, j0 + lane, indices) : -1;
      keyrow[lane] = enc;
      keyoff[lane] = enc < 0 ? 0u : uint(dsv41_dense_row(enc, p) * p.width);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // scores: this simdgroup's 128 dims of the block's 32 keys
    frag<float> lo_s = frag<float>(0), hi_s = frag<float>(0);
#ifndef TF_SIMD_FRAGS
    TF_UNROLL
    for (short t = 0; t < 8; t++) {
      const uint d = uint(d0) + 16u * uint(t) + uint(home.x);
      const bfloat4 a = dsv41_dense4_off(keyrow[home.y], keyoff[home.y], dense, d);
      const bfloat4 b = dsv41_dense4_off(keyrow[short(home.y + 8)], keyoff[short(home.y + 8)], dense, d);
      const bfloat4 c = dsv41_dense4_off(keyrow[short(16 + home.y)], keyoff[short(16 + home.y)], dense, d);
      const bfloat4 e = dsv41_dense4_off(keyrow[short(24 + home.y)], keyoff[short(24 + home.y)], dense, d);
      mma_16x32<false, true>(lo_s, hi_s, qf[t], frag<bfloat>(a, b), frag<bfloat>(c, e));
    }
#else
    TF_UNROLL
    for (short t = 0; t < 8; t++) {
      frag<bfloat> b0, b1;
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int k0 = keyrow[t_key(e, home)], k1 = keyrow[16 + t_key(e, home)];
        const uint d = uint(d0) + 16u * uint(t) + uint(t_dim(e, home));
        b0[e] = dsv41_dense_at_off(k0, keyoff[t_key(e, home)], dense, d);
        b1[e] = dsv41_dense_at_off(k1, keyoff[16 + t_key(e, home)], dense, d);
      }
      mma_16x32<false, true>(lo_s, hi_s, qf[t], b0, b1);
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
      sc[i] = keyrow[key] >= 0 ? v * p.scale : NO_SCORE;
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
    frag<bfloat> p1[2]; // the probabilities' one bf16 part, exactly the scalar's dsv41_round
    float bsum[2] = {0.0f, 0.0f};
    TF_UNROLL
    for (short i = 0; i < 16; i++) {
      const short e = i & 7, h = e >> 2, kb = i < 8 ? 0 : 1;
      const float pp = sc[i] > NO_SCORE ? fast::exp(sc[i] - m[h]) : 0.0f;
      bsum[h] += pp;
      p1[kb][e] = bfloat(dsv41_bf(dsv41_round(pp)));
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
      TF_UNROLL
      for (short c = 0; c < 4; c++) {
        const uint d = uint(d0) + 32u * uint(c) + uint(home.x);
#ifndef TF_SIMD_FRAGS
        // The dense pass already rounded every key this window reads, so the values are four loads and not four
        // packed decodes: same bytes, a thirteenth of the arithmetic.
        const bfloat4 a = dsv41_dense4_off(keyrow[16 * kb + home.y], keyoff[16 * kb + home.y], dense, d);
        const bfloat4 b = dsv41_dense4_off(keyrow[16 * kb + short(home.y + 8)], keyoff[16 * kb + short(home.y + 8)], dense, d);
        frag<bfloat> v0(a, b);
        const bfloat4 a2 = dsv41_dense4_off(keyrow[16 * kb + home.y], keyoff[16 * kb + home.y], dense, d + 16);
        const bfloat4 b2 = dsv41_dense4_off(keyrow[16 * kb + short(home.y + 8)], keyoff[16 * kb + short(home.y + 8)], dense, d + 16);
        frag<bfloat> v1(a2, b2);
#else
        frag<bfloat> v0, v1;
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          const int k = keyrow[16 * kb + p_row(e, home)];
          const uint dd = uint(d0) + 32u * uint(c) + uint(p_col(e, home));
          v0[e] = dsv41_decode_at(k, ring, chunk, main, dd, p);
          v1[e] = dsv41_decode_at(k, ring, chunk, main, dd + 16, p);
        }
#endif
        mma_16x32<false, false>(acc[c][0], acc[c][1], p1[kb], v0, v1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (p.splits > 1) {
    // This split's (max, sum, accumulators) by head group, then split order; a decode window is one row.
    // Each of the four SIMDgroups holds its own 128-dimensional slice, so the parts are keyed by SIMDgroup too.
    device float *part = parts + ((((ulong(tg.y) * ulong(p.splits) + ulong(split)) * ulong(p.heads / GROUP) + ulong(tg.x)) * 4 + ulong(s)) * 32 + ulong(lane)) * 68;
    part[0] = m[0];
    part[1] = m[1];
    part[2] = l[0];
    part[3] = l[1];
    TF_UNROLL
    for (short c = 0; c < 4; c++)
      TF_UNROLL
      for (short j = 0; j < 2; j++)
        TF_UNROLL
        for (short e = 0; e < 8; e++) part[4 + c * 16 + j * 8 + e] = acc[c][j][e];
    return;
  }
  // The source adds sink mass to the completed online denominator before the division; every store is dsv41_round.
  device ushort *obase = out + ulong(row) * p.q_stride + ulong(h0) * RANK + uint(d0);
  TF_UNROLL
  for (short c = 0; c < 4; c++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      frag<float> o = acc[c][j];
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const short h = e >> 2;
        const float denom = l[h] + fast::exp(sink[ulong(h0) + uint(home.y) + 8u * uint(h)] - m[h]);
        o[e] = denom == 0.0f ? 0.0f : o[e] / denom;
      }
      dsv41_frag_put(o, obase, RANK, 0, 32 * c + 16 * j, home);
    }
  }
}

// The same pass over only the compressed rows a one-row window actually reads: the picks name the positions, so
// the dense rows keep their position indexing and the sparse kernel is unchanged.
kernel void dsv41_dense_kv_picked(const device uchar *src [[buffer(0)]], device ushort *dense [[buffer(1)]],
                                 constant DenseKv &d [[buffer(2)]], const device int *picks [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= d.rows || gid.y * 4 >= d.width) return;
  const int pos = picks[gid.x];
  if (pos < 0) return;
  const device uchar *k = src + ulong(pos) * d.stride;
  const uint dd = gid.y * 4;
  device ushort *o = dense + ulong(d.first_row + uint(pos)) * d.width + dd;
  bfloat4 v;
  const float s = dsv41_e4(k[d.width / 2 + (dd >> 4)]);
  const uchar2 c = *(const device uchar2 *)(k + dd / 2);
  v[0] = bfloat(dsv41_bf(dsv41_round(dsv41_e2(c[0] & 15) * s)));
  v[1] = bfloat(dsv41_bf(dsv41_round(dsv41_e2(c[0] >> 4) * s)));
  v[2] = bfloat(dsv41_bf(dsv41_round(dsv41_e2(c[1] & 15) * s)));
  v[3] = bfloat(dsv41_bf(dsv41_round(dsv41_e2(c[1] >> 4) * s)));
  *(device bfloat4 *)o = v;
}

// THE MERGE'S GRID IS FOUR THREADGROUPS. This form gives each threadgroup ONE of the four accumulator chunks: the
// grid is four times wider (16 threadgroups), each lane reads its own chunk's sixteen floats per split (the same
// traffic, the same split order, the same per output sums) and the fragment's dim offset is the chunk's own -- so
// the output is BIT-IDENTICAL.
[[kernel]] void dsv41_sparse_merge_split(device const float *parts [[buffer(0)]], device ushort *out [[buffer(1)]],
                                         device const float *sink [[buffer(2)]], constant Dsv41Attention &p [[buffer(3)]],
                                         uint s [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                         uint3 tg [[threadgroup_position_in_grid]]) {
  if (tg.x * GROUP >= p.heads || tg.y >= p.rows) return;
  const uint row = tg.y, h0 = tg.x * GROUP, d0 = s * SLICE;
  const uint c = tg.z;                       // this threadgroup's accumulator chunk
  const short2 home = frag_home(ushort(lane));
  const ulong groups = ulong(p.heads / GROUP);
  float m[2] = {-1e30f, -1e30f};
  for (uint sp = 0; sp < p.splits; ++sp) {
    const device float *part = parts + ((((ulong(row) * p.splits + ulong(sp)) * groups + ulong(tg.x)) * 4 + ulong(s)) * 32 + ulong(lane)) * 68;
    m[0] = max(m[0], part[0]);
    m[1] = max(m[1], part[1]);
  }
  float l[2] = {0.0f, 0.0f};
  frag<float> acc[2];
  TF_UNROLL
  for (short j = 0; j < 2; j++) {
    acc[j] = frag<float>(0);
    TF_UNROLL
    for (short e = 0; e < 8; e++) acc[j][e] = 0.0f;
  }
  for (uint sp = 0; sp < p.splits; ++sp) {
    const device float *part = parts + ((((ulong(row) * p.splits + ulong(sp)) * groups + ulong(tg.x)) * 4 + ulong(s)) * 32 + ulong(lane)) * 68;
    float f[2];
    TF_UNROLL
    for (short h = 0; h < 2; ++h) {
      f[h] = fast::exp(part[h] - m[h]);
      l[h] += part[2 + h] * f[h];
    }
    TF_UNROLL
    for (short j = 0; j < 2; j++)
      TF_UNROLL
      for (short e = 0; e < 8; e++) acc[j][e] += part[4 + c * 16 + j * 8 + e] * f[e >> 2];
  }
  device ushort *obase = out + ulong(row) * p.q_stride + ulong(h0) * RANK + uint(d0);
  TF_UNROLL
  for (short j = 0; j < 2; j++) {
    frag<float> o = acc[j];
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      const short h = e >> 2;
      const float denom = l[h] + fast::exp(sink[ulong(h0) + uint(home.y) + 8u * uint(h)] - m[h]);
      o[e] = denom == 0.0f ? 0.0f : o[e] / denom;
    }
    dsv41_frag_put(o, obase, RANK, 0, 32 * c + 16 * j, home);
  }
}

// A decode window's splits, summed in split order, then the sink and the division the plain path does.
[[kernel]] void dsv41_sparse_merge(device const float *parts [[buffer(0)]], device ushort *out [[buffer(1)]],
                                  device const float *sink [[buffer(2)]], constant Dsv41Attention &p [[buffer(3)]],
                                  uint s [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                  uint3 tg [[threadgroup_position_in_grid]]) {
  if (tg.x * GROUP >= p.heads || tg.y >= p.rows) return;
  const uint row = tg.y, h0 = tg.x * GROUP, d0 = s * SLICE;
  const short2 home = frag_home(ushort(lane));
  const ulong groups = ulong(p.heads / GROUP);
  const uint base = lane * 68;
  float m[2] = {-1e30f, -1e30f};
  for (uint sp = 0; sp < p.splits; ++sp) {
    const device float *part = parts + ((((ulong(row) * p.splits + ulong(sp)) * groups + ulong(tg.x)) * 4 + ulong(s)) * 32 + ulong(lane)) * 68;
    m[0] = max(m[0], part[0]);
    m[1] = max(m[1], part[1]);
  }
  float l[2] = {0.0f, 0.0f};
  frag<float> acc[4][2];
  TF_UNROLL
  for (short c = 0; c < 4; c++)
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      acc[c][j] = frag<float>(0);
      TF_UNROLL
      for (short e = 0; e < 8; e++) acc[c][j][e] = 0.0f;
    }
  for (uint sp = 0; sp < p.splits; ++sp) {
    const device float *part = parts + ((((ulong(row) * p.splits + ulong(sp)) * groups + ulong(tg.x)) * 4 + ulong(s)) * 32 + ulong(lane)) * 68;
    float f[2];
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      f[h] = fast::exp(part[h] - m[h]);
      l[h] += part[2 + h] * f[h];
    }
    TF_UNROLL
    for (short c = 0; c < 4; c++)
      TF_UNROLL
      for (short j = 0; j < 2; j++)
        TF_UNROLL
        for (short e = 0; e < 8; e++) acc[c][j][e] += part[4 + c * 16 + j * 8 + e] * f[e >> 2];
  }
  device ushort *obase = out + ulong(row) * p.q_stride + ulong(h0) * RANK + uint(d0);
  TF_UNROLL
  for (short c = 0; c < 4; c++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      frag<float> o = acc[c][j];
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const short h = e >> 2;
        const float denom = l[h] + fast::exp(sink[ulong(h0) + uint(home.y) + 8u * uint(h)] - m[h]);
        o[e] = denom == 0.0f ? 0.0f : o[e] / denom;
      }
      dsv41_frag_put(o, obase, RANK, 0, 32 * c + 16 * j, home);
    }
  }
}
