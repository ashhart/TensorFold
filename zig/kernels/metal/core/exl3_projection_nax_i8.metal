// Native EXL3 on the tensor units: the same stored trellis, decoded into a threadgroup tile and reduced by
// cooperative matmul instead of per-lane scalar FMAs. The decode is unchanged (measured free); only the reduction moves.
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"
using namespace tfp;

struct TfExl3Args {
  uint k, n, rows, halves;
  uint codebook, su_signs, sv_signs, has_bias;
  uint output_f32, input_ids, output_ids, routed;
  uint expert_id, repeat, indexed, experts, splits;
  ulong x_stride, y_stride, tile_k_stride, tile_n_stride;
  uint pair;
};

// Metal 3 / argument-buffer tier 2 descriptors are populated once from resident buffers' GPU addresses.
struct TfExl3Expert {
  const device uchar* trellis;
  const device ushort* su;
  const device ushort* sv;
  const device half* bias;
  ulong tile_k_stride, tile_n_stride;
  uint halves, codebook, su_signs, sv_signs, has_bias, reserved;
};

inline bool tf_exl3_active(const device uint* routes, constant TfExl3Args& a, uint row) {
  return row < a.rows && (a.indexed ? routes[row] < a.experts : (!a.routed || routes[row] == a.expert_id));
}

// Two little-endian int16 words form one MSB-first stream word, including unaligned borrowed tiles.
inline uint tf_exl3_word(const device ushort* tile, uint i) {
  return uint(tile[2 * i]) | (uint(tile[2 * i + 1]) << 16);
}

inline ushort tf_exl3_state(const device ushort* tile, uint halves, uint p) {
  const uint ring = 128 * halves;
  const uint end = ((p + 1) * halves - ((halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
  uint start = end + ring - 16;
  if (start >= ring) start -= ring;
  const uint wi = start >> 5, off = start & 31u;
  const uint lo = tf_exl3_word(tile, wi);
  if (off <= 16) return ushort(lo >> (16 - off));
  const uint next = wi + 1 == 4 * halves ? 0 : wi + 1;
  const uint hi = tf_exl3_word(tile, next);
  return ushort((lo << (off - 16)) | (hi >> (48 - off)));
}

inline ushort tf_exl3_mapped_state(const device ushort* tile, uint map) {
  const uint wi = map & 63u, off = (map >> 6) & 31u;
  const uint lo = tf_exl3_word(tile, wi);
  const uint hi = tf_exl3_word(tile, map >> 11);
  const uint carry = (hi >> ((32 - off) & 31u)) & (0u - uint(off != 0));
  return ushort(((lo << off) | carry) >> 16);
}

// The codebook as a table: the same value tf_exl3_value computes, read instead of derived.
inline half tf_exl3_table_value(const device half* table, ushort state, uint codebook) {
  return table[codebook * 65536u + uint(state)];
}

// The same bitstream read from a tile staged in threadgroup memory: one coalesced pass over 256 bytes, then every
// decoded element hits threadgroup memory instead of a scattered device load.
inline uint tf_exl3_word_tg(const threadgroup ushort* tile, uint i) {
  return uint(tile[2 * i]) | (uint(tile[2 * i + 1]) << 16);
}

inline ushort tf_exl3_mapped_state_tg(const threadgroup ushort* tile, uint map) {
  const uint wi = map & 63u, off = (map >> 6) & 31u;
  const uint lo = tf_exl3_word_tg(tile, wi);
  const uint hi = tf_exl3_word_tg(tile, map >> 11);
  const uint carry = (hi >> ((32 - off) & 31u)) & (0u - uint(off != 0));
  return ushort(((lo << off) | carry) >> 16);
}

// Both operations are exact in FP32 before the single FP16 round; do not split the mul1 FMA.
inline half tf_exl3_value(ushort state, uint codebook) {
  uint x = uint(state) * (codebook == 2 ? 0x83DCD12Du : codebook == 1 ? 0xCBAC1FEDu : 89226354u);
  if (codebook == 2) {
    const uint h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24);
    return half(fma(float(h), float(as_type<half>(ushort(0x1EEE))), float(as_type<half>(ushort(0xC931)))));
  }
  if (codebook == 0) x += 64248484u;
  x = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
  return half(float(as_type<half>(ushort(x))) + float(as_type<half>(ushort(x >> 16))));
}

// Two 8-byte loads, one per fragment row, as nax.h's frag_get_tg does for bfloat: rows home.y and home.y + 8, four
// columns from home.x. `c + home.x` must be a multiple of 4 and `ld` a multiple of 4.
inline void tf_frag_get_tg4(thread frag<half>& f, const threadgroup half* p, int ld, int r, int c, short2 home) {
  const threadgroup half* q = p + (r + home.y) * ld + (c + home.x);
  f = frag<half>(*(const threadgroup half4*)q, *(const threadgroup half4*)(q + 8 * ld));
}

// The bit-stream map for element index p, the same value the old per-lane precompute produced.
inline uint tf_exl3_map_for(uint p, uint halves) {
  const uint ring = 128 * halves;
  const uint end = ((p + 1) * halves - ((halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
  // end + ring - 16 is below twice the ring, so one conditional subtract replaces the modulo by a runtime divisor.
  uint start = end + ring - 16;
  if (start >= ring) start -= ring;
  const uint wi = start >> 5, off = start & 31u, next = wi + 1 == 4 * halves ? 0 : wi + 1;
  return wi | (off << 6) | (next << 11);
}

// A threadgroup owns RG sixteen-row groups by CG sixty-four-column groups, one SIMDgroup a (row group, column
// group). Dense rows share one matrix; grouped rows are sixty-four-row aligned, so one expert owns the tile and
// its weights decode once. A wider column tile re-reads the activation tile for fewer output tiles. `Fast` keeps
// the per-state cursor out of the k loop and reads the three stream words an eight-state block can touch.

// One 16x16 fragment to int8 with its own peak (127 levels): the scalar int8 core's rule, on the tensor path.
inline float tf_frag_quantise(thread frag<int8_t>& q, thread const frag<half>& v) {
  float peak = 0.0f;
  TF_UNROLL
  for (short e = 0; e < 8; e++) peak = max(peak, fabs(float(v[e])));
  peak = simd_max(peak);
  const float s = peak > 0 ? peak / 127.0f : 1.0f;
  TF_UNROLL
  for (short e = 0; e < 8; e++) q[e] = int8_t(clamp(rint(float(v[e]) / s), -127.0f, 127.0f));
  return s;
}

template <bool Fast, uint RG, uint CG>
inline void tf_exl3_prefill_body(threadgroup half* wx, const device half* x, const device uchar* trellis,
                                 device float* sums, constant TfExl3Args& a, const device uint* routes,
                                 const device TfExl3Expert* descriptors,
                                 uint lane, uint sg, uint tid, uint tptg, uint3 tg) {
  const uint trows = RG * 16u;
  const uint rb = tg.y * trows, nb = tg.x * (64u * CG);
  if (a.indexed == 1 || rb >= a.rows) return;
  const uint tile_expert = a.indexed ? routes[rb] : a.expert_id;
  if (a.indexed && tile_expert >= a.experts) return;
  const uint didx = a.indexed ? tile_expert : 0;
  const device uchar* data = a.indexed ? descriptors[didx].trellis : trellis;
  const ulong ks = a.indexed ? descriptors[didx].tile_k_stride : a.tile_k_stride;
  const ulong ns = a.indexed ? descriptors[didx].tile_n_stride : a.tile_n_stride;
  const uint halves = a.indexed ? descriptors[didx].halves : a.halves;
  const uint codebook = a.indexed ? descriptors[didx].codebook : a.codebook;
  const short2 home = frag_home(ushort(lane));
  const uint rows_here = min(trows, a.rows - rb);
  const uint cols_here = min(64u * CG, a.n - nb);
  // Four warps decode a sixty-four-column group; a threadgroup with more warps a group than that splits the sixteen
  // states of each lane between them instead of leaving those warps to wait at the barrier.
  constexpr uint wpg = RG / 4u;                  // warps decoding one column group
  constexpr uint states = 8u / wpg;              // consecutive states a lane owns
  const uint gwarp = tid / 32u, group = gwarp / wpg, wg = gwarp % wpg;
  const uint h = group % 4u, hg = (group / 4u) * 64u;
  const uint p0 = (wg * 32u + tid % 32u) * states;
  const bool decodes = group < 4u * CG and nb + hg + h * 16u < a.n;
  uint at[8], off[8], sh[8], rel[8];
  const uint words = 4 * halves;
  const uint wi0 = tf_exl3_map_for(p0, halves) & 63u;
  if (Fast) {
    const uint ring = 128 * halves;
    const uint cur = (wi0 << 5) | ((tf_exl3_map_for(p0, halves) >> 6) & 31u);
    TF_UNROLL
    for (uint k = 0; k < states; ++k) {
      const uint p = p0 + k, l = p / 8, e = p % 8;
      at[k] = (2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u)) * (64u * CG) + hg + h * 16u + (l / 4 + 8 * (e >> 2));
      uint c8 = cur + k * (halves >> 1);
      if (c8 >= ring) c8 -= ring;
      const uint wi = c8 >> 5;
      off[k] = c8 & 31u;
      sh[k] = (32u - off[k]) & 31u;
      rel[k] = wi >= wi0 ? wi - wi0 : wi + words - wi0;
    }
  }
  const uint sgr = sg % RG, sgc = sg / RG;       // this SIMDgroup's row group and column group
  frag<float> lo0 = frag<float>(0), hi0 = frag<float>(0), lo1 = frag<float>(0), hi1 = frag<float>(0);
  // The four stream words a block needs are loaded a block early: issued before the barrier, their latency overlaps
  // the multiply and the next block's decode instead of stalling at the top of every k block.
  uint p0v, p1v, p2v, p3v;
  if (Fast and decodes) {
    const device ushort* tile0 = reinterpret_cast<const device ushort*>(data + ulong(nb / 16 + hg / 16u + h) * ns);
    uint a1 = wi0 + 1; if (a1 >= words) a1 -= words;
    uint a2 = wi0 + 2; if (a2 >= words) a2 -= words;
    uint a3 = wi0 + 3; if (a3 >= words) a3 -= words;
    p0v = tf_exl3_word(tile0, wi0);
    p1v = tf_exl3_word(tile0, a1);
    p2v = tf_exl3_word(tile0, a2);
    p3v = tf_exl3_word(tile0, a3);
  }
  for (uint kb = 0; kb < a.k; kb += 16) {
    // One thread owns 8 consecutive states of one 16x16 tile. The bit cursor advances by `halves` a state, so it
    // rolls forward with one add and one conditional subtract instead of recomputing the map formula.
    if (decodes) {
      {
        const device ushort* tile = reinterpret_cast<const device ushort*>(
            data + ulong(kb / 16) * ks + ulong(nb / 16 + hg / 16u + h) * ns);
        if (Fast and (halves & 1u) == 0 and halves <= 16) {
          // Eight states span seven steps, so at 8 bits a value their cursors reach three words past the first.
          const uint w0 = p0v, w1 = p1v, w2 = p2v, w3 = p3v;
          const uint nkb = kb + 16;
          if (nkb < a.k) {
            const device ushort* tile_n = reinterpret_cast<const device ushort*>(
                data + ulong(nkb / 16) * ks + ulong(nb / 16 + hg / 16u + h) * ns);
            p0v = tf_exl3_word(tile_n, wi0);
            p1v = tf_exl3_word(tile_n, wi0 + 1 < words ? wi0 + 1 : wi0 + 1 - words);
            p2v = tf_exl3_word(tile_n, wi0 + 2 < words ? wi0 + 2 : wi0 + 2 - words);
            p3v = tf_exl3_word(tile_n, wi0 + 3 < words ? wi0 + 3 : wi0 + 3 - words);
          }
          TF_UNROLL
          for (uint k = 0; k < states; ++k) {
            const uint lo = rel[k] == 0 ? w0 : rel[k] == 1 ? w1 : w2;
            const uint hi = rel[k] == 0 ? w1 : rel[k] == 1 ? w2 : w3;
            const uint carry = (hi >> sh[k]) & (0u - uint(off[k] != 0));
            wx[at[k]] = tf_exl3_value(ushort(((lo << off[k]) | carry) >> 16), codebook);
          }
        } else if (halves <= 16 and (halves & 1u) == 0 and states == 8) {
          const uint ring = 128 * halves;
          uint at1 = wi0 + 1; if (at1 >= words) at1 -= words;
          uint at2 = wi0 + 2; if (at2 >= words) at2 -= words;
          uint at3 = wi0 + 3; if (at3 >= words) at3 -= words;
          uint at4 = wi0 + 4; if (at4 >= words) at4 -= words;
          const uint w0 = tf_exl3_word(tile, wi0);
          const uint w1 = tf_exl3_word(tile, at1), w2 = tf_exl3_word(tile, at2);
          const uint w3 = tf_exl3_word(tile, at3), w4 = tf_exl3_word(tile, at4);
          uint cur = (wi0 << 5) | ((tf_exl3_map_for(p0, halves) >> 6) & 31u);
          TF_UNROLL
          for (uint k = 0; k < 8; ++k) {
            // Each state's cursor comes from the block's origin: the eight decodes are independent.
            uint c8 = cur + k * (halves >> 1);
            if (c8 >= ring) c8 -= ring;
            const uint wi = c8 >> 5, off = c8 & 31u;
            const uint rel = wi >= wi0 ? wi - wi0 : wi + words - wi0;
            const uint lo = rel == 0 ? w0 : rel == 1 ? w1 : rel == 2 ? w2 : w3;
            const uint hi = rel == 0 ? w1 : rel == 1 ? w2 : rel == 2 ? w3 : w4;
            const uint carry = (hi >> ((32 - off) & 31u)) & (0u - uint(off != 0));
            const uint p = p0 + k, l = p / 8, e = p % 8;
            const uint r = 2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u);
            const uint c = l / 4 + 8 * (e >> 2);
            wx[r * (64u * CG) + hg + h * 16 + c] = tf_exl3_value(ushort(((lo << off) | carry) >> 16), codebook);
          }
        } else {
          TF_UNROLL
          for (uint k = 0; k < 8; ++k) {
            const uint p = p0 + k;
            const uint l = p / 8, e = p % 8;
            const uint r = 2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u);
            const uint c = l / 4 + 8 * (e >> 2);
            wx[r * (64u * CG) + hg + h * 16 + c] = tf_exl3_value(tf_exl3_mapped_state(tile, tf_exl3_map_for(p, halves)), codebook);
          }
        }
      }
    }
    // This SIMDgroup's sixteen rows of the k block, read straight into fragments: no staging buffer, one barrier.
    frag<half> av;
    frag_get_in(av, (const device half*)x, int(a.k), int(rb + sgr * 16), int(kb), home, int(rb + rows_here), int(kb) + 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      frag<half> b0, b1;
      const int wc = int(sgc * 64u);
      tf_frag_get_tg4(b0, wx, int(64u * CG), 0, wc, home);
      tf_frag_get_tg4(b1, wx, int(64u * CG), 0, wc + 16, home);
      mma_16x32<false, false>(lo0, hi0, av, b0, b1);
      tf_frag_get_tg4(b0, wx, int(64u * CG), 0, wc + 32, home);
      tf_frag_get_tg4(b1, wx, int(64u * CG), 0, wc + 48, home);
      mma_16x32<false, false>(lo1, hi1, av, b0, b1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const int r0 = int(rb + sgr * 16);
  const int c0 = int(nb + sgc * 64u);
  // `sums` is the whole [rows, n] buffer and frag_put_in indexes it absolutely (rr, cc), so the
  // bounds must be absolute too: the last live row is rb + rows_here and the last live column is
  // nb + cols_here. The relative counts (rows_here - sg*16, cols_here) left every row past 32 and
  // every column past 16 unwritten, which is what made this kernel's answers wrong.
  frag_put_in(lo0, sums, int(a.n), r0, c0, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi0, sums, int(a.n), r0, c0 + 16, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo1, sums, int(a.n), r0, c0 + 32, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi1, sums, int(a.n), r0, c0 + 48, home, int(rb + rows_here), int(nb + cols_here));
}

template <bool Fast, uint RG, uint CG>
inline void tf_exl3_prefill_body_i8(threadgroup half* wx, const device half* x, const device uchar* trellis,
                                 device float* sums, constant TfExl3Args& a, const device uint* routes,
                                 const device TfExl3Expert* descriptors,
                                 uint lane, uint sg, uint tid, uint tptg, uint3 tg) {
  const uint trows = RG * 16u;
  const uint rb = tg.y * trows, nb = tg.x * (64u * CG);
  if (a.indexed == 1 || rb >= a.rows) return;
  const uint tile_expert = a.indexed ? routes[rb] : a.expert_id;
  if (a.indexed && tile_expert >= a.experts) return;
  const uint didx = a.indexed ? tile_expert : 0;
  const device uchar* data = a.indexed ? descriptors[didx].trellis : trellis;
  const ulong ks = a.indexed ? descriptors[didx].tile_k_stride : a.tile_k_stride;
  const ulong ns = a.indexed ? descriptors[didx].tile_n_stride : a.tile_n_stride;
  const uint halves = a.indexed ? descriptors[didx].halves : a.halves;
  const uint codebook = a.indexed ? descriptors[didx].codebook : a.codebook;
  const short2 home = frag_home(ushort(lane));
  const uint rows_here = min(trows, a.rows - rb);
  const uint cols_here = min(64u * CG, a.n - nb);
  // Four warps decode a sixty-four-column group; a threadgroup with more warps a group than that splits the sixteen
  // states of each lane between them instead of leaving those warps to wait at the barrier.
  constexpr uint wpg = RG / 4u;                  // warps decoding one column group
  constexpr uint states = 8u / wpg;              // consecutive states a lane owns
  const uint gwarp = tid / 32u, group = gwarp / wpg, wg = gwarp % wpg;
  const uint h = group % 4u, hg = (group / 4u) * 64u;
  const uint p0 = (wg * 32u + tid % 32u) * states;
  const bool decodes = group < 4u * CG and nb + hg + h * 16u < a.n;
  uint at[8], off[8], sh[8], rel[8];
  const uint words = 4 * halves;
  const uint wi0 = tf_exl3_map_for(p0, halves) & 63u;
  if (Fast) {
    const uint ring = 128 * halves;
    const uint cur = (wi0 << 5) | ((tf_exl3_map_for(p0, halves) >> 6) & 31u);
    TF_UNROLL
    for (uint k = 0; k < states; ++k) {
      const uint p = p0 + k, l = p / 8, e = p % 8;
      at[k] = (2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u)) * (64u * CG) + hg + h * 16u + (l / 4 + 8 * (e >> 2));
      uint c8 = cur + k * (halves >> 1);
      if (c8 >= ring) c8 -= ring;
      const uint wi = c8 >> 5;
      off[k] = c8 & 31u;
      sh[k] = (32u - off[k]) & 31u;
      rel[k] = wi >= wi0 ? wi - wi0 : wi + words - wi0;
    }
  }
  const uint sgr = sg % RG, sgc = sg / RG;       // this SIMDgroup's row group and column group
  frag<float> lo0 = frag<float>(0), hi0 = frag<float>(0), lo1 = frag<float>(0), hi1 = frag<float>(0);
  // The four stream words a block needs are loaded a block early: issued before the barrier, their latency overlaps
  // the multiply and the next block's decode instead of stalling at the top of every k block.
  uint p0v, p1v, p2v, p3v;
  if (Fast and decodes) {
    const device ushort* tile0 = reinterpret_cast<const device ushort*>(data + ulong(nb / 16 + hg / 16u + h) * ns);
    uint a1 = wi0 + 1; if (a1 >= words) a1 -= words;
    uint a2 = wi0 + 2; if (a2 >= words) a2 -= words;
    uint a3 = wi0 + 3; if (a3 >= words) a3 -= words;
    p0v = tf_exl3_word(tile0, wi0);
    p1v = tf_exl3_word(tile0, a1);
    p2v = tf_exl3_word(tile0, a2);
    p3v = tf_exl3_word(tile0, a3);
  }
  for (uint kb = 0; kb < a.k; kb += 16) {
    // One thread owns 8 consecutive states of one 16x16 tile. The bit cursor advances by `halves` a state, so it
    // rolls forward with one add and one conditional subtract instead of recomputing the map formula.
    if (decodes) {
      {
        const device ushort* tile = reinterpret_cast<const device ushort*>(
            data + ulong(kb / 16) * ks + ulong(nb / 16 + hg / 16u + h) * ns);
        if (Fast and (halves & 1u) == 0 and halves <= 16) {
          // Eight states span seven steps, so at 8 bits a value their cursors reach three words past the first.
          const uint w0 = p0v, w1 = p1v, w2 = p2v, w3 = p3v;
          const uint nkb = kb + 16;
          if (nkb < a.k) {
            const device ushort* tile_n = reinterpret_cast<const device ushort*>(
                data + ulong(nkb / 16) * ks + ulong(nb / 16 + hg / 16u + h) * ns);
            p0v = tf_exl3_word(tile_n, wi0);
            p1v = tf_exl3_word(tile_n, wi0 + 1 < words ? wi0 + 1 : wi0 + 1 - words);
            p2v = tf_exl3_word(tile_n, wi0 + 2 < words ? wi0 + 2 : wi0 + 2 - words);
            p3v = tf_exl3_word(tile_n, wi0 + 3 < words ? wi0 + 3 : wi0 + 3 - words);
          }
          TF_UNROLL
          for (uint k = 0; k < states; ++k) {
            const uint lo = rel[k] == 0 ? w0 : rel[k] == 1 ? w1 : w2;
            const uint hi = rel[k] == 0 ? w1 : rel[k] == 1 ? w2 : w3;
            const uint carry = (hi >> sh[k]) & (0u - uint(off[k] != 0));
            wx[at[k]] = tf_exl3_value(ushort(((lo << off[k]) | carry) >> 16), codebook);
          }
        } else if (halves <= 16 and (halves & 1u) == 0 and states == 8) {
          const uint ring = 128 * halves;
          uint at1 = wi0 + 1; if (at1 >= words) at1 -= words;
          uint at2 = wi0 + 2; if (at2 >= words) at2 -= words;
          uint at3 = wi0 + 3; if (at3 >= words) at3 -= words;
          uint at4 = wi0 + 4; if (at4 >= words) at4 -= words;
          const uint w0 = tf_exl3_word(tile, wi0);
          const uint w1 = tf_exl3_word(tile, at1), w2 = tf_exl3_word(tile, at2);
          const uint w3 = tf_exl3_word(tile, at3), w4 = tf_exl3_word(tile, at4);
          uint cur = (wi0 << 5) | ((tf_exl3_map_for(p0, halves) >> 6) & 31u);
          TF_UNROLL
          for (uint k = 0; k < 8; ++k) {
            // Each state's cursor comes from the block's origin: the eight decodes are independent.
            uint c8 = cur + k * (halves >> 1);
            if (c8 >= ring) c8 -= ring;
            const uint wi = c8 >> 5, off = c8 & 31u;
            const uint rel = wi >= wi0 ? wi - wi0 : wi + words - wi0;
            const uint lo = rel == 0 ? w0 : rel == 1 ? w1 : rel == 2 ? w2 : w3;
            const uint hi = rel == 0 ? w1 : rel == 1 ? w2 : rel == 2 ? w3 : w4;
            const uint carry = (hi >> ((32 - off) & 31u)) & (0u - uint(off != 0));
            const uint p = p0 + k, l = p / 8, e = p % 8;
            const uint r = 2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u);
            const uint c = l / 4 + 8 * (e >> 2);
            wx[r * (64u * CG) + hg + h * 16 + c] = tf_exl3_value(ushort(((lo << off) | carry) >> 16), codebook);
          }
        } else {
          TF_UNROLL
          for (uint k = 0; k < 8; ++k) {
            const uint p = p0 + k;
            const uint l = p / 8, e = p % 8;
            const uint r = 2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u);
            const uint c = l / 4 + 8 * (e >> 2);
            wx[r * (64u * CG) + hg + h * 16 + c] = tf_exl3_value(tf_exl3_mapped_state(tile, tf_exl3_map_for(p, halves)), codebook);
          }
        }
      }
    }
    // This SIMDgroup's sixteen rows of the k block, read straight into fragments: no staging buffer, one barrier.
    // int8 body, register-lean: one weight fragment and one int accumulator pair live at a time, each pair flushed
    // into the float accumulators before the registers are reused. The tile and the x block are quantised per 16x16
    // fragment with their own peak/127 scales, the int32 sums are exact, and every k block flushes because its
    // scales differ. This is the trellis skeleton's own decode with the multiply moved to int8 on the tensor units.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      const int wc = int(sgc * 64u);
      frag<half> xv;
      frag_get_in(xv, (const device half*)x, int(a.k), int(rb + sgr * 16), int(kb), home, int(rb + rows_here), int(kb) + 16);
      frag<int8_t> aq;
      const float xs = tf_frag_quantise(aq, xv);
      frag<half> wo;
      frag<int8_t> b0, b1;
      frag<int> acc;
      tf_frag_get_tg4(wo, wx, int(64u * CG), 0, wc, home);
      const float s0 = tf_frag_quantise(b0, wo) * xs;
      tf_frag_get_tg4(wo, wx, int(64u * CG), 0, wc + 16, home);
      const float s1 = tf_frag_quantise(b1, wo) * xs;
      acc = frag<int>(0);
      mma_16x32<false, false>(acc, acc, aq, b0, b1);
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        lo0[e] += float(acc[e]) * s0;
        hi0[e] += float(acc[e]) * s1;
      }
      tf_frag_get_tg4(wo, wx, int(64u * CG), 0, wc + 32, home);
      const float s2 = tf_frag_quantise(b0, wo) * xs;
      tf_frag_get_tg4(wo, wx, int(64u * CG), 0, wc + 48, home);
      const float s3 = tf_frag_quantise(b1, wo) * xs;
      acc = frag<int>(0);
      mma_16x32<false, false>(acc, acc, aq, b0, b1);
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        lo1[e] += float(acc[e]) * s2;
        hi1[e] += float(acc[e]) * s3;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const int r0 = int(rb + sgr * 16);
  const int c0 = int(nb + sgc * 64u);
  // `sums` is the whole [rows, n] buffer and frag_put_in indexes it absolutely (rr, cc), so the
  // bounds must be absolute too: the last live row is rb + rows_here and the last live column is
  // nb + cols_here. The relative counts (rows_here - sg*16, cols_here) left every row past 32 and
  // every column past 16 unwritten, which is what made this kernel's answers wrong.
  frag_put_in(lo0, sums, int(a.n), r0, c0, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi0, sums, int(a.n), r0, c0 + 16, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo1, sums, int(a.n), r0, c0 + 32, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi1, sums, int(a.n), r0, c0 + 48, home, int(rb + rows_here), int(nb + cols_here));
}

kernel void tf_exl3_matmul_prefill_nax(const device half* x [[buffer(0)]],
                                      const device uchar* trellis [[buffer(1)]],
                                      device float* sums [[buffer(2)]],
                                      constant TfExl3Args& a [[buffer(3)]],
                                      const device uint* routes [[buffer(4)]],
                                      const device TfExl3Expert* descriptors [[buffer(5)]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint sg [[simdgroup_index_in_threadgroup]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint3 tptg [[threads_per_threadgroup]],
                                      uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[16 * 64];
  tf_exl3_prefill_body<false, 4, 1>(wx, x, trellis, sums, a, routes, descriptors, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax2(const device half* x [[buffer(0)]],
                                       const device uchar* trellis [[buffer(1)]],
                                       device float* sums [[buffer(2)]],
                                       constant TfExl3Args& a [[buffer(3)]],
                                       const device uint* routes [[buffer(4)]],
                                       const device TfExl3Expert* descriptors [[buffer(5)]],
                                       uint lane [[thread_index_in_simdgroup]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint3 tptg [[threads_per_threadgroup]],
                                       uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[16 * 64];
  tf_exl3_prefill_body<true, 8, 1>(wx, x, trellis, sums, a, routes, descriptors, lane, sg, tid, tptg.x, tg);
}

// One row an expert, four rows a threadgroup: each SIMDgroup decodes its own expert's 16x32 tile once and
// multiplies the single row it owns, writing only that row of the sixteen-row tensor-unit product.
kernel void tf_exl3_matmul_decode_nax(const device half* x [[buffer(0)]],
                                     const device uchar* trellis [[buffer(1)]],
                                     device float* sums [[buffer(2)]],
                                     constant TfExl3Args& a [[buffer(3)]],
                                     const device uint* routes [[buffer(4)]],
                                     const device TfExl3Expert* descriptors [[buffer(5)]],
                                     device float* partials [[buffer(6)]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint3 tptg [[threads_per_threadgroup]],
                                     uint3 tg [[threadgroup_position_in_grid]]) {
  // Two weight tiles a SIMDgroup: the next block decodes while this one multiplies, and one barrier a block
  // replaces two because the tile a multiply reads is never the tile a decode writes.
#if defined(TF_DECODE_COMPACT)
  // Compact: one row a threadgroup, two 16x32 tiles only (1 KB), so many more groups fit the threadgroup
  // memory and the decode's issue rate is not occupancy-bound (round 173's probe: 0.84 against 0.50e12).
  threadgroup half wx[2 * 16 * 32];
#else
  threadgroup half wx[2 * 4 * 16 * 32];
#endif
  const uint nb = tg.x * 32;
  // One SIMDgroup a row: a threadgroup of four warps takes four rows, one of a single warp takes one, which is four
  // times the threadgroups for the same work (TF_DSV41_EXL3_MOE_ROWS).
#if defined(TF_DECODE_COMPACT)
  const uint rpt = 1;
  const uint row = tg.y;
  threadgroup half* wxa = wx;
  threadgroup half* wxb = wx + 16 * 32;
#else
  const uint rpt = tptg.x / 32u;
  const uint row = tg.y * rpt + sg;
  threadgroup half* wxa = wx + sg * (16 * 32);
  threadgroup half* wxb = wx + (4 + sg) * (16 * 32);
#endif
  if (row >= a.rows) return;
  const uint expert = routes[row];
  if (expert >= a.experts) return;
  const device uchar* data = descriptors[expert].trellis;
  const ulong ks = descriptors[expert].tile_k_stride;
  const ulong ns = descriptors[expert].tile_n_stride;
  const uint halves = descriptors[expert].halves;
  const uint codebook = descriptors[expert].codebook;
  const short2 home = frag_home(ushort(lane));
  const uint cols_here = min(32u, a.n - nb);
  // A decode tile has a handful of rows and one threadgroup a layer for the whole k range, so a split takes a
  // slice of the k blocks and the reduce kernel sums the parts in split order.
  const uint split = tg.z;
  const uint blocks = a.k / 16;
  const uint b0 = a.splits > 1 ? blocks * split / a.splits : 0;
  const uint b1 = a.splits > 1 ? blocks * (split + 1) / a.splits : blocks;
  // Eight consecutive states a thread of each sixteen-column tile, so the three stream words they can touch are
  // read once and the cursor and the tile slot come from registers (the prompt kernel's inner loop).
  uint at[8], off[8], sh[8], rel[8];
  const uint words = 4 * halves;
  const uint wi0 = tf_exl3_map_for(lane * 8u, halves) & 63u;
  {
    const uint ring = 128 * halves;
    const uint cur = (wi0 << 5) | ((tf_exl3_map_for(lane * 8u, halves) >> 6) & 31u);
    TF_UNROLL
    for (uint k = 0; k < 8; ++k) {
      const uint p = lane * 8u + k, l = p / 8, e = p % 8;
      off[k] = (cur + k * (halves >> 1)) % ring & 31u;
      const uint c8 = (cur + k * (halves >> 1)) % ring;
      sh[k] = (32u - off[k]) & 31u;
      rel[k] = (c8 >> 5) >= wi0 ? (c8 >> 5) - wi0 : (c8 >> 5) + words - wi0;
      at[k] = (2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u)) * 32u + (l / 4 + 8 * (e >> 2));
    }
  }
  frag<float> lo = frag<float>(0), hi = frag<float>(0);
  // The words a block needs are loaded a block early, so their latency overlaps the multiply instead of stalling.
  uint pre[2][4];
  const bool pipeline = (halves & 1u) == 0 and halves <= 16;
  uint a1 = wi0 + 1; if (a1 >= words) a1 -= words;
  uint a2 = wi0 + 2; if (a2 >= words) a2 -= words;
  uint a3 = wi0 + 3; if (a3 >= words) a3 -= words;
  if (pipeline) {
    const device ushort* t0 = reinterpret_cast<const device ushort*>(data + ulong(b0) * ks + ulong(nb / 16) * ns);
    pre[0][0] = tf_exl3_word(t0, wi0);
    pre[0][1] = tf_exl3_word(t0, a1);
    pre[0][2] = tf_exl3_word(t0, a2);
    pre[0][3] = tf_exl3_word(t0, a3);
    if (nb + 16 < a.n) {
      const device ushort* t1 = reinterpret_cast<const device ushort*>(data + ulong(b0) * ks + ulong(nb / 16 + 1) * ns);
      pre[1][0] = tf_exl3_word(t1, wi0);
      pre[1][1] = tf_exl3_word(t1, a1);
      pre[1][2] = tf_exl3_word(t1, a2);
      pre[1][3] = tf_exl3_word(t1, a3);
    }
  }
  for (uint bb = b0; bb < b1; ++bb) {
    const uint kb = bb * 16;
    threadgroup half* wxc = ((bb - b0) & 1u) ? wxb : wxa;
    if (pipeline) {
      TF_UNROLL
      for (uint h = 0; h < 2; ++h) {
        if (nb + h * 16 >= a.n) continue;
        const uint w0 = pre[h][0], w1 = pre[h][1], w2 = pre[h][2], w3 = pre[h][3];
        const uint nxt = bb + 1;
        if (nxt < b1) {
          const device ushort* tn = reinterpret_cast<const device ushort*>(data + ulong(nxt) * ks + ulong(nb / 16 + h) * ns);
          pre[h][0] = tf_exl3_word(tn, wi0);
          pre[h][1] = tf_exl3_word(tn, a1);
          pre[h][2] = tf_exl3_word(tn, a2);
          pre[h][3] = tf_exl3_word(tn, a3);
        }
        TF_UNROLL
        for (uint k = 0; k < 8; ++k) {
          const uint lo8 = rel[k] == 0 ? w0 : rel[k] == 1 ? w1 : w2;
          const uint hi8 = rel[k] == 0 ? w1 : rel[k] == 1 ? w2 : w3;
          const uint carry = (hi8 >> sh[k]) & (0u - uint(off[k] != 0));
          wxc[at[k] + h * 16] = tf_exl3_value(ushort(((lo8 << off[k]) | carry) >> 16), codebook);
        }
      }
    } else {
      TF_UNROLL
      for (uint t = 0; t < 16; ++t) {
        const uint idx = lane + 32 * t, h = idx / 256, p = idx % 256;
        if (nb + h * 16 >= a.n) continue;
        const device ushort* tile = reinterpret_cast<const device ushort*>(
            data + ulong(kb / 16) * ks + ulong(nb / 16 + h) * ns);
        const uint l = p / 8, e = p % 8;
        const uint r = 2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u);
        const uint c = l / 4 + 8 * (e >> 2);
        wxc[r * 32 + h * 16 + c] = tf_exl3_value(tf_exl3_mapped_state(tile, tf_exl3_map_for(p, halves)), codebook);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // A tile's first row is the token; the bound makes every other row read zero, so only row 0 is live.
    frag<half> av, b0f, b1f;
    frag_get_in(av, (const device half*)x, int(a.k), int(row), int(kb), home, int(row) + 1, int(kb) + 16);
    tf_frag_get_tg4(b0f, wxc, 32, 0, 0, home);
    tf_frag_get_tg4(b1f, wxc, 32, 0, 16, home);
    mma_16x32<false, false>(lo, hi, av, b0f, b1f);
  }
  if (a.splits > 1) {
    device float* part = partials + (ulong(split) * a.rows + ulong(row)) * a.n;
    frag_put_in(lo, part, int(a.n), 0, int(nb), home, 1, int(nb) + int(cols_here));
    frag_put_in(hi, part, int(a.n), 0, int(nb + 16), home, 1, int(nb) + int(cols_here));
    return;
  }
  frag_put_in(lo, sums, int(a.n), int(row), int(nb), home, int(row) + 1, int(nb) + int(cols_here));
  frag_put_in(hi, sums, int(a.n), int(row), int(nb + 16), home, int(row) + 1, int(nb) + int(cols_here));
}

// One row, four column tiles a threadgroup: a dense projection at decode width on the tensor units. Each
// SIMDgroup decodes its own 32-column tile once and multiplies the single row, writing only row 0.
kernel void tf_exl3_matvec_nax(const device half* x [[buffer(0)]],
                              const device uchar* trellis [[buffer(1)]],
                              device float* sums [[buffer(2)]],
                              constant TfExl3Args& a [[buffer(3)]],
                              const device uint* routes [[buffer(4)]],
                              const device TfExl3Expert* descriptors [[buffer(5)]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[4 * 16 * 32];
  threadgroup half xx[16 * 16];
  const uint nb = tg.x * 128 + sg * 32;
  if (nb >= a.n || a.rows == 0) return;
  const device uchar* base = trellis;
  const uint cols_here = min(32u, a.n - nb);
  const short2 home = frag_home(ushort(lane));
  threadgroup half* wxg = wx + sg * (16 * 32);
  frag<float> lo = frag<float>(0), hi = frag<float>(0);
  for (uint kb = 0; kb < a.k; kb += 16) {
    TF_UNROLL
    for (uint t = 0; t < 16; ++t) {
      const uint idx = lane + 32 * t, h = idx / 256, p = idx % 256;
      if (nb + h * 16 >= a.n) continue;
      const device ushort* tile = reinterpret_cast<const device ushort*>(
          base + ulong(kb / 16) * a.tile_k_stride + ulong(nb / 16 + h) * a.tile_n_stride);
      const uint l = p / 8, e = p % 8;
      const uint r = 2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u);
      const uint c = l / 4 + 8 * (e >> 2);
      wxg[r * 32 + h * 16 + c] = tf_exl3_value(tf_exl3_mapped_state(tile, tf_exl3_map_for(p, a.halves)), a.codebook);
    }
    TF_UNROLL
    for (uint t = 0; t < 8; ++t) {
      const uint idx = lane + 32 * t, r = idx / 16, c = idx % 16;
      xx[idx] = r == 0 ? *(const device half*)(x + ulong(kb) + c) : half(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    frag<half> av, b0, b1;
    tf_frag_get_tg4(av, xx, 16, 0, 0, home);
    tf_frag_get_tg4(b0, wxg, 32, 0, 0, home);
    tf_frag_get_tg4(b1, wxg, 32, 0, 16, home);
    mma_16x32<false, false>(lo, hi, av, b0, b1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  frag_put_in(lo, sums, int(a.n), 0, int(nb), home, 1, int(nb + cols_here));
  frag_put_in(hi, sums, int(a.n), 0, int(nb + 16), home, 1, int(nb + cols_here));
}

kernel void tf_exl3_matmul_prefill_nax3(const device half* x [[buffer(0)]],
                                        const device uchar* trellis [[buffer(1)]],
                                        device float* sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device uint* routes [[buffer(4)]],
                                        const device TfExl3Expert* descriptors [[buffer(5)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tptg [[threads_per_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[16 * 128];
  tf_exl3_prefill_body<true, 4, 2>(wx, x, trellis, sums, a, routes, descriptors, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax4(const device half* x [[buffer(0)]],
                                        const device uchar* trellis [[buffer(1)]],
                                        device float* sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device uint* routes [[buffer(4)]],
                                        const device TfExl3Expert* descriptors [[buffer(5)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tptg [[threads_per_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[16 * 128];
  tf_exl3_prefill_body<true, 8, 2>(wx, x, trellis, sums, a, routes, descriptors, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax5(const device half* x [[buffer(0)]],
                                        const device uchar* trellis [[buffer(1)]],
                                        device float* sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device uint* routes [[buffer(4)]],
                                        const device TfExl3Expert* descriptors [[buffer(5)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tptg [[threads_per_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[16 * 64];
  tf_exl3_prefill_body<true, 4, 1>(wx, x, trellis, sums, a, routes, descriptors, lane, sg, tid, tptg.x, tg);
}

// Sixteen row groups by one column group, every warp decoding: a long chunk's expert tile.
kernel void tf_exl3_matmul_prefill_nax6(const device half* const x [[buffer(0)]],
                                       const device uchar* const trellis [[buffer(1)]],
                                       device float* const sums [[buffer(2)]],
                                       constant TfExl3Args& a [[buffer(3)]],
                                       const device uint* const routes [[buffer(4)]],
                                       const device TfExl3Expert* const descriptors [[buffer(5)]],
                                       uint lane [[thread_index_in_simdgroup]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint3 tptg [[threads_per_threadgroup]],
                                       uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[16 * 64];
  tf_exl3_prefill_body<true, 16, 1>(wx, x, trellis, sums, a, routes, descriptors, lane, sg, tid, tptg.x, tg);
}

// The int8 routed body: the trellis skeleton's decode, the tile and x block quantised per 16x16 fragment and
// multiplied as int8 x int8 on the tensor units (TF_DSV41_INT8_BODY).
kernel void tf_exl3_matmul_prefill_nax_i8(const device half* x [[buffer(0)]],
                                         const device uchar* trellis [[buffer(1)]],
                                         device float* sums [[buffer(2)]],
                                         constant TfExl3Args& a [[buffer(3)]],
                                         const device uint* routes [[buffer(4)]],
                                         const device TfExl3Expert* descriptors [[buffer(5)]],
                                         uint lane [[thread_index_in_simdgroup]],
                                         uint sg [[simdgroup_index_in_threadgroup]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint3 tptg [[threads_per_threadgroup]],
                                         uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[16 * 64];
  tf_exl3_prefill_body_i8<false, 4, 1>(wx, x, trellis, sums, a, routes, descriptors, lane, sg, tid, tptg.x, tg);
}
