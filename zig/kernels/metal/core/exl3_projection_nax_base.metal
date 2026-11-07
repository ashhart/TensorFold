// Native EXL3 on the tensor units: the same stored trellis, decoded into a threadgroup tile and reduced by
// cooperative matmul instead of per-lane scalar FMAs. The decode is unchanged; only the reduction moves. The
// model-facing order follows the release's inference/model.py, revision 314113f8; the trellis arithmetic and the
// kernel shapes are the lane's own pack format.
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

// Both operations are exact in FP32 before the single FP16 round; do not split the mul1 FMA.
inline half tf_exl3_value(ushort state, uint codebook) {
  uint x = uint(state) * (codebook == 2 ? 0x83DCD12Du : codebook == 1 ? 0xCBAC1FEDu : 89226354u);
  if (codebook == 2) {
    // The four byte sums collapse into one 16-bit lane and the +1024 bias is added as half 0x6400, so the state maps
    // to the half h in five ALU ops instead of nine. The fma is the same one the scalar reference does.
    const uint bs = (x & 0x00FF00FFu) + ((x >> 8u) & 0x00FF00FFu);
    const half hs = as_type<half>(ushort((bs * 0x00010001u + 0x64000000u) >> 16u));
    return half(fma(float(hs), float(as_type<half>(ushort(0x1EEE))), float(as_type<half>(ushort(0xC931)))));
  }
  if (codebook == 0) x += 64248484u;
  x = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
  return half(float(as_type<half>(ushort(x))) + float(as_type<half>(ushort(x >> 16))));
}

// The mul1 value from a 1024-entry threadgroup LUT indexed by the byte sum: four ALU ops and a shared load, no 384 KB
// table and no per-weight L2 miss.
inline half tf_exl3_value_lut(const threadgroup half* lut, ushort state) {
  const uint x = uint(state) * 0x83DCD12Du;
  const uint bs = (x & 0x00FF00FFu) + ((x >> 8u) & 0x00FF00FFu);
  return lut[(bs * 0x00010001u) >> 16u];
}

// The lane's eight windows at compile-time shifts. C_cur is the 32-bit window at the lane's first state (its bit 31
// is that window's first stream bit) and C_next the window 32 bits later; S is the states' stride in bits (halves /
// 2) and J the state's index, so every shift below folds at compile time. The structure: one funnel shift a block,
// then the windows at fixed shifts.
template <uint S, uint J>
inline ushort tf_exl3_state_static(uint c_cur, uint c_next) {
  const uint bit = S * J;
  if (bit <= 16u) return ushort((c_cur >> (16u - bit)) & 0xFFFFu);
  if (bit <= 31u) return ushort(((c_cur << (bit - 16u)) | (c_next >> (48u - bit))) & 0xFFFFu);
  return ushort((c_next >> (48u - bit)) & 0xFFFFu);
}

// The value either way: the resident table when one is bound, the arithmetic otherwise.
inline half tf_exl3_pick(const device half* table, const threadgroup half* lut, ushort state, uint codebook) {
  if (lut != nullptr and codebook == 2) return tf_exl3_value_lut(lut, state);
  return table != nullptr ? tf_exl3_table_value(table, state, codebook) : tf_exl3_value(state, codebook);
}

// The device-side twin of frag_get_tg4: two 8-byte loads for a full 16-row tile, the scalar bounds path only when the
// lane's lower row (home.y + 8) falls past the last live row. The values loaded are the same elements frag_get_in
// reads, so the arithmetic is unchanged; this only replaces eight scalar loads with two vector ones.
inline void tf_frag_get_dev4(thread frag<half>& f, const device half* p, int ld, int r, int c, short2 home, int nr) {
  const int r0 = r + home.y, r1 = r0 + 8, c0 = c + home.x;
  if (r1 < nr) {
    f = frag<half>(*(const device half4*)(p + r0 * ld + c0), *(const device half4*)(p + r1 * ld + c0));
  } else {
    TF_UNROLL
    for (short e = 0; e < 8; ++e) {
      const int rr = r0 + (e >> 2) * 8;
      f[e] = rr < nr ? p[rr * ld + c0 + (e & 3)] : half(0);
    }
  }
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
  // End + ring - 16 is below twice the ring, so one conditional subtract replaces the modulo by a runtime divisor.
  uint start = end + ring - 16;
  if (start >= ring) start -= ring;
  const uint wi = start >> 5, off = start & 31u, next = wi + 1 == 4 * halves ? 0 : wi + 1;
  return wi | (off << 6) | (next << 11);
}

// A routed prompt tile's decode job. The release groups a prompt's rows into expert blocks (inference/model.py,
// revision 314113f8); a tile is one block's row window and one column window, and its weights are the resident pack
// slice the route names. The trellis strides, width and codebook are our EXL3 pack's own layout.
struct TfExl3DigJob {
  const device uchar* trellis;
  ulong tile_k_stride, tile_n_stride;
  uint halves, codebook;
  uint row_block, col_block, tile_rows;
  uint rows_here, cols_here;
};

template <uint CG>
inline TfExl3DigJob tf_exl3_dig_job(const device uchar* trellis, constant TfExl3Args& a,
                                    const device uint* routes, const device TfExl3Expert* descriptors,
                                    uint row_block, uint col_block, uint tile_rows) {
  const uint expert = a.indexed ? routes[row_block] : a.expert_id;
  const uint didx = a.indexed ? expert : 0;
  TfExl3DigJob job;
  job.trellis = a.indexed ? descriptors[didx].trellis : trellis;
  job.tile_k_stride = a.indexed ? descriptors[didx].tile_k_stride : a.tile_k_stride;
  job.tile_n_stride = a.indexed ? descriptors[didx].tile_n_stride : a.tile_n_stride;
  job.halves = a.indexed ? descriptors[didx].halves : a.halves;
  job.codebook = a.indexed ? descriptors[didx].codebook : a.codebook;
  job.row_block = row_block;
  job.col_block = col_block;
  job.tile_rows = tile_rows;
  job.rows_here = min(tile_rows, a.rows - row_block);
  job.cols_here = min(64u * CG, a.n - col_block);
  return job;
}

// The lane's window walk over one pack tile: the stream is a ring of 128*halves sixteen-bit states, and a lane owns
// `States` consecutive states from p0. State k starts halves/2 bits after state k-1 and wraps at the ring. `at[k]` is
// the threadgroup slot that state's weight lands in; `Cols` is the tile's column stride and `hg`/`h` place a column
// group. `off`, `sh` and `rel` are the general read's cursor: the bit offset in the state's first word, the shift that
// brings the second word down, and the word's distance from the lane's first word. `next1..3` are the ring words after
// the first. The arithmetic is fixed by the pack format; the walk is our own cursor.
struct TfExl3DigWindows {
  uint at[8], off[8], sh[8], rel[8];
  uint first_word, first_bit, words, next1, next2, next3;
};

template <uint States, uint CG, uint Cols>
inline TfExl3DigWindows tf_exl3_dig_windows(uint p0, uint halves, uint hg, uint h, bool cursor) {
  TfExl3DigWindows walk;
  walk.words = 4u * halves;
  const uint map = tf_exl3_map_for(p0, halves);
  walk.first_word = map & 63u;
  walk.first_bit = (map >> 6) & 31u;
  walk.next1 = walk.first_word + 1u;
  if (walk.next1 >= walk.words) walk.next1 -= walk.words;
  walk.next2 = walk.first_word + 2u;
  if (walk.next2 >= walk.words) walk.next2 -= walk.words;
  walk.next3 = walk.first_word + 3u;
  if (walk.next3 >= walk.words) walk.next3 -= walk.words;
  const uint ring = 128u * halves;
  const uint origin = (walk.first_word << 5) | walk.first_bit;
  TF_UNROLL
  for (uint k = 0; k < States; ++k) {
    const uint p = p0 + k, l = p / 8, e = p % 8;
    walk.at[k] = (2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u)) * Cols + hg + h * 16u + (l / 4 + 8 * (e >> 2));
    if (cursor) {
      uint c8 = origin + k * (halves >> 1);
      if (c8 >= ring) c8 -= ring;
      const uint wi = c8 >> 5;
      walk.off[k] = c8 & 31u;
      walk.sh[k] = (32u - walk.off[k]) & 31u;
      walk.rel[k] = wi >= walk.first_word ? wi - walk.first_word : wi + walk.words - walk.first_word;
    }
  }
  return walk;
}

// One routed or dense row's decode job: the pack slice a route names (or the dense call's own trellis), its k/n
// strides, width and codebook, and the row's output width. The release's MTP and draft rounds feed the same EXL3
// projection shapes (inference/model.py, revision 314113f8); the pack layout and the row walk are our own.
struct TfExl3DecodeJob {
  const device uchar* trellis;
  ulong tile_k_stride, tile_n_stride;
  uint halves, codebook, width;
};

inline TfExl3DecodeJob tf_exl3_dense_decode_job(const device uchar* trellis, constant TfExl3Args& a) {
  TfExl3DecodeJob job;
  job.trellis = trellis;
  job.tile_k_stride = a.tile_k_stride;
  job.tile_n_stride = a.tile_n_stride;
  job.halves = a.halves;
  job.codebook = a.codebook;
  job.width = a.n;
  return job;
}

inline TfExl3DecodeJob tf_exl3_routed_decode_job(const device uchar* trellis, constant TfExl3Args& a,
                                                const device TfExl3Expert* descriptors, uint expert) {
  (void)trellis;
  TfExl3DecodeJob job;
  job.trellis = descriptors[expert].trellis;
  job.tile_k_stride = descriptors[expert].tile_k_stride;
  job.tile_n_stride = descriptors[expert].tile_n_stride;
  job.halves = descriptors[expert].halves;
  job.codebook = descriptors[expert].codebook;
  job.width = descriptors[expert].reserved != 0 ? descriptors[expert].reserved : a.n;
  return job;
}

// The lane's row walk through one pack tile: its first word, the bit offset in it and the shift/mask that bring the
// next word down, and the next two words. One runtime funnel shift a block puts the lane's window at a known place;
// the eight states then sit at compile-time shifts. The pack's ring fixes this arithmetic.
struct TfExl3RowWalk {
  uint words, first_word, first_bit, shift, mask, next1, next2;
};

inline TfExl3RowWalk tf_exl3_row_walk(uint lane, uint halves) {
  TfExl3RowWalk walk;
  walk.words = 4u * halves;
  const uint map = tf_exl3_map_for(lane * 8u, halves);
  walk.first_word = map & 63u;
  walk.first_bit = (map >> 6) & 31u;
  walk.shift = (32u - walk.first_bit) & 31u;
  walk.mask = 0u - uint(walk.first_bit != 0u);
  walk.next1 = walk.first_word + 1u;
  if (walk.next1 >= walk.words) walk.next1 -= walk.words;
  walk.next2 = walk.first_word + 2u;
  if (walk.next2 >= walk.words) walk.next2 -= walk.words;
  return walk;
}

// One state block's four stream words from the next 16-row tile, with the ring's wrap folded in once.
inline uint tf_exl3_block_word(const device uchar* data, ulong ks, ulong ns, uint kb, uint nb, uint hg, uint h,
                               uint wi, uint words) {
  const device ushort* tile = reinterpret_cast<const device ushort*>(
      data + ulong(kb / 16) * ks + ulong(nb / 16 + hg / 16u + h) * ns);
  return tf_exl3_word(tile, wi < words ? wi : wi - words);
}

// The eight (or four, or two) states a lane owns, from four words already in registers. This is the Fast path's inner
// decode lifted out so the double-buffered body can run it against the buffer the multiply is not reading.
template <uint States>
inline void tf_exl3_decode_fast(threadgroup half* wx, uint codebook, const uint* at, const uint* off,
                                const uint* sh, const uint* rel, uint w0, uint w1, uint w2, uint w3) {
  TF_UNROLL
  for (uint k = 0; k < States; ++k) {
    const uint lo = rel[k] == 0 ? w0 : rel[k] == 1 ? w1 : w2;
    const uint hi = rel[k] == 0 ? w1 : rel[k] == 1 ? w2 : w3;
    const uint carry = (hi >> sh[k]) & (0u - uint(off[k] != 0));
    wx[at[k]] = tf_exl3_value(ushort(((lo << off[k]) | carry) >> 16), codebook);
  }
}

// The same decode through the resident codebook table: one load replaces the mul1's four byte sums and fma.
template <uint States>
inline void tf_exl3_decode_table(threadgroup half* wx, const device half* table, uint codebook,
                                 const uint* at, const uint* off, const uint* sh, const uint* rel,
                                 uint w0, uint w1, uint w2, uint w3) {
  TF_UNROLL
  for (uint k = 0; k < States; ++k) {
    const uint lo = rel[k] == 0 ? w0 : rel[k] == 1 ? w1 : w2;
    const uint hi = rel[k] == 0 ? w1 : rel[k] == 1 ? w2 : w3;
    const uint carry = (hi >> sh[k]) & (0u - uint(off[k] != 0));
    wx[at[k]] = tf_exl3_table_value(table, ushort(((lo << off[k]) | carry) >> 16), codebook);
  }
}

// The funnel read in the prompt body, on our map and our ring.
// A lane owns eight states; the states' sixteen-bit windows are `stride` bits apart (three at the routed pack's
// halves = 6), so the run of eight windows spans stride*7 + 16 bits -- thirty-seven at halves = 6 -- and two stream
// words hold it. The run is assembled once a block into the register R (R's top bit is the run's first bit, R's bit 0
// its last), and the eight windows then sit at compile-time shifts: one shift and one mask a state, where the
// cursor's select chain was. The values, their order and the k order are the shipped body's, so the output is
// bit-identical; only the word traffic and the select chain leave.
// The ring's order is MSB-first inside a word, so the run's bits are assembled the other way round: R's bit (run - 1)
// is the run's first bit, and the two words' pieces meet without overlapping.
template <uint States, uint Stride = 3>
inline void tf_exl3_decode_funnel(threadgroup half* wx, const device half* table, const threadgroup half* lut,
                                  uint codebook, const uint* at, uint b0, uint stride, uint w0, uint w1) {
  // The windows are Stride bits apart, so the run is 16 + Stride * (States - 1) bits. Halves = 6 is Stride 3 (25 bits
  // for a four-state lane, 37 for eight); halves = 8 is Stride 4 (28 bits for four). The stride is compile time now,
  // so the shifts below are literals; the runtime argument is kept for the callers and ignored.
  (void)stride;
  constexpr uint run = 16u + Stride * (States - 1u);
  // The run's first bit is w0's bit (31 - b0) and it has to land on r's bit (run - 1). A run of thirty-seven bits
  // (eight windows) always straddles two words; a run of twenty-five (four windows, the 128-row tile's lane) sits
  // inside one word when its start leaves room for it (b0 <= 32 - run) and straddles otherwise. The window offsets
  // are the same (stride a window apart), so the states are the shipped body's -- only the assembly of the run
  // changes.
  ulong r;
  if (run >= 32u) {
    r = (ulong(w0) << (run - 32u + b0)) | (ulong(w1) >> (64u - run - b0));
  } else if (b0 <= 32u - run) {
    r = ulong(w0) >> (32u - run - b0);
  } else {
    r = (ulong(w0) << (b0 + run - 32u)) | (ulong(w1) >> (64u - run - b0));
  }
  TF_UNROLL
  for (uint k = 0; k < States; ++k) {
    const ushort state = ushort((r >> (Stride * (States - 1u - k))) & 0xffff);
    wx[at[k]] = tf_exl3_pick(table, lut, state, codebook);
  }
}

// A threadgroup owns RG sixteen-row groups by CG sixty-four-column groups, one SIMDgroup a (row group, column group).
// Dense rows share one matrix; grouped rows are sixty-four-row aligned, so one expert owns the tile and its weights
// decode once. A wider column tile re-reads the activation tile for fewer output tiles. `Fast` keeps the per-state
// cursor out of the k loop and reads the three stream words an eight-state block can touch.
template <bool Fast, uint RG, uint CG, bool Table, bool Funnel = false, bool SkipPad = false, uint Rpg = 1, bool F6 = false>
