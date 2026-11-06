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

// The same stream word as one 4-byte load. The shipped read is two 2-byte loads (a word is a little-endian pair of
// int16s), so a lane issues four memory instructions a block where the wide form reads two; the bytes, the values and
// the order are the same. `W` is a template parameter, so only one path is compiled. The caller checks that the
// tile's byte address is 4-byte aligned.
template <bool W>
inline uint tf_exl3_word_t(const device ushort* tile, uint i) {
  if (W) return *reinterpret_cast<const device uint*>(reinterpret_cast<const device uchar*>(tile) + 4u * i);
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

// The funnel with the value read cheaply -- the window extraction and the store stay, the codebook arithmetic goes. A
// diagnostic: it isolates the value path's share of the tile (the value it stores is not the weight's, so its output
// differs by construction). The stored value is a small integer, so no denormal reaches the tensor units and the
// timing is the extraction's alone.
inline void tf_exl3_decode_funnel_cheap(threadgroup half* wx, const uint* at, uint b0, uint stride, uint w0, uint w1) {
  const ulong r = (ulong(w0) << (b0 + 5u)) | (ulong(w1) >> (27u - b0));
  TF_UNROLL
  for (uint k = 0; k < 8u; ++k) {
    const ushort state = ushort((r >> (stride * (7u - k))) & 0xffffu);
    wx[at[k]] = half(float(state & 0xFFu));
  }
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
template <uint States, bool NoValue = false, uint Stride = 3>
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
    wx[at[k]] = NoValue ? half(state) : tf_exl3_pick(table, lut, state, codebook);
  }
}

// The shipped body writes each lane's eight decoded weights as eight scattered two-byte threadgroup stores; a wide
// form writes a lane's values as contiguous stores. These modes isolate the difference. Mode 1: the decode stays, the
// eight stores go -- the value is folded into a register accumulator so the compiler cannot eliminate the decode. A
// diagnostic: the tile is never written, so the output differs by construction. Mode 2: the eight stores stay, the
// EXTRACTION goes -- the stored value is a cheap function of the k block, so the stores cannot be hoisted out of the
// loop. A diagnostic. Mode 3: the decode stays, the eight stores become CONTIGUOUS -- one lane's run of eight halves,
// the wide form. The tile's contents change, so the output differs by construction. Mode 4: the shipped store PLUS
// the Mode 1 accumulator -- the calibration for the accumulator's cost.
template <uint States, uint Mode>
inline uint tf_exl3_decode_funnel_store(threadgroup half* wx, const device half* table, const threadgroup half* lut,
                                        uint codebook, const uint* at, uint b0, uint stride, uint w0, uint w1,
                                        uint acc, uint salt, uint slot) {
  constexpr uint run = 16u + 3u * (States - 1u);
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
    const ushort state = ushort((r >> (stride * (States - 1u - k))) & 0xffff);
    const half v = tf_exl3_pick(table, lut, state, codebook);
    if (Mode == 1) {
      acc ^= uint(as_type<ushort>(v));
    } else if (Mode == 2) {
      wx[at[k]] = half(float(salt + k));
    } else if (Mode == 3) {
      wx[slot * States + k] = v;
    } else {
      wx[at[k]] = v;
      if (Mode == 4) acc ^= uint(as_type<ushort>(v));
    }
  }
  return acc;
}

// The funnel's extraction in thirty-two-bit words, the windows at compile-time shifts -- the structure: it reads its
// stream words into 32-bit registers and shifts them by literal amounts, because Apple GPUs emulate a 64-bit shift
// with several 32-bit ops, so the shipped funnel paid that emulation once a window). The lane's first window starts
// at bit b0 of the ring's word (the ring is MSB-first inside a word): G holds the thirty-two ring bits from there and
// h5 the five above them, so the eight windows sit at G's compile-time shifts (16 - 3k) and the last two take their
// high bits from h5. The same windows, the same values, the same order -- the output is bit-identical to the shipped
// funnel.
template <uint States, bool NoValue = false>
inline void tf_exl3_decode_funnel32(threadgroup half* wx, const device half* table, const threadgroup half* lut,
                                     uint codebook, const uint* at, uint b0, uint w0, uint w1) {
  const uint G = (w0 << b0) | ((w1 >> 1u) >> (31u - b0));
  const uint h5 = (w1 >> (27u - b0)) & 0x1Fu;
  TF_UNROLL
  for (uint k = 0; k < States; ++k) {
    uint state;
    if (States == 8u and k == 6u) {
      state = ((G & 0x3FFFu) << 2u) | ((h5 >> 3u) & 3u);
    } else if (States == 8u and k == 7u) {
      state = ((G & 0x7FFu) << 5u) | (h5 & 0x1Fu);
    } else {
      state = (G >> (16u - 3u * k)) & 0xFFFFu;
    }
    wx[at[k]] = NoValue ? half(state) : tf_exl3_pick(table, lut, ushort(state), codebook);
  }
}

// A threadgroup owns RG sixteen-row groups by CG sixty-four-column groups, one SIMDgroup a (row group, column group).
// Dense rows share one matrix; grouped rows are sixty-four-row aligned, so one expert owns the tile and its weights
// decode once. A wider column tile re-reads the activation tile for fewer output tiles. `Fast` keeps the per-state
// cursor out of the k loop and reads the three stream words an eight-state block can touch.
template <bool Fast, uint RG, uint CG, bool Table, bool Funnel = false, bool NoValue = false, bool Cheap = false, bool SkipPad = false, bool Funnel32 = false, bool Coop = false, uint Rpg = 1, bool W = false, bool F6 = false, uint Act = 0, uint Store = 0>
inline void tf_exl3_prefill_body(threadgroup half* wx, const device half* x, const device uchar* trellis,
                                 device float* sums, constant TfExl3Args& a, const device uint* routes,
                                 const device TfExl3Expert* descriptors, const device half* table,
                                 const threadgroup half* lut,
                                 uint lane, uint sg, uint tid, uint tptg, uint3 tg) {
  const uint trows = RG * 16u;
  // The activation's staging area, carved from the tail of the weight tile (the kernel entries allocate it): the
  // tile's rows by the block's sixteen k columns.
  threadgroup half* ax = wx + 16u * 64u * CG;
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
  // states of each lane between them instead of leaving those warps to wait at the barrier. The warps decoding one
  // sixteen-column block. With Rpg row groups a warp (the wide form's thirty-two-row simdgroup) the block is decoded
  // by one warp, so the lane still owns eight states and still reads eight bits a weight -- the tile's height and the
  // traffic per weight are decoupled.
  constexpr uint wpg = RG / (4u * Rpg);          // warps decoding one column group
  constexpr uint states = 8u / wpg;              // consecutive states a lane owns
  const uint gwarp = tid / 32u, group = gwarp / wpg, wg = gwarp % wpg;
  const uint h = group % 4u, hg = (group / 4u) * 64u;
  const uint p0 = (wg * 32u + tid % 32u) * states;
  const bool decodes = group < 4u * CG and nb + hg + h * 16u < a.n;
  uint at[8], off[8], sh[8], rel[8];
  const uint words = 4 * halves;
  const uint wi0 = tf_exl3_map_for(p0, halves) & 63u;
  const uint b0 = (tf_exl3_map_for(p0, halves) >> 6) & 31u;
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
      // With F6 the funnel path is the only one, and the cursor arrays the Fast path needs are dead there -- but they
      // are live registers across the whole k loop, so they are not computed at all.
      if (!F6) {
        off[k] = c8 & 31u;
        sh[k] = (32u - off[k]) & 31u;
        rel[k] = wi >= wi0 ? wi - wi0 : wi + words - wi0;
      }
    }
  }
  // This SIMDgroup's row groups (Rpg of them) and its column group.
  const uint sgr = Rpg * (sg % (RG / Rpg)), sgc = sg / (RG / Rpg);
  // The routed tile's padded rows need no multiply. Each expert's block is padded to sixty-four rows (the padding
  // rows carry the sentinel expert id), and a warp owns sixteen rows of the tile, so a warp reads its own sixteen
  // route ids and skips the multiply when they are all padding. The decode stays: the other warps read the whole
  // sixty-four-column tile. The real rows' values, their order and the k order are the shipped body's, so the reply
  // is bit-identical. The skip is per sixteen-row group, so a warp that owns two groups (the shape) still multiplies
  // the real one. The mask keeps the per-group answer in a scalar (no local memory).
  uint real_mask = (1u << Rpg) - 1u;
  if (SkipPad and a.indexed != 0) {
    real_mask = 0u;
    if (lane < 16u * Rpg) {
      const uint grow = rb + (sgr + lane / 16u) * 16u + (lane % 16u);
      if (grow < a.rows and routes[grow] != 0xffffffffu) real_mask = 1u << (lane / 16u);
    }
  }
  bool group_live[Rpg];
  bool any_live = false;
  TF_UNROLL
  for (uint r = 0; r < Rpg; ++r) {
    group_live[r] = simd_any((real_mask & (1u << r)) != 0u);
    any_live = any_live or group_live[r];
  }
  frag<float> lo0[Rpg], hi0[Rpg], lo1[Rpg], hi1[Rpg];
  TF_UNROLL
  for (uint r = 0; r < Rpg; ++r) {
    lo0[r] = frag<float>(0);
    hi0[r] = frag<float>(0);
    lo1[r] = frag<float>(0);
    hi1[r] = frag<float>(0);
  }
  // The four stream words a block needs are loaded a block early: issued before the barrier, their latency overlaps
  // the multiply and the next block's decode instead of stalling at the top of every k block.
  uint p0v = 0, p1v = 0, p2v = 0, p3v = 0;
  // The store path's accumulator (the decode's values kept live in a register).
  uint sacc = 0;
  // The cooperative read -- this lane holds the block's word at its own index and the two words it needs arrive by a
  // shuffle, so the warp reads the block's ninety-six bytes once instead of 2.67 times over.
  uint mine = 0, wi1s = 0;
  if (Funnel and Coop and decodes) {
    const device ushort* tile0 = reinterpret_cast<const device ushort*>(data + ulong(nb / 16 + hg / 16u + h) * ns);
    mine = (lane < words) ? tf_exl3_word_t<W>(tile0, lane) : 0u;
    wi1s = (wi0 + 1u < words) ? wi0 + 1u : wi0 + 1u - words;
  }
  if (Fast and decodes) {
    const device ushort* tile0 = reinterpret_cast<const device ushort*>(data + ulong(nb / 16 + hg / 16u + h) * ns);
    uint a1 = wi0 + 1; if (a1 >= words) a1 -= words;
    uint a2 = wi0 + 2; if (a2 >= words) a2 -= words;
    uint a3 = wi0 + 3; if (a3 >= words) a3 -= words;
    p0v = tf_exl3_word_t<W>(tile0, wi0);
    p1v = tf_exl3_word_t<W>(tile0, a1);
    p2v = tf_exl3_word_t<W>(tile0, a2);
    p3v = tf_exl3_word_t<W>(tile0, a3);
  }
  for (uint kb = 0; kb < a.k; kb += 16) {
    // One thread owns 8 consecutive states of one 16x16 tile. The bit cursor advances by `halves` a state, so it
    // rolls forward with one add and one conditional subtract instead of recomputing the map formula.
    if (decodes) {
      {
        const device ushort* tile = reinterpret_cast<const device ushort*>(
            data + ulong(kb / 16) * ks + ulong(nb / 16 + hg / 16u + h) * ns);
        if (Funnel and (F6 or halves == 6) and (states == 8 or states == 4)) {
          // Two words a lane: the funnel holds the lane's eight windows (three bits apart), so the state's window is
          // one shift and one mask from the funnel, with no cursor and no select chain.
          const uint nkb = kb + 16;
          if (Coop) {
            // The words come from the cooperative read -- a shuffle from the lane that holds the word.
            const uint w0 = simd_shuffle(mine, wi0);
            const uint w1 = simd_shuffle(mine, wi1s);
            if (nkb < a.k) {
              const device ushort* tile_n = reinterpret_cast<const device ushort*>(
                  data + ulong(nkb / 16) * ks + ulong(nb / 16 + hg / 16u + h) * ns);
              mine = (lane < words) ? tf_exl3_word_t<W>(tile_n, lane) : 0u;
            }
            if (Cheap) tf_exl3_decode_funnel_cheap(wx, at, b0, halves >> 1, w0, w1);
            else if (Funnel32) tf_exl3_decode_funnel32<states, NoValue>(wx, table, lut, codebook, at, b0, w0, w1);
            else tf_exl3_decode_funnel<states, NoValue>(wx, table, lut, codebook, at, b0, halves >> 1, w0, w1);
          } else {
          const uint w0 = p0v, w1 = p1v;
          if (nkb < a.k) {
            const device ushort* tile_n = reinterpret_cast<const device ushort*>(
                data + ulong(nkb / 16) * ks + ulong(nb / 16 + hg / 16u + h) * ns);
            p0v = tf_exl3_word_t<W>(tile_n, wi0);
            p1v = tf_exl3_word_t<W>(tile_n, wi0 + 1 < words ? wi0 + 1 : wi0 + 1 - words);
          }
          if (Store != 0) {
            sacc = tf_exl3_decode_funnel_store<states, Store>(wx, table, lut, codebook, at, b0, halves >> 1, w0, w1, sacc, kb, tid);
          } else if (Cheap) tf_exl3_decode_funnel_cheap(wx, at, b0, halves >> 1, w0, w1);
          else if (Funnel32) tf_exl3_decode_funnel32<states, NoValue>(wx, table, lut, codebook, at, b0, w0, w1);
          else tf_exl3_decode_funnel<states, NoValue>(wx, table, lut, codebook, at, b0, halves >> 1, w0, w1);
          }
        } else if (Fast and (halves & 1u) == 0 and halves <= 16) {
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
            wx[at[k]] = NoValue ? half(ushort(((lo << off[k]) | carry) >> 16))
                                : tf_exl3_pick(table, lut, ushort(((lo << off[k]) | carry) >> 16), codebook);
          }
        } else if (halves <= 16 and (halves & 1u) == 0 and states == 8) {
          const uint ring = 128 * halves;
          uint at1 = wi0 + 1; if (at1 >= words) at1 -= words;
          uint at2 = wi0 + 2; if (at2 >= words) at2 -= words;
          uint at3 = wi0 + 3; if (at3 >= words) at3 -= words;
          uint at4 = wi0 + 4; if (at4 >= words) at4 -= words;
          const uint w0 = tf_exl3_word_t<W>(tile, wi0);
          const uint w1 = tf_exl3_word_t<W>(tile, at1), w2 = tf_exl3_word_t<W>(tile, at2);
          const uint w3 = tf_exl3_word_t<W>(tile, at3), w4 = tf_exl3_word_t<W>(tile, at4);
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
            wx[r * (64u * CG) + hg + h * 16 + c] = tf_exl3_pick(table, lut, ushort(((lo << off) | carry) >> 16), codebook);
          }
        } else {
          TF_UNROLL
          for (uint k = 0; k < 8; ++k) {
            const uint p = p0 + k;
            const uint l = p / 8, e = p % 8;
            const uint r = 2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u);
            const uint c = l / 4 + 8 * (e >> 2);
            wx[r * (64u * CG) + hg + h * 16 + c] = tf_exl3_pick(table, lut, tf_exl3_mapped_state(tile, tf_exl3_map_for(p, halves)), codebook);
          }
        }
      }
    }
    // This SIMDgroup's rows of the k block, read straight into fragments: no staging buffer, one barrier. Rpg
    // sixteen-row groups a warp -- the weight fragments are read once for all of them. The activation's three forms.
    // Act == 1 reads the fragment ONCE, before the k loop (a diagnostic: the output differs by construction and the
    // delta against Act == 0 IS the per-block read). Act == 2 stages the fragment in threadgroup memory once a k
    // block, so the two column groups of a row group SHARE one device read -- the record's "occupancy-keeping form"
    // (one column tile's activation read shared by two column groups, the fragment in threadgroup memory). The
    // staging is published by the SAME barrier that publishes the weight tile, so the block's barrier count is
    // unchanged.
    frag<half> av[Rpg];
    if (Act == 1) {
      if (any_live) {
        TF_UNROLL
        for (uint r = 0; r < Rpg; ++r) {
          if (group_live[r]) tf_frag_get_dev4(av[r], (const device half*)x, int(a.k), int(rb + (sgr + r) * 16), 0, home, int(rb + rows_here));
        }
      }
    }
    if (Act == 2) {
      // The staging: the column group's first warp reads its row group's activation from the device once and writes
      // it into the tile's staging area (the tile's rows, the block's sixteen k columns). The other column group's
      // warps read it back from there AFTER the barrier that publishes it, so the device read happens once a row
      // group, not once a column group. The values, their order and the k order are the shipped body's.
      if (sgc == 0 and any_live) {
        TF_UNROLL
        for (uint r = 0; r < Rpg; ++r) {
          if (group_live[r]) {
            const int r0 = int(rb + (sgr + r) * 16) + int(home.y);
            const int r1 = r0 + 8;
            const int c0 = int(kb) + int(home.x);
            half4 v0 = half4(0.0h, 0.0h, 0.0h, 0.0h);
            half4 v1 = half4(0.0h, 0.0h, 0.0h, 0.0h);
            if (r0 < int(rb + rows_here)) v0 = *(const device half4*)(x + r0 * int(a.k) + c0);
            if (r1 < int(rb + rows_here)) v1 = *(const device half4*)(x + r1 * int(a.k) + c0);
            *(threadgroup half4*)(ax + int((sgr + r) * 16 + uint(home.y)) * 16 + int(home.x)) = v0;
            *(threadgroup half4*)(ax + int((sgr + r) * 16 + uint(home.y) + 8u) * 16 + int(home.x)) = v1;
          }
        }
      }
    }
    if (Act == 0) {
      if (any_live) {
        TF_UNROLL
        for (uint r = 0; r < Rpg; ++r) {
          if (group_live[r]) tf_frag_get_dev4(av[r], (const device half*)x, int(a.k), int(rb + (sgr + r) * 16), int(kb), home, int(rb + rows_here));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (Act == 2) {
      // The staged fragment, read back after the barrier that published it.
      TF_UNROLL
      for (uint r = 0; r < Rpg; ++r) {
        if (group_live[r]) tf_frag_get_tg4(av[r], ax, 16, int((sgr + r) * 16), 0, home);
      }
    }
    if (any_live) {
      frag<half> b0, b1;
      const int wc = int(sgc * 64u);
      tf_frag_get_tg4(b0, wx, int(64u * CG), 0, wc, home);
      tf_frag_get_tg4(b1, wx, int(64u * CG), 0, wc + 16, home);
      TF_UNROLL
      for (uint r = 0; r < Rpg; ++r) {
        if (group_live[r]) mma_16x32<false, false>(lo0[r], hi0[r], av[r], b0, b1);
      }
      tf_frag_get_tg4(b0, wx, int(64u * CG), 0, wc + 32, home);
      tf_frag_get_tg4(b1, wx, int(64u * CG), 0, wc + 48, home);
      TF_UNROLL
      for (uint r = 0; r < Rpg; ++r) {
        if (group_live[r]) mma_16x32<false, false>(lo1[r], hi1[r], av[r], b0, b1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // The accumulator is folded into the first output element, so the decode's values cannot be eliminated. A
  // diagnostic: the output differs by construction.
  if (Store == 1 or Store == 4) {
    const float t = lo0[0][0];
    lo0[0][0] = t + float(sacc);
  }
  const int r0 = int(rb + sgr * 16);
  const int c0 = int(nb + sgc * 64u);
  // `sums` is the whole [rows, n] buffer and frag_put_in indexes it absolutely (rr, cc), so the bounds must be
  // absolute too: the last live row is rb + rows_here and the last live column is nb + cols_here. The relative counts
  // (rows_here - sg*16, cols_here) left every row past 32 and every column past 16 unwritten, which is what made this
  // kernel's answers wrong.
  TF_UNROLL
  for (uint r = 0; r < Rpg; ++r) {
    if (group_live[r]) {
      frag_put_in(lo0[r], sums, int(a.n), r0 + int(r * 16), c0, home, int(rb + rows_here), int(nb + cols_here));
      frag_put_in(hi0[r], sums, int(a.n), r0 + int(r * 16), c0 + 16, home, int(rb + rows_here), int(nb + cols_here));
      frag_put_in(lo1[r], sums, int(a.n), r0 + int(r * 16), c0 + 32, home, int(rb + rows_here), int(nb + cols_here));
      frag_put_in(hi1[r], sums, int(a.n), r0 + int(r * 16), c0 + 48, home, int(rb + rows_here), int(nb + cols_here));
    }
  }
}


// Two-blocks-ahead prompt body: the stream words for block b+1 are issued two iterations before they are decoded, so
// a DRAM-latency miss is covered by two blocks of multiply and decode instead of one. Two threadgroup tiles and two
// register word sets; the arithmetic (values, k order, MMA order) is the double-buffered body's, unchanged.
template <uint RG, uint CG, uint Mmas>
inline void tf_exl3_prefill_body_pf(threadgroup half* wx, const device half* x, const device uchar* trellis,
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
  if ((halves & 1u) != 0 or halves > 16) {
    tf_exl3_prefill_body<true, RG, CG, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg, tg);
    return;
  }
  const short2 home = frag_home(ushort(lane));
  const uint rows_here = min(trows, a.rows - rb);
  const uint cols_here = min(64u * CG, a.n - nb);
  constexpr uint wpg = RG / 4u;
  constexpr uint states = 8u / wpg;
  const uint gwarp = tid / 32u, group = gwarp / wpg, wg = gwarp % wpg;
  const uint h = group % 4u, hg = (group / 4u) * 64u;
  const uint p0 = (wg * 32u + tid % 32u) * states;
  const bool decodes = group < 4u * CG and nb + hg + h * 16u < a.n;
  uint at[8], off[8], sh[8], rel[8];
  const uint words = 4 * halves;
  const uint wi0 = tf_exl3_map_for(p0, halves) & 63u;
  const uint ring = 128 * halves;
  {
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
  const uint sgr = sg % RG, sgc = sg / RG;
  frag<float> lo0 = frag<float>(0), hi0 = frag<float>(0), lo1 = frag<float>(0), hi1 = frag<float>(0);
  const uint blocks = a.k / 16;
  // Block 0 decodes straight from its words; blocks 1 and 2 are issued into the two register sets behind it.
  if (decodes) {
    const uint w0 = tf_exl3_block_word(data, ks, ns, 0, nb, hg, h, wi0, words);
    const uint w1 = tf_exl3_block_word(data, ks, ns, 0, nb, hg, h, wi0 + 1, words);
    const uint w2 = tf_exl3_block_word(data, ks, ns, 0, nb, hg, h, wi0 + 2, words);
    const uint w3 = tf_exl3_block_word(data, ks, ns, 0, nb, hg, h, wi0 + 3, words);
    tf_exl3_decode_fast<states>(wx, codebook, at, off, sh, rel, w0, w1, w2, w3);
  }
  uint A0 = 0, A1 = 0, A2 = 0, A3 = 0, B0 = 0, B1 = 0, B2 = 0, B3 = 0;
  if (decodes and blocks > 1) {
    A0 = tf_exl3_block_word(data, ks, ns, 16, nb, hg, h, wi0, words);
    A1 = tf_exl3_block_word(data, ks, ns, 16, nb, hg, h, wi0 + 1, words);
    A2 = tf_exl3_block_word(data, ks, ns, 16, nb, hg, h, wi0 + 2, words);
    A3 = tf_exl3_block_word(data, ks, ns, 16, nb, hg, h, wi0 + 3, words);
  }
  if (decodes and blocks > 2) {
    B0 = tf_exl3_block_word(data, ks, ns, 32, nb, hg, h, wi0, words);
    B1 = tf_exl3_block_word(data, ks, ns, 32, nb, hg, h, wi0 + 1, words);
    B2 = tf_exl3_block_word(data, ks, ns, 32, nb, hg, h, wi0 + 2, words);
    B3 = tf_exl3_block_word(data, ks, ns, 32, nb, hg, h, wi0 + 3, words);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint b = 0; b < blocks; ++b) {
    const uint cur_buf = b & 1u;
    // Decode block b+1 from the set issued two iterations ago (before this block's multiply, into the free tile).
    if (b + 1 < blocks and decodes) {
      const bool even = ((b + 1) & 1u) == 0u;
      const uint w0 = even ? A0 : B0, w1 = even ? A1 : B1, w2 = even ? A2 : B2, w3 = even ? A3 : B3;
      tf_exl3_decode_fast<states>(wx + (cur_buf ^ 1u) * (16u * 64u * CG), codebook, at, off, sh, rel, w0, w1, w2, w3);
    }
    frag<half> av;
    tf_frag_get_dev4(av, (const device half*)x, int(a.k), int(rb + sgr * 16), int(b * 16), home, int(rb + rows_here));
    threadgroup half* cur = wx + cur_buf * (16u * 64u * CG);
    frag<half> b0, b1;
    const int wc = int(sgc * 64u);
    tf_frag_get_tg4(b0, cur, int(64u * CG), 0, wc, home);
    tf_frag_get_tg4(b1, cur, int(64u * CG), 0, wc + 16, home);
    mma_16x32<false, false>(lo0, hi0, av, b0, b1);
    if (Mmas > 1) {
      tf_frag_get_tg4(b0, cur, int(64u * CG), 0, wc + 32, home);
      tf_frag_get_tg4(b1, cur, int(64u * CG), 0, wc + 48, home);
      mma_16x32<false, false>(lo1, hi1, av, b0, b1);
    }
    // Issue block b+3 into the set just decoded (its consumers are two iterations away).
    if (b + 3 < blocks and decodes) {
      const bool even = ((b + 3) & 1u) == 0u;
      const uint n3 = (b + 3) * 16;
      const uint q0 = tf_exl3_block_word(data, ks, ns, n3, nb, hg, h, wi0, words);
      const uint q1 = tf_exl3_block_word(data, ks, ns, n3, nb, hg, h, wi0 + 1, words);
      const uint q2 = tf_exl3_block_word(data, ks, ns, n3, nb, hg, h, wi0 + 2, words);
      const uint q3 = tf_exl3_block_word(data, ks, ns, n3, nb, hg, h, wi0 + 3, words);
      if (even) { A0 = q0; A1 = q1; A2 = q2; A3 = q3; }
      else { B0 = q0; B1 = q1; B2 = q2; B3 = q3; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const int r0 = int(rb + sgr * 16);
  const int c0 = int(nb + sgc * 64u);
  frag_put_in(lo0, sums, int(a.n), r0, c0, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi0, sums, int(a.n), r0, c0 + 16, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo1, sums, int(a.n), r0, c0 + 32, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi1, sums, int(a.n), r0, c0 + 48, home, int(rb + rows_here), int(nb + cols_here));
}

// Double-buffered prompt body: the next k block decodes into the tile the multiply is not reading, and the multiply
// is issued before that decode, so the tensor units and the codebook ALU are in flight together. One barrier a block
// instead of two; the even-halves Fast cursor is the only path doubled (the shipped body covers the rest).
template <uint RG, uint CG, bool DoDecode, uint Mmas, bool UseTable, bool Funnel = false, uint Rpg = 1>
inline void tf_exl3_prefill_body_db(threadgroup half* wx, const device half* x, const device uchar* trellis,
                                    device float* sums, constant TfExl3Args& a, const device uint* routes,
                                    const device TfExl3Expert* descriptors, const device half* table,
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
  // With Rpg row groups a SIMDgroup (the shape: a simdgroup owns thirty-two rows), the warps decoding one column
  // group are RG / (4 * Rpg) and a lane owns 8 / that many states. Rpg = 1 is the shipped body, bit for bit.
  constexpr uint wpg = RG / (4u * Rpg);
  constexpr uint states = 8u / wpg;
  const uint gwarp = tid / 32u, group = gwarp / wpg, wg = gwarp % wpg;
  const uint h = group % 4u, hg = (group / 4u) * 64u;
  const uint p0 = (wg * 32u + tid % 32u) * states;
  const bool decodes = group < 4u * CG and nb + hg + h * 16u < a.n;
  // The funnel is admissible where the lane's run fits two 32-bit words: halves = 6 (stride 3) with four or eight
  // states, and halves = 8 (stride 4) with four states. Everything else keeps the cursor path.
  const bool use_funnel = Funnel and ((halves == 6u and (states == 8u or states == 4u)) or
                                      (halves == 8u and states == 4u));
  if ((halves & 1u) != 0 or halves > 16) {
    tf_exl3_prefill_body<true, RG, CG, UseTable, false, false, false, false, false, false, Rpg>(wx, x, trellis, sums, a, routes, descriptors, table, nullptr, lane, sg, tid, tptg, tg);
    return;
  }
  uint at[8], off[8], sh[8], rel[8];
  const uint words = 4 * halves;
  const uint wi0 = tf_exl3_map_for(p0, halves) & 63u;
  const uint b0 = (tf_exl3_map_for(p0, halves) >> 6) & 31u;
  {
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
  const uint sgr = Rpg * (sg % (RG / Rpg)), sgc = sg / (RG / Rpg);
  frag<float> lo0[Rpg], hi0[Rpg], lo1[Rpg], hi1[Rpg];
  TF_UNROLL
  for (uint r = 0; r < Rpg; ++r) {
    lo0[r] = frag<float>(0);
    hi0[r] = frag<float>(0);
    lo1[r] = frag<float>(0);
    hi1[r] = frag<float>(0);
  }
  // Prologue: block 0 into buffer 0, then one barrier before the first multiply.
  if (decodes) {
    const uint w0 = tf_exl3_block_word(data, ks, ns, 0, nb, hg, h, wi0, words);
    const uint w1 = tf_exl3_block_word(data, ks, ns, 0, nb, hg, h, wi0 + 1, words);
    if (use_funnel) {
      if (halves == 8u) tf_exl3_decode_funnel<states, false, 4>(wx, table, nullptr, codebook, at, b0, 4, w0, w1);
      else tf_exl3_decode_funnel<states>(wx, table, nullptr, codebook, at, b0, halves >> 1, w0, w1);
    } else {
      const uint w2 = tf_exl3_block_word(data, ks, ns, 0, nb, hg, h, wi0 + 2, words);
      const uint w3 = tf_exl3_block_word(data, ks, ns, 0, nb, hg, h, wi0 + 3, words);
      if (UseTable) tf_exl3_decode_table<states>(wx, table, codebook, at, off, sh, rel, w0, w1, w2, w3);
      else tf_exl3_decode_fast<states>(wx, codebook, at, off, sh, rel, w0, w1, w2, w3);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint cur_buf = 0;
  for (uint kb = 0; kb < a.k; kb += 16) {
    const uint nkb = kb + 16;
    // Issue the next block's stream loads before the multiply, so their latency overlaps the tensor work.
    uint nw0 = 0, nw1 = 0, nw2 = 0, nw3 = 0;
    if (nkb < a.k and decodes and DoDecode) {
      nw0 = tf_exl3_block_word(data, ks, ns, nkb, nb, hg, h, wi0, words);
      nw1 = tf_exl3_block_word(data, ks, ns, nkb, nb, hg, h, wi0 + 1, words);
      if (!use_funnel) {
        nw2 = tf_exl3_block_word(data, ks, ns, nkb, nb, hg, h, wi0 + 2, words);
        nw3 = tf_exl3_block_word(data, ks, ns, nkb, nb, hg, h, wi0 + 3, words);
      }
    }
    frag<half> av[Rpg];
    TF_UNROLL
    for (uint r = 0; r < Rpg; ++r)
      tf_frag_get_dev4(av[r], (const device half*)x, int(a.k), int(rb + (sgr + r) * 16), int(kb), home, int(rb + rows_here));
    if (Mmas > 0) {
      threadgroup half* cur = wx + cur_buf * (16u * 64u * CG);
      frag<half> b0, b1;
      const int wc = int(sgc * 64u);
      tf_frag_get_tg4(b0, cur, int(64u * CG), 0, wc, home);
      tf_frag_get_tg4(b1, cur, int(64u * CG), 0, wc + 16, home);
      TF_UNROLL
      for (uint r = 0; r < Rpg; ++r) mma_16x32<false, false>(lo0[r], hi0[r], av[r], b0, b1);
      if (Mmas > 1) {
        tf_frag_get_tg4(b0, cur, int(64u * CG), 0, wc + 32, home);
        tf_frag_get_tg4(b1, cur, int(64u * CG), 0, wc + 48, home);
        TF_UNROLL
        for (uint r = 0; r < Rpg; ++r) mma_16x32<false, false>(lo1[r], hi1[r], av[r], b0, b1);
      }
    }
    if (nkb < a.k and decodes and DoDecode) {
      threadgroup half* nxt = wx + (cur_buf ^ 1u) * (16u * 64u * CG);
      if (use_funnel) {
        if (halves == 8u) tf_exl3_decode_funnel<states, false, 4>(nxt, table, nullptr, codebook, at, b0, 4, nw0, nw1);
        else tf_exl3_decode_funnel<states>(nxt, table, nullptr, codebook, at, b0, halves >> 1, nw0, nw1);
      } else if (UseTable) tf_exl3_decode_table<states>(nxt, table, codebook, at, off, sh, rel, nw0, nw1, nw2, nw3);
      else tf_exl3_decode_fast<states>(nxt, codebook, at, off, sh, rel, nw0, nw1, nw2, nw3);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (DoDecode) cur_buf ^= 1u;
  }
  const int c0 = int(nb + sgc * 64u);
  TF_UNROLL
  for (uint r = 0; r < Rpg; ++r) {
    const int r0 = int(rb + (sgr + r) * 16);
    frag_put_in(lo0[r], sums, int(a.n), r0, c0, home, int(rb + rows_here), int(nb + cols_here));
    frag_put_in(hi0[r], sums, int(a.n), r0, c0 + 16, home, int(rb + rows_here), int(nb + cols_here));
    frag_put_in(lo1[r], sums, int(a.n), r0, c0 + 32, home, int(rb + rows_here), int(nb + cols_here));
    frag_put_in(hi1[r], sums, int(a.n), r0, c0 + 48, home, int(rb + rows_here), int(nb + cols_here));
  }
  if (Mmas == 0 and tid < 256u) {
    // Keep every decoded store live: read the states this thread wrote, not just the first.
    float acc = 0;
    TF_UNROLL
    for (uint k = 0; k < states; ++k) acc += float(wx[cur_buf * (16u * 64u * CG) + at[k]]);
    sums[tid] = acc;
  }
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
  tf_exl3_prefill_body<false, 4, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
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
  tf_exl3_prefill_body<true, 8, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// One row an expert, four rows a threadgroup: each SIMDgroup decodes its own expert's 16x32 tile once and multiplies
// the single row it owns, writing only that row of the sixteen-row tensor-unit product.
kernel void tf_exl3_matmul_decode_nax(const device half* x [[buffer(0)]],
                                     const device uchar* trellis [[buffer(1)]],
                                     device float* sums [[buffer(2)]],
                                     constant TfExl3Args& a [[buffer(3)]],
                                     const device uint* routes [[buffer(4)]],
                                     const device TfExl3Expert* descriptors [[buffer(5)]],
                                     device float* partials [[buffer(6)]],
#if defined(TF_DECODE_TABLE)
                                     const device half* table [[buffer(7)]],
#endif
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint3 tptg [[threads_per_threadgroup]],
                                     uint3 tg [[threadgroup_position_in_grid]]) {
  // Two weight tiles a SIMDgroup: the next block decodes while this one multiplies, and one barrier a block replaces
  // two because the tile a multiply reads is never the tile a decode writes. TF_DECODE_TABLE: the codebook value is
  // read from the resident table (the engine binds it at buffer 7 for every stage) instead of derived in the ALU. The
  // table holds the format's own decoder's values, qualified entry for entry against the arithmetic path, so the
  // decoded tile -- and the product -- is bit-identical.
#if defined(TF_DECODE_COMPACT)
  // Compact: one row a threadgroup, two 16x32 tiles only (1 KB), so many more groups fit the threadgroup memory and
  // the decode's issue rate is not occupancy-bound (probe: 0.84 against 0.50e12).
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
  // A decode tile has a handful of rows and one threadgroup a layer for the whole k range, so a split takes a slice
  // of the k blocks and the reduce kernel sums the parts in split order.
  const uint split = tg.z;
  const uint blocks = a.k / 16;
  const uint b0 = a.splits > 1 ? blocks * split / a.splits : 0;
  const uint b1 = a.splits > 1 ? blocks * (split + 1) / a.splits : blocks;
  // Eight consecutive states a thread of each sixteen-column tile, so the three stream words they can touch are read
  // once and the cursor and the tile slot come from registers (the prompt kernel's inner loop).
  uint at[8], off[8], sh[8], rel[8];
  const uint words = 4 * halves;
  const uint wi0 = tf_exl3_map_for(lane * 8u, halves) & 63u;
  // TF_DECODE_FUNNEL: the routed pack's halves = 6. Eight consecutive states a lane span 37 bits from start + 16, so
  // two 32-bit stream words cover all eight and the per-state word-pair selection collapses to a two-way select. For
  // halves = 6 the rel = 1 states always start at bit offset <= 16, so the next word after the pair contributes only
  // below the 16-bit state and can be dropped. Bit-identical to the four-word path (verified against
  // tf_exl3_mapped_state for every lane and state); halves ! = 6 keeps the general path.
#if defined(TF_DECODE_FUNNEL)
  // Halves = 8 (the drafter's routed pack) joins halves = 6 ONLY under TF_DECODE_FUNNEL_H8, an optional arm. At
  // halves = 8 every lane's run starts at off0 = 20 (the map's start is 1012 + 32*lane mod 1024), so eight
  // four-bit-spaced states span exactly 44 bits and sit in two words with every rel = 1 state at an offset <= 16 --
  // the same two-word form.
#if defined(TF_DECODE_FUNNEL_H8)
  const bool funnel = halves == 6u or halves == 8u;
#else
  const bool funnel = halves == 6u;
#endif
#else
  const bool funnel = false;
#endif
// THE TWO-WORD READ AT halves = 8 WITHOUT THE FUNNEL'S ASSEMBLY. Every lane's eight states start at off0 = 20 and sit
// in exactly two stream words (rel in {0, 1}); the general path loads four. This drops the two words it cannot touch
// and keeps the cursor's select-chain extraction. Bit-identical by construction (the two words it drops feed only rel
// >= 2, which does not occur at halves = 8).
#if defined(TF_DECODE_TWO)
  const bool two_word = halves == 8u;
#else
  const bool two_word = false;
#endif
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
    if (!funnel and !two_word) {
      pre[0][2] = tf_exl3_word(t0, a2);
      pre[0][3] = tf_exl3_word(t0, a3);
    }
    if (nb + 16 < a.n) {
      const device ushort* t1 = reinterpret_cast<const device ushort*>(data + ulong(b0) * ks + ulong(nb / 16 + 1) * ns);
      pre[1][0] = tf_exl3_word(t1, wi0);
      pre[1][1] = tf_exl3_word(t1, a1);
      if (!funnel and !two_word) {
        pre[1][2] = tf_exl3_word(t1, a2);
        pre[1][3] = tf_exl3_word(t1, a3);
      }
    }
  }
  for (uint bb = b0; bb < b1; ++bb) {
    const uint kb = bb * 16;
    threadgroup half* wxc = ((bb - b0) & 1u) ? wxb : wxa;
    if (pipeline) {
      TF_UNROLL
      for (uint h = 0; h < 2; ++h) {
        if (nb + h * 16 >= a.n) continue;
        const uint w0 = pre[h][0], w1 = pre[h][1];
        uint w2 = 0, w3 = 0;
        if (!funnel and !two_word) {
          w2 = pre[h][2];
          w3 = pre[h][3];
        }
        const uint nxt = bb + 1;
        if (nxt < b1) {
          const device ushort* tn = reinterpret_cast<const device ushort*>(data + ulong(nxt) * ks + ulong(nb / 16 + h) * ns);
          pre[h][0] = tf_exl3_word(tn, wi0);
          pre[h][1] = tf_exl3_word(tn, a1);
          if (!funnel and !two_word) {
            pre[h][2] = tf_exl3_word(tn, a2);
            pre[h][3] = tf_exl3_word(tn, a3);
          }
        }
        if (funnel) {
          // Two words a lane, on our map and with our left-shift funnel. The rel = 1 states start at bit offset <= 16
          // at halves = 6, so the second word's carry is the whole state's source.
          TF_UNROLL
          for (uint k = 0; k < 8; ++k) {
            const uint lo8 = rel[k] == 0 ? w0 : w1;
            const uint hi8 = rel[k] == 0 ? w1 : 0;
            const uint carry = (hi8 >> sh[k]) & (0u - uint(off[k] != 0));
#if defined(TF_DECODE_TABLE)
            wxc[at[k] + h * 16] = tf_exl3_table_value(table, ushort(((lo8 << off[k]) | carry) >> 16), codebook);
#else
            wxc[at[k] + h * 16] = tf_exl3_value(ushort(((lo8 << off[k]) | carry) >> 16), codebook);
#endif
          }
        } else {
          TF_UNROLL
          for (uint k = 0; k < 8; ++k) {
            const uint lo8 = rel[k] == 0 ? w0 : rel[k] == 1 ? w1 : w2;
            const uint hi8 = rel[k] == 0 ? w1 : rel[k] == 1 ? w2 : w3;
            const uint carry = (hi8 >> sh[k]) & (0u - uint(off[k] != 0));
#if defined(TF_DECODE_TABLE)
            wxc[at[k] + h * 16] = tf_exl3_table_value(table, ushort(((lo8 << off[k]) | carry) >> 16), codebook);
#else
            wxc[at[k] + h * 16] = tf_exl3_value(ushort(((lo8 << off[k]) | carry) >> 16), codebook);
#endif
          }
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
#if defined(TF_DECODE_TABLE)
        wxc[r * 32 + h * 16 + c] = tf_exl3_table_value(table, tf_exl3_mapped_state(tile, tf_exl3_map_for(p, halves)), codebook);
#else
        wxc[r * 32 + h * 16 + c] = tf_exl3_value(tf_exl3_mapped_state(tile, tf_exl3_map_for(p, halves)), codebook);
#endif
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

// One row, four column tiles a threadgroup: a dense projection at decode width on the tensor units. Each SIMDgroup
// decodes its own 32-column tile once and multiplies the single row, writing only row 0.
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


// Timing only, values meaningless; no engine path selects it.
kernel void tf_exl3_matvec_nax_regfloor(const device half* x [[buffer(0)]],
                                       const device uchar* trellis [[buffer(1)]],
                                       device float* sums [[buffer(2)]],
                                       constant TfExl3Args& a [[buffer(3)]],
                                       const device uint* routes [[buffer(4)]],
                                       const device TfExl3Expert* descriptors [[buffer(5)]],
                                       uint lane [[thread_index_in_simdgroup]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half xx[16 * 16];
  const uint nb = tg.x * 128 + sg * 32;
  if (nb >= a.n || a.rows == 0) return;
  const uint cols_here = min(32u, a.n - nb);
  const short2 home = frag_home(ushort(lane));
  frag<float> lo = frag<float>(0), hi = frag<float>(0);
  frag<half> b0, b1;
  TF_UNROLL
  for (uint e = 0; e < 8; ++e) { b0[e] = half(1); b1[e] = half(1); }
  for (uint kb = 0; kb < a.k; kb += 16) {
    TF_UNROLL
    for (uint t = 0; t < 8; ++t) {
      const uint idx = lane + 32 * t, r = idx / 16, c = idx % 16;
      xx[idx] = r == 0 ? *(const device half*)(x + ulong(kb) + c) : half(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    frag<half> av;
    tf_frag_get_tg4(av, xx, 16, 0, 0, home);
    mma_16x32<false, false>(lo, hi, av, b0, b1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  frag_put_in(lo, sums, int(a.n), 0, int(nb), home, 1, int(nb + cols_here));
  frag_put_in(hi, sums, int(a.n), 0, int(nb + 16), home, 1, int(nb + cols_here));
}


// The routed decode gives a simdgroup one output column and lets its lanes walk k: that works on its affine pack
// because a column's weights are self-contained. Our pack's trellis is a 16-column x 16-row bit-sharing tile (256
// states in 96 bytes, each state a 16-bit window advancing three bits), so a column's sixteen states are interleaved
// with its fifteen neighbours in the same bytes, and this arm must read every tile its column shares.
kernel void tf_exl3_matvec_col_nax(const device half* x [[buffer(0)]],
                                   const device uchar* trellis [[buffer(1)]],
                                   device float* sums [[buffer(2)]],
                                   constant TfExl3Args& a [[buffer(3)]],
                                   const device uint* routes [[buffer(4)]],
                                   const device TfExl3Expert* descriptors [[buffer(5)]],
                                   const device half* table [[buffer(7)]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]],
                                   uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y;
  const uint col = tg.x * 4u + sg;
  if (row >= a.rows || col >= a.n) return;
  const uint expert = routes[row];
  if (expert >= a.experts) return;
  const device uchar* data = descriptors[expert].trellis;
  const ulong ks = descriptors[expert].tile_k_stride;
  const ulong ns = descriptors[expert].tile_n_stride;
  const uint halves = descriptors[expert].halves;
  const uint codebook = descriptors[expert].codebook;
  const uint cb = col / 16u, cc = col & 15u, q = cc & 7u;
  const device half* xr = x + ulong(row) * a.k;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  uint slot = 0u;
  for (uint k = lane; k < a.k; k += 32u) {
    // The inverse of the tile's state map: (row-in-tile r, column c) -> the stream position p.
    const uint b = k / 16u, r = k % 16u;
    const uint u = (r >> 3) & 1u, m = (r >> 1) & 3u, rb = r & 1u;
    const uint e = 2u * u + rb + 4u * (cc >> 3);
    const uint p = 8u * (4u * q + m) + e;
    const device ushort* tile = reinterpret_cast<const device ushort*>(data + ulong(b) * ks + ulong(cb) * ns);
    const ushort s = tf_exl3_mapped_state(tile, tf_exl3_map_for(p, halves));
    const float v = float(tf_exl3_table_value(table, s, codebook));
    acc[slot] = fma(float(xr[k]), v, acc[slot]);
    slot = (slot + 1u) & 7u;
  }
  float t = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
  t += simd_shuffle_xor(t, 16u);
  t += simd_shuffle_xor(t, 8u);
  t += simd_shuffle_xor(t, 4u);
  t += simd_shuffle_xor(t, 2u);
  t += simd_shuffle_xor(t, 1u);
  if (lane == 0u) sums[ulong(row) * a.n + col] = t;
}

#ifndef TF_AFF_WARPS
#define TF_AFF_WARPS 16
#endif

// This body is what a repack would actually dispatch: a 16-column x 16-k tile of straight 3-bit codes (256 codes in
// 96 bytes, no bit sharing), group-64 bf16 scales, our sixteen-warp k split, and the fma body's activation reads and
// reduction. Codes are exact integers times their group's scale; the trellis decode is gone.
kernel void tf_affine_tile3_nax(const device half* x [[buffer(0)]],
                                const device uchar* codes [[buffer(1)]],
                                device float* sums [[buffer(2)]],
                                constant TfExl3Args& a [[buffer(3)]],
                                const device uint* routes [[buffer(4)]],
                                const device TfExl3Expert* descriptors [[buffer(5)]],
                                const device half* scales [[buffer(7)]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y;
  const uint nb = tg.x * 16u;
  if (row >= a.rows) return;
  const uint expert = routes[row];
  if (expert >= a.experts) return;
  const device uchar* data = descriptors[expert].trellis;
  const ulong ks = descriptors[expert].tile_k_stride;
  const ulong ns = descriptors[expert].tile_n_stride;
  const uint blocks = a.k / 16u;
  const uint G = a.k / 64u;
  const uint W = TF_AFF_WARPS;
  const uint b0 = blocks * sg / W, b1 = blocks * (sg + 1u) / W;
  // The lane's eight codes are one 24-bit run of the tile: words (3L)/4 and +1, at a byte phase of 3L mod 4.
  const uint w0i = (3u * lane) / 4u, sh = 8u * (3u * lane & 3u);
  const uint cn = lane / 4u, cf = cn + 8u;
  const device half* scn = scales + (ulong(expert) * a.n + nb + cn) * G;
  const device half* scf = scales + (ulong(expert) * a.n + nb + cf) * G;
  const device half* xr = x + ulong(row) * a.k;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float sn = 0.0f, sf = 0.0f;
  // The next block's two words are loaded a block early (the shipped body's own prefetch): without it every block's
  // DRAM latency is exposed once and the body runs at a third of its rate.
  const ulong base_nb = ulong(nb / 16u) * ns;
  uint lo = 0u, hi = 0u;
  {
    const device uint* t0 = reinterpret_cast<const device uint*>(data + ulong(b0) * ks + base_nb);
    lo = t0[w0i];
    hi = t0[w0i + 1u];
  }
  for (uint bb = b0; bb < b1; ++bb) {
    const uint kb = bb * 16u;
    const uint c0 = lo, c1 = hi;
    if (bb + 1u < b1) {
      const device uint* tn = reinterpret_cast<const device uint*>(data + ulong(bb + 1u) * ks + base_nb);
      lo = tn[w0i];
      hi = tn[w0i + 1u];
    }
    const uint funnel = sh == 0u ? c0 : ((c0 >> sh) | (c1 << ((32u - sh) & 31u)));
    if ((bb & 3u) == 0u) {
      sn = float(scn[bb / 4u]);
      sf = float(scf[bb / 4u]);
    }
    const device half* xrow = xr + kb + 2u * (lane & 3u);
    const half2 xa = *(const device half2*)xrow;
    const half2 xb = *(const device half2*)(xrow + 8);
    const float x0 = float(xa.x), x1 = float(xa.y), x2 = float(xb.x), x3 = float(xb.y);
    TF_UNROLL
    for (uint k = 0; k < 8; ++k) {
      const float code = float((funnel >> (3u * k)) & 7u);
      const float scale = k < 4u ? sn : sf;
      const uint odd = k & 1u, up = (k >> 1) & 1u;
      const float xv = odd == 0 ? (up == 0 ? x0 : x2) : (up == 0 ? x1 : x3);
      acc[k] = fma(xv, code * scale, acc[k]);
    }
  }
  float acc0 = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  float acc1 = (acc[4] + acc[5]) + (acc[6] + acc[7]);
  threadgroup float red[TF_AFF_WARPS][64];
  red[sg][lane] = acc0;
  red[sg][32 + lane] = acc1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0u) return;
  acc0 = 0.0f;
  acc1 = 0.0f;
  for (uint w = 0; w < W; ++w) {
    acc0 += red[w][lane];
    acc1 += red[w][32 + lane];
  }
  acc0 += simd_shuffle_xor(acc0, 1);
  acc0 += simd_shuffle_xor(acc0, 2);
  acc1 += simd_shuffle_xor(acc1, 1);
  acc1 += simd_shuffle_xor(acc1, 2);
  if ((lane & 3u) == 0u) {
    sums[ulong(row) * a.n + nb + lane / 4u] = acc0;
    sums[ulong(row) * a.n + nb + 8u + lane / 4u] = acc1;
  }
}

// This body is the form a repack would dispatch: a 16-column x 16-k tile of straight 3-bit codes (256 codes, 96
// bytes, no bit sharing), group-64 bf16 scales, W warps a threadgroup splitting k, register-only accumulators, a
// two-block prefetch, and no threadgroup round trip at W = 1. The code index of (k, col) inside a tile is 32*(col &
// 7) + 8*((k >> 1) & 3) + ((k & 1) + 2*((k >> 3) & 1)) + 4*((col >> 3) & 1), which is the pack's own choice;
// `tf_affine_tile3_ref` decodes the same bytes by the same map.
#ifndef TF_AFF_WARPS
#define TF_AFF_WARPS 16
#endif

kernel void tf_affine_tile3b_nax(const device half* x [[buffer(0)]],
                                 const device uchar* codes [[buffer(1)]],
                                 device float* sums [[buffer(2)]],
                                 constant TfExl3Args& a [[buffer(3)]],
                                 const device uint* routes [[buffer(4)]],
                                 const device TfExl3Expert* descriptors [[buffer(5)]],
                                 const device half* scales [[buffer(7)]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y;
  const uint nb = tg.x * 16u;
  if (row >= a.rows) return;
  const uint expert = routes[row];
  if (expert >= a.experts) return;
  const device uchar* data = descriptors[expert].trellis;
  const ulong ks = descriptors[expert].tile_k_stride;
  const ulong ns = descriptors[expert].tile_n_stride;
  const uint blocks = a.k / 16u;
  const uint G = a.k / 64u;
  const uint W = TF_AFF_WARPS;
  const uint b0 = blocks * sg / W, b1 = blocks * (sg + 1u) / W;
  if (b0 >= b1) return;
  // The lane's eight codes are one 24-bit run: words (3L)/4 and +1 at a byte phase of 3L mod 4.
  const uint w0i = (3u * lane) / 4u, sh = 8u * (3u * lane & 3u);
  const uint cn = lane / 4u, cf = cn + 8u;
  const device half* scn = scales + (ulong(expert) * a.n + nb + cn) * G;
  const device half* scf = scales + (ulong(expert) * a.n + nb + cf) * G;
  const device half* xr = x + ulong(row) * a.k;
  const ulong base_nb = ulong(nb / 16u) * ns;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float sn = 0.0f, sf = 0.0f;
  uint c0 = 0u, c1 = 0u, n0 = 0u, n1 = 0u;
  {
    const device uint* t0 = reinterpret_cast<const device uint*>(data + ulong(b0) * ks + base_nb);
    c0 = t0[w0i];
    c1 = t0[w0i + 1u];
  }
  if (b0 + 1u < b1) {
    const device uint* t1 = reinterpret_cast<const device uint*>(data + ulong(b0 + 1u) * ks + base_nb);
    n0 = t1[w0i];
    n1 = t1[w0i + 1u];
  }
  for (uint bb = b0; bb < b1; ++bb) {
    const uint l0 = c0, l1 = c1;
    c0 = n0;
    c1 = n1;
    if (bb + 2u < b1) {
      const device uint* tn = reinterpret_cast<const device uint*>(data + ulong(bb + 2u) * ks + base_nb);
      n0 = tn[w0i];
      n1 = tn[w0i + 1u];
    } else {
      n0 = 0u;
      n1 = 0u;
    }
    const uint funnel = sh == 0u ? l0 : ((l0 >> sh) | (l1 << ((32u - sh) & 31u)));
    if ((bb & 3u) == 0u) {
      sn = float(scn[bb / 4u]);
      sf = float(scf[bb / 4u]);
    }
    const device half* xrow = xr + bb * 16u + 2u * (lane & 3u);
    const half2 xa = *(const device half2*)xrow;
    const half2 xb = *(const device half2*)(xrow + 8);
    const float x0 = float(xa.x), x1 = float(xa.y), x2 = float(xb.x), x3 = float(xb.y);
    TF_UNROLL
    for (uint k = 0; k < 8; ++k) {
      const float code = float((funnel >> (3u * k)) & 7u);
      const float scale = k < 4u ? sn : sf;
      const uint odd = k & 1u, up = (k >> 1) & 1u;
      const float xv = odd == 0 ? (up == 0 ? x0 : x2) : (up == 0 ? x1 : x3);
      acc[k] = fma(xv, code * scale, acc[k]);
    }
  }
  float t0 = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  float t1 = (acc[4] + acc[5]) + (acc[6] + acc[7]);
  t0 += simd_shuffle_xor(t0, 1u);
  t0 += simd_shuffle_xor(t0, 2u);
  t1 += simd_shuffle_xor(t1, 1u);
  t1 += simd_shuffle_xor(t1, 2u);
  if (W == 1u) {
    if ((lane & 3u) == 0u) {
      sums[ulong(row) * a.n + nb + lane / 4u] = t0;
      sums[ulong(row) * a.n + nb + 8u + lane / 4u] = t1;
    }
    return;
  }
  threadgroup float red[TF_AFF_WARPS][64];
  red[sg][lane] = t0;
  red[sg][32u + lane] = t1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0u) return;
  float u0 = 0.0f, u1 = 0.0f;
  for (uint w = 0; w < W; ++w) {
    u0 += red[w][lane];
    u1 += red[w][32u + lane];
  }
  if ((lane & 3u) == 0u) {
    sums[ulong(row) * a.n + nb + lane / 4u] = u0;
    sums[ulong(row) * a.n + nb + 8u + lane / 4u] = u1;
  }
}

kernel void tf_affine_fill_one(device ushort* t [[buffer(0)]],
                               uint gid [[thread_position_in_grid]]) {
  t[gid] = 0x3C00u;
}

kernel void tf_affine_fill_alt(device ushort* t [[buffer(0)]],
                               constant uint& G [[buffer(1)]],
                               uint gid [[thread_position_in_grid]]) {
  t[gid] = ((gid / G) & 1u) == 0u ? 0x3C00u : 0x4000u;
}


// A min/max affine quantizer needs both a scale and a bias (eight uniform levels spanning the group's range), and the
// own tile_qmv body carries scales AND biases. Buffer 7 holds them interleaved per group: [scale, bias] at
// 2*((expert*n + col)*G + g). The extra per-weight op is one fma: code*scale + bias.
kernel void tf_affine_tile3s_nax(const device half* x [[buffer(0)]],
                                 const device uchar* codes [[buffer(1)]],
                                 device float* sums [[buffer(2)]],
                                 constant TfExl3Args& a [[buffer(3)]],
                                 const device uint* routes [[buffer(4)]],
                                 const device TfExl3Expert* descriptors [[buffer(5)]],
                                 const device half* scales [[buffer(7)]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y;
  const uint nb = tg.x * 16u;
  if (row >= a.rows) return;
  const uint expert = routes[row];
  if (expert >= a.experts) return;
  const device uchar* data = descriptors[expert].trellis;
  const ulong ks = descriptors[expert].tile_k_stride;
  const ulong ns = descriptors[expert].tile_n_stride;
  const uint blocks = a.k / 16u;
  const uint G = a.k / 64u;
  const uint W = TF_AFF_WARPS;
  const uint b0 = blocks * sg / W, b1 = blocks * (sg + 1u) / W;
  if (b0 >= b1) return;
  const uint w0i = (3u * lane) / 4u, sh = 8u * (3u * lane & 3u);
  const uint cn = lane / 4u, cf = cn + 8u;
  const device half* scn = scales + 2ull * (ulong(expert) * a.n + nb + cn) * G;
  const device half* scf = scales + 2ull * (ulong(expert) * a.n + nb + cf) * G;
  const device half* xr = x + ulong(row) * a.k;
  const ulong base_nb = ulong(nb / 16u) * ns;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float sn = 0.0f, sf = 0.0f, bn = 0.0f, bf = 0.0f;
  uint c0 = 0u, c1 = 0u, n0 = 0u, n1 = 0u;
  {
    const device uint* t0 = reinterpret_cast<const device uint*>(data + ulong(b0) * ks + base_nb);
    c0 = t0[w0i];
    c1 = t0[w0i + 1u];
  }
  if (b0 + 1u < b1) {
    const device uint* t1 = reinterpret_cast<const device uint*>(data + ulong(b0 + 1u) * ks + base_nb);
    n0 = t1[w0i];
    n1 = t1[w0i + 1u];
  }
  for (uint bb = b0; bb < b1; ++bb) {
    const uint l0 = c0, l1 = c1;
    c0 = n0;
    c1 = n1;
    if (bb + 2u < b1) {
      const device uint* tn = reinterpret_cast<const device uint*>(data + ulong(bb + 2u) * ks + base_nb);
      n0 = tn[w0i];
      n1 = tn[w0i + 1u];
    } else {
      n0 = 0u;
      n1 = 0u;
    }
    const uint funnel = sh == 0u ? l0 : ((l0 >> sh) | (l1 << ((32u - sh) & 31u)));
    if ((bb & 3u) == 0u) {
      const uint gi = 2u * (bb / 4u);
      sn = float(scn[gi]);
      bn = float(scn[gi + 1u]);
      sf = float(scf[gi]);
      bf = float(scf[gi + 1u]);
    }
    const device half* xrow = xr + bb * 16u + 2u * (lane & 3u);
    const half2 xa = *(const device half2*)xrow;
    const half2 xb = *(const device half2*)(xrow + 8);
    const float x0 = float(xa.x), x1 = float(xa.y), x2 = float(xb.x), x3 = float(xb.y);
    TF_UNROLL
    for (uint k = 0; k < 8; ++k) {
      const float code = float((funnel >> (3u * k)) & 7u);
      const float scale = k < 4u ? sn : sf;
      const float bias = k < 4u ? bn : bf;
      const uint odd = k & 1u, up = (k >> 1) & 1u;
      const float xv = odd == 0 ? (up == 0 ? x0 : x2) : (up == 0 ? x1 : x3);
      acc[k] = fma(xv, fma(code, scale, bias), acc[k]);
    }
  }
  float t0 = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  float t1 = (acc[4] + acc[5]) + (acc[6] + acc[7]);
  t0 += simd_shuffle_xor(t0, 1u);
  t0 += simd_shuffle_xor(t0, 2u);
  t1 += simd_shuffle_xor(t1, 1u);
  t1 += simd_shuffle_xor(t1, 2u);
  if (W == 1u) {
    if ((lane & 3u) == 0u) {
      sums[ulong(row) * a.n + nb + lane / 4u] = t0;
      sums[ulong(row) * a.n + nb + 8u + lane / 4u] = t1;
    }
    return;
  }
  threadgroup float red[TF_AFF_WARPS][64];
  red[sg][lane] = t0;
  red[sg][32u + lane] = t1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0u) return;
  float u0 = 0.0f, u1 = 0.0f;
  for (uint w = 0; w < W; ++w) {
    u0 += red[w][lane];
    u1 += red[w][32u + lane];
  }
  if ((lane & 3u) == 0u) {
    sums[ulong(row) * a.n + nb + lane / 4u] = u0;
    sums[ulong(row) * a.n + nb + 8u + lane / 4u] = u1;
  }
}

// One thread an output, the same bytes, the same map, a serial k walk: a mismatch is a port bug in the warp body, not
// a numerics class.
kernel void tf_affine_tile3_ref(const device half* x [[buffer(0)]],
                                const device uchar* codes [[buffer(1)]],
                                device float* sums [[buffer(2)]],
                                constant TfExl3Args& a [[buffer(3)]],
                                const device uint* routes [[buffer(4)]],
                                const device TfExl3Expert* descriptors [[buffer(5)]],
                                const device half* scales [[buffer(7)]],
                                uint2 gid [[thread_position_in_grid]]) {
  const uint col = gid.x, row = gid.y;
  if (row >= a.rows || col >= a.n) return;
  const uint expert = routes[row];
  if (expert >= a.experts) return;
  const device uchar* data = descriptors[expert].trellis;
  const ulong ks = descriptors[expert].tile_k_stride;
  const ulong ns = descriptors[expert].tile_n_stride;
  const uint p = col & 7u, h = (col >> 3u) & 1u, cb = col / 16u;
  const uint G = a.k / 64u;
  const device half* sc = scales + (ulong(expert) * a.n + col) * G;
  const device half* xr = x + ulong(row) * a.k;
  const ulong tile = ulong(cb) * ns;
  float acc = 0.0f;
  for (uint k = 0; k < a.k; ++k) {
    const uint b = k / 16u, kk = k & 15u;
    const uint q = (kk >> 1) & 3u;
    const uint j = (kk & 1u) + 2u * ((kk >> 3u) & 1u) + 4u * h;
    const uint idx = 32u * p + 8u * q + j;
    const uint bit = 3u * idx;
    const device uchar* tp = data + ulong(b) * ks + tile;
    // A 3-bit code can straddle a 32-bit word, so read the two bytes that hold it (a code never starts in the last
    // byte of a tile at an offset that would leave the 96-byte tile).
    const device uchar* bp = tp + (bit >> 3u);
    const uint two = uint(bp[0]) | (uint(bp[1]) << 8);
    const uint code = (two >> (bit & 7u)) & 7u;
    acc = fma(float(xr[k]), float(code) * float(sc[k >> 6u]), acc);
  }
  sums[ulong(row) * a.n + col] = acc;
}

// The repack can order the pack any way, so this is the form the plan actually needs: a column's k-run is contiguous
// (a 32-code block is 12 aligned bytes), the lanes walk blocks, and the 16-column bit-sharing tile -- and its 13.8 KB
// k-stride -- are gone. Four warps a threadgroup, a warp a column; codes are exact integers times their group-64
// scale.
kernel void tf_affine_col3_nax(const device half* x [[buffer(0)]],
                               const device uchar* codes [[buffer(1)]],
                               device float* sums [[buffer(2)]],
                               constant TfExl3Args& a [[buffer(3)]],
                               const device uint* routes [[buffer(4)]],
                               const device TfExl3Expert* descriptors [[buffer(5)]],
                               const device half* scales [[buffer(7)]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]],
                               uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y;
  const uint col = tg.x * 4u + sg;
  if (row >= a.rows || col >= a.n) return;
  const uint expert = routes[row];
  if (expert >= a.experts) return;
  const uint blocks = a.k / 32u;
  const device uint* colp = reinterpret_cast<const device uint*>(
      descriptors[expert].trellis + ulong(col) * ulong(blocks) * 12u);
  const device half* scp = scales + (ulong(expert) * a.n + col) * (a.k / 64u);
  const device half* xr = x + ulong(row) * a.k;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint b = lane; b < blocks; b += 32u) {
    const uint k0 = b * 32u;
    const uint w0 = colp[3u * b], w1 = colp[3u * b + 1u], w2 = colp[3u * b + 2u];
    const float s = float(scp[b >> 1u]);
    const device half* xb = xr + k0;
    TF_UNROLL
    for (uint i = 0; i < 32; ++i) {
      const uint bit = 3u * i, word = bit >> 5, sh = bit & 31u;
      const uint wc = word == 0u ? w0 : (word == 1u ? w1 : w2);
      const uint wn = word == 0u ? w1 : (word == 1u ? w2 : 0u);
      const uint carry = sh == 0u ? 0u : (wn << (32u - sh));
      const float code = float(((wc >> sh) | carry) & 7u);
      acc[i & 7u] = fma(float(xb[i]), code * s, acc[i & 7u]);
    }
  }
  float t = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
  t += simd_shuffle_xor(t, 16u);
  t += simd_shuffle_xor(t, 8u);
  t += simd_shuffle_xor(t, 4u);
  t += simd_shuffle_xor(t, 2u);
  t += simd_shuffle_xor(t, 1u);
  if (lane == 0u) sums[ulong(row) * a.n + col] = t;
}

// Queue item 2: the routed 64 x 128 tile on four warps. Eight accumulator fragments a lane (64 float registers). k
// order, MMA order and per-value arithmetic are 's, so the output is bit-identical; only the dispatch geometry moves.
template <bool Fast>
inline void tf_exl3_prefill_body_wide(threadgroup half* wx, const device half* x, const device uchar* trellis,
                                      device float* sums, constant TfExl3Args& a, const device uint* routes,
                                      const device TfExl3Expert* descriptors,
                                      uint lane, uint sg, uint tid, uint tptg, uint3 tg) {
  const uint rb = tg.y * 64u, nb = tg.x * 128u;
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
  const uint rows_here = min(64u, a.rows - rb);
  const uint cols_here = min(128u, a.n - nb);
  // Four warps: this warp is row group `sg` and decodes column tiles h and h+4 of the 128-column group.
  const uint h = sg;
  const uint p0 = lane * 8u;
  uint at[8], off[8], sh[8], rel[8];
  const uint words = 4 * halves;
  const uint wi0 = tf_exl3_map_for(p0, halves) & 63u;
  {
    const uint ring = 128 * halves;
    const uint cur = (wi0 << 5) | ((tf_exl3_map_for(p0, halves) >> 6) & 31u);
    TF_UNROLL
    for (uint k = 0; k < 8; ++k) {
      const uint p = p0 + k, l = p / 8, e = p % 8;
      at[k] = (2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u)) * 128u + h * 16u + (l / 4 + 8 * (e >> 2));
      uint c8 = cur + k * (halves >> 1);
      if (c8 >= ring) c8 -= ring;
      const uint wi = c8 >> 5;
      off[k] = c8 & 31u;
      sh[k] = (32u - off[k]) & 31u;
      rel[k] = wi >= wi0 ? wi - wi0 : wi + words - wi0;
    }
  }
  const uint a1 = wi0 + 1 < words ? wi0 + 1 : wi0 + 1 - words;
  const uint a2 = wi0 + 2 < words ? wi0 + 2 : wi0 + 2 - words;
  const uint a3 = wi0 + 3 < words ? wi0 + 3 : wi0 + 3 - words;
  uint p0v = 0, p1v = 0, p2v = 0, p3v = 0, q0v = 0, q1v = 0, q2v = 0, q3v = 0;
  if (Fast) {
    const device ushort* t0 = reinterpret_cast<const device ushort*>(data + ulong(nb / 16 + h) * ns);
    const device ushort* t1 = reinterpret_cast<const device ushort*>(data + ulong(nb / 16 + h + 4) * ns);
    p0v = tf_exl3_word(t0, wi0); p1v = tf_exl3_word(t0, a1); p2v = tf_exl3_word(t0, a2); p3v = tf_exl3_word(t0, a3);
    q0v = tf_exl3_word(t1, wi0); q1v = tf_exl3_word(t1, a1); q2v = tf_exl3_word(t1, a2); q3v = tf_exl3_word(t1, a3);
  }
  frag<float> lo0 = frag<float>(0), hi0 = frag<float>(0), lo1 = frag<float>(0), hi1 = frag<float>(0);
  frag<float> lo2 = frag<float>(0), hi2 = frag<float>(0), lo3 = frag<float>(0), hi3 = frag<float>(0);
  for (uint kb = 0; kb < a.k; kb += 16) {
    {
      const device ushort* t0 = reinterpret_cast<const device ushort*>(data + ulong(kb / 16) * ks + ulong(nb / 16 + h) * ns);
      const device ushort* t1 = reinterpret_cast<const device ushort*>(data + ulong(kb / 16) * ks + ulong(nb / 16 + h + 4) * ns);
      if (Fast and (halves & 1u) == 0 and halves <= 16) {
        const uint w0 = p0v, w1 = p1v, w2 = p2v, w3 = p3v;
        const uint v0 = q0v, v1 = q1v, v2 = q2v, v3 = q3v;
        const uint nkb = kb + 16;
        if (nkb < a.k) {
          const device ushort* n0 = reinterpret_cast<const device ushort*>(data + ulong(nkb / 16) * ks + ulong(nb / 16 + h) * ns);
          const device ushort* n1 = reinterpret_cast<const device ushort*>(data + ulong(nkb / 16) * ks + ulong(nb / 16 + h + 4) * ns);
          p0v = tf_exl3_word(n0, wi0); p1v = tf_exl3_word(n0, a1); p2v = tf_exl3_word(n0, a2); p3v = tf_exl3_word(n0, a3);
          q0v = tf_exl3_word(n1, wi0); q1v = tf_exl3_word(n1, a1); q2v = tf_exl3_word(n1, a2); q3v = tf_exl3_word(n1, a3);
        }
        TF_UNROLL
        for (uint k = 0; k < 8; ++k) {
          const uint lo = rel[k] == 0 ? w0 : rel[k] == 1 ? w1 : w2;
          const uint hi = rel[k] == 0 ? w1 : rel[k] == 1 ? w2 : w3;
          const uint carry = (hi >> sh[k]) & (0u - uint(off[k] != 0));
          wx[at[k]] = tf_exl3_value(ushort(((lo << off[k]) | carry) >> 16), codebook);
          const uint loq = rel[k] == 0 ? v0 : rel[k] == 1 ? v1 : v2;
          const uint hiq = rel[k] == 0 ? v1 : rel[k] == 1 ? v2 : v3;
          const uint carryq = (hiq >> sh[k]) & (0u - uint(off[k] != 0));
          wx[at[k] + 64u] = tf_exl3_value(ushort(((loq << off[k]) | carryq) >> 16), codebook);
        }
      } else {
        TF_UNROLL
        for (uint k = 0; k < 8; ++k) {
          const uint p = p0 + k;
          wx[at[k]] = tf_exl3_value(tf_exl3_mapped_state(t0, tf_exl3_map_for(p, halves)), codebook);
          wx[at[k] + 64u] = tf_exl3_value(tf_exl3_mapped_state(t1, tf_exl3_map_for(p, halves)), codebook);
        }
      }
    }
    frag<half> av;
    tf_frag_get_dev4(av, (const device half*)x, int(a.k), int(rb + sg * 16), int(kb), home, int(rb + rows_here));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      frag<half> b0, b1;
      tf_frag_get_tg4(b0, wx, 128, 0, 0, home);
      tf_frag_get_tg4(b1, wx, 128, 0, 16, home);
      mma_16x32<false, false>(lo0, hi0, av, b0, b1);
      tf_frag_get_tg4(b0, wx, 128, 0, 32, home);
      tf_frag_get_tg4(b1, wx, 128, 0, 48, home);
      mma_16x32<false, false>(lo1, hi1, av, b0, b1);
      tf_frag_get_tg4(b0, wx, 128, 0, 64, home);
      tf_frag_get_tg4(b1, wx, 128, 0, 80, home);
      mma_16x32<false, false>(lo2, hi2, av, b0, b1);
      tf_frag_get_tg4(b0, wx, 128, 0, 96, home);
      tf_frag_get_tg4(b1, wx, 128, 0, 112, home);
      mma_16x32<false, false>(lo3, hi3, av, b0, b1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const int r0 = int(rb + sg * 16);
  const int c0 = int(nb);
  frag_put_in(lo0, sums, int(a.n), r0, c0, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi0, sums, int(a.n), r0, c0 + 16, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo1, sums, int(a.n), r0, c0 + 32, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi1, sums, int(a.n), r0, c0 + 48, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo2, sums, int(a.n), r0, c0 + 64, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi2, sums, int(a.n), r0, c0 + 80, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo3, sums, int(a.n), r0, c0 + 96, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi3, sums, int(a.n), r0, c0 + 112, home, int(rb + rows_here), int(nb + cols_here));
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
  tf_exl3_prefill_body<true, 4, 2, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
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
  tf_exl3_prefill_body<true, 8, 2, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
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
  tf_exl3_prefill_body<true, 4, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}


// The OCCUPANCY-KEEPING FORM -- the routed prompt tile with the activation staged in threadgroup memory, once a k
// block, and SHARED by the two column groups of a row group (one column tile's activation read shared by two column
// groups; the fragment in threadgroup memory; one more barrier a block -- here the SAME barrier that publishes the
// weight tile, so the block's barrier count is unchanged). The two column groups of a row group read the device ONCE
// and read the staged fragment back, so the activation's device traffic falls by half against the same tile widened
// to 128 columns. The values, their order, the k order and the MMA order are the shipped body's, so the output is
// bit-identical.
kernel void tf_exl3_matmul_prefill_nax61(const device half* x [[buffer(0)]],
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
  threadgroup half wx[16 * 64 * 2 + 4 * 16 * 16];
  tf_exl3_prefill_body<true, 4, 2, false, true, false, false, true, false, false, 1, false, false, 2>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The ABLATION -- the activation fragment is read ONCE, before the k loop, and reused. A diagnostic: the output
// differs by construction, and the delta against the same body reading it every block IS the activation's per-block
// cost (the record's instrument first: the activation read's share of the routed tile).
kernel void tf_exl3_matmul_prefill_nax62(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, false, 1>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The CONTROL for -- the same 128-column tile with the shipped per-warp activation read (so the two arms differ only
// in the staging). Bit-identical to by construction.
kernel void tf_exl3_matmul_prefill_nax63(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 2, false, true, false, false, true, false, false, 1, false, false, 0>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The engine's routed 64 x 64 prompt tile with the funnel read -- two stream words a lane, the eight windows at one
// shift a state, on our map and our layout. The values, their order and the k order are the shipped body's, so the
// output is bit-identical; only the word traffic and the cursor's select chain leave.
kernel void tf_exl3_matmul_prefill_nax40(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The same body with the mul1 value dropped (the raw state stored): the window extraction and the stores without the
// codebook arithmetic. A diagnostic -- it isolates the value path inside the decode; its output differs.
kernel void tf_exl3_matmul_prefill_nax41(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The engine's routed 64 x 64 prompt tile with the funnel read -- the two-word funnel and the eight windows at
// compile-time shifts, on our map and our ring. The values, their order and the k order are the shipped body's, so
// the output is bit-identical; only the word traffic and the cursor's select chain leave. The funnel needs the routed
// pack's width (halves = 6, three bits a state); at any other width the kernel reads the shipped path's four words
// and decodes as before.
kernel void tf_exl3_matmul_prefill_nax42(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The funnel read with the value read from the resident codebook table (buffer 7). One load a state instead of the
// mul1's ALU. The table holds the format's own decoder's values, so the values, their order and the k order are the
// shipped body's: bit-identical.
kernel void tf_exl3_matmul_prefill_nax43(const device half* x [[buffer(0)]],
                                        const device uchar* trellis [[buffer(1)]],
                                        device float* sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device uint* routes [[buffer(4)]],
                                        const device TfExl3Expert* descriptors [[buffer(5)]],
                                        const device half* table [[buffer(7)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tptg [[threads_per_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[16 * 64];
  tf_exl3_prefill_body<true, 4, 1, false, true, false>(wx, x, trellis, sums, a, routes, descriptors, table, nullptr, lane, sg, tid, tptg.x, tg);
}

// The funnel read with the value read from the 1024-entry threadgroup LUT (the prefill kernel's read). The LUT's
// entries are the table's, so the values are the same, bit-identical.
kernel void tf_exl3_matmul_prefill_nax44(const device half* x [[buffer(0)]],
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
  threadgroup half lut[1024];
  const half scale = as_type<half>(ushort(0x1EEEu));
  const half bias = as_type<half>(ushort(0xC931u));
  TF_UNROLL
  for (uint j = 0; j < 8u; ++j) lut[tid + 128u * j] = half(fma(float(tid + 128u * j + 1024u), float(scale), float(bias)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tf_exl3_prefill_body<true, 4, 1, false, true, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lut, lane, sg, tid, tptg.x, tg);
}

// The funnel read with the value read cheaply -- a diagnostic; its output differs by construction.
kernel void tf_exl3_matmul_prefill_nax47(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The double-buffered routed tile with the funnel read. The double-buffered body decodes the next k block into the
// tile the multiply is not reading and issues the multiply before that decode, so the tensor units and the codebook
// ALU are in flight together; the funnel makes the decode cheap enough for the overlap to pay.
kernel void tf_exl3_matmul_prefill_nax46(const device half* x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_db<4, 1, true, 4, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

// The routed 64 x 64 prompt tile with the funnel read and the padded rows skipped. The engine's grouping pads each
// expert's block to sixty-four rows, so a 3,619-token prompt's 21,714 routed rows are dispatched as 45,888 rows --
// 2.11 times the real work; the padding rows carry the sentinel expert id, so a warp reads its own sixteen route ids
// and skips its multiply when they are all padding. The real rows' values, their order and the k order are the
// shipped body's, so the reply is bit-identical. The tile's contents change, so the output differs by construction.
kernel void tf_exl3_matmul_prefill_nax64(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, false, 0, 3>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// the DECODE KEPT, the eight scattered stores GONE (the values are folded into a register accumulator
// instead). The floor of the store path. A diagnostic: the tile is never written.
kernel void tf_exl3_matmul_prefill_nax65(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, false, 0, 1>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The shipped body plus the Mode 1 accumulator: the calibration for the accumulator's own cost, so the store
// path's cost falls out of the two.
kernel void tf_exl3_matmul_prefill_nax66(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, false, 0, 4>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// the shipped body with the codebook arithmetic dropped (the raw window stored, the extraction and the
// stores kept). The value path's price, in the shipped configuration. A diagnostic.
kernel void tf_exl3_matmul_prefill_nax67(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, true, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// the eight stores KEPT, the EXTRACTION dropped (the stored value is a cheap function of the k block, so the
// stores stay in the loop). The extraction's price, in the shipped configuration.
kernel void tf_exl3_matmul_prefill_nax68(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, false, 0, 2>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax48(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The routed 64x64 prompt tile with the stream word read as one 4-byte load -- the wide form's read (its lane reads
// 32-bit stream words; ours read the same bytes as two 2-byte loads, so a lane issued four memory instructions a
// block where it now issues two). The values, their order, the k order and the MMA order are 's, so the output is
// bit-identical. The kernel falls back to the shipped read when the tile's byte address is not 4-byte aligned.
kernel void tf_exl3_matmul_prefill_nax57(const device half* x [[buffer(0)]],
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
  // The tile's byte address: the expert's trellis plus the tile's k and n offsets. A 4-byte load needs it 4-byte
  // aligned, and the strides must keep it there.
  const uint rb = tg.y * 64u;
  uint e = 0u;
  if (a.indexed != 1u and rb < a.rows) e = routes[rb];
  if (a.experts != 0u and e >= a.experts) e = 0u;
  const device uchar* data = (a.indexed != 0u) ? descriptors[e].trellis : trellis;
  const ulong ks = (a.indexed != 0u) ? descriptors[e].tile_k_stride : a.tile_k_stride;
  const ulong ns = (a.indexed != 0u) ? descriptors[e].tile_n_stride : a.tile_n_stride;
  const bool wide = (reinterpret_cast<ulong>(data) & 3ul) == 0ul and (ks & 3ul) == 0ul and (ns & 3ul) == 0ul;
  if (wide) {
    tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
  } else {
    tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
  }
}

// The store path's modes on the shipped body (the funnel path as its only path, the twenty-four dead cursor
// registers not computed). The store path's costs are only worth anything on the body the engine runs. Mode 1: the
// eight decoded values written CONTIGUOUSLY, one lane's run of eight halves -- the wide form's store form (a lane's
// values go out as wide stores, not as eight scattered two-byte stores). Mode 2: the decode KEPT, the eight scattered
// stores GONE (the values are folded into a register accumulator). The floor of the store path. Mode 3: the shipped
// store PLUS the accumulator -- the calibration for the accumulator's own cost. Mode 4: the codebook arithmetic
// dropped (the extraction and the stores kept) -- the value path's cost. Mode 5: the eight stores KEPT, the
// EXTRACTION dropped -- the extraction's cost. Modes 2-5 are diagnostics: their outputs differ by construction.
kernel void tf_exl3_matmul_prefill_nax69(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, true, 0, 3>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax70(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, true, 0, 1>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax71(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, true, 0, 4>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax72(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, true, true, false, false, 1, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax73(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, true, 0, 2>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The routed 64x64 prompt tile with the funnel path as the only path . The cursor arrays the Fast path needs -- off,
// sh and rel, twenty-four registers -- are dead in the funnel path but live across the k loop, so the shipped body
// carries them for nothing. Here they are not computed. The values, their order, the k order and the MMA order are
// 's, so the output is bit-identical.
kernel void tf_exl3_matmul_prefill_nax59(const device half* x [[buffer(0)]],
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
  // The funnel path is the one the routed pack's width (halves = 6) takes; the expert's own descriptor says so, so
  // the kernel picks the specialised body per expert and falls back to the shipped one otherwise.
  const uint rb = tg.y * 64u;
  uint e = 0u;
  if (a.indexed != 1u and rb < a.rows) e = routes[rb];
  if (a.experts != 0u and e >= a.experts) e = 0u;
  const uint halves = (a.indexed != 0u) ? descriptors[e].halves : a.halves;
  if (halves == 6u) {
    tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
  } else {
    tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
  }
}

// THE EXPERT'S SLAB DECODED ONCE FOR THE TILE'S ROWS . The values, their order, the k order and the MMA order are the
// shipped body's, so the output is bit-identical.
kernel void tf_exl3_matmul_prefill_nax76(const device half* x [[buffer(0)]],
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
  const uint rb = tg.y * 128u;
  uint e = 0u;
  if (a.indexed != 1u and rb < a.rows) e = routes[rb];
  if (a.experts != 0u and e >= a.experts) e = 0u;
  const uint halves = (a.indexed != 0u) ? descriptors[e].halves : a.halves;
  if (halves == 6u) {
    tf_exl3_prefill_body<true, 8, 1, false, true, false, false, true, false, false, 2, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
  } else {
    tf_exl3_prefill_body<true, 8, 1, false, true, false, false, true, false, false, 2, false, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
  }
}

// The same form one step further -- TWO HUNDRED AND FIFTY-SIX ROWS TALL with FOUR ROW GROUPS A WARP . Sixteen
// accumulator fragments a lane: the register wall is the price of the form.
kernel void tf_exl3_matmul_prefill_nax77(const device half* x [[buffer(0)]],
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
  const uint rb = tg.y * 256u;
  uint e = 0u;
  if (a.indexed != 1u and rb < a.rows) e = routes[rb];
  if (a.experts != 0u and e >= a.experts) e = 0u;
  const uint halves = (a.indexed != 0u) ? descriptors[e].halves : a.halves;
  if (halves == 6u) {
    tf_exl3_prefill_body<true, 16, 1, false, true, false, false, true, false, false, 4, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
  } else {
    tf_exl3_prefill_body<true, 16, 1, false, true, false, false, true, false, false, 4, false, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
  }
}

// The routed 128-row tile (eight row groups, eight warps). A 128-row tile halves that (the plan's own bound: 384 +
// (21,714 - 384) / 128 tiles), and the weight decode is per tile, so the tile's decode cost falls with it. The eight
// warps make a lane own four states instead of eight, so the funnel reads a twenty-five-bit run (four windows)
// instead of a thirty-seven-bit one; the values, their order, the k order and the per-row arithmetic are the shipped
// body's, so the reply is bit-identical.
kernel void tf_exl3_matmul_prefill_nax49(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 8, 1, false, true, false, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The routed 64 x 64 prompt tile with the funnel read and the extraction in thirty-two-bit words (the funnel's own
// dig-job structure). The windows, the values, their order and the k order are the shipped funnel's, so the output is
// bit-identical.
kernel void tf_exl3_matmul_prefill_nax50(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The routed 64 x 64 prompt tile with the funnel read and the block's words delivered by a cooperative load and a
// shuffle (one lane a word loads the block's twenty-four words; the two a lane needs arrive by a shuffle). The words,
// the windows, the values and the order are the shipped funnel's, so the output is bit-identical.
kernel void tf_exl3_matmul_prefill_nax52(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, false, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The routed 64 x 64 prompt tile with the funnel read, the grouping's padded rows skipped, and the block's words
// delivered by a cooperative load and a shuffle (one lane a word loads the block's twenty-four words; the two a lane
// needs arrive by a shuffle). The words, the windows, the values and the order are the shipped tile's, so the output
// is bit-identical.
kernel void tf_exl3_matmul_prefill_nax53(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The routed 128-row tile with a warp that does thirty-two rows (the NAXTile shape: a simdgroup owns thirty-two rows,
// so the block is decoded by one warp and the lane keeps eight states and eight bits a weight). The tile covers twice
// the rows, so the grouped plan spends half the tiles, and the weight fragments are read once for two row groups. The
// values, their order, the k order and the MMA order per output are the shipped tile's, so the output is
// bit-identical.
kernel void tf_exl3_matmul_prefill_nax54(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 8, 1, false, true, false, false, true, false, false, 2>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The routed 64 x 128 tile on four warps ('s ); 128 threads, one 16 x 128 threadgroup tile.
kernel void tf_exl3_matmul_prefill_nax37(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body_wide<true>(wx, x, trellis, sums, a, routes, descriptors, lane, sg, tid, tptg.x, tg);
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
  tf_exl3_prefill_body<true, 16, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The routed prompt tile, double-buffered: two 16x128 tiles (8 KB) and one barrier a k block.
kernel void tf_exl3_matmul_prefill_nax7(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 128];
  tf_exl3_prefill_body_db<4, 2, true, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

// Diagnostics: is the same tile with the per-block decode removed (the prologue tile reused), keeps the decode but
// issues one of the four multiplies. - isolates the decode, isolates the multiply.
kernel void tf_exl3_matmul_prefill_nax8(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 128];
  tf_exl3_prefill_body_db<4, 2, false, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax9(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 128];
  tf_exl3_prefill_body_db<4, 2, true, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

// The routed tile with the codebook as a resident table (buffer 7): the mul1 state maps to a half by one load.
kernel void tf_exl3_matmul_prefill_nax10(const device half* const x [[buffer(0)]],
                                        const device uchar* const trellis [[buffer(1)]],
                                        device float* const sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device uint* const routes [[buffer(4)]],
                                        const device TfExl3Expert* const descriptors [[buffer(5)]],
                                        const device half* const table [[buffer(7)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tptg [[threads_per_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 16 * 128];
  tf_exl3_prefill_body_db<4, 2, true, 4, true>(wx, x, trellis, sums, a, routes, descriptors, table, lane, sg, tid, tptg.x, tg);
}

// Warp-specialised routed tile: eight warps multiply the current block, eight decode the next into the other buffer,
// so the tensor units and the codebook ALU are in flight on different warps instead of interleaved in one.
template <uint CG>
inline void tf_exl3_prefill_body_ws(threadgroup half* wx, const device half* x, const device uchar* trellis,
                                    device float* sums, constant TfExl3Args& a, const device uint* routes,
                                    const device TfExl3Expert* descriptors, const device half* table,
                                    uint lane, uint sg0, uint tid, uint3 tg) {
  const uint rb = tg.y * 64u, nb = tg.x * (64u * CG);
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
  const uint rows_here = min(64u, a.rows - rb);
  const uint cols_here = min(64u * CG, a.n - nb);
  constexpr uint mma_warps = 4u * CG;
  const bool is_decode = tid >= mma_warps * 32u;
  const uint ws = tid / 32u;
  // MMA role: four row groups by CG column groups.
  const uint sgr = ws % 4u, sgc = ws / 4u;
  // Decode role: four column groups by CG sixty-four-column halves.
  const uint dg = ws - mma_warps, h = dg % 4u, hg = (dg / 4u) * 64u;
  const bool decodes = is_decode && (nb + hg + h * 16u < a.n);
  const uint p0 = lane * 8u;
  uint at[8], off[8], sh[8], rel[8];
  const uint words = 4 * halves;
  const uint wi0 = tf_exl3_map_for(p0, halves) & 63u;
  {
    const uint ring = 128 * halves;
    const uint cur = (wi0 << 5) | ((tf_exl3_map_for(p0, halves) >> 6) & 31u);
    TF_UNROLL
    for (uint k = 0; k < 8; ++k) {
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
  frag<float> lo0 = frag<float>(0), hi0 = frag<float>(0), lo1 = frag<float>(0), hi1 = frag<float>(0);
  if (decodes) {
    const uint w0 = tf_exl3_block_word(data, ks, ns, 0, nb, hg, h, wi0, words);
    const uint w1 = tf_exl3_block_word(data, ks, ns, 0, nb, hg, h, wi0 + 1, words);
    const uint w2 = tf_exl3_block_word(data, ks, ns, 0, nb, hg, h, wi0 + 2, words);
    const uint w3 = tf_exl3_block_word(data, ks, ns, 0, nb, hg, h, wi0 + 3, words);
    if (table != nullptr) tf_exl3_decode_table<8>(wx, table, codebook, at, off, sh, rel, w0, w1, w2, w3);
    else tf_exl3_decode_fast<8>(wx, codebook, at, off, sh, rel, w0, w1, w2, w3);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint cur_buf = 0;
  for (uint kb = 0; kb < a.k; kb += 16) {
    const uint nkb = kb + 16;
    if (!is_decode) {
      frag<half> av;
      tf_frag_get_dev4(av, (const device half*)x, int(a.k), int(rb + sgr * 16), int(kb), home, int(rb + rows_here));
      threadgroup half* cur = wx + cur_buf * (16u * 64u * CG);
      frag<half> b0, b1;
      const int wc = int(sgc * 64u);
      tf_frag_get_tg4(b0, cur, int(64u * CG), 0, wc, home);
      tf_frag_get_tg4(b1, cur, int(64u * CG), 0, wc + 16, home);
      mma_16x32<false, false>(lo0, hi0, av, b0, b1);
      tf_frag_get_tg4(b0, cur, int(64u * CG), 0, wc + 32, home);
      tf_frag_get_tg4(b1, cur, int(64u * CG), 0, wc + 48, home);
      mma_16x32<false, false>(lo1, hi1, av, b0, b1);
    } else if (nkb < a.k and decodes) {
      const uint nw0 = tf_exl3_block_word(data, ks, ns, nkb, nb, hg, h, wi0, words);
      const uint nw1 = tf_exl3_block_word(data, ks, ns, nkb, nb, hg, h, wi0 + 1, words);
      const uint nw2 = tf_exl3_block_word(data, ks, ns, nkb, nb, hg, h, wi0 + 2, words);
      const uint nw3 = tf_exl3_block_word(data, ks, ns, nkb, nb, hg, h, wi0 + 3, words);
      threadgroup half* nxt = wx + (cur_buf ^ 1u) * (16u * 64u * CG);
      if (table != nullptr) tf_exl3_decode_table<8>(nxt, table, codebook, at, off, sh, rel, nw0, nw1, nw2, nw3);
      else tf_exl3_decode_fast<8>(nxt, codebook, at, off, sh, rel, nw0, nw1, nw2, nw3);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    cur_buf ^= 1u;
  }
  if (!is_decode) {
    const int r0 = int(rb + sgr * 16);
    const int c0 = int(nb + sgc * 64u);
    frag_put_in(lo0, sums, int(a.n), r0, c0, home, int(rb + rows_here), int(nb + cols_here));
    frag_put_in(hi0, sums, int(a.n), r0, c0 + 16, home, int(rb + rows_here), int(nb + cols_here));
    frag_put_in(lo1, sums, int(a.n), r0, c0 + 32, home, int(rb + rows_here), int(nb + cols_here));
    frag_put_in(hi1, sums, int(a.n), r0, c0 + 48, home, int(rb + rows_here), int(nb + cols_here));
  }
}

kernel void tf_exl3_matmul_prefill_nax11(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 128];
  tf_exl3_prefill_body_ws<2>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tg);
}

kernel void tf_exl3_matmul_prefill_nax12(const device half* const x [[buffer(0)]],
                                        const device uchar* const trellis [[buffer(1)]],
                                        device float* const sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device uint* const routes [[buffer(4)]],
                                        const device TfExl3Expert* const descriptors [[buffer(5)]],
                                        const device half* const table [[buffer(7)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tptg [[threads_per_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 16 * 128];
  tf_exl3_prefill_body_ws<2>(wx, x, trellis, sums, a, routes, descriptors, table, lane, sg, tid, tg);
}

// Wider column tiles: CG=4 re-reads the activation tile nine times instead of eighteen, CG=8 five times.
kernel void tf_exl3_matmul_prefill_nax13(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 256];
  tf_exl3_prefill_body_db<4, 4, true, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax14(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 512];
  tf_exl3_prefill_body_db<4, 8, true, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax15(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 256];
  tf_exl3_prefill_body_ws<4>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tg);
}

kernel void tf_exl3_matmul_prefill_nax16(const device half* const x [[buffer(0)]],
                                        const device uchar* const trellis [[buffer(1)]],
                                        device float* const sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device uint* const routes [[buffer(4)]],
                                        const device TfExl3Expert* const descriptors [[buffer(5)]],
                                        const device half* const table [[buffer(7)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tptg [[threads_per_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 16 * 128];
  tf_exl3_prefill_body_db<4, 2, true, 0, true>(wx, x, trellis, sums, a, routes, descriptors, table, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax17(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 128];
  tf_exl3_prefill_body_db<4, 2, true, 0, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

// The 64x64 grouped tile the engine actually selects ('s tile): table decode and double buffer.
kernel void tf_exl3_matmul_prefill_nax18(const device half* const x [[buffer(0)]],
                                        const device uchar* const trellis [[buffer(1)]],
                                        device float* const sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device uint* const routes [[buffer(4)]],
                                        const device TfExl3Expert* const descriptors [[buffer(5)]],
                                        const device half* const table [[buffer(7)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tptg [[threads_per_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_db<4, 1, true, 4, true>(wx, x, trellis, sums, a, routes, descriptors, table, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax19(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_db<4, 1, true, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax20(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_db<4, 1, false, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax21(const device half* const x [[buffer(0)]],
                                        const device uchar* const trellis [[buffer(1)]],
                                        device float* const sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device uint* const routes [[buffer(4)]],
                                        const device TfExl3Expert* const descriptors [[buffer(5)]],
                                        const device half* const table [[buffer(7)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tptg [[threads_per_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_db<4, 1, true, 0, true>(wx, x, trellis, sums, a, routes, descriptors, table, lane, sg, tid, tptg.x, tg);
}

// The engine's 64x64 grouped tile with the codebook table and the shipped single buffer: the table gain without the
// double buffer's extra 2 KB of threadgroup memory (which cost more occupancy than it returned).
kernel void tf_exl3_matmul_prefill_nax22(const device half* const x [[buffer(0)]],
                                        const device uchar* const trellis [[buffer(1)]],
                                        device float* const sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device uint* const routes [[buffer(4)]],
                                        const device TfExl3Expert* const descriptors [[buffer(5)]],
                                        const device half* const table [[buffer(7)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tptg [[threads_per_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[16 * 64];
  tf_exl3_prefill_body<true, 4, 1, true>(wx, x, trellis, sums, a, routes, descriptors, table, nullptr, lane, sg, tid, tptg.x, tg);
}

// Activation-fragment reuse: four warps (one a row group), each decodes two 16x64 weight tiles into one 16x128
// buffer, loads its `av` fragment once, and multiplies both tiles.
template <bool Table>
inline void tf_exl3_prefill_body_reuse(threadgroup half* wx, const device half* x, const device uchar* trellis,
                                       device float* sums, constant TfExl3Args& a, const device uint* routes,
                                       const device TfExl3Expert* descriptors, const device half* table,
                                       uint lane, uint sg, uint tid, uint3 tg) {
  const uint rb = tg.y * 64u, nb = tg.x * 128u;
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
  const uint rows_here = min(64u, a.rows - rb);
  const uint cols_here = min(128u, a.n - nb);
  const uint h = sg; // four warps, one a row group
  const uint p0 = lane * 8u;
  const uint words = 4 * halves;
  const uint wi0 = tf_exl3_map_for(p0, halves) & 63u;
  uint atA[8], atB[8], off[8], sh[8], rel[8];
  {
    const uint ring = 128 * halves;
    const uint cur = (wi0 << 5) | ((tf_exl3_map_for(p0, halves) >> 6) & 31u);
    TF_UNROLL
    for (uint k = 0; k < 8; ++k) {
      const uint p = p0 + k, l = p / 8, e = p % 8;
      const uint at = (2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u)) * 128u + h * 16u + (l / 4 + 8 * (e >> 2));
      atA[k] = at;
      atB[k] = at + 64u;
      uint c8 = cur + k * (halves >> 1);
      if (c8 >= ring) c8 -= ring;
      const uint wi = c8 >> 5;
      off[k] = c8 & 31u;
      sh[k] = (32u - off[k]) & 31u;
      rel[k] = wi >= wi0 ? wi - wi0 : wi + words - wi0;
    }
  }
  frag<float> lo0 = frag<float>(0), hi0 = frag<float>(0), lo1 = frag<float>(0), hi1 = frag<float>(0);
  frag<float> lo2 = frag<float>(0), hi2 = frag<float>(0), lo3 = frag<float>(0), hi3 = frag<float>(0);
  for (uint kb = 0; kb < a.k; kb += 16) {
    const uint nA0 = tf_exl3_block_word(data, ks, ns, kb, nb, 0, h, wi0, words);
    const uint nA1 = tf_exl3_block_word(data, ks, ns, kb, nb, 0, h, wi0 + 1, words);
    const uint nA2 = tf_exl3_block_word(data, ks, ns, kb, nb, 0, h, wi0 + 2, words);
    const uint nA3 = tf_exl3_block_word(data, ks, ns, kb, nb, 0, h, wi0 + 3, words);
    const uint nB0 = tf_exl3_block_word(data, ks, ns, kb, nb + 64u, 0, h, wi0, words);
    const uint nB1 = tf_exl3_block_word(data, ks, ns, kb, nb + 64u, 0, h, wi0 + 1, words);
    const uint nB2 = tf_exl3_block_word(data, ks, ns, kb, nb + 64u, 0, h, wi0 + 2, words);
    const uint nB3 = tf_exl3_block_word(data, ks, ns, kb, nb + 64u, 0, h, wi0 + 3, words);
    if (Table) {
      tf_exl3_decode_table<8>(wx, table, codebook, atA, off, sh, rel, nA0, nA1, nA2, nA3);
      tf_exl3_decode_table<8>(wx, table, codebook, atB, off, sh, rel, nB0, nB1, nB2, nB3);
    } else {
      tf_exl3_decode_fast<8>(wx, codebook, atA, off, sh, rel, nA0, nA1, nA2, nA3);
      tf_exl3_decode_fast<8>(wx, codebook, atB, off, sh, rel, nB0, nB1, nB2, nB3);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    frag<half> av;
    frag_get_in(av, (const device half*)x, int(a.k), int(rb + h * 16), int(kb), home, int(rb + rows_here), int(kb) + 16);
    frag<half> b0, b1;
    tf_frag_get_tg4(b0, wx, 128, 0, 0, home);
    tf_frag_get_tg4(b1, wx, 128, 0, 16, home);
    mma_16x32<false, false>(lo0, hi0, av, b0, b1);
    tf_frag_get_tg4(b0, wx, 128, 0, 32, home);
    tf_frag_get_tg4(b1, wx, 128, 0, 48, home);
    mma_16x32<false, false>(lo1, hi1, av, b0, b1);
    tf_frag_get_tg4(b0, wx, 128, 0, 64, home);
    tf_frag_get_tg4(b1, wx, 128, 0, 80, home);
    mma_16x32<false, false>(lo2, hi2, av, b0, b1);
    tf_frag_get_tg4(b0, wx, 128, 0, 96, home);
    tf_frag_get_tg4(b1, wx, 128, 0, 112, home);
    mma_16x32<false, false>(lo3, hi3, av, b0, b1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const int r0 = int(rb + h * 16);
  const int c0 = int(nb);
  frag_put_in(lo0, sums, int(a.n), r0, c0, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi0, sums, int(a.n), r0, c0 + 16, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo1, sums, int(a.n), r0, c0 + 32, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi1, sums, int(a.n), r0, c0 + 48, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo2, sums, int(a.n), r0, c0 + 64, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi2, sums, int(a.n), r0, c0 + 80, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo3, sums, int(a.n), r0, c0 + 96, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi3, sums, int(a.n), r0, c0 + 112, home, int(rb + rows_here), int(nb + cols_here));
}

kernel void tf_exl3_matmul_prefill_nax23(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[16 * 128];
  tf_exl3_prefill_body_reuse<false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tg);
}

kernel void tf_exl3_matmul_prefill_nax24(const device half* const x [[buffer(0)]],
                                        const device uchar* const trellis [[buffer(1)]],
                                        device float* const sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device uint* const routes [[buffer(4)]],
                                        const device TfExl3Expert* const descriptors [[buffer(5)]],
                                        const device half* const table [[buffer(7)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tptg [[threads_per_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[16 * 128];
  tf_exl3_prefill_body_reuse<true>(wx, x, trellis, sums, a, routes, descriptors, table, lane, sg, tid, tg);
}

// The 64x64 grouped tile with the mul1 byte-sum LUT built in threadgroup memory (1024 halves, 2 KB): the decode
// shape, no 384 KB table, no per-weight L2 miss.
kernel void tf_exl3_matmul_prefill_nax25(const device half* const x [[buffer(0)]],
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
  threadgroup half lut[1024];
  const half scale = as_type<half>(ushort(0x1EEEu));
  const half bias = as_type<half>(ushort(0xC931u));
  TF_UNROLL
  for (uint j = 0; j < 8u; ++j) lut[tid + 128u * j] = half(fma(float(tid + 128u * j + 1024u), float(scale), float(bias)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tf_exl3_prefill_body<true, 4, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lut, lane, sg, tid, tptg.x, tg);
}

// Dense 128x64 tile ('s shape): double-buffered + codebook table, and single-buffer + threadgroup LUT.
kernel void tf_exl3_matmul_prefill_nax26(const device half* const x [[buffer(0)]],
                                        const device uchar* const trellis [[buffer(1)]],
                                        device float* const sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device uint* const routes [[buffer(4)]],
                                        const device TfExl3Expert* const descriptors [[buffer(5)]],
                                        const device half* const table [[buffer(7)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tptg [[threads_per_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_db<8, 1, true, 4, true>(wx, x, trellis, sums, a, routes, descriptors, table, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax27(const device half* const x [[buffer(0)]],
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
  threadgroup half lut[1024];
  const half scale = as_type<half>(ushort(0x1EEEu));
  const half bias = as_type<half>(ushort(0xC931u));
  TF_UNROLL
  for (uint j = 0; j < 4u; ++j) lut[tid + 256u * j] = half(fma(float(tid + 256u * j + 1024u), float(scale), float(bias)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tf_exl3_prefill_body<true, 8, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lut, lane, sg, tid, tptg.x, tg);
}

// Dense 128x128 tile: sixteen warps, the tile shape, with the 1024-entry LUT amortised over twice the decode (2
// entries built a thread, 4 states decoded) and half the activation re-read.
kernel void tf_exl3_matmul_prefill_nax28(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[16 * 128];
  threadgroup half lut[1024];
  const half scale = as_type<half>(ushort(0x1EEEu));
  const half bias = as_type<half>(ushort(0xC931u));
  TF_UNROLL
  for (uint j = 0; j < 2u; ++j) lut[tid + 512u * j] = half(fma(float(tid + 512u * j + 1024u), float(scale), float(bias)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tf_exl3_prefill_body<true, 8, 2, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lut, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax29(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[16 * 128];
  tf_exl3_prefill_body<true, 8, 2, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// Diagnostics for the dense 128x64 tile (RG=8, 256 threads), the attention-dense body: the same body with the
// per-block decode removed (the prologue tile reused), and the same body that keeps it but issues one of the four
// multiplies. They isolate the decode at the dense tile and the decode-free multiply.
kernel void tf_exl3_matmul_prefill_nax30(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_db<8, 1, false, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax31(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_db<8, 1, true, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

// THE DOUBLE-BUFFERED 128-COLUMN DENSE BODY. 's decode, k order and MMA order per output, at twice the tile width:
// one activation block is read for 128 columns instead of 64 and one barrier covers twice the multiply. reads the
// codebook from the resident device table (buffer 7); is the same body on the ALU codebook. Both must agree with
// bit-for-bit at every output.
kernel void tf_exl3_matmul_prefill_nax79(const device half* const x [[buffer(0)]],
                                        const device uchar* const trellis [[buffer(1)]],
                                        device float* const sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device uint* const routes [[buffer(4)]],
                                        const device TfExl3Expert* const descriptors [[buffer(5)]],
                                        const device half* const table [[buffer(7)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tptg [[threads_per_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 16 * 128];
  tf_exl3_prefill_body_db<8, 2, true, 4, true>(wx, x, trellis, sums, a, routes, descriptors, table, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax80(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 128];
  tf_exl3_prefill_body_db<8, 2, true, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

// The 64-row x 128-column double-buffered dense body: the same 8,192-element tile area as, eight warps / 256 threads
// ('s occupancy), eight decode groups so one threadgroup owns all 128 columns the fused output transform needs. 's k
// order and MMA order per output.
kernel void tf_exl3_matmul_prefill_nax82(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 128];
  tf_exl3_prefill_body_db<4, 2, true, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

// ONE 8-WARP THREADGROUP OVER 128 ROWS AND BOTH 64-COLUMN HALVES. Two accumulator sets (eight fragments, 64 f32 a
// lane) and two double-buffered 16x64 tiles; the activation block is loaded once and multiplied into both halves. Per
// output this is 's weight-decode amortisation (128 rows) and 's eight-warp threadgroup, with half its activation
// reads. 's k order and MMA order per output, so `sums` is bit-identical.
template <bool UseTable>
inline void tf_exl3_prefill_body_db2(threadgroup half* wx, const device half* x, const device uchar* trellis,
                                     device float* sums, constant TfExl3Args& a, const device uint* routes,
                                     const device TfExl3Expert* descriptors, const device half* table,
                                     uint lane, uint sg, uint tid, uint tptg, uint3 tg) {
  const uint trows = 128u;
  const uint rb = tg.y * trows, nb = tg.x * 128u;
  if (a.indexed == 1 || rb >= a.rows || nb >= a.n) return;
  const device uchar* data = trellis;
  const ulong ks = a.tile_k_stride, ns = a.tile_n_stride;
  const uint halves = a.halves, codebook = a.codebook;
  if ((halves & 1u) != 0 or halves > 16) return;
  const short2 home = frag_home(ushort(lane));
  const uint rows_here = min(trows, a.rows - rb);
  const uint cols_here = min(128u, a.n - nb);
  constexpr uint wpg = 2u;
  constexpr uint states = 4u;
  const uint gwarp = tid / 32u, group = gwarp / wpg, wg = gwarp % wpg;
  const uint h = group % 4u;
  const uint p0 = (wg * 32u + tid % 32u) * states;
  const bool decodes = group < 4u and nb + 64u + h * 16u < a.n;
  uint at[8], off[8], sh[8], rel[8];
  const uint words = 4 * halves;
  const uint wi0 = tf_exl3_map_for(p0, halves) & 63u;
  {
    const uint ring = 128 * halves;
    const uint cur = (wi0 << 5) | ((tf_exl3_map_for(p0, halves) >> 6) & 31u);
    TF_UNROLL
    for (uint k = 0; k < states; ++k) {
      const uint p = p0 + k, l = p / 8, e = p % 8;
      at[k] = (2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u)) * 64u + h * 16u + (l / 4 + 8 * (e >> 2));
      uint c8 = cur + k * (halves >> 1);
      if (c8 >= ring) c8 -= ring;
      const uint wi = c8 >> 5;
      off[k] = c8 & 31u;
      sh[k] = (32u - off[k]) & 31u;
      rel[k] = wi >= wi0 ? wi - wi0 : wi + words - wi0;
    }
  }
  const uint sgr = sg;
  frag<float> lo0 = frag<float>(0), hi0 = frag<float>(0), lo1 = frag<float>(0), hi1 = frag<float>(0);
  frag<float> lo2 = frag<float>(0), hi2 = frag<float>(0), lo3 = frag<float>(0), hi3 = frag<float>(0);
  if (decodes) {
    uint w0 = tf_exl3_block_word(data, ks, ns, 0, nb, 0, h, wi0, words);
    uint w1 = tf_exl3_block_word(data, ks, ns, 0, nb, 0, h, wi0 + 1, words);
    uint w2 = tf_exl3_block_word(data, ks, ns, 0, nb, 0, h, wi0 + 2, words);
    uint w3 = tf_exl3_block_word(data, ks, ns, 0, nb, 0, h, wi0 + 3, words);
    if (UseTable) tf_exl3_decode_table<states>(wx, table, codebook, at, off, sh, rel, w0, w1, w2, w3);
    else tf_exl3_decode_fast<states>(wx, codebook, at, off, sh, rel, w0, w1, w2, w3);
    w0 = tf_exl3_block_word(data, ks, ns, 0, nb + 64u, 0, h, wi0, words);
    w1 = tf_exl3_block_word(data, ks, ns, 0, nb + 64u, 0, h, wi0 + 1, words);
    w2 = tf_exl3_block_word(data, ks, ns, 0, nb + 64u, 0, h, wi0 + 2, words);
    w3 = tf_exl3_block_word(data, ks, ns, 0, nb + 64u, 0, h, wi0 + 3, words);
    if (UseTable) tf_exl3_decode_table<states>(wx + 2u * (16u * 64u), table, codebook, at, off, sh, rel, w0, w1, w2, w3);
    else tf_exl3_decode_fast<states>(wx + 2u * (16u * 64u), codebook, at, off, sh, rel, w0, w1, w2, w3);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint cur0 = 0u, cur1 = 2u;
  for (uint kb = 0; kb < a.k; kb += 16) {
    const uint nkb = kb + 16;
    uint nw0 = 0, nw1 = 0, nw2 = 0, nw3 = 0, nw4 = 0, nw5 = 0, nw6 = 0, nw7 = 0;
    if (nkb < a.k and decodes) {
      nw0 = tf_exl3_block_word(data, ks, ns, nkb, nb, 0, h, wi0, words);
      nw1 = tf_exl3_block_word(data, ks, ns, nkb, nb, 0, h, wi0 + 1, words);
      nw2 = tf_exl3_block_word(data, ks, ns, nkb, nb, 0, h, wi0 + 2, words);
      nw3 = tf_exl3_block_word(data, ks, ns, nkb, nb, 0, h, wi0 + 3, words);
      nw4 = tf_exl3_block_word(data, ks, ns, nkb, nb + 64u, 0, h, wi0, words);
      nw5 = tf_exl3_block_word(data, ks, ns, nkb, nb + 64u, 0, h, wi0 + 1, words);
      nw6 = tf_exl3_block_word(data, ks, ns, nkb, nb + 64u, 0, h, wi0 + 2, words);
      nw7 = tf_exl3_block_word(data, ks, ns, nkb, nb + 64u, 0, h, wi0 + 3, words);
    }
    frag<half> av;
    tf_frag_get_dev4(av, (const device half*)x, int(a.k), int(rb + sgr * 16u), int(kb), home, int(rb + rows_here));
    {
      const threadgroup half* cur = wx + cur0 * (16u * 64u);
      frag<half> b0f, b1f;
      tf_frag_get_tg4(b0f, cur, 64, 0, 0, home);
      tf_frag_get_tg4(b1f, cur, 64, 0, 16, home);
      mma_16x32<false, false>(lo0, hi0, av, b0f, b1f);
      tf_frag_get_tg4(b0f, cur, 64, 0, 32, home);
      tf_frag_get_tg4(b1f, cur, 64, 0, 48, home);
      mma_16x32<false, false>(lo1, hi1, av, b0f, b1f);
    }
    {
      const threadgroup half* cur = wx + cur1 * (16u * 64u);
      frag<half> b0f, b1f;
      tf_frag_get_tg4(b0f, cur, 64, 0, 0, home);
      tf_frag_get_tg4(b1f, cur, 64, 0, 16, home);
      mma_16x32<false, false>(lo2, hi2, av, b0f, b1f);
      tf_frag_get_tg4(b0f, cur, 64, 0, 32, home);
      tf_frag_get_tg4(b1f, cur, 64, 0, 48, home);
      mma_16x32<false, false>(lo3, hi3, av, b0f, b1f);
    }
    if (nkb < a.k and decodes) {
      if (UseTable) {
        tf_exl3_decode_table<states>(wx + (cur0 ^ 1u) * (16u * 64u), table, codebook, at, off, sh, rel, nw0, nw1, nw2, nw3);
        tf_exl3_decode_table<states>(wx + (cur1 ^ 1u) * (16u * 64u), table, codebook, at, off, sh, rel, nw4, nw5, nw6, nw7);
      } else {
        tf_exl3_decode_fast<states>(wx + (cur0 ^ 1u) * (16u * 64u), codebook, at, off, sh, rel, nw0, nw1, nw2, nw3);
        tf_exl3_decode_fast<states>(wx + (cur1 ^ 1u) * (16u * 64u), codebook, at, off, sh, rel, nw4, nw5, nw6, nw7);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    cur0 ^= 1u;
    cur1 ^= 1u;
  }
  const int r0 = int(rb + sgr * 16u);
  const int c0 = int(nb), c1 = int(nb + 64u);
  frag_put_in(lo0, sums, int(a.n), r0, c0, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi0, sums, int(a.n), r0, c0 + 16, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo1, sums, int(a.n), r0, c0 + 32, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi1, sums, int(a.n), r0, c0 + 48, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo2, sums, int(a.n), r0, c1, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi2, sums, int(a.n), r0, c1 + 16, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo3, sums, int(a.n), r0, c1 + 32, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi3, sums, int(a.n), r0, c1 + 48, home, int(rb + rows_here), int(nb + cols_here));
}

kernel void tf_exl3_matmul_prefill_nax83(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[4 * 16 * 64];
  tf_exl3_prefill_body_db2<false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

// The dense 128x64 prompt tile ('s geometry) with the funnel read. The dense tile is 128 rows (eight row groups), so
// a lane owns four states and the funnel reads a twenty-five-bit run -- two words, where the cursor path reads four.
// The values, their order, the k order and the MMA order are 's, so the output is bit-identical; only the word
// traffic and the select chain leave.
kernel void tf_exl3_matmul_prefill_nax51(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_db<8, 1, true, 4, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

// The dense 128x64 prompt tile with a SIMDgroup that owns thirty-two rows -- the wide form's NAXTile shape (a
// simdgroup owns thirty-two rows, so one threadgroup's eight row groups become four groups of two). The lane keeps
// eight states, so the stream words it reads cover twice the states, and the weight fragments are read once for two
// row groups, so the tile's threadgroup traffic per row halves. The values, their order, the k order and the MMA
// order per output are 's, so the output is bit-identical.
kernel void tf_exl3_matmul_prefill_nax56(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_db<8, 1, true, 4, false, false, 2>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax32(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_db<8, 1, true, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

// Dense row-height diagnostics: RG=16 (256 rows a threadgroup, 512 threads) and RG=32 (512 rows, 1024 threads). The
// decoded 16x64 weight tile is shared by every row group, so the per-weight decode halves at RG=16 and quarters at
// RG=32; the per-row MMA order is unchanged (each SIMDgroup still owns its sixteen rows).
kernel void tf_exl3_matmul_prefill_nax33(const device half* const x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 16, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax34(const device half* const x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 32, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// Two-blocks-ahead prompt bodies: the routed 64x64 tile and the dense 128x64 tile .
kernel void tf_exl3_matmul_prefill_nax35(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_pf<4, 1, 4>(wx, x, trellis, sums, a, routes, descriptors, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax36(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_pf<8, 1, 4>(wx, x, trellis, sums, a, routes, descriptors, lane, sg, tid, tptg.x, tg);
}

// The register-only decode structure on our map -- one threadgroup a sixteen-column tile, one k walk, no split-K, no
// partials and no reduce dispatch, and the decoded weights stay in REGISTERS: there is no threadgroup weight tile, no
// barrier in the k loop and no threadgroup memory round trip. One warp owns a sixteen-column tile of one row, each
// lane holds two output columns and the four k rows it reads, the eight decoded weights stay in registers, and the
// four lanes that share a column reduce once, at the end of the walk. TF_FMA_WARPS: the warps a threadgroup. Each
// warp walks its own slice of the k blocks and the warps' partial sums are added once, at the end of the walk,
// through a threadgroup reduction -- so a split-K dispatch's parallelism without its partials round trip and its
// reduce dispatch. Our map is unchanged: the lane's eight states are the same eight the tensor-unit body reads (the
// same stream words, the same bit offsets, the same codebook values), so the products are the same products. The
// reduction order is the FP32 walk's, not the tensor unit's: the sums are FP32 either way, but the tensor unit
// reduces sixteen k at a time and this body adds them one at a time, so the two are not bit-identical (the numerics
// band shows the difference).
#ifndef TF_FMA_WARPS
#define TF_FMA_WARPS 1
#endif
// THE PREFETCH DEPTH OF THE REGISTER-ONLY BODY'S K-WALK. The shipped body keeps ONE block of stream words in flight
// (the next block's three words are read at the top of the walk and consumed at the top of the next step), so a
// block's DRAM latency is exposed once a block unless the block's own work hides it -- and a block's work is eight
// states, far less than a DRAM round trip. TF_FMA_PF blocks in flight (a ring of word triples) hides it behind PF
// blocks of work. The values, their order and the arithmetic are untouched: the ring only moves the LOADS in time, so
// every arm is BIT-IDENTICAL to the shipped body.
#ifndef TF_FMA_PF
#define TF_FMA_PF 1
#endif
// THE STREAM WORD AS ONE 4-BYTE LOAD (the read, 's form). The shipped read is two 2-byte loads a word (a stream word
// is a little-endian pair of int16s), so the decode's lane issues SIX load instructions a k-block where the wide form
// issues three; the bytes, the values and the order are the same, so the arm is BIT-IDENTICAL. The prefill tiles
// already read their words this way (tf_exl3_word_t<W> with W = true); the register-only decode body did not.
// TF_FMA_WIDE_WORD = 1 turns it on there.
#ifndef TF_FMA_WIDE_WORD
#define TF_FMA_WIDE_WORD 0
#endif

// THE TWO-WORD FUNNEL AT halves = 6 IN THE STATIC BODY. The static body reads three stream words a lane a block (a
// run of 7*S + 16 bits; at S = 3 that is 37 bits), but at halves = 6 every lane run starts at off0 in {3,11,19,27},
// so the run is contained in two 32-bit words and only the top five bits of the funnel second register are read
// (states J = 6, 7 at bit 18/21 use c_next >> 30 / >> 27). The third word feeds only the low off0 <= 27 bits of
// c_next, which the extraction never reads. Dropping it is bit-identical BY CONSTRUCTION (the same argument as routed
// funnel); The two forms are hash-identical.
#ifndef TF_FMA_TWO
#define TF_FMA_TWO 0
#endif
// THE VALUE PATH'S FORM. The alternative form is the ARITHMETIC one the format's own decoder uses (tf_exl3_value: the
// same byte sum, the same half, the same fma) -- the LUT's entries ARE those values, entry for entry, so the two
// forms are BIT-IDENTICAL and the choice is a pure cost question. TF_FMA_VALUE_ARITH = 1 takes the arithmetic form.
#ifndef TF_FMA_VALUE_ARITH
#define TF_FMA_VALUE_ARITH 0
#endif
kernel void tf_exl3_matmul_decode_fma(const device half* x [[buffer(0)]],
                                      const device uchar* trellis [[buffer(1)]],
                                      device float* sums [[buffer(2)]],
                                      constant TfExl3Args& a [[buffer(3)]],
                                      const device uint* routes [[buffer(4)]],
                                      const device TfExl3Expert* descriptors [[buffer(5)]],
                                      device float* partials [[buffer(6)]],
#if defined(TF_DECODE_TABLE)
                                      const device half* table [[buffer(7)]],
#endif
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint sg [[simdgroup_index_in_threadgroup]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint3 tptg [[threads_per_threadgroup]],
                                      uint3 tg [[threadgroup_position_in_grid]]) {
#if TF_FMA_WARPS > 1
  // The warps' partial sums: two columns a lane, one entry a warp a lane.
  threadgroup float red[TF_FMA_WARPS][64];
#endif
  const uint row = tg.y;
  const uint nb = tg.x * 16u;
  if (row >= a.rows) return;
  const uint expert = routes[row];
  if (expert >= a.experts) return;
  // The entry's own output width bounds this row's column tiles (0 = the call's n).
  if (nb >= (descriptors[expert].reserved != 0 ? descriptors[expert].reserved : a.n)) return;
  const device uchar* data = descriptors[expert].trellis;
  const ulong ks = descriptors[expert].tile_k_stride;
  const ulong ns = descriptors[expert].tile_n_stride;
  const uint halves = descriptors[expert].halves;
  const uint codebook = descriptors[expert].codebook;
  const uint blocks = a.k / 16;
  const uint W = TF_FMA_WARPS;
  const uint b0 = blocks * sg / W, b1 = blocks * (sg + 1u) / W;
  // The lane's eight states: the same cursor, word and bit offsets the tensor-unit body computes.
  uint off[8], sh[8], rel[8];
  const uint words = 4 * halves;
  const uint ring = 128 * halves;
  const uint wi0 = tf_exl3_map_for(lane * 8u, halves) & 63u;
#if defined(TF_DECODE_FUNNEL)
  // The routed pack's halves = 6: two 32-bit stream words cover the lane's eight states.
  const bool funnel = halves == 6u;
#else
  const bool funnel = false;
#endif
  {
    const uint cur = (wi0 << 5) | ((tf_exl3_map_for(lane * 8u, halves) >> 6) & 31u);
    TF_UNROLL
    for (uint k = 0; k < 8; ++k) {
      const uint c8 = (cur + k * (halves >> 1)) % ring;
      off[k] = c8 & 31u;
      sh[k] = (32u - off[k]) & 31u;
      rel[k] = (c8 >> 5) >= wi0 ? (c8 >> 5) - wi0 : (c8 >> 5) + words - wi0;
    }
  }
  uint a1 = wi0 + 1; if (a1 >= words) a1 -= words;
  uint a2 = wi0 + 2; if (a2 >= words) a2 -= words;
  uint a3 = wi0 + 3; if (a3 >= words) a3 -= words;
  // The words a block needs are loaded a block early, so their latency overlaps the multiply.
  uint pre[4];
  {
    const device ushort* t0 = reinterpret_cast<const device ushort*>(data + ulong(b0) * ks + ulong(nb / 16) * ns);
    pre[0] = tf_exl3_word(t0, wi0);
    pre[1] = tf_exl3_word(t0, a1);
    if (!funnel) {
      pre[2] = tf_exl3_word(t0, a2);
      pre[3] = tf_exl3_word(t0, a3);
    }
  }
  // The four-word pipeline read covers an even width up to sixteen halves; the general path reads the tile per state
  // (the tensor-unit body's own fallback) for any other width.
  const bool pipeline = (halves & 1u) == 0 and halves <= 16;
  // One accumulator a state, so the eight products of a block meet in a tree at the end instead of in two chains of
  // four. Eight accumulators make the two the same class.
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint bb = b0; bb < b1; ++bb) {
    const uint kb = bb * 16;
    const uint w0 = pre[0], w1 = pre[1];
    uint w2 = 0, w3 = 0;
    if (!funnel) {
      w2 = pre[2];
      w3 = pre[3];
    }
    if (bb + 1 < b1) {
      const device ushort* tn = reinterpret_cast<const device ushort*>(data + ulong(bb + 1) * ks + ulong(nb / 16) * ns);
      pre[0] = tf_exl3_word(tn, wi0);
      pre[1] = tf_exl3_word(tn, a1);
      if (!funnel) {
        pre[2] = tf_exl3_word(tn, a2);
        pre[3] = tf_exl3_word(tn, a3);
      }
    }
    // The lane's four k rows of this block, read as two halves each (the rows are adjacent in pairs).
#if defined(TF_DIAG_DECODE) && TF_DIAG_DECODE == 4
    // The FMA stays (the weight is still needed) and the constant is a RUNTIME value (the args' rows), so the
    // compiler cannot fold the multiply away and the arm isolates the read, not the arithmetic.
    const float x0 = float(a.rows), x1 = float(a.rows), x2 = float(a.rows), x3 = float(a.rows);
#else
    const device half* xr = x + ulong(row) * a.k + kb + 2 * (lane & 3u);
    const half2 xa = *(const device half2*)xr;
    const half2 xb = *(const device half2*)(xr + 8);
    const float x0 = float(xa.x), x1 = float(xa.y), x2 = float(xb.x), x3 = float(xb.y);
#endif
    if (pipeline) {
      TF_UNROLL
      for (uint k = 0; k < 8; ++k) {
        const uint lo8 = funnel ? (rel[k] == 0 ? w0 : w1) : (rel[k] == 0 ? w0 : rel[k] == 1 ? w1 : w2);
        const uint hi8 = funnel ? (rel[k] == 0 ? w1 : 0) : (rel[k] == 0 ? w1 : rel[k] == 1 ? w2 : w3);
        const uint carry = (hi8 >> sh[k]) & (0u - uint(off[k] != 0));
        const ushort s = ushort(((lo8 << off[k]) | carry) >> 16);
#if defined(TF_DECODE_TABLE)
        const float w = float(tf_exl3_table_value(table, s, codebook));
#else
        const float w = float(tf_exl3_value(s, codebook));
#endif
        // The state's row (four of them a lane) and its column (two of them a lane) in the tile.
        const uint odd = k & 1u, up = (k >> 1) & 1u;
        const float xv = odd == 0 ? (up == 0 ? x0 : x2) : (up == 0 ? x1 : x3);
        acc[k] = fma(xv, w, acc[k]);
      }
    } else {
      const device ushort* tile = reinterpret_cast<const device ushort*>(data + ulong(bb) * ks + ulong(nb / 16) * ns);
      TF_UNROLL
      for (uint k = 0; k < 8; ++k) {
        const ushort s = tf_exl3_mapped_state(tile, tf_exl3_map_for(lane * 8u + k, halves));
#if defined(TF_DECODE_TABLE)
        const float w = float(tf_exl3_table_value(table, s, codebook));
#else
        const float w = float(tf_exl3_value(s, codebook));
#endif
        const uint odd = k & 1u, up = (k >> 1) & 1u;
        const float xv = odd == 0 ? (up == 0 ? x0 : x2) : (up == 0 ? x1 : x3);
        acc[k] = fma(xv, w, acc[k]);
      }
    }
  }
  // The eight accumulators meet in a tree: the first four are the near column, the last four the far one.
  float acc0 = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  float acc1 = (acc[4] + acc[5]) + (acc[6] + acc[7]);
#if TF_FMA_WARPS > 1
  // The warps' partial sums meet once, here, and the four lanes that share a column meet once, here.
  red[sg][lane] = acc0;
  red[sg][32 + lane] = acc1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  acc0 = 0.0f;
  acc1 = 0.0f;
  for (uint w = 0; w < W; ++w) {
    acc0 += red[w][lane];
    acc1 += red[w][32 + lane];
  }
#endif
  acc0 += simd_shuffle_xor(acc0, 1);
  acc0 += simd_shuffle_xor(acc0, 2);
  acc1 += simd_shuffle_xor(acc1, 1);
  acc1 += simd_shuffle_xor(acc1, 2);
  if ((lane & 3u) == 0u) {
    sums[ulong(row) * a.n + nb + lane / 4] = acc0;
    sums[ulong(row) * a.n + nb + 8 + lane / 4] = acc1;
  }
}

// One state, one product: the state's window at a compile-time shift (the static selection) and the lane's activation
// row for that state. The state's value alone (the window at a compile-time shift, then the codebook's value). The
// decode is the ALU-bound part of the register-only body, so the multi-row body decodes it ONCE a block and gives the
// value to every row's product.
template <uint S, uint J, bool Lut = false>
inline float tf_exl3_fma_weight(uint c_cur, uint c_next, const device half* table, uint codebook,
                                const threadgroup half* lut = nullptr) {
  const ushort s = tf_exl3_state_static<S, J>(c_cur, c_next);
#if defined(TF_DIAG_DECODE) && (TF_DIAG_DECODE == 1 || TF_DIAG_DECODE == 2)
  // TF_DIAG_DECODE = 1 keeps the state's extraction and the trellis read and drops the VALUE path (the hash, the byte
  // sum, the index and the LUT load); = 2 drops the whole decode, so the compiler removes the trellis reads too.
  if (TF_DIAG_DECODE == 1) return float(half(float(s & 0xFFu)));
  return 1.0f;
#elif defined(TF_DIAG_DECODE) && (TF_DIAG_DECODE == 6 || TF_DIAG_DECODE == 7)
  // The ladder (TF_DIAG_DECODE = 1) isolates the whole value path (the hash, the byte sum, the index and the load) as
  // one term; this splits it. d6 KEEPS THE LOAD and drops the index arithmetic (the value is the LUT's entry at a
  // state-derived index -- the same random access pattern, none of the arithmetic); d7 KEEPS THE INDEX ARITHMETIC and
  // drops the load (the value is the index's low bit). Both arms' outputs differ from the body's by construction --
  // they are timing arms, never dispatched by the engine. The mul1 codebook's index is the byte sum of the hash,
  // folded under 1,024; the table codebooks' value IS the load, so their d6 is the shipped arm and their d7 is the
  // load alone.
  if (codebook == 2u) {
#if TF_DIAG_DECODE == 6
    // The LOAD kept, the index arithmetic gone.
    return float(Lut ? lut[s & 1023u] : table[codebook * 65536u + uint(s)]);
#else
    // The index arithmetic kept, the LOAD gone.
    const uint x = uint(s) * 0x83DCD12Du;
    const uint bs = (x & 0x00FF00FFu) + ((x >> 8u) & 0x00FF00FFu);
    const uint idx = (bs * 0x00010001u) >> 16u;
    return float(half(float(idx & 1u)));
#endif
  }
#if TF_DIAG_DECODE == 6
  // The table codebooks: the value IS the load -- the shipped arm.
  return float(table[codebook * 65536u + uint(s)]);
#else
  // The table codebooks: the load alone (no index arithmetic exists).
  return float(half(float(s & 1u)));
#endif
// THE BRANCH MUST TEST THE VALUE, NOT defined: the define defaults to 0, and defined is true for 0, so the engine was
// silently taking the ARITHMETIC value path (the format-s own decoder) instead of the threadgroup LUT.
#elif defined(TF_FMA_VALUE_ARITH) && TF_FMA_VALUE_ARITH
  // The ARITHMETIC value path (the format's own decoder) instead of the shared LUT.
  return float(tf_exl3_value(s, codebook));
#elif defined(TF_DECODE_TABLE)
  return Lut ? float(tf_exl3_value_lut(lut, s)) : float(tf_exl3_table_value(table, s, codebook));
#else
  return float(tf_exl3_value(s, codebook));
#endif
}

// One state, one product, with the weight already decoded. The product is the shipped body's: the same activation
// pick and the same fma.
template <uint J>
inline void tf_exl3_fma_step_w(float* acc_j, float w, float x0, float x1, float x2, float x3) {
  const uint odd = J & 1u, up = (J >> 1) & 1u;
  const float xv = odd == 0 ? (up == 0 ? x0 : x2) : (up == 0 ? x1 : x3);
#if defined(TF_DIAG_DECODE) && TF_DIAG_DECODE == 3
  // A removal arm would let the compiler delete the whole decode (the weight would be dead), so the price of the
  // arithmetic is read as the MARGINAL cost of one more FMA -- the doubling-arm discipline: a doubling arm isolates a
  // dispatch's marginal cost.
  *acc_j = fma(xv, w, *acc_j);
#endif
  *acc_j = fma(xv, w, *acc_j);
}

template <uint S, uint J, bool Lut = false>
inline void tf_exl3_fma_step(float* acc_j, uint c_cur, uint c_next, const device half* table, uint codebook,
                             float x0, float x1, float x2, float x3, const threadgroup half* lut = nullptr) {
  const float w = tf_exl3_fma_weight<S, J, Lut>(c_cur, c_next, table, codebook, lut);
  tf_exl3_fma_step_w<J>(acc_j, w, x0, x1, x2, x3);
}

// The register-only routed decode with the static state selection. The shipped body picks each state's word pair and
// bit offset at run time (the `rel[k]` cursor); here one funnel shift a block puts the lane's window at a known place
// and the eight windows then sit at compile-time shifts, so the cursor and the per-state bit offset leave the inner
// loop. The lane's eight states, the stream words, the bit offsets, the codebook values, the products and the
// reduction tree are the shipped body's, so the output is bit-identical. S is the states' stride in bits (halves /
// 2); the body is specialised for the widths the pack uses. `Dense` is the dense one-row path's variant of the same
// body. The routed decode reads its weights through descriptors[expert] (one expert a row); a dense matrix has one
// trellis and its strides live in the args, so the dense variant reads them from there and needs no descriptor table
// at all. The two differ only in where the five fields come from: the walk, the windows, the products, the reduction
// tree and the output are the same code, so the routed path's values do not move.
template <bool Dense, uint S, bool Lut = false, bool Two = false>
inline void tf_exl3_decode_fma_static_body(const device half* x, const device uchar* trellis,
                                            device float* sums, constant TfExl3Args& a,
                                            const device uint* routes, const device TfExl3Expert* descriptors,
                                            device float* partials, const device half* table,
                                            uint lane, uint sg, uint3 tg, threadgroup float* red_all,
                                            threadgroup half* lut = nullptr) {
  const uint row = tg.y;
  const uint nb = tg.x * 16u;
  if (row >= a.rows) return;
  const device uchar* data;
  ulong ks;
  ulong ns;
  uint halves;
  uint codebook;
  uint width = a.n;
  if (Dense) {
    data = trellis;
    ks = a.tile_k_stride;
    ns = a.tile_n_stride;
    halves = a.halves;
    codebook = a.codebook;
  } else {
    const uint expert = routes[row];
    if (expert >= a.experts) return;
    data = descriptors[expert].trellis;
    ks = descriptors[expert].tile_k_stride;
    ns = descriptors[expert].tile_n_stride;
    halves = descriptors[expert].halves;
    codebook = descriptors[expert].codebook;
    // The entry own output width (0 = the call n), for a batch of entries that differ in n.
    if (descriptors[expert].reserved != 0) width = descriptors[expert].reserved;
  }
  if (nb >= width) return;
  const uint blocks = a.k / 16;
  const uint W = TF_FMA_WARPS;
  const uint b0 = blocks * sg / W, b1 = blocks * (sg + 1u) / W;
  const uint words = 4 * halves;
  const uint wi0 = tf_exl3_map_for(lane * 8u, halves) & 63u;
  // The lane's bit offset inside its first word: the one runtime shift of the body.
  const uint off0 = (tf_exl3_map_for(lane * 8u, halves) >> 6) & 31u;
  const uint sh0 = (32u - off0) & 31u;
  const uint m0 = 0u - uint(off0 != 0u);
  uint a1 = wi0 + 1; if (a1 >= words) a1 -= words;
  uint a2 = wi0 + 2; if (a2 >= words) a2 -= words;
  // Three stream words: the lane's window spans at most 7*S + 16 bits from its first state. The ring of TF_FMA_PF
  // triples (one block in flight is the shipped form).
  uint pre[TF_FMA_PF][3];
  {
    const device ushort* t0 = reinterpret_cast<const device ushort*>(data + ulong(b0) * ks + ulong(nb / 16) * ns);
#if TF_FMA_WIDE_WORD
    pre[0][0] = tf_exl3_word_t<true>(t0, wi0);
    pre[0][1] = tf_exl3_word_t<true>(t0, a1);
    pre[0][2] = Two ? 0u : tf_exl3_word_t<true>(t0, a2);
#else
    pre[0][0] = tf_exl3_word(t0, wi0);
    pre[0][1] = tf_exl3_word(t0, a1);
    pre[0][2] = Two ? 0u : tf_exl3_word(t0, a2);
#endif
#if TF_FMA_PF > 1
    // The ring is primed: the first TF_FMA_PF blocks' words are in flight before the walk starts.
    for (uint p = 1; p < TF_FMA_PF; ++p) {
      if (b0 + p < b1) {
        const device ushort* tp = reinterpret_cast<const device ushort*>(data + ulong(b0 + p) * ks + ulong(nb / 16) * ns);
        pre[p][0] = tf_exl3_word(tp, wi0);
        pre[p][1] = tf_exl3_word(tp, a1);
        pre[p][2] = Two ? 0u : tf_exl3_word(tp, a2);
      }
    }
#endif
  }
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint bb = b0; bb < b1; ++bb) {
    const uint kb = bb * 16;
#if TF_FMA_PF > 1
    const uint slot = (bb - b0) % TF_FMA_PF;
    const uint w0 = pre[slot][0], w1 = pre[slot][1], w2 = pre[slot][2];
    // The block TF_FMA_PF ahead takes this slot: its words are in flight while PF blocks are walked.
    if (bb + TF_FMA_PF < b1) {
      const device ushort* tn = reinterpret_cast<const device ushort*>(data + ulong(bb + TF_FMA_PF) * ks + ulong(nb / 16) * ns);
#if TF_FMA_WIDE_WORD
      pre[slot][0] = tf_exl3_word_t<true>(tn, wi0);
      pre[slot][1] = tf_exl3_word_t<true>(tn, a1);
      pre[slot][2] = Two ? 0u : tf_exl3_word_t<true>(tn, a2);
#else
      pre[slot][0] = tf_exl3_word(tn, wi0);
      pre[slot][1] = tf_exl3_word(tn, a1);
      pre[slot][2] = Two ? 0u : tf_exl3_word(tn, a2);
#endif
    }
#else
    const uint w0 = pre[0][0], w1 = pre[0][1], w2 = pre[0][2];
    if (bb + 1 < b1) {
      const device ushort* tn = reinterpret_cast<const device ushort*>(data + ulong(bb + 1) * ks + ulong(nb / 16) * ns);
#if TF_FMA_WIDE_WORD
      pre[0][0] = tf_exl3_word_t<true>(tn, wi0);
      pre[0][1] = tf_exl3_word_t<true>(tn, a1);
      pre[0][2] = Two ? 0u : tf_exl3_word_t<true>(tn, a2);
#else
      pre[0][0] = tf_exl3_word(tn, wi0);
      pre[0][1] = tf_exl3_word(tn, a1);
      pre[0][2] = Two ? 0u : tf_exl3_word(tn, a2);
#endif
    }
#endif
    // The lane's window: one runtime funnel shift, then the eight windows at compile-time shifts.
    const uint c_cur = (w0 << off0) | ((w1 >> sh0) & m0);
    const uint c_next = Two ? (w1 << off0) : ((w1 << off0) | ((w2 >> sh0) & m0));
    // The lane's four k rows of this block, read as two halves each (the rows are adjacent in pairs).
#if defined(TF_DIAG_DECODE) && TF_DIAG_DECODE == 4
    // The FMA stays (the weight is still needed) and the constant is a RUNTIME value (the args' rows), so the
    // compiler cannot fold the multiply away and the arm isolates the read, not the arithmetic.
    const float x0 = float(a.rows), x1 = float(a.rows), x2 = float(a.rows), x3 = float(a.rows);
#else
    const device half* xr = x + ulong(row) * a.k + kb + 2 * (lane & 3u);
    const half2 xa = *(const device half2*)xr;
    const half2 xb = *(const device half2*)(xr + 8);
    const float x0 = float(xa.x), x1 = float(xa.y), x2 = float(xb.x), x3 = float(xb.y);
#endif
    tf_exl3_fma_step<S, 0, Lut>(&acc[0], c_cur, c_next, table, codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 1, Lut>(&acc[1], c_cur, c_next, table, codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 2, Lut>(&acc[2], c_cur, c_next, table, codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 3, Lut>(&acc[3], c_cur, c_next, table, codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 4, Lut>(&acc[4], c_cur, c_next, table, codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 5, Lut>(&acc[5], c_cur, c_next, table, codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 6, Lut>(&acc[6], c_cur, c_next, table, codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 7, Lut>(&acc[7], c_cur, c_next, table, codebook, x0, x1, x2, x3, lut);
  }
  // The eight accumulators meet in a tree: the first four are the near column, the last four the far one.
  float acc0 = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  float acc1 = (acc[4] + acc[5]) + (acc[6] + acc[7]);
#if defined(TF_DIAG_DECODE) && TF_DIAG_DECODE == 5
#elif TF_FMA_WARPS > 1
  red_all[sg * 64u + lane] = acc0;
  red_all[sg * 64u + 32u + lane] = acc1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  acc0 = 0.0f;
  acc1 = 0.0f;
  for (uint w = 0; w < W; ++w) {
    acc0 += red_all[w * 64u + lane];
    acc1 += red_all[w * 64u + 32u + lane];
  }
#endif
  acc0 += simd_shuffle_xor(acc0, 1);
  acc0 += simd_shuffle_xor(acc0, 2);
  acc1 += simd_shuffle_xor(acc1, 1);
  acc1 += simd_shuffle_xor(acc1, 2);
  const bool store_it = (lane & 3u) == 0u
#if defined(TF_DIAG_DECODE) && TF_DIAG_DECODE == 5
      and sg == 0
#endif
      ;
  if (store_it) {
    sums[ulong(row) * a.n + nb + lane / 4] = acc0;
    sums[ulong(row) * a.n + nb + 8 + lane / 4] = acc1;
  }
}

// THE MULTI-ROW DENSE DECODE BODY. One threadgroup computes R rows of the SAME dense matrix, so the weight decode --
// the ALU-bound part of the register-only body -- is shared by the rows instead of paid once a row (the shipped body
// is one row a threadgroup: `row = tg.y`, and the dispatch's height is the row count, so a four-row window decodes
// the attention's dense weights four times over). Each row keeps the shipped per-row arithmetic exactly: the same k
// blocks, the same warp split, the same accumulator tree, the same cross-warp sum and the same lane shuffles -- so
// every row's sum is bit-identical to the one-row body's. One decode for the rows that share a weight: the shape the
// pair and multi-row bodies use.
template <uint S, bool Lut = false, uint R = 2>
inline void tf_exl3_decode_fma_dense_multi_body(const device half* x, const device uchar* trellis,
                                                 device float* sums, constant TfExl3Args& a,
                                                 const device half* table, uint lane, uint sg, uint3 tg,
                                                 threadgroup float* red_all, threadgroup half* lut = nullptr) {
  const uint row0 = tg.y * R;
  const uint nb = tg.x * 16u;
  if (row0 >= a.rows or nb >= a.n) return;
  const device uchar* data = trellis;
  const ulong ks = a.tile_k_stride;
  const ulong ns = a.tile_n_stride;
  const uint halves = a.halves;
  const uint codebook = a.codebook;
  const uint blocks = a.k / 16;
  const uint W = TF_FMA_WARPS;
  const uint b0 = blocks * sg / W, b1 = blocks * (sg + 1u) / W;
  const uint words = 4 * halves;
  const uint wi0 = tf_exl3_map_for(lane * 8u, halves) & 63u;
  // The lane's bit offset inside its first word: the one runtime shift of the body.
  const uint off0 = (tf_exl3_map_for(lane * 8u, halves) >> 6) & 31u;
  const uint sh0 = (32u - off0) & 31u;
  const uint m0 = 0u - uint(off0 != 0u);
  uint a1 = wi0 + 1; if (a1 >= words) a1 -= words;
  uint a2 = wi0 + 2; if (a2 >= words) a2 -= words;
  // Three stream words: the lane's window spans at most 7*S + 16 bits from its first state.
  uint pre[3];
  {
    const device ushort* t0 = reinterpret_cast<const device ushort*>(data + ulong(b0) * ks + ulong(nb / 16) * ns);
    pre[0] = tf_exl3_word(t0, wi0);
    pre[1] = tf_exl3_word(t0, a1);
    pre[2] = tf_exl3_word(t0, a2);
  }
  float acc[R][8];
  for (uint r = 0; r < R; ++r) for (uint j = 0; j < 8u; ++j) acc[r][j] = 0.0f;
  for (uint bb = b0; bb < b1; ++bb) {
    const uint kb = bb * 16;
    const uint w0 = pre[0], w1 = pre[1], w2 = pre[2];
    if (bb + 1 < b1) {
      const device ushort* tn = reinterpret_cast<const device ushort*>(data + ulong(bb + 1) * ks + ulong(nb / 16) * ns);
      pre[0] = tf_exl3_word(tn, wi0);
      pre[1] = tf_exl3_word(tn, a1);
      pre[2] = tf_exl3_word(tn, a2);
    }
    // The lane's window: one runtime funnel shift, then the eight windows at compile-time shifts.
    const uint c_cur = (w0 << off0) | ((w1 >> sh0) & m0);
    const uint c_next = (w1 << off0) | ((w2 >> sh0) & m0);
    // The eight weights are decoded ONCE for the block, and every row's product uses them.
    float wgt[8];
    wgt[0] = tf_exl3_fma_weight<S, 0, Lut>(c_cur, c_next, table, codebook, lut);
    wgt[1] = tf_exl3_fma_weight<S, 1, Lut>(c_cur, c_next, table, codebook, lut);
    wgt[2] = tf_exl3_fma_weight<S, 2, Lut>(c_cur, c_next, table, codebook, lut);
    wgt[3] = tf_exl3_fma_weight<S, 3, Lut>(c_cur, c_next, table, codebook, lut);
    wgt[4] = tf_exl3_fma_weight<S, 4, Lut>(c_cur, c_next, table, codebook, lut);
    wgt[5] = tf_exl3_fma_weight<S, 5, Lut>(c_cur, c_next, table, codebook, lut);
    wgt[6] = tf_exl3_fma_weight<S, 6, Lut>(c_cur, c_next, table, codebook, lut);
    wgt[7] = tf_exl3_fma_weight<S, 7, Lut>(c_cur, c_next, table, codebook, lut);
    for (uint r = 0; r < R; ++r) {
      const uint row = row0 + r;
      if (row >= a.rows) break;
      // The lane's four k rows of this block, read as two halves each (the rows are adjacent in pairs).
#if defined(TF_DIAG_DECODE) && TF_DIAG_DECODE == 4
      const float x0 = float(a.rows), x1 = float(a.rows), x2 = float(a.rows), x3 = float(a.rows);
#else
      const device half* xr = x + ulong(row) * a.k + kb + 2 * (lane & 3u);
      const half2 xa = *(const device half2*)xr;
      const half2 xb = *(const device half2*)(xr + 8);
      const float x0 = float(xa.x), x1 = float(xa.y), x2 = float(xb.x), x3 = float(xb.y);
#endif
      tf_exl3_fma_step_w<0>(&acc[r][0], wgt[0], x0, x1, x2, x3);
      tf_exl3_fma_step_w<1>(&acc[r][1], wgt[1], x0, x1, x2, x3);
      tf_exl3_fma_step_w<2>(&acc[r][2], wgt[2], x0, x1, x2, x3);
      tf_exl3_fma_step_w<3>(&acc[r][3], wgt[3], x0, x1, x2, x3);
      tf_exl3_fma_step_w<4>(&acc[r][4], wgt[4], x0, x1, x2, x3);
      tf_exl3_fma_step_w<5>(&acc[r][5], wgt[5], x0, x1, x2, x3);
      tf_exl3_fma_step_w<6>(&acc[r][6], wgt[6], x0, x1, x2, x3);
      tf_exl3_fma_step_w<7>(&acc[r][7], wgt[7], x0, x1, x2, x3);
    }
  }
  // The eight accumulators meet in a tree: the first four are the near column, the last four the far one. One row a
  // turn, with the threadgroup's partials reused: the tree, the cross-warp sum and the lane shuffles are the shipped
  // body's, so the row's sum does not move.
  for (uint r = 0; r < R; ++r) {
    const uint row = row0 + r;
    if (row >= a.rows) break;
    float acc0 = (acc[r][0] + acc[r][1]) + (acc[r][2] + acc[r][3]);
    float acc1 = (acc[r][4] + acc[r][5]) + (acc[r][6] + acc[r][7]);
#if defined(TF_DIAG_DECODE) && TF_DIAG_DECODE == 5
#elif TF_FMA_WARPS > 1
    red_all[sg * 64u + lane] = acc0;
    red_all[sg * 64u + 32u + lane] = acc1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      acc0 = 0.0f;
      acc1 = 0.0f;
      for (uint w = 0; w < W; ++w) {
        acc0 += red_all[w * 64u + lane];
        acc1 += red_all[w * 64u + 32u + lane];
      }
    }
#endif
    acc0 += simd_shuffle_xor(acc0, 1);
    acc0 += simd_shuffle_xor(acc0, 2);
    acc1 += simd_shuffle_xor(acc1, 1);
    acc1 += simd_shuffle_xor(acc1, 2);
    if (sg == 0 and (lane & 3u) == 0u) {
      sums[ulong(row) * a.n + nb + lane / 4] = acc0;
      sums[ulong(row) * a.n + nb + 8 + lane / 4] = acc1;
    }
#if TF_FMA_WARPS > 1
    // The next row reuses the partials: every thread reaches this barrier.
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
  }
}

// The LUT's values are the table's, entry for entry, so the output is bit-identical; what the arm removes is the
// table's L2 traffic.
kernel void tf_exl3_matmul_decode_fma_static_lut(const device half* x [[buffer(0)]],
                                                  const device uchar* trellis [[buffer(1)]],
                                                  device float* sums [[buffer(2)]],
                                                  constant TfExl3Args& a [[buffer(3)]],
                                                  const device uint* routes [[buffer(4)]],
                                                  const device TfExl3Expert* descriptors [[buffer(5)]],
                                                  device float* partials [[buffer(6)]],
#if defined(TF_DECODE_TABLE)
                                                  const device half* table [[buffer(7)]],
#endif
                                                  uint lane [[thread_index_in_simdgroup]],
                                                  uint sg [[simdgroup_index_in_threadgroup]],
                                                  uint tid [[thread_index_in_threadgroup]],
                                                  uint3 tptg [[threads_per_threadgroup]],
                                                  uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float red[TF_FMA_WARPS][64];
  threadgroup half lut[1024];
  const uint row = tg.y;
  if (row >= a.rows) return;
  const uint expert = routes[row];
  if (expert >= a.experts) return;
  const uint halves = descriptors[expert].halves;
  // The LUT holds the mul1 values, so it is only the right table for that codebook; any other codebook reads the
  // resident table (and the branch is uniform across the threadgroup, which shares one expert).
  const bool mul1 = descriptors[expert].codebook == 2u;
  if (mul1) {
    const float scale = float(as_type<half>(ushort(0x1EEEu)));
    const float bias = float(as_type<half>(ushort(0xC931u)));
    for (uint b = tid; b < 1024u; b += tptg.x) lut[b] = half(fma(float(b + 1024u), scale, bias));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (halves == 6u) {
    if (mul1) tf_exl3_decode_fma_static_body<false, 3, true, (TF_FMA_TWO != 0)>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0], lut);
    else tf_exl3_decode_fma_static_body<false, 3, false, (TF_FMA_TWO != 0)>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  } else if (halves == 10u) {
    if (mul1) tf_exl3_decode_fma_static_body<false, 5, true>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0], lut);
    else tf_exl3_decode_fma_static_body<false, 5, false>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  } else if (halves == 12u) {
    if (mul1) tf_exl3_decode_fma_static_body<false, 6, true>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0], lut);
    else tf_exl3_decode_fma_static_body<false, 6, false>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  } else if (halves == 8u) {
    if (mul1) tf_exl3_decode_fma_static_body<false, 4, true>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0], lut);
    else tf_exl3_decode_fma_static_body<false, 4, false>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  } else if (halves == 4u) {
    if (mul1) tf_exl3_decode_fma_static_body<false, 2, true>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0], lut);
    else tf_exl3_decode_fma_static_body<false, 2, false>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  } else if (halves == 2u) {
    if (mul1) tf_exl3_decode_fma_static_body<false, 1, true>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0], lut);
    else tf_exl3_decode_fma_static_body<false, 1, false>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  }
}

// The register-only body on the DENSE one-row path. The dense matrix has no expert descriptor, so this kernel reads
// the trellis from the buffer and the strides, the width and the codebook from the args: the walk, the windows, the
// products and the reduction tree are the routed body's own code, so the only difference from the routed body is
// where the five fields come from.
kernel void tf_exl3_matmul_decode_fma_dense(const device half* x [[buffer(0)]],
                                             const device uchar* trellis [[buffer(1)]],
                                             device float* sums [[buffer(2)]],
                                             constant TfExl3Args& a [[buffer(3)]],
                                             const device uint* routes [[buffer(4)]],
                                             const device TfExl3Expert* descriptors [[buffer(5)]],
                                             device float* partials [[buffer(6)]],
#if defined(TF_DECODE_TABLE)
                                             const device half* table [[buffer(7)]],
#endif
                                             uint lane [[thread_index_in_simdgroup]],
                                             uint sg [[simdgroup_index_in_threadgroup]],
                                             uint tid [[thread_index_in_threadgroup]],
                                             uint3 tptg [[threads_per_threadgroup]],
                                             uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float red[TF_FMA_WARPS][64];
  threadgroup half lut[1024];
  const uint row = tg.y;
  if (row >= a.rows) return;
  const uint nb = tg.x * 16u;
  if (nb >= a.n) return;
  const uint halves = a.halves;
  // The mul1 codebook's values depend only on the state's byte sum, so 1,024 entries hold every value; the LUT's
  // values are the resident table's, entry for entry, so the output is the table's.
  const bool mul1 = a.codebook == 2u;
  if (mul1) {
    const float scale = float(as_type<half>(ushort(0x1EEEu)));
    const float bias = float(as_type<half>(ushort(0xC931u)));
    for (uint b = tid; b < 1024u; b += tptg.x) lut[b] = half(fma(float(b + 1024u), scale, bias));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (halves == 6u) {
    if (mul1) tf_exl3_decode_fma_static_body<true, 3, true>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0], lut);
    else tf_exl3_decode_fma_static_body<true, 3, false>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  } else
  if (halves == 10u) {
    if (mul1) tf_exl3_decode_fma_static_body<true, 5, true>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0], lut);
    else tf_exl3_decode_fma_static_body<true, 5, false>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  } else
  if (halves == 12u) {
    if (mul1) tf_exl3_decode_fma_static_body<true, 6, true>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0], lut);
    else tf_exl3_decode_fma_static_body<true, 6, false>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  } else
  if (halves == 8u) {
    if (mul1) tf_exl3_decode_fma_static_body<true, 4, true>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0], lut);
    else tf_exl3_decode_fma_static_body<true, 4, false>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  } else
  if (halves == 4u) {
    if (mul1) tf_exl3_decode_fma_static_body<true, 2, true>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0], lut);
    else tf_exl3_decode_fma_static_body<true, 2, false>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  } else
  if (halves == 2u) {
    if (mul1) tf_exl3_decode_fma_static_body<true, 1, true>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0], lut);
    else tf_exl3_decode_fma_static_body<true, 1, false>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  }
}

// The multi-row dense body, two rows a threadgroup and four rows a threadgroup. The width is the matrix's (the dense
// pack's are 6, 10, 12 and 8 halves), so the body is specialised on it at dispatch.
template <uint R>
inline void tf_exl3_decode_fma_dense_multi_kernel(const device half* x, const device uchar* trellis,
                                                   device float* sums, constant TfExl3Args& a,
                                                   const device half* table, uint lane, uint sg, uint tid,
                                                   uint3 tptg, uint3 tg, threadgroup float* red,
                                                   threadgroup half* lut) {
  const uint halves = a.halves;
  const bool mul1 = a.codebook == 2u;
  if (halves == 6u) {
    if (mul1) tf_exl3_decode_fma_dense_multi_body<3, true, R>(x, trellis, sums, a, table, lane, sg, tg, red, lut);
    else tf_exl3_decode_fma_dense_multi_body<3, false, R>(x, trellis, sums, a, table, lane, sg, tg, red);
  } else
  if (halves == 10u) {
    if (mul1) tf_exl3_decode_fma_dense_multi_body<5, true, R>(x, trellis, sums, a, table, lane, sg, tg, red, lut);
    else tf_exl3_decode_fma_dense_multi_body<5, false, R>(x, trellis, sums, a, table, lane, sg, tg, red);
  } else
  if (halves == 12u) {
    if (mul1) tf_exl3_decode_fma_dense_multi_body<6, true, R>(x, trellis, sums, a, table, lane, sg, tg, red, lut);
    else tf_exl3_decode_fma_dense_multi_body<6, false, R>(x, trellis, sums, a, table, lane, sg, tg, red);
  } else
  if (halves == 8u) {
    if (mul1) tf_exl3_decode_fma_dense_multi_body<4, true, R>(x, trellis, sums, a, table, lane, sg, tg, red, lut);
    else tf_exl3_decode_fma_dense_multi_body<4, false, R>(x, trellis, sums, a, table, lane, sg, tg, red);
  } else
  if (halves == 4u) {
    if (mul1) tf_exl3_decode_fma_dense_multi_body<2, true, R>(x, trellis, sums, a, table, lane, sg, tg, red, lut);
    else tf_exl3_decode_fma_dense_multi_body<2, false, R>(x, trellis, sums, a, table, lane, sg, tg, red);
  } else
  if (halves == 2u) {
    if (mul1) tf_exl3_decode_fma_dense_multi_body<1, true, R>(x, trellis, sums, a, table, lane, sg, tg, red, lut);
    else tf_exl3_decode_fma_dense_multi_body<1, false, R>(x, trellis, sums, a, table, lane, sg, tg, red);
  }
}

// The shipped register-only dense body gives a lane EIGHT consecutive states -- two columns by four k-rows of a 16x16
// tile -- and reads three 32-bit stream words to decode them (7*S + 16 bits from its first state). This body gives a
// lane SIXTEEN consecutive states -- two columns by eight k-rows, p = 16L..16L+15 -- so the same lane reads four
// words (S = 5: the run is 15*5 + 16 = 91 bits) and the words a weight costs fall from 3/8 to 4/16. The warp still
// owns ONE sixteen-column tile: lanes 0..15 walk one 16-k block and lanes 16..31 the next, so the dispatch geometry,
// the column coverage and the split-K warp count are the shipped body's. The body is only compiled when TF_FMA_S16 is
// defined; no engine path selects it.
#if defined(TF_FMA_S16)
// The sixteen windows of a lane at compile-time shifts, from the four aligned 32-bit windows c0..c3 of its stream
// run. Same semantics as tf_exl3_state_static: the state's 16-bit window starts S*J bits into the run.
template <uint S, uint J>
inline ushort tf_exl3_state_s16(uint c0, uint c1, uint c2, uint c3) {
  const uint bit = S * J;
  if (bit <= 16u) return ushort((c0 >> (16u - bit)) & 0xFFFFu);
  if (bit <= 32u) return ushort(((c0 << (bit - 16u)) | (c1 >> (48u - bit))) & 0xFFFFu);
  if (bit <= 48u) return ushort((c1 >> (48u - bit)) & 0xFFFFu);
  if (bit <= 64u) return ushort(((c1 << (bit - 48u)) | (c2 >> (80u - bit))) & 0xFFFFu);
  if (bit <= 80u) return ushort((c2 >> (80u - bit)) & 0xFFFFu);
  if (bit <= 96u) return ushort(((c2 << (bit - 80u)) | (c3 >> (112u - bit))) & 0xFFFFu);
  return ushort((c3 >> (112u - bit)) & 0xFFFFu);
}

// One state, one product: the value from the resident table and the activation row the state's (row, column) pair
// names. The xv order is the caller's: index 4*(J>>3) + 2*((J>>1)&1) + (J&1), rows B + 2*(J>>3) + (J&1) +
// 8*((J>>1)&1) for the lane's B = 4*(L&1).
template <uint S, uint J>
inline void tf_exl3_s16_step(float* acc, uint c0, uint c1, uint c2, uint c3,
                             const device half* table, uint codebook, const float* xv) {
  const ushort s = tf_exl3_state_s16<S, J>(c0, c1, c2, c3);
  const float w = float(tf_exl3_table_value(table, s, codebook));
  const uint idx = 4u * (J >> 3) + 2u * ((J >> 1) & 1u) + (J & 1u);
  *acc = fma(xv[idx], w, *acc);
}

template <uint S>
inline void tf_exl3_decode_s16_body(const device half* x, const device uchar* trellis,
                                    device float* sums, constant TfExl3Args& a,
                                    const device half* table, uint lane, uint sg,
                                    uint3 tg, threadgroup float* red_all) {
  const uint row = tg.y;
  const uint nb = tg.x * 16u;
  if (row >= a.rows or nb >= a.n) return;
  const uint L = lane & 15u;
  const uint g = lane >> 4;
  const uint blocks = a.k / 16u;
  const uint W = TF_FMA_WARPS;
  const uint b0 = blocks * sg / W, b1 = blocks * (sg + 1u) / W;
  const uint words = 4u * a.halves;
  const uint p0 = L * 16u;
  const uint wi0 = tf_exl3_map_for(p0, a.halves) & 63u;
  const uint off0 = (tf_exl3_map_for(p0, a.halves) >> 6) & 31u;
  const uint sh0 = (32u - off0) & 31u;
  const uint m0 = 0u - uint(off0 != 0u);
  uint a1 = wi0 + 1u; if (a1 >= words) a1 -= words;
  uint a2 = wi0 + 2u; if (a2 >= words) a2 -= words;
  uint a3 = wi0 + 3u; if (a3 >= words) a3 -= words;
  uint a4 = wi0 + 4u; if (a4 >= words) a4 -= words;
  // Four words cover S = 5 (91 bits); five cover S = 6 (106 bits from an offset of 22).
  uint pre[5];
  const uint first = b0 + g;
  if (first < b1) {
    const device ushort* t0 = reinterpret_cast<const device ushort*>(
        trellis + ulong(first) * a.tile_k_stride + ulong(nb / 16u) * a.tile_n_stride);
    pre[0] = tf_exl3_word(t0, wi0);
    pre[1] = tf_exl3_word(t0, a1);
    pre[2] = tf_exl3_word(t0, a2);
    pre[3] = tf_exl3_word(t0, a3);
    if (S > 5u) pre[4] = tf_exl3_word(t0, a4);
  }
  float acc[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint bb = b0 + g; bb < b1; bb += 2u) {
    const uint kb = bb * 16u;
    const uint w0 = pre[0], w1 = pre[1], w2 = pre[2], w3 = pre[3];
    uint w4 = 0u;
    if (S > 5u) w4 = pre[4];
    if (bb + 2u < b1) {
      const device ushort* tn = reinterpret_cast<const device ushort*>(
          trellis + ulong(bb + 2u) * a.tile_k_stride + ulong(nb / 16u) * a.tile_n_stride);
      pre[0] = tf_exl3_word(tn, wi0);
      pre[1] = tf_exl3_word(tn, a1);
      pre[2] = tf_exl3_word(tn, a2);
      pre[3] = tf_exl3_word(tn, a3);
      if (S > 5u) pre[4] = tf_exl3_word(tn, a4);
    }
    // The lane's run as aligned 32-bit windows: one funnel shift a word, then fixed shifts a state.
    const uint c0 = (w0 << off0) | ((w1 >> sh0) & m0);
    const uint c1 = (w1 << off0) | ((w2 >> sh0) & m0);
    const uint c2 = (w2 << off0) | ((w3 >> sh0) & m0);
    uint c3 = 0u;
    if (S > 5u) c3 = (w3 << off0) | ((w4 >> sh0) & m0);
    // Eight activation values: the lane's eight k-rows, the same rows its two columns share.
    const uint xo = kb + 4u * (L & 1u);
    const device half* xr = x + ulong(row) * a.k + xo;
    const half2 xa0 = *(const device half2*)xr;
    const half2 xa1 = *(const device half2*)(xr + 2);
    const half2 xb0 = *(const device half2*)(xr + 8);
    const half2 xb1 = *(const device half2*)(xr + 10);
    float xv[8];
    xv[0] = float(xa0.x); xv[1] = float(xa0.y);
    xv[2] = float(xb0.x); xv[3] = float(xb0.y);
    xv[4] = float(xa1.x); xv[5] = float(xa1.y);
    xv[6] = float(xb1.x); xv[7] = float(xb1.y);
    tf_exl3_s16_step<S, 0>(&acc[0], c0, c1, c2, c3, table, a.codebook, xv);
    tf_exl3_s16_step<S, 1>(&acc[1], c0, c1, c2, c3, table, a.codebook, xv);
    tf_exl3_s16_step<S, 2>(&acc[2], c0, c1, c2, c3, table, a.codebook, xv);
    tf_exl3_s16_step<S, 3>(&acc[3], c0, c1, c2, c3, table, a.codebook, xv);
    tf_exl3_s16_step<S, 4>(&acc[4], c0, c1, c2, c3, table, a.codebook, xv);
    tf_exl3_s16_step<S, 5>(&acc[5], c0, c1, c2, c3, table, a.codebook, xv);
    tf_exl3_s16_step<S, 6>(&acc[6], c0, c1, c2, c3, table, a.codebook, xv);
    tf_exl3_s16_step<S, 7>(&acc[7], c0, c1, c2, c3, table, a.codebook, xv);
    tf_exl3_s16_step<S, 8>(&acc[8], c0, c1, c2, c3, table, a.codebook, xv);
    tf_exl3_s16_step<S, 9>(&acc[9], c0, c1, c2, c3, table, a.codebook, xv);
    tf_exl3_s16_step<S, 10>(&acc[10], c0, c1, c2, c3, table, a.codebook, xv);
    tf_exl3_s16_step<S, 11>(&acc[11], c0, c1, c2, c3, table, a.codebook, xv);
    tf_exl3_s16_step<S, 12>(&acc[12], c0, c1, c2, c3, table, a.codebook, xv);
    tf_exl3_s16_step<S, 13>(&acc[13], c0, c1, c2, c3, table, a.codebook, xv);
    tf_exl3_s16_step<S, 14>(&acc[14], c0, c1, c2, c3, table, a.codebook, xv);
    tf_exl3_s16_step<S, 15>(&acc[15], c0, c1, c2, c3, table, a.codebook, xv);
  }
  // Eight accumulators a column meet in a tree: the near column is J mod 8 < 4, the far one the rest.
  float accA = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[8] + acc[9]) + (acc[10] + acc[11]));
  float accB = ((acc[4] + acc[5]) + (acc[6] + acc[7])) + ((acc[12] + acc[13]) + (acc[14] + acc[15]));
#if TF_FMA_WARPS > 1
  red_all[sg * 64u + lane] = accA;
  red_all[sg * 64u + 32u + lane] = accB;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  accA = 0.0f;
  accB = 0.0f;
  TF_UNROLL
  for (uint w = 0; w < W; ++w) {
    accA += red_all[w * 64u + lane];
    accB += red_all[w * 64u + 32u + lane];
  }
#endif
  // Two lanes a column pair inside a 16-k block (the 8-row halves), then the two block halves.
  accA += simd_shuffle_xor(accA, 1);
  accB += simd_shuffle_xor(accB, 1);
  accA += simd_shuffle_xor(accA, 16);
  accB += simd_shuffle_xor(accB, 16);
  if (sg == 0 and (lane & 17u) == 0u) {
    const uint j = lane >> 1;
    sums[ulong(row) * a.n + nb + j] = accA;
    sums[ulong(row) * a.n + nb + 8u + j] = accB;
  }
}

kernel void tf_exl3_matmul_decode_fma_s16(const device half* x [[buffer(0)]],
                                          const device uchar* trellis [[buffer(1)]],
                                          device float* sums [[buffer(2)]],
                                          constant TfExl3Args& a [[buffer(3)]],
                                          const device uint* routes [[buffer(4)]],
                                          const device TfExl3Expert* descriptors [[buffer(5)]],
                                          device float* partials [[buffer(6)]],
                                          const device half* table [[buffer(7)]],
                                          uint lane [[thread_index_in_simdgroup]],
                                          uint sg [[simdgroup_index_in_threadgroup]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint3 tptg [[threads_per_threadgroup]],
                                          uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float red[TF_FMA_WARPS][64];
  if (a.halves == 10u) tf_exl3_decode_s16_body<5>(x, trellis, sums, a, table, lane, sg, tg, &red[0][0]);
  else if (a.halves == 12u) tf_exl3_decode_s16_body<6>(x, trellis, sums, a, table, lane, sg, tg, &red[0][0]);
}
#endif

kernel void tf_exl3_matmul_decode_fma_dense_multi2(const device half* x [[buffer(0)]],
                                                    const device uchar* trellis [[buffer(1)]],
                                                    device float* sums [[buffer(2)]],
                                                    constant TfExl3Args& a [[buffer(3)]],
                                                    const device uint* routes [[buffer(4)]],
                                                    const device TfExl3Expert* descriptors [[buffer(5)]],
                                                    device float* partials [[buffer(6)]],
#if defined(TF_DECODE_TABLE)
                                                    const device half* table [[buffer(7)]],
#endif
                                                    uint lane [[thread_index_in_simdgroup]],
                                                    uint sg [[simdgroup_index_in_threadgroup]],
                                                    uint tid [[thread_index_in_threadgroup]],
                                                    uint3 tptg [[threads_per_threadgroup]],
                                                    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float red[TF_FMA_WARPS][64];
  threadgroup half lut[1024];
  const bool mul1 = a.codebook == 2u;
  if (mul1) {
    const float scale = float(as_type<half>(ushort(0x1EEEu)));
    const float bias = float(as_type<half>(ushort(0xC931u)));
    for (uint b = tid; b < 1024u; b += tptg.x) lut[b] = half(fma(float(b + 1024u), scale, bias));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  tf_exl3_decode_fma_dense_multi_kernel<2>(x, trellis, sums, a,
#if defined(TF_DECODE_TABLE)
    table,
#else
    nullptr,
#endif
    lane, sg, tid, tptg, tg, &red[0][0], lut);
}

kernel void tf_exl3_matmul_decode_fma_dense_multi4(const device half* x [[buffer(0)]],
                                                    const device uchar* trellis [[buffer(1)]],
                                                    device float* sums [[buffer(2)]],
                                                    constant TfExl3Args& a [[buffer(3)]],
                                                    const device uint* routes [[buffer(4)]],
                                                    const device TfExl3Expert* descriptors [[buffer(5)]],
                                                    device float* partials [[buffer(6)]],
#if defined(TF_DECODE_TABLE)
                                                    const device half* table [[buffer(7)]],
#endif
                                                    uint lane [[thread_index_in_simdgroup]],
                                                    uint sg [[simdgroup_index_in_threadgroup]],
                                                    uint tid [[thread_index_in_threadgroup]],
                                                    uint3 tptg [[threads_per_threadgroup]],
                                                    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float red[TF_FMA_WARPS][64];
  threadgroup half lut[1024];
  const bool mul1 = a.codebook == 2u;
  if (mul1) {
    const float scale = float(as_type<half>(ushort(0x1EEEu)));
    const float bias = float(as_type<half>(ushort(0xC931u)));
    for (uint b = tid; b < 1024u; b += tptg.x) lut[b] = half(fma(float(b + 1024u), scale, bias));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  tf_exl3_decode_fma_dense_multi_kernel<4>(x, trellis, sums, a,
#if defined(TF_DECODE_TABLE)
    table,
#else
    nullptr,
#endif
    lane, sg, tid, tptg, tg, &red[0][0], lut);
}

template <uint S, bool Lut = false>
inline void tf_exl3_decode_fma_static_pair_body(const device half* x, const device uchar* trellis,
                                                 device float* sums, constant TfExl3Args& a,
                                                 const device uint* routes, const device TfExl3Expert* descriptors,
                                                 const device half* table,
                                                 uint lane, uint sg, uint3 tg, threadgroup float* red_all,
                                                 threadgroup half* lut = nullptr) {
  const uint row = tg.y;
  const uint nb = tg.x * 16u;
  if (row >= a.rows or nb >= a.n) return;
  // THE PAIR. The entries that name the same expert, in entry order, are ranked; the even ranks lead a pair (their
  // entry and the next one with the same expert) and the odd ranks exit, so the weight decode -- and the weight
  // traffic -- is paid once for the pair.: one decode for the rows that share a weight. Each row keeps the shipped
  // per-row arithmetic, so the pair's rows are bit-identical to the single-row body's.
  const uint slot = routes[row];
  uint rank = 0u;
  uint mate = row;
  for (uint j = 0; j < a.rows; ++j) {
    const uint same = (routes[j] == slot) ? 1u : 0u;
    rank += (j < row) ? same : 0u;
    if (mate == row and j > row and same == 1u) mate = j;
  }
  if ((rank & 1u) != 0u) return;   // the odd rank's row is computed by its leader
  const bool paired = mate != row;
  const device uchar* data;
  ulong ks;
  ulong ns;
  uint halves;
  uint codebook;
  {
    const uint expert = slot;
    if (expert >= a.experts) return;
    data = descriptors[expert].trellis;
    ks = descriptors[expert].tile_k_stride;
    ns = descriptors[expert].tile_n_stride;
    halves = descriptors[expert].halves;
    codebook = descriptors[expert].codebook;
  }
  const uint blocks = a.k / 16;
  const uint W = TF_FMA_WARPS;
  const uint b0 = blocks * sg / W, b1 = blocks * (sg + 1u) / W;
  const uint words = 4 * halves;
  const uint wi0 = tf_exl3_map_for(lane * 8u, halves) & 63u;
  // The lane's bit offset inside its first word: the one runtime shift of the body.
  const uint off0 = (tf_exl3_map_for(lane * 8u, halves) >> 6) & 31u;
  const uint sh0 = (32u - off0) & 31u;
  const uint m0 = 0u - uint(off0 != 0u);
  uint a1 = wi0 + 1; if (a1 >= words) a1 -= words;
  uint a2 = wi0 + 2; if (a2 >= words) a2 -= words;
  // Three stream words: the lane's window spans at most 7*S + 16 bits from its first state.
  uint pre[3];
  {
    const device ushort* t0 = reinterpret_cast<const device ushort*>(data + ulong(b0) * ks + ulong(nb / 16) * ns);
    pre[0] = tf_exl3_word(t0, wi0);
    pre[1] = tf_exl3_word(t0, a1);
    pre[2] = tf_exl3_word(t0, a2);
  }
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float accb[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint bb = b0; bb < b1; ++bb) {
    const uint kb = bb * 16;
    const uint w0 = pre[0], w1 = pre[1], w2 = pre[2];
    if (bb + 1 < b1) {
      const device ushort* tn = reinterpret_cast<const device ushort*>(data + ulong(bb + 1) * ks + ulong(nb / 16) * ns);
      pre[0] = tf_exl3_word(tn, wi0);
      pre[1] = tf_exl3_word(tn, a1);
      pre[2] = tf_exl3_word(tn, a2);
    }
    // The lane's window: one runtime funnel shift, then the eight windows at compile-time shifts.
    const uint c_cur = (w0 << off0) | ((w1 >> sh0) & m0);
    const uint c_next = (w1 << off0) | ((w2 >> sh0) & m0);
    // The lane's four k rows of this block, read as two halves each (the rows are adjacent in pairs).
#if defined(TF_DIAG_DECODE) && TF_DIAG_DECODE == 4
    // The FMA stays (the weight is still needed) and the constant is a RUNTIME value (the args' rows), so the
    // compiler cannot fold the multiply away and the arm isolates the read, not the arithmetic.
    const float x0 = float(a.rows), x1 = float(a.rows), x2 = float(a.rows), x3 = float(a.rows);
#else
    const device half* xr = x + ulong(row) * a.k + kb + 2 * (lane & 3u);
    const half2 xa = *(const device half2*)xr;
    const half2 xb = *(const device half2*)(xr + 8);
    const float x0 = float(xa.x), x1 = float(xa.y), x2 = float(xb.x), x3 = float(xb.y);
#endif
    tf_exl3_fma_step<S, 0, Lut>(&acc[0], c_cur, c_next, table, codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 1, Lut>(&acc[1], c_cur, c_next, table, codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 2, Lut>(&acc[2], c_cur, c_next, table, codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 3, Lut>(&acc[3], c_cur, c_next, table, codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 4, Lut>(&acc[4], c_cur, c_next, table, codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 5, Lut>(&acc[5], c_cur, c_next, table, codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 6, Lut>(&acc[6], c_cur, c_next, table, codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 7, Lut>(&acc[7], c_cur, c_next, table, codebook, x0, x1, x2, x3, lut);
    if (paired) {
      // The mate's row: the same weights, the same products, the mate's own activations.
      const device half* xm = x + ulong(mate) * a.k + kb + 2 * (lane & 3u);
      const half2 ma = *(const device half2*)xm;
      const half2 mb = *(const device half2*)(xm + 8);
      const float y0 = float(ma.x), y1 = float(ma.y), y2 = float(mb.x), y3 = float(mb.y);
      tf_exl3_fma_step<S, 0, Lut>(&accb[0], c_cur, c_next, table, codebook, y0, y1, y2, y3, lut);
      tf_exl3_fma_step<S, 1, Lut>(&accb[1], c_cur, c_next, table, codebook, y0, y1, y2, y3, lut);
      tf_exl3_fma_step<S, 2, Lut>(&accb[2], c_cur, c_next, table, codebook, y0, y1, y2, y3, lut);
      tf_exl3_fma_step<S, 3, Lut>(&accb[3], c_cur, c_next, table, codebook, y0, y1, y2, y3, lut);
      tf_exl3_fma_step<S, 4, Lut>(&accb[4], c_cur, c_next, table, codebook, y0, y1, y2, y3, lut);
      tf_exl3_fma_step<S, 5, Lut>(&accb[5], c_cur, c_next, table, codebook, y0, y1, y2, y3, lut);
      tf_exl3_fma_step<S, 6, Lut>(&accb[6], c_cur, c_next, table, codebook, y0, y1, y2, y3, lut);
      tf_exl3_fma_step<S, 7, Lut>(&accb[7], c_cur, c_next, table, codebook, y0, y1, y2, y3, lut);
    }
  }
  // The eight accumulators meet in a tree: the first four are the near column, the last four the far one. The shipped
  // tree, the shipped cross-warp sum and the shipped lane shuffles, one row a turn.
  for (uint part = 0; part < 2u; ++part) {
    const uint out_row = part == 0 ? row : mate;
    if (part == 1u and !paired) break;
    float acc0 = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    float acc1 = (acc[4] + acc[5]) + (acc[6] + acc[7]);
    if (part == 1u) {
      acc0 = (accb[0] + accb[1]) + (accb[2] + accb[3]);
      acc1 = (accb[4] + accb[5]) + (accb[6] + accb[7]);
    }
#if defined(TF_DIAG_DECODE) && TF_DIAG_DECODE == 5
#elif TF_FMA_WARPS > 1
    red_all[sg * 64u + lane] = acc0;
    red_all[sg * 64u + 32u + lane] = acc1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      acc0 = 0.0f;
      acc1 = 0.0f;
      for (uint w = 0; w < W; ++w) {
        acc0 += red_all[w * 64u + lane];
        acc1 += red_all[w * 64u + 32u + lane];
      }
    }
#endif
    acc0 += simd_shuffle_xor(acc0, 1);
    acc0 += simd_shuffle_xor(acc0, 2);
    acc1 += simd_shuffle_xor(acc1, 1);
    acc1 += simd_shuffle_xor(acc1, 2);
    if (sg == 0 and (lane & 3u) == 0u) {
      sums[ulong(out_row) * a.n + nb + lane / 4] = acc0;
      sums[ulong(out_row) * a.n + nb + 8 + lane / 4] = acc1;
    }
#if TF_FMA_WARPS > 1
    // The mate's turn reuses the partials: every thread reaches this barrier.
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
  }
}


// The paired routed decode (the resident table): the entries that name the same expert are computed in one
// threadgroup, one weight decode for the pair.
kernel void tf_exl3_matmul_decode_fma_static_pair(const device half* x [[buffer(0)]],
                             const device uchar* trellis [[buffer(1)]],
                             device float* sums [[buffer(2)]],
                             constant TfExl3Args& a [[buffer(3)]],
                             const device uint* routes [[buffer(4)]],
                             const device TfExl3Expert* descriptors [[buffer(5)]],
                             device float* partials [[buffer(6)]],
#if defined(TF_DECODE_TABLE)
                             const device half* table [[buffer(7)]],
#endif
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint3 tptg [[threads_per_threadgroup]],
                             uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float red[TF_FMA_WARPS][64];
  threadgroup half lut[1024];
  const uint row = tg.y;
  if (row >= a.rows) return;
  const uint expert = routes[row];
  if (expert >= a.experts) return;
  const uint halves = descriptors[expert].halves;
  // The mul1 codebook's values depend only on the state's byte sum, so 1,024 entries hold every value; the LUT's
  // values are the resident table's, entry for entry, so the output is the table's.
  const bool mul1 = descriptors[expert].codebook == 2u;
  if (mul1) {
    const float scale = float(as_type<half>(ushort(0x1EEEu)));
    const float bias = float(as_type<half>(ushort(0xC931u)));
    for (uint b = tid; b < 1024u; b += tptg.x) lut[b] = half(fma(float(b + 1024u), scale, bias));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (halves == 6u)
    tf_exl3_decode_fma_static_pair_body<3, false>(x, trellis, sums, a, routes, descriptors,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  else if (halves == 10u)
    tf_exl3_decode_fma_static_pair_body<5, false>(x, trellis, sums, a, routes, descriptors,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  else if (halves == 12u)
    tf_exl3_decode_fma_static_pair_body<6, false>(x, trellis, sums, a, routes, descriptors,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  else if (halves == 8u)
    tf_exl3_decode_fma_static_pair_body<4, false>(x, trellis, sums, a, routes, descriptors,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  else if (halves == 4u)
    tf_exl3_decode_fma_static_pair_body<2, false>(x, trellis, sums, a, routes, descriptors,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  else if (halves == 2u)
    tf_exl3_decode_fma_static_pair_body<1, false>(x, trellis, sums, a, routes, descriptors,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
}

// The paired routed decode (with the codebook LUT): the entries that name the same expert are computed in one
// threadgroup, one weight decode for the pair.
kernel void tf_exl3_matmul_decode_fma_static_pair_lut(const device half* x [[buffer(0)]],
                             const device uchar* trellis [[buffer(1)]],
                             device float* sums [[buffer(2)]],
                             constant TfExl3Args& a [[buffer(3)]],
                             const device uint* routes [[buffer(4)]],
                             const device TfExl3Expert* descriptors [[buffer(5)]],
                             device float* partials [[buffer(6)]],
#if defined(TF_DECODE_TABLE)
                             const device half* table [[buffer(7)]],
#endif
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint3 tptg [[threads_per_threadgroup]],
                             uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float red[TF_FMA_WARPS][64];
  threadgroup half lut[1024];
  const uint row = tg.y;
  if (row >= a.rows) return;
  const uint expert = routes[row];
  if (expert >= a.experts) return;
  const uint halves = descriptors[expert].halves;
  // The mul1 codebook's values depend only on the state's byte sum, so 1,024 entries hold every value; the LUT's
  // values are the resident table's, entry for entry, so the output is the table's.
  const bool mul1 = descriptors[expert].codebook == 2u;
  if (mul1) {
    const float scale = float(as_type<half>(ushort(0x1EEEu)));
    const float bias = float(as_type<half>(ushort(0xC931u)));
    for (uint b = tid; b < 1024u; b += tptg.x) lut[b] = half(fma(float(b + 1024u), scale, bias));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (halves == 6u)
    tf_exl3_decode_fma_static_pair_body<3, true>(x, trellis, sums, a, routes, descriptors,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0], lut);
  else if (halves == 10u)
    tf_exl3_decode_fma_static_pair_body<5, true>(x, trellis, sums, a, routes, descriptors,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0], lut);
  else if (halves == 12u)
    tf_exl3_decode_fma_static_pair_body<6, true>(x, trellis, sums, a, routes, descriptors,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0], lut);
  else if (halves == 8u)
    tf_exl3_decode_fma_static_pair_body<4, true>(x, trellis, sums, a, routes, descriptors,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0], lut);
  else if (halves == 4u)
    tf_exl3_decode_fma_static_pair_body<2, true>(x, trellis, sums, a, routes, descriptors,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0], lut);
  else if (halves == 2u)
    tf_exl3_decode_fma_static_pair_body<1, true>(x, trellis, sums, a, routes, descriptors,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0], lut);
}
// The width is the expert descriptor's (the routed pack's is 6 halves), so the body is specialised on it at dispatch.
kernel void tf_exl3_matmul_decode_fma_static(const device half* x [[buffer(0)]],
                                              const device uchar* trellis [[buffer(1)]],
                                              device float* sums [[buffer(2)]],
                                              constant TfExl3Args& a [[buffer(3)]],
                                              const device uint* routes [[buffer(4)]],
                                              const device TfExl3Expert* descriptors [[buffer(5)]],
                                              device float* partials [[buffer(6)]],
#if defined(TF_DECODE_TABLE)
                                              const device half* table [[buffer(7)]],
#endif
                                              uint lane [[thread_index_in_simdgroup]],
                                              uint sg [[simdgroup_index_in_threadgroup]],
                                              uint tid [[thread_index_in_threadgroup]],
                                              uint3 tptg [[threads_per_threadgroup]],
                                              uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float red[TF_FMA_WARPS][64];
  const uint row = tg.y;
  if (row >= a.rows) return;
  const uint expert = routes[row];
  if (expert >= a.experts) return;
  const uint halves = descriptors[expert].halves;
  if (halves == 6u)
    tf_exl3_decode_fma_static_body<false, 3, false, (TF_FMA_TWO != 0)>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  else if (halves == 10u)
    tf_exl3_decode_fma_static_body<false, 5>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  else if (halves == 12u)
    tf_exl3_decode_fma_static_body<false, 6>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  else if (halves == 8u)
    tf_exl3_decode_fma_static_body<false, 4>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  else if (halves == 4u)
    tf_exl3_decode_fma_static_body<false, 2>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
  else if (halves == 2u)
    tf_exl3_decode_fma_static_body<false, 1>(x, trellis, sums, a, routes, descriptors, partials,
#if defined(TF_DECODE_TABLE)
      table,
#else
      nullptr,
#endif
      lane, sg, tg, &red[0][0]);
}
// THE ROUTED PREFILL GEOMETRY. BM 128, BN 64, BK 64: eight simdgroups (WM 4 x WN 2), each owning a 32 x 32 output
// tile; the eight simdgroups cooperatively decode the whole 64 x 64 weight stage (two 16x16 trellis tiles each) into
// one row-major B tile, and every simdgroup multiplies the shared stage. Per output the k order and the MMA order are
// 's, so `sums` is bit-identical; only the stage depth, the warp tile and the cooperative decode differ. The stage is
// DOUBLE-BUFFERED and the next stage's decode is issued before the current stage's multiply, so the codebook ALU and
// the stream loads overlap the tensor units (the deeper stage, our overlap).
template <bool UseTable>
inline void tf_exl3_cmp_decode_stage(threadgroup half* wx, const device uchar* data, ulong ks, ulong ns,
                                     uint nb, uint s, uint sg, uint wi0, uint words, uint b0, uint halves,
                                     uint codebook, const device half* table,
                                     const uint* k_l, const uint* c_l, const uint* off, const uint* sh, const uint* rel) {
  const uint tiles_t = sg * 2u;
  TF_UNROLL
  for (uint j = 0; j < 2u; ++j) {
    const uint t = tiles_t + j;
    const uint nt = t & 3u, kt = t >> 2u;
    uint at[8];
    TF_UNROLL
    for (uint k = 0; k < 8u; ++k) at[k] = (kt * 16u + k_l[k]) * 72u + nt * 16u + c_l[k];
    const uint kb = s + kt * 16u;
    const uint w0 = tf_exl3_block_word(data, ks, ns, kb, nb, 0, nt, wi0, words);
    const uint w1 = tf_exl3_block_word(data, ks, ns, kb, nb, 0, nt, wi0 + 1u, words);
    if (UseTable and halves == 6u) {
      // The lane's eight 3-bit windows are 37 bits, so two words hold the run (the funnel path).
      tf_exl3_decode_funnel<8>(wx, table, nullptr, codebook, at, b0, halves >> 1, w0, w1);
    } else if (UseTable) {
      const uint w2 = tf_exl3_block_word(data, ks, ns, kb, nb, 0, nt, wi0 + 2u, words);
      const uint w3 = tf_exl3_block_word(data, ks, ns, kb, nb, 0, nt, wi0 + 3u, words);
      tf_exl3_decode_table<8>(wx, table, codebook, at, off, sh, rel, w0, w1, w2, w3);
    } else {
      const uint w2 = tf_exl3_block_word(data, ks, ns, kb, nb, 0, nt, wi0 + 2u, words);
      const uint w3 = tf_exl3_block_word(data, ks, ns, kb, nb, 0, nt, wi0 + 3u, words);
      tf_exl3_decode_fast<8>(wx, codebook, at, off, sh, rel, w0, w1, w2, w3);
    }
  }
}

template <bool UseTable>
inline void tf_exl3_prefill_body_cmp(threadgroup half* wx, const device half* x, const device uchar* trellis,
                                     device float* sums, constant TfExl3Args& a, const device uint* routes,
                                     const device TfExl3Expert* descriptors, const device half* table,
                                     uint lane, uint sg, uint tid, uint3 tg) {
  constexpr uint LDB = 72u;
  constexpr uint STAGE = 64u * 72u;
  const uint trows = 128u;
  const uint rb = tg.y * trows, nb = tg.x * 64u;
  if (a.indexed == 1 || rb >= a.rows || nb >= a.n) return;
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
  const uint cols_here = min(64u, a.n - nb);
  const uint tm = 32u * (sg >> 1u), tn = 32u * (sg & 1u);
  // This lane's eight consecutive states of one 16x16 tile; the same map the shipped body uses.
  const uint p0 = lane * 8u;
  const uint m0 = tf_exl3_map_for(p0, halves);
  const uint wi0 = m0 & 63u;
  const uint b0 = (m0 >> 6) & 31u;
  const uint words = 4u * halves;
  const uint ring = 128u * halves;
  const uint cur0 = (wi0 << 5) | b0;
  uint k_l[8], c_l[8], off[8], sh[8], rel[8];
  TF_UNROLL
  for (uint k = 0; k < 8u; ++k) {
    const uint p = p0 + k, l = p / 8u, e = p % 8u;
    // The pack's per-state map: k_local is the 64-stride axis, c_local the +1 axis.
    k_l[k] = 2u * (l % 4u) + (e & 1u) + 8u * ((e >> 1u) & 1u);
    c_l[k] = l / 4u + 8u * (e >> 2u);
    uint c8 = cur0 + k * (halves >> 1);
    if (c8 >= ring) c8 -= ring;
    const uint wi = c8 >> 5;
    off[k] = c8 & 31u;
    sh[k] = (32u - off[k]) & 31u;
    rel[k] = wi >= wi0 ? wi - wi0 : wi + words - wi0;
  }
  // The padded-row skip at the 16-row group (/'s form).
  bool lo_live = true, hi_live = true;
  if (a.indexed != 0) {
    const uint g = rb + tm + lane;
    const bool gv = g < a.rows and routes[g] != 0xffffffffu;
    lo_live = simd_any(gv and lane < 16u);
    hi_live = simd_any(gv and lane >= 16u);
  }
  frag<float> lo0 = frag<float>(0), hi0 = frag<float>(0);
  frag<float> lo1 = frag<float>(0), hi1 = frag<float>(0);
  // Stage 0 into buffer 0, then one barrier before the first multiply.
  tf_exl3_cmp_decode_stage<UseTable>(wx, data, ks, ns, nb, 0u, sg, wi0, words, b0, halves, codebook, table,
                                     k_l, c_l, off, sh, rel);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint nst = 0u;
  for (uint s = 0; s + 64u <= a.k; s += 64u, ++nst) {
    const threadgroup half* bs = wx + (nst & 1u) * STAGE;
    if (lo_live or hi_live) {
      TF_UNROLL
      for (uint i = 0; i < 2u; ++i) {
        frag<half> av0, av1;
        if (lo_live) tf_frag_get_dev4(av0, x, int(a.k), int(rb + tm), int(s + i * 32u), home, int(rb + rows_here));
        if (hi_live) tf_frag_get_dev4(av1, x, int(a.k), int(rb + tm + 16u), int(s + i * 32u), home, int(rb + rows_here));
        TF_UNROLL
        for (uint kk = 0; kk < 2u; ++kk) {
          const int kc = int(i * 32u + kk * 16u);
          frag<half> bb0, bb1;
          tf_frag_get_tg4(bb0, bs, int(LDB), kc, int(tn), home);
          tf_frag_get_tg4(bb1, bs, int(LDB), kc, int(tn + 16u), home);
          if (lo_live) mma_16x32<false, false>(lo0, hi0, av0, bb0, bb1);
          if (hi_live) mma_16x32<false, false>(lo1, hi1, av1, bb0, bb1);
        }
      }
    }
    if (s + 64u < a.k) {
      tf_exl3_cmp_decode_stage<UseTable>(wx + ((nst & 1u) ^ 1u) * STAGE, data, ks, ns, nb, s + 64u, sg,
                                         wi0, words, b0, halves, codebook, table, k_l, c_l, off, sh, rel);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const int r0 = int(rb + tm), c0 = int(nb + tn);
  frag_put_in(lo0, sums, int(a.n), r0, c0, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi0, sums, int(a.n), r0, c0 + 16, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo1, sums, int(a.n), r0 + 16, c0, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi1, sums, int(a.n), r0 + 16, c0 + 16, home, int(rb + rows_here), int(nb + cols_here));
}

kernel void tf_exl3_matmul_prefill_nax84(const device half* const x [[buffer(0)]],
                                        const device uchar* const trellis [[buffer(1)]],
                                        device float* const sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device uint* const routes [[buffer(4)]],
                                        const device TfExl3Expert* const descriptors [[buffer(5)]],
                                        const device half* const table [[buffer(7)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tptg [[threads_per_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 64 * 72];
  tf_exl3_prefill_body_cmp<true>(wx, x, trellis, sums, a, routes, descriptors, table, lane, sg, tid, tg);
}


// They are never selected by the engine (no dispatcher references them). The sink index is tg.x * 128 + tg.y * 8192 +
// tid, so every arm writes a distinct word and no load is dead.
// ---------------------------------------------------------------------------------------------------------------
kernel void tf_exl3_stream_read_nax90(const device half* x [[buffer(0)]],
                                     const device uchar* trellis [[buffer(1)]],
                                     device uint* sink [[buffer(2)]],
                                     constant TfExl3Args& a [[buffer(3)]],
                                     const device uint* routes [[buffer(4)]],
                                     const device TfExl3Expert* descriptors [[buffer(5)]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint3 tptg [[threads_per_threadgroup]],
                                     uint3 tg [[threadgroup_position_in_grid]]) {
  const uint rb = tg.y * 64u, nb = tg.x * 64u;
  if (rb >= a.rows) return;
  const uint e = routes[rb];
  if (e >= a.experts) return;
  const device uchar* data = descriptors[e].trellis;
  const ulong ks = descriptors[e].tile_k_stride, ns = descriptors[e].tile_n_stride;
  const uint halves = descriptors[e].halves, words = 4u * halves;
  const uint h = sg & 3u;
  const device ushort* t0 = reinterpret_cast<const device ushort*>(data + ulong(nb / 16u + h) * ns);
  uint acc = 0u;
  uint cur = (lane < words) ? tf_exl3_word(t0, lane) : 0u;
  for (uint kb = 0; kb < a.k; kb += 16u) {
    const uint nkb = kb + 16u;
    uint nxt = 0u;
    if (nkb < a.k and lane < words) {
      const device ushort* tn = reinterpret_cast<const device ushort*>(data + ulong(nkb / 16u) * ks + ulong(nb / 16u + h) * ns);
      nxt = tf_exl3_word(tn, lane);
    }
    acc ^= cur;
    cur = nxt;
  }
  sink[tg.x * 128u + tg.y * 8192u + tid] = acc;
}

kernel void tf_exl3_stream_read_nax91(const device half* x [[buffer(0)]],
                                     const device uchar* trellis [[buffer(1)]],
                                     device uint* sink [[buffer(2)]],
                                     constant TfExl3Args& a [[buffer(3)]],
                                     const device uint* routes [[buffer(4)]],
                                     const device TfExl3Expert* descriptors [[buffer(5)]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint3 tptg [[threads_per_threadgroup]],
                                     uint3 tg [[threadgroup_position_in_grid]]) {
  const uint rb = tg.y * 64u, nb = tg.x * 64u;
  if (rb >= a.rows) return;
  const uint e = routes[rb];
  if (e >= a.experts) return;
  const device uchar* data = descriptors[e].trellis;
  const ulong ks = descriptors[e].tile_k_stride, ns = descriptors[e].tile_n_stride;
  const uint halves = descriptors[e].halves, words = 4u * halves;
  const uint h = sg & 3u;
  const uint nwords = words / 4u;
  const device ushort* t0 = reinterpret_cast<const device ushort*>(data + ulong(nb / 16u + h) * ns);
  uint acc = 0u;
  uint4 cur = uint4(0u, 0u, 0u, 0u);
  if (lane < nwords) cur = reinterpret_cast<const device uint4*>(t0)[lane];
  for (uint kb = 0; kb < a.k; kb += 16u) {
    const uint nkb = kb + 16u;
    uint4 nxt = uint4(0u, 0u, 0u, 0u);
    if (nkb < a.k and lane < nwords) {
      const device ushort* tn = reinterpret_cast<const device ushort*>(data + ulong(nkb / 16u) * ks + ulong(nb / 16u + h) * ns);
      nxt = reinterpret_cast<const device uint4*>(tn)[lane];
    }
    acc ^= cur.x ^ cur.y ^ cur.z ^ cur.w;
    cur = nxt;
  }
  sink[tg.x * 128u + tg.y * 8192u + tid] = acc;
}

// The shipped per-lane cursor: two 32-bit words a lane, eight states a lane (the funnel's read count).
kernel void tf_exl3_stream_read_nax92(const device half* x [[buffer(0)]],
                                     const device uchar* trellis [[buffer(1)]],
                                     device uint* sink [[buffer(2)]],
                                     constant TfExl3Args& a [[buffer(3)]],
                                     const device uint* routes [[buffer(4)]],
                                     const device TfExl3Expert* descriptors [[buffer(5)]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint3 tptg [[threads_per_threadgroup]],
                                     uint3 tg [[threadgroup_position_in_grid]]) {
  const uint rb = tg.y * 64u, nb = tg.x * 64u;
  if (rb >= a.rows) return;
  const uint e = routes[rb];
  if (e >= a.experts) return;
  const device uchar* data = descriptors[e].trellis;
  const ulong ks = descriptors[e].tile_k_stride, ns = descriptors[e].tile_n_stride;
  const uint halves = descriptors[e].halves, words = 4u * halves;
  const uint h = sg & 3u;
  const uint p0 = (tid % 32u) * 8u;
  const uint wi0 = tf_exl3_map_for(p0, halves) & 63u;
  const uint wi1 = (wi0 + 1u < words) ? wi0 + 1u : wi0 + 1u - words;
  const device ushort* t0 = reinterpret_cast<const device ushort*>(data + ulong(nb / 16u + h) * ns);
  uint acc = 0u;
  uint w0 = tf_exl3_word(t0, wi0), w1 = tf_exl3_word(t0, wi1);
  for (uint kb = 0; kb < a.k; kb += 16u) {
    const uint nkb = kb + 16u;
    uint n0 = 0u, n1 = 0u;
    if (nkb < a.k) {
      const device ushort* tn = reinterpret_cast<const device ushort*>(data + ulong(nkb / 16u) * ks + ulong(nb / 16u + h) * ns);
      n0 = tf_exl3_word(tn, wi0); n1 = tf_exl3_word(tn, wi1);
    }
    acc ^= w0 ^ w1;
    w0 = n0; w1 = n1;
  }
  sink[tg.x * 128u + tg.y * 8192u + tid] = acc;
}

// The whole expert slab, one threadgroup an expert: the fully sequential schedule (144 n-tiles a k block).
kernel void tf_exl3_stream_read_nax93(const device half* x [[buffer(0)]],
                                     const device uchar* trellis [[buffer(1)]],
                                     device uint* sink [[buffer(2)]],
                                     constant TfExl3Args& a [[buffer(3)]],
                                     const device uint* routes [[buffer(4)]],
                                     const device TfExl3Expert* descriptors [[buffer(5)]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint3 tptg [[threads_per_threadgroup]],
                                     uint3 tg [[threadgroup_position_in_grid]]) {
  const uint e = tg.y;
  if (e >= a.experts) return;
  const device uchar* data = descriptors[e].trellis;
  const ulong ks = descriptors[e].tile_k_stride, ns = descriptors[e].tile_n_stride;
  const uint halves = descriptors[e].halves, words = 4u * halves;
  const uint nwords = words / 4u;
  const uint base = sg * 36u;
  uint acc = 0u;
  for (uint kb = 0; kb < a.k; kb += 16u) {
    for (uint g = 0; g < 36u; g += 4u) {
      uint4 v0 = uint4(0u, 0u, 0u, 0u), v1 = uint4(0u, 0u, 0u, 0u), v2 = uint4(0u, 0u, 0u, 0u), v3 = uint4(0u, 0u, 0u, 0u);
      if (lane < nwords) {
        const ulong koff = ulong(kb / 16u) * ks;
        const ulong boff = ulong(base + g) * ns;
        v0 = reinterpret_cast<const device uint4*>(data + koff + boff)[lane];
        v1 = reinterpret_cast<const device uint4*>(data + koff + boff + ns)[lane];
        v2 = reinterpret_cast<const device uint4*>(data + koff + boff + 2u * ns)[lane];
        v3 = reinterpret_cast<const device uint4*>(data + koff + boff + 3u * ns)[lane];
      }
      acc ^= v0.x ^ v0.y ^ v0.z ^ v0.w ^ v1.x ^ v1.y ^ v1.z ^ v1.w ^
             v2.x ^ v2.y ^ v2.z ^ v2.w ^ v3.x ^ v3.y ^ v3.z ^ v3.w;
    }
  }
  sink[tg.x * 128u + tg.y * 8192u + tid] = acc;
}

// The same read with the GRID TRANSPOSED: consecutive threadgroups are experts at the same column tile.
kernel void tf_exl3_stream_read_nax94(const device half* x [[buffer(0)]],
                                     const device uchar* trellis [[buffer(1)]],
                                     device uint* sink [[buffer(2)]],
                                     constant TfExl3Args& a [[buffer(3)]],
                                     const device uint* routes [[buffer(4)]],
                                     const device TfExl3Expert* descriptors [[buffer(5)]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint3 tptg [[threads_per_threadgroup]],
                                     uint3 tg [[threadgroup_position_in_grid]]) {
  const uint rb = tg.x * 64u, nb = tg.y * 64u;
  if (rb >= a.rows) return;
  const uint e = routes[rb];
  if (e >= a.experts) return;
  const device uchar* data = descriptors[e].trellis;
  const ulong ks = descriptors[e].tile_k_stride, ns = descriptors[e].tile_n_stride;
  const uint halves = descriptors[e].halves, words = 4u * halves;
  const uint h = sg & 3u;
  const device ushort* t0 = reinterpret_cast<const device ushort*>(data + ulong(nb / 16u + h) * ns);
  uint acc = 0u;
  uint cur = (lane < words) ? tf_exl3_word(t0, lane) : 0u;
  for (uint kb = 0; kb < a.k; kb += 16u) {
    const uint nkb = kb + 16u;
    uint nxt = 0u;
    if (nkb < a.k and lane < words) {
      const device ushort* tn = reinterpret_cast<const device ushort*>(data + ulong(nkb / 16u) * ks + ulong(nb / 16u + h) * ns);
      nxt = tf_exl3_word(tn, lane);
    }
    acc ^= cur;
    cur = nxt;
  }
  sink[tg.x * 128u + tg.y * 8192u + tid] = acc;
}

// Eight warps a threadgroup, the k walk split in two halves: more independent streams per threadgroup.
kernel void tf_exl3_stream_read_nax95(const device half* x [[buffer(0)]],
                                     const device uchar* trellis [[buffer(1)]],
                                     device uint* sink [[buffer(2)]],
                                     constant TfExl3Args& a [[buffer(3)]],
                                     const device uint* routes [[buffer(4)]],
                                     const device TfExl3Expert* descriptors [[buffer(5)]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint3 tptg [[threads_per_threadgroup]],
                                     uint3 tg [[threadgroup_position_in_grid]]) {
  const uint rb = tg.y * 64u, nb = tg.x * 64u;
  if (rb >= a.rows) return;
  const uint e = routes[rb];
  if (e >= a.experts) return;
  const device uchar* data = descriptors[e].trellis;
  const ulong ks = descriptors[e].tile_k_stride, ns = descriptors[e].tile_n_stride;
  const uint halves = descriptors[e].halves, words = 4u * halves;
  const uint h = sg & 3u, kh = sg >> 2;
  const device ushort* t0 = reinterpret_cast<const device ushort*>(data + ulong(nb / 16u + h) * ns);
  uint acc = 0u;
  uint cur = (lane < words) ? tf_exl3_word(reinterpret_cast<const device ushort*>(data + ulong(kh) * ks + ulong(nb / 16u + h) * ns), lane) : 0u;
  for (uint kb = kh * 16u; kb < a.k; kb += 32u) {
    const uint nkb = kb + 32u;
    uint nxt = 0u;
    if (nkb < a.k and lane < words) {
      const device ushort* tn = reinterpret_cast<const device ushort*>(data + ulong(nkb / 16u) * ks + ulong(nb / 16u + h) * ns);
      nxt = tf_exl3_word(tn, lane);
    }
    acc ^= cur;
    cur = nxt;
  }
  sink[tg.x * 128u + tg.y * 8192u + tid] = acc;
}


// The 64-column tile's four 16-column n-tiles are CONTIGUOUS (ns = 96 B), so one 384-byte chunk a k block. This arm
// reads that chunk with 24 lanes of 16-byte loads -- the same bytes, a quarter of the requests.
kernel void tf_exl3_stream_read_nax96(const device half* x [[buffer(0)]],
                                     const device uchar* trellis [[buffer(1)]],
                                     device uint* sink [[buffer(2)]],
                                     constant TfExl3Args& a [[buffer(3)]],
                                     const device uint* routes [[buffer(4)]],
                                     const device TfExl3Expert* descriptors [[buffer(5)]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint3 tptg [[threads_per_threadgroup]],
                                     uint3 tg [[threadgroup_position_in_grid]]) {
  const uint rb = tg.y * 64u, nb = tg.x * 64u;
  if (rb >= a.rows) return;
  const uint e = routes[rb];
  if (e >= a.experts) return;
  const device uchar* data = descriptors[e].trellis;
  const ulong ks = descriptors[e].tile_k_stride, ns = descriptors[e].tile_n_stride;
  const uint blocks = a.k / 16u;
  const uint per = (blocks + 3u) / 4u;
  const uint kb0 = sg * per;
  const uint kb1 = min(kb0 + per, blocks);
  uint acc = 0u;
  uint4 cur = uint4(0u, 0u, 0u, 0u);
  if (lane < 24u and kb0 < kb1) {
    const device uchar* ch = data + ulong(kb0) * ks + ulong(nb / 16u) * ns;
    cur = reinterpret_cast<const device uint4*>(ch)[lane];
  }
  for (uint kb = kb0; kb < kb1; ++kb) {
    uint4 nxt = uint4(0u, 0u, 0u, 0u);
    if (lane < 24u and kb + 1u < kb1) {
      const device uchar* ch = data + ulong(kb + 1u) * ks + ulong(nb / 16u) * ns;
      nxt = reinterpret_cast<const device uint4*>(ch)[lane];
    }
    acc ^= cur.x ^ cur.y ^ cur.z ^ cur.w;
    cur = nxt;
  }
  sink[tg.x * 128u + tg.y * 8192u + tid] = acc;
}

kernel void tf_exl3_stream_read_nax97(const device half* x [[buffer(0)]],
                                     const device uchar* trellis [[buffer(1)]],
                                     device uint* sink [[buffer(2)]],
                                     constant TfExl3Args& a [[buffer(3)]],
                                     const device uint* routes [[buffer(4)]],
                                     const device TfExl3Expert* descriptors [[buffer(5)]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint3 tptg [[threads_per_threadgroup]],
                                     uint3 tg [[threadgroup_position_in_grid]]) {
  const uint rb = tg.y * 64u, nb = tg.x * 64u;
  if (rb >= a.rows) return;
  const uint e = routes[rb];
  if (e >= a.experts) return;
  const device uchar* data = descriptors[e].trellis;
  const ulong ks = descriptors[e].tile_k_stride, ns = descriptors[e].tile_n_stride;
  const uint blocks = a.k / 16u;
  const uint per = (blocks + 7u) / 8u;
  const uint kb0 = sg * per;
  const uint kb1 = min(kb0 + per, blocks);
  uint acc = 0u;
  uint4 cur = uint4(0u, 0u, 0u, 0u);
  if (lane < 24u and kb0 < kb1) {
    const device uchar* ch = data + ulong(kb0) * ks + ulong(nb / 16u) * ns;
    cur = reinterpret_cast<const device uint4*>(ch)[lane];
  }
  for (uint kb = kb0; kb < kb1; ++kb) {
    uint4 nxt = uint4(0u, 0u, 0u, 0u);
    if (lane < 24u and kb + 1u < kb1) {
      const device uchar* ch = data + ulong(kb + 1u) * ks + ulong(nb / 16u) * ns;
      nxt = reinterpret_cast<const device uint4*>(ch)[lane];
    }
    acc ^= cur.x ^ cur.y ^ cur.z ^ cur.w;
    cur = nxt;
  }
  sink[tg.x * 128u + (tg.y * 8192u) + tid] = acc;
}

// The expert slab with sixteen warps, each nine n-tiles a k block: nine independent 16-byte loads a lane.
kernel void tf_exl3_stream_read_nax98(const device half* x [[buffer(0)]],
                                     const device uchar* trellis [[buffer(1)]],
                                     device uint* sink [[buffer(2)]],
                                     constant TfExl3Args& a [[buffer(3)]],
                                     const device uint* routes [[buffer(4)]],
                                     const device TfExl3Expert* descriptors [[buffer(5)]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint3 tptg [[threads_per_threadgroup]],
                                     uint3 tg [[threadgroup_position_in_grid]]) {
  const uint e = tg.y;
  if (e >= a.experts) return;
  const device uchar* data = descriptors[e].trellis;
  const ulong ks = descriptors[e].tile_k_stride, ns = descriptors[e].tile_n_stride;
  const uint base = sg * 9u;
  uint acc = 0u;
  for (uint kb = 0; kb < a.k; kb += 16u) {
    const ulong koff = ulong(kb / 16u) * ks;
    uint4 v0 = uint4(0u, 0u, 0u, 0u), v1 = uint4(0u, 0u, 0u, 0u), v2 = uint4(0u, 0u, 0u, 0u);
    uint4 v3 = uint4(0u, 0u, 0u, 0u), v4 = uint4(0u, 0u, 0u, 0u), v5 = uint4(0u, 0u, 0u, 0u);
    uint4 v6 = uint4(0u, 0u, 0u, 0u), v7 = uint4(0u, 0u, 0u, 0u), v8 = uint4(0u, 0u, 0u, 0u);
    if (lane < 6u) {
      v0 = reinterpret_cast<const device uint4*>(data + koff + ulong(base) * ns)[lane];
      v1 = reinterpret_cast<const device uint4*>(data + koff + ulong(base + 1u) * ns)[lane];
      v2 = reinterpret_cast<const device uint4*>(data + koff + ulong(base + 2u) * ns)[lane];
      v3 = reinterpret_cast<const device uint4*>(data + koff + ulong(base + 3u) * ns)[lane];
      v4 = reinterpret_cast<const device uint4*>(data + koff + ulong(base + 4u) * ns)[lane];
      v5 = reinterpret_cast<const device uint4*>(data + koff + ulong(base + 5u) * ns)[lane];
      v6 = reinterpret_cast<const device uint4*>(data + koff + ulong(base + 6u) * ns)[lane];
      v7 = reinterpret_cast<const device uint4*>(data + koff + ulong(base + 7u) * ns)[lane];
      v8 = reinterpret_cast<const device uint4*>(data + koff + ulong(base + 8u) * ns)[lane];
    }
    acc ^= v0.x ^ v0.y ^ v0.z ^ v0.w ^ v1.x ^ v1.y ^ v1.z ^ v1.w ^ v2.x ^ v2.y ^ v2.z ^ v2.w ^
           v3.x ^ v3.y ^ v3.z ^ v3.w ^ v4.x ^ v4.y ^ v4.z ^ v4.w ^ v5.x ^ v5.y ^ v5.z ^ v5.w ^
           v6.x ^ v6.y ^ v6.z ^ v6.w ^ v7.x ^ v7.y ^ v7.z ^ v7.w ^ v8.x ^ v8.y ^ v8.z ^ v8.w;
  }
  sink[tg.x * 128u + tg.y * 8192u + tid] = acc;
}


// The absolute ceiling control -- 1.716 GB read LINEARLY, one threadgroup a 122,880-byte contiguous range, four
// independent 16-byte loads a thread a wave. If this does not reach the box's peak, the peak is not available to a
// cold stream of this size; if it does, the scattered slabs' pattern is the difference.
kernel void tf_exl3_stream_read_nax99(const device half* x [[buffer(0)]],
                                     const device uchar* trellis [[buffer(1)]],
                                     device uint* sink [[buffer(2)]],
                                     constant TfExl3Args& a [[buffer(3)]],
                                     const device uint* routes [[buffer(4)]],
                                     const device TfExl3Expert* descriptors [[buffer(5)]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint3 tptg [[threads_per_threadgroup]],
                                     uint3 tg [[threadgroup_position_in_grid]]) {
  const ulong base = (ulong(tg.y) * 36u + ulong(tg.x)) * 122880u;
  const ulong stride = ulong(tptg.x) * 64u;
  uint acc = 0u;
  for (ulong off = ulong(tid) * 64u; off < 122880u; off += stride) {
    const uint4 v0 = reinterpret_cast<const device uint4*>(trellis + base + off)[0];
    const uint4 v1 = reinterpret_cast<const device uint4*>(trellis + base + off)[1];
    const uint4 v2 = reinterpret_cast<const device uint4*>(trellis + base + off)[2];
    const uint4 v3 = reinterpret_cast<const device uint4*>(trellis + base + off)[3];
    acc ^= v0.x ^ v0.y ^ v0.z ^ v0.w ^ v1.x ^ v1.y ^ v1.z ^ v1.w ^
           v2.x ^ v2.y ^ v2.z ^ v2.w ^ v3.x ^ v3.y ^ v3.z ^ v3.w;
  }
  sink[tg.x * 128u + tg.y * 8192u + tid] = acc;
}

// The weight tensor is repacked to one byte a weight, in the kernel's own 16-k x 64-n block order ([k][n] inside a
// block), so the decode is one 4-byte load a thread the e4m3 -> half bit expansion, no codebook, no table, no
// per-state shift. This arm isolates the RATE at the attention shapes; per-row scales are not applied here.
inline void tf_fp8_expand8(threadgroup half* wx, uint tid, uint2 q) {
  const half2 k256 = half2(256.0h, 256.0h);
  const uint u0 = q.x, u1 = q.y;
  const half2 a0 = as_type<half2>(((u0 & 0x007F007Fu) << 7) | ((u0 << 8) & 0x80008000u));
  const half2 b0 = as_type<half2>(((u0 >> 1) & 0x3F803F80u) | (u0 & 0x80008000u));
  const half2 a1 = as_type<half2>(((u1 & 0x007F007Fu) << 7) | ((u1 << 8) & 0x80008000u));
  const half2 b1 = as_type<half2>(((u1 >> 1) & 0x3F803F80u) | (u1 & 0x80008000u));
  threadgroup half2* o = (threadgroup half2*)(wx + 8u * tid);
  o[0] = half2(a0.x, b0.x) * k256;
  o[1] = half2(a0.y, b0.y) * k256;
  o[2] = half2(a1.x, b1.x) * k256;
  o[3] = half2(a1.y, b1.y) * k256;
}

inline void tf_i8_expand4(threadgroup half* wx, uint tid, uint u) {
  const int b0 = int(u << 24) >> 24, b1 = int(u << 16) >> 24, b2 = int(u << 8) >> 24, b3 = int(u) >> 24;
  threadgroup half2* o = (threadgroup half2*)(wx + 4u * tid);
  o[0] = half2(half(float(b0)), half(float(b1)));
  o[1] = half2(half(float(b2)), half(float(b3)));
}

inline void tf_i8_expand8(threadgroup half* wx, uint tid, uint2 q) {
  const uint u0 = q.x, u1 = q.y;
  const int a0 = int(u0 << 24) >> 24, a1 = int(u0 << 16) >> 24, a2 = int(u0 << 8) >> 24, a3 = int(u0) >> 24;
  const int b0 = int(u1 << 24) >> 24, b1 = int(u1 << 16) >> 24, b2 = int(u1 << 8) >> 24, b3 = int(u1) >> 24;
  threadgroup half2* o = (threadgroup half2*)(wx + 8u * tid);
  o[0] = half2(half(float(a0)), half(float(a1)));
  o[1] = half2(half(float(a2)), half(float(a3)));
  o[2] = half2(half(float(b0)), half(float(b1)));
  o[3] = half2(half(float(b2)), half(float(b3)));
}

inline void tf_fp8_expand4(threadgroup half* wx, uint tid, uint u) {
  const half2 a = as_type<half2>(((u & 0x007F007Fu) << 7) | ((u << 8) & 0x80008000u));
  const half2 b = as_type<half2>(((u >> 1) & 0x3F803F80u) | (u & 0x80008000u));
  const half2 k256 = half2(256.0h, 256.0h);
  threadgroup half2* o = (threadgroup half2*)(wx + 4u * tid);
  o[0] = half2(a.x, b.x) * k256;
  o[1] = half2(a.y, b.y) * k256;
}

template <bool WIDE, typename O>
inline void tf_exl3_prefill_body_fp8(threadgroup half* wx, const device half* x, const device uchar* w8,
                                     const device float* scales, device O* sums, constant TfExl3Args& a,
                                     uint lane, uint sg, uint tid, uint3 tg) {
  constexpr uint RG = 8u, CG = 1u;
  const uint trows = RG * 16u;
  const uint rb = tg.y * trows, nb = tg.x * 64u;
  if (rb >= a.rows) return;
  const uint rows_here = min(trows, a.rows - rb);
  const uint cols_here = min(64u, a.n - nb);
  const short2 home = frag_home(ushort(lane));
  const uint nb64 = a.n >> 6, block = nb >> 6;
  const uint sgr = sg % RG, sgc = sg / RG;
  frag<float> lo0, hi0, lo1, hi1;
  lo0 = frag<float>(0); hi0 = frag<float>(0); lo1 = frag<float>(0); hi1 = frag<float>(0);
  if (WIDE) {
    if (tid < 128u) {
      const uint2 q = *reinterpret_cast<const device uint2*>(w8 + ((block) << 10) + 8u * tid);
      if (a.halves == 0u) tf_i8_expand8(wx, tid, q); else tf_fp8_expand8(wx, tid, q);
    }
  } else if (tid < 256u) {
    const uint u = *reinterpret_cast<const device uint*>(w8 + ((block) << 10) + 4u * tid);
    if (a.halves == 0u) tf_i8_expand4(wx, tid, u); else tf_fp8_expand4(wx, tid, u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint cur_buf = 0;
  for (uint kb = 0; kb < a.k; kb += 16u) {
    const uint nkb = kb + 16u;
    uint nu = 0u;
    uint2 nq = uint2(0u);
    if (nkb < a.k) {
      if (WIDE) {
        if (tid < 128u) nq = *reinterpret_cast<const device uint2*>(w8 + (((nkb >> 4) * nb64 + block) << 10) + 8u * tid);
      } else if (tid < 256u) {
        nu = *reinterpret_cast<const device uint*>(w8 + (((nkb >> 4) * nb64 + block) << 10) + 4u * tid);
      }
    }
    frag<half> av;
    tf_frag_get_dev4(av, x, int(a.x_stride), int(rb + sgr * 16u), int(kb), home, int(rb + rows_here));
    threadgroup half* cur = wx + cur_buf * 1024u;
    frag<half> b0, b1;
    const int wc = int(sgc * 64u);
    tf_frag_get_tg4(b0, cur, 64, 0, wc, home);
    tf_frag_get_tg4(b1, cur, 64, 0, wc + 16, home);
    mma_16x32<false, false>(lo0, hi0, av, b0, b1);
    tf_frag_get_tg4(b0, cur, 64, 0, wc + 32, home);
    tf_frag_get_tg4(b1, cur, 64, 0, wc + 48, home);
    mma_16x32<false, false>(lo1, hi1, av, b0, b1);
    if (nkb < a.k) {
      if (WIDE) {
        if (tid < 128u) {
          if (a.halves == 0u) tf_i8_expand8(wx + (cur_buf ^ 1u) * 1024u, tid, nq);
          else tf_fp8_expand8(wx + (cur_buf ^ 1u) * 1024u, tid, nq);
        }
      } else if (tid < 256u) {
        if (a.halves == 0u) tf_i8_expand4(wx + (cur_buf ^ 1u) * 1024u, tid, nu);
        else tf_fp8_expand4(wx + (cur_buf ^ 1u) * 1024u, tid, nu);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    cur_buf ^= 1u;
  }
  const int c0 = int(nb + sgc * 64u);
  const int r0 = int(rb + sgr * 16u);
  // A bf16 destination is the caller's final output (the pre-rotated pack has no epilogue transform), so it carries
  // the call's y stride; an f32 destination is the internal sums tile and is contiguous at n. sizeof(float) == 4
  // separates the two instantiations.
  const int out_ld = (sizeof(O) == 4) ? int(a.n) : int(a.y_stride);
  if (scales != nullptr) {
    // The per-output-row E4M3 scale. One multiply an output, applied to the finished accumulator, so the MMA ran on
    // the codes alone and the row scale stays exact.
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      lo0[e] *= scales[c0 + home.x + (e & 3)];
      hi0[e] *= scales[c0 + 16 + home.x + (e & 3)];
      lo1[e] *= scales[c0 + 32 + home.x + (e & 3)];
      hi1[e] *= scales[c0 + 48 + home.x + (e & 3)];
    }
  }
  frag_put_in(lo0, sums, out_ld, r0, c0, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi0, sums, out_ld, r0, c0 + 16, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo1, sums, out_ld, r0, c0 + 32, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi1, sums, out_ld, r0, c0 + 48, home, int(rb + rows_here), int(nb + cols_here));
}

// THE BLOCK-SCALED INT8 BODY for the pre-rotated pack. W_target's columns are quantized with one int8 grid per 128-k
// block (scales[kb * n + col], f32). The block scale is folded into the half weight at the expand step (the exact
// half product of the int8 code and the half scale), so the MMA loop keeps the shipped body's single accumulator and
// geometry.
inline void tf_i8_expand4s(threadgroup half* wx, uint tid, uint u, half2 s01, half2 s23) {
  const int b0 = int(u << 24) >> 24, b1 = int(u << 16) >> 24, b2 = int(u << 8) >> 24, b3 = int(u) >> 24;
  threadgroup half2* o = (threadgroup half2*)(wx + 4u * tid);
  o[0] = half2(half(float(b0)), half(float(b1))) * s01;
  o[1] = half2(half(float(b2)), half(float(b3))) * s23;
}

inline void tf_i8_expand8s(threadgroup half* wx, uint tid, uint2 q, half2 s01, half2 s23, half2 s45, half2 s67) {
  const uint u0 = q.x, u1 = q.y;
  const int a0 = int(u0 << 24) >> 24, a1 = int(u0 << 16) >> 24, a2 = int(u0 << 8) >> 24, a3 = int(u0) >> 24;
  const int b0 = int(u1 << 24) >> 24, b1 = int(u1 << 16) >> 24, b2 = int(u1 << 8) >> 24, b3 = int(u1) >> 24;
  threadgroup half2* o = (threadgroup half2*)(wx + 8u * tid);
  o[0] = half2(half(float(a0)), half(float(a1))) * s01;
  o[1] = half2(half(float(a2)), half(float(a3))) * s23;
  o[2] = half2(half(float(b0)), half(float(b1))) * s45;
  o[3] = half2(half(float(b2)), half(float(b3))) * s67;
}

template <bool WIDE>
inline void tf_exl3_prefill_body_i8b(threadgroup half* wx, const device half* x, const device uchar* w8,
                                     const device float* bscales, device float* sums, constant TfExl3Args& a,
                                     uint lane, uint sg, uint tid, uint3 tg) {
  constexpr uint RG = 8u;
  const uint trows = RG * 16u;
  const uint rb = tg.y * trows, nb = tg.x * 64u;
  if (rb >= a.rows) return;
  const uint rows_here = min(trows, a.rows - rb);
  const uint cols_here = min(64u, a.n - nb);
  const short2 home = frag_home(ushort(lane));
  const uint nb64 = a.n >> 6, block = nb >> 6;
  const uint sgr = sg % RG, sgc = sg / RG;
  // The four (narrow) or eight (wide) output columns this thread's packed bytes cover, constant across k.
  const uint n0 = WIDE ? ((tid & 7u) * 8u) : ((tid & 15u) * 4u);
  half2 s01, s23, s45, s67;
  const device float* bs0 = bscales + nb;
  s01 = half2(half(bs0[n0]), half(bs0[n0 + 1u]));
  s23 = half2(half(bs0[n0 + 2u]), half(bs0[n0 + 3u]));
  s45 = half2(half(bs0[n0 + 4u]), half(bs0[n0 + 5u]));
  s67 = half2(half(bs0[n0 + 6u]), half(bs0[n0 + 7u]));
  frag<float> lo0, hi0, lo1, hi1;
  lo0 = frag<float>(0); hi0 = frag<float>(0); lo1 = frag<float>(0); hi1 = frag<float>(0);
  if (WIDE) {
    if (tid < 128u) {
      const uint2 q = *reinterpret_cast<const device uint2*>(w8 + ((block) << 10) + 8u * tid);
      tf_i8_expand8s(wx, tid, q, s01, s23, s45, s67);
    }
  } else if (tid < 256u) {
    const uint u = *reinterpret_cast<const device uint*>(w8 + ((block) << 10) + 4u * tid);
    tf_i8_expand4s(wx, tid, u, s01, s23);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint cur_buf = 0;
  for (uint kb = 0; kb < a.k; kb += 16u) {
    const uint nkb = kb + 16u;
    uint nu = 0u;
    uint2 nq = uint2(0u);
    if (nkb < a.k) {
      if (WIDE) {
        if (tid < 128u) nq = *reinterpret_cast<const device uint2*>(w8 + (((nkb >> 4) * nb64 + block) << 10) + 8u * tid);
      } else if (tid < 256u) {
        nu = *reinterpret_cast<const device uint*>(w8 + (((nkb >> 4) * nb64 + block) << 10) + 4u * tid);
      }
    }
    frag<half> av;
    tf_frag_get_dev4(av, x, int(a.k), int(rb + sgr * 16u), int(kb), home, int(rb + rows_here));
    threadgroup half* cur = wx + cur_buf * 1024u;
    frag<half> b0, b1;
    const int wc = int(sgc * 64u);
    tf_frag_get_tg4(b0, cur, 64, 0, wc, home);
    tf_frag_get_tg4(b1, cur, 64, 0, wc + 16, home);
    mma_16x32<false, false>(lo0, hi0, av, b0, b1);
    tf_frag_get_tg4(b0, cur, 64, 0, wc + 32, home);
    tf_frag_get_tg4(b1, cur, 64, 0, wc + 48, home);
    mma_16x32<false, false>(lo1, hi1, av, b0, b1);
    if (nkb < a.k) {
      // One scale load per 128-k block, reused by its eight 16-k expands.
      if ((nkb & 127u) == 0u) {
        const device float* bs = bscales + ulong(nkb >> 7) * a.n + nb;
        s01 = half2(half(bs[n0]), half(bs[n0 + 1u]));
        s23 = half2(half(bs[n0 + 2u]), half(bs[n0 + 3u]));
        s45 = half2(half(bs[n0 + 4u]), half(bs[n0 + 5u]));
        s67 = half2(half(bs[n0 + 6u]), half(bs[n0 + 7u]));
      }
      if (WIDE) {
        if (tid < 128u) tf_i8_expand8s(wx + (cur_buf ^ 1u) * 1024u, tid, nq, s01, s23, s45, s67);
      } else if (tid < 256u) {
        tf_i8_expand4s(wx + (cur_buf ^ 1u) * 1024u, tid, nu, s01, s23);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    cur_buf ^= 1u;
  }
  const int c0 = int(nb + sgc * 64u);
  const int r0 = int(rb + sgr * 16u);
  frag_put_in(lo0, sums, int(a.n), r0, c0, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi0, sums, int(a.n), r0, c0 + 16, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo1, sums, int(a.n), r0, c0 + 32, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi1, sums, int(a.n), r0, c0 + 48, home, int(rb + rows_here), int(nb + cols_here));
}

kernel void tf_exl3_matmul_prefill_i8b(const device half* const x [[buffer(0)]],
                                       const device uchar* const w8 [[buffer(1)]],
                                       device float* const sums [[buffer(2)]],
                                       constant TfExl3Args& a [[buffer(3)]],
                                       const device float* const bscales [[buffer(4)]],
                                       uint lane [[thread_index_in_simdgroup]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_i8b<false>(wx, x, w8, bscales, sums, a, lane, sg, tid, tg);
}

kernel void tf_exl3_matmul_prefill_i8bw(const device half* const x [[buffer(0)]],
                                        const device uchar* const w8 [[buffer(1)]],
                                        device float* const sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device float* const bscales [[buffer(4)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_i8b<true>(wx, x, w8, bscales, sums, a, lane, sg, tid, tg);
}

// (v1, exact f32 accumulate): the block-scaled int8 body for the pre-rotated pack. The accumulate order is the same
// 16-k MMA order; each block's finished fragment is scaled and added to the running f32 accumulator, so the result is
// sum_kb scale_kb * sum_{k in kb} code[k,n]*x[k] -- the exact linear map of the stored W_hat.
template <bool WIDE>
inline void tf_exl3_prefill_body_i8a(threadgroup half* wx, const device half* x, const device uchar* w8,
                                     const device float* bscales, device float* sums, constant TfExl3Args& a,
                                     uint lane, uint sg, uint tid, uint3 tg) {
  constexpr uint RG = 8u;
  const uint trows = RG * 16u;
  const uint rb = tg.y * trows, nb = tg.x * 64u;
  if (rb >= a.rows) return;
  const uint rows_here = min(trows, a.rows - rb);
  const uint cols_here = min(64u, a.n - nb);
  const short2 home = frag_home(ushort(lane));
  const uint nb64 = a.n >> 6, block = nb >> 6;
  const uint sgr = sg % RG, sgc = sg / RG;
  frag<float> lo0, hi0, lo1, hi1;      // the current 128-k block
  frag<float> acc0, acc1, acc2, acc3;  // the scaled running sum
  lo0 = frag<float>(0); hi0 = frag<float>(0); lo1 = frag<float>(0); hi1 = frag<float>(0);
  acc0 = frag<float>(0); acc1 = frag<float>(0); acc2 = frag<float>(0); acc3 = frag<float>(0);
  const int c0 = int(nb + sgc * 64u);
  if (WIDE) {
    if (tid < 128u) {
      const uint2 q = *reinterpret_cast<const device uint2*>(w8 + ((block) << 10) + 8u * tid);
      tf_i8_expand8(wx, tid, q);
    }
  } else if (tid < 256u) {
    const uint u = *reinterpret_cast<const device uint*>(w8 + ((block) << 10) + 4u * tid);
    tf_i8_expand4(wx, tid, u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint cur_buf = 0;
  for (uint kb = 0; kb < a.k; kb += 16u) {
    const uint nkb = kb + 16u;
    uint nu = 0u;
    uint2 nq = uint2(0u);
    if (nkb < a.k) {
      if (WIDE) {
        if (tid < 128u) nq = *reinterpret_cast<const device uint2*>(w8 + (((nkb >> 4) * nb64 + block) << 10) + 8u * tid);
      } else if (tid < 256u) {
        nu = *reinterpret_cast<const device uint*>(w8 + (((nkb >> 4) * nb64 + block) << 10) + 4u * tid);
      }
    }
    if ((kb & 127u) == 0u) {
      lo0 = frag<float>(0); hi0 = frag<float>(0); lo1 = frag<float>(0); hi1 = frag<float>(0);
    }
    frag<half> av;
    tf_frag_get_dev4(av, x, int(a.k), int(rb + sgr * 16u), int(kb), home, int(rb + rows_here));
    threadgroup half* cur = wx + cur_buf * 1024u;
    frag<half> b0, b1;
    const int wc = int(sgc * 64u);
    tf_frag_get_tg4(b0, cur, 64, 0, wc, home);
    tf_frag_get_tg4(b1, cur, 64, 0, wc + 16, home);
    mma_16x32<false, false>(lo0, hi0, av, b0, b1);
    tf_frag_get_tg4(b0, cur, 64, 0, wc + 32, home);
    tf_frag_get_tg4(b1, cur, 64, 0, wc + 48, home);
    mma_16x32<false, false>(lo1, hi1, av, b0, b1);
    if (nkb < a.k) {
      if (WIDE) {
        if (tid < 128u) tf_i8_expand8(wx + (cur_buf ^ 1u) * 1024u, tid, nq);
      } else if (tid < 256u) {
        tf_i8_expand4(wx + (cur_buf ^ 1u) * 1024u, tid, nu);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    cur_buf ^= 1u;
    // Nkb is the first k of the NEXT block: at every 128 boundary the finished fragment is scaled and folded.
    if ((nkb & 127u) == 0u) {
      const uint bi = kb >> 7;
      const device float* bs = bscales + ulong(bi) * a.n;
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int cA = c0 + home.x + (e & 3);
        acc0[e] = fma(lo0[e], bs[cA], acc0[e]);
        acc1[e] = fma(hi0[e], bs[cA + 16], acc1[e]);
        acc2[e] = fma(lo1[e], bs[cA + 32], acc2[e]);
        acc3[e] = fma(hi1[e], bs[cA + 48], acc3[e]);
      }
    }
  }
  const int r0 = int(rb + sgr * 16u);
  frag_put_in(acc0, sums, int(a.n), r0, c0, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(acc1, sums, int(a.n), r0, c0 + 16, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(acc2, sums, int(a.n), r0, c0 + 32, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(acc3, sums, int(a.n), r0, c0 + 48, home, int(rb + rows_here), int(nb + cols_here));
}


kernel void tf_exl3_matmul_prefill_i8a(const device half* const x [[buffer(0)]],
                                       const device uchar* const w8 [[buffer(1)]],
                                       device float* const sums [[buffer(2)]],
                                       constant TfExl3Args& a [[buffer(3)]],
                                       const device float* const bscales [[buffer(4)]],
                                       uint lane [[thread_index_in_simdgroup]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_i8a<false>(wx, x, w8, bscales, sums, a, lane, sg, tid, tg);
}

kernel void tf_exl3_matmul_prefill_fp8(const device half* const x [[buffer(0)]],
                                       const device uchar* const w8 [[buffer(1)]],
                                       device float* const sums [[buffer(2)]],
                                       constant TfExl3Args& a [[buffer(3)]],
                                       const device float* const scales [[buffer(4)]],
                                       uint lane [[thread_index_in_simdgroup]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_fp8<false, float>(wx, x, w8, scales, sums, a, lane, sg, tid, tg);
}

kernel void tf_exl3_matmul_prefill_fp8w(const device half* const x [[buffer(0)]],
                                        const device uchar* const w8 [[buffer(1)]],
                                        device float* const sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device float* const scales [[buffer(4)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_fp8<true, float>(wx, x, w8, scales, sums, a, lane, sg, tid, tg);
}

// The same byte body with a bf16 destination, for a pre-rotated pack (TF_DSV41_I8_ROT=1): the row scale is applied in
// the body's own epilogue, so no rotate_out (Hadamard + svh) and no f32 sums round trip.
kernel void tf_exl3_matmul_prefill_fp8b(const device half* const x [[buffer(0)]],
                                        const device uchar* const w8 [[buffer(1)]],
                                        device bfloat* const y [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device float* const scales [[buffer(4)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_fp8<false, bfloat>(wx, x, w8, scales, y, a, lane, sg, tid, tg);
}
