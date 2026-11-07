inline void tf_exl3_matmul_wide_static_pf_impl(const device half* x, const device uchar* trellis,
                                               device float* sums, constant TfExl3Args& a,
                                               const device uint* routes, const device TfExl3Expert* descriptors,
                                               device float* partials, const device half* table,
                                               uint lane, uint sg, uint3 tg,
                                               threadgroup half* wx_all, threadgroup float* xx_all,
                                               threadgroup half* lut) {
  const uint split = a.splits > 1 ? tg.y : 0;
  const uint rb = a.splits > 1 ? tg.z * 8 : tg.y * 8, nb = tg.x * 128 + sg * 16;
  if (!simd_any(tf_exl3_active(routes, a, rb + lane % 8))) return;
  const device uchar* data = trellis;
  const ulong ks = a.tile_k_stride;
  const ulong ns = a.tile_n_stride;
  const uint halves = a.halves;
  const uint codebook = a.codebook;
  threadgroup half* wx = wx_all + sg * 256;
  uint at[8];
  const uint words = 4 * halves;
  const uint wi0 = tf_exl3_map_of(lane * 8, halves) & 63u;
  const uint off0 = (tf_exl3_map_of(lane * 8, halves) >> 6) & 31u;
  TF_UNROLL
  for (uint j = 0; j < 8; ++j) {
    const uint p = lane * 8 + j, l = p / 8, e = p % 8;
    at[j] = (2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u)) * 16u + (l / 4 + 8 * (e >> 2));
  }
  float high[4] = {0,0,0,0}, low[4] = {0,0,0,0};
  const uint blocks = a.k / 16;
  const uint b0 = a.splits > 1 ? blocks * split / a.splits : 0;
  const uint b1 = a.splits > 1 ? blocks * (split + 1) / a.splits : blocks;
  uint a1 = wi0 + 1; if (a1 >= words) a1 -= words;
  uint a2 = wi0 + 2; if (a2 >= words) a2 -= words;
  const uint sh0 = (32u - off0) & 31u;
  const uint m0 = 0u - uint(off0 != 0u);
  // The prefetch ring: PF blocks of three stream words a lane, in registers (PF a compile-time constant, so the array
  // never lands in thread-local memory).
  uint pf[PF][3];
  TF_UNROLL
  for (uint p = 0; p < PF; ++p) {
    if (b0 + p < b1) {
      const device ushort* t = reinterpret_cast<const device ushort*>(data + ulong(b0 + p) * ks + ulong(nb / 16) * ns);
      pf[p][0] = tf_exl3_word(t, wi0);
      pf[p][1] = tf_exl3_word(t, a1);
      pf[p][2] = tf_exl3_word(t, a2);
    }
  }
  uint b = b0;
  const uint main_end = b0 + ((b1 - b0) / PF) * PF;
  while (b < main_end) {
    TF_UNROLL
    for (uint p = 0; p < PF; ++p) {
      const uint blk = b + p;
      threadgroup float* xx = xx_all + ((blk - b0) & 1u) * 128;
      if (sg == 0) {
        for (uint j = lane; j < 128; j += 32) {
          const uint row = rb + j / 16;
          xx[j] = tf_exl3_active(routes, a, row) ? float(x[ulong(row) * a.k + blk * 16 + j % 16]) : 0.0f;
        }
      }
      const uint w0 = pf[p][0], w1 = pf[p][1], w2 = pf[p][2];
      // The next PF blocks' words, issued now: PF loads in flight a lane instead of one.
      if (blk + PF < b1) {
        const device ushort* tn = reinterpret_cast<const device ushort*>(data + ulong(blk + PF) * ks + ulong(nb / 16) * ns);
        pf[p][0] = tf_exl3_word(tn, wi0);
        pf[p][1] = tf_exl3_word(tn, a1);
        pf[p][2] = tf_exl3_word(tn, a2);
      }
      const uint c_cur = (w0 << off0) | ((w1 >> sh0) & m0);
      const uint c_next = (w1 << off0) | ((w2 >> sh0) & m0);
      #define TF_EXL3_W(J) tf_exl3_state_static<S, J>(c_cur, c_next)
      #define TF_EXL3_V(J) (Lut ? lut[tf_exl3_mul1_index(TF_EXL3_W(J))] : tf_exl3_table_value(table, codebook, TF_EXL3_W(J)))
      wx[at[0]] = TF_EXL3_V(0);
      wx[at[1]] = TF_EXL3_V(1);
      wx[at[2]] = TF_EXL3_V(2);
      wx[at[3]] = TF_EXL3_V(3);
      wx[at[4]] = TF_EXL3_V(4);
      wx[at[5]] = TF_EXL3_V(5);
      wx[at[6]] = TF_EXL3_V(6);
      wx[at[7]] = TF_EXL3_V(7);
      #undef TF_EXL3_V
      #undef TF_EXL3_W
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = 0; i < 4; ++i) {
        const uint at_i = lane + 32 * i, row = rb + at_i / 16, col = at_i % 16;
        if (!tf_exl3_active(routes, a, row)) continue;
        for (uint k = 0; k < 16; ++k) high[i] += xx[(at_i / 16) * 16 + k] * float(wx[k * 16 + col]);
      }
    }
    b += PF;
  }
  // The tail: the blocks the pipelined loop did not cover (fewer than PF).
  for (; b < b1; ++b) {
    threadgroup float* xx = xx_all + ((b - b0) & 1u) * 128;
    if (sg == 0) {
      for (uint j = lane; j < 128; j += 32) {
        const uint row = rb + j / 16;
        xx[j] = tf_exl3_active(routes, a, row) ? float(x[ulong(row) * a.k + b * 16 + j % 16]) : 0.0f;
      }
    }
    const device ushort* t = reinterpret_cast<const device ushort*>(data + ulong(b) * ks + ulong(nb / 16) * ns);
    const uint w0 = tf_exl3_word(t, wi0), w1 = tf_exl3_word(t, a1), w2 = tf_exl3_word(t, a2);
    const uint c_cur = (w0 << off0) | ((w1 >> sh0) & m0);
    const uint c_next = (w1 << off0) | ((w2 >> sh0) & m0);
    #define TF_EXL3_W(J) tf_exl3_state_static<S, J>(c_cur, c_next)
    #define TF_EXL3_V(J) (Lut ? lut[tf_exl3_mul1_index(TF_EXL3_W(J))] : tf_exl3_table_value(table, codebook, TF_EXL3_W(J)))
    wx[at[0]] = TF_EXL3_V(0);
    wx[at[1]] = TF_EXL3_V(1);
    wx[at[2]] = TF_EXL3_V(2);
    wx[at[3]] = TF_EXL3_V(3);
    wx[at[4]] = TF_EXL3_V(4);
    wx[at[5]] = TF_EXL3_V(5);
    wx[at[6]] = TF_EXL3_V(6);
    wx[at[7]] = TF_EXL3_V(7);
    #undef TF_EXL3_V
    #undef TF_EXL3_W
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
      const uint at_i = lane + 32 * i, row = rb + at_i / 16, col = at_i % 16;
      if (!tf_exl3_active(routes, a, row)) continue;
      for (uint k = 0; k < 16; ++k) high[i] += xx[(at_i / 16) * 16 + k] * float(wx[k * 16 + col]);
    }
  }
  for (uint i = 0; i < 4; ++i) {
    const uint at_i = lane + 32 * i, row = rb + at_i / 16;
    if (tf_exl3_active(routes, a, row)) {
      if (a.splits > 1) partials[(ulong(split) * a.rows + row) * a.n + nb + at_i % 16] = high[i] + low[i];
      else sums[ulong(row) * a.n + nb + at_i % 16] = high[i] + low[i];
    }
  }
}





// 32-k continuation of the wide one-row tile: eight SIMDgroups own 128 columns, and each block decodes two
// consecutive 16-k tiles before one barrier, so the barrier count halves and a threadgroup reads 1.5 KB of contiguous
// trellis a block. Each 16-k tile is decoded exactly as tf_exl3_matmul_wide_tab decodes it (same maps, same table
// value) and the accumulation still ascends k in order, so the output is bit-identical by construction.
inline void tf_exl3_wide2_decode16(const device ushort* tile, uint wi0, uint a1, uint a2, uint a3,
                                   threadgroup half* dst,
                                   const thread uint* at, const thread uint* off, const thread uint* sh,
                                   const thread uint* rel, const thread uint* maps, uint lane,
                                   const device half* table, uint codebook, bool table_on, bool fast) {
  if (fast) {
    const uint w0 = tf_exl3_word(tile, wi0), w1 = tf_exl3_word(tile, a1);
    const uint w2 = tf_exl3_word(tile, a2), w3 = tf_exl3_word(tile, a3);
    TF_UNROLL
    for (uint j = 0; j < 8; ++j) {
      const uint lo = rel[j] == 0 ? w0 : rel[j] == 1 ? w1 : w2;
      const uint hi = rel[j] == 0 ? w1 : rel[j] == 1 ? w2 : w3;
      const uint carry = (hi >> sh[j]) & (0u - uint(off[j] != 0));
      const ushort s = ushort(((lo << off[j]) | carry) >> 16);
      dst[at[j]] = table_on ? tf_exl3_table_value(table, codebook, s) : tf_exl3_value(s, codebook);
    }
  } else {
    for (uint j = 0; j < 8; ++j) {
      const uint p = lane + 32 * j, l = p / 8, e = p % 8;
      const uint r = 2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u);
      const uint c = l / 4 + 8 * (e >> 2);
      const ushort s = tf_exl3_mapped_state(tile, maps[j]);
      dst[r * 16 + c] = table_on ? tf_exl3_table_value(table, codebook, s) : tf_exl3_value(s, codebook);
    }
  }
}

template <bool Table>
inline void tf_exl3_matmul_wide2_impl(const device half* x, const device uchar* trellis,
                                      device float* sums, constant TfExl3Args& a,
                                      const device uint* routes, const device TfExl3Expert* descriptors,
                                      device float* partials, const device half* table,
                                      uint lane, uint sg, uint3 tg,
                                      threadgroup half* wx_all, threadgroup float* xx_all) {
  const uint split = a.splits > 1 ? tg.y : 0;
  const uint rb = a.splits > 1 ? tg.z * 8 : tg.y * 8, nb = tg.x * 128 + sg * 16;
  if (!simd_any(tf_exl3_active(routes, a, rb + lane % 8))) return;
  const device uchar* data = trellis;
  const ulong ks = a.tile_k_stride;
  const ulong ns = a.tile_n_stride;
  const uint halves = a.halves;
  const uint codebook = a.codebook;
  threadgroup half* wx = wx_all + sg * 512;
  uint maps[8], at[8], off[8], sh[8], rel[8];
  const uint words = 4 * halves;
  const uint ring = 128 * halves;
  const uint wi0 = tf_exl3_map_of(lane * 8, halves) & 63u;
  const uint cur0 = (wi0 << 5) | ((tf_exl3_map_of(lane * 8, halves) >> 6) & 31u);
  TF_UNROLL
  for (uint j = 0; j < 8; ++j) {
    const uint p = lane * 8 + j, l = p / 8, e = p % 8;
    uint c8 = cur0 + j * (halves >> 1);
    if (c8 >= ring) c8 -= ring;
    off[j] = c8 & 31u;
    sh[j] = (32u - off[j]) & 31u;
    rel[j] = (c8 >> 5) >= wi0 ? (c8 >> 5) - wi0 : (c8 >> 5) + words - wi0;
    at[j] = (2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u)) * 16u + (l / 4 + 8 * (e >> 2));
  }
  for (uint j = 0; j < 8; ++j) {
    const uint p = lane + 32 * j, ring2 = 128 * halves;
    const uint end = ((p + 1) * halves - ((halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
    uint start = end + ring2 - 16;
    if (start >= ring2) start -= ring2;
    const uint wi = start >> 5, o = start & 31u, next = wi + 1 == 4 * halves ? 0 : wi + 1;
    maps[j] = wi | (o << 6) | (next << 11);
  }
  float high[4] = {0,0,0,0};
  const uint blocks = a.k / 16;
  const uint b0 = a.splits > 1 ? blocks * split / a.splits : 0;
  const uint b1 = a.splits > 1 ? blocks * (split + 1) / a.splits : blocks;
  const bool fast = (halves & 1u) == 0 and halves <= 16;
  const uint a1 = wi0 + 1 < words ? wi0 + 1 : wi0 + 1 - words;
  const uint a2 = wi0 + 2 < words ? wi0 + 2 : wi0 + 2 - words;
  const uint a3 = wi0 + 3 < words ? wi0 + 3 : wi0 + 3 - words;
  uint b = b0;
  for (; b + 1 < b1; b += 2) {
    threadgroup float* xx = xx_all + (((b - b0) >> 1) & 1u) * 256;
    // One warp loads the 8-row x 32-k activation tile for all eight; zeros past a.rows.
    if (sg == 0) {
      for (uint j = lane; j < 256; j += 32) {
        const uint row = rb + j / 32;
        const uint kk = b * 16 + j % 32;
        xx[j] = (kk < a.k and tf_exl3_active(routes, a, row)) ? float(x[ulong(row) * a.k + kk]) : 0.0f;
      }
    }
    const device ushort* tileA = reinterpret_cast<const device ushort*>(data + ulong(b) * ks + ulong(nb / 16) * ns);
    const device ushort* tileB = reinterpret_cast<const device ushort*>(data + ulong(b + 1) * ks + ulong(nb / 16) * ns);
    tf_exl3_wide2_decode16(tileA, wi0, a1, a2, a3, wx, at, off, sh, rel, maps, lane, table, codebook, Table, fast);
    tf_exl3_wide2_decode16(tileB, wi0, a1, a2, a3, wx + 256, at, off, sh, rel, maps, lane, table, codebook, Table, fast);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
      const uint at_i = lane + 32 * i, row = rb + at_i / 16, col = at_i % 16;
      if (!tf_exl3_active(routes, a, row)) continue;
      for (uint k = 0; k < 32; ++k) high[i] += xx[(at_i / 16) * 32 + k] * float(wx[k * 16 + col]);
    }
  }
  if (b < b1) {
    threadgroup float* xx = xx_all + (((b - b0) >> 1) & 1u) * 256;
    if (sg == 0) {
      for (uint j = lane; j < 128; j += 32) {
        const uint row = rb + j / 16;
        xx[(j / 16) * 32 + j % 16] = tf_exl3_active(routes, a, row) ? float(x[ulong(row) * a.k + b * 16 + j % 16]) : 0.0f;
      }
    }
    const device ushort* tileA = reinterpret_cast<const device ushort*>(data + ulong(b) * ks + ulong(nb / 16) * ns);
    tf_exl3_wide2_decode16(tileA, wi0, a1, a2, a3, wx, at, off, sh, rel, maps, lane, table, codebook, Table, fast);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
      const uint at_i = lane + 32 * i, row = rb + at_i / 16, col = at_i % 16;
      if (!tf_exl3_active(routes, a, row)) continue;
      for (uint k = 0; k < 16; ++k) high[i] += xx[(at_i / 16) * 32 + k] * float(wx[k * 16 + col]);
    }
  }
  for (uint i = 0; i < 4; ++i) {
    const uint at_i = lane + 32 * i, row = rb + at_i / 16;
    if (tf_exl3_active(routes, a, row)) {
      if (a.splits > 1) partials[(ulong(split) * a.rows + row) * a.n + nb + at_i % 16] = high[i];
      else sums[ulong(row) * a.n + nb + at_i % 16] = high[i];
    }
  }
}


// Direct-register one-row decode: a lane owns one output column, decodes that column's 16 states for a 16-k block
// into registers (eight stream words; the window is sized for halves <= 14), reads the codebook table and accumulates
// ascending k. No threadgroup weight tile, no barrier, no shared round trip; a warp covers 32 columns. The k order,
// the table entry and the FP32 FMA sequence are the wide body's, so the one-row output is bit-identical by
// construction (halves = 6, the shipped pack; guarded on that in the caller).
kernel void tf_exl3_matmul_direct_tab(const device half* x [[buffer(0)]],
                                      const device uchar* trellis [[buffer(1)]],
                                      device float* sums [[buffer(2)]],
                                      constant TfExl3Args& a [[buffer(3)]],
                                      const device uint* routes [[buffer(4)]],
                                      const device TfExl3Expert* descriptors [[buffer(5)]],
                                      device float* partials [[buffer(6)]],
                                      const device half* table [[buffer(7)]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint sg [[simdgroup_index_in_threadgroup]],
                                      uint3 tptg [[threads_per_threadgroup]],
                                      uint3 tg [[threadgroup_position_in_grid]]) {
  const uint split = a.splits > 1 ? tg.y : 0;
  const uint span = tptg.x;
  const uint nb = tg.x * span + sg * 32 + lane;
  if (nb >= a.n) return;
  const device uchar* data = trellis;
  const ulong ks = a.tile_k_stride;
  const ulong ns = a.tile_n_stride;
  const uint halves = a.halves;
  const uint codebook = a.codebook;
  const uint words = 4 * halves;
  const uint c = lane & 15u;
  // The tile state of (k, c): p = base + 8*(k/2) + (k&1) - 30*(k/8), from the wide body's at[] map.
  const uint base = 32u * (c % 8u) + 4u * (c / 8u);
  uint rel[16], off[16], wnext[8];
  uint wbase = 0;
#if defined(TF_DIRECT_NW)
  uint maxrel = 0;
#endif
  for (uint k = 0; k < 16; ++k) {
    const uint p = base + 8u * (k >> 1) + (k & 1u) - 30u * ((k >> 3) & 1u);
    const uint map = tf_exl3_map_of(p, halves);
    const uint wi = map & 63u;
    if (k == 0) wbase = wi;
    rel[k] = (wi + words - wbase) % words;
#if defined(TF_DIRECT_NW)
    maxrel = max(maxrel, rel[k]);
#endif
    off[k] = (map >> 6) & 31u;
  }
  for (uint i = 0; i < 8; ++i) wnext[i] = (wbase + i) % words;
#if defined(TF_DIRECT_NW)
  // The sixteen states of a column span 27 ring positions; at halves = 10 the union is six words (max(rel) = 4), at
  // halves = 12 seven (max(rel) = 5), at halves = 6 five. The body's eight-word load is two or three words more than
  // the `rel[k] + 1` access ever reaches; load only what the map names.
  const uint nw = min(maxrel + 2u, words);
#endif
  float high = 0;
  const uint blocks = a.k / 16;
  const uint b0 = a.splits > 1 ? blocks * split / a.splits : 0;
  const uint b1 = a.splits > 1 ? blocks * (split + 1) / a.splits : blocks;
  for (uint b = b0; b < b1; ++b) {
    const device ushort* tile = reinterpret_cast<const device ushort*>(data + ulong(b) * ks + ulong(nb / 16) * ns);
    uint w[8];
#if defined(TF_DIRECT_NW)
    switch (nw) {
      case 3: { TF_UNROLL for (uint i = 0; i < 3; ++i) w[i] = tf_exl3_word(tile, wnext[i]); break; }
      case 4: { TF_UNROLL for (uint i = 0; i < 4; ++i) w[i] = tf_exl3_word(tile, wnext[i]); break; }
      case 5: { TF_UNROLL for (uint i = 0; i < 5; ++i) w[i] = tf_exl3_word(tile, wnext[i]); break; }
      case 6: { TF_UNROLL for (uint i = 0; i < 6; ++i) w[i] = tf_exl3_word(tile, wnext[i]); break; }
      case 7: { TF_UNROLL for (uint i = 0; i < 7; ++i) w[i] = tf_exl3_word(tile, wnext[i]); break; }
      default: { TF_UNROLL for (uint i = 0; i < 8; ++i) w[i] = tf_exl3_word(tile, wnext[i]); break; }
    }
#else
    TF_UNROLL
    for (uint i = 0; i < 8; ++i) w[i] = tf_exl3_word(tile, wnext[i]);
#endif
    TF_UNROLL
    for (uint k = 0; k < 16; ++k) {
      const uint lo = w[rel[k]], hi = w[rel[k] + 1];
      const uint carry = (hi >> ((32u - off[k]) & 31u)) & (0u - uint(off[k] != 0));
      const ushort s = ushort(((lo << off[k]) | carry) >> 16);
      high += float(x[ulong(b) * 16 + k]) * float(tf_exl3_table_value(table, codebook, s));
    }
  }
  if (a.splits > 1) partials[ulong(split) * a.rows * a.n + nb] = high;
  else sums[nb] = high;
}



