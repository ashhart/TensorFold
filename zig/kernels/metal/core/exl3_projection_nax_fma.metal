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
  if (!Dense) {
    const uint expert = routes[row];
    if (expert >= a.experts) return;
  }
  TfExl3DecodeJob job;
  if (Dense) job = tf_exl3_dense_decode_job(trellis, a);
  else job = tf_exl3_routed_decode_job(trellis, a, descriptors, routes[row]);
  if (nb >= job.width) return;
  const uint blocks = a.k / 16;
  const uint W = TF_FMA_WARPS;
  const uint b0 = blocks * sg / W, b1 = blocks * (sg + 1u) / W;
  const TfExl3RowWalk walk = tf_exl3_row_walk(lane, job.halves);
  // Three stream words: the lane's window spans at most 7*S + 16 bits from its first state. The ring of TF_FMA_PF
  // triples (one block in flight is the one-block-in-flight form).
  uint pre[TF_FMA_PF][3];
  {
    const device ushort* t0 = reinterpret_cast<const device ushort*>(job.trellis + ulong(b0) * job.tile_k_stride + ulong(nb / 16) * job.tile_n_stride);
#if TF_FMA_WIDE_WORD
    pre[0][0] = tf_exl3_word_t<true>(t0, walk.first_word);
    pre[0][1] = tf_exl3_word_t<true>(t0, walk.next1);
    pre[0][2] = Two ? 0u : tf_exl3_word_t<true>(t0, walk.next2);
#else
    pre[0][0] = tf_exl3_word(t0, walk.first_word);
    pre[0][1] = tf_exl3_word(t0, walk.next1);
    pre[0][2] = Two ? 0u : tf_exl3_word(t0, walk.next2);
#endif
#if TF_FMA_PF > 1
    // The ring is primed: the first TF_FMA_PF blocks' words are in flight before the walk starts.
    for (uint p = 1; p < TF_FMA_PF; ++p) {
      if (b0 + p < b1) {
        const device ushort* tp = reinterpret_cast<const device ushort*>(job.trellis + ulong(b0 + p) * job.tile_k_stride + ulong(nb / 16) * job.tile_n_stride);
        pre[p][0] = tf_exl3_word(tp, walk.first_word);
        pre[p][1] = tf_exl3_word(tp, walk.next1);
        pre[p][2] = Two ? 0u : tf_exl3_word(tp, walk.next2);
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
      const device ushort* tn = reinterpret_cast<const device ushort*>(job.trellis + ulong(bb + TF_FMA_PF) * job.tile_k_stride + ulong(nb / 16) * job.tile_n_stride);
#if TF_FMA_WIDE_WORD
      pre[slot][0] = tf_exl3_word_t<true>(tn, walk.first_word);
      pre[slot][1] = tf_exl3_word_t<true>(tn, walk.next1);
      pre[slot][2] = Two ? 0u : tf_exl3_word_t<true>(tn, walk.next2);
#else
      pre[slot][0] = tf_exl3_word(tn, walk.first_word);
      pre[slot][1] = tf_exl3_word(tn, walk.next1);
      pre[slot][2] = Two ? 0u : tf_exl3_word(tn, walk.next2);
#endif
    }
#else
    const uint w0 = pre[0][0], w1 = pre[0][1], w2 = pre[0][2];
    if (bb + 1 < b1) {
      const device ushort* tn = reinterpret_cast<const device ushort*>(job.trellis + ulong(bb + 1) * job.tile_k_stride + ulong(nb / 16) * job.tile_n_stride);
#if TF_FMA_WIDE_WORD
      pre[0][0] = tf_exl3_word_t<true>(tn, walk.first_word);
      pre[0][1] = tf_exl3_word_t<true>(tn, walk.next1);
      pre[0][2] = Two ? 0u : tf_exl3_word_t<true>(tn, walk.next2);
#else
      pre[0][0] = tf_exl3_word(tn, walk.first_word);
      pre[0][1] = tf_exl3_word(tn, walk.next1);
      pre[0][2] = Two ? 0u : tf_exl3_word(tn, walk.next2);
#endif
    }
#endif
    // The lane's window: one runtime funnel shift, then the eight windows at compile-time shifts.
    const uint c_cur = (w0 << walk.first_bit) | ((w1 >> walk.shift) & walk.mask);
    const uint c_next = Two ? (w1 << walk.first_bit) : ((w1 << walk.first_bit) | ((w2 >> walk.shift) & walk.mask));
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
    tf_exl3_fma_step<S, 0, Lut>(&acc[0], c_cur, c_next, table, job.codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 1, Lut>(&acc[1], c_cur, c_next, table, job.codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 2, Lut>(&acc[2], c_cur, c_next, table, job.codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 3, Lut>(&acc[3], c_cur, c_next, table, job.codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 4, Lut>(&acc[4], c_cur, c_next, table, job.codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 5, Lut>(&acc[5], c_cur, c_next, table, job.codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 6, Lut>(&acc[6], c_cur, c_next, table, job.codebook, x0, x1, x2, x3, lut);
    tf_exl3_fma_step<S, 7, Lut>(&acc[7], c_cur, c_next, table, job.codebook, x0, x1, x2, x3, lut);
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
