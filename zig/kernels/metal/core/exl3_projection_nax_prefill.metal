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
  // the decode's issue rate is not occupancy-bound.
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
  // One routed row's pack slice: the expert's trellis, k/n strides, width and codebook. The release's MTP and draft
  // rounds feed the same EXL3 projection shapes (inference/model.py, revision 314113f8); the pack layout is our own.
  const TfExl3DecodeJob job = tf_exl3_routed_decode_job(trellis, a, descriptors, expert);
  const short2 home = frag_home(ushort(lane));
  const uint cols_here = min(32u, a.n - nb);
  // A decode tile has a handful of rows and one threadgroup a layer for the whole k range, so a split takes a slice
  // of the k blocks and the reduce kernel sums the parts in split order.
  const uint split = tg.z;
  const uint blocks = a.k / 16;
  const uint b0 = a.splits > 1 ? blocks * split / a.splits : 0;
  const uint b1 = a.splits > 1 ? blocks * (split + 1) / a.splits : blocks;
  // Eight consecutive states a thread of each sixteen-column tile, so the stream words they can touch are read once.
  // The walk names the lane's first words and its window; the cursor and the tile slot come from registers.
  const TfExl3DigWindows walk = tf_exl3_dig_windows<8u, 1u, 32u>(lane * 8u, job.halves, 0u, 0u, true);
  // TF_DECODE_FUNNEL: the routed pack's halves = 6. Eight consecutive states a lane span 37 bits from start + 16, so
  // two 32-bit stream words cover all eight and the per-state word-pair selection collapses to a two-way select. For
  // halves = 6 the rel = 1 states always start at bit offset <= 16, so the next word after the pair contributes only
  // below the 16-bit state and can be dropped. The two-word and four-word paths agree state for state; halves ! = 6
  // keeps the general path.
#if defined(TF_DECODE_FUNNEL)
  // Halves = 8 (the drafter's routed pack) joins halves = 6 ONLY under TF_DECODE_FUNNEL_H8, an optional arm. At
  // halves = 8 every lane's run starts at the same bit offset, so eight four-bit-spaced states span exactly 44 bits
  // and sit in two words with every rel = 1 state at an offset <= 16 -- the same two-word form.
#if defined(TF_DECODE_FUNNEL_H8)
  const bool funnel = job.halves == 6u or job.halves == 8u;
#else
  const bool funnel = job.halves == 6u;
#endif
#else
  const bool funnel = false;
#endif
// THE TWO-WORD READ AT halves = 8 WITHOUT THE FUNNEL'S ASSEMBLY. Every lane's eight states start at off0 = 20 and sit
// in exactly two stream words (rel in {0, 1}); the general path loads four. This drops the two words it cannot touch
// and keeps the cursor's select-chain extraction. Bit-identical by construction (the two words it drops feed only rel
// >= 2, which does not occur at halves = 8).
#if defined(TF_DECODE_TWO)
  const bool two_word = job.halves == 8u;
#else
  const bool two_word = false;
#endif
  frag<float> lo = frag<float>(0), hi = frag<float>(0);
  // The words a block needs are loaded a block early, so their latency overlaps the multiply instead of stalling.
  uint pre[2][4];
  const bool pipeline = (job.halves & 1u) == 0 and job.halves <= 16;
  if (pipeline) {
    const device ushort* t0 = reinterpret_cast<const device ushort*>(job.trellis + ulong(b0) * job.tile_k_stride + ulong(nb / 16) * job.tile_n_stride);
    pre[0][0] = tf_exl3_word(t0, walk.first_word);
    pre[0][1] = tf_exl3_word(t0, walk.next1);
    if (!funnel and !two_word) {
      pre[0][2] = tf_exl3_word(t0, walk.next2);
      pre[0][3] = tf_exl3_word(t0, walk.next3);
    }
    if (nb + 16 < a.n) {
      const device ushort* t1 = reinterpret_cast<const device ushort*>(job.trellis + ulong(b0) * job.tile_k_stride + ulong(nb / 16 + 1) * job.tile_n_stride);
      pre[1][0] = tf_exl3_word(t1, walk.first_word);
      pre[1][1] = tf_exl3_word(t1, walk.next1);
      if (!funnel and !two_word) {
        pre[1][2] = tf_exl3_word(t1, walk.next2);
        pre[1][3] = tf_exl3_word(t1, walk.next3);
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
          const device ushort* tn = reinterpret_cast<const device ushort*>(job.trellis + ulong(nxt) * job.tile_k_stride + ulong(nb / 16 + h) * job.tile_n_stride);
          pre[h][0] = tf_exl3_word(tn, walk.first_word);
          pre[h][1] = tf_exl3_word(tn, walk.next1);
          if (!funnel and !two_word) {
            pre[h][2] = tf_exl3_word(tn, walk.next2);
            pre[h][3] = tf_exl3_word(tn, walk.next3);
          }
        }
        if (funnel) {
          // Two words a lane, on our map and with our left-shift funnel. The rel = 1 states start at bit offset <= 16
          // at halves = 6, so the second word's carry is the whole state's source.
          TF_UNROLL
          for (uint k = 0; k < 8; ++k) {
            const uint lo8 = walk.rel[k] == 0 ? w0 : w1;
            const uint hi8 = walk.rel[k] == 0 ? w1 : 0;
            const uint carry = (hi8 >> walk.sh[k]) & (0u - uint(walk.off[k] != 0));
#if defined(TF_DECODE_TABLE)
            wxc[walk.at[k] + h * 16] = tf_exl3_table_value(table, ushort(((lo8 << walk.off[k]) | carry) >> 16), job.codebook);
#else
            wxc[walk.at[k] + h * 16] = tf_exl3_value(ushort(((lo8 << walk.off[k]) | carry) >> 16), job.codebook);
#endif
          }
        } else {
          TF_UNROLL
          for (uint k = 0; k < 8; ++k) {
            const uint lo8 = walk.rel[k] == 0 ? w0 : walk.rel[k] == 1 ? w1 : w2;
            const uint hi8 = walk.rel[k] == 0 ? w1 : walk.rel[k] == 1 ? w2 : w3;
            const uint carry = (hi8 >> walk.sh[k]) & (0u - uint(walk.off[k] != 0));
#if defined(TF_DECODE_TABLE)
            wxc[walk.at[k] + h * 16] = tf_exl3_table_value(table, ushort(((lo8 << walk.off[k]) | carry) >> 16), job.codebook);
#else
            wxc[walk.at[k] + h * 16] = tf_exl3_value(ushort(((lo8 << walk.off[k]) | carry) >> 16), job.codebook);
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
            job.trellis + ulong(kb / 16) * job.tile_k_stride + ulong(nb / 16 + h) * job.tile_n_stride);
        const uint l = p / 8, e = p % 8;
        const uint r = 2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u);
        const uint c = l / 4 + 8 * (e >> 2);
#if defined(TF_DECODE_TABLE)
        wxc[r * 32 + h * 16 + c] = tf_exl3_table_value(table, tf_exl3_mapped_state(tile, tf_exl3_map_for(p, job.halves)), job.codebook);
#else
        wxc[r * 32 + h * 16 + c] = tf_exl3_value(tf_exl3_mapped_state(tile, tf_exl3_map_for(p, job.halves)), job.codebook);
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
  const TfExl3DecodeJob job = tf_exl3_dense_decode_job(trellis, a);
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
          job.trellis + ulong(kb / 16) * job.tile_k_stride + ulong(nb / 16 + h) * job.tile_n_stride);
      const uint l = p / 8, e = p % 8;
      const uint r = 2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u);
      const uint c = l / 4 + 8 * (e >> 2);
      wxg[r * 32 + h * 16 + c] = tf_exl3_value(tf_exl3_mapped_state(tile, tf_exl3_map_for(p, job.halves)), job.codebook);
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
