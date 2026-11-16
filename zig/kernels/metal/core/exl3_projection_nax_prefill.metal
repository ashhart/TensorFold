// The routed prompt tile's job follows the release's expert blocks; the pack layout and row walk are our own.

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

// One row an expert, four rows a threadgroup.
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
  // Two weight tiles a SIMDgroup: the next block decodes while this one multiplies.
#if defined(TF_DECODE_COMPACT)
  // Compact: one row a threadgroup, two 16x32 tiles only (1 KB).
  threadgroup half wx[2 * 16 * 32];
#else
  threadgroup half wx[2 * 4 * 16 * 32];
#endif
  const uint nb = tg.x * 32;
  // One SIMDgroup a row: a threadgroup of four warps takes four rows, one of a single warp takes one.
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
  // One routed row's pack slice: the expert's trellis, k/n strides, width and codebook.
  const TfExl3DecodeJob job = tf_exl3_routed_decode_job(trellis, a, descriptors, expert);
  const short2 home = frag_home(ushort(lane));
  const uint cols_here = min(32u, a.n - nb);
  // A decode tile has a handful of rows and one threadgroup a layer for the whole k range.
  const uint split = tg.z;
  const uint blocks = a.k / 16;
  const uint b0 = a.splits > 1 ? blocks * split / a.splits : 0;
  const uint b1 = a.splits > 1 ? blocks * (split + 1) / a.splits : blocks;
  // Eight consecutive states a thread of each sixteen-column tile, so the stream words they can touch are read once.
  const TfExl3DigWindows walk = tf_exl3_dig_windows<8u, 1u, 32u>(lane * 8u, job.halves, 0u, 0u, true);
  // TF_DECODE_FUNNEL: the routed pack's halves = 6.
#if defined(TF_DECODE_FUNNEL)
  // Halves = 8 (the drafter's routed pack) joins halves = 6 ONLY under TF_DECODE_FUNNEL_H8, an optional arm.
#if defined(TF_DECODE_FUNNEL_H8)
  const bool funnel = job.halves == 6u or job.halves == 8u;
#else
  const bool funnel = job.halves == 6u;
#endif
#else
  const bool funnel = false;
#endif
// THE TWO-WORD READ AT halves = 8 WITHOUT THE FUNNEL'S ASSEMBLY.
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
          // Two words a lane, on our map and with our left-shift funnel.
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
