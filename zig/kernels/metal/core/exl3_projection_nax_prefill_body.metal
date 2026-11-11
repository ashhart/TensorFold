inline void tf_exl3_prefill_body(threadgroup half* wx, const device half* x, const device uchar* trellis,
                                 device float* sums, constant TfExl3Args& a, const device uint* routes,
                                 const device TfExl3Expert* descriptors, const device half* table,
                                 const threadgroup half* lut,
                                 uint lane, uint sg, uint tid, uint tptg, uint3 tg) {
  const uint trows = RG * 16u;
  const uint rb = tg.y * trows, nb = tg.x * (64u * CG);
  if (a.indexed == 1 || rb >= a.rows) return;
  const uint tile_expert = a.indexed ? routes[rb] : a.expert_id;
  if (a.indexed && tile_expert >= a.experts) return;
  const TfExl3DigJob job = tf_exl3_dig_job<CG>(trellis, a, routes, descriptors, rb, nb, trows);
  const short2 home = frag_home(ushort(lane));
  // Four warps decode a sixty-four-column group; more warps split each lane's sixteen states instead of waiting.
  constexpr uint wpg = RG / (4u * Rpg);          // warps decoding one column group
  constexpr uint states = 8u / wpg;              // consecutive states a lane owns
  const uint gwarp = tid / 32u, group = gwarp / wpg, wg = gwarp % wpg;
  const uint h = group % 4u, hg = (group / 4u) * 64u;
  const uint p0 = (wg * 32u + tid % 32u) * states;
  const bool decodes = group < 4u * CG and nb + hg + h * 16u < a.n;
  const uint words = 4 * job.halves;
  const uint wi0 = tf_exl3_map_for(p0, job.halves) & 63u;
  // The Fast path's cursor is the shared walk; with F6 the funnel path is the only one.
  TfExl3DigWindows walk;
  if (Fast) walk = tf_exl3_dig_windows<states, CG, 64u * CG>(p0, job.halves, hg, h, !F6);
  // This SIMDgroup's row groups (Rpg of them) and its column group.
  const uint sgr = Rpg * (sg % (RG / Rpg)), sgc = sg / (RG / Rpg);
  // The routed tile's padded rows need no multiply.
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
  // The four stream words a block needs are loaded a block early: issued before the barrier.
  uint p0v = 0, p1v = 0, p2v = 0, p3v = 0;
  if (Fast and decodes) {
    const device ushort* tile0 = reinterpret_cast<const device ushort*>(job.trellis + ulong(nb / 16 + hg / 16u + h) * job.tile_n_stride);
    uint a1 = wi0 + 1; if (a1 >= words) a1 -= words;
    uint a2 = wi0 + 2; if (a2 >= words) a2 -= words;
    uint a3 = wi0 + 3; if (a3 >= words) a3 -= words;
    p0v = tf_exl3_word(tile0, wi0);
    p1v = tf_exl3_word(tile0, a1);
    p2v = tf_exl3_word(tile0, a2);
    p3v = tf_exl3_word(tile0, a3);
  }
  for (uint kb = 0; kb < a.k; kb += 16) {
    // One thread owns 8 consecutive states of one 16x16 tile.
    if (decodes) {
      {
        const device ushort* tile = reinterpret_cast<const device ushort*>(
            job.trellis + ulong(kb / 16) * job.tile_k_stride + ulong(nb / 16 + hg / 16u + h) * job.tile_n_stride);
        if (Funnel and (F6 or job.halves == 6) and (states == 8 or states == 4)) {
          // Two words a lane: the funnel holds the lane's eight windows (three bits apart).
          const uint nkb = kb + 16;
          const uint w0 = p0v, w1 = p1v;
          if (nkb < a.k) {
            const device ushort* tile_n = reinterpret_cast<const device ushort*>(
                job.trellis + ulong(nkb / 16) * job.tile_k_stride + ulong(nb / 16 + hg / 16u + h) * job.tile_n_stride);
            p0v = tf_exl3_word(tile_n, wi0);
            p1v = tf_exl3_word(tile_n, wi0 + 1 < words ? wi0 + 1 : wi0 + 1 - words);
          }
          tf_exl3_decode_funnel<states>(wx, table, lut, job.codebook, walk.at, walk.first_bit, job.halves >> 1, w0, w1);
        } else if (Fast and (job.halves & 1u) == 0 and job.halves <= 16) {
          // Eight states span seven steps, so at 8 bits a value their cursors reach three words past the first.
          const uint w0 = p0v, w1 = p1v, w2 = p2v, w3 = p3v;
          const uint nkb = kb + 16;
          if (nkb < a.k) {
            const device ushort* tile_n = reinterpret_cast<const device ushort*>(
                job.trellis + ulong(nkb / 16) * job.tile_k_stride + ulong(nb / 16 + hg / 16u + h) * job.tile_n_stride);
            p0v = tf_exl3_word(tile_n, wi0);
            p1v = tf_exl3_word(tile_n, wi0 + 1 < words ? wi0 + 1 : wi0 + 1 - words);
            p2v = tf_exl3_word(tile_n, wi0 + 2 < words ? wi0 + 2 : wi0 + 2 - words);
            p3v = tf_exl3_word(tile_n, wi0 + 3 < words ? wi0 + 3 : wi0 + 3 - words);
          }
          TF_UNROLL
          for (uint k = 0; k < states; ++k) {
            const uint lo = walk.rel[k] == 0 ? w0 : walk.rel[k] == 1 ? w1 : w2;
            const uint hi = walk.rel[k] == 0 ? w1 : walk.rel[k] == 1 ? w2 : w3;
            const uint carry = (hi >> walk.sh[k]) & (0u - uint(walk.off[k] != 0));
            wx[walk.at[k]] = tf_exl3_pick(table, lut, ushort(((lo << walk.off[k]) | carry) >> 16), job.codebook);
          }
        } else if (job.halves <= 16 and (job.halves & 1u) == 0 and states == 8) {
          const uint ring = 128 * job.halves;
          uint at1 = wi0 + 1; if (at1 >= words) at1 -= words;
          uint at2 = wi0 + 2; if (at2 >= words) at2 -= words;
          uint at3 = wi0 + 3; if (at3 >= words) at3 -= words;
          uint at4 = wi0 + 4; if (at4 >= words) at4 -= words;
          const uint w0 = tf_exl3_word(tile, wi0);
          const uint w1 = tf_exl3_word(tile, at1), w2 = tf_exl3_word(tile, at2);
          const uint w3 = tf_exl3_word(tile, at3), w4 = tf_exl3_word(tile, at4);
          uint cur = (wi0 << 5) | ((tf_exl3_map_for(p0, job.halves) >> 6) & 31u);
          TF_UNROLL
          for (uint k = 0; k < 8; ++k) {
            // Each state's cursor comes from the block's origin: the eight decodes are independent.
            uint c8 = cur + k * (job.halves >> 1);
            if (c8 >= ring) c8 -= ring;
            const uint wi = c8 >> 5, off = c8 & 31u;
            const uint rel = wi >= wi0 ? wi - wi0 : wi + words - wi0;
            const uint lo = rel == 0 ? w0 : rel == 1 ? w1 : rel == 2 ? w2 : w3;
            const uint hi = rel == 0 ? w1 : rel == 1 ? w2 : rel == 2 ? w3 : w4;
            const uint carry = (hi >> ((32 - off) & 31u)) & (0u - uint(off != 0));
            const uint p = p0 + k, l = p / 8, e = p % 8;
            const uint r = 2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u);
            const uint c = l / 4 + 8 * (e >> 2);
            wx[r * (64u * CG) + hg + h * 16 + c] = tf_exl3_pick(table, lut, ushort(((lo << off) | carry) >> 16), job.codebook);
          }
        } else {
          TF_UNROLL
          for (uint k = 0; k < 8; ++k) {
            const uint p = p0 + k;
            const uint l = p / 8, e = p % 8;
            const uint r = 2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u);
            const uint c = l / 4 + 8 * (e >> 2);
            wx[r * (64u * CG) + hg + h * 16 + c] = tf_exl3_pick(table, lut, tf_exl3_mapped_state(tile, tf_exl3_map_for(p, job.halves)), job.codebook);
          }
        }
      }
    }
    // This SIMDgroup's rows of the k block, read straight into fragments: no staging buffer, one barrier.
    frag<half> av[Rpg];
    if (any_live) {
      TF_UNROLL
      for (uint r = 0; r < Rpg; ++r) {
        if (group_live[r]) tf_frag_get_dev4(av[r], (const device half*)x, int(a.k), int(rb + (sgr + r) * 16), int(kb), home, int(rb + job.rows_here));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
  const int r0 = int(rb + sgr * 16);
  const int c0 = int(nb + sgc * 64u);
  // `sums` is the whole [rows, n] buffer and frag_put_in indexes it absolutely (rr, cc).
  TF_UNROLL
  for (uint r = 0; r < Rpg; ++r) {
    if (group_live[r]) {
      frag_put_in(lo0[r], sums, int(a.n), r0 + int(r * 16), c0, home, int(rb + job.rows_here), int(nb + job.cols_here));
      frag_put_in(hi0[r], sums, int(a.n), r0 + int(r * 16), c0 + 16, home, int(rb + job.rows_here), int(nb + job.cols_here));
      frag_put_in(lo1[r], sums, int(a.n), r0 + int(r * 16), c0 + 32, home, int(rb + job.rows_here), int(nb + job.cols_here));
      frag_put_in(hi1[r], sums, int(a.n), r0 + int(r * 16), c0 + 48, home, int(rb + job.rows_here), int(nb + job.cols_here));
    }
  }
}


// Double-buffered prompt body: the next k block decodes into the tile the multiply is not reading.
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
  const TfExl3DigJob job = tf_exl3_dig_job<CG>(trellis, a, routes, descriptors, rb, nb, trows);
  const short2 home = frag_home(ushort(lane));
  // With Rpg row groups a SIMDgroup (a SIMDgroup owns thirty-two rows in the wide tile).
  constexpr uint wpg = RG / (4u * Rpg);
  constexpr uint states = 8u / wpg;
  const uint gwarp = tid / 32u, group = gwarp / wpg, wg = gwarp % wpg;
  const uint h = group % 4u, hg = (group / 4u) * 64u;
  const uint p0 = (wg * 32u + tid % 32u) * states;
  const bool decodes = group < 4u * CG and nb + hg + h * 16u < a.n;
  // The funnel is admissible where the lane's run fits two 32-bit words: halves = 6.
  const bool use_funnel = Funnel and ((job.halves == 6u and (states == 8u or states == 4u)) or
                                      (job.halves == 8u and states == 4u));
  if ((job.halves & 1u) != 0 or job.halves > 16) {
    tf_exl3_prefill_body<true, RG, CG, UseTable, false, false, Rpg>(wx, x, trellis, sums, a, routes, descriptors, table, nullptr, lane, sg, tid, tptg, tg);
    return;
  }
  const TfExl3DigWindows walk = tf_exl3_dig_windows<states, CG, 64u * CG>(p0, job.halves, hg, h, true);
  const uint words = walk.words;
  const uint wi0 = walk.first_word;
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
    const uint w0 = tf_exl3_block_word(job.trellis, job.tile_k_stride, job.tile_n_stride, 0, nb, hg, h, wi0, words);
    const uint w1 = tf_exl3_block_word(job.trellis, job.tile_k_stride, job.tile_n_stride, 0, nb, hg, h, wi0 + 1, words);
    if (use_funnel) {
      if (job.halves == 8u) tf_exl3_decode_funnel<states, 4>(wx, table, nullptr, job.codebook, walk.at, walk.first_bit, 4, w0, w1);
      else tf_exl3_decode_funnel<states>(wx, table, nullptr, job.codebook, walk.at, walk.first_bit, job.halves >> 1, w0, w1);
    } else {
      const uint w2 = tf_exl3_block_word(job.trellis, job.tile_k_stride, job.tile_n_stride, 0, nb, hg, h, wi0 + 2, words);
      const uint w3 = tf_exl3_block_word(job.trellis, job.tile_k_stride, job.tile_n_stride, 0, nb, hg, h, wi0 + 3, words);
      if (UseTable) tf_exl3_decode_table<states>(wx, table, job.codebook, walk.at, walk.off, walk.sh, walk.rel, w0, w1, w2, w3);
      else tf_exl3_decode_fast<states>(wx, job.codebook, walk.at, walk.off, walk.sh, walk.rel, w0, w1, w2, w3);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint cur_buf = 0;
  for (uint kb = 0; kb < a.k; kb += 16) {
    const uint nkb = kb + 16;
    // Issue the next block's stream loads before the multiply, so their latency overlaps the tensor work.
    uint nw0 = 0, nw1 = 0, nw2 = 0, nw3 = 0;
    if (nkb < a.k and decodes and DoDecode) {
      nw0 = tf_exl3_block_word(job.trellis, job.tile_k_stride, job.tile_n_stride, nkb, nb, hg, h, wi0, words);
      nw1 = tf_exl3_block_word(job.trellis, job.tile_k_stride, job.tile_n_stride, nkb, nb, hg, h, wi0 + 1, words);
      if (!use_funnel) {
        nw2 = tf_exl3_block_word(job.trellis, job.tile_k_stride, job.tile_n_stride, nkb, nb, hg, h, wi0 + 2, words);
        nw3 = tf_exl3_block_word(job.trellis, job.tile_k_stride, job.tile_n_stride, nkb, nb, hg, h, wi0 + 3, words);
      }
    }
    frag<half> av[Rpg];
    TF_UNROLL
    for (uint r = 0; r < Rpg; ++r)
      tf_frag_get_dev4(av[r], (const device half*)x, int(a.k), int(rb + (sgr + r) * 16), int(kb), home, int(rb + job.rows_here));
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
        if (job.halves == 8u) tf_exl3_decode_funnel<states, 4>(nxt, table, nullptr, job.codebook, walk.at, walk.first_bit, 4, nw0, nw1);
        else tf_exl3_decode_funnel<states>(nxt, table, nullptr, job.codebook, walk.at, walk.first_bit, job.halves >> 1, nw0, nw1);
      } else if (UseTable) tf_exl3_decode_table<states>(nxt, table, job.codebook, walk.at, walk.off, walk.sh, walk.rel, nw0, nw1, nw2, nw3);
      else tf_exl3_decode_fast<states>(nxt, job.codebook, walk.at, walk.off, walk.sh, walk.rel, nw0, nw1, nw2, nw3);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (DoDecode) cur_buf ^= 1u;
  }
  const int c0 = int(nb + sgc * 64u);
  TF_UNROLL
  for (uint r = 0; r < Rpg; ++r) {
    const int r0 = int(rb + (sgr + r) * 16);
    frag_put_in(lo0[r], sums, int(a.n), r0, c0, home, int(rb + job.rows_here), int(nb + job.cols_here));
    frag_put_in(hi0[r], sums, int(a.n), r0, c0 + 16, home, int(rb + job.rows_here), int(nb + job.cols_here));
    frag_put_in(lo1[r], sums, int(a.n), r0, c0 + 32, home, int(rb + job.rows_here), int(nb + job.cols_here));
    frag_put_in(hi1[r], sums, int(a.n), r0, c0 + 48, home, int(rb + job.rows_here), int(nb + job.cols_here));
  }
  if (Mmas == 0 and tid < 256u) {
    // Keep every decoded store live: read the states this thread wrote, not just the first.
    float acc = 0;
    TF_UNROLL
    for (uint k = 0; k < states; ++k) acc += float(wx[cur_buf * (16u * 64u * CG) + walk.at[k]]);
    sums[tid] = acc;
  }
}
