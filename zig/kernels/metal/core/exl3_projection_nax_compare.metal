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

