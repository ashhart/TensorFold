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
  // One routed row's pack slice; a column reads the tiles its sixteen states share.
  const TfExl3DecodeJob job = tf_exl3_routed_decode_job(trellis, a, descriptors, expert);
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
    const device ushort* tile = reinterpret_cast<const device ushort*>(job.trellis + ulong(b) * job.tile_k_stride + ulong(cb) * job.tile_n_stride);
    const ushort s = tf_exl3_mapped_state(tile, tf_exl3_map_for(p, job.halves));
    const float v = float(tf_exl3_table_value(table, s, job.codebook));
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


