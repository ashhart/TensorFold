kernel void tf_exl3_contract_fma_reference(const device half* x [[buffer(0)]],
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
  threadgroup float red[TF_FMA_WARPS][64];
#endif
  const uint row = tg.y;
  const uint nb = tg.x * 16u;
  if (row >= a.rows) return;
  const uint expert = routes[row];
  if (expert >= a.experts) return;
  if (nb >= (descriptors[expert].reserved != 0 ? descriptors[expert].reserved : a.n)) return;
  const device uchar* data = descriptors[expert].trellis;
  const ulong ks = descriptors[expert].tile_k_stride;
  const ulong ns = descriptors[expert].tile_n_stride;
  const uint halves = descriptors[expert].halves;
  const uint codebook = descriptors[expert].codebook;
  const uint blocks = a.k / 16;
  const uint W = TF_FMA_WARPS;
  const uint b0 = blocks * sg / W, b1 = blocks * (sg + 1u) / W;
  uint off[8], sh[8], rel[8];
  const uint words = 4 * halves;
  const uint ring = 128 * halves;
  const uint wi0 = tf_exl3_map_for(lane * 8u, halves) & 63u;
#if defined(TF_DECODE_FUNNEL)
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
  const bool pipeline = (halves & 1u) == 0 and halves <= 16;
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
#if defined(TF_DIAG_DECODE) && TF_DIAG_DECODE == 4
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
  float acc0 = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  float acc1 = (acc[4] + acc[5]) + (acc[6] + acc[7]);
#if TF_FMA_WARPS > 1
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
