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
// the single-tile body's, so `sums` is bit-identical; only the stage depth, the warp tile and the cooperative decode
// differ. The stage is
// DOUBLE-BUFFERED and the next stage's decode is issued before the current stage's multiply, so the codebook ALU and
// the stream loads overlap the tensor units (the deeper stage, our overlap).
