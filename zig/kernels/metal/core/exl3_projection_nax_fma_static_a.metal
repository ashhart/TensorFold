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
  // The LUT holds the mul1 values, so it is only the right table for that codebook.
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

// The register-only body on the DENSE one-row path.
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
  // The mul1 codebook's values depend only on the state's byte sum, so 1,024 entries hold every value.
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

// The multi-row dense body, two rows a threadgroup and four rows a threadgroup.
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
