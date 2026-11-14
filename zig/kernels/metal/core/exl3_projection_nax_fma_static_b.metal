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
