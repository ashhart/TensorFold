// The routed 64x64 prompt tile with the funnel path as the only path.
kernel void tf_exl3_matmul_prefill_nax59(const device half* x [[buffer(0)]],
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
  threadgroup half wx[2 * TF_KU * 16 * 64];
  // The funnel path is the one the routed pack's width (halves = 6) takes; the expert's own descriptor says so.
  const uint rb = tg.y * 64u;
  uint e = 0u;
  if (a.indexed != 1u and rb < a.rows) e = routes[rb];
  if (a.experts != 0u and e >= a.experts) e = 0u;
  const uint halves = (a.indexed != 0u) ? descriptors[e].halves : a.halves;
  if (halves == 6u and a.k % (16u * TF_KU) == 0u) {
    tf_exl3_prefill_body_ku<4, 1, TF_KU, 1, true>(wx, x, trellis, sums, a, routes, descriptors, lane, sg, tid, tptg.x, tg);
  } else if (halves == 6u) {
    tf_exl3_prefill_body<true, 4, 1, false, true, true, 1, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
  } else {
    tf_exl3_prefill_body<true, 4, 1, false, true, true, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
  }
}

// The routed 128-row tile (eight row groups, eight warps): twice the rows a tile, so half the per-tile weight decodes.
kernel void tf_exl3_matmul_prefill_nax49(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 8, 1, false, true, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The routed 128-row tile with a warp doing thirty-two rows: one warp decodes the block and a lane keeps eight states.
kernel void tf_exl3_matmul_prefill_nax54(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 8, 1, false, true, true, 2>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}


// Sixteen row groups by one column group, every warp decoding: a long chunk's expert tile.
kernel void tf_exl3_matmul_prefill_nax6(const device half* const x [[buffer(0)]],
                                       const device uchar* const trellis [[buffer(1)]],
                                       device float* const sums [[buffer(2)]],
                                       constant TfExl3Args& a [[buffer(3)]],
                                       const device uint* const routes [[buffer(4)]],
                                       const device TfExl3Expert* const descriptors [[buffer(5)]],
                                       uint lane [[thread_index_in_simdgroup]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint3 tptg [[threads_per_threadgroup]],
                                       uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[16 * 64];
  tf_exl3_prefill_body<true, 16, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}
