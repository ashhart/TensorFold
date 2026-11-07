// The routed 64x64 prompt tile with the funnel path as the only path. The cursor arrays the Fast path needs -- off,
// sh and rel, twenty-four registers -- are dead in the funnel path but live across the k loop, so a body that always
// takes the funnel path would carry them for nothing. Here they are not computed. The values, their order, the k
// order and the MMA order are the general body's, so the output is bit-identical.
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
  threadgroup half wx[16 * 64];
  // The funnel path is the one the routed pack's width (halves = 6) takes; the expert's own descriptor says so, so
  // the kernel picks the specialised body per expert and falls back to the shipped one otherwise.
  const uint rb = tg.y * 64u;
  uint e = 0u;
  if (a.indexed != 1u and rb < a.rows) e = routes[rb];
  if (a.experts != 0u and e >= a.experts) e = 0u;
  const uint halves = (a.indexed != 0u) ? descriptors[e].halves : a.halves;
  if (halves == 6u) {
    tf_exl3_prefill_body<true, 4, 1, false, true, true, 1, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
  } else {
    tf_exl3_prefill_body<true, 4, 1, false, true, true, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
  }
}

// The routed 128-row tile (eight row groups, eight warps). A 128-row tile halves that (the plan's own bound: 384 +
// (21,714 - 384) / 128 tiles), and the weight decode is per tile, so the tile's decode cost falls with it. The eight
// warps make a lane own four states instead of eight, so the funnel reads a twenty-five-bit run (four windows)
// instead of a thirty-seven-bit one; the values, their order, the k order and the per-row arithmetic are the shipped
// body's, so the reply is bit-identical.
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

// The routed 128-row tile with a warp that does thirty-two rows (the NAXTile shape: a simdgroup owns thirty-two rows,
// so the block is decoded by one warp and the lane keeps eight states and eight bits a weight). The tile covers twice
// the rows, so the grouped plan spends half the tiles, and the weight fragments are read once for two row groups. The
// values, their order, the k order and the MMA order per output are the shipped tile's, so the output is
// bit-identical.
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
