kernel void tf_exl3_matmul_prefill_nax72(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, true, true, false, false, 1, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax73(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, true, 0, 2>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

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
    tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
  } else {
    tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
  }
}

// THE EXPERT'S SLAB DECODED ONCE FOR THE TILE'S ROWS . The values, their order, the k order and the MMA order are the
// shipped body's, so the output is bit-identical.
kernel void tf_exl3_matmul_prefill_nax76(const device half* x [[buffer(0)]],
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
  const uint rb = tg.y * 128u;
  uint e = 0u;
  if (a.indexed != 1u and rb < a.rows) e = routes[rb];
  if (a.experts != 0u and e >= a.experts) e = 0u;
  const uint halves = (a.indexed != 0u) ? descriptors[e].halves : a.halves;
  if (halves == 6u) {
    tf_exl3_prefill_body<true, 8, 1, false, true, false, false, true, false, false, 2, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
  } else {
    tf_exl3_prefill_body<true, 8, 1, false, true, false, false, true, false, false, 2, false, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
  }
}

// The same form one step further -- TWO HUNDRED AND FIFTY-SIX ROWS TALL with FOUR ROW GROUPS A WARP . Sixteen
// accumulator fragments a lane: the register wall is the price of the form.
kernel void tf_exl3_matmul_prefill_nax77(const device half* x [[buffer(0)]],
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
  const uint rb = tg.y * 256u;
  uint e = 0u;
  if (a.indexed != 1u and rb < a.rows) e = routes[rb];
  if (a.experts != 0u and e >= a.experts) e = 0u;
  const uint halves = (a.indexed != 0u) ? descriptors[e].halves : a.halves;
  if (halves == 6u) {
    tf_exl3_prefill_body<true, 16, 1, false, true, false, false, true, false, false, 4, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
  } else {
    tf_exl3_prefill_body<true, 16, 1, false, true, false, false, true, false, false, 4, false, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
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
  tf_exl3_prefill_body<true, 8, 1, false, true, false, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The routed 64 x 64 prompt tile with the funnel read and the extraction in thirty-two-bit words (the funnel's own
// dig-job structure). The windows, the values, their order and the k order are the shipped funnel's, so the output is
// bit-identical.
kernel void tf_exl3_matmul_prefill_nax50(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The routed 64 x 64 prompt tile with the funnel read and the block's words delivered by a cooperative load and a
// shuffle (one lane a word loads the block's twenty-four words; the two a lane needs arrive by a shuffle). The words,
// the windows, the values and the order are the shipped funnel's, so the output is bit-identical.
kernel void tf_exl3_matmul_prefill_nax52(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, false, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The routed 64 x 64 prompt tile with the funnel read, the grouping's padded rows skipped, and the block's words
// delivered by a cooperative load and a shuffle (one lane a word loads the block's twenty-four words; the two a lane
// needs arrive by a shuffle). The words, the windows, the values and the order are the shipped tile's, so the output
// is bit-identical.
kernel void tf_exl3_matmul_prefill_nax53(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
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
  tf_exl3_prefill_body<true, 8, 1, false, true, false, false, true, false, false, 2>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
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

// The routed prompt tile, double-buffered: two 16x128 tiles (8 KB) and one barrier a k block.
kernel void tf_exl3_matmul_prefill_nax7(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 128];
  tf_exl3_prefill_body_db<4, 2, true, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

// Diagnostics: is the same tile with the per-block decode removed (the prologue tile reused), keeps the decode but
// issues one of the four multiplies. - isolates the decode, isolates the multiply.
kernel void tf_exl3_matmul_prefill_nax8(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 128];
  tf_exl3_prefill_body_db<4, 2, false, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax9(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 128];
  tf_exl3_prefill_body_db<4, 2, true, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

// The routed tile with the codebook as a resident table (buffer 7): the mul1 state maps to a half by one load.
kernel void tf_exl3_matmul_prefill_nax10(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 128];
  tf_exl3_prefill_body_db<4, 2, true, 4, true>(wx, x, trellis, sums, a, routes, descriptors, table, lane, sg, tid, tptg.x, tg);
}




// Wider column tiles: CG=4 re-reads the activation tile nine times instead of eighteen, CG=8 five times.
kernel void tf_exl3_matmul_prefill_nax13(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 256];
  tf_exl3_prefill_body_db<4, 4, true, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax14(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 512];
  tf_exl3_prefill_body_db<4, 8, true, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}


kernel void tf_exl3_matmul_prefill_nax16(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 128];
  tf_exl3_prefill_body_db<4, 2, true, 0, true>(wx, x, trellis, sums, a, routes, descriptors, table, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax17(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 128];
  tf_exl3_prefill_body_db<4, 2, true, 0, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

// The 64x64 grouped tile with the table decode and the double buffer.
kernel void tf_exl3_matmul_prefill_nax18(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_db<4, 1, true, 4, true>(wx, x, trellis, sums, a, routes, descriptors, table, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax19(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_db<4, 1, true, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax20(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_db<4, 1, false, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax21(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_db<4, 1, true, 0, true>(wx, x, trellis, sums, a, routes, descriptors, table, lane, sg, tid, tptg.x, tg);
}

// The engine's 64x64 grouped tile with the codebook table and the shipped single buffer: the table gain without the
// double buffer's extra 2 KB of threadgroup memory (which cost more occupancy than it returned).
kernel void tf_exl3_matmul_prefill_nax22(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[16 * 64];
  tf_exl3_prefill_body<true, 4, 1, true>(wx, x, trellis, sums, a, routes, descriptors, table, nullptr, lane, sg, tid, tptg.x, tg);
}

// Activation-fragment reuse: four warps (one a row group), each decodes two 16x64 weight tiles into one 16x128
// buffer, loads its `av` fragment once, and multiplies both tiles.
