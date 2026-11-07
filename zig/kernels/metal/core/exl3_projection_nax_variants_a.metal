kernel void tf_exl3_matmul_prefill_nax3(const device half* x [[buffer(0)]],
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
  threadgroup half wx[16 * 128];
  tf_exl3_prefill_body<true, 4, 2, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax4(const device half* x [[buffer(0)]],
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
  threadgroup half wx[16 * 128];
  tf_exl3_prefill_body<true, 8, 2, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax5(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}


// The OCCUPANCY-KEEPING FORM -- the routed prompt tile with the activation staged in threadgroup memory, once a k
// block, and SHARED by the two column groups of a row group (one column tile's activation read shared by two column
// groups; the fragment in threadgroup memory; one more barrier a block -- here the SAME barrier that publishes the
// weight tile, so the block's barrier count is unchanged). The two column groups of a row group read the device ONCE
// and read the staged fragment back, so the activation's device traffic falls by half against the same tile widened
// to 128 columns. The values, their order, the k order and the MMA order are the shipped body's, so the output is
// bit-identical.
kernel void tf_exl3_matmul_prefill_nax61(const device half* x [[buffer(0)]],
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
  threadgroup half wx[16 * 64 * 2 + 4 * 16 * 16];
  tf_exl3_prefill_body<true, 4, 2, false, true, false, false, true, false, false, 1, false, false, 2>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The ABLATION -- the activation fragment is read ONCE, before the k loop, and reused. A diagnostic: the output
// differs by construction, and the delta against the same body reading it every block IS the activation's per-block
// cost (the record's instrument first: the activation read's share of the routed tile).
kernel void tf_exl3_matmul_prefill_nax62(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, false, 1>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The CONTROL for -- the same 128-column tile with the shipped per-warp activation read (so the two arms differ only
// in the staging). Bit-identical to by construction.
kernel void tf_exl3_matmul_prefill_nax63(const device half* x [[buffer(0)]],
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
  threadgroup half wx[16 * 128];
  tf_exl3_prefill_body<true, 4, 2, false, true, false, false, true, false, false, 1, false, false, 0>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The engine's routed 64 x 64 prompt tile with the funnel read -- two stream words a lane, the eight windows at one
// shift a state, on our map and our layout. The values, their order and the k order are the shipped body's, so the
// output is bit-identical; only the word traffic and the cursor's select chain leave.
kernel void tf_exl3_matmul_prefill_nax40(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The same body with the mul1 value dropped (the raw state stored): the window extraction and the stores without the
// codebook arithmetic. A diagnostic -- it isolates the value path inside the decode; its output differs.
kernel void tf_exl3_matmul_prefill_nax41(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The engine's routed 64 x 64 prompt tile with the funnel read -- the two-word funnel and the eight windows at
// compile-time shifts, on our map and our ring. The values, their order and the k order are the shipped body's, so
// the output is bit-identical; only the word traffic and the cursor's select chain leave. The funnel needs the routed
// pack's width (halves = 6, three bits a state); at any other width the kernel reads the shipped path's four words
// and decodes as before.
kernel void tf_exl3_matmul_prefill_nax42(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The funnel read with the value read from the resident codebook table (buffer 7). One load a state instead of the
// mul1's ALU. The table holds the format's own decoder's values, so the values, their order and the k order are the
// shipped body's: bit-identical.
kernel void tf_exl3_matmul_prefill_nax43(const device half* x [[buffer(0)]],
                                        const device uchar* trellis [[buffer(1)]],
                                        device float* sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device uint* routes [[buffer(4)]],
                                        const device TfExl3Expert* descriptors [[buffer(5)]],
                                        const device half* table [[buffer(7)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tptg [[threads_per_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[16 * 64];
  tf_exl3_prefill_body<true, 4, 1, false, true, false>(wx, x, trellis, sums, a, routes, descriptors, table, nullptr, lane, sg, tid, tptg.x, tg);
}

// The funnel read with the value read from the 1024-entry threadgroup LUT (the prefill kernel's read). The LUT's
// entries are the table's, so the values are the same, bit-identical.
kernel void tf_exl3_matmul_prefill_nax44(const device half* x [[buffer(0)]],
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
  threadgroup half lut[1024];
  const half scale = as_type<half>(ushort(0x1EEEu));
  const half bias = as_type<half>(ushort(0xC931u));
  TF_UNROLL
  for (uint j = 0; j < 8u; ++j) lut[tid + 128u * j] = half(fma(float(tid + 128u * j + 1024u), float(scale), float(bias)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tf_exl3_prefill_body<true, 4, 1, false, true, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lut, lane, sg, tid, tptg.x, tg);
}

// The funnel read with the value read cheaply -- a diagnostic; its output differs by construction.
kernel void tf_exl3_matmul_prefill_nax47(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The double-buffered routed tile with the funnel read. The double-buffered body decodes the next k block into the
// tile the multiply is not reading and issues the multiply before that decode, so the tensor units and the codebook
// ALU are in flight together; the funnel makes the decode cheap enough for the overlap to pay.
kernel void tf_exl3_matmul_prefill_nax46(const device half* x [[buffer(0)]],
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
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_db<4, 1, true, 4, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

// The routed 64 x 64 prompt tile with the funnel read and the padded rows skipped. The engine's grouping pads each
// expert's block to sixty-four rows, so the routed rows are dispatched with padding; the padding rows carry the
// sentinel expert id, so a warp reads its own sixteen route ids
// and skips its multiply when they are all padding. The real rows' values, their order and the k order are the
// shipped body's, so the reply is bit-identical. The tile's contents change, so the output differs by construction.
kernel void tf_exl3_matmul_prefill_nax64(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, false, 0, 3>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// the DECODE KEPT, the eight scattered stores GONE (the values are folded into a register accumulator
// instead). The floor of the store path. A diagnostic: the tile is never written.
kernel void tf_exl3_matmul_prefill_nax65(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, false, 0, 1>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The shipped body plus the Mode 1 accumulator: the calibration for the accumulator's own cost, so the store
// path's cost falls out of the two.
kernel void tf_exl3_matmul_prefill_nax66(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, false, 0, 4>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// the shipped body with the codebook arithmetic dropped (the raw window stored, the extraction and the
// stores kept). The value path's price, in the shipped configuration. A diagnostic.
kernel void tf_exl3_matmul_prefill_nax67(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, true, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// the eight stores KEPT, the EXTRACTION dropped (the stored value is a cheap function of the k block, so the
// stores stay in the loop). The extraction's price, in the shipped configuration.
kernel void tf_exl3_matmul_prefill_nax68(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, false, 0, 2>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax48(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// The routed 64x64 prompt tile with the stream word read as one 4-byte load -- the wide form's read (its lane reads
// 32-bit stream words; ours read the same bytes as two 2-byte loads, so a lane issued four memory instructions a
// block where it now issues two). The values, their order, the k order and the MMA order are the single-tile body's,
// so the output is bit-identical. The kernel falls back to the two-load read when the tile's byte address is not
// 4-byte aligned.
kernel void tf_exl3_matmul_prefill_nax57(const device half* x [[buffer(0)]],
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
  // The tile's byte address: the expert's trellis plus the tile's k and n offsets. A 4-byte load needs it 4-byte
  // aligned, and the strides must keep it there.
  const uint rb = tg.y * 64u;
  uint e = 0u;
  if (a.indexed != 1u and rb < a.rows) e = routes[rb];
  if (a.experts != 0u and e >= a.experts) e = 0u;
  const device uchar* data = (a.indexed != 0u) ? descriptors[e].trellis : trellis;
  const ulong ks = (a.indexed != 0u) ? descriptors[e].tile_k_stride : a.tile_k_stride;
  const ulong ns = (a.indexed != 0u) ? descriptors[e].tile_n_stride : a.tile_n_stride;
  const bool wide = (reinterpret_cast<ulong>(data) & 3ul) == 0ul and (ks & 3ul) == 0ul and (ns & 3ul) == 0ul;
  if (wide) {
    tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
  } else {
    tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
  }
}

// The store path's modes on the shipped body (the funnel path as its only path, the twenty-four dead cursor
// registers not computed). The store path's costs are only worth anything on the body the engine runs. Mode 1: the
// eight decoded values written CONTIGUOUSLY, one lane's run of eight halves -- the wide form's store form (a lane's
// values go out as wide stores, not as eight scattered two-byte stores). Mode 2: the decode KEPT, the eight scattered
// stores GONE (the values are folded into a register accumulator). The floor of the store path. Mode 3: the shipped
// store PLUS the accumulator -- the calibration for the accumulator's own cost. Mode 4: the codebook arithmetic
// dropped (the extraction and the stores kept) -- the value path's cost. Mode 5: the eight stores KEPT, the
// EXTRACTION dropped -- the extraction's cost. Modes 2-5 are diagnostics: their outputs differ by construction.
kernel void tf_exl3_matmul_prefill_nax69(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, true, 0, 3>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax70(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, true, 0, 1>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax71(const device half* x [[buffer(0)]],
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
  tf_exl3_prefill_body<true, 4, 1, false, true, false, false, true, false, false, 1, false, true, 0, 4>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

