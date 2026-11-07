template <bool Table>
inline void tf_exl3_prefill_body_reuse(threadgroup half* wx, const device half* x, const device uchar* trellis,
                                       device float* sums, constant TfExl3Args& a, const device uint* routes,
                                       const device TfExl3Expert* descriptors, const device half* table,
                                       uint lane, uint sg, uint tid, uint3 tg) {
  const uint rb = tg.y * 64u, nb = tg.x * 128u;
  if (a.indexed == 1 || rb >= a.rows) return;
  const uint tile_expert = a.indexed ? routes[rb] : a.expert_id;
  if (a.indexed && tile_expert >= a.experts) return;
  const uint didx = a.indexed ? tile_expert : 0;
  const device uchar* data = a.indexed ? descriptors[didx].trellis : trellis;
  const ulong ks = a.indexed ? descriptors[didx].tile_k_stride : a.tile_k_stride;
  const ulong ns = a.indexed ? descriptors[didx].tile_n_stride : a.tile_n_stride;
  const uint halves = a.indexed ? descriptors[didx].halves : a.halves;
  const uint codebook = a.indexed ? descriptors[didx].codebook : a.codebook;
  const short2 home = frag_home(ushort(lane));
  const uint rows_here = min(64u, a.rows - rb);
  const uint cols_here = min(128u, a.n - nb);
  const uint h = sg; // four warps, one a row group
  const uint p0 = lane * 8u;
  const uint words = 4 * halves;
  const uint wi0 = tf_exl3_map_for(p0, halves) & 63u;
  uint atA[8], atB[8], off[8], sh[8], rel[8];
  {
    const uint ring = 128 * halves;
    const uint cur = (wi0 << 5) | ((tf_exl3_map_for(p0, halves) >> 6) & 31u);
    TF_UNROLL
    for (uint k = 0; k < 8; ++k) {
      const uint p = p0 + k, l = p / 8, e = p % 8;
      const uint at = (2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u)) * 128u + h * 16u + (l / 4 + 8 * (e >> 2));
      atA[k] = at;
      atB[k] = at + 64u;
      uint c8 = cur + k * (halves >> 1);
      if (c8 >= ring) c8 -= ring;
      const uint wi = c8 >> 5;
      off[k] = c8 & 31u;
      sh[k] = (32u - off[k]) & 31u;
      rel[k] = wi >= wi0 ? wi - wi0 : wi + words - wi0;
    }
  }
  frag<float> lo0 = frag<float>(0), hi0 = frag<float>(0), lo1 = frag<float>(0), hi1 = frag<float>(0);
  frag<float> lo2 = frag<float>(0), hi2 = frag<float>(0), lo3 = frag<float>(0), hi3 = frag<float>(0);
  for (uint kb = 0; kb < a.k; kb += 16) {
    const uint nA0 = tf_exl3_block_word(data, ks, ns, kb, nb, 0, h, wi0, words);
    const uint nA1 = tf_exl3_block_word(data, ks, ns, kb, nb, 0, h, wi0 + 1, words);
    const uint nA2 = tf_exl3_block_word(data, ks, ns, kb, nb, 0, h, wi0 + 2, words);
    const uint nA3 = tf_exl3_block_word(data, ks, ns, kb, nb, 0, h, wi0 + 3, words);
    const uint nB0 = tf_exl3_block_word(data, ks, ns, kb, nb + 64u, 0, h, wi0, words);
    const uint nB1 = tf_exl3_block_word(data, ks, ns, kb, nb + 64u, 0, h, wi0 + 1, words);
    const uint nB2 = tf_exl3_block_word(data, ks, ns, kb, nb + 64u, 0, h, wi0 + 2, words);
    const uint nB3 = tf_exl3_block_word(data, ks, ns, kb, nb + 64u, 0, h, wi0 + 3, words);
    if (Table) {
      tf_exl3_decode_table<8>(wx, table, codebook, atA, off, sh, rel, nA0, nA1, nA2, nA3);
      tf_exl3_decode_table<8>(wx, table, codebook, atB, off, sh, rel, nB0, nB1, nB2, nB3);
    } else {
      tf_exl3_decode_fast<8>(wx, codebook, atA, off, sh, rel, nA0, nA1, nA2, nA3);
      tf_exl3_decode_fast<8>(wx, codebook, atB, off, sh, rel, nB0, nB1, nB2, nB3);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    frag<half> av;
    frag_get_in(av, (const device half*)x, int(a.k), int(rb + h * 16), int(kb), home, int(rb + rows_here), int(kb) + 16);
    frag<half> b0, b1;
    tf_frag_get_tg4(b0, wx, 128, 0, 0, home);
    tf_frag_get_tg4(b1, wx, 128, 0, 16, home);
    mma_16x32<false, false>(lo0, hi0, av, b0, b1);
    tf_frag_get_tg4(b0, wx, 128, 0, 32, home);
    tf_frag_get_tg4(b1, wx, 128, 0, 48, home);
    mma_16x32<false, false>(lo1, hi1, av, b0, b1);
    tf_frag_get_tg4(b0, wx, 128, 0, 64, home);
    tf_frag_get_tg4(b1, wx, 128, 0, 80, home);
    mma_16x32<false, false>(lo2, hi2, av, b0, b1);
    tf_frag_get_tg4(b0, wx, 128, 0, 96, home);
    tf_frag_get_tg4(b1, wx, 128, 0, 112, home);
    mma_16x32<false, false>(lo3, hi3, av, b0, b1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const int r0 = int(rb + h * 16);
  const int c0 = int(nb);
  frag_put_in(lo0, sums, int(a.n), r0, c0, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi0, sums, int(a.n), r0, c0 + 16, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo1, sums, int(a.n), r0, c0 + 32, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi1, sums, int(a.n), r0, c0 + 48, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo2, sums, int(a.n), r0, c0 + 64, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi2, sums, int(a.n), r0, c0 + 80, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo3, sums, int(a.n), r0, c0 + 96, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi3, sums, int(a.n), r0, c0 + 112, home, int(rb + rows_here), int(nb + cols_here));
}

kernel void tf_exl3_matmul_prefill_nax23(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[16 * 128];
  tf_exl3_prefill_body_reuse<false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tg);
}

kernel void tf_exl3_matmul_prefill_nax24(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[16 * 128];
  tf_exl3_prefill_body_reuse<true>(wx, x, trellis, sums, a, routes, descriptors, table, lane, sg, tid, tg);
}

// The 64x64 grouped tile with the mul1 byte-sum LUT built in threadgroup memory (1024 halves, 2 KB): the decode
// shape, no 384 KB table, no per-weight L2 miss.
kernel void tf_exl3_matmul_prefill_nax25(const device half* const x [[buffer(0)]],
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
  threadgroup half lut[1024];
  const half scale = as_type<half>(ushort(0x1EEEu));
  const half bias = as_type<half>(ushort(0xC931u));
  TF_UNROLL
  for (uint j = 0; j < 8u; ++j) lut[tid + 128u * j] = half(fma(float(tid + 128u * j + 1024u), float(scale), float(bias)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tf_exl3_prefill_body<true, 4, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lut, lane, sg, tid, tptg.x, tg);
}

// Dense 128x64 tile: double-buffered with the codebook table, and single-buffer with the threadgroup LUT.
kernel void tf_exl3_matmul_prefill_nax26(const device half* const x [[buffer(0)]],
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
  tf_exl3_prefill_body_db<8, 1, true, 4, true>(wx, x, trellis, sums, a, routes, descriptors, table, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax27(const device half* const x [[buffer(0)]],
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
  threadgroup half lut[1024];
  const half scale = as_type<half>(ushort(0x1EEEu));
  const half bias = as_type<half>(ushort(0xC931u));
  TF_UNROLL
  for (uint j = 0; j < 4u; ++j) lut[tid + 256u * j] = half(fma(float(tid + 256u * j + 1024u), float(scale), float(bias)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tf_exl3_prefill_body<true, 8, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lut, lane, sg, tid, tptg.x, tg);
}

// Dense 128x128 tile: sixteen warps, the tile shape, with the 1024-entry LUT amortised over twice the decode (2
// entries built a thread, 4 states decoded) and half the activation re-read.
kernel void tf_exl3_matmul_prefill_nax28(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[16 * 128];
  threadgroup half lut[1024];
  const half scale = as_type<half>(ushort(0x1EEEu));
  const half bias = as_type<half>(ushort(0xC931u));
  TF_UNROLL
  for (uint j = 0; j < 2u; ++j) lut[tid + 512u * j] = half(fma(float(tid + 512u * j + 1024u), float(scale), float(bias)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tf_exl3_prefill_body<true, 8, 2, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lut, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax29(const device half* const x [[buffer(0)]],
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
  threadgroup half wx[16 * 128];
  tf_exl3_prefill_body<true, 8, 2, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, nullptr, lane, sg, tid, tptg.x, tg);
}

// Diagnostics for the dense 128x64 tile (RG=8, 256 threads), the attention-dense body: the same body with the
// per-block decode removed (the prologue tile reused), and the same body that keeps it but issues one of the four
// multiplies. They isolate the decode at the dense tile and the decode-free multiply.
kernel void tf_exl3_matmul_prefill_nax30(const device half* const x [[buffer(0)]],
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
  tf_exl3_prefill_body_db<8, 1, false, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax31(const device half* const x [[buffer(0)]],
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
  tf_exl3_prefill_body_db<8, 1, true, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

// THE DOUBLE-BUFFERED 128-COLUMN DENSE BODY. The per-output decode, k order and MMA order are the single-tile body's,
// at twice the tile width: one activation block is read for 128 columns instead of 64 and one barrier covers twice
// the multiply. The first kernel reads the codebook from the resident device table (buffer 7); the second is the
// same body on the ALU codebook. Both must agree with the single-tile body bit for bit at every output.
kernel void tf_exl3_matmul_prefill_nax79(const device half* const x [[buffer(0)]],
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
  tf_exl3_prefill_body_db<8, 2, true, 4, true>(wx, x, trellis, sums, a, routes, descriptors, table, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax80(const device half* const x [[buffer(0)]],
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
  tf_exl3_prefill_body_db<8, 2, true, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

// The 64-row x 128-column double-buffered dense body: the same 8,192-element tile area as the 128-row form, eight
// warps / 256 threads, eight decode groups so one threadgroup owns all 128 columns the fused output transform needs.
// The k order and MMA order per output are the single-tile body's.
kernel void tf_exl3_matmul_prefill_nax82(const device half* const x [[buffer(0)]],
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



// The dense 128x64 prompt tile with the funnel read. The dense tile is 128 rows (eight row groups), so
// a lane owns four states and the funnel reads a twenty-five-bit run -- two words, where the cursor path reads four.
// The values, their order, the k order and the MMA order are the single-tile body's, so the output is bit-identical;
// only the word traffic and the select chain leave.
kernel void tf_exl3_matmul_prefill_nax51(const device half* const x [[buffer(0)]],
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
  tf_exl3_prefill_body_db<8, 1, true, 4, false, true>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

// The dense 128x64 prompt tile with a SIMDgroup that owns thirty-two rows -- the wide form's NAXTile shape (a
// simdgroup owns thirty-two rows, so one threadgroup's eight row groups become four groups of two). The lane keeps
// eight states, so the stream words it reads cover twice the states, and the weight fragments are read once for two
// row groups, so the tile's threadgroup traffic per row halves. The values, their order, the k order and the MMA
// order per output are the single-tile body's, so the output is bit-identical.
kernel void tf_exl3_matmul_prefill_nax56(const device half* const x [[buffer(0)]],
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
  tf_exl3_prefill_body_db<8, 1, true, 4, false, false, 2>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

kernel void tf_exl3_matmul_prefill_nax32(const device half* const x [[buffer(0)]],
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
  tf_exl3_prefill_body_db<8, 1, true, 1, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

// Dense row-height diagnostics: RG=16 (256 rows a threadgroup, 512 threads) and RG=32 (512 rows, 1024 threads). The
// decoded 16x64 weight tile is shared by every row group, so the per-weight decode halves at RG=16 and quarters at
// RG=32; the per-row MMA order is unchanged (each SIMDgroup still owns its sixteen rows).
kernel void tf_exl3_matmul_prefill_nax33(const device half* const x [[buffer(0)]],
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




// The register-only decode structure on our map -- one threadgroup a sixteen-column tile, one k walk, no split-K, no
// partials and no reduce dispatch, and the decoded weights stay in REGISTERS: there is no threadgroup weight tile, no
// barrier in the k loop and no threadgroup memory round trip. One warp owns a sixteen-column tile of one row, each
// lane holds two output columns and the four k rows it reads, the eight decoded weights stay in registers, and the
// four lanes that share a column reduce once, at the end of the walk. TF_FMA_WARPS: the warps a threadgroup. Each
// warp walks its own slice of the k blocks and the warps' partial sums are added once, at the end of the walk,
// through a threadgroup reduction -- so a split-K dispatch's parallelism without its partials round trip and its
// reduce dispatch. Our map is unchanged: the lane's eight states are the same eight the tensor-unit body reads (the
// same stream words, the same bit offsets, the same codebook values), so the products are the same products. The
// reduction order is the FP32 walk's, not the tensor unit's: the sums are FP32 either way, but the tensor unit
// reduces sixteen k at a time and this body adds them one at a time, so the two are not bit-identical (the numerics
// band shows the difference).
#ifndef TF_FMA_WARPS
#define TF_FMA_WARPS 1
#endif
// THE PREFETCH DEPTH OF THE REGISTER-ONLY BODY'S K-WALK. The shipped body keeps ONE block of stream words in flight
// (the next block's three words are read at the top of the walk and consumed at the top of the next step), so a
// block's DRAM latency is exposed once a block unless the block's own work hides it -- and a block's work is eight
// states, far less than a DRAM round trip. TF_FMA_PF blocks in flight (a ring of word triples) hides it behind PF
// blocks of work. The values, their order and the arithmetic are untouched: the ring only moves the LOADS in time, so
// every arm is BIT-IDENTICAL to the shipped body.
#ifndef TF_FMA_PF
#define TF_FMA_PF 1
#endif
// THE STREAM WORD AS ONE 4-BYTE LOAD. The default read is two 2-byte loads a word (a stream word is a little-endian
// pair of int16s), so the decode's lane issues SIX load instructions a k-block where the wide form issues three; the
// bytes, the values and the order are the same, so the arm is BIT-IDENTICAL. The prefill tiles already read their
// words this way (tf_exl3_word_t<W> with W = true); the register-only decode body did not.
// TF_FMA_WIDE_WORD = 1 turns it on there.
#ifndef TF_FMA_WIDE_WORD
#define TF_FMA_WIDE_WORD 0
#endif

// THE TWO-WORD FUNNEL AT halves = 6 IN THE STATIC BODY. The static body reads three stream words a lane a block (a
// run of 7*S + 16 bits; at S = 3 that is 37 bits), but at halves = 6 every lane run starts at off0 in {3,11,19,27},
// so the run is contained in two 32-bit words and only the top five bits of the funnel second register are read
// (states J = 6, 7 at bit 18/21 use c_next >> 30 / >> 27). The third word feeds only the low off0 <= 27 bits of
// c_next, which the extraction never reads. Dropping it is bit-identical BY CONSTRUCTION (the same argument as routed
// funnel); The two forms are hash-identical.
#ifndef TF_FMA_TWO
#define TF_FMA_TWO 0
#endif
// THE VALUE PATH'S FORM. The alternative form is the ARITHMETIC one the format's own decoder uses (tf_exl3_value: the
// same byte sum, the same half, the same fma) -- the LUT's entries ARE those values, entry for entry, so the two
// forms are BIT-IDENTICAL and the choice is a pure cost question. TF_FMA_VALUE_ARITH = 1 takes the arithmetic form.
#ifndef TF_FMA_VALUE_ARITH
#define TF_FMA_VALUE_ARITH 0
#endif
