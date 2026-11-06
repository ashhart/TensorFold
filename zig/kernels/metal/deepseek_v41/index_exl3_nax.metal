// Source is compiled after kv.metal and index_exl3.metal; the tensor-unit score kernels only.
using namespace tfp;

// ---------------------------------------------------------------- prompt scores on the tensor units
// The scalar scorer's arithmetic, a tile at a time: 64 keys by 64 rows, four SIMDgroups of 32x32, the
// 64x128 key tile staged once and read by every row group; one 128-dim transposed MMA a head instead of a
// warp a key with sixty-four simd_sums. The epilogue is the scalar kernel's own term -- bf16 dot, relu, times
// the bf16-rounded scaled weight, summed in fp32 in head order -- so only the dot's summation order differs.
constant constexpr int INAX_D = 128, INAX_PAD = INAX_D + 8;

inline frag<bfloat> inax_frag(const device bfloat* r0, const device bfloat* r1, int c, short2 home) {
#ifndef TF_SIMD_FRAGS
  return frag<bfloat>(*(const device bfloat4*)(r0 + c + home.x), *(const device bfloat4*)(r1 + c + home.x));
#else
  frag<bfloat> f;
  TF_UNROLL
  for (short e = 0; e < 8; e++) f[e] = (e < 4 ? r0 : r1)[c + home.x + TF_COL(e)];
  return f;
#endif
}

kernel void dsv41_index_scores_nax(device const ushort *qp [[buffer(0)]], device const ushort *wp [[buffer(1)]],
                                   device const ushort *kp [[buffer(2)]], device float *scores [[buffer(3)]],
                                   constant Dsv41Index &p [[buffer(4)]], uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const device bfloat* Q = reinterpret_cast<const device bfloat*>(qp);
  const device bfloat* W = reinterpret_cast<const device bfloat*>(wp);
  const device bfloat* K = reinterpret_cast<const device bfloat*>(kp);
  threadgroup bfloat tile[64 * INAX_PAD];
  const int t0 = int(p.tile_start) + int(tg.x) * 64, r0 = int(tg.y) * 64;
  const int rows = min(64, int(p.rows) - r0);
  const uint t = sg * 32u + lane;
  if (p.keys == 0 or p.ratio == 0) return;
  // A tile past the last row's visible keys scores -inf and is not staged.
  if (t0 >= int((p.position + uint(r0) + uint(rows)) / p.ratio)) {
    for (uint i = t; i < 64u * 64u; i += 128u) {
      const int r = int(i) / 64, k = t0 + int(i) % 64;
      if (r < rows && k < int(p.keys))
        *(dsv41_score_slot(scores, p, r0 + r, p.tile_start / p.tile_width) + (k - int(p.tile_start))) = -INFINITY;
    }
    return;
  }
  for (uint i = t; i < 64u * (INAX_D / 8u); i += 128u) {
    const uint rr = i / (INAX_D / 8u), cc = (i % (INAX_D / 8u)) * 8u;
    const int k = min(t0 + int(rr), int(p.keys) - 1);
    *(threadgroup uint4*)(tile + rr * INAX_PAD + cc) = *(const device uint4*)(K + size_t(k) * INAX_D + cc);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const short2 home = frag_home(ushort(lane));
  const int tm = 32 * int(sg / 2), tn = 32 * int(sg % 2);
  const device bfloat* qr[2][2];
  const device bfloat* wr[2][2];
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = tm + 16 * m + home.y + 8 * h;
      const int at = r0 + (r < rows ? r : 0); // a dead row reads a live one; its scores are never stored
      qr[m][h] = Q + size_t(at) * p.q_stride;
      wr[m][h] = W + size_t(at) * p.weight_stride;
    }
  }
  frag<float> total[2][2];
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    total[m][0] = frag<float>(0);
    total[m][1] = frag<float>(0);
  }
  for (int hd = 0; hd < int(p.heads); hd++) {
    frag<float> s[2][2];
    TF_UNROLL
    for (short m = 0; m < 2; m++) {
      s[m][0] = frag<float>(0);
      s[m][1] = frag<float>(0);
    }
    TF_UNROLL
    for (short q = 0; q < INAX_D / 16; q++) {
      frag<bfloat> k0, k1;
      frag_get_t(k0, (const threadgroup bfloat*)tile, INAX_PAD, tn, 16 * q, home);
      frag_get_t(k1, (const threadgroup bfloat*)tile, INAX_PAD, tn + 16, 16 * q, home);
      TF_UNROLL
      for (short m = 0; m < 2; m++) {
        const frag<bfloat> qa = inax_frag(qr[m][0] + hd * INAX_D, qr[m][1] + hd * INAX_D, 16 * q, home);
        mma_16x32<false, true>(s[m][0], s[m][1], qa, k0, k1);
      }
    }
    // the head's term as the scalar makes it: bf16 dot, relu, times the bf16-rounded scaled weight, summed in fp32
    TF_UNROLL
    for (short m = 0; m < 2; m++) {
      const float w0 = dsv41_bf(dsv41_round(float(wr[m][0][hd]) * p.scale));
      const float w1 = dsv41_bf(dsv41_round(float(wr[m][1][hd]) * p.scale));
      TF_UNROLL
      for (short j = 0; j < 2; j++) {
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          const float dot = dsv41_bf(dsv41_round(s[m][j][e]));
          total[m][j][e] += dsv41_bf(dsv41_round(max(dot, 0.0f) * (e < 4 ? w0 : w1)));
        }
      }
    }
  }
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int r = tm + 16 * m + home.y + (e >> 2) * 8;
        const int k = t0 + tn + 16 * j + home.x + TF_COL(e);
        if (r < rows && k < int(p.keys))
          *(dsv41_score_slot(scores, p, r0 + r, p.tile_start / p.tile_width) + (k - int(p.tile_start))) =
              dsv41_bf(dsv41_round(total[m][j][e]));
      }
    }
  }
}

// This is that kernel's own body with the tile's row extent cut to sixteen -- the MMA's M granularity, so nothing
// narrower can help -- and the same operand roles, the same 128-dim k steps in the same order, the same epilogue
// term by term and the same store expression, so a live row's score is BIT-IDENTICAL to dsv41_index_scores_nax's. A
// threadgroup keeps its 64 keys and two SIMDgroups take 32 keys each (tn), the other two stage and stop: the tile's
// MMA work is a quarter of the prompt tile's at one row whatever the window's row count.
kernel void dsv41_index_scores_row(device const ushort *qp [[buffer(0)]], device const ushort *wp [[buffer(1)]],
                                   device const ushort *kp [[buffer(2)]], device float *scores [[buffer(3)]],
                                   constant Dsv41Index &p [[buffer(4)]], uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const device bfloat* Q = reinterpret_cast<const device bfloat*>(qp);
  const device bfloat* W = reinterpret_cast<const device bfloat*>(wp);
  const device bfloat* K = reinterpret_cast<const device bfloat*>(kp);
  threadgroup bfloat tile[64 * INAX_PAD];
  const int t0 = int(p.tile_start) + int(tg.x) * 64, r0 = int(tg.y) * 16;
  const int rows = min(16, int(p.rows) - r0);
  const uint t = sg * 32u + lane;
  if (p.keys == 0 or p.ratio == 0) return;
  if (t0 >= int((p.position + uint(r0) + uint(rows)) / p.ratio)) {
    for (uint i = t; i < 16u * 64u; i += 128u) {
      const int r = int(i) / 64, k = t0 + int(i) % 64;
      if (r < rows && k < int(p.keys))
        *(dsv41_score_slot(scores, p, r0 + r, p.tile_start / p.tile_width) + (k - int(p.tile_start))) = -INFINITY;
    }
    return;
  }
  for (uint i = t; i < 64u * (INAX_D / 8u); i += 128u) {
    const uint rr = i / (INAX_D / 8u), cc = (i % (INAX_D / 8u)) * 8u;
    const int k = min(t0 + int(rr), int(p.keys) - 1);
    *(threadgroup uint4*)(tile + rr * INAX_PAD + cc) = *(const device uint4*)(K + size_t(k) * INAX_D + cc);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(sg) >= 2) return;  // sixteen rows by sixty-four keys is two 16x32 column blocks
  const short2 home = frag_home(ushort(lane));
  const int tn = 32 * int(sg);
  const int tm = 0;
  const device bfloat* qr[2];
  const device bfloat* wr[2];
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const int r = tm + home.y + 8 * h;
    const int at = r0 + (r < rows ? r : 0);
    qr[h] = Q + size_t(at) * p.q_stride;
    wr[h] = W + size_t(at) * p.weight_stride;
  }
  frag<float> total0 = frag<float>(0), total1 = frag<float>(0);
  for (int hd = 0; hd < int(p.heads); hd++) {
    frag<float> s0 = frag<float>(0), s1 = frag<float>(0);
    TF_UNROLL
    for (short q = 0; q < INAX_D / 16; q++) {
      frag<bfloat> k0, k1;
      frag_get_t(k0, (const threadgroup bfloat*)tile, INAX_PAD, tn, 16 * q, home);
      frag_get_t(k1, (const threadgroup bfloat*)tile, INAX_PAD, tn + 16, 16 * q, home);
      const frag<bfloat> qa = inax_frag(qr[0] + hd * INAX_D, qr[1] + hd * INAX_D, 16 * q, home);
      mma_16x32<false, true>(s0, s1, qa, k0, k1);
    }
    // the head's term as the scalar makes it: bf16 dot, relu, times the bf16-rounded scaled weight, fp32 sum
    const float w0 = dsv41_bf(dsv41_round(float(wr[0][hd]) * p.scale));
    const float w1 = dsv41_bf(dsv41_round(float(wr[1][hd]) * p.scale));
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const float dot = dsv41_bf(dsv41_round((j == 0 ? s0 : s1)[e]));
        const float add = dsv41_bf(dsv41_round(max(dot, 0.0f) * (e < 4 ? w0 : w1)));
        if (j == 0) total0[e] += add; else total1[e] += add;
      }
    }
  }
  TF_UNROLL
  for (short j = 0; j < 2; j++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      const int r = tm + home.y + (e >> 2) * 8;
      const int k = t0 + tn + 16 * j + home.x + TF_COL(e);
      if (r < rows && k < int(p.keys))
        *(dsv41_score_slot(scores, p, r0 + r, p.tile_start / p.tile_width) + (k - int(p.tile_start))) =
            dsv41_bf(dsv41_round((j == 0 ? total0 : total1)[e]));
    }
  }
}

// The keys-on-M score tile The row tile above cuts the prompt tile's row extent to the MMA's sixteen, but a one-row
// step still runs a sixteen-row MMA: fifteen of sixteen M rows are dead. This form swaps the operand roles -- the
// KEYS are the M operand (four 16-key blocks, one a SIMDgroup) and the head queries are the N operand (32 heads) --
// so a threadgroup scores 64 keys x 32 heads with eight MMAs a SIMDgroup instead of a quarter of 1024. The k chain
// (128 dims in sixteen-wide steps, ascending), the per-head term (bf16 dot, relu, the bf16-rounded scaled weight)
// and the head-ordered fp32 sum are the scalar's own: the head terms are staged as bf16 (the value is already bf16)
// and summed 0..31 in order by one lane a key, so a score is BIT-IDENTICAL to dsv41_index_scores_row's. The key
// tile is staged once and reused by every row of the window.
kernel void dsv41_index_scores_keys(device const ushort *qp [[buffer(0)]], device const ushort *wp [[buffer(1)]],
                                    device const ushort *kp [[buffer(2)]], device float *scores [[buffer(3)]],
                                    constant Dsv41Index &p [[buffer(4)]], uint sg [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const device bfloat* Q = reinterpret_cast<const device bfloat*>(qp);
  const device bfloat* W = reinterpret_cast<const device bfloat*>(wp);
  const device bfloat* K = reinterpret_cast<const device bfloat*>(kp);
  threadgroup bfloat tile[64 * INAX_PAD];
  threadgroup bfloat terms[64 * 32];
  const int t0 = int(p.tile_start) + int(tg.x) * 64;
  const uint t = sg * 32u + lane;
  if (p.keys == 0 or p.ratio == 0) return;
  const int rows = int(p.rows);
  // A tile past the last row's visible keys scores -inf and is not staged.
  if (t0 >= int((p.position + uint(rows)) / p.ratio)) {
    for (uint r = 0; r < uint(rows); r++)
      for (uint i = t; i < 64u; i += 128u) {
        const int k = t0 + int(i);
        if (k < int(p.keys))
          *(dsv41_score_slot(scores, p, int(r), p.tile_start / p.tile_width) + (k - int(p.tile_start))) = -INFINITY;
      }
    return;
  }
  for (uint i = t; i < 64u * (INAX_D / 8u); i += 128u) {
    const uint rr = i / (INAX_D / 8u), cc = (i % (INAX_D / 8u)) * 8u;
    const int k = min(t0 + int(rr), int(p.keys) - 1);
    *(threadgroup uint4*)(tile + rr * INAX_PAD + cc) = *(const device uint4*)(K + size_t(k) * INAX_D + cc);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const short2 home = frag_home(ushort(lane));
  const int m0 = 16 * int(sg);
  for (int r = 0; r < rows; r++) {
    const device bfloat* qrow = Q + size_t(r) * p.q_stride;
    const device bfloat* wrow = W + size_t(r) * p.weight_stride;
    frag<float> s0 = frag<float>(0), s1 = frag<float>(0);
    TF_UNROLL
    for (short q = 0; q < INAX_D / 16; q++) {
      frag<bfloat> ka, b0, b1;
      frag_get(ka, (const threadgroup bfloat*)tile, INAX_PAD, m0, 16 * q, home);
      frag_get_t(b0, qrow, INAX_D, 0, 16 * q, home);
      frag_get_t(b1, qrow, INAX_D, 16, 16 * q, home);
      mma_16x32<false, true>(s0, s1, ka, b0, b1);
    }
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int key = m0 + home.y + (e >> 2) * 8;
        const int h = 16 * j + home.x + TF_COL(e);
        const float w = dsv41_bf(dsv41_round(float(wrow[h]) * p.scale));
        const float dot = dsv41_bf(dsv41_round((j == 0 ? s0 : s1)[e]));
        terms[key * 32 + h] = bfloat(dsv41_bf(dsv41_round(max(dot, 0.0f) * w)));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 64u) {
      const int k = t0 + int(t);
      if (k < int(p.keys)) {
        float total = 0.0f;
        TF_UNROLL
        for (short h = 0; h < 32; h++) total += float(terms[t * 32u + uint(h)]);
        *(dsv41_score_slot(scores, p, r, p.tile_start / p.tile_width) + (k - int(p.tile_start))) =
            dsv41_bf(dsv41_round(total));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// The decode selection, one dispatch: at decode rows the heap path costs 27 dispatches and ~1,000 one-thread pushes
// a source. This does the same selection in ONE threadgroup a row: a radix select on the score's monotone image
// (four 8-bit passes over the resident scores) fixes the k-th largest exactly, a threshold filter keeps the k keys
// above it (plus the lowest-positioned ties), and the finish's bitonic sort writes the ascending positions. The
// heap's total order is (score desc, position asc), so the SET and the array are the same.
inline void dsv41_ixsort(threadgroup uint *sh, uint n, uint tid) {
    const uint TG = 256u;
    for (uint k = 2; k <= n; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = tid; i < n; i += TG) {
                const uint ixj = i ^ j;
                if (ixj > i) {
                    const bool asc = (i & k) == 0;
                    const uint a = sh[i], b = sh[ixj];
                    if (asc ? (a > b) : (a < b)) { sh[i] = b; sh[ixj] = a; }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void dsv41_index_select(device const float *scores [[buffer(0)]],
                               device int *positions [[buffer(1)]],
                               constant Dsv41Index &p [[buffer(2)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    const uint TG = 256u, topk = 512u;
    if (row >= p.rows) return;
    threadgroup uint hist[256];
    threadgroup uint sel[512];
    threadgroup uint ties[512];
    threadgroup atomic_uint counter;
    const uint visible = min(p.keys, (p.position + row + 1) / p.ratio);
    if (visible <= topk) { // every visible key is kept: the identity branch's array
        for (uint i = tid; i < topk; i += TG) positions[row * topk + i] = i < visible ? int(i) : -1;
        return;
    }
    // ---- the radix select: `prefix` converges to the k-th largest's image, `above` is the count above it
    uint prefix = 0u, above = 0u, need = topk;
    for (uint pass = 0; pass < 4u; ++pass) {
        const uint shift = 24u - 8u * pass;
        for (uint i = tid; i < 256u; i += TG) hist[i] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0, base = 0; base < visible; ++t, base += p.tile_width) {
            const device float *slot = dsv41_score_slot_ro(scores, p, row, t);
            const uint n = min(p.tile_width, visible - base);
            for (uint i = tid; i < n; i += TG) {
                const uint u = dsv41_index_mono(slot[i]);
                if (pass == 0u || (u >> (shift + 8u)) == prefix) {
                    atomic_fetch_add_explicit((threadgroup atomic_uint *)&hist[(u >> shift) & 0xFFu], 1u,
                                              memory_order_relaxed);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup uint digit, ran;
        // Digit/ran are threadgroup memory. Initialising them here was a race: every thread wrote 0 to them and tid
        // 0 wrote the chosen digit, with no ordering between the two writes before the barrier, so a late thread's
        // zero could land on top of the digit. A zero prefix then made the next pass's histogram empty and the
        // filter count every visible key as above the threshold, which the first 512 atomic arrivals truncated -- a
        // nondeterministic key SET from unchanged scores. Only tid 0 writes them now, and it writes them once.
        if (tid == 0u) {
            uint cum = 0u;
            digit = 0u; ran = 0u;
            for (int d = 255; d >= 0; --d) {
                const uint c = hist[uint(d)];
                if (cum + c >= need) { digit = uint(d); ran = cum; break; }
                cum += c;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = (pass == 0u) ? digit : ((prefix << 8u) | digit);
        above += ran;
        need -= ran;
    }
    // ---- the filter: every key above the threshold (exactly `above` of them), compacted in arrival order
    if (tid == 0u) atomic_store_explicit(&counter, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0, base = 0; base < visible; ++t, base += p.tile_width) {
        const device float *slot = dsv41_score_slot_ro(scores, p, row, t);
        const uint n = min(p.tile_width, visible - base);
        for (uint i = tid; i < n; i += TG) {
            if (dsv41_index_mono(slot[i]) > prefix) {
                const uint at = atomic_fetch_add_explicit(&counter, 1u, memory_order_relaxed);
                if (at < topk) sel[at] = base + i;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- the ties at the threshold, the lowest `need` positions of them
    const uint have = min(atomic_load_explicit(&counter, memory_order_relaxed), topk);
    // The ties pass below reuses this counter, so it has to be zeroed AFTER every thread has read `have`. Without
    // the barrier the zero could land first and leave `have` divergent across the threadgroup -- the ties then got
    // written at different slots (and the sentinel fill started at a different index) per thread, which is a second
    // nondeterministic key SET.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) atomic_store_explicit(&counter, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0, base = 0; base < visible; ++t, base += p.tile_width) {
        const device float *slot = dsv41_score_slot_ro(scores, p, row, t);
        const uint n = min(p.tile_width, visible - base);
        for (uint i = tid; i < n; i += TG) {
            if (dsv41_index_mono(slot[i]) == prefix) {
                const uint at = atomic_fetch_add_explicit(&counter, 1u, memory_order_relaxed);
                if (at < 512u) ties[at] = base + i;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nties = atomic_load_explicit(&counter, memory_order_relaxed);
    if (need > 0u) {
        if (nties <= 512u) { // sort the ties ascending and take the first `need`
            for (uint i = nties + tid; i < 512u; i += TG) ties[i] = 0xFFFFFFFFu;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dsv41_ixsort(ties, 512u, tid);
            for (uint i = tid; i < need && have + i < topk; i += TG) sel[have + i] = ties[i];
        } else { // more ties than a threadgroup array: the lowest positions, scanned in order
            if (tid == 0u) {
                uint got = 0u;
                for (uint t = 0, base = 0; base < visible && got < need; ++t, base += p.tile_width) {
                    const device float *slot = dsv41_score_slot_ro(scores, p, row, t);
                    const uint n = min(p.tile_width, visible - base);
                    for (uint i = 0; i < n && got < need; ++i)
                        if (dsv41_index_mono(slot[i]) == prefix && have + got < topk) { sel[have + got] = base + i; ++got; }
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- the ascending positions array the finish writes: the same array, sorted the same way
    for (uint i = have + need + tid; i < topk; i += TG) sel[i] = 0xFFFFFFFFu;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dsv41_ixsort(sel, 512u, tid);
    for (uint i = tid; i < topk; i += TG)
        positions[row * topk + i] = sel[i] == 0xFFFFFFFFu ? -1 : int(sel[i]);
}



// The one-dispatch select with vector reads, a two-pass radix and the ties folded into the filter. Exactness: every
// score this path ranks is bf16-valued (the scorer stores dsv41_bf(x)), so mono(score)'s low 16 bits are zero and
// its top 16 bits fix the order; two 8-bit passes converge on the k-th image. The filter/ties compaction order
// differs from the shipped kernel's, but the array is sorted before it is written, so the output is the same array.
kernel void dsv41_index_select_v(device const float *scores [[buffer(0)]],
                                 device int *positions [[buffer(1)]],
                                 constant Dsv41Index &p [[buffer(2)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
    const uint TG = 256u, topk = 512u;
    if (row >= p.rows) return;
    threadgroup uint hist[256];
    threadgroup uint sel[512];
    threadgroup uint ties[512];
    threadgroup atomic_uint counter;
    threadgroup atomic_uint tcounter;
    const uint visible = min(p.keys, (p.position + row + 1) / p.ratio);
    if (visible <= topk) { // every visible key is kept: the identity branch's array
        for (uint i = tid; i < topk; i += TG) positions[row * topk + i] = i < visible ? int(i) : -1;
        return;
    }
    // ---- the two-pass radix: bits 31..24 then 23..16; the low 16 are zero by the bf16 invariant
    uint prefix = 0u, above = 0u, need = topk;
    for (uint pass = 0; pass < 2u; ++pass) {
        const uint shift = 24u - 8u * pass;
        for (uint i = tid; i < 256u; i += TG) hist[i] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0, base = 0; base < visible; ++t, base += p.tile_width) {
            const device float *slot = dsv41_score_slot_ro(scores, p, row, t);
            const uint n = min(p.tile_width, visible - base);
            const uint n4 = n & ~3u;
            const device float4 *s4 = (const device float4 *)slot;
            for (uint i = tid; i < n4 / 4u; i += TG) {
                const float4 v = s4[i];
                const uint u[4] = {dsv41_index_mono(v.x), dsv41_index_mono(v.y),
                                   dsv41_index_mono(v.z), dsv41_index_mono(v.w)};
                for (uint j = 0; j < 4u; ++j)
                    if (pass == 0u || (u[j] >> (shift + 8u)) == prefix)
                        atomic_fetch_add_explicit((threadgroup atomic_uint *)&hist[(u[j] >> shift) & 0xFFu],
                                                  1u, memory_order_relaxed);
            }
            for (uint i = n4 + tid; i < n; i += TG) {
                const uint u = dsv41_index_mono(slot[i]);
                if (pass == 0u || (u >> (shift + 8u)) == prefix)
                    atomic_fetch_add_explicit((threadgroup atomic_uint *)&hist[(u >> shift) & 0xFFu],
                                              1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup uint digit, ran;
        // Digit/ran are threadgroup memory. Initialising them here was a race: every thread wrote 0 to them and tid
        // 0 wrote the chosen digit, with no ordering between the two writes before the barrier, so a late thread's
        // zero could land on top of the digit. A zero prefix then made the next pass's histogram empty and the
        // filter count every visible key as above the threshold, which the first 512 atomic arrivals truncated -- a
        // nondeterministic key SET from unchanged scores. Only tid 0 writes them now, and it writes them once.
        if (tid == 0u) {
            uint cum = 0u;
            digit = 0u; ran = 0u;
            for (int d = 255; d >= 0; --d) {
                const uint c = hist[uint(d)];
                if (cum + c >= need) { digit = uint(d); ran = cum; break; }
                cum += c;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = (pass == 0u) ? digit : ((prefix << 8u) | digit);
        above += ran;
        need -= ran;
    }
    // The two passes leave the top sixteen bits in the low half. mono()'s low sixteen are 0 for a
    // positive bf16 score and 0xFFFF for a negative one (mono is b's complement there), so the full
    // image is (prefix << 16) | (negative ? 0xFFFF : 0).
    prefix = (prefix << 16u) | (((prefix & 0x8000u) == 0u) ? 0xFFFFu : 0u);
    // ---- the filter and the ties in one pass: every key above the threshold (exactly `above` of them)
    if (tid == 0u) {
        atomic_store_explicit(&counter, 0u, memory_order_relaxed);
        atomic_store_explicit(&tcounter, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0, base = 0; base < visible; ++t, base += p.tile_width) {
        const device float *slot = dsv41_score_slot_ro(scores, p, row, t);
        const uint n = min(p.tile_width, visible - base);
        const uint n4 = n & ~3u;
        const device float4 *s4 = (const device float4 *)slot;
        for (uint i = tid; i < n4 / 4u; i += TG) {
            const float4 v = s4[i];
            const uint u[4] = {dsv41_index_mono(v.x), dsv41_index_mono(v.y),
                               dsv41_index_mono(v.z), dsv41_index_mono(v.w)};
            for (uint j = 0; j < 4u; ++j) {
                const uint at = base + i * 4u + j;
                if (u[j] > prefix) {
                    const uint k = atomic_fetch_add_explicit(&counter, 1u, memory_order_relaxed);
                    if (k < topk) sel[k] = at;
                } else if (u[j] == prefix && need > 0u) {
                    const uint k = atomic_fetch_add_explicit(&tcounter, 1u, memory_order_relaxed);
                    if (k < 512u) ties[k] = at;
                }
            }
        }
        for (uint i = n4 + tid; i < n; i += TG) {
            const uint u = dsv41_index_mono(slot[i]);
            const uint at = base + i;
            if (u > prefix) {
                const uint k = atomic_fetch_add_explicit(&counter, 1u, memory_order_relaxed);
                if (k < topk) sel[k] = at;
            } else if (u == prefix && need > 0u) {
                const uint k = atomic_fetch_add_explicit(&tcounter, 1u, memory_order_relaxed);
                if (k < 512u) ties[k] = at;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint have = min(atomic_load_explicit(&counter, memory_order_relaxed), topk);
    const uint nties = atomic_load_explicit(&tcounter, memory_order_relaxed);
    if (need > 0u) {
        if (nties <= 512u) { // sort the ties ascending and take the first `need`
            for (uint i = nties + tid; i < 512u; i += TG) ties[i] = 0xFFFFFFFFu;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dsv41_ixsort(ties, 512u, tid);
            for (uint i = tid; i < need && have + i < topk; i += TG) sel[have + i] = ties[i];
        } else { // more ties than a threadgroup array: the lowest positions, scanned in order
            if (tid == 0u) {
                uint got = 0u;
                for (uint t = 0, base = 0; base < visible && got < need; ++t, base += p.tile_width) {
                    const device float *slot = dsv41_score_slot_ro(scores, p, row, t);
                    const uint n = min(p.tile_width, visible - base);
                    for (uint i = 0; i < n && got < need; ++i)
                        if (dsv41_index_mono(slot[i]) == prefix && have + got < topk) { sel[have + got] = base + i; ++got; }
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- the ascending positions array the finish writes: the same array, sorted the same way
    for (uint i = have + need + tid; i < topk; i += TG) sel[i] = 0xFFFFFFFFu;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dsv41_ixsort(sel, 512u, tid);
    for (uint i = tid; i < topk; i += TG)
        positions[row * topk + i] = sel[i] == 0xFFFFFFFFu ? -1 : int(sel[i]);
}
