// DeepSeek-V4.1 indexer top-512 of long rows, appended to topk512.metal: tf_topk512's indices from chunk bounds.
#define TF_CH 32u      // keys a chunk: a simdgroup's lanes
#define TF_CAP 2048u   // candidates a row holds; past it the row ranks every key
#define TF_LONG 16384u // a row bounds its chunks past 512 chunks of keys

// 512 chunks reach the 512th largest bound T, so every top-512 key is at least T and so is its chunk's bound.
struct TfSelArgs {
  uint width;  // scores a row (the row stride)
  uint chunks; // bounds a row (the bound row stride)
};

// Chunk bounds: four simdgroups a threadgroup, a 32-key chunk each; rows of TF_LONG keys or fewer keep none.
[[max_total_threads_per_threadgroup(128)]] kernel void tf_topk512_bounds(
    constant TfSelArgs& args [[buffer(0)]],
    device const uint* scores [[buffer(1)]],
    device const int* visible [[buffer(2)]],
    device uint* bounds [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  const uint row = tg.y;
  const uint n = min(uint(max(visible[row], 0)), args.width);
  const uint c = tg.x * 4u + simd;
  if (n <= TF_LONG || c * TF_CH >= n) return;
  const uint i = c * TF_CH + lane;
#if TF_CTL == 5 // planted: a chunk's last key left out of its bound
  const uint key = (i < n && lane != TF_CH - 1u) ? tf_topk512_key(scores[ulong(row) * args.width + i]) : 0u;
#else
  const uint key = i < n ? tf_topk512_key(scores[ulong(row) * args.width + i]) : 0u;
#endif
  const uint top = simd_max(key);
  if (lane == 0u) bounds[ulong(row) * args.chunks + c] = top;
}

// A list's i-th key and the index it stands for: a row's scores, its chunk bounds, or its candidates.
struct TfSelRow {
  device const uint* s;
  uint key(uint i) const { return tf_topk512_key(s[i]); }
  uint at(uint i) const { return i; }
};
struct TfSelBounds {
  device const uint* b;
  uint key(uint i) const { return b[i]; }
  uint at(uint i) const { return i; }
};
struct TfSelCands {
  device const uint* s;
  threadgroup const uint* ci;
  uint key(uint i) const { return tf_topk512_key(s[ci[i]]); }
  uint at(uint i) const { return ci[i]; }
};

// One count a bin; a simdgroup whose live keys share a bin adds them at once (the masked keys' -inf bin, mostly).
inline void tf_sel_count(threadgroup atomic_uint* bins, bool live, uint bin) {
  const uint lo = simd_min(live ? bin : 0xffffffffu), hi = simd_max(live ? bin : 0u);
  if (lo == hi) {
    const uint all = simd_sum(live ? 1u : 0u);
    if (simd_is_first() && all != 0u) atomic_fetch_add_explicit(bins + lo, all, memory_order_relaxed);
  } else if (live) {
    atomic_fetch_add_explicit(bins + bin, 1u, memory_order_relaxed);
  }
}

// The k-th largest of m > k keys by three bin passes (11, 11, 10 bits): (that key, keys strictly above it).
template <typename L>
inline uint2 tf_sel_kth(L l, uint m, uint k, threadgroup atomic_uint* bins, threadgroup uint* sc, uint tid, uint lane,
                        uint simd) {
  uint prefix = 0u, above = 0u;
  for (uint pass = 0; pass < 3u; ++pass) {
    const uint shift = pass == 0u ? 21u : (pass == 1u ? 10u : 0u);
    const uint mask = pass == 2u ? 0x3ffu : 0x7ffu;
    const uint up = pass == 1u ? 21u : 10u;
    for (uint i = tid; i < TF_BINS; i += TF_TG) atomic_store_explicit(bins + i, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint base = 0; base < m; base += TF_TG) {
      const uint i = base + tid;
      const uint key = i < m ? l.key(i) : 0u;
      tf_sel_count(bins, i < m && (pass == 0u || (key >> up) == prefix), (key >> shift) & mask);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint2 r = tf_topk512_pick(bins, k, sc, tid, lane, simd);
    prefix = (prefix << (pass == 2u ? 10u : 11u)) | r.x;
    k -= r.y;
    above += r.y;
  }
  return uint2(prefix, above);
}

// A list's top-512 in list order, as tf_topk512: those above its 512th key, then its ties at the lowest positions.
template <typename L>
inline void tf_sel_list(L l, uint m, device int* out, threadgroup atomic_uint* bins, threadgroup uint* sc,
                        threadgroup uint* part, uint tid, uint lane, uint simd) {
  const bool all = m <= TF_K;
  const uint2 t = all ? uint2(0u) : tf_sel_kth(l, m, TF_K, bins, sc, tid, lane, simd);
  const uint keep = min(m, TF_K), wanted = keep - t.y;
  uint gt_run = 0u, eq_run = 0u;
  for (uint base = 0; base < m && gt_run + min(eq_run, wanted) < keep; base += TF_TG) {
    const uint i = base + tid;
    const uint key = i < m ? l.key(i) : 0u;
    const uint gt = (i < m && (all || key > t.x)) ? 1u : 0u;
    const uint eq = (i < m && !all && key == t.x) ? 1u : 0u;
    const uint packed = gt | (eq << 16);
    const uint ex = simd_prefix_exclusive_sum(packed);
    threadgroup uint* p = part + 32u * ((base / TF_TG) & 1u);
    if (lane == 31u) p[simd] = ex + packed;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = ex, total = 0u;
    for (uint q = 0; q < 32u; ++q) {
      const uint v = p[q];
      before += q < simd ? v : 0u;
      total += v;
    }
    const uint gt_x = gt_run + (before & 0xffffu), eq_x = eq_run + (before >> 16);
    if (gt != 0u || (eq != 0u && eq_x < wanted)) out[gt_x + min(eq_x, wanted)] = int(l.at(i));
    gt_run += total & 0xffffu;
    eq_run += total >> 16;
  }
  for (uint i = keep + tid; i < TF_K; i += TF_TG) out[i] = -1;
}

// A key is a candidate at or above the 512th bound; a padding key (0) never is, every real key being above 0.
inline bool tf_sel_in(uint key, uint thr) {
#if TF_CTL == 6 // planted: the keys equal to the bound left out
  return key > thr;
#else
  return key >= thr;
#endif
}

// The candidates' indices in index order, chunk by chunk; returns their count (past TF_CAP: the row ranks every key).
inline uint tf_sel_gather(device const uint* s, device const uint* b, uint n, uint thr, threadgroup uint* ci,
                          threadgroup uint* part, uint tid, uint lane, uint simd) {
  const uint chunks = (n + TF_CH - 1u) / TF_CH;
  uint run = 0u;
  for (uint base = 0; base < chunks && run <= TF_CAP; base += TF_TG) {
    // simdgroup `simd` takes chunks base + 32 simd + j in turn, each lane one key of a chunk
    const uint c0 = base + TF_CH * simd;
    const uint mine = c0 + lane;
    const simd_vote::vote_t hit = static_cast<simd_vote::vote_t>(simd_ballot(mine < chunks && b[mine] >= thr));
    uint count = 0u;
    for (simd_vote::vote_t v = hit; v != 0u; v &= v - 1u) {
      const uint i = (c0 + ctz(v)) * TF_CH + lane;
      count += simd_sum(tf_sel_in(i < n ? tf_topk512_key(s[i]) : 0u, thr) ? 1u : 0u);
    }
    threadgroup uint* p = part + 32u * ((base / TF_TG) & 1u);
    if (lane == 0u) p[simd] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint at = run, total = 0u;
    for (uint q = 0; q < 32u; ++q) {
      at += q < simd ? p[q] : 0u;
      total += p[q];
    }
    for (simd_vote::vote_t v = hit; v != 0u; v &= v - 1u) {
      const uint i = (c0 + ctz(v)) * TF_CH + lane;
      const bool in = tf_sel_in(i < n ? tf_topk512_key(s[i]) : 0u, thr);
      const uint to = at + simd_prefix_exclusive_sum(in ? 1u : 0u);
      if (in && to < TF_CAP) ci[to] = i;
      at += simd_sum(in ? 1u : 0u);
    }
    run += total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return run;
}

// The top-512 of rows tf_topk512_bounds bounded (n > TF_LONG): bins over the bounds, then over the candidates.
[[max_total_threads_per_threadgroup(TF_TG)]] kernel void tf_topk512_bounded(
    constant TfSelArgs& args [[buffer(0)]],
    device const uint* scores [[buffer(1)]],
    device const int* visible [[buffer(2)]],
    device int* selected [[buffer(3)]],
    device const uint* bounds [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup atomic_uint bins[TF_BINS];
  threadgroup uint sc[34], part[64], ci[TF_CAP];
  const uint n = min(uint(max(visible[row], 0)), args.width);
  device const uint* s = scores + ulong(row) * args.width;
  device int* out = selected + ulong(row) * TF_K;
  if (n > TF_LONG) {
    device const uint* b = bounds + ulong(row) * args.chunks;
    const uint chunks = (n + TF_CH - 1u) / TF_CH;
    const uint thr = tf_sel_kth(TfSelBounds{b}, chunks, TF_K, bins, sc, tid, lane, simd).x;
    const uint m = tf_sel_gather(s, b, n, thr, ci, part, tid, lane, simd);
    if (m <= TF_CAP) return tf_sel_list(TfSelCands{s, ci}, m, out, bins, sc, part, tid, lane, simd);
  }
  tf_sel_list(TfSelRow{s}, n, out, bins, sc, part, tid, lane, simd);
}
