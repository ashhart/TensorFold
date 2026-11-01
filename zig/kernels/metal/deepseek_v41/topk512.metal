// DeepSeek-V4.1 indexer top-512, a threadgroup a row: stable rank, NaN as -inf, -0 as +0, ascending indices, then -1.
#include <metal_stdlib>
using namespace metal;

#ifndef TF_CTL
#define TF_CTL 0 // planted faults: 1 NaN ranks highest, 2 ties go to the higher index, 3 last score dropped, 4 input write
#endif
#define TF_TG 1024u
#define TF_K 512u
#define TF_BINS 2048u

struct TfTopk512Args {
  uint width; // scores a row (the row stride)
};

inline uint tf_topk512_key(uint u) {
  if ((u & 0x7fffffffu) > 0x7f800000u) u = TF_CTL == 1 ? 0x7f800000u : 0xff800000u;
  if (u == 0x80000000u) u = 0u;
  return (u >> 31) ? ~u : (u | 0x80000000u);
}

// The bin holding the k-th largest key (bins scanned from the top) and the keys in the bins above it.
inline uint2 tf_topk512_pick(threadgroup atomic_uint* bins, uint k, threadgroup uint* sc, uint tid, uint lane, uint simd) {
  const uint a = atomic_load_explicit(bins + (TF_BINS - 1u - 2u * tid), memory_order_relaxed);
  const uint b = atomic_load_explicit(bins + (TF_BINS - 2u - 2u * tid), memory_order_relaxed);
  const uint local = a + b, ex = simd_prefix_exclusive_sum(local);
  if (lane == 31u) sc[simd] = ex + local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint before = ex;
  for (uint s = 0; s < simd; ++s) before += sc[s];
  if (before < k && before + local >= k) {
    const bool first = before + a >= k;
    sc[32] = TF_BINS - (first ? 1u : 2u) - 2u * tid;
    sc[33] = first ? before : before + a;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint2 r = uint2(sc[32], sc[33]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}

[[max_total_threads_per_threadgroup(TF_TG)]] kernel void tf_topk512(
    constant TfTopk512Args& args [[buffer(0)]],
    device const uint* scores [[buffer(1)]],
    device const int* visible [[buffer(2)]],
    device int* selected [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup atomic_uint bins[TF_BINS];
  threadgroup uint sc[34], part[64];
  const uint n = min(uint(max(visible[row], 0)), args.width);
  device const uint* s = scores + ulong(row) * args.width;
  device int* out = selected + ulong(row) * TF_K;
  const uint keep = min(n, TF_K);
  const uint m = (TF_CTL == 3 && n > 0u) ? n - 1u : n;
#if TF_CTL == 4
  if (tid == 0u && n > 0u) ((device uint*)s)[0] ^= 1u;
#endif
  uint threshold = 0u, above = 0u, ties = 0u;
  if (n > TF_K) {
    uint prefix = 0u, k = TF_K;
    for (uint pass = 0; pass < 3u; ++pass) {
      const uint shift = pass == 0u ? 21u : (pass == 1u ? 10u : 0u);
      const uint mask = pass == 2u ? 0x3ffu : 0x7ffu;
      const uint up = pass == 1u ? 21u : 10u;
      for (uint i = tid; i < TF_BINS; i += TF_TG) atomic_store_explicit(bins + i, 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = tid; i < m; i += TF_TG) {
        const uint key = tf_topk512_key(s[i]);
        if (pass == 0u || (key >> up) == prefix) atomic_fetch_add_explicit(bins + ((key >> shift) & mask), 1u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint2 r = tf_topk512_pick(bins, k, sc, tid, lane, simd);
      prefix = (prefix << (pass == 2u ? 10u : 11u)) | r.x;
      k -= r.y;
      above += r.y;
    }
    threshold = prefix;
    ties = atomic_load_explicit(bins + (prefix & 0x3ffu), memory_order_relaxed);
  }
  const uint wanted = keep - above; // keys equal to the threshold that make the set: the lowest indices
  const uint skip = TF_CTL == 2 ? ties - wanted : 0u;
  uint gt_run = 0u, eq_run = 0u;
  for (uint base = 0; base < m && gt_run + min(eq_run, wanted) < keep; base += TF_TG) {
    const uint i = base + tid;
    const uint key = i < m ? tf_topk512_key(s[i]) : 0u;
    const uint gt = (i < m && (n <= TF_K || key > threshold)) ? 1u : 0u;
    const uint eq = (i < m && n > TF_K && key == threshold) ? 1u : 0u;
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
    const uint eq_taken = TF_CTL == 2 ? (eq_x > skip ? eq_x - skip : 0u) : min(eq_x, wanted);
    if (gt != 0u || (eq != 0u && (TF_CTL == 2 ? eq_x >= skip : eq_x < wanted))) out[gt_x + eq_taken] = int(i);
    gt_run += total & 0xffffu;
    eq_run += total >> 16;
  }
  for (uint i = keep + tid; i < TF_K; i += TF_TG) out[i] = -1;
}
