// DeepSeek-V4.1 candidate blocks, a threadgroup a row: block maxima, the newest block pinned, the top blocks as a mask.
#include <metal_stdlib>
using namespace metal;

#ifndef TF_CTL
#define TF_CTL 0 // planted faults: 1 NaN ranks highest, 2 ties to the higher block, 3 no pin, 4 input write
#endif
#define TF_TG 1024u
#define TF_BINS 2048u
#define TF_NINF 0x007fffffu // the order key of -inf: a block at or below it is never a candidate
#define TF_PINF 0xff800000u // the order key of +inf, which the pinned block takes

struct TfCandArgs {
  uint keys;   // scores a row (the row stride of the scores)
  uint blocks; // the row stride of the keyed scratch and of the mask
  uint topk;   // blocks kept, at most 2048
};

// Order-preserving key of a float's bits: NaN as -inf (control 1: +inf), -0 as +0.
inline uint tf_cand_key(uint u) {
  if ((u & 0x7fffffffu) > 0x7f800000u) u = TF_CTL == 1 ? 0x7f800000u : 0xff800000u;
  if (u == 0x80000000u) u = 0u;
  return (u >> 31) ? ~u : (u | 0x80000000u);
}

// The bin holding the k-th largest key (bins scanned from the top) and the keys in the bins above it.
inline uint2 tf_cand_pick(threadgroup atomic_uint* bins, uint k, threadgroup uint* sc, uint tid, uint lane, uint simd) {
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

// scores [rows][keys], visible [rows], keyed [rows][blocks] scratch, mask [rows][blocks] bytes.
[[max_total_threads_per_threadgroup(TF_TG)]] kernel void tf_cand(
    constant TfCandArgs& args [[buffer(0)]],
    device const uint* scores [[buffer(1)]],
    device const int* visible [[buffer(2)]],
    device uint* keyed [[buffer(3)]],
    device uchar* mask [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup atomic_uint bins[TF_BINS];
  threadgroup uint sc[34], part[64];
  const uint n = min((args.keys + 7u) / 8u, args.blocks);
  const uint k = min(args.topk, 2048u);
  const int vis = visible[row];
  const uint last = vis >= 1 ? uint(vis - 1) / 8u : 0xffffffffu;
  device const uint* s = scores + ulong(row) * args.keys;
  device uint* kd = keyed + ulong(row) * args.blocks;
  device uchar* out = mask + ulong(row) * args.blocks;
#if TF_CTL == 4
  if (tid == 0u && args.keys > 0u) ((device uint*)s)[0] ^= 1u;
#endif
  for (uint j = tid; j < args.blocks; j += TF_TG) {
    uint m = TF_NINF;
    if (j < n) {
      const uint end = min(8u * j + 8u, args.keys);
      for (uint i = 8u * j; i < end; ++i) m = max(m, tf_cand_key(s[i]));
    }
    kd[j] = (TF_CTL != 3 && j == last && j < n) ? TF_PINF : m;
  }
  uint threshold = 0u, above = 0u, ties = 0u;
  if (n > k && k > 0u) {
    uint prefix = 0u, left = k;
    for (uint pass = 0; pass < 3u; ++pass) {
      const uint shift = pass == 0u ? 21u : (pass == 1u ? 10u : 0u);
      const uint bmask = pass == 2u ? 0x3ffu : 0x7ffu;
      const uint up = pass == 1u ? 21u : 10u;
      for (uint i = tid; i < TF_BINS; i += TF_TG) atomic_store_explicit(bins + i, 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = tid; i < n; i += TF_TG) {
        const uint key = kd[i];
        if (pass == 0u || (key >> up) == prefix) atomic_fetch_add_explicit(bins + ((key >> shift) & bmask), 1u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint2 r = tf_cand_pick(bins, left, sc, tid, lane, simd);
      prefix = (prefix << (pass == 2u ? 10u : 11u)) | r.x;
      left -= r.y;
      above += r.y;
    }
    threshold = prefix;
    ties = atomic_load_explicit(bins + (prefix & 0x3ffu), memory_order_relaxed);
  }
  const bool all = n <= k;
  const uint wanted = k - above; // keys equal to the threshold that make the set: the lowest blocks
  const bool scan = !all && (TF_CTL == 2 || wanted < ties);
  const uint skip = TF_CTL == 2 ? ties - wanted : 0u;
  uint eq_run = 0u;
  for (uint base = 0; base < args.blocks; base += TF_TG) {
    const uint j = base + tid;
    const uint key = j < n ? kd[j] : 0u;
    const bool real = key > TF_NINF;
    const bool eq = real && !all && key == threshold;
    bool pick = real && k > 0u && (all || key > threshold);
    if (scan) {
      const uint ex = simd_prefix_exclusive_sum(eq ? 1u : 0u);
      threadgroup uint* p = part + 32u * ((base / TF_TG) & 1u);
      if (lane == 31u) p[simd] = ex + (eq ? 1u : 0u);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      uint before = ex, total = 0u;
      for (uint q = 0; q < 32u; ++q) {
        const uint v = p[q];
        before += q < simd ? v : 0u;
        total += v;
      }
      const uint eq_x = eq_run + before;
      pick = pick || (eq && (TF_CTL == 2 ? eq_x >= skip : eq_x < wanted));
      eq_run += total;
    } else {
      pick = pick || (eq && k > 0u);
    }
    if (j < args.blocks) out[j] = pick ? 1 : 0;
  }
}
