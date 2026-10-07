#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;

inline uint tf_key(float v) { uint b = as_type<uint>(v); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
// The cut bin of the 256 in `hist` (lane l scans bins 255 - 8l down to 248 - 8l): the cut key's next 8 bits, and
// how many of the `need` best keys lie in the bins above it.
inline void cut_bin(threadgroup atomic_uint* hist, uint lane, uint prefix, int shift, uint need,
                    threadgroup uint& cut, threadgroup uint& rest) {
  uint c[8], mine = 0u;
  for (int i = 0; i < 8; i++) {
    c[i] = atomic_load_explicit(&hist[255 - 8 * int(lane) - i], memory_order_relaxed);
    mine += c[i];
  }
  uint above = simd_prefix_exclusive_sum(mine);
  if (above < need && above + mine >= need) {
    int bin = 248 - 8 * int(lane);
    for (int i = 0; i < 8; i++) {
      if (above + c[i] >= need) { bin = 255 - 8 * int(lane) - i; break; }
      above += c[i];
    }
    cut = prefix | (uint(bin) << shift);
    rest = need - above;
  }
}

[[max_total_threads_per_threadgroup(1024)]]
[[kernel]] void custom_kernel_q4_idx_select_4092b8bd389cc7ed_float_int32_t_int32_t_int32_t(
  const device float* SC [[buffer(0)]],
  const constant int* SC_shape [[buffer(1)]],
  const device int32_t* COMPLETE [[buffer(2)]],
  const device int32_t* ENDS [[buffer(3)]],
  device int32_t* KEYS [[buffer(4)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  constexpr int TOP = 512;
  constexpr int KW = 2051;
  constexpr int CAP = 2048;
  constexpr int SAMPLED = 4096;

  // One threadgroup (1024 threads) a row past TOP complete blocks. From SAMPLED blocks on, thread t samples block
  // t nb / 1024; the J-th best sample (J = 2 TOP 1024 / nb) is a threshold about 2 TOP blocks clear. Whenever TOP or
  // more clear it, every block at or above the TOP-th score does, so when TOP to CAP clear it they are kept in block
  // order in threadgroup memory and the radix select (8 bits a pass; among keys equal to the cut, the lowest block
  // ids) runs on them, else on every block: the same blocks in the same order. Then the tail keys [4 complete, ENDS).
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int r = int(threadgroup_position_in_grid.x);
  const int nb = COMPLETE[r];
  if (nb <= TOP) return;
  const int ends = ENDS[r];
  const device float* sc = SC + size_t(r) * SC_shape[1];
  device int* keys = KEYS + size_t(r) * KW;
  threadgroup atomic_uint hist[256];
  threadgroup uint cut_t, need_t;
  threadgroup int tot_a[32], tot_e[32], count_t;
  threadgroup uint ck[CAP];
  threadgroup int cid[CAP];
  const int chunk0 = (nb + 1023) / 1024;
  const int lo0 = min(nb, int(t) * chunk0), hi0 = min(nb, lo0 + chunk0);
  uint thr = 0u;
  int at = 0, C = 0;
  if (nb >= SAMPLED) {                       // below SAMPLED blocks the radix over every block is as quick
    const uint sample = tf_key(sc[(long(t) * nb) / 1024]);
    uint prefix = 0u, mask = 0u, need = uint(clamp((2 * TOP * 1024) / nb, 1, 1024));
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if ((sample & mask) == prefix)
        atomic_fetch_add_explicit(&hist[(sample >> shift) & 255u], 1u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0) cut_bin(hist, lane, prefix, shift, need, cut_t, need_t);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      prefix = cut_t;
      need = need_t;
      mask |= 255u << shift;
    }
    thr = prefix;
    int cnt = 0;
    for (int b = lo0; b < hi0; b++) cnt += tf_key(sc[b]) >= thr ? 1 : 0;
    at = simd_prefix_exclusive_sum(cnt);
    if (lane == 31) tot_a[sg] = at + cnt;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      const int a = tot_a[lane], ex = simd_prefix_exclusive_sum(a);
      tot_a[lane] = ex;
      if (lane == 31) count_t = ex + a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    at += tot_a[sg];
    C = count_t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (C >= TOP && C <= CAP) {
    for (int b = lo0; b < hi0; b++) {
      const uint k = tf_key(sc[b]);
      if (k >= thr) { ck[at] = k; cid[at] = b; at++; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

  {
    uint prefix = 0u, mask = 0u, need = TOP;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = int(t); i < C; i += 1024) {
        const uint k = ck[i];
        if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0) cut_bin(hist, lane, prefix, shift, need, cut_t, need_t);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      prefix = cut_t;
      need = need_t;
      mask |= 255u << shift;
    }
    const int chunk = (C + 1023) / 1024;
    const int lo = min(C, int(t) * chunk), hi = min(C, lo + chunk);
    int n_above = 0, n_equal = 0;
    for (int i = lo; i < hi; i++) {
      const uint k = ck[i];
      n_above += k > prefix ? 1 : 0;
      n_equal += k == prefix ? 1 : 0;
    }
    int pa = simd_prefix_exclusive_sum(n_above), pe = simd_prefix_exclusive_sum(n_equal);
    if (lane == 31) { tot_a[sg] = pa + n_above; tot_e[sg] = pe + n_equal; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      const int a = tot_a[lane], e = tot_e[lane];
      tot_a[lane] = simd_prefix_exclusive_sum(a);
      tot_e[lane] = simd_prefix_exclusive_sum(e);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pa += tot_a[sg];
    pe += tot_e[sg];
    int out = pa + min(pe, int(need));
    for (int i = lo; i < hi; i++) {
      const uint k = ck[i];
      bool take = k > prefix;
      if (k == prefix) { take = pe < int(need); pe++; }
      if (take) {
        for (int j = 0; j < 4; j++) keys[out * 4 + j] = 4 * (cid[i]) + j;
        out++;
      }
    }
  }

  } else {

  {
    uint prefix = 0u, mask = 0u, need = TOP;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = int(t); i < nb; i += 1024) {
        const uint k = tf_key(sc[i]);
        if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0) cut_bin(hist, lane, prefix, shift, need, cut_t, need_t);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      prefix = cut_t;
      need = need_t;
      mask |= 255u << shift;
    }
    const int chunk = (nb + 1023) / 1024;
    const int lo = min(nb, int(t) * chunk), hi = min(nb, lo + chunk);
    int n_above = 0, n_equal = 0;
    for (int i = lo; i < hi; i++) {
      const uint k = tf_key(sc[i]);
      n_above += k > prefix ? 1 : 0;
      n_equal += k == prefix ? 1 : 0;
    }
    int pa = simd_prefix_exclusive_sum(n_above), pe = simd_prefix_exclusive_sum(n_equal);
    if (lane == 31) { tot_a[sg] = pa + n_above; tot_e[sg] = pe + n_equal; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      const int a = tot_a[lane], e = tot_e[lane];
      tot_a[lane] = simd_prefix_exclusive_sum(a);
      tot_e[lane] = simd_prefix_exclusive_sum(e);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pa += tot_a[sg];
    pe += tot_e[sg];
    int out = pa + min(pe, int(need));
    for (int i = lo; i < hi; i++) {
      const uint k = tf_key(sc[i]);
      bool take = k > prefix;
      if (k == prefix) { take = pe < int(need); pe++; }
      if (take) {
        for (int j = 0; j < 4; j++) keys[out * 4 + j] = 4 * (i) + j;
        out++;
      }
    }
  }

  }
  if (t == 0)
    for (int k = 4 * nb; k < ends; k++) keys[4 * TOP + (k - 4 * nb)] = k;

}
