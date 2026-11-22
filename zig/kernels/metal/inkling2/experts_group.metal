// Route pairs grouped by expert, for the MMA dots of prompt chunks and the dedup dots of decode windows.
#define I2X_PLAN_THREADS 1024u

struct Inkling2ExpPlanArgs {
  uint pairs, experts, tile, gate_up_groups;
  uint down_groups, swap, p1, p2;
};

// Sort pair slots by expert (bucket `experts` is inactive) into tiles {bucket, first pair, pairs}; tiles[0].x counts.
[[kernel, max_total_threads_per_threadgroup(I2X_PLAN_THREADS)]]
void inkling2_exp_group_plan(const device uint* ids [[buffer(0)]], device uint* order [[buffer(1)]],
                             device uint4* tiles [[buffer(2)]], device uint* launch [[buffer(3)]],
                             constant Inkling2ExpPlanArgs& a [[buffer(4)]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint seen[130];
  threadgroup uint first[130];
  if (t < 130u) atomic_store_explicit(&seen[t], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint p = t; p < a.pairs; p += I2X_PLAN_THREADS)
    atomic_fetch_add_explicit(&seen[min(ids[p], a.experts)], 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0u) {
    uint begin = 0u, n = 0u;
    for (uint e = 0u; e <= a.experts; ++e) {
      const uint count = atomic_load_explicit(&seen[e], memory_order_relaxed);
      first[e] = begin;
      for (uint at = 0u; at < count; at += a.tile) tiles[1u + n++] = uint4(e, begin + at, min(a.tile, count - at), 0u);
      begin += count;
      atomic_store_explicit(&seen[e], 0u, memory_order_relaxed);
    }
    tiles[0] = uint4(n, 0u, 0u, 0u);
    // swap: columns along grid x and tiles along y (the dedup dots); else tiles along x (the MMA dots)
    launch[0] = a.swap ? a.gate_up_groups : n; launch[1] = a.swap ? n : a.gate_up_groups; launch[2] = 1u;
    launch[4] = a.swap ? a.down_groups : n; launch[5] = a.swap ? n : a.down_groups; launch[6] = 1u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint p = t; p < a.pairs; p += I2X_PLAN_THREADS) {
    const uint e = min(ids[p], a.experts);
    order[first[e] + atomic_fetch_add_explicit(&seen[e], 1u, memory_order_relaxed)] = p;
  }
}
