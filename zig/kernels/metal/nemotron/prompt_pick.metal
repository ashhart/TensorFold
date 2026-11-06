// Scan only a prompt's newly written keys that an earlier query must not see; pick clears the reusable flag.
[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void nemotron_prompt_scan(const device bfloat* V [[buffer(0)]], const device int* P [[buffer(1)]],
    device atomic_uint* flag [[buffer(2)]], uint3 tg [[threadgroup_position_in_grid]],
    uint3 groups [[threadgroups_per_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  const device bfloat* v = V + long(tg.z) * P[19];
  uint inf = 0;
  for (int key = P[11] + 1 + int(tg.y) * 4 + sg; key < P[3]; key += int(groups.y) * 4) {
    const uint2 bits = *(const device uint2*)(v + long(key) * P[20] + lane * 4);
    inf |= ((bits.x & 0x7F807F80u) + 0x00800080u) & 0x80008000u;
    inf |= ((bits.y & 0x7F807F80u) + 0x00800080u) & 0x80008000u;
  }
  if (simd_any(inf != 0) && lane == 0) atomic_fetch_or_explicit(flag, 1u, memory_order_relaxed);
}

[[max_total_threads_per_threadgroup(1)]]
[[kernel]] void nemotron_prompt_pick(device atomic_uint* flag [[buffer(0)]], constant uint* grid [[buffer(1)]],
    device uint* grids [[buffer(2)]]) {
  const bool masked = atomic_load_explicit(flag, memory_order_relaxed) != 0;
  atomic_store_explicit(flag, 0u, memory_order_relaxed);
  for (int i = 0; i < 3; i++) { grids[i] = masked ? 0u : grid[i]; grids[3 + i] = masked ? grid[i] : 0u; }
}
