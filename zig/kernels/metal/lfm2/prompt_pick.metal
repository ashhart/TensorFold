// Include every staged tail key, even past the live prompt: a full final block reads those cache rows too.
[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void lfm2_prompt_scan(const device bfloat* V [[buffer(0)]], constant lfm2_attn_args& p [[buffer(1)]],
    device atomic_uint* flag [[buffer(2)]], constant uint& rows [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 groups [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint end = min(p.cap, ((p.len + rows + KB - 1) / KB) * KB);
  const device ushort* v = (const device ushort*)(V + size_t(tg.z) * p.cap * HD);
  bool exceptional = false;
  for (uint key = p.len + 1 + tg.y * 2 + tid / HD; key < end; key += groups.y * 2)
    exceptional = exceptional || ((v[size_t(key) * HD + tid % HD] & 0x7f80u) == 0x7f80u);
  if (simd_any(exceptional) && tid % 32 == 0) atomic_fetch_or_explicit(flag, 1u, memory_order_relaxed);
}
[[max_total_threads_per_threadgroup(1)]]
[[kernel]] void lfm2_prompt_pick(device atomic_uint* flag [[buffer(0)]], constant uint* grid [[buffer(1)]],
    device uint* grids [[buffer(2)]]) {
  const bool masked = atomic_load_explicit(flag, memory_order_relaxed) != 0;
  atomic_store_explicit(flag, 0u, memory_order_relaxed);
  for (uint i = 0; i < 3; i++) { grids[i] = masked ? 0u : grid[i]; grids[3 + i] = masked ? grid[i] : 0u; }
}
