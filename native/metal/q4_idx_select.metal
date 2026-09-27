
  // One threadgroup (1024 threads) a row past TOP complete blocks: its TOP best blocks by score (radix select over
  // order-preserving keys, 8 bits a pass; among scores equal to the cut, the lowest block ids), written as the keys
  // they cover (4 a block) in position order, then the row's tail keys [4 complete, ENDS).
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int r = int(threadgroup_position_in_grid.x);
  const int nb = COMPLETE[r];
  if (nb <= TOP) return;
  const int ends = ENDS[r];
  const int stride = SC_shape[1];
  const device float* sc = SC + size_t(r) * stride;
  device int* keys = KEYS + size_t(r) * KW;
  threadgroup atomic_uint hist[256];
  threadgroup uint cut_t, need_t;
  threadgroup int tot_a[32], tot_e[32];
  uint prefix = 0u, mask = 0u, need = TOP;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int b = int(t); b < nb; b += 1024) {
      const uint k = tf_key(sc[b]);
      if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      uint above = 0u;
      int bin = 255;
      for (; bin > 0; bin--) {
        const uint n = atomic_load_explicit(&hist[bin], memory_order_relaxed);
        if (above + n >= need) break;
        above += n;
      }
      cut_t = prefix | (uint(bin) << shift);
      need_t = need - above;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix = cut_t;
    need = need_t;
    mask |= 255u << shift;
  }
  // `prefix` is the cut score's key: every block above it is taken, and the first `need` equal to it
  const int chunk = (nb + 1023) / 1024;
  const int lo = min(nb, int(t) * chunk), hi = min(nb, lo + chunk);
  int n_above = 0, n_equal = 0;
  for (int b = lo; b < hi; b++) {
    const uint k = tf_key(sc[b]);
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
  for (int b = lo; b < hi; b++) {
    const uint k = tf_key(sc[b]);
    bool take = k > prefix;
    if (k == prefix) { take = pe < int(need); pe++; }
    if (take) {
      for (int j = 0; j < 4; j++) keys[out * 4 + j] = 4 * b + j;
      out++;
    }
  }
  if (t == 0)
    for (int k = 4 * nb; k < ends; k++) keys[4 * TOP + (k - 4 * nb)] = k;
