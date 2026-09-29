
  // Thread e lists expert e's picks (row * TOPK + slot); distinct experts get places in id order
  const int e = int(thread_position_in_threadgroup.x);
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int picks = int(IDX_shape[0]) * TOPK;
  threadgroup int offs[32];
  int members[MAXR];
  int count = 0;
  if (e < NE)
    for (int p = 0; p < picks; p++)
      if (int(IDX[p]) == e) members[count++] = p;
  const int used = count > 0 ? 1 : 0;
  const int before = simd_prefix_exclusive_sum(used);
  if (lane == 31) offs[g] = before + used;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int base = 0;
  for (int q = 0; q < int(g); q++) base += offs[q];
  if (used) {
    const int u = base + before;
    UIDS[u] = e;
    for (int j = 0; j < MAXR; j++) UMEM[u * MAXR + j] = j < count ? members[j] : -1;
  }
  if (e == ((NE + 31) / 32) * 32 - 1) UCOUNT[0] = base + before + used;
