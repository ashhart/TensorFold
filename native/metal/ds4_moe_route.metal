
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int e = int(thread_position_in_threadgroup.x);
  const int R = int(LOGITS_shape[0]);
  constexpr int PER = (NE + 31) / 32;
  threadgroup int picks[MAXR * TOPK];
  threadgroup int offs[32];
  if (int(g) < R) {
    const int r = int(g);
    float c[PER], sc[PER];
    for (int j = 0; j < PER; j++) {
      const int id = j * 32 + int(lane);
      sc[j] = id < NE ? sqrtsoftplus(LOGITS[r * NE + id]) : 0.0f;
      c[j] = id < NE ? sc[j] + BIAS[id] : -INFINITY;
    }
    int ids[TOPK];
    float w[TOPK];
    for (int k = 0; k < TOPK; k++) {
      if (HASHED) {
        const int id = TABLE[size_t(TOKENS[r]) * TOPK + k];
        float mine = 0.0f;
        for (int j = 0; j < PER; j++) if (j == id / 32) mine = sc[j];
        ids[k] = id;
        w[k] = simd_shuffle(mine, ushort(id % 32));
        continue;
      }
      float best = -INFINITY, bsc = 0.0f;
      int bid = NE;
      for (int j = 0; j < PER; j++) {
        const int id = j * 32 + int(lane);
        if (id < NE && (c[j] > best || (c[j] == best && id < bid))) { best = c[j]; bid = id; bsc = sc[j]; }
      }
      for (int off = 16; off > 0; off /= 2) {
        const float ob = simd_shuffle_xor(best, off);
        const int oi = simd_shuffle_xor(bid, off);
        const float os = simd_shuffle_xor(bsc, off);
        if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; bsc = os; }
      }
      ids[k] = bid;
      w[k] = bsc;
      for (int j = 0; j < PER; j++) if (j * 32 + int(lane) == bid) c[j] = -INFINITY;
    }
    for (int a = 1; a < TOPK; a++)
      for (int b = a; b > 0 && ids[b - 1] > ids[b]; b--) {
        const int ti = ids[b]; ids[b] = ids[b - 1]; ids[b - 1] = ti;
        const float tw = w[b]; w[b] = w[b - 1]; w[b - 1] = tw;
      }
    if (lane == 0) {
      float total = w[0];
      for (int k = 1; k < TOPK; k++) total = total + w[k];
      for (int k = 0; k < TOPK; k++) {
        picks[r * TOPK + k] = ids[k];
        WTS[r * TOPK + k] = (w[k] / (total + 1e-20f)) * SCALE[0];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int members[MAXR];
  int count = 0;
  if (e < NE)
    for (int p = 0; p < R * TOPK; p++)
      if (picks[p] == e) members[count++] = p;
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
  if (e == int(NT) - 1) UCOUNT[0] = base + before + used;
