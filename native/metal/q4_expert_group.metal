
  // One threadgroup of NE threads. Simdgroup r (rows in turn) ranks row r's experts by router logit (simd_topk:
  // largest first, lowest id among ties) and writes PICK[r][k] and the weights exp(l_k - l_0) / their sum (fp32,
  // bf16-rounded, as expert_down). Then thread e lists the (row, slot) pairs that picked expert e, in row order;
  // the distinct experts get places u in increasing id order: UIDS[u], UMEM[u][j] = row * 32 + slot (-1 after the
  // last), UCOUNT[0] = their number.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int R = rows[0];
  threadgroup int picks[MAXR][TOPK];
  threadgroup int offs[NE / 32];
  for (int r = int(g); r < R; r += NE / 32) {
    float picked[TOPK];
    int ids[TOPK];
    simd_topk_all<NE, TOPK>(LOGITS + r * NL, lane, ids, picked);
    if (lane == 0) {
      for (int k = 0; k < TOPK; k++) { picks[r][k] = ids[k]; PICK[r * TOPK + k] = uint32_t(ids[k]); }
      float ex[TOPK], total = 0.0f;
      for (int k = 0; k < TOPK; k++) { ex[k] = metal::exp(picked[k] - picked[0]); total += ex[k]; }
      for (int k = 0; k < TOPK; k++) WTS[r * TOPK + k] = float(bfloat(ex[k] / total));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int e = int(t);
  int members[MAXR];
  int count = 0;
  for (int r = 0; r < R; r++)
    for (int k = 0; k < TOPK; k++)
      if (picks[r][k] == e) members[count++] = r * 32 + k;
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
  if (t == NE - 1) UCOUNT[0] = base + before + used;
