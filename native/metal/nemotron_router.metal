
  // bf16 router logits for R rows: one threadgroup of SG simdgroups per expert. Simdgroup g sums its D / SG inputs
  // (lane l: 4 consecutive inputs at a time, 128 apart), then simd_sum; the simdgroups' sums are added in order.
  // A row's logits have the same bits at any row count (the row count is a runtime value).
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int e = int(threadgroup_position_in_grid.y);
  const int R = rows[0];
  constexpr int PART = D / SG;
  threadgroup float part[MAXR][SG];
  float acc[MAXR];
  for (int r = 0; r < MAXR; r++) acc[r] = 0.0f;
  const int begin = int(g) * PART;
  for (int c = begin + 4 * int(lane); c < begin + PART; c += 128) {
    const float w0 = float(GW[size_t(e) * D + c]), w1 = float(GW[size_t(e) * D + c + 1]);
    const float w2 = float(GW[size_t(e) * D + c + 2]), w3 = float(GW[size_t(e) * D + c + 3]);
    for (int r = 0; r < MAXR; r++) {
      if (r >= R) break;
      const device bfloat* xr = X + r * D + c;
      acc[r] = fma(float(xr[3]), w3, fma(float(xr[2]), w2, fma(float(xr[1]), w1, fma(float(xr[0]), w0, acc[r]))));
    }
  }
  for (int r = 0; r < MAXR; r++) {
    if (r >= R) break;
    const float total = simd_sum(acc[r]);
    if (lane == 0) part[r][g] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (g == 0 && int(lane) < R) {
    float total = 0.0f;
    for (int k = 0; k < SG; k++) total += part[lane][k];
    OUT[int(lane) * NE + e] = bfloat(total);
  }
