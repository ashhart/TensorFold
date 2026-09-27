
  // Threadgroup (j, r): dims 256 j .. 256 j + 255 of row r in every stream: write the block's branch back into the
  // S streams (bf16 ops) and each stream's partial sum of squares over these dims (fp32, simdgroups in order).
  // Consumers take a stream's inverse RMS from its NT partials, added in j order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int j = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D;
  constexpr int NT = D / 256;
  threadgroup float part[8][S];
  const int d = j * 256 + int(t);
  float ss[S];
  
  for (int s = 0; s < S; s++) {
    const int e = s * D + d;
    float hv = float(H[r * W + e]);
    
    HN[r * W + e] = bfloat(hv);
    ss[s] = simd_sum(hv * hv);
  }
  if (lane == 0) for (int s = 0; s < S; s++) part[g][s] = ss[s];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < S) {
    float total = 0.0f;
    for (int k = 0; k < 8; k++) total += part[k][t];
    SSP[(r * NT + j) * S + t] = total;
  }
