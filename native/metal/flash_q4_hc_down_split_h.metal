
  // split-K matvec of one row's normed streams: simdgroup o of threadgroup (i, k, r) over input groups 32 k .. 32 k + 31
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int R = rows[0];
  constexpr int W = S * D;
  constexpr int GROUPS = W / 32;
  const int o = int(threadgroup_position_in_grid.x) * 8 + int(g);
  const int k = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  const int c0 = k * 32 * 32;
  threadgroup float xs[32 * 33];                      // group j at 33 j: a lane's reads hit distinct banks
  threadgroup float rinv[S];
  if (t < S) rinv[t] = stream_rinv(SSP, r, int(t), D / 256, S, D, eps[0]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(t); i < 32 * 32; i += 256) {
    const int e = c0 + i;
    xs[(i / 32) * 33 + i % 32] = float(bfloat((float(HN[r * W + e]) * rinv[e / D]) * NW[e]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (o < ND) {
    const int grp = k * 32 + int(lane);
    float x[32];
    for (int n = 0; n < 32; n++) x[n] = xs[lane * 33 + n];
    float acc = qgroup_doth(QW + (size_t(o) * GROUPS + grp) * 4, float(QS[o * GROUPS + grp]), float(QB[o * GROUPS + grp]), x);
    acc = simd_sum(acc);
    if (lane == 0) PART[(k * R + r) * ND + o] = acc;
  }
