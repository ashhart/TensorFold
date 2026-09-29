
  constexpr int S = 4;
  constexpr int F = S * D;                          // flattened streams
  constexpr int MIX = (2 + S) * S;                  // 24 mixes

  // Threadgroup (og, r): MLX's gemv for mixes og 4 .. og 4 + 3 of row r on z = x inv (the rms_norm output).
  const int og = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  const uint lane = thread_index_in_simdgroup, sgn = simdgroup_index_in_threadgroup;
  threadgroup float part[8][4];
  const float inv = INV[r];
  device const bfloat* xs = X + size_t(r) * F;
  float res[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int bn = (32 * int(sgn) + int(lane)) * 4; bn < F; bn += 1024) {
    float vc[4];
    for (int tn = 0; tn < 4; tn++) vc[tn] = float(xs[bn + tn]) * inv;
    for (int tm = 0; tm < 4; tm++) {
      const device float* mrow = FN + size_t(og * 4 + tm) * F;
      float inter[4];
      for (int tn = 0; tn < 4; tn++) inter[tn] = mrow[bn + tn];
      for (int tn = 0; tn < 4; tn++) res[tm] += inter[tn] * vc[tn];
    }
  }
  for (int tm = 0; tm < 4; tm++)
    for (ushort sn = 16; sn >= 1; sn >>= 1) res[tm] += simd_shuffle_down(res[tm], sn);
  if (lane == 0) for (int tm = 0; tm < 4; tm++) part[sgn][tm] = res[tm];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgn == 0 && lane < 4) {
    float a = part[0][lane];
    for (int k = 1; k < 8; k++) a += part[k][lane];
    MIXES[r * MIX + og * 4 + lane] = a;
  }
