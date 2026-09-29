
  // one simdgroup per (row, v head): out = SiLU(z) * RMSNorm(y) * w, and the out projection's group sums
  const uint lane = thread_index_in_simdgroup;
  const uint hv = threadgroup_position_in_grid.y;
  const uint m = threadgroup_position_in_grid.z;
  const int M = dims[0], MP = dims[1];
  constexpr int PER = DV / 32;
  constexpr int GPH = DV / 64;                          // 64-groups per head
  threadgroup bfloat ob[DV];
  if (int(m) >= M) {
    if (lane < GPH) XS[(hv * GPH + lane) * MP + m] = 0.0f;
    return;
  }
  float yv[PER];
  float ss = 0.0f;
  for (int j = 0; j < PER; j++) {
    yv[j] = float(Y[(m * NV + hv) * DV + lane * PER + j]);
    ss += yv[j] * yv[j];
  }
  ss = simd_sum(ss);
  const float inv = metal::rsqrt(ss / float(DV) + eps[0]);
  for (int j = 0; j < PER; j++) {
    const int d = int(lane) * PER + j;
    const float x = float(bfloat(float(NW[d]) * (yv[j] * inv)));
    const float zf = float(Z[m * NV * DV + hv * DV + d]);
    const bfloat o = bfloat(zf / (1.0f + metal::exp(-zf)) * x);
    OUT[m * NV * DV + hv * DV + d] = o;
    ob[d] = o;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < GPH) {
    float acc = 0.0f;
    for (int i = 0; i < 64; i++) acc += float(ob[lane * 64 + i]);
    XS[(hv * GPH + lane) * MP + m] = acc;
  }
