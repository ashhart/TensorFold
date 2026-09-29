
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int slot = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  const int R = int(threadgroups_per_grid.y);
  constexpr int PER = DH / 32;
  threadgroup float raw[DH];
  for (int p = sg; p < DH / RPS; p += SG) {
    const int row0 = slot * DH + p * RPS;
    float acc[RPS];
    tf_rowdot<KD, GS, RPS>((const device uint8_t*)W + size_t(row0) * (KD / 2), S + size_t(row0) * (KD / GS),
                           B + size_t(row0) * (KD / GS), X + size_t(r) * KD, lane, acc);
    if (lane == 0)
      for (int j = 0; j < RPS; j++) raw[p * RPS + j] = float(bfloat(acc[j]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  const int kind = slot < NQ ? 0 : (slot < NQ + NK ? 1 : 2);
  const int head = kind == 0 ? slot : (kind == 1 ? slot - NQ : slot - NQ - NK);
  float xv[PER];
  for (int i = 0; i < PER; i++) xv[i] = raw[int(lane) + 32 * i];

  {
    float ss = 0.0f;
    for (int i = 0; i < PER; i++) ss = fma(xv[i], xv[i], ss);
    ss = simd_sum(ss);
    const float inv = metal::precise::rsqrt(ss / float(DH) + eps[0]);
    if (kind == 2) {
      for (int i = 0; i < PER; i++) V[(head * R + r) * DH + int(lane) + 32 * i] = bfloat(xv[i] * inv);
    } else {
      const device bfloat* w = kind == 0 ? QW : KW;
      float y[PER];
      for (int i = 0; i < PER; i++) y[i] = float(bfloat(float(w[int(lane) + 32 * i]) * float(bfloat(xv[i] * inv))));
      const float pos = float(POS[r]);
      for (int i = 0; i < PER / 2; i++) {
        const float theta = pos * INVF[int(lane) + 32 * i];
        const float c = metal::fast::cos(theta), s = metal::fast::sin(theta);
        const float x1 = y[i], x2 = y[i + PER / 2];
        y[i] = x1 * c - x2 * s;
        y[i + PER / 2] = x1 * s + x2 * c;
      }
      if (kind == 0)
        for (int i = 0; i < PER; i++) Q[(r * NQ + head) * DH + int(lane) + 32 * i] = bfloat(y[i]);
      else
        for (int i = 0; i < PER; i++) K[(head * R + r) * DH + int(lane) + 32 * i] = bfloat(y[i]);
    }
  }

  if (VK && kind == 1) {

  {
    float ss = 0.0f;
    for (int i = 0; i < PER; i++) ss = fma(xv[i], xv[i], ss);
    ss = simd_sum(ss);
    const float inv = metal::precise::rsqrt(ss / float(DH) + eps[0]);
    if (2 == 2) {
      for (int i = 0; i < PER; i++) V[(head * R + r) * DH + int(lane) + 32 * i] = bfloat(xv[i] * inv);
    } else {
      const device bfloat* w = 2 == 0 ? QW : KW;
      float y[PER];
      for (int i = 0; i < PER; i++) y[i] = float(bfloat(float(w[int(lane) + 32 * i]) * float(bfloat(xv[i] * inv))));
      const float pos = float(POS[r]);
      for (int i = 0; i < PER / 2; i++) {
        const float theta = pos * INVF[int(lane) + 32 * i];
        const float c = metal::fast::cos(theta), s = metal::fast::sin(theta);
        const float x1 = y[i], x2 = y[i + PER / 2];
        y[i] = x1 * c - x2 * s;
        y[i + PER / 2] = x1 * s + x2 * c;
      }
      if (2 == 0)
        for (int i = 0; i < PER; i++) Q[(r * NQ + head) * DH + int(lane) + 32 * i] = bfloat(y[i]);
      else
        for (int i = 0; i < PER; i++) K[(head * R + r) * DH + int(lane) + 32 * i] = bfloat(y[i]);
    }
  }

  }
