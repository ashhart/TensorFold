
  const uint lane = thread_index_in_simdgroup;
  const uint n = threadgroup_position_in_grid.y;
  const int r = int(n) / H;
  constexpr int PER = D / 32;
  const device bfloat* x = X + size_t(n) * D + lane * PER;
  float v[PER];
  for (int i = 0; i < PER; i++) v[i] = float(x[i]);
  if (NORM) {
    float ss = 0.0f;
    for (int i = 0; i < PER; i++) ss += v[i] * v[i];
    ss = simd_sum(ss);
    const float inv = metal::precise::rsqrt(ss / float(D) + EPS[0]);
    for (int i = 0; i < PER; i++) {
      float y = v[i] * inv;
      if (WEIGHTED) y = y * float(W[lane * PER + i]);
      v[i] = float(bfloat(y));
    }
  }
  const int base = int(lane) * PER;
  if (base + PER > D - PE) {
    const float pos = float(POS[r]);
    for (int i = 0; i < PER; i += 2) {
      const int d = base + i;
      if (d >= D - PE) {
        const float th = pos * INV[(d - (D - PE)) / 2];
        const float c = metal::precise::cos(th);
        const float s = INVERSE ? -metal::precise::sin(th) : metal::precise::sin(th);
        const float a = v[i], b = v[i + 1];
        v[i] = a * c - b * s;
        v[i + 1] = a * s + b * c;
      }
    }
  }
  device bfloat* o = OUT + size_t(n) * D + lane * PER;
  for (int i = 0; i < PER; i++) o[i] = bfloat(v[i]);
