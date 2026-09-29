
  const uint lane = thread_index_in_simdgroup;
  const int b = int(threadgroup_position_in_grid.y);
  constexpr int PER = D / 32;
  constexpr int S = (1 + OV) * R;
  const int w = META[0] + b, base = META[1], ring = META[2], col = META[3];
  const int p0 = OV ? (w - 1) * R : w * R;
  float v[PER];
  for (int i = 0; i < PER; i++) {
    const int c = int(lane) * PER + i;
    float top = -INFINITY;
    for (int j = 0; j < S; j++) {
      const int pos = p0 + j;
      if (pos < 0) continue;
      const int fc = OV && j >= R ? D + c : c;
      const size_t row = size_t(ring ? pos % ring : pos - base) * WT;
      const float s = P[row + col + W + fc] + APE[(j % R) * W + fc];
      top = s > top ? s : top;
    }
    float total = 0.0f;
    for (int j = 0; j < S; j++) {
      const int pos = p0 + j;
      if (pos < 0) continue;
      const int fc = OV && j >= R ? D + c : c;
      const size_t row = size_t(ring ? pos % ring : pos - base) * WT;
      const float s = P[row + col + W + fc] + APE[(j % R) * W + fc];
      const float e = metal::precise::exp(s - top);
      total = total + e;
    }
    float acc = 0.0f;
    for (int j = 0; j < S; j++) {
      const int pos = p0 + j;
      if (pos < 0) continue;
      const int fc = OV && j >= R ? D + c : c;
      const size_t row = size_t(ring ? pos % ring : pos - base) * WT;
      const float s = P[row + col + W + fc] + APE[(j % R) * W + fc];
      const float e = metal::precise::exp(s - top);
      const float share = e / total;
      const float term = share * P[row + col + fc];
      acc = acc + term;
    }
    v[i] = float(bfloat(acc));
  }
  float ss = 0.0f;
  for (int i = 0; i < PER; i++) ss += v[i] * v[i];
  ss = simd_sum(ss);
  const float inv = metal::precise::rsqrt(ss / float(D) + EPS[0]);
  for (int i = 0; i < PER; i++) {
    float y = v[i] * inv;
    y = y * float(NORM[lane * PER + i]);
    v[i] = float(bfloat(y));
  }
  const int d0 = int(lane) * PER;
  if (d0 + PER > D - PE) {
    const float at = float(w * R);
    for (int i = 0; i < PER; i += 2) {
      const int d = d0 + i;
      if (d >= D - PE) {
        const float th = at * INV[(d - (D - PE)) / 2];
        const float cs = metal::precise::cos(th);
        const float sn = metal::precise::sin(th);
        const float a = v[i], bb = v[i + 1];
        v[i] = a * cs - bb * sn;
        v[i + 1] = a * sn + bb * cs;
      }
    }
  }
  device bfloat* o = OUT + size_t(b) * D + d0;
  for (int i = 0; i < PER; i++) o[i] = bfloat(v[i]);
