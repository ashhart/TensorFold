
  const int r = int(threadgroup_position_in_grid.y) / 64;
  const int h = int(threadgroup_position_in_grid.y) % 64;
  const uint lane = thread_index_in_simdgroup;
  const int g = int(simdgroup_index_in_threadgroup);
  constexpr int D = 512;
  constexpr int KB = 8;
  constexpr float NEG = -INFINITY;
  const int prefix = META[0], pm = META[1], rs = META[2];
  const float scale = SCALE[0];
  const int pc = PCOUNT[r];
  const int wlo = WPOS[2 * r], whi = WPOS[2 * r + 1];
  const int n = pc + (whi - wlo + 1);
  const int per = (n + S - 1) / S;
  const int j0 = g * per, j1 = min(n, j0 + per);
  auto key = [&](int j) -> const device bfloat* {
    if (j < pc) return POOL + size_t(prefix ? j : PIDX[size_t(r) * pm + j]) * D + lane * 16;
    return RING + size_t((wlo + j - pc) % rs) * D + lane * 16;
  };
  float q[16], o[16], k[16];
  load16(Q + (size_t(r) * 64 + h) * D + lane * 16, q);
  for (int d = 0; d < 16; d++) o[d] = 0.0f;
  float m = NEG, l = 0.0f;
  for (int b = j0; b < j1; b += KB) {
    float sc[KB];
    float mb = NEG;
    for (int i = 0; i < KB; i++) {
      float a = 0.0f;
      if (b + i < j1) {
        load16(key(b + i), k);
        for (int d = 0; d < 16; d++) a = fma(q[d], k[d], a);
      }
      a = simd_sum(a) * scale;
      sc[i] = b + i < j1 ? a : NEG;
      mb = max(mb, sc[i]);
    }
    const float mn = max(m, mb);
    const float c = m == NEG ? 0.0f : metal::exp(m - mn);
    l *= c;
    for (int d = 0; d < 16; d++) o[d] *= c;
    for (int i = 0; i < KB; i++) {
      if (b + i < j1) {
        const float p = metal::exp(sc[i] - mn);
        l += p;
        const float pb = float(bfloat(p));
        load16(key(b + i), k);
        for (int d = 0; d < 16; d++) o[d] = fma(pb, k[d], o[d]);
      }
    }
    m = mn;
  }
  threadgroup float tm[S], tl[S], to[S * D];
  if (lane == 0) { tm[g] = m; tl[g] = l; }
  for (int d = 0; d < 16; d++) to[g * D + lane * 16 + d] = o[d];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (g != 0) return;
  float top = NEG;
  for (int i = 0; i < S; i++) top = max(top, tm[i]);
  float lt = 0.0f, v[16];
  for (int d = 0; d < 16; d++) v[d] = 0.0f;
  for (int i = 0; i < S; i++) {
    const float w = tm[i] == NEG ? 0.0f : metal::exp(tm[i] - top);
    lt += tl[i] * w;
    for (int d = 0; d < 16; d++) v[d] = fma(to[i * D + lane * 16 + d], w, v[d]);
  }
  lt += metal::exp(SINK[h] - top);
  for (int d = 0; d < 16; d++) v[d] = float(bfloat(v[d] / lt));
  if (ROT && int(lane) * 16 + 16 > D - PE) {
    const float pos = float(whi + META[3]);
    for (int d = 0; d < 16; d += 2) {
      const int dd = int(lane) * 16 + d;
      if (dd >= D - PE) {
        const float th = pos * INV[(dd - (D - PE)) / 2];
        const float c = metal::precise::cos(th);
        const float s = -metal::precise::sin(th);
        const float a = v[d], bb = v[d + 1];
        v[d] = a * c - bb * s;
        v[d + 1] = a * s + bb * c;
      }
    }
  }
  device bfloat* op = OUT + (size_t(r) * 64 + h) * D + lane * 16;
  for (int d = 0; d < 16; d++) op[d] = bfloat(v[d]);
