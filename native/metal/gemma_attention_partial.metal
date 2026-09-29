  const float SCALE = as_type<float>(uint(SCALE_BITS));

  // threadgroup (chunk C0 + x, key head h, row r); simdgroup (g, s): query head h G + g, keys k0 + s, k0 + s + S, ...
  const uint lane = thread_index_in_simdgroup;
  const int sgi = int(simdgroup_index_in_threadgroup);
  const int g = sgi / S, s = sgi % S;
  const int C0 = META[0], NCH = META[1], RING = META[2], R = META[3];
  const int c = C0 + int(threadgroup_position_in_grid.x);
  const int h = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  const int qh = h * G + g;
  const int CAP = K_shape[2];
  const int P0 = POS[0];                                    // the call's first row: keys from P0 on are its new rows
  constexpr int DPL = D / 32;
  const int k0 = metal::max(c * CK, LO[r]), k1 = metal::min((c + 1) * CK, POS[r] + 1);
  float q[DPL], o[DPL];
  const device bfloat* qp = Q + (size_t(r) * (HK * G) + qh) * D + lane * DPL;
  for (int i = 0; i < DPL; i++) { q[i] = float(qp[i]) * SCALE; o[i] = 0.0f; }
  float m = -INFINITY, l = 0.0f;
  const device bfloat* kb = K + size_t(h) * CAP * D + lane * DPL;
  const device bfloat* vb = V + size_t(h) * CAP * D + lane * DPL;
  const device bfloat* kn = KN + size_t(h) * R * D + lane * DPL;
  const device bfloat* vn = VN + size_t(h) * R * D + lane * DPL;
  for (int base = k0 + s; base < k1; base += S * BLK) {
    float sc[BLK];
    size_t at[BLK];
    bool fresh[BLK], live[BLK];
    float bm = -INFINITY;
    for (int j = 0; j < BLK; j++) {
      const int p = base + j * S;
      live[j] = p < k1;
      fresh[j] = p >= P0;
      at[j] = size_t(fresh[j] ? p - P0 : (RING ? p % RING : p)) * D;
      float d = 0.0f;
      if (live[j]) {
        const device bfloat* kr = (fresh[j] ? kn : kb) + at[j];
        for (int i = 0; i < DPL; i++) d = fma(q[i], float(kr[i]), d);
      }
      sc[j] = simd_sum(d);
      if (live[j]) bm = metal::max(bm, sc[j]);
    }
    const float mn = metal::max(m, bm);
    const float a = metal::exp(m - mn);
    l *= a;
    for (int i = 0; i < DPL; i++) o[i] *= a;
    for (int j = 0; j < BLK; j++) {
      if (!live[j]) continue;
      const float b = metal::exp(sc[j] - mn);
      l += b;
      const device bfloat* vr = (fresh[j] ? vn : vb) + at[j];
      for (int i = 0; i < DPL; i++) o[i] = fma(b, float(vr[i]), o[i]);
    }
    m = mn;
  }
  const size_t slot = (size_t(qh) * R + r) * NCH + (c - C0);
  if (S == 1) {
    if (lane == 0) { PM[slot] = m; PL[slot] = l; }
    for (int i = 0; i < DPL; i++) PO[slot * D + lane * DPL + i] = o[i];
    return;
  }
  threadgroup float sm[G * S], sl[G * S];
  threadgroup float so[S > 1 ? G * S : 1][S > 1 ? D : 1];
  if (lane == 0) { sm[sgi] = m; sl[sgi] = l; }
  for (int i = 0; i < DPL; i++) so[sgi][lane * DPL + i] = o[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (s != 0) return;
  float top = -INFINITY;
  for (int t = 0; t < S; t++) top = metal::max(top, sm[g * S + t]);
  float lsum = 0.0f, acc[DPL];
  for (int i = 0; i < DPL; i++) acc[i] = 0.0f;
  for (int t = 0; t < S; t++) {
    const float e = sl[g * S + t] > 0.0f ? metal::exp(sm[g * S + t] - top) : 0.0f;
    lsum = fma(sl[g * S + t], e, lsum);
    for (int i = 0; i < DPL; i++) acc[i] = fma(so[g * S + t][lane * DPL + i], e, acc[i]);
  }
  if (lane == 0) { PM[slot] = top; PL[slot] = lsum; }
  for (int i = 0; i < DPL; i++) PO[slot * D + lane * DPL + i] = acc[i];
