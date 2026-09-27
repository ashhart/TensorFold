
  const uint lane = thread_index_in_simdgroup;
  const uint hk = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;              // row t * G + g
  const int NCH = dims[1], NQ = dims[2], SGA = dims[4];
  const int RP = 16 * SGA;
  constexpr int DP = D / 32;
  float m = -INFINITY, l = 0.0f, o[DP];
  for (int i = 0; i < DP; i++) o[i] = 0.0f;
  for (int c = 0; c < NCH; c++) {
    const int64_t row = ((int64_t)hk * NCH + c) * RP + r;
    const float mc = PM[row];
    if (mc == -INFINITY) continue;
    const float lc = PL[row];
    const float nm = max(m, mc);
    const float f1 = fast::exp(m - nm), f2 = fast::exp(mc - nm);
    l = l * f1 + lc * f2;
    for (int i = 0; i < DP; i++) o[i] = o[i] * f1 + PO[row * D + lane * DP + i] * f2;
    m = nm;
  }
  const int t = r / G, g = r % G, h = hk * G + g;
  for (int i = 0; i < DP; i++) OUT[((int64_t)h * NQ + t) * D + lane * DP + i] = static_cast<bfloat>(o[i] / l);
