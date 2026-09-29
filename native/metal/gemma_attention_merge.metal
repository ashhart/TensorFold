
  // one simdgroup per (query head, row): the chunks' partials in chunk order
  const uint lane = thread_index_in_simdgroup;
  const int qh = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  const int NCH = META[1], R = META[3];
  constexpr int DPL = D / 32;
  const size_t base = (size_t(qh) * R + r) * NCH;
  float top = -INFINITY;
  for (int c = 0; c < NCH; c++) top = metal::max(top, PM[base + c]);
  float lsum = 0.0f, acc[DPL];
  for (int i = 0; i < DPL; i++) acc[i] = 0.0f;
  for (int c = 0; c < NCH; c++) {
    const float e = PL[base + c] > 0.0f ? metal::exp(PM[base + c] - top) : 0.0f;
    lsum = fma(PL[base + c], e, lsum);
    for (int i = 0; i < DPL; i++) acc[i] = fma(PO[(base + c) * D + lane * DPL + i], e, acc[i]);
  }
  for (int i = 0; i < DPL; i++) OUT[(size_t(r) * H + qh) * D + lane * DPL + i] = bfloat(acc[i] / lsum);
