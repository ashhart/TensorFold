
  // Thread (e, r): element e of row r's streams: bf16((h * rinv(stream)) * scale), kernels.hc_project's normed input.
  const int e = int(thread_position_in_grid.x);
  const int r = int(thread_position_in_grid.y);
  constexpr int W = S * D;
  const float rv = stream_rinv(SSP, r, e / D, D / 256, S, D, eps[0]);
  NORMED[size_t(r) * W + e] = bfloat((float(HN[size_t(r) * W + e]) * rv) * NW[e]);
