
  // SiLU(gate) * up: bf16(bf16(silu(g)) * u); gate and up are N-wide runs of
  // their rows (stride S, offsets GO and UO: two arrays, or two parts of one stacked projection)
  const uint i = thread_position_in_grid.x;
  const int row = int(i) / N, col = int(i) % N;
  const float g = float(G[row * GSTRIDE + GO + col]);
  const float u = float(U[row * USTRIDE + UO + col]);
  OUT[i] = bfloat(bsilu(g) * u);
