
  // SiLU(gate) * up over [gate | up] rows of 2N
  const uint i = thread_position_in_grid.x;
  const uint m = thread_position_in_grid.y;
  if (i >= uint(N)) return;
  const float gf = float(GU[m * 2 * N + i]);
  HOUT[m * N + i] = bfloat(gf / (1.0f + metal::precise::exp(-gf)) * float(GU[m * 2 * N + N + i]));
