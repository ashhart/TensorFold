
  // the partial sums of squares of row m as a residual epilogue writes them: partial t covers [8 t, 8 t + 8), two
  // sequential fma runs of 4 (one a simdgroup there) added in order
  const uint t = thread_position_in_grid.x;
  const uint m = thread_position_in_grid.y;
  if (t >= uint(K / 8)) return;
  float total = 0.0f;
  for (int s2 = 0; s2 < 2; s2++) {
    float ss = 0.0f;
    for (int j = 0; j < 4; j++) {
      const float h = float(H[m * K + t * 8 + s2 * 4 + j]);
      ss = fma(h, h, ss);
    }
    total += ss;
  }
  PO[m * (K / 8) + t] = total;
