
  // Thread (d, r): dim d of row r's block input: mean over streams of bf16(sigmoid(up) * normed).
  const int d = int(thread_position_in_grid.x);
  const int r = int(thread_position_in_grid.y);
  constexpr int W = S * D;
  float total = 0.0f;
  for (int s = 0; s < S; s++) {
    const size_t e = size_t(r) * W + s * D + d;
    total += float(bfloat(bsig(float(UP[e])) * float(NORMED[e])));
  }
  MIXED[size_t(r) * D + d] = bfloat(total / float(S));
