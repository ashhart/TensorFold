
  // Threadgroup (h, r), a thread a dim: the P parts of head h of row r combined in part order.
  const int d = int(thread_position_in_threadgroup.x);
  const int h = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  const size_t at = (size_t(r) * H + h) * P;
  float top = -INFINITY;
  for (int k = 0; k < P; k++) top = metal::max(top, PM[(at + k) * 2]);
  float total = 0.0f, acc = 0.0f;
  for (int k = 0; k < P; k++) {
    const float mk = PM[(at + k) * 2];
    const float w = mk == -INFINITY ? 0.0f : metal::exp(mk - top);
    total = fma(PM[(at + k) * 2 + 1], w, total);
    acc = fma(PO[(at + k) * D + d], w, acc);
  }
  OUT[(size_t(r) * H + h) * D + d] = bfloat(acc / total);
