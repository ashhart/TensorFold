
  const uint gid = thread_position_in_grid.x;
  const int r = int(gid / uint(D)), d = int(gid % uint(D));
  if (r >= int(YS_shape[0])) return;
  const device bfloat* y = Y + size_t(r) * TOPK * D + d;
  float acc = 0.0f;
  for (int k = 0; k < TOPK; k++) acc += float(y[size_t(k) * D]);
  OUT[size_t(r) * D + d] = bfloat(acc + float(YS[size_t(r) * D + d]));
