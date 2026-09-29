
  // _MOE_COMBINE with the routed experts' outputs Y [R][TOPK][D] and the shared expert's YS [R][D] apart
  const uint gid = thread_position_in_grid.x;
  const int r = int(gid / uint(D)), d = int(gid % uint(D));
  if (r >= int(WTS_shape[0])) return;
  const device bfloat* y = Y + size_t(r) * TOPK * D + d;
  float acc = WTS[r * TOPK] * float(y[0]);
  for (int k = 1; k < TOPK; k++) acc = mul_add(acc, WTS[r * TOPK + k], float(y[size_t(k) * D]));
  OUT[size_t(r) * D + d] = bfloat(acc) + YS[size_t(r) * D + d];
