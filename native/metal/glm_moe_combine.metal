
  // out[r][d] = bf16(bf16(sum_k w_k y_k, fp32 in slot order, each product rounded before its add) + shared)
  const uint gid = thread_position_in_grid.x;
  const int r = int(gid / uint(D)), d = int(gid % uint(D));
  if (r >= int(WTS_shape[0])) return;
  constexpr int SLOTS = TOPK + 1;
  const device bfloat* y = Y + size_t(r) * SLOTS * D + d;
  float acc = WTS[r * TOPK] * float(y[0]);
  for (int k = 1; k < TOPK; k++) acc = mul_add(acc, WTS[r * TOPK + k], float(y[size_t(k) * D]));
  OUT[size_t(r) * D + d] = bfloat(acc) + y[size_t(TOPK) * D];
