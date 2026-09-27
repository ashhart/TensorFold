
  // fc1 and mlx_lm's relu2 on its bf16 output: bf16(max(bf16(sum), 0)^2)
  const uint lane = thread_index_in_simdgroup;
  const int p = int(threadgroup_position_in_grid.z);
  const size_t e = size_t(IDS[p]);
  const int row0 = (int(threadgroup_position_in_grid.y) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
  const size_t at = e * N + size_t(row0);
  float acc[RPS];
  tf_rowdot<K, GS, RPS>((const device uint8_t*)W + at * (K / 2), S + at * (K / GS), B + at * (K / GS),
                        X + size_t(p / TOPK) * K, lane, acc);
  if (lane == 0)
    for (int j = 0; j < RPS; j++) {
      const float h = metal::max(float(bfloat(acc[j])), 0.0f);
      ACT[size_t(p) * N + row0 + j] = bfloat(h * h);
    }
