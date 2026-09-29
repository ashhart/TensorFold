
  // expert_down_y for any width: simdgroup (row, slot) pair, dims 8 b .. 8 b + 7, one fp32 sum a dim
  const uint lane = thread_index_in_simdgroup;
  const int R = rows[0];
  const int pair = int(threadgroup_position_in_grid.z) * SG + int(simdgroup_index_in_threadgroup);
  constexpr int SLOTS = TOPK + 1;
  if (pair >= R * SLOTS) return;
  const int r = pair / SLOTS, k = pair % SLOTS;
  const int d0 = int(threadgroup_position_in_grid.y) * 8;
  const bool shared = k == TOPK;
  const device bfloat* x = ACT + (r * SLOTS + k) * NI;
  float out[8];
  if (shared) down_rows<SWB, SWG, NI>(SDW, SDS, SDB, size_t(d0), x, lane, out);
  else down_rows<WB, WG, NI>(DW, DS, DB, size_t(PICK[r * TOPK + k]) * D + d0, x, lane, out);
  for (int row = 0; row < 8; row++) {
    const float v = simd_sum(out[row]);
    if (lane == 0) Y[(r * SLOTS + k) * D + d0 + row] = bfloat(v);
  }
