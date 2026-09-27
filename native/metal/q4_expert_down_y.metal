
  // Threadgroup (b, z): SG simdgroups, each one (row, slot) pair (z SG + g in row-major order; slot TOPK: the shared
  // expert; routed slots' experts from expert_gateup's PICK) for model dims 8 b .. 8 b + 7: lane l reads 16-input
  // chunks l and (l < NC - 32) 32 + l of the activation; Y[row][slot][d] = bf16(sum), expert_down's sums. The
  // combine (weights, shared gate) is the next hc_norm's "grouped" write-back, expert_down's arithmetic.
  const uint lane = thread_index_in_simdgroup;
  const int R = rows[0];
  const int pair = int(threadgroup_position_in_grid.z) * SG + int(simdgroup_index_in_threadgroup);
  constexpr int SLOTS = TOPK + 1;
  if (pair >= R * SLOTS) return;
  const int r = pair / SLOTS, k = pair % SLOTS;
  const int d0 = int(threadgroup_position_in_grid.y) * 8;
  constexpr int KB = NI / 2;
  constexpr int KG = NI / 32;
  constexpr int NC = NI / 16;
  const bool shared = k == TOPK;
  const size_t e = shared ? 0 : size_t(PICK[r * TOPK + k]);
  const device uint32_t* DWp = shared ? SDW : DW;
  const device bfloat* DSp = shared ? SDS : DS;
  const device bfloat* DBp = shared ? SDB : DB;
  const device bfloat* x = ACT + (r * SLOTS + k) * NI;
  float xa[16], xb[16];
  const float sa = load16(x + lane * 16, xa);
  const bool second = int(lane) < NC - 32;
  const float sb = second ? load16(x + (32 + lane) * 16, xb) : 0.0f;
  for (int row = 0; row < 8; row++) {
    const size_t at = e * D + d0 + row;
    const device uint8_t* w = (const device uint8_t*)DWp + at * KB;
    float acc = qdot16(w + lane * 8, xa, float(DSp[at * KG + lane / 2]), float(DBp[at * KG + lane / 2]), sa);
    if (second)
      acc += qdot16(w + (32 + lane) * 8, xb, float(DSp[at * KG + (32 + lane) / 2]), float(DBp[at * KG + (32 + lane) / 2]), sb);
    acc = simd_sum(acc);
    if (lane == 0) Y[(r * SLOTS + k) * D + d0 + row] = bfloat(acc);
  }
