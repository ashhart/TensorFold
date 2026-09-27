
  // Threadgroup (b, u): MAXM simdgroups, simdgroup m takes the m-th row that picked distinct expert u (MAXU: the
  // shared expert) for model dims 8 b .. 8 b + 7: lane l reads 16-input chunks l and (l < NC - 32) 32 + l of the
  // row's activation; Y[row][slot][d] = bf16(sum), expert_down's sums. Weight rows read once for all members.
  const uint lane = thread_index_in_simdgroup;
  const int m = int(simdgroup_index_in_threadgroup);
  const int u = int(threadgroup_position_in_grid.z);
  const int R = rows[0];
  constexpr int SLOTS = TOPK + 1;
  const bool shared = u == MAXU;
  if (!shared && u >= UCOUNT[0]) return;
  const int member = shared ? (m < R ? m * 32 + TOPK : -1) : UMEM[u * MAXR + m];
  if (member < 0) return;
  const size_t e = shared ? 0 : size_t(UIDS[u]);
  const int r = member / 32, slot = member % 32;
  const int d0 = int(threadgroup_position_in_grid.y) * 8;
  constexpr int KB = NI / 2;
  constexpr int KG = NI / 32;
  constexpr int NC = NI / 16;
  const device uint32_t* DWp = shared ? SDW : DW;
  const device bfloat* DSp = shared ? SDS : DS;
  const device bfloat* DBp = shared ? SDB : DB;
  const bool second = int(lane) < NC - 32;
  const device bfloat* x = ACT + (r * SLOTS + slot) * NI;
  float xa[16], xb[16];
  const bool first = int(lane) < NC;
  const float sa = first ? load16(x + lane * 16, xa) : 0.0f;
  const float sb = second ? load16(x + (32 + lane) * 16, xb) : 0.0f;
  for (int row = 0; row < 8; row++) {
    const size_t at = e * D + d0 + row;
    const device uint8_t* w = (const device uint8_t*)DWp + at * KB;
    float acc = first ? qdot16(w + lane * 8, xa, float(DSp[at * KG + lane / 2]), float(DBp[at * KG + lane / 2]), sa) : 0.0f;
    if (second)
      acc += qdot16(w + (32 + lane) * 8, xb, float(DSp[at * KG + (32 + lane) / 2]), float(DBp[at * KG + (32 + lane) / 2]), sb);
    acc = simd_sum(acc);
    if (lane == 0) Y[(r * SLOTS + slot) * D + d0 + row] = bfloat(acc);
  }
