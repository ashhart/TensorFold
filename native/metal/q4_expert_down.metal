
  // Threadgroup (b, r): TOPK + SHARED simdgroups, simdgroup k takes slot k (the last: the shared expert; the
  // routed slots' experts and weights from expert_gateup's PICK and WTS) for model dims 8 b .. 8 b + 7: lane l reads 16-input chunks l and (l < NC - 32) 32 + l of each row (qmv's inner loop);
  // y = bf16(sum). routed = bf16(sum_k y_k w_k) (fp32, slots in order), shared = bf16(y_s * bf16(sigmoid(logit))),
  // out = bf16(routed + shared).
  const uint lane = thread_index_in_simdgroup;
  const int k = int(simdgroup_index_in_threadgroup);
  const int r = int(threadgroup_position_in_grid.z);
  const int d0 = int(threadgroup_position_in_grid.y) * 8;
  constexpr int SLOTS = TOPK + SHARED;
  constexpr int KB = NI / 2;
  constexpr int KG = NI / 32;
  constexpr int NC = NI / 16;                        // 16-input chunks a row
  threadgroup float ys[SLOTS][8];
  threadgroup float wts[TOPK];
  const bool shared = k == TOPK;
  // the routing expert_gateup found: slot k's expert, and the renormalized top-k weights
  const size_t e = shared ? 0 : size_t(PICK[r * TOPK + k]);
  if (k == 0 && int(lane) < TOPK) wts[lane] = WTS[r * TOPK + lane];
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
    if (lane == 0) ys[k][row] = float(bfloat(acc));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (k == 0 && lane < 8) {
    float routed = 0.0f;
    for (int kk = 0; kk < TOPK; kk++) routed = fma(ys[kk][lane], wts[kk], routed);
    float out = float(bfloat(routed));
    if (SHARED) out = float(bfloat(out + float(bfloat(ys[TOPK][lane] * bsig(float(bfloat(LOGITS[r * NL + NL - 1])))))));
    ROUTED[r * D + d0 + int(lane)] = bfloat(out);
  }
