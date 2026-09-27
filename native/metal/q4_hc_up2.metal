
  // One threadgroup of 10 * 32 threads per 8 model dims. Prologue: the down projection's outputs from the
  // split-K partials (summed in chunk order) -> bf16 -> / S -> SiLU (bf16); threadgroup 0 also writes the
  // inject gates 2 sigmoid(inject / S). Then its 32 up rows {s * D + d0 + j}: thread t = 10 i + q takes group q of
  // row i; row sums in group order, sigmoid, times the normed stream, mean over streams.
  const uint t = thread_position_in_threadgroup.x;
  const int R = rows[0];
  const int r = int(threadgroup_position_in_grid.y);    // rows run in parallel threadgroups
  constexpr int W = S * D;
  constexpr int GPR = LOW / 32;
  constexpr int ROWS = S * 8;
  const int d0 = int(threadgroup_position_in_grid.x) * 8;
  threadgroup float act[GPR * 33];                    // group q at 33 q: distinct banks across a simdgroup
  threadgroup float part[ROWS][GPR];
  threadgroup float prod[ROWS];
  const int i = int(t) / GPR, q = int(t) % GPR;
  const int s = i / 8, d = d0 + i % 8;
  const int row = s * D + d;
  {
    for (int c = int(t); c < ND; c += ROWS * GPR) {
      float v = 0.0f;
      for (int k = 0; k < KS; k++) v += PART[(k * R + r) * ND + c];
      const float v4 = float(bfloat(float(bfloat(v)) / float(S)));
      if (c < LOW) act[(c / 32) * 33 + c % 32] = bsilu(v4);
      else if (threadgroup_position_in_grid.x == 0) INJOUT[r * S + (c - LOW)] = bfloat(2.0f * bsig(v4));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      float x[32];
      for (int n = 0; n < 32; n++) x[n] = act[q * 33 + n];
      part[i][q] = qgroup_dot(QW + (size_t(row) * GPR + q) * 4, float(QS[row * GPR + q]), float(QB[row * GPR + q]), x);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (q == 0) {
      float u = 0.0f;
      for (int qq = 0; qq < GPR; qq++) u += part[i][qq];
      const float rv = stream_rinv(SSP, r, s, D / 256, S, D, eps[0]);
      const float normed = float(bfloat((float(HN[r * W + row]) * rv) * NW[row]));
      prod[i] = float(bfloat(bsig(float(bfloat(u))) * normed));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 8) {
      float total = 0.0f;
      for (int ss = 0; ss < S; ss++) total += prod[ss * 8 + int(t)];
      MIXED[r * D + d0 + int(t)] = bfloat(total / float(S));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
