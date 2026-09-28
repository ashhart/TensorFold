
  // One threadgroup of 32 simdgroups per value head hv (key head hv / (NV / NK)); simdgroup s owns state rows
  // dv = 4 s .. 4 s + 3, lane l their columns dk = 4 l .. 4 l + 3 (the layout of mlx_lm's gated_delta kernel).
  // P rows are [qkv (C) | z (NV DV) | b (NV) | a (NV)]; the conv reads [conv state (TAPS - 1 rows); P rows].
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int hv = int(threadgroup_position_in_grid.x);
  const int hk = hv / (NV / NK);
  const int R = rows[0];
  constexpr int C = 2 * NK * DK + NV * DV;
  constexpr int PW = C + NV * DV + 2 * NV;
  constexpr int RPS = DV / 32;                              // state rows a simdgroup
  threadgroup float qs[DK], ks[DK], vs[DV], ys[DV];
  threadgroup float red[2][32];
  threadgroup float gates[2];
  // this head's conv channels: q (hk), k (hk), v (hv)
  int c = -1;
  if (int(t) < DK) c = hk * DK + int(t);
  else if (int(t) < 2 * DK) c = NK * DK + hk * DK + int(t) - DK;
  else if (int(t) < 2 * DK + DV) c = 2 * NK * DK + hv * DV + int(t) - 2 * DK;
  const bool writes_qk = (hv % (NV / NK)) == 0;
  float state[RPS][4];
  for (int j = 0; j < RPS; j++)
    for (int i = 0; i < 4; i++)
      state[j][i] = HAS_STATE ? SIN[(size_t(hv) * DV + sg * RPS + j) * DK + lane * 4 + i] : 0.0f;
  for (int r = 0; r < R; r++) {
    if (c >= 0) {
      float conv = 0.0f;
      for (int tap = 0; tap < TAPS; tap++) {
        const int at = r + tap;                               // into [conv state; P rows]
        const float xv = at < TAPS - 1 ? float(CS[at * C + c]) : float(P[(at - (TAPS - 1)) * PW + c]);
        conv = fma(float(CW[c * TAPS + tap]), xv, conv);
      }
      const float act = bsilu(conv);                          // conv + SiLU in fp32, stored as bf16
      if (int(t) < DK) qs[t] = act;
      else if (int(t) < 2 * DK) ks[int(t) - DK] = act;
      else vs[int(t) - 2 * DK] = act;
      if (c < 2 * NK * DK ? writes_qk : true) {
        for (int j = 0; j < TAPS - 1; j++) {                  // the conv window after this row
          const int at = r + 1 + j;
          CSO[(r * (TAPS - 1) + j) * C + c] = at < TAPS - 1 ? CS[at * C + c] : P[(at - (TAPS - 1)) * PW + c];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < 2) {
      // q (sg 0) and k (sg 1): x / sqrt(sum(x^2) + 1e-6) in fp32 (the delta-rule kernel's in-kernel L2 norm),
      // q also times DK^-0.5; both stay fp32
      threadgroup float* x = sg == 0 ? qs : ks;
      float ss = 0.0f;
      for (int i = 0; i < DK / 32; i++) {
        const float v = x[lane * (DK / 32) + i];
        ss = fma(v, v, ss);
      }
      ss = simd_sum(ss);
      const float inv = metal::precise::rsqrt(ss + 1e-6f) * (sg == 0 ? metal::precise::rsqrt(float(DK)) : 1.0f);
      for (int i = 0; i < DK / 32; i++) x[lane * (DK / 32) + i] *= inv;
    } else if (sg == 2 && lane == 0) {
      // g = exp(-exp(A_log) * softplus(a + dt_bias)) in fp32, beta = sigmoid(b) as bf16
      const float b = float(P[r * PW + C + NV * DV + hv]);
      const float a = float(P[r * PW + C + NV * DV + NV + hv]);
      gates[0] = metal::precise::exp(-metal::precise::exp(float(ALOG[hv])) * fsoftplus(a + float(DT[hv])));
      gates[1] = bsig(b);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float g = gates[0], beta = gates[1];
    float kk[4], qq[4];
    for (int i = 0; i < 4; i++) { kk[i] = ks[lane * 4 + i]; qq[i] = qs[lane * 4 + i]; }
    for (int j = 0; j < RPS; j++) {
      const int dv = int(sg) * RPS + j;
      float kv = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv += state[j][i] * kk[i];
      }
      kv = simd_sum(kv);
      const float delta = (vs[dv] - kv) * beta;
      float out = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] + kk[i] * delta;
        out += state[j][i] * qq[i];
      }
      out = simd_sum(out);
      if (lane == 0) ys[dv] = float(bfloat(out));
      for (int i = 0; i < 4; i++) SO[((size_t(r) * NV + hv) * DV + dv) * DK + lane * 4 + i] = state[j][i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float ss = 0.0f;
      for (int i = 0; i < DV / 32; i++) { const float v = ys[lane * (DV / 32) + i]; ss = fma(v, v, ss); }
      ss = simd_sum(ss);
      if (lane == 0) red[0][0] = metal::precise::rsqrt(ss / float(DV) + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(t) < DV) {
      // sigmoid-gated RMSNorm: mx.fast.rms_norm's bf16(w * bf16(y * inv)), times sigmoid(z) in fp32, bf16 out
      const float y = float(bfloat(float(NW[t]) * float(bfloat(ys[t] * red[0][0]))));
      const float z = float(P[r * PW + C + hv * DV + int(t)]);
      OUT[r * NV * DV + hv * DV + int(t)] = bfloat(y * fsig(z));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
