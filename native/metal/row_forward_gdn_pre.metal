
  // one simdgroup per (row w, head): q heads [0, NK), k heads [NK, 2 NK), v heads [2 NK, 2 NK + NV)
  const uint lane = thread_index_in_simdgroup;
  const uint head = threadgroup_position_in_grid.y;
  const uint w = threadgroup_position_in_grid.z;
  constexpr int C = 2 * NK * DK + NV * DV;
  const bool isq = head < NK, isk = !isq && head < 2 * NK;
  const int c0 = isq ? int(head) * DK : (isk ? NK * DK + (int(head) - NK) * DK : 2 * NK * DK + (int(head) - 2 * NK) * DV);
  constexpr int PER = DK / 32;                          // channels per lane (DK == DV)
  float vals[PER];
  for (int j = 0; j < PER; j++) {
    const int c = c0 + int(lane) * PER + j;
    float acc = 0.0f;
    for (int tap = 0; tap < TAPS; tap++) {
      const int row = windows[w * TAPS + tap];          // into [conv state rows; window rows]
      const float xv = row < TAPS - 1 ? float(CS[row * C + c]) : float(QKV[(row - (TAPS - 1)) * ZS + c]);
      acc += float(CW[c * TAPS + tap]) * xv;
    }
    const float conv = float(bfloat(acc));
    const float sig = float(bfloat(1.0f / (1.0f + metal::exp(-conv))));
    vals[j] = float(bfloat(conv * sig));                // SiLU, bf16 like mlx_lm's two ops
  }
  if (isq || isk) {
    float ss = 0.0f;
    for (int j = 0; j < PER; j++) ss += vals[j] * vals[j];
    ss = simd_sum(ss);
    const float inv = metal::rsqrt(ss / float(DK) + 1e-6f);
    // mlx_lm: q = (DK^-0.5)^2 * rms_norm(q), k = DK^-0.5 * rms_norm(k), scales rounded to bf16
    const float scale = isq ? float(bfloat(1.0f / float(DK))) : float(bfloat(metal::rsqrt(float(DK))));
    for (int j = 0; j < PER; j++) {
      const bfloat out = bfloat(scale * float(bfloat(vals[j] * inv)));
      if (isq) Q[(w * NK + head) * DK + lane * PER + j] = out;
      else Kout[(w * NK + head - NK) * DK + lane * PER + j] = out;
    }
  } else {
    const int hv = int(head) - 2 * NK;
    for (int j = 0; j < PER; j++) Vout[(w * NV + hv) * DV + lane * PER + j] = bfloat(vals[j]);
    if (lane == 0) {
      // g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b) (mlx_lm's compute_g, sigmoid)
      const float s = float(bfloat(float(Ain[w * ZS + AO + hv]) + float(DT[hv])));
      const float sp = float(bfloat(metal::max(s, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(s)))));
      G[w * NV + hv] = metal::exp(-metal::exp(float(ALOG[hv])) * sp);
      BETA[w * NV + hv] = bfloat(1.0f / (1.0f + metal::exp(-float(Bin[w * ZS + BO + hv]))));
    }
  }

  // the conv tail after the last row: the last TAPS - 1 rows of [conv state; window rows], this lane's channels
  if (int(w) == nodes[0] - 1)
    for (int r = 0; r < TAPS - 1; r++) {
      const int row = windows[w * TAPS + 1 + r];
      for (int j = 0; j < PER; j++) {
        const int c = c0 + int(lane) * PER + j;
        CO[r * C + c] = row < TAPS - 1 ? CS[row * C + c] : QKV[(row - (TAPS - 1)) * ZS + c];
      }
    }
