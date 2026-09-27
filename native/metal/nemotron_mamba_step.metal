
  // grid (32, DH, H): lane = 4 state elements, y = channel d of head h. Rows are consecutive tokens.
  const uint lane = thread_position_in_threadgroup.x;
  const uint d = thread_position_in_grid.y;
  const uint h = thread_position_in_grid.z;
  const uint g = h / (H / NG);
  const int R = dims[0];
  constexpr int NS = DS / 32;
  constexpr int CD = XD + 2 * NG * DS;          // conv channels: x, B, C
  const int cx = int(h) * DH + int(d);
  const int cb = XD + int(g) * DS + int(lane) * NS;
  const int cc = XD + NG * DS + int(g) * DS + int(lane) * NS;
  float st[NS];
  const int sbase = (cx * DS) + int(lane) * NS;
  for (int i = 0; i < NS; i++) st[i] = float(S_IN[sbase + i]);
  const float A = -metal::exp(float(A_LOG[h]));
  const float dskip = float(bfloat(float(DSKIP[h])));
  const float dtb = float(DT_BIAS[h]);

  // conv of channel ch at row rr: taps over inputs rr-3 .. rr (rows < 0 come from the conv state)
  #define TAP(ch, pos) ((pos) < 0 ? float(CS_IN[((pos) + KC - 1) * CD + (ch)]) : float(P[(pos) * PROJ + XOFF + (ch)]))
  #define CONV(ch, rr, out) { \
      float a_ = float(CB[ch]); \
      for (int k_ = 0; k_ < KC; k_++) a_ = fma(CW[k_ * CD + (ch)], TAP(ch, (rr) - (KC - 1) + k_), a_); \
      const float cv_ = float(bfloat(a_)); \
      out = float(bfloat(cv_ / (1.0f + metal::exp(-cv_)))); }

  // B and C of this head's group, every row, computed once per threadgroup (thread tid owns one of 2 DS channels)
  threadgroup float bc[MAXR * 2 * DS];
  const uint tid = thread_position_in_threadgroup.y * 32 + lane;
  for (int rr = 0; rr < R; rr++) {
    for (uint c = tid; c < 2 * DS; c += 32 * TGY) {
      const int ch = c < DS ? XD + int(g) * DS + int(c) : XD + NG * DS + int(g) * DS + int(c - DS);
      float v; CONV(ch, rr, v);
      bc[rr * 2 * DS + c] = v;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int rr = 0; rr < R; rr++) {
    float xv = 0.0f;
    if (lane == 0) { CONV(cx, rr, xv); }
    xv = simd_broadcast(xv, 0);
    float bv[NS], cvv[NS];
    for (int i = 0; i < NS; i++) {
      bv[i] = bc[rr * 2 * DS + int(lane) * NS + i];
      cvv[i] = bc[rr * 2 * DS + DS + int(lane) * NS + i];
    }
    float dt = float(P[rr * PROJ + DTOFF + int(h)]) + dtb;
    dt = metal::max(dt, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(dt)));   // softplus (logaddexp(x, 0))
    dt = metal::clamp(dt, limits[0], limits[1]);
    const float dA = metal::exp(A * dt);
    const float xdt = xv * dt;
    float acc = 0.0f;
    for (int i = 0; i < NS; i++) {
      const float s = dA * st[i] + xdt * bv[i];
      st[i] = s;
      acc += s * cvv[i];
    }
    acc = simd_sum(acc);
    if (lane == 0) {
      const float y = float(bfloat(acc + xv * dskip));
      const float z = float(P[rr * PROJ + cx]);
      const float sz = float(bfloat(z / (1.0f + metal::exp(-z))));
      Y[rr * XD + cx] = bfloat(sz * y);
    }
    // the SSM state after this row (a verify window keeps the state of its last accepted row)
    for (int i = 0; i < NS; i++) S_OUT[size_t(rr) * SSZ + sbase + i] = st[i];
    // the conv state after this row: its last KC-1 inputs; each channel written by one thread
    if (lane == 0) {
      for (int k = 0; k < KC - 1; k++) {
        const int pos = rr - (KC - 2) + k;
        CS_OUT[(rr * (KC - 1) + k) * CD + cx] = pos < 0 ? CS_IN[(pos + KC - 1) * CD + cx] : P[pos * PROJ + XOFF + cx];
      }
    }
    if ((h % (H / NG)) == 0 && d == 0) {
      for (int i = 0; i < NS; i++) {
        for (int k = 0; k < KC - 1; k++) {
          const int pos = rr - (KC - 2) + k;
          CS_OUT[(rr * (KC - 1) + k) * CD + cb + i] =
              pos < 0 ? CS_IN[(pos + KC - 1) * CD + cb + i] : P[pos * PROJ + XOFF + cb + i];
          CS_OUT[(rr * (KC - 1) + k) * CD + cc + i] =
              pos < 0 ? CS_IN[(pos + KC - 1) * CD + cc + i] : P[pos * PROJ + XOFF + cc + i];
        }
      }
    }
  }
