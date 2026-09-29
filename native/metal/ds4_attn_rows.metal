
  const int r = int(threadgroup_position_in_grid.y) / (64 / HTG);
  const int hg = int(threadgroup_position_in_grid.y) % (64 / HTG);
  const uint t = thread_position_in_threadgroup.x;
  constexpr int TPG = HTG * 16;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  constexpr int D = 512;
  constexpr int H = 64;
  constexpr float NEG = -INFINITY;
  threadgroup bfloat Ks[BK * D];
  const int prefix = META[0], pm = META[1], rs = META[2];
  const float scale = SCALE[0];
  const int pc = PCOUNT[r];
  const int wlo = WPOS[2 * r], whi = WPOS[2 * r + 1];
  const int n = pc + (whi - wlo + 1);
  const int h0 = hg * HTG + int(sg) * 2;
  float q0[16], q1[16], o0[16], o1[16];
  const device bfloat* qp = Q + (size_t(r) * H + h0) * D + lane * 16;
  for (int d = 0; d < 16; d++) { q0[d] = float(qp[d]); q1[d] = float(qp[D + d]); o0[d] = 0.0f; o1[d] = 0.0f; }
  float m0 = NEG, m1 = NEG, l0 = 0.0f, l1 = 0.0f;
  for (int b = 0; b < n; b += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = int(t); e < BK * D / 8; e += TPG) {
      const int i = e / (D / 8), c = e - i * (D / 8), g = b + i;
      uint4 v = uint4(0);
      if (g < pc) {
        const int row = prefix ? g : PIDX[size_t(r) * pm + g];
        v = ((const device uint4*)(POOL + size_t(row) * D))[c];
      } else if (g < n) {
        v = ((const device uint4*)(RING + size_t((wlo + g - pc) % rs) * D))[c];
      }
      ((threadgroup uint4*)Ks)[e] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s0[BK], s1[BK];
    float mb0 = NEG, mb1 = NEG;
    for (int i = 0; i < BK; i++) {
      const threadgroup bfloat* kp = Ks + i * D + lane * 16;
      float a0 = 0.0f, a1 = 0.0f;
      for (int d = 0; d < 16; d++) { const float k = float(kp[d]); a0 = fma(q0[d], k, a0); a1 = fma(q1[d], k, a1); }
      a0 = simd_sum(a0) * scale;
      a1 = simd_sum(a1) * scale;
      s0[i] = (b + i < n) ? a0 : NEG;
      s1[i] = (b + i < n) ? a1 : NEG;
      mb0 = max(mb0, s0[i]);
      mb1 = max(mb1, s1[i]);
    }
    const float n0 = max(m0, mb0), n1 = max(m1, mb1);
    const float c0 = m0 == NEG ? 0.0f : metal::exp(m0 - n0), c1 = m1 == NEG ? 0.0f : metal::exp(m1 - n1);
    l0 *= c0; l1 *= c1;
    for (int d = 0; d < 16; d++) { o0[d] *= c0; o1[d] *= c1; }
    for (int i = 0; i < BK; i++) {
      const float p0 = s0[i] == NEG ? 0.0f : metal::exp(s0[i] - n0);
      const float p1 = s1[i] == NEG ? 0.0f : metal::exp(s1[i] - n1);
      l0 += p0; l1 += p1;
      const float b0 = float(bfloat(p0)), b1 = float(bfloat(p1));
      const threadgroup bfloat* kp = Ks + i * D + lane * 16;
      for (int d = 0; d < 16; d++) {
        const float k = float(kp[d]);
        o0[d] = fma(b0, k, o0[d]);
        o1[d] = fma(b1, k, o1[d]);
      }
    }
    m0 = n0; m1 = n1;
  }
  l0 += metal::exp(SINK[h0] - m0);
  l1 += metal::exp(SINK[h0 + 1] - m1);
  device bfloat* op = OUT + (size_t(r) * H + h0) * D + lane * 16;
  float v0[16], v1[16];
  for (int d = 0; d < 16; d++) { v0[d] = float(bfloat(o0[d] / l0)); v1[d] = float(bfloat(o1[d] / l1)); }
  if (ROT && int(lane) * 16 + 16 > D - PE) {
    const float pos = float(whi + META[3]);
    for (int d = 0; d < 16; d += 2) {
      const int dd = int(lane) * 16 + d;
      if (dd >= D - PE) {
        const float th = pos * INV[(dd - (D - PE)) / 2];
        const float c = metal::precise::cos(th);
        const float s = -metal::precise::sin(th);
        const float a0 = v0[d], b0 = v0[d + 1], a1 = v1[d], b1 = v1[d + 1];
        v0[d] = a0 * c - b0 * s;
        v0[d + 1] = a0 * s + b0 * c;
        v1[d] = a1 * c - b1 * s;
        v1[d + 1] = a1 * s + b1 * c;
      }
    }
  }
  for (int d = 0; d < 16; d++) { op[d] = bfloat(v0[d]); op[D + d] = bfloat(v1[d]); }
