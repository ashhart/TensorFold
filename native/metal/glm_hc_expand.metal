
  constexpr int S = 4;
  constexpr int F = S * D;                          // flattened streams
  constexpr int MIX = (2 + S) * S;                  // 24 mixes

  // Threadgroup r (1024 threads): the pending write-back (EXPAND) and the streams' RMS scale (SPLIT).
  const int r = int(threadgroup_position_in_grid.x);
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  threadgroup float red[32];
  device const bfloat* xo = XOLD + size_t(r) * F;
  device bfloat* xn = XNEW + size_t(r) * F;
  float ss = 0.0f;
  for (int k = 0; k < F / 4096; ++k) {
    for (int i = 0; i < 4; ++i) {
      const int f = int(t) * 4 + 4096 * k + i;
      float v;
      if (EXPAND) {
        const int s = f / D, d = f - s * D;
        const float y = POST[r * S + s] * float(BRANCH[size_t(r) * D + d]);
        const device float* c = COMB + r * S * S;
        float mm = c[0 * S + s] * float(xo[0 * D + d]);
        mm = fma(c[1 * S + s], float(xo[1 * D + d]), mm);
        mm = fma(c[2 * S + s], float(xo[2 * D + d]), mm);
        mm = fma(c[3 * S + s], float(xo[3 * D + d]), mm);
        const bfloat nb = bfloat(add_nc(y, mm));
        xn[f] = nb;
        v = float(nb);
      } else {
        v = float(xo[f]);
      }
      ss = sq_acc<SQ_FMA>(ss, v);
    }
  }
  if (!SPLIT) return;
  ss = simd_sum(ss);
  if (sg == 0) red[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) red[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float a = simd_sum(red[lane]);
    if (lane == 0) INV[r] = metal::precise::rsqrt(a / float(F) + EPS[0]);
  }
