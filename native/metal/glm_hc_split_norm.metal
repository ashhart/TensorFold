
  constexpr int S = 4;
  constexpr int F = S * D;                          // flattened streams
  constexpr int MIX = (2 + S) * S;                  // 24 mixes

  // Threadgroup r: sinkhorn, pre / post / comb (hc_sinkhorn_collapse), the collapse and the RMSNorm
  constexpr float HC_EPS = HC_EPS_INT * 1e-9;       // as the hc_split kernel spells its eps
  const int r = int(threadgroup_position_in_grid.x);
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  threadgroup float red[32];
  threadgroup float pre_s[S];
  threadgroup float inv_s[1];
  device const float* mixes = MIXES + r * MIX;
  device const bfloat* xs = X + size_t(r) * F;
  if (sg == 0) {
    constexpr int BASE_OFF = 2 * S;
    const float pre_scale = SCALE[0], post_scale = SCALE[1], comb_scale = SCALE[2];
    const float active = (lane < (uint)S) ? 1.0f : 0.0f;
    const uint llane = metal::min(lane, (uint)(S - 1));
    const float pre_z = mixes[llane] * pre_scale + BASEV[llane];
    const float post_z = mixes[S + llane] * post_scale + BASEV[S + llane];
    const float pre_v = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + HC_EPS;
    const float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
    if (lane < (uint)S) { pre_s[lane] = pre_v; POST_OUT[r * S + lane] = post_v; }
    float4 v = (*(const device float4*)(mixes + BASE_OFF + llane * S) * comb_scale
                + *(const device float4*)(BASEV + BASE_OFF + llane * S)) * active;
    const float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
    const float4 e = metal::fast::exp(v - row_max) * active;
    float4 rr = e * (1.0f / (e.x + e.y + e.z + e.w + HC_EPS)) + HC_EPS * active;
    float4 col_inv = 1.0f / (float4(simd_sum(rr.x), simd_sum(rr.y), simd_sum(rr.z), simd_sum(rr.w)) + HC_EPS);
    rr *= col_inv;
    for (int iter = 1; iter < ITERS; ++iter) {
      rr *= (1.0f / (rr.x + rr.y + rr.z + rr.w + HC_EPS)) * active;
      col_inv = 1.0f / (float4(simd_sum(rr.x), simd_sum(rr.y), simd_sum(rr.z), simd_sum(rr.w)) + HC_EPS);
      rr *= col_inv;
    }
    if (lane < (uint)S) *(device float4*)(COMB_OUT + r * S * S + lane * S) = rr;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float p0 = pre_s[0], p1 = pre_s[1], p2 = pre_s[2], p3 = pre_s[3];
  float xc[4];
  float acc = 0.0f;
  for (int i = 0; i < 4; ++i) {
    const int d = int(t) * 4 + i;
    const float res = fma(p0, float(xs[d]), fma(p1, float(xs[D + d]),
                          fma(p2, float(xs[2 * D + d]), p3 * float(xs[3 * D + d]))));
    xc[i] = float(bfloat(res));
    acc = sq_acc<SQ_FMA>(acc, xc[i]);
  }
  acc = simd_sum(acc);
  if (sg == 0) red[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) red[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float a = simd_sum(red[lane]);
    if (lane == 0) inv_s[0] = metal::precise::rsqrt(a / float(D) + EPS[0]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < 4; ++i) {
    const int d = int(t) * 4 + i;
    NORMED[size_t(r) * D + d] = NORMW[d] * bfloat(xc[i] * inv_s[0]);
  }
