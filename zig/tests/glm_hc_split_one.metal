// tf-glm-hc-rows' reference, the one-group split: slice sums, mixes, pre, post, Sinkhorn, then the collapse and norm.
[[kernel]] void tf_hc_split_one(const device bfloat* X [[buffer(0)]], const device float* PART [[buffer(1)]],
                        const constant float* SCALE [[buffer(2)]], const device float* BASEV [[buffer(3)]],
                        const device bfloat* NORMW [[buffer(4)]], const constant float* EPS [[buffer(5)]],
                        device bfloat* NORMED [[buffer(6)]], device float* POST_OUT [[buffer(7)]],
                        device float* COMB_OUT [[buffer(8)]], uint tg [[threadgroup_position_in_grid]],
                        uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr float HC_EPS = TF_HC_EPS_INT * 1e-9;
  const int r = int(tg);
  threadgroup float tot[PARTS];
  threadgroup float mixes[MIX];
  threadgroup float red[32];
  threadgroup float pre_s[S];
  threadgroup float inv_s[1];
  device const bfloat* xs = X + size_t(r) * F;
  float xv[4][S]; // this thread's 4 columns of the streams, loaded while the mixes and the Sinkhorn run
  for (int i = 0; i < 4; ++i) {
    for (int s = 0; s < S; ++s) xv[i][s] = float(xs[s * TF_D + int(t) * 4 + i]);
  }
  if (int(t) < PARTS) {
    const device float* p = PART + size_t(r) * SLICES * PARTS + t;
    float a = p[0];
    for (int j = 1; j < SLICES; j++) a += p[j * PARTS];
    tot[t] = a;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) < MIX) mixes[t] = tot[t] * metal::precise::rsqrt(tot[MIX] / float(F) + EPS[0]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    constexpr int BASE_OFF = 2 * S;
    const float pre_scale = SCALE[0], post_scale = SCALE[1], comb_scale = SCALE[2];
    const float active = (lane < (uint)S) ? 1.0f : 0.0f;
    const uint llane = metal::min(lane, (uint)(S - 1));
    const float pre_z = mixes[llane] * pre_scale + BASEV[llane];
    const float post_z = mixes[S + llane] * post_scale + BASEV[S + llane];
    const float pre_v = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + HC_EPS;
    const float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
    if (lane < (uint)S) {
      pre_s[lane] = pre_v;
      POST_OUT[r * S + lane] = post_v;
    }
    const float4 m4 = float4(mixes[BASE_OFF + llane * S], mixes[BASE_OFF + llane * S + 1], mixes[BASE_OFF + llane * S + 2],
                             mixes[BASE_OFF + llane * S + 3]);
    float4 v = (m4 * comb_scale + *(const device float4*)(BASEV + BASE_OFF + llane * S)) * active;
    const float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
    const float4 e = metal::fast::exp(v - row_max) * active;
    float4 rr = e * (1.0f / (e.x + e.y + e.z + e.w + HC_EPS)) + HC_EPS * active;
    float4 col_inv = 1.0f / (hc_col4(rr) + HC_EPS);
    rr *= col_inv;
    for (int iter = 1; iter < TF_ITERS; ++iter) {
      rr *= (1.0f / (rr.x + rr.y + rr.z + rr.w + HC_EPS)) * active;
      col_inv = 1.0f / (hc_col4(rr) + HC_EPS);
      rr *= col_inv;
    }
    if (lane < (uint)S) *(device float4*)(COMB_OUT + r * S * S + lane * S) = rr;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float p0 = pre_s[0], p1 = pre_s[1], p2 = pre_s[2], p3 = pre_s[3];
  float xc[4];
  float acc = 0.0f;
  for (int i = 0; i < 4; ++i) {
    const float res = fma(p0, xv[i][0], fma(p1, xv[i][1], fma(p2, xv[i][2], p3 * xv[i][3])));
    xc[i] = float(bfloat(res));
    acc = hc_sq_acc(acc, xc[i]);
  }
  acc = simd_sum(acc);
  if (sg == 0) red[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) red[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float a = simd_sum(red[lane]);
    if (lane == 0) inv_s[0] = metal::precise::rsqrt(a / float(TF_D) + EPS[0]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < 4; ++i) {
    const int d = int(t) * 4 + i;
    NORMED[size_t(r) * TF_D + d] = NORMW[d] * bfloat(xc[i] * inv_s[0]);
  }
}
