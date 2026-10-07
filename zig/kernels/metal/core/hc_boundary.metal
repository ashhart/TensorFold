// A hyper-connection boundary in two launches: the expand with the mix's and the norm's partial sums, then the split.
#include <metal_stdlib>
using namespace metal;

// Defines: TF_D (stream width, a multiple of 1024), TF_ITERS (Sinkhorn iterations), TF_HC_EPS_INT (the split's eps in 1e-9).
constant constexpr int S = 4, F = S * TF_D, MIX = (2 + S) * S, SLICES = F / 1024, PARTS = MIX + 1;

#pragma clang fp contract(off)
inline float hc_sq_acc(float acc, float v) { return v * v + acc; }
inline float hc_add_nc(float a, float b) { return a + b; }
#pragma clang fp contract(on)

// Grid (SLICES, rows) of 256 threads: thread t of slice j takes columns 1024 j + 4 t.. of the streams, expands them (the family's per-element arithmetic) and sums their squares and 24 mix products into PART [rows, SLICES, 25].
template <bool EXPAND>
[[kernel]] void tf_hc_expand_mix(const device bfloat* XOLD [[buffer(0)]], const device bfloat* BRANCH [[buffer(1)]],
                             const device float* POST [[buffer(2)]], const device float* COMB [[buffer(3)]],
                             const device uint4* FNP [[buffer(4)]], device bfloat* XNEW [[buffer(5)]],
                             device float* PART [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                             uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[8][PARTS];
  const int j = int(tg.x), r = int(tg.y);
  const uint t = tpos.x;
  const device bfloat* xo = XOLD + size_t(r) * F;
  float vals[4];
  float ss = 0.0f;
  for (int tn = 0; tn < 4; tn++) {
    const int f = 1024 * j + 4 * int(t) + tn;
    float v;
    if (EXPAND) {
      const int s = f / TF_D, d = f - s * TF_D;
      const float y = POST[r * S + s] * float(BRANCH[size_t(r) * TF_D + d]);
      const device float* c = COMB + r * S * S;
      float mm = c[0 * S + s] * float(xo[0 * TF_D + d]);
      mm = fma(c[1 * S + s], float(xo[1 * TF_D + d]), mm);
      mm = fma(c[2 * S + s], float(xo[2 * TF_D + d]), mm);
      mm = fma(c[3 * S + s], float(xo[3 * TF_D + d]), mm);
      const bfloat nb = bfloat(hc_add_nc(y, mm));
      XNEW[size_t(r) * F + f] = nb;
      v = float(nb);
    } else {
      v = float(xo[f]);
    }
    vals[tn] = v;
    ss = hc_sq_acc(ss, v);
  }
  float acc[PARTS];
  for (int og = 0; og < MIX / 4; og++) { // the packed mix: [og][sg][lane][slice][tm][tn]
    const device uint4* w = FNP + (size_t(og * 8 + int(sg)) * 32 + lane) * SLICES * 2 + j * 2;
    float inter[4][4];
    for (int h = 0; h < 2; h++) {
      const uint4 v = w[h];
      const uint words[4] = {v.x, v.y, v.z, v.w};
      for (int k = 0; k < 4; k++) {
        const int e = h * 8 + k * 2;
        inter[e / 4][e % 4] = as_type<float>(words[k] << 16);
        inter[(e + 1) / 4][(e + 1) % 4] = as_type<float>(words[k] & 0xffff0000u);
      }
    }
    for (int tm = 0; tm < 4; tm++) {
      float a = 0.0f;
      for (int tn = 0; tn < 4; tn++) a += inter[tm][tn] * vals[tn];
      acc[og * 4 + tm] = a;
    }
  }
  acc[MIX] = ss;
  for (int i = 0; i < PARTS; i++) {
    const float v = simd_sum(acc[i]);
    if (lane == 0) red[sg][i] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) < PARTS) {
    float a = red[0][t];
    for (int k = 1; k < 8; k++) a += red[k][t];
    PART[(size_t(r) * SLICES + j) * PARTS + t] = a;
  }
}

template [[host_name("tf_hc_expand_mix")]] [[kernel]] decltype(tf_hc_expand_mix<true>) tf_hc_expand_mix<true>;
template [[host_name("tf_hc_first_mix")]] [[kernel]] decltype(tf_hc_expand_mix<false>) tf_hc_expand_mix<false>;

// One threadgroup of 1024 a row: the slices' sums added in order, the mixes scaled by the streams' inverse norm, then the family's split: pre, post and the Sinkhorn comb, the streams collapsed and normed.
[[kernel]] void tf_hc_split(const device bfloat* X [[buffer(0)]], const device float* PART [[buffer(1)]],
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
    float4 col_inv = 1.0f / (float4(simd_sum(rr.x), simd_sum(rr.y), simd_sum(rr.z), simd_sum(rr.w)) + HC_EPS);
    rr *= col_inv;
    for (int iter = 1; iter < TF_ITERS; ++iter) {
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
    const float res = fma(p0, float(xs[d]), fma(p1, float(xs[TF_D + d]), fma(p2, float(xs[2 * TF_D + d]), p3 * float(xs[3 * TF_D + d]))));
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
