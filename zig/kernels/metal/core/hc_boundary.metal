// A hyper-connection block boundary in one launch, six threadgroups of 256 a row: each expands the streams (the
// 1024-thread expand's per-thread sums, four virtual threads a thread), mixes its four outputs, and the last to finish
// splits the mixes (Sinkhorn) and norms the collapsed row (the 1024-thread kernel's sums again). Same bits as three launches.
// Defines: TF_D (width), TF_ITERS (Sinkhorn iterations), TF_HC_EPS_INT (eps in 1e-9), TF_U (mix unroll), TF_SQ_FMA.
#include <metal_stdlib>
using namespace metal;

constant constexpr int S = 4;
constant constexpr int D = TF_D;
constant constexpr int F = S * D;
constant constexpr int MIX = (2 + S) * S;
constant constexpr int GROUPS = MIX / 4; // mix threadgroups a row

#pragma clang fp contract(off)
template <int FMA>
inline float tf_sq_acc(float acc, float v) { return FMA ? fma(v, v, acc) : v * v + acc; }
inline float tf_add_nc(float a, float b) { return a + b; }
#pragma clang fp contract(on)

struct HcArgs {
  float eps;
};

// EXPAND: the pending branch written into the streams first (else the streams as they are).
template <int EXPAND>
inline void tf_hc_boundary(const device bfloat* XOLD, const device bfloat* BRANCH, const device float* POST,
                           const device float* COMB, const device bfloat* FNP, const constant float* SCALE,
                           const device float* BASEV, const device bfloat* NORMW, float eps, device bfloat* XNEW,
                           device atomic_uint* MIXES, device bfloat* NORMED, device float* POST_OUT, device float* COMB_OUT,
                           device atomic_uint* CNT, uint og, uint r, uint t, uint lane, uint sg, threadgroup float* red,
                           threadgroup float* inv_s, threadgroup float (*part)[4], threadgroup float* mix_s,
                           threadgroup float* pre_s, threadgroup uint* last_s) {
  device const bfloat* xo = XOLD + size_t(r) * F;
  device bfloat* xn = XNEW + size_t(r) * F;
  device const bfloat* xs = EXPAND ? (device const bfloat*)xn : xo;
  // expand: virtual thread T = t + 256 j does the 1024-thread kernel's thread T
  for (int j = 0; j < 4; j++) {
    const int T = int(t) + 256 * j;
    float ss = 0.0f;
    for (int k = 0; k < F / 4096; ++k) {
      for (int i = 0; i < 4; ++i) {
        const int f = T * 4 + 4096 * k + i;
        float v;
        if (EXPAND) {
          const int s = f / D, d = f - s * D;
          const float y = POST[r * S + s] * float(BRANCH[size_t(r) * D + d]);
          const device float* c = COMB + r * S * S;
          float mm = c[0 * S + s] * float(xo[0 * D + d]);
          mm = fma(c[1 * S + s], float(xo[1 * D + d]), mm);
          mm = fma(c[2 * S + s], float(xo[2 * D + d]), mm);
          mm = fma(c[3 * S + s], float(xo[3 * D + d]), mm);
          const bfloat nb = bfloat(tf_add_nc(y, mm));
          xn[f] = nb;
          v = float(nb);
        } else {
          v = float(xo[f]);
        }
        ss = tf_sq_acc<TF_SQ_FMA>(ss, v);
      }
    }
    ss = simd_sum(ss);
    if (lane == 0) red[8 * j + int(sg)] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  if (sg == 0) {
    const float a = simd_sum(red[lane]);
    if (lane == 0) inv_s[0] = metal::precise::rsqrt(a / float(F) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // mix: this threadgroup's four outputs (hc_mix_packed's sums)
  {
    constexpr int ITERS = F / 1024;
    const float inv = inv_s[0];
    const device uint4* w = (const device uint4*)(FNP + ((size_t(og) * 8 + sg) * 32 + lane) * ITERS * 16);
    const int bn0 = (32 * int(sg) + int(lane)) * 4;
    float res[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i0 = 0; i0 < ITERS; i0 += TF_U) {
      uint4 raw[TF_U][2];
      float xv[TF_U][4];
      for (int u = 0; u < TF_U; u++) {
        raw[u][0] = w[(i0 + u) * 2]; raw[u][1] = w[(i0 + u) * 2 + 1];
        for (int tn = 0; tn < 4; tn++) xv[u][tn] = float(xs[bn0 + 1024 * (i0 + u) + tn]);
      }
      for (int u = 0; u < TF_U; u++) {
        float vc[4];
        for (int tn = 0; tn < 4; tn++) vc[tn] = xv[u][tn] * inv;
        float inter[4][4];
        for (int h = 0; h < 2; h++) {
          const uint4 v = raw[u][h];
          const uint words[4] = {v.x, v.y, v.z, v.w};
          for (int q = 0; q < 4; q++) {
            const int e = h * 8 + q * 2;
            inter[e / 4][e % 4] = as_type<float>(words[q] << 16);
            inter[(e + 1) / 4][(e + 1) % 4] = as_type<float>(words[q] & 0xffff0000u);
          }
        }
        for (int tm = 0; tm < 4; tm++)
          for (int tn = 0; tn < 4; tn++) res[tm] += inter[tm][tn] * vc[tn];
      }
    }
    for (int tm = 0; tm < 4; tm++)
      for (ushort sn = 16; sn >= 1; sn >>= 1) res[tm] += simd_shuffle_down(res[tm], sn);
    if (lane == 0) for (int tm = 0; tm < 4; tm++) part[sg][tm] = res[tm];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < 4) {
      float a = part[0][lane];
      for (int k = 1; k < 8; k++) a += part[k][lane];
      atomic_store_explicit(&MIXES[r * MIX + og * 4 + lane], as_type<uint>(a), memory_order_relaxed);
    }
  }
  // the last of the row's threadgroups splits and norms
  threadgroup_barrier(mem_flags::mem_device);
  if (t == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    const uint prev = atomic_fetch_add_explicit(&CNT[r], 1u, memory_order_relaxed);
    last_s[0] = prev == uint(GROUPS - 1) ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (last_s[0] == 0) return;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  if (t == 0) atomic_store_explicit(&CNT[r], 0u, memory_order_relaxed);
  if (t < uint(MIX)) mix_s[t] = as_type<float>(atomic_load_explicit(&MIXES[r * MIX + t], memory_order_relaxed));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr float HC_EPS = TF_HC_EPS_INT * 1e-9;
  if (sg == 0) {
    constexpr int BASE_OFF = 2 * S;
    const float pre_scale = SCALE[0], post_scale = SCALE[1], comb_scale = SCALE[2];
    const float active = (lane < (uint)S) ? 1.0f : 0.0f;
    const uint llane = metal::min(lane, (uint)(S - 1));
    const float pre_z = mix_s[llane] * pre_scale + BASEV[llane];
    const float post_z = mix_s[S + llane] * post_scale + BASEV[S + llane];
    const float pre_v = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + HC_EPS;
    const float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
    if (lane < (uint)S) { pre_s[lane] = pre_v; POST_OUT[r * S + lane] = post_v; }
    const float4 mv = float4(mix_s[BASE_OFF + llane * S], mix_s[BASE_OFF + llane * S + 1], mix_s[BASE_OFF + llane * S + 2],
                             mix_s[BASE_OFF + llane * S + 3]);
    float4 v = (mv * comb_scale + *(const device float4*)(BASEV + BASE_OFF + llane * S)) * active;
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
  float xc[4][4];
  for (int j = 0; j < 4; j++) {
    const int T = int(t) + 256 * j;
    float acc = 0.0f;
    for (int i = 0; i < 4; ++i) {
      const int d = T * 4 + i;
      const float res = fma(p0, float(xs[d]), fma(p1, float(xs[D + d]),
                            fma(p2, float(xs[2 * D + d]), p3 * float(xs[3 * D + d]))));
      xc[j][i] = float(bfloat(res));
      acc = tf_sq_acc<TF_SQ_FMA>(acc, xc[j][i]);
    }
    acc = simd_sum(acc);
    if (lane == 0) red[8 * j + int(sg)] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float a = simd_sum(red[lane]);
    if (lane == 0) inv_s[0] = metal::precise::rsqrt(a / float(D) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int j = 0; j < 4; j++)
    for (int i = 0; i < 4; ++i) {
      const int d = (int(t) + 256 * j) * 4 + i;
      NORMED[size_t(r) * D + d] = NORMW[d] * bfloat(xc[j][i] * inv_s[0]);
    }
}

#define TF_HC_KERNEL(NAME, EXPAND)                                                                                       \
  [[kernel]] void NAME(const device bfloat* XOLD [[buffer(0)]], const device bfloat* BRANCH [[buffer(1)]],              \
                       const device float* POST [[buffer(2)]], const device float* COMB [[buffer(3)]],                  \
                       const device bfloat* FNP [[buffer(4)]], const constant float* SCALE [[buffer(5)]],               \
                       const device float* BASEV [[buffer(6)]], const device bfloat* NORMW [[buffer(7)]],               \
                       constant HcArgs& a [[buffer(8)]], device bfloat* XNEW [[buffer(9)]],                              \
                       device atomic_uint* MIXES [[buffer(10)]], device bfloat* NORMED [[buffer(11)]],                  \
                       device float* POST_OUT [[buffer(12)]], device float* COMB_OUT [[buffer(13)]],                    \
                       device atomic_uint* CNT [[buffer(14)]], uint2 tg [[threadgroup_position_in_grid]],                \
                       uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],           \
                       uint sg [[simdgroup_index_in_threadgroup]]) {                                                     \
    threadgroup float red[32], inv_s[1], part[8][4], mix_s[MIX], pre_s[S];                                               \
    threadgroup uint last_s[1];                                                                                           \
    tf_hc_boundary<EXPAND>(XOLD, BRANCH, POST, COMB, FNP, SCALE, BASEV, NORMW, a.eps, XNEW, MIXES, NORMED, POST_OUT,    \
                           COMB_OUT, CNT, tg.x, tg.y, tpos.x, lane, sg, red, inv_s, part, mix_s, pre_s, last_s);         \
  }

TF_HC_KERNEL(tf_hc_boundary_expand, 1)
TF_HC_KERNEL(tf_hc_boundary_first, 0)
