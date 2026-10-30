// DeepSeek V4.1 decode kernels that split or fuse the phase-1 glue launches, each keeping the glue's bits exactly.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

#define TF_F_MIX 24u
#define TF_F_SG 8u

// A float's adjacent-lane butterfly: the same tree whatever the launch (glue_tmp's tf_g_tree).
inline float tf_f_tree(float v) {
  for (ushort m = 1; m < 32; m *= 2) v += simd_shuffle_xor(v, m);
  return v;
}

struct TfFHcArgs {
  uint dim, rows, iters;
  float eps, hc_eps, alpha;
};

// The hc mix's sums by simdgroup: group (m, sg, row) is simdgroup sg of matvec m's 256-thread sum (m 24: squares).
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_f_hcmix_part(
    const device bfloat* X [[buffer(0)]], const device float* FN [[buffer(1)]], device float* P [[buffer(2)]],
    constant TfFHcArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint m = tg.x, sg = tg.y, r = tg.z;
  const device bfloat* x = X + ulong(r) * a.dim;
  const uint step = 32u * TF_F_SG;
  float v = 0.0f;
  uint i = sg * 32u + lane;
  // eight steps' loads issued before their chain: the order of the chain is unchanged
  if (m < TF_F_MIX) {
    const device float* fn = FN + ulong(m) * a.dim;
    for (; i + 7u * step < a.dim; i += 8u * step) {
      float f[8], xv[8];
      for (uint u = 0; u < 8u; ++u) f[u] = fn[i + u * step], xv[u] = float(x[i + u * step]);
      for (uint u = 0; u < 8u; ++u) v = fma(f[u], xv[u], v);
    }
    for (; i < a.dim; i += step) v = fma(fn[i], float(x[i]), v);
  } else {
    for (; i + 7u * step < a.dim; i += 8u * step) {
      float xv[8];
      for (uint u = 0; u < 8u; ++u) xv[u] = float(x[i + u * step]);
      for (uint u = 0; u < 8u; ++u) v = v + xv[u] * xv[u];
    }
    for (; i < a.dim; i += step) v = v + float(x[i]) * float(x[i]);
  }
  v = tf_f_tree(v);
  if (lane == 0) P[(ulong(r) * (TF_F_MIX + 1u) + m) * TF_F_SG + sg] = v;
}

// glue_tmp's hcmix tail, a simdgroup a row: lane m holds mix m (lane 8 + 4j + k comb[j][k]), sums in the same order.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_f_hcmix_fin(
    const device float* P [[buffer(0)]], const device float* SCALE [[buffer(1)]], const device float* BASE [[buffer(2)]],
    device float* PRE [[buffer(3)]], device float* POST [[buffer(4)]], device float* COMB [[buffer(5)]],
    constant TfFHcArgs& a [[buffer(6)]], uint r [[threadgroup_position_in_grid]], ushort L [[thread_index_in_simdgroup]]) {
  if (r >= a.rows) return;
  const device float* p = P + ulong(r) * (TF_F_MIX + 1u) * TF_F_SG;
  float tot = p[TF_F_MIX * TF_F_SG];
  for (uint k = 1; k < TF_F_SG; ++k) tot += p[TF_F_MIX * TF_F_SG + k];
  const float inv = precise::rsqrt(tot / float(a.dim) + a.eps);
  const uint m = min(uint(L), TF_F_MIX - 1u);
  float dot = p[m * TF_F_SG];
  for (uint k = 1; k < TF_F_SG; ++k) dot += p[m * TF_F_SG + k];
  const float mix = dot * inv;
  if (L < 4) PRE[r * 4 + L] = precise::divide(1.0f, 1.0f + precise::exp(-(mix * SCALE[0] + BASE[L]))) + a.hc_eps;
  else if (L < 8) POST[r * 4 + L - 4] = a.alpha * precise::divide(1.0f, 1.0f + precise::exp(-(mix * SCALE[1] + BASE[L])));
  const ushort q = (L >= 8 && L < 24) ? L - 8 : 0;
  const ushort rj = 8 + (q & ~ushort(3)), ck = 8 + (q & 3);
  float c = mix * SCALE[2] + BASE[8 + q];
  {
    const float c0 = simd_shuffle(c, rj), c1 = simd_shuffle(c, ushort(rj + 1));
    const float c2 = simd_shuffle(c, ushort(rj + 2)), c3 = simd_shuffle(c, ushort(rj + 3));
    const float mx = max(max(c0, c1), max(c2, c3));
    const float ex = precise::exp(c - mx);
    float s = 0.0f;
    for (ushort k = 0; k < 4; ++k) s += simd_shuffle(ex, ushort(rj + k));
    c = precise::divide(ex, s) + a.hc_eps;
  }
  for (uint it = 0; it < a.iters; ++it) {
    if (it > 0) {
      const float s = ((simd_shuffle(c, rj) + simd_shuffle(c, ushort(rj + 1))) + simd_shuffle(c, ushort(rj + 2))) + simd_shuffle(c, ushort(rj + 3));
      c = precise::divide(c, s + a.hc_eps);
    }
    const float s = ((simd_shuffle(c, ck) + simd_shuffle(c, ushort(ck + 4))) + simd_shuffle(c, ushort(ck + 8))) + simd_shuffle(c, ushort(ck + 12));
    c = precise::divide(c, s + a.hc_eps);
  }
  if (L >= 8 && L < 24) COMB[r * 16 + q] = c;
}

// glue_tmp's act_quant helpers, the same expressions.
inline float tf_f_pow2(int e) { return as_type<float>(uint(e + 127) << 23); }

inline int tf_f_log2c(float v) {
  const uint b = as_type<uint>(v);
  return int((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) != 0u ? 1 : 0);
}

inline uint tf_f_e4m3(float a) {
  uint b = as_type<uint>(a);
  b += 0x7FFFFu + ((b >> 20) & 1u);
  return a < 0.015625f ? uint(rint(a * 512.0f)) : (((b >> 23) - 120u) << 3) | ((b >> 20) & 7u);
}

inline float tf_f_e4m3v(uint c) { return float(as_type<half>(ushort((c & 0x7Fu) << 7))) * 256.0f; }

// act_quant's round trip of the value a lane holds, its 32-value block being the simdgroup's 32 lanes.
inline bfloat tf_f_actq(float v) {
  const bool bad = simd_any(!isfinite(v));
  const float amax = simd_max(fabs(v));
  const int e = tf_f_log2c(max(amax, 1e-4f) * (1.0f / 448.0f));
  const float q = tf_f_e4m3v(tf_f_e4m3(min(fabs(v) * tf_f_pow2(-e), 448.0f))) * tf_f_pow2(e);
  return bfloat(bad ? NAN : (signbit(v) ? -q : q));
}

// A threadgroup of 256's sum: each simdgroup's tree, then the eight in order (glue_tmp's tf_g_sum).
inline float tf_f_sum(float v, threadgroup float* part, uint lane, uint sg) {
  v = tf_f_tree(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = part[0];
  for (uint k = 1; k < 8u; ++k) t += part[k];
  return t;
}

struct TfFNormArgs {
  uint dim, rows;
  float eps;
};

// glue_tmp's (hcpre with COLLAPSE,) rmsnorm and actq of a row in a group of 256; dim a multiple of 32, <= 256 NV.
template <bool COLLAPSE, uint NV>
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_f_norm(
    const device bfloat* X [[buffer(0)]], const device float* PRE [[buffer(1)]], const device float* Wt [[buffer(2)]],
    device bfloat* H [[buffer(3)]], device bfloat* HN [[buffer(4)]], device bfloat* HQ [[buffer(5)]],
    constant TfFNormArgs& a [[buffer(6)]], uint r [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[8];
  // every load before the first store, so none waits behind a store it might alias
  float y[NV], wv[NV];
  if (COLLAPSE) {
    const device bfloat* x = X + ulong(r) * 4u * a.dim;
    const device float* p = PRE + r * 4u;
    const float p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
    for (uint j = 0; j < NV; ++j) {
      const uint i = min(t + 256u * j, a.dim - 1u); // in bounds, so the loads issue unguarded
      y[j] = float(bfloat(((p0 * float(x[i]) + p1 * float(x[a.dim + i])) + p2 * float(x[2u * a.dim + i])) + p3 * float(x[3u * a.dim + i])));
    }
  } else {
    const device bfloat* x = X + ulong(r) * a.dim;
    for (uint j = 0; j < NV; ++j) y[j] = float(x[min(t + 256u * j, a.dim - 1u)]);
  }
  for (uint j = 0; j < NV; ++j) wv[j] = Wt[min(t + 256u * j, a.dim - 1u)];
  if (COLLAPSE) {
    for (uint j = 0; j < NV; ++j) {
      const uint i = t + 256u * j;
      if (i < a.dim) H[ulong(r) * a.dim + i] = bfloat(y[j]);
    }
  }
  float s = 0.0f;
  for (uint j = 0; j < NV; ++j)
    if (t + 256u * j < a.dim) s = s + y[j] * y[j];
  const float tot = tf_f_sum(s, part, lane, sg);
  const float inv = precise::rsqrt(tot / float(a.dim) + a.eps);
  for (uint j = 0; j < NV; ++j) {
    const uint i = t + 256u * j;
    if (i - lane < a.dim) {
      const bfloat hn = bfloat(wv[j] * (y[j] * inv));
      HN[ulong(r) * a.dim + i] = hn;
      HQ[ulong(r) * a.dim + i] = tf_f_actq(float(hn));
    }
  }
}

#define TF_F_NORM(NV) \
  template [[host_name("tf_dsv41_f_prenorm_v" #NV)]] [[kernel]] decltype(tf_dsv41_f_norm<true, NV>) tf_dsv41_f_norm<true, NV>; \
  template [[host_name("tf_dsv41_f_normq_v" #NV)]] [[kernel]] decltype(tf_dsv41_f_norm<false, NV>) tf_dsv41_f_norm<false, NV>;
TF_F_NORM(1)
TF_F_NORM(2)
TF_F_NORM(5)
TF_F_NORM(8)
TF_F_NORM(20)
TF_F_NORM(32)

struct TfFKvArgs {
  uint dim, rows, reps;
  float eps;
};

// glue_tmp's rmsnorm, rope, actq and scatter of a window key row (dim 512) in a group of 256, into D[reps r + k].
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_f_kv(
    const device bfloat* X [[buffer(0)]], const device float* Wt [[buffer(1)]], device bfloat* KVN [[buffer(2)]],
    device bfloat* KVR [[buffer(3)]], device bfloat* KVQ [[buffer(4)]], device bfloat* WIN [[buffer(5)]],
    const device int* P [[buffer(6)]], const device float* CS [[buffer(7)]], const device int* D [[buffer(8)]],
    constant TfFKvArgs& a [[buffer(9)]], uint r [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[8];
  const int pos = P[r];
  if (pos < 0) return;
  const device bfloat* x = X + ulong(r) * a.dim;
  const float y0 = float(x[t]), y1 = float(x[t + 256u]);
  float s = 0.0f;
  s = s + y0 * y0;
  s = s + y1 * y1;
  const float tot = tf_f_sum(s, part, lane, sg);
  const float inv = precise::rsqrt(tot / float(a.dim) + a.eps);
  for (uint j = 0; j < 2u; ++j) {
    const uint i = t + 256u * j;
    const float kn = float(bfloat(Wt[i] * ((j == 0 ? y0 : y1) * inv)));
    KVN[ulong(r) * a.dim + i] = bfloat(kn);
    const float other = simd_shuffle_xor(kn, ushort(1));
    float kr = kn;
    if (i >= a.dim - 64u) {
      const uint pr = (i - (a.dim - 64u)) / 2u;
      const float c = CS[ulong(pos) * 64u + pr], sn = CS[ulong(pos) * 64u + 32u + pr];
      kr = (i & 1u) == 0u ? float(bfloat(kn * c - other * sn)) : float(bfloat(other * sn + kn * c));
    }
    KVR[ulong(r) * a.dim + i] = bfloat(kr);
    const bfloat q = tf_f_actq(kr);
    KVQ[ulong(r) * a.dim + i] = q;
    for (uint k = 0; k < a.reps; ++k) {
      const int d = D[a.reps * r + k];
      if (d >= 0) WIN[ulong(d) * a.dim + i] = q;
    }
  }
}

inline float tf_f_clip(float x, float lo, float hi) { return x != x ? x : clamp(x, lo, hi); }

struct TfFSwiArgs {
  uint n;
  float limit;
};

// The shared expert's clamped SwiGLU (glue_tmp's swiglu) and its act_quant, a block of 32 a simdgroup; n % 32 == 0.
kernel void tf_dsv41_f_swiglu(const device bfloat* G [[buffer(0)]], const device bfloat* U [[buffer(1)]],
                              device bfloat* Y [[buffer(2)]], device bfloat* YQ [[buffer(3)]],
                              constant TfFSwiArgs& a [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i - i % 32u >= a.n) return;
  const float g = tf_f_clip(float(G[i]), -INFINITY, a.limit), u = tf_f_clip(float(U[i]), -a.limit, a.limit);
  const bfloat y = bfloat(precise::divide(g, 1.0f + precise::exp(-g)) * u);
  Y[i] = y;
  YQ[i] = tf_f_actq(float(y));
}

struct TfFPostArgs {
  uint dim, rows;
};

// The MoE join (routed fp32 plus shared bf16, one rounding, into B) and hc_post of it (glue_tmp's join and hcpost).
kernel void tf_dsv41_f_joinpost(const device float* RT [[buffer(0)]], const device bfloat* SH [[buffer(1)]],
                                device bfloat* B [[buffer(2)]], const device bfloat* RES [[buffer(3)]],
                                const device float* POST [[buffer(4)]], const device float* COMB [[buffer(5)]],
                                device bfloat* Y [[buffer(6)]], constant TfFPostArgs& a [[buffer(7)]],
                                uint2 g [[thread_position_in_grid]]) {
  if (g.x >= a.dim || g.y >= a.rows) return;
  const ulong at = ulong(g.y) * a.dim + g.x;
  const bfloat bv = bfloat(RT[at] + float(SH[at]));
  B[at] = bv;
  const float b = float(bv);
  const device bfloat* res = RES + ulong(g.y) * 4u * a.dim + g.x;
  const device float* c = COMB + g.y * 16u;
  for (uint k = 0; k < 4; ++k) {
    const float mm = ((c[k] * float(res[0]) + c[4 + k] * float(res[a.dim])) + c[8 + k] * float(res[2u * a.dim])) +
                     c[12 + k] * float(res[3u * a.dim]);
    Y[ulong(g.y) * 4u * a.dim + k * a.dim + g.x] = bfloat(POST[g.y * 4u + k] * b + mm);
  }
}

// hcpre into H and rmsnorm's sum by simdgroup: group (sg, row) is simdgroup sg of the 256-thread row; dim <= 256 NV.
template <uint NV>
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_f_pre_a(
    const device bfloat* X [[buffer(0)]], const device float* PRE [[buffer(1)]], device bfloat* H [[buffer(2)]],
    device float* P [[buffer(3)]], constant TfFNormArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint sg = tg.x, r = tg.y, t = sg * 32u + lane;
  const device bfloat* x = X + ulong(r) * 4u * a.dim;
  const device float* p = PRE + r * 4u;
  const float p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
  float y[NV];
  for (uint j = 0; j < NV; ++j) {
    const uint i = min(t + 256u * j, a.dim - 1u); // in bounds, so the loads issue unguarded
    y[j] = float(bfloat(((p0 * float(x[i]) + p1 * float(x[a.dim + i])) + p2 * float(x[2u * a.dim + i])) + p3 * float(x[3u * a.dim + i])));
  }
  float s = 0.0f;
  for (uint j = 0; j < NV; ++j) {
    const uint i = t + 256u * j;
    if (i < a.dim) {
      H[ulong(r) * a.dim + i] = bfloat(y[j]);
      s = s + y[j] * y[j];
    }
  }
  s = tf_f_tree(s);
  if (lane == 0) P[r * 8u + sg] = s;
}

// rmsnorm's output from H and the eight partials (in order), and actq: a thread a value, a simdgroup a block.
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_f_pre_b(
    const device bfloat* H [[buffer(0)]], const device float* P [[buffer(1)]], const device float* Wt [[buffer(2)]],
    device bfloat* HN [[buffer(3)]], device bfloat* HQ [[buffer(4)]], constant TfFNormArgs& a [[buffer(5)]],
    uint2 g [[thread_position_in_grid]]) {
  const uint i = g.x, r = g.y;
  if (i - i % 32u >= a.dim) return;
  const device float* p = P + r * 8u;
  float tot = p[0];
  for (uint k = 1; k < 8u; ++k) tot += p[k];
  const float inv = precise::rsqrt(tot / float(a.dim) + a.eps);
  const bfloat hn = bfloat(Wt[i] * (float(H[ulong(r) * a.dim + i]) * inv));
  HN[ulong(r) * a.dim + i] = hn;
  HQ[ulong(r) * a.dim + i] = tf_f_actq(float(hn));
}

template [[host_name("tf_dsv41_f_pre_a_v20")]] [[kernel]] decltype(tf_dsv41_f_pre_a<20>) tf_dsv41_f_pre_a<20>;
template [[host_name("tf_dsv41_f_pre_a_v32")]] [[kernel]] decltype(tf_dsv41_f_pre_a<32>) tf_dsv41_f_pre_a<32>;

struct TfFQuantArgs {
  uint k, nb, na, sa, sb, ta, tb;
};

// glue_tmp's actq a thread a value (its strides), a block of 32 on one simdgroup's lanes; k a multiple of 32.
kernel void tf_dsv41_f_actq(const device bfloat* X [[buffer(0)]], device bfloat* Y [[buffer(1)]],
                            constant TfFQuantArgs& a [[buffer(2)]], uint3 g [[thread_position_in_grid]]) {
  if (g.x - g.x % 32u >= a.k || g.y >= a.nb || g.z >= a.na) return;
  const float v = float(X[ulong(g.z) * a.sa + ulong(g.y) * a.sb + g.x]);
  Y[ulong(g.z) * a.ta + ulong(g.y) * a.tb + g.x] = tf_f_actq(v);
}
