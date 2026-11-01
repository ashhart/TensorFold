// DeepSeek V4.1 decode kernels that fold a dependent glue launch into its neighbour, each with the glue's bits.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

#ifndef TF_CTL
#define TF_CTL 0
#endif

#define TF_O_NOWHERE 0xFFFFFFFFu
#define TF_O_TG 1024u
#define TF_O_MAX 5120u

// Control 1 of every kernel: a non-finite input is read as 0, so the output class changes.
inline float tf_o_in(float v) { return (TF_CTL == 1 && !isfinite(v)) ? 0.0f : v; }

// Control 3 of every kernel: a running total held in fp16.
inline float tf_o_run(float v) { return TF_CTL == 3 ? float(half(v)) : v; }

struct TfOPostArgs {
  uint dim, rows, topk, per_group, groups;
};

// Kimi's routed combine (group fmas, then the group tree) plus the shared expert into B, then hc_post into Y.
[[kernel, max_total_threads_per_threadgroup(64)]] void tf_dsv41_o_combpost(
    const device bfloat* OUT [[buffer(0)]], const device uint* WHERE [[buffer(1)]],
    const device float* WTS [[buffer(2)]], const device uint* IDS [[buffer(3)]], const device bfloat* SH [[buffer(4)]],
    device bfloat* B [[buffer(5)]], const device bfloat* RES [[buffer(6)]], const device float* POST [[buffer(7)]],
    const device float* COMB [[buffer(8)]], device bfloat* Y [[buffer(9)]], constant TfOPostArgs& a [[buffer(10)]],
    device float* YR [[buffer(11)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= a.dim || g.y >= a.rows) return;
  float part[8];
  for (uint i = 0; i < 8; ++i) part[i] = 0.0f;
  for (uint k = 0; k < a.topk; ++k) {
    const uint p = WHERE[g.y * a.topk + k];
    if (p == TF_O_NOWHERE) continue;
    const uint grp = IDS[g.y * a.topk + k] / a.per_group;
    part[grp] = tf_o_run(fma(WTS[g.y * a.topk + k], tf_o_in(float(OUT[ulong(p) * a.dim + g.x])), part[grp]));
  }
  for (uint w = a.groups; w > 1; w /= 2)
    for (uint i = 0; i < w / 2; ++i) part[i] = part[2 * i] + part[2 * i + 1];
  const ulong at = ulong(g.y) * a.dim + g.x;
  YR[at] = part[0];
  const bfloat bv = bfloat(part[0] + tf_o_in(float(SH[at])));
  B[at] = bv;
  const float b = float(bv);
  const device bfloat* res = RES + ulong(g.y) * 4u * a.dim + g.x;
  const device float* c = COMB + g.y * 16u;
  const float r3 = TF_CTL == 2 ? 0.0f : tf_o_in(float(res[3u * a.dim]));
  for (uint k = 0; k < 4; ++k) {
    const float mm = ((c[k] * tf_o_in(float(res[0])) + c[4 + k] * tf_o_in(float(res[a.dim]))) +
                      c[8 + k] * tf_o_in(float(res[2u * a.dim]))) + c[12 + k] * r3;
    Y[ulong(g.y) * 4u * a.dim + k * a.dim + g.x] = bfloat(POST[g.y * 4u + k] * b + mm);
  }
  if (TF_CTL == 4 && g.x == 0) Y[ulong(a.rows) * 4u * a.dim + g.y] = bfloat(1.0f);
}

// glue_tmp's helpers (tf_g_tree, the act_quant round trip), the same expressions.
inline float tf_o_tree(float v) {
  for (ushort m = 1; m < 32; m *= 2) v += simd_shuffle_xor(v, m);
  return v;
}

inline float tf_o_pow2(int e) { return as_type<float>(uint(e + 127) << 23); }

inline int tf_o_log2c(float v) {
  const uint b = as_type<uint>(v);
  return int((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) != 0u ? 1 : 0);
}

inline uint tf_o_e4m3(float a) {
  uint b = as_type<uint>(a);
  b += 0x7FFFFu + ((b >> 20) & 1u);
  return a < 0.015625f ? uint(rint(a * 512.0f)) : (((b >> 23) - 120u) << 3) | ((b >> 20) & 7u);
}

inline float tf_o_e4m3v(uint c) { return float(as_type<half>(ushort((c & 0x7Fu) << 7))) * 256.0f; }

// act_quant's round trip of the value a lane holds, its 32-value block being the simdgroup's 32 lanes.
inline bfloat tf_o_actq(float v) {
  const bool bad = simd_any(!isfinite(v));
  const float amax = simd_max(fabs(v));
  const int e = tf_o_log2c(max(amax, 1e-4f) * (1.0f / 448.0f));
  const float q = tf_o_e4m3v(tf_o_e4m3(min(fabs(v) * tf_o_pow2(-e), 448.0f))) * tf_o_pow2(e);
  return bfloat(bad ? NAN : (signbit(v) ? -q : q));
}

struct TfONormArgs {
  uint dim, rows;
  float eps;
};

// hc_pre (H), rmsnorm (HN) and act_quant (HQ) of a row in a group of 1024; POST first makes X1 = hc_post(BR, X).

// The sum of squares runs on threads 0..255 as glue_tmp's rmsnorm: every 256th value, the butterfly, eight partials.
template <bool POST>
[[kernel, max_total_threads_per_threadgroup(TF_O_TG)]] void tf_dsv41_o_prenorm(
    const device bfloat* X [[buffer(0)]], const device float* PRE [[buffer(1)]], const device float* Wt [[buffer(2)]],
    device bfloat* H [[buffer(3)]], device bfloat* HN [[buffer(4)]], device bfloat* HQ [[buffer(5)]],
    constant TfONormArgs& a [[buffer(6)]], const device bfloat* BR [[buffer(7)]], const device float* PO [[buffer(8)]],
    const device float* CB [[buffer(9)]], device bfloat* X1 [[buffer(10)]], uint r [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float hs[TF_O_MAX];
  threadgroup float part[8];
  if (r >= a.rows) return;
  const device bfloat* x = X + ulong(r) * 4u * a.dim;
  const device float* p = PRE + r * 4u;
  const float p0 = p[0], p1 = p[TF_CTL == 2 ? 2 : 1], p2 = p[TF_CTL == 2 ? 1 : 2], p3 = p[3];
  for (uint i = t; i < a.dim; i += TF_O_TG) {
    float v[4];
    for (uint k = 0; k < 4; ++k) v[k] = tf_o_in(float(x[k * a.dim + i]));
    if (POST) {
      const float b = tf_o_in(float(BR[ulong(r) * a.dim + i]));
      const device float* c = CB + r * 16u;
      float o[4];
      for (uint k = 0; k < 4; ++k) {
        const float mm = ((c[k] * v[0] + c[4 + k] * v[1]) + c[8 + k] * v[2]) + c[12 + k] * v[3];
        const bfloat ok = bfloat(PO[r * 4u + k] * b + mm);
        X1[ulong(r) * 4u * a.dim + k * a.dim + i] = ok;
        o[k] = float(ok);
      }
      for (uint k = 0; k < 4; ++k) v[k] = o[k];
    }
    const float y = float(bfloat(((p0 * v[0] + p1 * v[1]) + p2 * v[2]) + p3 * v[3]));
    hs[i] = y;
    H[ulong(r) * a.dim + i] = bfloat(y);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 256u) {
    float s = 0.0f;
    for (uint i = t; i < a.dim; i += 256u) s = tf_o_run(s + hs[i] * hs[i]);
    s = tf_o_tree(s);
    if (lane == 0) part[sg] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = part[0];
  for (uint k = 1; k < 8u; ++k) tot = tf_o_run(tot + part[k]);
  const float inv = precise::rsqrt(tot / float(a.dim) + a.eps);
  for (uint i = t; i - lane < a.dim; i += TF_O_TG) {
    const float hv = i < a.dim ? hs[i] : 0.0f;
    const bfloat hn = bfloat(Wt[min(i, a.dim - 1u)] * (hv * inv));
    const bfloat q = tf_o_actq(float(hn));
    if (i < a.dim) HN[ulong(r) * a.dim + i] = hn, HQ[ulong(r) * a.dim + i] = q;
  }
  if (TF_CTL == 4 && t == 0) HQ[ulong(a.rows) * a.dim + r] = bfloat(1.0f);
}

template [[host_name("tf_dsv41_o_prenorm")]] [[kernel]] decltype(tf_dsv41_o_prenorm<false>) tf_dsv41_o_prenorm<false>;
template [[host_name("tf_dsv41_o_postnorm")]] [[kernel]] decltype(tf_dsv41_o_prenorm<true>) tf_dsv41_o_prenorm<true>;
