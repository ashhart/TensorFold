// DeepSeek-V4.1's hyper-connection boundary on Metal: the mixes, the collapse with the LAGGED pre mix, the expand.

// model.py hc_mixes 951-957, hc_pre 959-963, hc_post 964-969; kernel.py hc_split_sinkhorn_kernel 407-462.

// hc_mult 4, hc_sinkhorn_iters, hc_eps and hc_post_alpha arrive as arguments; hc_fn is fp32 and exp is precise.

// A sublayer's coefficients are used by the NEXT one (model.py:984-997), so `pre` arrives from outside unfused.

// O1 the norm's squares and each of the 24 matvecs: a lane chains its own i = t, t+256, t+512, ... upward.

// The lane total folds through the 5-level adjacent-lane butterfly (xor 1, 2, 4, 8, 16).

// The eight simdgroup partials then add in order sg 0, 1, ..., 7, on thread 0 only, then the fold's last adds.

// O2 a softmax row's exp terms add in index order k = 0, 1, 2, 3, and the divide-back is k = 0, 1, 2, 3.

// O3 Sinkhorn (kernel.py:438-445): the row softmax, then iters-1 passes of (row sums, then column sums).

// Every inner sum runs in index order, with the +eps inside the divisor exactly where the reference puts it.

// O4 hc_pre: the four streams add in stream order 0, 1, 2, 3, one thread an output, then one round to bf16.

// O5 hc_post: post[k]*branch, then comb[0][k]*r0 + comb[1][k]*r1 + comb[2][k]*r2 + comb[3][k]*r3, then bf16.

// Controls (TF_CTL, all three kernels): 1 a non-finite input reads as 0; 2 a part of the formula is lost.

// 3 a running fp32 sum flushes to fp16; 4 one element is written past the output, into its guard.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

#ifndef TF_CTL
#define TF_CTL 0
#endif

#define TF_HC_TG 256u
#define TF_HC_MULT 4u
#define TF_HC_MIX 24u // (2 + hc_mult) * hc_mult: pre 0-3, post 4-7, comb 8-23 (kernel.py:430-435)
#define TF_HC_SG (TF_HC_TG / 32u)

// O1's fold: the same 5-level butterfly whatever the launch.
inline float tf_hc_tree(float v) {
  for (ushort m = 1; m < 32; m *= 2) v += simd_shuffle_xor(v, m);
  return v;
}

// Control 1 turns a non-finite input into 0, so the output's class changes. Over bf16 and fp32 inputs alike.
template <typename T>
inline float tf_hc_nv(const device T* p, ulong i) {
#if TF_CTL == 1
  const float v = float(p[i]);
  return isfinite(v) ? v : 0.0f;
#else
  return float(p[i]);
#endif
}

struct TfHcMixArgs {
  uint dim, rows, iters;
  float eps, hc_eps, alpha;
};

// hc_mixes and hc_split_sinkhorn of one row: the norm and the 24 matvecs share one scan, each its own O1 chain.
[[kernel, max_total_threads_per_threadgroup(TF_HC_TG)]] void tf_dsv41_hc_mix(
    const device bfloat* X [[buffer(0)]], const device float* FN [[buffer(1)]], const device float* SCALE [[buffer(2)]],
    const device float* BASE [[buffer(3)]], device float* PRE [[buffer(4)]], device float* POST [[buffer(5)]],
    device float* COMB [[buffer(6)]], constant TfHcMixArgs& a [[buffer(7)]], uint r [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float ss_part[TF_HC_SG];
  threadgroup float part[TF_HC_SG * TF_HC_MIX];
  const device bfloat* x = X + ulong(r) * a.dim;
  float ss = 0.0f;
  float acc[TF_HC_MIX];
  for (uint m = 0; m < TF_HC_MIX; ++m) acc[m] = 0.0f;
  for (uint i = t; i < a.dim; i += TF_HC_TG) {
    const float v = tf_hc_nv(x, i);
    ss = ss + v * v;
    for (uint m = 0; m < TF_HC_MIX; ++m) {
      acc[m] = fma(tf_hc_nv(FN, ulong(m) * a.dim + i), v, acc[m]); // fma, as the glue's hcmix: one rounding
#if TF_CTL == 3 // control 3: the lane's running fp32 total flushes to fp16
      acc[m] = float(half(acc[m]));
#endif
    }
  }
  const float sst = tf_hc_tree(ss); // the norm's own O1 chain, kept apart from the matvecs'
  float tots[TF_HC_MIX];
  for (uint m = 0; m < TF_HC_MIX; ++m) tots[m] = tf_hc_tree(acc[m]); // O1: the matvecs fold in m order
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) {
    ss_part[sg] = sst;
    for (uint m = 0; m < TF_HC_MIX; ++m) part[sg * TF_HC_MIX + m] = tots[m];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t != 0) return;
  float tot = ss_part[0];
  for (uint s = 1; s < TF_HC_SG; ++s) tot += ss_part[s]; // O1's tail: the eight partials, in order
  const float inv = precise::rsqrt(tot / float(a.dim) + a.eps);
  float mix[TF_HC_MIX];
  for (uint m = 0; m < TF_HC_MIX; ++m) {
    float dot = part[m];
    for (uint s = 1; s < TF_HC_SG; ++s) dot += part[s * TF_HC_MIX + m];
    mix[m] = dot * inv;
  }
  // kernel.py:430-435. Control 2 takes the post mix from the pre slot, so the coefficients are the wrong ones.
  float c[TF_HC_MULT][TF_HC_MULT];
  for (uint j = 0; j < TF_HC_MULT; ++j) {
    PRE[r * TF_HC_MULT + j] = precise::divide(1.0f, 1.0f + precise::exp(-(mix[j] * SCALE[0] + BASE[j]))) + a.hc_eps;
#if TF_CTL == 2
    const uint wj = j;
#else
    const uint wj = 4u + j;
#endif
    POST[r * TF_HC_MULT + j] = a.alpha * precise::divide(1.0f, 1.0f + precise::exp(-(mix[wj] * SCALE[1] + BASE[wj])));
    for (uint k = 0; k < TF_HC_MULT; ++k) c[j][k] = mix[8u + 4u * j + k] * SCALE[2] + BASE[8u + 4u * j + k];
  }
  // O2, then O3. Control 2 also stops before the Sinkhorn has run at all.
  for (uint j = 0; j < TF_HC_MULT; ++j) {
    const float mx = max(max(c[j][0], c[j][1]), max(c[j][2], c[j][3]));
    float s = 0.0f;
    for (uint k = 0; k < TF_HC_MULT; ++k) c[j][k] = precise::exp(c[j][k] - mx), s += c[j][k];
    for (uint k = 0; k < TF_HC_MULT; ++k) c[j][k] = precise::divide(c[j][k], s) + a.hc_eps;
  }
#if TF_CTL == 2
  const uint iters = 0u;
#else
  const uint iters = a.iters;
#endif
  for (uint it = 0; it < iters; ++it) {
    if (it > 0) {
      for (uint j = 0; j < TF_HC_MULT; ++j) {
        const float s = ((c[j][0] + c[j][1]) + c[j][2]) + c[j][3];
        for (uint k = 0; k < TF_HC_MULT; ++k) c[j][k] = precise::divide(c[j][k], s + a.hc_eps);
      }
    }
    for (uint k = 0; k < TF_HC_MULT; ++k) {
      const float s = ((c[0][k] + c[1][k]) + c[2][k]) + c[3][k];
      for (uint j = 0; j < TF_HC_MULT; ++j) c[j][k] = precise::divide(c[j][k], s + a.hc_eps);
    }
  }
  for (uint j = 0; j < TF_HC_MULT; ++j)
    for (uint k = 0; k < TF_HC_MULT; ++k) COMB[r * 16u + 4u * j + k] = c[j][k];
#if TF_CTL == 4 // control 4: one float past the last comb, into its guard
  if (r == 0u && t == 0u) COMB[a.rows * 16u] = 1.0f;
#endif
}

struct TfHcArgs2 {
  uint dim, rows;
};

// hc_pre (model.py:959-963): O4, `pre` being the mix the PREVIOUS sublayer made; no cross-thread sum here.
[[kernel, max_total_threads_per_threadgroup(TF_HC_TG)]] void tf_dsv41_hc_pre(const device bfloat* X [[buffer(0)]],
                                                                                 const device float* PRE [[buffer(1)]],
                                                                                 device bfloat* Y [[buffer(2)]],
                                                                                 constant TfHcArgs2& a [[buffer(3)]],
                                                                                 uint2 g [[thread_position_in_grid]]) {
  if (g.x >= a.dim || g.y >= a.rows) return;
  const device bfloat* x = X + ulong(g.y) * TF_HC_MULT * a.dim + g.x;
  const device float* p = PRE + g.y * TF_HC_MULT;
#if TF_CTL == 2 // control 2: the weights rotated by one stream
  const float w0 = p[1], w1 = p[2], w2 = p[3], w3 = p[0];
#else
  const float w0 = p[0], w1 = p[1], w2 = p[2], w3 = p[3];
#endif
  float y = w0 * tf_hc_nv(x, 0) + w1 * tf_hc_nv(x, a.dim);
#if TF_CTL == 3 // control 3: the running total, holding the small term, flushes to fp16 and drops it
  y = float(half(y));
#endif
  y = (y + w2 * tf_hc_nv(x, 2u * a.dim)) + w3 * tf_hc_nv(x, 3u * a.dim);
  Y[ulong(g.y) * a.dim + g.x] = bfloat(y);
#if TF_CTL == 4 // control 4: the last thread also writes one bf16 past the payload, into the buffer's guard
  if (g.y == a.rows - 1u && g.x == a.dim - 1u) Y[ulong(a.rows) * a.dim] = bfloat(1.0f);
#endif
}

// hc_post (model.py:964-969): O5, the branch term first, then the residual streams through comb in j order.
[[kernel, max_total_threads_per_threadgroup(TF_HC_TG)]] void tf_dsv41_hc_post(const device bfloat* B [[buffer(0)]],
                                                                                  const device bfloat* RES [[buffer(1)]],
                                                                                  const device float* POST [[buffer(2)]],
                                                                                  const device float* COMB [[buffer(3)]],
                                                                                  device bfloat* Y [[buffer(4)]],
                                                                                  constant TfHcArgs2& a [[buffer(5)]],
                                                                                  uint2 g [[thread_position_in_grid]]) {
  if (g.x >= a.dim || g.y >= a.rows) return;
  const device bfloat* res = RES + ulong(g.y) * TF_HC_MULT * a.dim + g.x;
  const device float* c = COMB + g.y * 16u;
  const float b = tf_hc_nv(B, ulong(g.y) * a.dim + g.x);
  const float r0 = tf_hc_nv(res, 0), r1 = tf_hc_nv(res, a.dim), r2 = tf_hc_nv(res, 2u * a.dim),
              r3 = tf_hc_nv(res, 3u * a.dim);
  for (uint k = 0; k < TF_HC_MULT; ++k) {
    float mm = c[k] * r0 + c[4u + k] * r1;
#if TF_CTL == 3 // control 3: the running total, holding the small term, flushes to fp16 and drops it
    mm = float(half(mm));
#endif
    mm = (mm + c[8u + k] * r2) + c[12u + k] * r3;
#if TF_CTL == 2 // control 2: the branch added without its post weight
    const float v = b + mm;
#else
    const float v = POST[g.y * TF_HC_MULT + k] * b + mm;
#endif
    Y[ulong(g.y) * TF_HC_MULT * a.dim + k * a.dim + g.x] = bfloat(v);
#if TF_CTL == 4 // control 4: the last thread also writes past the payload, into the buffer's guard
    if (g.y == a.rows - 1u && k == TF_HC_MULT - 1u && g.x == a.dim - 1u) Y[ulong(a.rows) * TF_HC_MULT * a.dim] = bfloat(1.0f);
#endif
  }
}
