// DeepSeek V4.1's vision tower: epilogues, 2D RoPE, full attention, SwiGLU, the aligner's unshuffle, the merge.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

#ifndef TF_CTL
#define TF_CTL 0
#endif

inline float tf_vis_bf(float v) {
  return float(bfloat(v));
}

// erf to fp32 precision (W. J. Cody's rational approximations, as libm's erff).
inline float tf_vis_erf(float x) {
  const float ax = fabs(x);
  if (ax < 0.84375f) {
    const float z = x * x;
    const float r = 1.28379167e-01f + z * (-3.25042108e-01f + z * (-2.84817495e-02f + z * (-5.77027029e-03f + z * -2.37630166e-05f)));
    const float s = 1.0f + z * (3.97917233e-01f + z * (6.50222499e-02f + z * (5.08130628e-03f + z * (1.32494738e-04f + z * -3.96022827e-06f))));
    return x + x * (r / s);
  }
  if (ax < 1.25f) {
    const float s = ax - 1.0f;
    const float P = -2.36211857e-03f + s * (4.14856106e-01f + s * (-3.72207481e-01f + s * (3.18346620e-01f + s * (-1.10894694e-01f + s * (3.54783031e-02f + s * -2.16637559e-03f)))));
    const float Q = 1.0f + s * (1.06420882e-01f + s * (5.40397942e-01f + s * (7.18286547e-02f + s * (1.26171216e-01f + s * (1.36370836e-02f + s * 1.19844998e-02f)))));
    const float e = 8.45062911e-01f + P / Q;
    return x >= 0 ? e : -e;
  }
  if (ax >= 6.0f) return x >= 0 ? 1.0f : -1.0f;
  const float s = 1.0f / (ax * ax);
  float R, S;
  if (ax < 1.0f / 0.35f) {
    R = -9.86494403e-03f + s * (-6.93858570e-01f + s * (-1.05586262e+01f + s * (-6.23753357e+01f + s * (-1.62396660e+02f + s * (-1.84605087e+02f + s * (-8.12874374e+01f + s * -9.81432934e+00f))))));
    S = 1.0f + s * (1.96512714e+01f + s * (1.37657754e+02f + s * (4.34565887e+02f + s * (6.45387268e+02f + s * (4.29008148e+02f + s * (1.08635002e+02f + s * (6.57024977e+00f + s * -6.04244141e-02f)))))));
  } else {
    R = -9.86494310e-03f + s * (-7.99283242e-01f + s * (-1.77579556e+01f + s * (-1.60636383e+02f + s * (-6.37566467e+02f + s * (-1.02509509e+03f + s * -4.83519196e+02f)))));
    S = 1.0f + s * (3.03380604e+01f + s * (3.25792511e+02f + s * (1.53672961e+03f + s * (3.19985820e+03f + s * (2.55305029e+03f + s * (4.74528534e+02f + s * -2.24409515e+01f))))));
  }
  const float zh = as_type<float>(as_type<uint>(ax) & 0xfffff000u);
  const float r = metal::precise::exp(-zh * zh - 0.5625f) * metal::precise::exp((zh - ax) * (zh + ax) + R / S);
  const float e = 1.0f - r / ax;
  return x >= 0 ? e : -e;
}

struct TfVisEpiArgs {
  uint rows, n, gelu, has_bias, has_res;
};

// A linear's fp32 sums to bf16: + bias, one rounding; then GELU (rounded) and the residual add (rounded) when asked.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_vis_epi(const device float* ACC [[buffer(0)]], const device bfloat* BIAS [[buffer(1)]],
                                 const device bfloat* RES [[buffer(2)]], device bfloat* Y [[buffer(3)]],
                                 constant TfVisEpiArgs& a [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= a.n || g.y >= a.rows) return;
  const ulong at = ulong(g.y) * a.n + g.x;
  float v = ACC[at];
  if (a.has_bias != 0) v = v + float(BIAS[g.x]);
  float z = tf_vis_bf(v);
  if (a.gelu != 0) z = tf_vis_bf(0.5f * z * (1.0f + tf_vis_erf(z * 0.70710678118654752f)));
  if (a.has_res != 0) z = tf_vis_bf(float(RES[at]) + z);
  Y[at] = bfloat(z);
}

struct TfVisQkvArgs {
  uint rows, heads, hd;
};

// The qkv linear's sums [rows][3][heads][hd] to bf16 with the bias, q and k rotated by their patch's (cos, sin) pairs.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_vis_qkv(const device float* ACC [[buffer(0)]], const device bfloat* BIAS [[buffer(1)]],
                                 const device float2* CS [[buffer(2)]], device bfloat* QKV [[buffer(3)]],
                                 constant TfVisQkvArgs& a [[buffer(4)]], uint3 g [[thread_position_in_grid]]) {
  const uint half_hd = a.hd / 2;
  if (g.x >= half_hd || g.y >= a.rows || g.z >= 3 * a.heads) return;
  const uint part = g.z / a.heads, head = g.z % a.heads;
  const uint D = a.heads * a.hd;
  const uint c1 = part * D + head * a.hd + g.x;
  const ulong at = ulong(g.y) * 3 * D + c1;
  const float x1 = tf_vis_bf(ACC[at] + float(BIAS[c1]));
  const float x2 = tf_vis_bf(ACC[at + half_hd] + float(BIAS[c1 + half_hd]));
  if (part == 2) {
    QKV[at] = bfloat(x1);
    QKV[at + half_hd] = bfloat(x2);
    return;
  }
  const float2 cs = CS[ulong(g.y) * half_hd + g.x];
  const float p = x1 * cs.x, q = x2 * cs.y, r = x2 * cs.x, s = x1 * cs.y;
  QKV[at] = bfloat(p - q);
  QKV[at + half_hd] = bfloat(r + s);
}

struct TfVisAttnArgs {
  uint n, heads;
  float scale;
};

// One simdgroup a (query, head): 32 keys a step, online softmax in fp32, values by the shuffled weights.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_vis_attn(
    const device bfloat* QKV [[buffer(0)]], device bfloat* O [[buffer(1)]], constant TfVisAttnArgs& a [[buffer(2)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint D = a.heads * 64u;
  const ulong stride = 3ul * D;
  const device bfloat* qp = QKV + ulong(tg.x) * stride + tg.y * 64u;
  float q[64];
  for (uint d = 0; d < 64u; ++d) q[d] = float(qp[d]);
  float m = -3.0e38f, l = 0.0f, acc0 = 0.0f, acc1 = 0.0f;
  for (uint j0 = 0; j0 < a.n; j0 += 32u) {
    const uint j = j0 + lane;
    float s = -3.0e38f;
    if (j < a.n) {
      const device bfloat* kp = QKV + ulong(j) * stride + D + tg.y * 64u;
      float t = 0.0f;
      for (uint d = 0; d < 64u; ++d) t = t + q[d] * float(kp[d]);
      s = t * a.scale;
    }
    const float mn = max(m, simd_max(s));
    const float alpha = metal::precise::exp(m - mn);
    const float p = j < a.n ? metal::precise::exp(s - mn) : 0.0f;
    l = l * alpha + simd_sum(p);
    acc0 = acc0 * alpha;
    acc1 = acc1 * alpha;
    const uint cnt = min(32u, a.n - j0);
    for (uint jj = 0; jj < cnt; ++jj) {
      const float pj = simd_shuffle(p, ushort(jj));
      const device bfloat* vp = QKV + ulong(j0 + jj) * stride + 2u * D + tg.y * 64u + 2u * lane;
      acc0 = acc0 + pj * float(vp[0]);
      acc1 = acc1 + pj * float(vp[1]);
    }
    m = mn;
  }
  device bfloat* op = O + ulong(tg.x) * D + tg.y * 64u + 2u * lane;
  op[0] = bfloat(acc0 / l);
  op[1] = bfloat(acc1 / l);
}

struct TfVisSwiArgs {
  uint rows, inter;
};

// The MLP's w1 sums [rows][2][inter] (gate, then up), each rounded to bf16: silu(gate) rounded, times up, rounded.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_vis_swiglu(const device float* ACC [[buffer(0)]], device bfloat* Y [[buffer(1)]],
                                    constant TfVisSwiArgs& a [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= a.inter || g.y >= a.rows) return;
  const ulong at = ulong(g.y) * 2ul * a.inter + g.x;
  const float gt = tf_vis_bf(ACC[at]);
  const float up = tf_vis_bf(ACC[at + a.inter]);
  const float s = tf_vis_bf(gt / (1.0f + metal::precise::exp(-gt)));
  Y[ulong(g.y) * a.inter + g.x] = bfloat(s * up);
}

struct TfVisUnshufArgs {
  uint nh, nw, lh, lw, dim, r;
};

// The aligner's F.unfold of the zero-padded patch grid: out [lh * lw][dim * r * r], column c * r * r + ky * r + kx.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_vis_unshuffle(const device bfloat* X [[buffer(0)]], device bfloat* Y [[buffer(1)]],
                                       constant TfVisUnshufArgs& a [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const uint rr = a.r * a.r;
  if (g.x >= a.dim * rr || g.y >= a.lh * a.lw) return;
  const uint c = g.x / rr, k = g.x % rr;
  const uint py = (g.y / a.lw) * a.r + k / a.r, px = (g.y % a.lw) * a.r + k % a.r;
  const bool in = py < a.nh && px < a.nw;
  Y[ulong(g.y) * a.dim * rr + g.x] = in ? X[(ulong(py) * a.nw + px) * a.dim + c] : bfloat(0.0f);
}

struct TfVisMergeArgs {
  uint dim, rows;
};

// Stream rows [rows][4][dim] take an aligner row (SRC >= 0) or a delimiter (-1 start, -2 end, -3 newline).
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_vis_merge(const device bfloat* FEAT [[buffer(0)]], const device bfloat* DELIM [[buffer(1)]],
                                   const device int* SRC [[buffer(2)]], device bfloat* Y [[buffer(3)]],
                                   constant TfVisMergeArgs& a [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= a.dim || g.y >= a.rows) return;
  const int s = SRC[g.y];
  if (s <= -4) return;
  const bfloat v = s >= 0 ? FEAT[ulong(s) * a.dim + g.x] : DELIM[ulong(-1 - s) * a.dim + g.x];
  for (uint k = 0; k < 4u; ++k) Y[(ulong(g.y) * 4ul + k) * a.dim + g.x] = v;
}
