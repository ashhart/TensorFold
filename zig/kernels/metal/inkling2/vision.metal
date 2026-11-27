// Inkling 2 image tower glue (HF Inkling2VisionModel): each torch op's output rounded once to bf16, sums in fp32.

// The patches as bf16, the dtype the reference casts pixel_values to before the patch projection.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2v_pack(
    const device float* x [[buffer(0)]], device bfloat* y [[buffer(1)]], constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i < n) y[i] = bfloat(x[i]);
}

struct Ik2vEmbed {
  uint patches;  // patch rows of the image; the sequence is one class row and then these
  uint width;
};

// Row 0: bf16(class + class position). Row 1 + i: bf16(patch_i + bf16(sum of the 4 bilinear taps of table * w)).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2v_embed(
    const device bfloat* pe [[buffer(0)]], const device bfloat* table [[buffer(1)]],
    const device uint* taps [[buffer(2)]], const device float* weights [[buffer(3)]],
    const device bfloat* cls [[buffer(4)]], const device bfloat* cls_pos [[buffer(5)]], device bfloat* x [[buffer(6)]],
    constant Ik2vEmbed& p [[buffer(7)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.width || at.y > p.patches) return;
  if (at.y == 0) {
    x[at.x] = bfloat(float(cls[at.x]) + float(cls_pos[at.x]));
    return;
  }
  const ulong i = ulong(at.y - 1);
  float pos = 0.0f;
  for (uint k = 0; k < 4; k++) pos += float(table[ulong(taps[i * 4 + k]) * p.width + at.x]) * weights[i * 4 + k];
  x[ulong(at.y) * p.width + at.x] = bfloat(float(pe[i * p.width + at.x]) + tf_vis_bf(pos));
}

struct Ik2vRows {
  uint rows;
  uint width;
  float eps;
};

#define IK2V_ROW(i) float(xr[i])

// nn.LayerNorm on bf16 rows: fp32 statistics, one rounding of (x - mean) * inv * w + b.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2v_layer_norm(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], const device bfloat* b [[buffer(2)]],
    device bfloat* y [[buffer(3)]], constant Ik2vRows& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const device bfloat* xr = x + ulong(row) * p.width;
  device bfloat* yr = y + ulong(row) * p.width;
  TF_VIS_LAYER_NORM_STATS(IK2V_ROW)
  TF_VIS_LAYER_NORM_OUT(IK2V_ROW)
}

struct Ik2vQkv {
  uint rows;  // sequence rows in all
  uint heads;
  uint dim;
  uint r0;    // the first sequence row of acc's rows
};

// acc [n, 3, heads * dim] fp32 q, k, v sums to head-major fp32 Q, K, V [heads, rows, dim]: bias, bf16, q k rotated.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2v_qkv(
    const device float* acc [[buffer(0)]], const device bfloat* bq [[buffer(1)]], const device bfloat* bk [[buffer(2)]],
    const device bfloat* bv [[buffer(3)]], const device float* cs [[buffer(4)]], const device float* sn [[buffer(5)]],
    device float* Q [[buffer(6)]], device float* K [[buffer(7)]], device float* V [[buffer(8)]],
    constant Ik2vQkv& p [[buffer(9)]], uint3 at [[thread_position_in_grid]]) {
  const uint d = at.x, which = at.y / p.heads, h = at.y % p.heads, r = at.z, g = p.r0 + at.z;
  const uint width = p.heads * p.dim;
  const ulong base = ulong(r) * 3 * width + ulong(which) * width + h * p.dim;
  const device bfloat* bias = which == 0 ? bq : (which == 1 ? bk : bv);
  const float v = tf_vis_bf(acc[base + d] + float(bias[h * p.dim + d]));
  const ulong out = (ulong(h) * p.rows + g) * p.dim + d;
  if (which == 2) {
    V[out] = v;
    return;
  }
  const uint e = d ^ 1u;
  const float partner = tf_vis_bf(acc[base + e] + float(bias[h * p.dim + e]));
  const float c = cs[ulong(g) * p.dim + d], s = sn[ulong(g) * p.dim + d];
  const float turned = (d & 1u) == 0 ? v * c - partner * s : v * c + partner * s;
  (which == 0 ? Q : K)[out] = tf_vis_bf(turned);
}

// bf16 scores (rounded by vision_gemm.metal's epilogue) to bf16 probabilities in place; the sum's order fixes the bits.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2v_softmax(
    device bfloat* S [[buffer(0)]], constant uint& n [[buffer(1)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint tn [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  device bfloat* s = S + ulong(row) * n;
  float m = -INFINITY;
  for (uint i = t; i < n; i += tn) m = max(m, float(s[i]));
  m = tf_vis_max(m, part, tn, lane, sg);
  float z = 0.0f;
  for (uint i = t; i < n; i += tn) {
    const float e = metal::precise::exp(float(s[i]) - m);
    z += e;
  }
  const float inv = 1.0f / tf_vis_sum(z, part, tn, lane, sg);
  for (uint i = t; i < n; i += tn) s[i] = bfloat(metal::precise::exp(float(s[i]) - m) * inv);
}

struct Ik2vHeads {
  uint rows;
  uint r0;
  uint heads;
  uint dim;
};

// O [heads, rows, dim] (one chunk of queries, fp32) to rows [r0, r0 + rows) of out [S, heads * dim], bf16.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2v_heads(
    const device float* O [[buffer(0)]], device bfloat* out [[buffer(1)]], constant Ik2vHeads& p [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  const uint h = at.x / p.dim, d = at.x % p.dim;
  out[ulong(p.r0 + at.y) * (p.heads * p.dim) + at.x] = bfloat(O[(ulong(h) * p.rows + at.y) * p.dim + d]);
}

// The exact GELU (nn.GELU's erf form); erf by Abramowitz and Stegun 7.1.26, |error| < 1.5e-7, far under bf16's step.
inline float ik2v_erf(float x) {
  const float a = fabs(x), t = 1.0f / (1.0f + 0.3275911f * a);
  const float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  const float e = 1.0f - poly * metal::precise::exp(-a * a);
  return x < 0.0f ? -e : e;
}

inline float ik2v_gelu(float v) { return 0.5f * v * (1.0f + ik2v_erf(v * 0.70710678118654752f)); }

// y = bf16(gelu(bf16(acc + b))) over acc [rows, width] fp32 sums: the MLP's first linear and its activation.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2v_bias_gelu(
    const device float* acc [[buffer(0)]], const device bfloat* b [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant uint& width [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  y[i] = bfloat(ik2v_gelu(tf_vis_bf(acc[i] + float(b[at.x]))));
}

// y = bf16(gelu(y)): the merger's activation between its two bias-free linears.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2v_gelu_rows(
    device bfloat* y [[buffer(0)]], constant uint& width [[buffer(1)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  y[i] = bfloat(ik2v_gelu(float(y[i])));
}

// x = bf16(x + bf16(lambda * bf16(acc + b))): a layer-scaled branch joins the bf16 residual stream.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2v_residual(
    const device float* acc [[buffer(0)]], const device bfloat* b [[buffer(1)]], const device bfloat* lambda [[buffer(2)]],
    device bfloat* x [[buffer(3)]], constant uint& width [[buffer(4)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  const float branch = tf_vis_bf(acc[i] + float(b[at.x]));
  x[i] = bfloat(float(x[i]) + tf_vis_bf(float(lambda[at.x]) * branch));
}
