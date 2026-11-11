// Qwen-VL image towers (the Qwen3.5 tower, any width): bf16 weights, fp32 sums and residual; gemm_nax.metal first.

// x [M, K] w^T: every linear layer (bf16 in and out, fp32 sums)
TF_GEMM_NAX_KERNEL(qwv_gemm_bf16_nt_00, bfloat16_t, false, true, false, false)
TF_GEMM_NAX_KERNEL(qwv_gemm_bf16_nt_01, bfloat16_t, false, true, false, true)
TF_GEMM_NAX_KERNEL(qwv_gemm_bf16_nt_10, bfloat16_t, false, true, true, false)
TF_GEMM_NAX_KERNEL(qwv_gemm_bf16_nt_11, bfloat16_t, false, true, true, true)
// q k^T per head in fp32
TF_GEMM_NAX_KERNEL(qwv_gemm_f32_nt_00, float, false, true, false, false)
TF_GEMM_NAX_KERNEL(qwv_gemm_f32_nt_01, float, false, true, false, true)
TF_GEMM_NAX_KERNEL(qwv_gemm_f32_nt_10, float, false, true, true, false)
TF_GEMM_NAX_KERNEL(qwv_gemm_f32_nt_11, float, false, true, true, true)
// p v per head in fp32
TF_GEMM_NAX_KERNEL(qwv_gemm_f32_nn_00, float, false, false, false, false)
TF_GEMM_NAX_KERNEL(qwv_gemm_f32_nn_01, float, false, false, false, true)
TF_GEMM_NAX_KERNEL(qwv_gemm_f32_nn_10, float, false, false, true, false)
TF_GEMM_NAX_KERNEL(qwv_gemm_f32_nn_11, float, false, false, true, true)

#ifdef TF_NAX_STRICT_FP32
#define QWV_EXACT_GEMM(NAME, AM, AN) \
  [[max_total_threads_per_threadgroup(256)]] kernel void NAME( \
      const device float* A, const device float* B, const device int* P, device float* D, \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
      uint3 tg [[threadgroup_position_in_grid]]) { \
    tfp::gemm_nax<float, false, true, AM, AN, false, 64, 128, 256, 2, 4, true>(A, B, D, P, tg, sg, lane); \
  }
QWV_EXACT_GEMM(qwv_gemm_f32_exact_nt_00, false, false)
QWV_EXACT_GEMM(qwv_gemm_f32_exact_nt_01, false, true)
QWV_EXACT_GEMM(qwv_gemm_f32_exact_nt_10, true, false)
QWV_EXACT_GEMM(qwv_gemm_f32_exact_nt_11, true, true)
#undef QWV_EXACT_GEMM
#endif

// The fp32 residual stream's start: the patch projection p plus its bias and the patch's four learned-grid taps.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void qwv_embed(
    const device bfloat* p [[buffer(0)]], const device bfloat* b [[buffer(1)]], const device bfloat* table [[buffer(2)]],
    const device uint* taps [[buffer(3)]], const device float* weights [[buffer(4)]], constant uint& width [[buffer(5)]],
    device float* x [[buffer(6)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  float pos = 0.0f;
  for (uint k = 0; k < 4; k++) pos += float(table[ulong(taps[at.y * 4 + k]) * width + at.x]) * weights[at.y * 4 + k];
  x[i] = float(p[i]) + float(b[at.x]) + pos;
}

struct QwvRows {
  uint rows;
  uint width;
  float eps;
};

// LayerNorm of the fp32 residual stream with weight and bias, one threadgroup a row: bf16 of the fp32 result.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void qwv_layer_norm(
    const device float* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], const device bfloat* b [[buffer(2)]],
    device bfloat* y [[buffer(3)]], constant QwvRows& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const device float* xr = x + ulong(row) * p.width;
  device bfloat* yr = y + ulong(row) * p.width;
  TF_VIS_LAYER_NORM_STATS(TF_VIS_ROW)
  TF_VIS_LAYER_NORM_OUT(TF_VIS_ROW)
}

struct QwvQkv {
  uint rows;
  uint heads;
  uint dim;
};

// qkv [rows, 3, heads, dim] + bias to head-major fp32 Q, K, V; q, k turned (rotate_half, fp32), all bf16-rounded.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void qwv_qkv(
    const device bfloat* qkv [[buffer(0)]], const device bfloat* bias [[buffer(1)]], const device float* cs [[buffer(2)]],
    const device float* sn [[buffer(3)]], device float* Q [[buffer(4)]], device float* K [[buffer(5)]],
    device float* V [[buffer(6)]], constant QwvQkv& p [[buffer(7)]], uint3 at [[thread_position_in_grid]]) {
  const uint d = at.x, which = at.y / p.heads, h = at.y % p.heads, r = at.z, mid = p.dim / 2;
  const uint col = (which * p.heads + h) * p.dim;
  const ulong row = ulong(r) * 3 * p.heads * p.dim;
  const float v = float(qkv[row + col + d]) + float(bias[col + d]);
  const ulong out = (ulong(h) * p.rows + r) * p.dim + d;
  if (which == 2) {
    V[out] = tf_vis_bf(v);
    return;
  }
  const uint e = d < mid ? d + mid : d - mid;
  const float partner = float(qkv[row + col + e]) + float(bias[col + e]);
  const float c = cs[ulong(r) * p.dim + d], s = sn[ulong(r) * p.dim + d];
  const float turned = d < mid ? v * c - partner * s : v * c + partner * s;
  (which == 0 ? Q : K)[out] = tf_vis_bf(turned);
}

// Each row of S [rows, n] (fp32 scores) as softmax(scale * s) in place, one threadgroup a row.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void qwv_softmax(
    device float* S [[buffer(0)]], constant uint& n [[buffer(1)]], constant float& scale [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint tn [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  device float* s = S + ulong(row) * n;
  float m = -INFINITY;
  for (uint i = t; i < n; i += tn) m = max(m, s[i] * scale);
  m = tf_vis_max(m, part, tn, lane, sg);
  float z = 0.0f;
  for (uint i = t; i < n; i += tn) {
    const float e = metal::precise::exp(s[i] * scale - m);
    s[i] = e;
    z += e;
  }
  const float inv = 1.0f / tf_vis_sum(z, part, tn, lane, sg);
  for (uint i = t; i < n; i += tn) s[i] *= inv;
}

struct QwvHeads {
  uint rows;
  uint r0;
  uint heads;
  uint dim;
};

// O [heads, rows, dim] (one chunk of queries, fp32) to rows [r0, r0 + rows) of out [L, heads * dim], bf16.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void qwv_heads(
    const device float* O [[buffer(0)]], device bfloat* out [[buffer(1)]], constant QwvHeads& p [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  const uint h = at.x / p.dim, d = at.x % p.dim;
  out[ulong(p.r0 + at.y) * (p.heads * p.dim) + at.x] = bfloat(O[(ulong(h) * p.rows + at.y) * p.dim + d]);
}

// erf to fp32's need here (Abramowitz and Stegun 7.1.26, |error| < 1.5e-7, far under a bf16 output's step).
inline float qwv_erf(float x) {
  const float a = fabs(x), t = 1.0f / (1.0f + 0.3275911f * a);
  const float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  const float e = 1.0f - poly * metal::precise::exp(-a * a);
  return x < 0.0f ? -e : e;
}

struct QwvAct {
  uint width;
  uint exact;  // 0: gelu_pytorch_tanh (the blocks' MLP), 1: the exact GELU (the merger's)
};

// y = gelu(y + b) in fp32, one rounding.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void qwv_bias_gelu(
    device bfloat* y [[buffer(0)]], const device bfloat* b [[buffer(1)]], constant QwvAct& p [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * p.width + at.x;
  const float v = float(y[i]) + float(b[at.x]);
  const float inner = p.exact != 0 ? qwv_erf(v * 0.70710678118654752f)
                                   : metal::precise::tanh(0.79788456080286536f * (v + 0.044715f * v * v * v));
  y[i] = bfloat(0.5f * v * (1.0f + inner));
}

// x += y + b in fp32: a block's residual after its projection and that projection's bias.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void qwv_residual(
    device float* x [[buffer(0)]], const device bfloat* y [[buffer(1)]], const device bfloat* b [[buffer(2)]],
    constant uint& width [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  x[i] += float(y[i]) + float(b[at.x]);
}

// y = y + b (the merger's output bias).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void qwv_bias(
    device bfloat* y [[buffer(0)]], const device bfloat* b [[buffer(1)]], constant uint& width [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  y[i] = bfloat(float(y[i]) + float(b[at.x]));
}
