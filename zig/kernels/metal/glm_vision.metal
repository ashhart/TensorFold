// GLM-5.3-Flash's image tower (GLM5-Next vision: 24 pre-norm blocks of 1024, 16 heads of 64, 2-D rotary, a 2x2 merge
// and a SwiGLU merger to the language model's 4096). Compiled after prefill/gemm_nax.metal, whose tfp::gemm_nax runs
// every projection; the rest is MLX's per-op arithmetic as mlx-vlm's glm5_next/vision.py runs it in bf16.

// ---- the projections and attention products: tfp::gemm_nax (bm64 bn128 bk256 wm2 wn4) by operand layout ----
#define GLMV_GEMM(NAME, T, TA, TB, AM, AN)                                                                         \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                                \
      const device T* A [[buffer(0)]], const device T* B [[buffer(1)]], const device int32_t* P [[buffer(2)]],     \
      device T* D [[buffer(3)]], uint sg [[simdgroup_index_in_threadgroup]],                                      \
      uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {                        \
    tfp::gemm_nax<T, TA, TB, AM, AN, false, 64, 128, 256, 2, 4>(A, B, D, P, tg, sg, lane);                         \
  }
// x [M, K] w^T: every linear layer (bf16 in and out, fp32 sums)
GLMV_GEMM(glmv_gemm_bf16_nt_00, bfloat16_t, false, true, false, false)
GLMV_GEMM(glmv_gemm_bf16_nt_01, bfloat16_t, false, true, false, true)
GLMV_GEMM(glmv_gemm_bf16_nt_10, bfloat16_t, false, true, true, false)
GLMV_GEMM(glmv_gemm_bf16_nt_11, bfloat16_t, false, true, true, true)
// q k^T per head in fp32 (the scores MLX's attention keeps in fp32)
GLMV_GEMM(glmv_gemm_f32_nt_00, float, false, true, false, false)
GLMV_GEMM(glmv_gemm_f32_nt_01, float, false, true, false, true)
GLMV_GEMM(glmv_gemm_f32_nt_10, float, false, true, true, false)
GLMV_GEMM(glmv_gemm_f32_nt_11, float, false, true, true, true)
// p v per head in fp32
GLMV_GEMM(glmv_gemm_f32_nn_00, float, false, false, false, false)
GLMV_GEMM(glmv_gemm_f32_nn_01, float, false, false, false, true)
GLMV_GEMM(glmv_gemm_f32_nn_10, float, false, false, true, false)
GLMV_GEMM(glmv_gemm_f32_nn_11, float, false, false, true, true)

struct GlmvRows {
  uint rows;
  uint width;
  float eps;
};

inline float glmv_bf(float x) { return float(bfloat(x)); }

// One threadgroup's sum of every thread's `v` (at most 1024 threads), returned to all of them.
inline float glmv_group_sum(float v, threadgroup float* part, uint t, uint n, uint lane, uint sg) {
  v = simd_sum(v);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float s = lane < (n + 31) / 32 ? part[lane] : 0.0f;
    s = simd_sum(s);
    if (lane == 0) part[32] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float out = part[32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out;
}

// y[r, c] = y + b[c] (a linear layer's bias, added after its product as nn.Linear's bf16 add rounds it)
[[kernel]] void glmv_bias(device bfloat* y [[buffer(0)]], const device bfloat* b [[buffer(1)]],
                          constant uint& width [[buffer(2)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  y[i] = bfloat(float(y[i]) + float(b[at.x]));
}

// x += y (a block's residual)
[[kernel]] void glmv_add(device bfloat* x [[buffer(0)]], const device bfloat* y [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
  x[i] = bfloat(float(x[i]) + float(y[i]));
}

// mx.fast.rms_norm, one threadgroup a row: w * bf16(x * rsqrt(mean(x^2) + eps))
[[kernel]] void glmv_rms(const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                         device bfloat* y [[buffer(2)]], constant GlmvRows& p [[buffer(3)]],
                         uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                         uint n [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device bfloat* xr = x + ulong(row) * p.width;
  device bfloat* yr = y + ulong(row) * p.width;
  float s = 0.0f;
  for (uint i = t; i < p.width; i += n) {
    const float v = float(xr[i]);
    s += v * v;
  }
  s = glmv_group_sum(s, part, t, n, lane, sg);
  const float inv = metal::precise::rsqrt(s / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += n) yr[i] = bfloat(float(w[i]) * glmv_bf(float(xr[i]) * inv));
}

// mx.fast.layer_norm with weight and bias, one threadgroup a row: w * bf16((x - mean) * rsqrt(var + eps)) + b
[[kernel]] void glmv_layer_norm(const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                                const device bfloat* b [[buffer(2)]], device bfloat* y [[buffer(3)]],
                                constant GlmvRows& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                                uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device bfloat* xr = x + ulong(row) * p.width;
  device bfloat* yr = y + ulong(row) * p.width;
  float s = 0.0f;
  for (uint i = t; i < p.width; i += n) s += float(xr[i]);
  const float mean = glmv_group_sum(s, part, t, n, lane, sg) / float(p.width);
  float v = 0.0f;
  for (uint i = t; i < p.width; i += n) {
    const float d = float(xr[i]) - mean;
    v += d * d;
  }
  const float inv = metal::precise::rsqrt(glmv_group_sum(v, part, t, n, lane, sg) / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += n) {
    yr[i] = bfloat(glmv_bf(float(w[i]) * glmv_bf((float(xr[i]) - mean) * inv)) + float(b[i]));
  }
}

struct GlmvQkv {
  uint L;     // rows (patches) of this image
  float eps;  // the q and k norms'
};

// One simdgroup per (row, head, q|k|v) of qkv [L, 3, 16, 64]: q and k RMS-normed, rotated by the row's 2-D angles
// (cos and sin [L, 64]: the height's 16 frequencies, then the width's, twice), and written head-major in fp32
// ([16, L, 64], each value a bf16 as MLX hands its attention); v as stored.
[[kernel]] void glmv_qkv(const device bfloat* qkv [[buffer(0)]], const device bfloat* qn [[buffer(1)]],
                         const device bfloat* kn [[buffer(2)]], const device float* cs [[buffer(3)]],
                         const device float* sn [[buffer(4)]], device float* Q [[buffer(5)]],
                         device float* K [[buffer(6)]], device float* V [[buffer(7)]],
                         constant GlmvQkv& p [[buffer(8)]], uint g [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint which = g % 3, h = (g / 3) % 16, r = g / 48;
  const device bfloat* src = qkv + ulong(r) * 3072 + which * 1024 + h * 64;
  const ulong out = (ulong(h) * p.L + r) * 64;
  const float a = float(src[lane]), b = float(src[lane + 32]);
  if (which == 2) {
    V[out + lane] = a;
    V[out + lane + 32] = b;
    return;
  }
  const device bfloat* w = which == 0 ? qn : kn;
  const float inv = metal::precise::rsqrt(simd_sum(a * a + b * b) / 64.0f + p.eps);
  const float na = glmv_bf(float(w[lane]) * glmv_bf(a * inv));
  const float nb = glmv_bf(float(w[lane + 32]) * glmv_bf(b * inv));
  const device float* c = cs + ulong(r) * 64;
  const device float* s = sn + ulong(r) * 64;
  device float* dst = which == 0 ? Q : K;
  dst[out + lane] = glmv_bf(na * c[lane] - nb * s[lane]);
  dst[out + lane + 32] = glmv_bf(nb * c[lane + 32] + na * s[lane + 32]);
}

// Each row of S [rows, n] (fp32 scores) as softmax(scale * s), in place; one threadgroup a row.
[[kernel]] void glmv_softmax(device float* S [[buffer(0)]], constant uint& n [[buffer(1)]],
                             constant float& scale [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                             uint t [[thread_position_in_threadgroup]], uint tn [[threads_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  device float* s = S + ulong(row) * n;
  float m = -INFINITY;
  for (uint i = t; i < n; i += tn) m = max(m, s[i] * scale);
  m = simd_max(m);
  if (lane == 0) part[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v = lane < (tn + 31) / 32 ? part[lane] : -INFINITY;
    v = simd_max(v);
    if (lane == 0) part[32] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = part[32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float z = 0.0f;
  for (uint i = t; i < n; i += tn) {
    const float e = metal::precise::exp(s[i] * scale - m);
    s[i] = e;
    z += e;
  }
  const float inv = 1.0f / glmv_group_sum(z, part, t, tn, lane, sg);
  for (uint i = t; i < n; i += tn) s[i] *= inv;
}

// O [16, rows, 64] (fp32, one chunk's attention) into rows [r0, r0 + rows) of out [L, 1024] (bf16, heads side by side)
[[kernel]] void glmv_heads(const device float* O [[buffer(0)]], device bfloat* out [[buffer(1)]],
                           constant uint2& p [[buffer(2)]], uint2 at [[thread_position_in_grid]]) {
  const uint rows = p.x, r0 = p.y, h = at.x / 64, d = at.x % 64, r = at.y;
  out[ulong(r0 + r) * 1024 + at.x] = bfloat(O[(ulong(h) * rows + r) * 64 + d]);
}

// The vision MLP's and merger's gate: silu(min(g, limit)) * clip(u, -limit, limit), each op rounded as MLX does.
// gu: g then u, `width` each, a row at a time (stride 2 * width); act [rows, width].
[[kernel]] void glmv_swiglu(const device bfloat* g [[buffer(0)]], const device bfloat* u [[buffer(1)]],
                            device bfloat* act [[buffer(2)]], constant float& limit [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
  const float gate = glmv_bf(min(float(g[i]), limit));
  const float up = glmv_bf(clamp(float(u[i]), -limit, limit));
  const float sig = glmv_bf(1.0f / (1.0f + metal::precise::exp(-gate)));
  act[i] = bfloat(glmv_bf(gate * sig) * up);
}

// erf to fp32 precision (W. J. Cody's rational approximations, as libm's erff).
inline float glmv_erf(float x) {
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

// nn.gelu (exact): x * (1 + erf(x / sqrt(2))) / 2, each op rounded to bf16 as the compiled MLX graph does
[[kernel]] void glmv_gelu(device bfloat* x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  const float v = float(x[i]);
  const float t = glmv_bf(v / 1.41421356237309504880f);
  const float e = glmv_bf(glmv_erf(t));
  const float w = glmv_bf(v * glmv_bf(1.0f + e));
  x[i] = bfloat(w / 2.0f);
}
