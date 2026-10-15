// Gemma 4's image tower at torch's per-op bf16 rounding (transformers' modeling_gemma4.py); after gemm_nax.metal.

#define G4V_GEMM(NAME, T, TA, TB, AM, AN)                                                                          \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                                \
      const device T* A [[buffer(0)]], const device T* B [[buffer(1)]], const device int32_t* P [[buffer(2)]],     \
      device T* D [[buffer(3)]], uint sg [[simdgroup_index_in_threadgroup]],                                      \
      uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {                        \
    tfp::gemm_nax<T, TA, TB, AM, AN, false, 64, 128, 256, 2, 4>(A, B, D, P, tg, sg, lane);                         \
  }
G4V_GEMM(g4v_gemm_bf16_nt_00, bfloat16_t, false, true, false, false)
G4V_GEMM(g4v_gemm_bf16_nt_01, bfloat16_t, false, true, false, true)
G4V_GEMM(g4v_gemm_bf16_nt_10, bfloat16_t, false, true, true, false)
G4V_GEMM(g4v_gemm_bf16_nt_11, bfloat16_t, false, true, true, true)
G4V_GEMM(g4v_gemm_f32_nt_00, float, false, true, false, false)
G4V_GEMM(g4v_gemm_f32_nt_01, float, false, true, false, true)
G4V_GEMM(g4v_gemm_f32_nt_10, float, false, true, true, false)
G4V_GEMM(g4v_gemm_f32_nt_11, float, false, true, true, true)
G4V_GEMM(g4v_gemm_f32_nn_00, float, false, false, false, false)
G4V_GEMM(g4v_gemm_f32_nn_01, float, false, false, false, true)
G4V_GEMM(g4v_gemm_f32_nn_10, float, false, false, true, false)
G4V_GEMM(g4v_gemm_f32_nn_11, float, false, false, true, true)

constant constexpr uint G4V_W = 1152, G4V_HEADS = 16, G4V_DH = 72, G4V_HALF = 36, G4V_QKV = 3 * G4V_W;

inline float g4v_bf(float x) { return float(bfloat(x)); }

inline float g4v_group_sum(float v, threadgroup float* part, uint n, uint lane, uint sg) {
  v = simd_sum(v);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float w = lane < (n + 31) / 32 ? part[lane] : 0.0f;
    w = simd_sum(w);
    if (lane == 0) part[32] = w;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float r = part[32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}

struct G4vRows { uint width; uint weighted; float eps; };

// The patches' x and y position rows into the projected patches: x = bf16(x + bf16(tx[px] + ty[py])), grid gw wide.
[[kernel]] void g4v_pos(device bfloat* x [[buffer(0)]], const device bfloat* table [[buffer(1)]],
                        constant uint& gw [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= G4V_W) return;
  const uint px = p.y % gw, py = p.y / gw;
  const float e = g4v_bf(float(table[ulong(px) * G4V_W + p.x]) + float(table[(10240ul + py) * G4V_W + p.x]));
  const ulong i = ulong(p.y) * G4V_W + p.x;
  x[i] = bfloat(float(x[i]) + e);
}

// Gemma4RMSNorm: x * (mean(x^2) + eps)^-0.5 in fp32, times the weight when `weighted`, rounded to bf16 once.
[[kernel]] void g4v_rms(const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], device bfloat* y [[buffer(2)]],
                        constant G4vRows& p [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
                        uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device bfloat* xr = x + ulong(row) * p.width;
  device bfloat* yr = y + ulong(row) * p.width;
  float s = 0.0f;
  for (uint i = t; i < p.width; i += n) s += float(xr[i]) * float(xr[i]);
  const float r = metal::precise::rsqrt(g4v_group_sum(s, part, n, lane, sg) / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += n) yr[i] = bfloat(p.weighted ? float(xr[i]) * r * float(w[i]) : float(xr[i]) * r);
}

// A branch's post norm into the residual: x = bf16(x + Gemma4RMSNorm(y) * w).
[[kernel]] void g4v_rms_add(device bfloat* x [[buffer(0)]], const device bfloat* y [[buffer(1)]], const device bfloat* w [[buffer(2)]],
                            constant float& eps [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
                            uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  device bfloat* xr = x + ulong(row) * G4V_W;
  const device bfloat* yr = y + ulong(row) * G4V_W;
  float s = 0.0f;
  for (uint i = t; i < G4V_W; i += n) s += float(yr[i]) * float(yr[i]);
  const float r = metal::precise::rsqrt(g4v_group_sum(s, part, n, lane, sg) / float(G4V_W) + eps);
  for (uint i = t; i < G4V_W; i += n) xr[i] = bfloat(float(xr[i]) + g4v_bf(float(yr[i]) * r * float(w[i])));
}

// qkv [P, 3456] to head-major q, k, v fp32: per-head norms (v unweighted), q and k by the 2D RoPE (x, then y).
[[kernel]] void g4v_qkv(const device bfloat* qkv [[buffer(0)]], const device bfloat* qw [[buffer(1)]],
                        const device bfloat* kw [[buffer(2)]], const device bfloat* cs [[buffer(3)]],
                        const device bfloat* sn [[buffer(4)]], device float* Q [[buffer(5)]], device float* K [[buffer(6)]],
                        device float* V [[buffer(7)]], constant uint& P [[buffer(8)]], constant float& eps [[buffer(9)]],
                        uint g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float nrm[G4V_DH];
  const uint part = g % 3, head = (g / 3) % G4V_HEADS, row = g / (3 * G4V_HEADS);
  if (row >= P) return;
  const device bfloat* x = qkv + ulong(row) * G4V_QKV + part * G4V_W + head * G4V_DH;
  const device bfloat* w = part == 0 ? qw : kw;
  float s = 0.0f;
  for (uint d = lane; d < G4V_DH; d += 32) s += float(x[d]) * float(x[d]);
  const float r = metal::precise::rsqrt(simd_sum(s) / float(G4V_DH) + eps);
  for (uint d = lane; d < G4V_DH; d += 32) nrm[d] = g4v_bf(part == 2 ? float(x[d]) * r : float(x[d]) * r * float(w[d]));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float* out = (part == 0 ? Q : part == 1 ? K : V) + (ulong(head) * P + row) * G4V_DH;
  for (uint d = lane; d < G4V_DH; d += 32) {
    if (part == 2) { out[d] = nrm[d]; continue; }
    const uint base = d / G4V_HALF * G4V_HALF, j = d % G4V_HALF;
    const float other = j < G4V_HALF / 2 ? -nrm[base + j + G4V_HALF / 2] : nrm[base + j - G4V_HALF / 2];
    const float c = float(cs[row * G4V_DH + d]), sv = float(sn[row * G4V_DH + d]);
    out[d] = g4v_bf(g4v_bf(nrm[d] * c) + g4v_bf(other * sv));
  }
}

// Each row of S (n scores) to its softmax, in place (the tower's attention scale is 1).
[[kernel]] void g4v_softmax(device float* S [[buffer(0)]], constant uint& n [[buffer(1)]], uint row [[threadgroup_position_in_grid]],
                            uint t [[thread_position_in_threadgroup]], uint tn [[threads_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  device float* s = S + ulong(row) * n;
  float m = -INFINITY;
  for (uint i = t; i < n; i += tn) m = max(m, s[i]);
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
    const float e = metal::precise::exp(s[i] - m);
    s[i] = e;
    z += e;
  }
  const float inv = 1.0f / g4v_group_sum(z, part, tn, lane, sg);
  for (uint i = t; i < n; i += tn) s[i] *= inv;
}

// O [16, rows, 72] fp32 to rows r0.. of out [P, 1152] bf16.
[[kernel]] void g4v_heads(const device float* O [[buffer(0)]], device bfloat* out [[buffer(1)]],
                          constant uint2& p [[buffer(2)]], uint2 at [[thread_position_in_grid]]) {
  const uint rows = p.x, r0 = p.y, h = at.x / G4V_DH, d = at.x % G4V_DH, r = at.y;
  if (at.x >= G4V_W || r >= rows) return;
  out[ulong(r0 + r) * G4V_W + at.x] = bfloat(O[(ulong(h) * rows + r) * G4V_DH + d]);
}

// gate | up [P, 2 n] to bf16(bf16(gelu_tanh(gate)) * up) [P, n] (torch's gelu, approximate "tanh", on bf16).
[[kernel]] void g4v_geglu(const device bfloat* gu [[buffer(0)]], device bfloat* act [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= n) return;
  const float g = float(gu[ulong(p.y) * 2 * n + p.x]), u = float(gu[ulong(p.y) * 2 * n + n + p.x]);
  const float a = g4v_bf(0.5f * g * (1.0f + metal::precise::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g))));
  act[ulong(p.y) * n + p.x] = bfloat(a * u);
}

// Gemma4VisionPooler and standardize: 3 x 3 block means to bf16, then (x sqrt(1152) - bias) scale in fp32, to bf16.
[[kernel]] void g4v_pool(const device bfloat* x [[buffer(0)]], const device bfloat* bias [[buffer(1)]],
                         const device bfloat* scale [[buffer(2)]], device bfloat* out [[buffer(3)]],
                         constant uint& gw [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= G4V_W) return;
  const uint bw = gw / 3, ty = p.y / bw, tx = p.y % bw;
  float s = 0.0f;
  for (uint dy = 0; dy < 3; dy++)
    for (uint dx = 0; dx < 3; dx++) s += float(x[(ulong(3 * ty + dy) * gw + 3 * tx + dx) * G4V_W + p.x]) * (1.0f / 9.0f);
  const float h = g4v_bf(s) * 33.941125496954285f;
  out[ulong(p.y) * G4V_W + p.x] = bfloat((h - float(bias[p.x])) * float(scale[p.x]));
}

// An image's rows over the prompt's embedded rows, 5376 bf16 a row: the soft tokens go in unscaled.
[[kernel]] void g4v_rows(const device uint4* src [[buffer(0)]], device uint4* dst [[buffer(1)]],
                         uint2 p [[thread_position_in_grid]]) {
  if (p.x >= 5376 / 8) return;
  dst[ulong(p.y) * (5376 / 8) + p.x] = src[ulong(p.y) * (5376 / 8) + p.x];
}
