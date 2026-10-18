// Step-5's perception encoder at torch's per-op bf16 rounding (vision_encoder.py); after gemm_nax.metal.

#define S5V_GEMM(NAME, T, TA, TB, AM, AN)                                                                          \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                                \
      const device T* A [[buffer(0)]], const device T* B [[buffer(1)]], const device int32_t* P [[buffer(2)]],     \
      device T* D [[buffer(3)]], uint sg [[simdgroup_index_in_threadgroup]],                                      \
      uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {                        \
    tfp::gemm_nax<T, TA, TB, AM, AN, false, 64, 128, 256, 2, 4>(A, B, D, P, tg, sg, lane);                         \
  }
S5V_GEMM(s5v_gemm_bf16_nt_00, bfloat16_t, false, true, false, false)
S5V_GEMM(s5v_gemm_bf16_nt_01, bfloat16_t, false, true, false, true)
S5V_GEMM(s5v_gemm_bf16_nt_10, bfloat16_t, false, true, true, false)
S5V_GEMM(s5v_gemm_bf16_nt_11, bfloat16_t, false, true, true, true)
S5V_GEMM(s5v_gemm_f32_nt_00, float, false, true, false, false)
S5V_GEMM(s5v_gemm_f32_nt_01, float, false, true, false, true)
S5V_GEMM(s5v_gemm_f32_nt_10, float, false, true, true, false)
S5V_GEMM(s5v_gemm_f32_nt_11, float, false, true, true, true)
S5V_GEMM(s5v_gemm_f32_nn_00, float, false, false, false, false)
S5V_GEMM(s5v_gemm_f32_nn_01, float, false, false, false, true)
S5V_GEMM(s5v_gemm_f32_nn_10, float, false, false, true, false)
S5V_GEMM(s5v_gemm_f32_nn_11, float, false, false, true, true)

inline float s5v_bf(float x) { return float(bfloat(x)); }

inline float s5v_group_sum(float v, threadgroup float* part, uint t, uint n, uint lane, uint sg) {
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

struct S5vRows { uint rows, width; float eps; };

// y[r, i] = bf16(y + b[i]): a linear layer's bias.
[[kernel]] void s5v_bias(device bfloat* y [[buffer(0)]], const device bfloat* b [[buffer(1)]],
                         constant uint& width [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= width) return;
  const ulong i = ulong(p.y) * width + p.x;
  y[i] = bfloat(float(y[i]) + float(b[p.x]));
}

// x = bf16(x + bf16(y * gamma)): a block's branch through its layer scale into the residual; gamma null: no scale.
[[kernel]] void s5v_scale_add(device bfloat* x [[buffer(0)]], const device bfloat* y [[buffer(1)]],
                              const device bfloat* gamma [[buffer(2)]], constant uint& width [[buffer(3)]],
                              uint2 p [[thread_position_in_grid]]) {
  if (p.x >= width) return;
  const ulong i = ulong(p.y) * width + p.x;
  x[i] = bfloat(float(x[i]) + s5v_bf(float(y[i]) * float(gamma[p.x])));
}

// x = bf16(x + pos): the patches' absolute position embedding.
[[kernel]] void s5v_add(device bfloat* x [[buffer(0)]], const device bfloat* pos [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
  x[i] = bfloat(float(x[i]) + float(pos[i]));
}

// nn.LayerNorm in bf16: fp32 mean and variance, bf16((x - mean) * rstd) * w + b rounded as torch's bf16 ops do.
[[kernel]] void s5v_layer_norm(const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                               const device bfloat* b [[buffer(2)]], device bfloat* y [[buffer(3)]],
                               constant S5vRows& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                               uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device bfloat* xr = x + ulong(row) * p.width;
  device bfloat* yr = y + ulong(row) * p.width;
  float s = 0.0f;
  for (uint i = t; i < p.width; i += n) s += float(xr[i]);
  const float mean = s5v_group_sum(s, part, t, n, lane, sg) / float(p.width);
  float v = 0.0f;
  for (uint i = t; i < p.width; i += n) {
    const float d = float(xr[i]) - mean;
    v += d * d;
  }
  const float inv = metal::precise::rsqrt(s5v_group_sum(v, part, t, n, lane, sg) / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += n) yr[i] = bfloat((float(xr[i]) - mean) * inv * float(w[i]) + float(b[i]));
}

// qkv [P, 4608] to q, k, v [16, P, 96] fp32, q and k by the 2D RoPE (interleaved pairs, bf16 cos and sin [P, 96]).
[[kernel]] void s5v_qkv(const device bfloat* qkv [[buffer(0)]], const device bfloat* cs [[buffer(1)]],
                        const device bfloat* sn [[buffer(2)]], device float* Q [[buffer(3)]], device float* K [[buffer(4)]],
                        device float* V [[buffer(5)]], constant uint& P [[buffer(6)]], uint g [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const uint part = g % 3, head = (g / 3) % 16, row = g / 48;
  if (row >= P) return;
  const device bfloat* x = qkv + ulong(row) * 4608 + part * 1536 + head * 96;
  device float* out = (part == 0 ? Q : part == 1 ? K : V) + (ulong(head) * P + row) * 96;
  for (uint d = lane; d < 96; d += 32) {
    if (part == 2) { out[d] = float(x[d]); continue; }
    const float c = float(cs[row * 96 + d]), s = float(sn[row * 96 + d]);
    const float other = (d & 1) ? float(x[d - 1]) : -float(x[d + 1]); // rotate_half: (-x2, x1) per pair
    out[d] = s5v_bf(s5v_bf(float(x[d]) * c) + s5v_bf(other * s));
  }
}

// Each row of S (n scores) to its softmax at `scale`.
[[kernel]] void s5v_softmax(device float* S [[buffer(0)]], constant uint& n [[buffer(1)]],
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
  const float inv = 1.0f / s5v_group_sum(z, part, t, tn, lane, sg);
  for (uint i = t; i < n; i += tn) s[i] *= inv;
}

// O [16, rows, 96] fp32 to out rows r0.. of [P, 1536] bf16.
[[kernel]] void s5v_heads(const device float* O [[buffer(0)]], device bfloat* out [[buffer(1)]],
                          constant uint2& p [[buffer(2)]], uint2 at [[thread_position_in_grid]]) {
  const uint rows = p.x, r0 = p.y, h = at.x / 96, d = at.x % 96, r = at.y;
  if (at.x >= 1536 || r >= rows) return;
  out[ulong(r0 + r) * 1536 + at.x] = bfloat(O[(ulong(h) * rows + r) * 96 + d]);
}

// x * sigmoid(1.702 x) in bf16 ops, in place.
[[kernel]] void s5v_quick_gelu(device bfloat* x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  const float v = float(x[i]);
  const float a = s5v_bf(1.702f * v);
  const float sg = s5v_bf(1.0f / (1.0f + metal::precise::exp(-a)));
  x[i] = bfloat(v * sg);
}

// A 3x3 stride-2 pad-1 convolution's input windows: x [gh * gw, C] to cols [oh * ow, 9 C], (ky, kx, c), zeros past.
[[kernel]] void s5v_im2col(const device bfloat* x [[buffer(0)]], device bfloat* cols [[buffer(1)]],
                           constant uint4& p [[buffer(2)]], uint2 at [[thread_position_in_grid]]) {
  const uint gh = p.x, gw = p.y, C = p.z, ow = (gw - 1) / 2 + 1, oh = (gh - 1) / 2 + 1;
  const uint o = at.y, k = at.x;
  if (o >= oh * ow || k >= 9 * C) return;
  const uint oy = o / ow, ox = o % ow, tap = k / C, c = k % C;
  const int y = int(oy * 2 + tap / 3) - 1, xx = int(ox * 2 + tap % 3) - 1;
  const bool in = y >= 0 && y < int(gh) && xx >= 0 && xx < int(gw);
  cols[ulong(o) * 9 * C + k] = in ? x[(ulong(y) * gw + xx) * C + c] : bfloat(0.0f);
}
