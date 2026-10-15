// Step-5's glue at StepFun's rounding points (tools/zig/step5_ref.py): fp32 residual, zero-centred norms, bf16 tables.
#include <metal_stdlib>
using namespace metal;

struct s5_args { uint rows, width, heads, dim, span, pos, stride, bits, group; float eps, theta, limit; uint ring; };

// Row r's slot: (pos + r) % ring in a ring, else r.
inline size_t s5_slot(constant s5_args& a, uint row) { return a.ring ? (a.pos + row) % a.ring : row; }

// Row r of residual R (fp32) = dequantized affine embedding row ids[r]: bf16(scale * q + bias).
[[kernel]] void s5_embed(const device uint* ids [[buffer(0)]], const device uint* W [[buffer(1)]],
    const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]], constant s5_args& a [[buffer(4)]],
    device float* R [[buffer(5)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t row = ids[p.y];
  const uint per = 32 / a.bits, mask = (1u << a.bits) - 1;
  const uint word = W[row * (a.width / per) + p.x / per];
  const float q = float((word >> ((p.x % per) * a.bits)) & mask);
  const size_t g = row * (a.width / a.group) + p.x / a.group;
  R[(size_t)p.y * a.width + p.x] = float(bfloat(float(S[g]) * q + float(B[g])));
}

// out = bf16(x / rms(x) * (1 + w)) of fp32 rows R; with round_in, x is R rounded to bf16 first (the final norm).
template <bool ROUND_IN>
inline void s5_norm_row(const device float* R, const device float* w, device bfloat* out, constant s5_args& a,
    uint row, uint t, uint n, threadgroup float* part) {
  const device float* x = R + (size_t)row * a.width;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += n) {
    const float v = ROUND_IN ? float(bfloat(x[i])) : x[i];
    sum = fma(v, v, sum);
  }
  sum = simd_sum(sum);
  if (t % 32 == 0) part[t / 32] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < n / 32; ++s) total += part[s];
  const float inv = rsqrt(total / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += n) {
    const float v = ROUND_IN ? float(bfloat(x[i])) : x[i];
    out[(size_t)row * (a.stride ? a.stride : a.width) + i] = bfloat(v * inv * (w[i] + 1.0f));
  }
}

[[kernel]] void s5_norm(const device float* R [[buffer(0)]], const device float* w [[buffer(1)]],
    constant s5_args& a [[buffer(2)]], device bfloat* out [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  s5_norm_row<false>(R, w, out, a, row, t, n, part);
}

[[kernel]] void s5_final_norm(const device float* R [[buffer(0)]], const device float* w [[buffer(1)]],
    constant s5_args& a [[buffer(2)]], device bfloat* out [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  s5_norm_row<true>(R, w, out, a, row, t, n, part);
}

// A simdgroup a (row, head): zero-centred head norm to bf16, then NEOX rotation of the span by bf16 cos/sin, in fp32.
[[kernel]] void s5_qk(const device bfloat* X [[buffer(0)]], const device float* w [[buffer(1)]],
    constant s5_args& a [[buffer(2)]], device bfloat* Y [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, row = tg.y;
  if (head >= a.heads || row >= a.rows) return;
  const device bfloat* x = X + ((size_t)row * a.heads + head) * a.dim;
  const uint per = a.dim / 32;
  float v[8];
  float sum = 0.0f;
  for (uint j = 0; j < per; ++j) {
    v[j] = float(x[lane + 32 * j]);
    sum = fma(v[j], v[j], sum);
  }
  const float inv = rsqrt(simd_sum(sum) / float(a.dim) + a.eps);
  threadgroup float n[256];
  for (uint j = 0; j < per; ++j) n[lane + 32 * j] = float(bfloat(v[j] * inv * (w[lane + 32 * j] + 1.0f)));
  simdgroup_barrier(mem_flags::mem_threadgroup);
  device bfloat* y = Y + s5_slot(a, row) * a.stride + (size_t)head * a.dim;
  const uint half_span = a.span / 2;
  const float position = float(a.pos + row);
  for (uint j = 0; j < per; ++j) {
    const uint d = lane + 32 * j;
    if (d >= a.span) { y[d] = bfloat(n[d]); continue; }
    const uint i = d % half_span;
    const float freq = 1.0f / precise::pow(a.theta, float(2 * i) / float(a.span));
    const float angle = position * freq;
    const float c = float(bfloat(precise::cos(angle))), s = float(bfloat(precise::sin(angle)));
    const float re = n[i], im = n[i + half_span];
    y[d] = bfloat(d < half_span ? re * c - im * s : re * s + im * c);
  }
}

// Y[slot(row) * stride + i] = X[row * width + i]: values into the cache at their positions.
[[kernel]] void s5_copy(const device bfloat* X [[buffer(0)]], constant s5_args& a [[buffer(1)]],
    device bfloat* Y [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x < a.width && p.y < a.rows) Y[s5_slot(a, p.y) * a.stride + p.x] = X[(size_t)p.y * a.width + p.x];
}

// O[row, head, :] *= bf16(sigmoid(G[row, head])), in bf16 as the reference multiplies.
[[kernel]] void s5_gate(device bfloat* O [[buffer(0)]], const device bfloat* G [[buffer(1)]],
    constant s5_args& a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.heads * a.dim || p.y >= a.rows) return;
  const float g = float(G[(size_t)p.y * a.heads + p.x / a.dim]);
  const float s = float(bfloat(1.0f / (1.0f + precise::exp(-g))));
  const size_t i = (size_t)p.y * a.heads * a.dim + p.x;
  O[i] = bfloat(float(O[i]) * s);
}

// A = bf16(min(silu(gate), limit) * clamp(up, -limit, limit)) in fp32, over rows x width.
[[kernel]] void s5_swiglu(const device bfloat* Gt [[buffer(0)]], const device bfloat* U [[buffer(1)]],
    constant s5_args& a [[buffer(2)]], device bfloat* A [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t i = (size_t)p.y * a.width + p.x;
  const float g = float(Gt[i]), u = float(U[i]);
  const float silu = g / (1.0f + precise::exp(-g));
  A[i] = bfloat(min(silu, a.limit) * clamp(u, -a.limit, a.limit));
}

// R += float(X): a sublayer's bf16 output into the fp32 residual.
[[kernel]] void s5_add(device float* R [[buffer(0)]], const device bfloat* X [[buffer(1)]],
    constant s5_args& a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t i = (size_t)p.y * a.width + p.x;
  R[i] += float(X[i]);
}

// R += float(bf16(sum_k wts[r, k] * Y[r, k, :])) + float(shared): routed picks in pick order, rounded once.
[[kernel]] void s5_moe_add(device float* R [[buffer(0)]], const device bfloat* Y [[buffer(1)]],
    const device float* wts [[buffer(2)]], const device bfloat* shared [[buffer(3)]],
    constant s5_args& a [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const uint k = a.heads; // picks a row
  float sum = 0.0f;
  for (uint j = 0; j < k; ++j) sum += wts[p.y * k + j] * float(Y[((size_t)p.y * k + j) * a.width + p.x]);
  const size_t i = (size_t)p.y * a.width + p.x;
  R[i] += float(bfloat(sum)) + float(shared[i]);
}

// A chunk's a.rows picks by expert (a.width of them): offsets[e] its first pair, order the pairs, each in pick order.
[[kernel]] void s5_sort(const device int* pick [[buffer(0)]], constant s5_args& a [[buffer(1)]],
    device int* offsets [[buffer(2)]], device int* order [[buffer(3)]], uint t [[thread_position_in_threadgroup]]) {
  threadgroup int picks[4096];
  threadgroup int counts[1024];
  for (uint i = t; i < a.rows; i += 1024) picks[i] = pick[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int n = 0;
  if (t < a.width)
    for (uint i = 0; i < a.rows; ++i) n += picks[i] == int(t) ? 1 : 0;
  counts[t] = n;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    int sum = 0;
    for (uint e = 0; e < a.width; ++e) { const int c = counts[e]; counts[e] = sum; sum += c; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t >= a.width) return;
  int at = counts[t];
  offsets[t] = at;
  for (uint i = 0; i < a.rows; ++i) if (picks[i] == int(t)) order[at++] = int(i);
}

// Y[i] = X[order[i] / a.heads]: each sorted pair's input row (a.width wide).
[[kernel]] void s5_rows_by_order(const device bfloat* X [[buffer(0)]], const device int* order [[buffer(1)]],
    constant s5_args& a [[buffer(2)]], device bfloat* Y [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x < a.width && p.y < a.rows) Y[(size_t)p.y * a.width + p.x] = X[(size_t)(order[p.y] / int(a.heads)) * a.width + p.x];
}

// Y[slot(row) * stride + i] = X[row * width + i], fp32: a window's residual rows kept for the draft head.
[[kernel]] void s5_copy32(const device float* X [[buffer(0)]], constant s5_args& a [[buffer(1)]],
    device float* Y [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x < a.width && p.y < a.rows) Y[s5_slot(a, p.y) * a.stride + p.x] = X[(size_t)p.y * a.width + p.x];
}

// R = float(X): eh_proj's bf16 rows as an MTP layer's fp32 residual.
[[kernel]] void s5_widen(device float* R [[buffer(0)]], const device bfloat* X [[buffer(1)]],
    constant s5_args& a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x < a.width && p.y < a.rows) R[(size_t)p.y * a.width + p.x] = float(X[(size_t)p.y * a.width + p.x]);
}
