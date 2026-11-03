// Unlimited-OCR-2's image encoder steps (SAM ViT-B, its neck, CLIP-L), each rounded to bf16 where the publisher's
// torch code rounds; the projections run on core_qwen_vision's tensor-unit GEMM beside these.

#include <metal_stdlib>
using namespace metal;

inline float uv_bf(float x) { return float(bfloat(x)); }

// One threadgroup's sum of every thread's `v` (at most 1,024 threads), returned to all of them.
inline float uv_sum(float v, threadgroup float* part, uint n, uint lane, uint sg) {
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

struct UvConv {
  uint in_h, in_w, channels, k, stride, pad, out_h, out_w, chw;
};

// Convolution patches: row (oy, ox) of [out_h * out_w, channels * k * k] in the weight's (c, ky, kx) order, zero
// outside the input. The input is [channels, h, w] (pixels, chw = 1) or [h, w, channels] (feature maps).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void uv_im2col(
    const device bfloat* x [[buffer(0)]], device bfloat* y [[buffer(1)]], constant UvConv& p [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  const uint width = p.channels * p.k * p.k;
  if (at.x >= width || at.y >= p.out_h * p.out_w) return;
  const uint c = at.x / (p.k * p.k), ky = (at.x / p.k) % p.k, kx = at.x % p.k;
  const int iy = int((at.y / p.out_w) * p.stride + ky) - int(p.pad);
  const int ix = int((at.y % p.out_w) * p.stride + kx) - int(p.pad);
  bfloat v = bfloat(0.0f);
  if (iy >= 0 && ix >= 0 && iy < int(p.in_h) && ix < int(p.in_w)) {
    v = p.chw ? x[(ulong(c) * p.in_h + iy) * p.in_w + ix] : x[(ulong(iy) * p.in_w + ix) * p.channels + c];
  }
  y[ulong(at.y) * width + at.x] = v;
}

// y = bf16(a + b) over `n` values (torch's bf16 add: both widened, one rounding).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void uv_add(
    const device bfloat* a [[buffer(0)]], const device bfloat* b [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i < n) y[i] = bfloat(float(a[i]) + float(b[i]));
}

// y [rows, width] = bf16(c + b): fp32 sums and a bias vector, rounded once (a Linear's bias in its sums).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void uv_bias(
    const device float* c [[buffer(0)]], const device bfloat* b [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant uint& width [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= width) return;
  const ulong i = ulong(at.y) * width + at.x;
  y[i] = bfloat(c[i] + float(b[at.x]));
}

struct UvNorm {
  uint width;
  float eps;
};

// nn.LayerNorm on bf16 rows: the mean and variance of the widened row, the affine in fp32, one rounding.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void uv_layer_norm(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], const device bfloat* b [[buffer(2)]],
    device bfloat* y [[buffer(3)]], constant UvNorm& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device bfloat* xr = x + ulong(row) * p.width;
  device bfloat* yr = y + ulong(row) * p.width;
  float s = 0.0f;
  for (uint i = t; i < p.width; i += n) s += float(xr[i]);
  const float mean = uv_sum(s, part, n, lane, sg) / float(p.width);
  float v = 0.0f;
  for (uint i = t; i < p.width; i += n) {
    const float d = float(xr[i]) - mean;
    v += d * d;
  }
  const float inv = metal::precise::rsqrt(uv_sum(v, part, n, lane, sg) / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += n) yr[i] = bfloat((float(xr[i]) - mean) * inv * float(w[i]) + float(b[i]));
}

// SAM's LayerNorm2d over a position's channels, op by op in bf16 as its module computes it in the input's dtype:
// u = mean(x); s = mean((x - u)^2); x = (x - u) / sqrt(s + eps); w * x + b, every step rounded.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void uv_layer_norm2d(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], const device bfloat* b [[buffer(2)]],
    device bfloat* y [[buffer(3)]], constant UvNorm& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device bfloat* xr = x + ulong(row) * p.width;
  device bfloat* yr = y + ulong(row) * p.width;
  float s = 0.0f;
  for (uint i = t; i < p.width; i += n) s += float(xr[i]);
  const float u = uv_bf(uv_sum(s, part, n, lane, sg) / float(p.width));
  float v = 0.0f;
  for (uint i = t; i < p.width; i += n) {
    const float d = uv_bf(float(xr[i]) - u);
    v += uv_bf(d * d);
  }
  const float var = uv_bf(uv_sum(v, part, n, lane, sg) / float(p.width));
  const float den = uv_bf(metal::precise::sqrt(uv_bf(var + p.eps)));
  for (uint i = t; i < p.width; i += n) {
    const float d = uv_bf(uv_bf(float(xr[i]) - u) / den);
    yr[i] = bfloat(uv_bf(float(w[i]) * d) + float(b[i]));
  }
}

// erf to under one ulp (Juffa's faithfully rounded erff, as spark25/glue.metal's sp_erf).
inline float uv_erf(float x) {
  const float t = fabs(x), s = x * x;
  float r;
  if (t > 0.927734375f) {
    r = fma(-1.72853470e-5f, t, 3.83197126e-4f);
    const float u = fma(-3.88396438e-3f, t, 2.42546219e-2f);
    r = fma(r, s, u);
    r = fma(r, t, -1.06777877e-1f);
    r = fma(r, t, -6.34846687e-1f);
    r = fma(r, t, -1.28717512e-1f);
    r = fma(r, t, -t);
    r = copysign(1.0f - precise::exp(r), x);
  } else {
    r = -5.96761703e-4f;
    r = fma(r, s, 4.99119423e-3f);
    r = fma(r, s, -2.67681349e-2f);
    r = fma(r, s, 1.12819925e-1f);
    r = fma(r, s, -3.76125336e-1f);
    r = fma(r, s, 1.28379166e-1f);
    r = fma(r, x, x);
  }
  return r;
}

// nn.GELU (erf) on bf16 values, in fp32, one rounding.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void uv_gelu(
    device bfloat* x [[buffer(0)]], constant uint& n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float v = float(x[i]);
  x[i] = bfloat(0.5f * v * (1.0f + uv_erf(v * 0.70710678118654752f)));
}

// CLIP's quick_gelu as its module runs it on bf16: bf16(x * bf16(sigmoid(bf16(1.702 * x)))).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void uv_quick_gelu(
    device bfloat* x [[buffer(0)]], constant uint& n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float v = float(x[i]);
  const float z = uv_bf(1.702f * v);
  const float s = uv_bf(1.0f / (1.0f + metal::precise::exp(-z)));
  x[i] = bfloat(v * s);
}

struct UvWindow {
  uint side, padded, window, channels;
};

// SAM's window partition: [side, side, C] zero-padded to `padded`, as [windows, window^2, C] row-major windows.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void uv_window(
    const device bfloat* x [[buffer(0)]], device bfloat* y [[buffer(1)]], constant UvWindow& p [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.channels || at.y >= p.padded * p.padded) return;
  const uint per = p.padded / p.window, w2 = p.window * p.window;
  const uint win = at.y / w2, in = at.y % w2;
  const uint gy = (win / per) * p.window + in / p.window, gx = (win % per) * p.window + in % p.window;
  y[ulong(at.y) * p.channels + at.x] = (gy < p.side && gx < p.side) ? x[(ulong(gy) * p.side + gx) * p.channels + at.x] : bfloat(0.0f);
}

// The partition undone and cropped, then the residual: x[side, side, C] = bf16(x + windows' row).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void uv_unwindow_add(
    device bfloat* x [[buffer(0)]], const device bfloat* win [[buffer(1)]], constant UvWindow& p [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.channels || at.y >= p.side * p.side) return;
  const uint gy = at.y / p.side, gx = at.y % p.side;
  const uint per = p.padded / p.window;
  const uint w = (gy / p.window) * per + gx / p.window, in = (gy % p.window) * p.window + gx % p.window;
  const ulong i = ulong(at.y) * p.channels + at.x;
  x[i] = bfloat(float(x[i]) + float(win[(ulong(w) * p.window * p.window + in) * p.channels + at.x]));
}

struct UvRel {
  uint groups, side, heads, width;
};

// SAM's decomposed relative positions: rel_h[h, g, q, kh] = q . Rh[qh - kh + side - 1] and rel_w likewise over kw,
// with q unscaled from the qkv rows [groups x side^2, 3 x width], each sum rounded to bf16 (the einsums' outputs).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void uv_rel_pos(
    const device bfloat* qkv [[buffer(0)]], const device bfloat* tab_h [[buffer(1)]],
    const device bfloat* tab_w [[buffer(2)]], device bfloat* rel_h [[buffer(3)]], device bfloat* rel_w [[buffer(4)]],
    constant UvRel& p [[buffer(5)]], uint3 at [[thread_position_in_grid]]) {
  const uint L = p.side * p.side;
  if (at.x >= 2 * p.side || at.y >= p.groups * L || at.z >= p.heads) return;
  const uint g = at.y / L, qi = at.y % L, qh = qi / p.side, qw = qi % p.side;
  const bool rows = at.x < p.side;
  const uint kk = rows ? at.x : at.x - p.side;
  const device bfloat* q = qkv + ulong(at.y) * 3 * p.width + at.z * 64;
  const device bfloat* r = (rows ? tab_h : tab_w) + ulong((rows ? qh : qw) - kk + p.side - 1) * 64;
  float s = 0.0f;
  for (uint d = 0; d < 64; d++) s += float(q[d]) * float(r[d]);
  device bfloat* out = rows ? rel_h : rel_w;
  out[((ulong(at.z) * p.groups + g) * L + qi) * p.side + kk] = bfloat(s);
}

struct UvAttn {
  uint groups, len, heads, width;  // groups of `len` rows; qkv rows [groups x len, 3 x width], heads of 64
  uint side;                       // the relative positions' side (0: no bias)
  float scale;
};

// o [groups x len, width] = softmax(q k^T scale + bias) v within each group, a simdgroup a query row and head:
// fp32 scores, the bias bf16(rel_h + rel_w) as the module's bf16 mask, exp weights rounded to bf16 before p v.
[[max_total_threads_per_threadgroup(32)]] [[kernel]] void uv_attention(
    const device bfloat* qkv [[buffer(0)]], const device bfloat* rel_h [[buffer(1)]],
    const device bfloat* rel_w [[buffer(2)]], device bfloat* o [[buffer(3)]], constant UvAttn& p [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float score[4096];
  const uint row = tg.x, h = tg.y;
  const uint g = row / p.len, qi = row % p.len;
  const ulong stride = 3 * ulong(p.width);
  const device bfloat* q = qkv + ulong(row) * stride + h * 64;
  const device bfloat* kbase = qkv + ulong(g) * p.len * stride + p.width + h * 64;
  const device bfloat* vbase = kbase + p.width;
  float qv[64];
  for (uint d = 0; d < 64; d++) qv[d] = float(q[d]);
  const ulong rel = ((ulong(h) * p.groups + g) * p.len + qi) * max(p.side, 1u);
  float m = -INFINITY;
  for (uint j = lane; j < p.len; j += 32) {
    const device bfloat* k = kbase + ulong(j) * stride;
    float s = 0.0f;
    for (uint d = 0; d < 64; d++) s += qv[d] * float(k[d]);
    s *= p.scale;
    if (p.side > 0) s += uv_bf(float(rel_h[rel + j / p.side]) + float(rel_w[rel + j % p.side]));
    score[j] = s;
    m = max(m, s);
  }
  m = simd_max(m);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum = 0.0f;
  for (uint j = lane; j < p.len; j += 32) {
    const float e = metal::precise::exp(score[j] - m);
    sum += e;
    score[j] = uv_bf(e);
  }
  sum = simd_sum(sum);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float a0 = 0.0f, a1 = 0.0f;
  for (uint j = 0; j < p.len; j++) {
    const device bfloat* v = vbase + ulong(j) * stride;
    a0 += score[j] * float(v[lane]);
    a1 += score[j] * float(v[lane + 32]);
  }
  device bfloat* out = o + ulong(row) * p.width + h * 64;
  out[lane] = bfloat(a0 / sum);
  out[lane + 32] = bfloat(a1 / sum);
}

struct UvCat {
  uint rows, left, right;
};

// y [rows, left + right] = [a row | b row]: CLIP's rows (past its class row) beside SAM's.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void uv_concat(
    const device bfloat* a [[buffer(0)]], const device bfloat* b [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant UvCat& p [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  const uint width = p.left + p.right;
  if (at.x >= width || at.y >= p.rows) return;
  y[ulong(at.y) * width + at.x] = at.x < p.left ? a[ulong(at.y) * p.left + at.x] : b[ulong(at.y) * p.right + at.x - p.left];
}

// CLIP's embeddings on SAM's map: y[0] = bf16(cls + pos[0]), y[r] = bf16(sam[r - 1] + pos[r]); rows of `width`.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void uv_clip_embed(
    const device bfloat* cls [[buffer(0)]], const device bfloat* sam [[buffer(1)]], const device bfloat* pos [[buffer(2)]],
    device bfloat* y [[buffer(3)]], constant uint& width [[buffer(4)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= width) return;
  const float v = at.y == 0 ? float(cls[at.x]) : float(sam[ulong(at.y - 1) * width + at.x]);
  y[ulong(at.y) * width + at.x] = bfloat(v + float(pos[ulong(at.y) * width + at.x]));
}
