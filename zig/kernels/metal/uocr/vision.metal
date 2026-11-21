// Unlimited-OCR-2's image encoder steps (SAM ViT-B, its neck, CLIP-L), rounded to bf16 where torch rounds.

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

struct UvCols {
  uint channels, in_side, out_side, span, stride, pad, cstride, pstride;
};

// Convolution rows [out_side^2, channels span^2], columns (c, ky, kx); map value (c, y, x) at c cs + (y side + x) ps.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void uv_im2col(
    const device bfloat* x [[buffer(0)]], device bfloat* y [[buffer(1)]], constant UvCols& p [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  const uint taps = p.span * p.span, width = p.channels * taps;
  if (at.x >= width || at.y >= p.out_side * p.out_side) return;
  const uint c = at.x / taps, t = at.x % taps;
  const int iy = int((at.y / p.out_side) * p.stride + t / p.span) - int(p.pad);
  const int ix = int((at.y % p.out_side) * p.stride + t % p.span) - int(p.pad);
  const bool inside = iy >= 0 && ix >= 0 && iy < int(p.in_side) && ix < int(p.in_side);
  y[ulong(at.y) * width + at.x] =
      inside ? x[ulong(c) * p.cstride + (ulong(iy) * p.in_side + uint(ix)) * p.pstride] : bfloat(0.0f);
}

struct UvWindow {
  uint side, padded, window, channels;
};

// The map's rows [side^2, channels] as the padded map's window x window squares, both in row order; padding rows are 0.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void uv_window(
    const device bfloat* x [[buffer(0)]], device bfloat* y [[buffer(1)]], constant UvWindow& p [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.channels || at.y >= p.padded * p.padded) return;
  const uint area = p.window * p.window, across = p.padded / p.window;
  const uint w = at.y / area, i = at.y % area;
  const uint r = (w / across) * p.window + i / p.window, c = (w % across) * p.window + i % p.window;
  y[ulong(at.y) * p.channels + at.x] = r < p.side && c < p.side ? x[(ulong(r) * p.side + c) * p.channels + at.x] : bfloat(0.0f);
}

// The windows' rows back onto the map, added to it in place: x = bf16(x + window row); padding rows are dropped.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void uv_unwindow_add(
    device bfloat* x [[buffer(0)]], const device bfloat* win [[buffer(1)]], constant UvWindow& p [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.channels || at.y >= p.side * p.side) return;
  const uint r = at.y / p.side, c = at.y % p.side, across = p.padded / p.window;
  const uint w = (r / p.window) * across + c / p.window, i = (r % p.window) * p.window + c % p.window;
  const ulong to = ulong(at.y) * p.channels + at.x;
  x[to] = bfloat(float(x[to]) + float(win[(ulong(w) * p.window * p.window + i) * p.channels + at.x]));
}

struct UvRel {
  uint groups, side, heads, width;
};

// Relative-position terms: for query i at (a, b) of its group, rel_h[h, i, j] = bf16(q . Rh[a - j + side - 1]); w by b.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void uv_rel_pos(
    const device bfloat* qkv [[buffer(0)]], const device bfloat* rh [[buffer(1)]], const device bfloat* rw [[buffer(2)]],
    device bfloat* out_h [[buffer(3)]], device bfloat* out_w [[buffer(4)]], constant UvRel& p [[buffer(5)]],
    uint3 at [[thread_position_in_grid]]) {
  const uint len = p.side * p.side, rows = p.groups * len;
  if (at.x >= 2 * p.side || at.y >= rows || at.z >= p.heads) return;
  const bool across = at.x >= p.side;
  const uint j = across ? at.x - p.side : at.x, i = at.y % len;
  const uint from = (across ? i % p.side : i / p.side) + p.side - 1 - j;
  const device bfloat* q = qkv + ulong(at.y) * 3 * p.width + at.z * 64;
  const device bfloat* r = (across ? rw : rh) + ulong(from) * 64;
  float s = 0.0f;
  for (uint c = 0; c < 64; c++) s = fma(float(q[c]), float(r[c]), s);
  (across ? out_w : out_h)[(ulong(at.z) * rows + at.y) * p.side + j] = bfloat(s);
}

struct UvAttn {
  uint groups, len, heads, width, side;
  float scale;
};

constant constexpr uint uv_q_rows = 8;    // query rows a simdgroup
constant constexpr uint uv_keys = 32;     // keys a step
constant constexpr uint uv_head = 64;     // head width
constant constexpr uint uv_block = 512;   // keys a softmax block: its own running top score
constant constexpr uint uv_blocks = 8;    // blocks a row at most (4,096 keys)

// One step's scores of this simdgroup's 8 query rows against the 32 keys in `ks`, into `stage` [8, 32] (fp32 sums).
inline void uv_scores(thread simdgroup_float8x8* qf, threadgroup const float* ks, threadgroup float* stage) {
  for (uint kc = 0; kc < uv_keys / 8; kc++) {
    simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint dc = 0; dc < uv_head / 8; dc++) {
      simdgroup_float8x8 kt;
      simdgroup_load(kt, ks + kc * 8 * uv_head + dc * 8, uv_head, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(s, qf[dc], kt, s);
    }
    simdgroup_store(s, stage + kc * 8, uv_keys);
  }
}

// A score as the softmax sees it: scale * (q . k) plus its query's two relative-position terms summed in bf16.
inline float uv_logit(float dot, uint key, uint row, uint head, constant UvAttn& p, const device bfloat* rel_h,
                      const device bfloat* rel_w) {
  float s = p.scale * dot;
  if (p.side > 0) {
    const ulong at = (ulong(head) * p.groups * p.len + row) * p.side;
    s += float(bfloat(float(rel_h[at + key / p.side]) + float(rel_w[at + key % p.side])));
  }
  return s;
}

// Attention in groups of len rows, 512-key blocks: fp32 weights on each block's running top, bf16 ones times values.
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void uv_attention(
    const device bfloat* qkv [[buffer(0)]], const device bfloat* rel_h [[buffer(1)]],
    const device bfloat* rel_w [[buffer(2)]], device bfloat* o [[buffer(3)]], constant UvAttn& p [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float ks[uv_keys * uv_head];
  threadgroup float vs[uv_keys * uv_head];
  threadgroup float stages[4 * uv_q_rows * uv_head];
  threadgroup float* stage = stages + sg * uv_q_rows * uv_head;
  const uint head = tg.y, base = tg.z * p.len, stride = 3 * p.width;
  const uint q0 = tg.x * 4 * uv_q_rows + sg * uv_q_rows;
  // this simdgroup's queries, rows past the group as zeros
  for (uint e = lane; e < uv_q_rows * uv_head; e += 32) {
    const uint r = q0 + e / uv_head;
    stage[e] = r < p.len ? float(qkv[ulong(base + r) * stride + head * uv_head + e % uv_head]) : 0.0f;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 qf[uv_head / 8];
  for (uint dc = 0; dc < uv_head / 8; dc++) simdgroup_load(qf[dc], stage + dc * 8, uv_head);
  // lane: row lane / 4 of the 8, keys (lane % 4) * 8 .. + 8 of each step
  const uint my = lane / 4, part = lane % 4, qr = min(q0 + my, p.len - 1), row = base + qr;
  float top[uv_blocks];
  for (uint b = 0; b < uv_blocks; b++) top[b] = -INFINITY;
  for (uint k0 = 0; k0 < p.len; k0 += uv_keys) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = t; e < uv_keys * uv_head; e += 128) {
      const uint key = k0 + e / uv_head;
      ks[e] = key < p.len ? float(qkv[ulong(base + key) * stride + p.width + head * uv_head + e % uv_head]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uv_scores(qf, ks, stage);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < 8; j++) {
      const uint key = k0 + part * 8 + j;
      if (key < p.len) top[k0 / uv_block] = max(top[k0 / uv_block], uv_logit(stage[my * uv_keys + part * 8 + j], key, row, head, p, rel_h, rel_w));
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint b = 0; b < uv_blocks; b++) {
    top[b] = max(top[b], simd_shuffle_xor(top[b], 1));
    top[b] = max(top[b], simd_shuffle_xor(top[b], 2));
    if (b > 0) top[b] = max(top[b], top[b - 1]);
  }
  simdgroup_float8x8 acc[uv_head / 8];
  for (uint dc = 0; dc < uv_head / 8; dc++) acc[dc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  float mass = 0.0f;
  for (uint k0 = 0; k0 < p.len; k0 += uv_keys) {
    const uint b = k0 / uv_block;
    if (b > 0 && k0 % uv_block == 0) {
      // a new block: the row's sums so far times exp(previous top - this top), as a diagonal product
      const float c = metal::precise::exp(top[b - 1] - top[b]);
      mass *= c;
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < 64; e += 32) stage[e] = 0.0f;
      simdgroup_barrier(mem_flags::mem_threadgroup);
      if (part == 0) stage[my * 9] = c;
      simdgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_float8x8 d;
      simdgroup_load(d, stage, 8);
      for (uint dc = 0; dc < uv_head / 8; dc++) simdgroup_multiply(acc[dc], d, acc[dc]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = t; e < uv_keys * uv_head; e += 128) {
      const uint key = k0 + e / uv_head;
      const ulong at = ulong(base + key) * stride + head * uv_head + e % uv_head;
      ks[e] = key < p.len ? float(qkv[at + p.width]) : 0.0f;
      vs[e] = key < p.len ? float(qkv[at + 2 * p.width]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uv_scores(qf, ks, stage);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < 8; j++) {
      const uint key = k0 + part * 8 + j, at = my * uv_keys + part * 8 + j;
      const float w = key < p.len ? metal::precise::exp(uv_logit(stage[at], key, row, head, p, rel_h, rel_w) - top[b]) : 0.0f;
      mass += w;
      stage[at] = float(bfloat(w));
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kc = 0; kc < uv_keys / 8; kc++) {
      simdgroup_float8x8 pf;
      simdgroup_load(pf, stage + kc * 8, uv_keys);
      for (uint dc = 0; dc < uv_head / 8; dc++) {
        simdgroup_float8x8 vf;
        simdgroup_load(vf, vs + kc * 8 * uv_head + dc * 8, uv_head);
        simdgroup_multiply_accumulate(acc[dc], pf, vf, acc[dc]);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  mass += simd_shuffle_xor(mass, 1);
  mass += simd_shuffle_xor(mass, 2);
  for (uint dc = 0; dc < uv_head / 8; dc++) simdgroup_store(acc[dc], stage + dc * 8, uv_head);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (q0 + my >= p.len) return;
  device bfloat* out = o + ulong(row) * p.width + head * uv_head;
  for (uint c = part * 16; c < part * 16 + 16; c++) out[c] = bfloat(stage[my * uv_head + c] / mass);
}
