// Spark2.5's glue at the reference's rounding points (its PyTorch code in bf16): an fp32 residual, fp32 norms and rope.
#include <metal_stdlib>
using namespace metal;

struct sp_args { uint rows, width, heads, bits, group; float eps; };

// Row r of residual R = dequantized affine embedding row ids[r]: bf16(scale * q + bias), held in fp32.
[[kernel]] void sp_embed(const device uint* ids [[buffer(0)]], const device uint* W [[buffer(1)]],
    const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]], constant sp_args& a [[buffer(4)]],
    device float* R [[buffer(5)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t row = ids[p.y];
  const uint per = 32 / a.bits, mask = (1u << a.bits) - 1;
  const uint word = W[row * (a.width / per) + p.x / per];
  const float q = float((word >> ((p.x % per) * a.bits)) & mask);
  const size_t g = row * (a.width / a.group) + p.x / a.group;
  R[(size_t)p.y * a.width + p.x] = float(bfloat(float(S[g]) * q + float(B[g])));
}

// A row's RMS norm as the reference takes it: out = bf16(w * (x * rsqrt(mean(x^2) + eps))), all in fp32.
inline void sp_norm_row(const device float* x, const device bfloat* w, constant sp_args& a, device bfloat* out,
    float sum, uint t, uint n, threadgroup float* part) {
  sum = simd_sum(sum);
  if (t % 32 == 0) part[t / 32] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < n / 32; ++s) total += part[s];
  const float inv = precise::rsqrt(total / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += n) out[i] = bfloat(float(w[i]) * (x[i] * inv));
}

// The residual's rows normed by w into out (bf16).
[[kernel]] void sp_norm(const device float* R [[buffer(0)]], const device bfloat* w [[buffer(1)]],
    constant sp_args& a [[buffer(2)]], device bfloat* out [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const device float* x = R + (size_t)row * a.width;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += n) sum = fma(x[i], x[i], sum);
  sp_norm_row(x, w, a, out + (size_t)row * a.width, sum, t, n, part);
}

// R += X (a sublayer's bf16 output), the sum kept in fp32 as the reference's residual is.
[[kernel]] void sp_add(device float* R [[buffer(0)]], const device bfloat* X [[buffer(1)]],
    constant sp_args& a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t i = (size_t)p.y * a.width + p.x;
  R[i] = R[i] + float(X[i]);
}

// sp_add then sp_norm on a row, one threadgroup of 256.
[[kernel]] void sp_add_norm(device float* R [[buffer(0)]], const device bfloat* X [[buffer(1)]],
    const device bfloat* w [[buffer(2)]], constant sp_args& a [[buffer(3)]], device bfloat* out [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint n [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  device float* x = R + (size_t)row * a.width;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += n) {
    const float v = x[i] + float(X[(size_t)row * a.width + i]);
    x[i] = v;
    sum = fma(v, v, sum);
  }
  sp_norm_row(x, w, a, out + (size_t)row * a.width, sum, t, n, part);
}

struct sp_prep_args { uint rows, heads, kv_heads, pos, ring, rot, stride; };

// q rotated to QN, k rotated into the cache, v beside it: the first a.rot dims in fp32, pairs (d, d + rot / 2).
[[kernel]] void sp_prep(const device bfloat* X [[buffer(0)]], const device float* inv [[buffer(1)]],
    constant sp_prep_args& a [[buffer(2)]], device bfloat* QN [[buffer(3)]], device bfloat* KC [[buffer(4)]],
    device bfloat* VC [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint DH = 256, PER = DH / 32;
  const uint unit = tg.x, row = tg.y;
  if (unit >= a.heads + a.kv_heads || row >= a.rows) return;
  const bool is_q = unit < a.heads;
  const uint head = is_q ? unit : unit - a.heads;
  const device bfloat* src = X + (size_t)row * a.stride + (is_q ? 0 : a.heads * DH) + (size_t)head * DH;
  const size_t slot = a.ring ? (a.pos + row) % a.ring : a.pos + row;
  const size_t kvw = (size_t)a.kv_heads * DH;
  device bfloat* y = is_q ? QN + ((size_t)row * a.heads + head) * DH : KC + slot * kvw + (size_t)head * DH;
  const uint half_rot = a.rot / 2;
  const float position = float(a.pos + row);
  for (uint j = 0; j < PER; ++j) {
    const uint d = lane + 32 * j;
    const float x = float(src[d]);
    if (d >= a.rot) { y[d] = bfloat(x); continue; }
    const uint i = d % half_rot;
    const float angle = position * inv[i];
    const float c = precise::cos(angle), s = precise::sin(angle);
    const float other = d < half_rot ? -float(src[d + half_rot]) : float(src[d - half_rot]);
    y[d] = bfloat(x * c + other * s);
  }
  if (is_q) return;
  const device bfloat* vs = X + (size_t)row * a.stride + (a.heads + a.kv_heads) * DH + (size_t)head * DH;
  device bfloat* vd = VC + slot * kvw + (size_t)head * DH;
  for (uint j = 0; j < PER; ++j) vd[lane + 32 * j] = vs[lane + 32 * j];
}

// The head gate: O = bf16(att * bf16(sigmoid(g))), g a head's column of the fused projection (rows a.width apart).
[[kernel]] void sp_gate(const device bfloat* att [[buffer(0)]], const device bfloat* G [[buffer(1)]],
    constant sp_args& a [[buffer(2)]], device bfloat* O [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
  constexpr uint DH = 256;
  if (p.x >= a.heads * DH || p.y >= a.rows) return;
  const float g = float(G[(size_t)p.y * a.width + p.x / DH]);
  const float s = float(bfloat(1.0f / (1.0f + precise::exp(-g))));
  const size_t i = (size_t)p.y * a.heads * DH + p.x;
  O[i] = bfloat(float(att[i]) * s);
}

// erf to under one ulp (Juffa's faithfully rounded erff, as MLX's and CUDA's erff compute it).
inline float sp_erf(float x) {
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

// PyTorch's exact GELU on a stacked gate|up row (gate at 0, up at a.width): A = bf16(bf16(gelu(g)) * u).
[[kernel]] void sp_gelu_gu(const device bfloat* GU [[buffer(0)]], constant sp_args& a [[buffer(1)]],
    device bfloat* A [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t at = (size_t)p.y * 2 * a.width + p.x;
  const float g = float(GU[at]), u = float(GU[at + a.width]);
  const float gelu = float(bfloat(g * 0.5f * (1.0f + sp_erf(g * 0.70710678118654752f))));
  A[(size_t)p.y * a.width + p.x] = bfloat(gelu * u);
}

#define TF_UNROLL_4 _Pragma("clang loop unroll(full)")

struct sp_vec_args { uint rows, pos, heads, groups, ring, window, blocks, chunk, first; float scale; };

// Decode attention partials (sdpa_vector lanes, eight dims a lane): a (key head, row, key block) a threadgroup.
[[kernel]] void sp_decode_vec(const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]], constant sp_vec_args& a [[buffer(3)]], device float* PM [[buffer(4)]],
    device float* PL [[buffer(5)]], device float* PO [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint g [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint DH = 256;
  const uint kv = tg.x, row = tg.y, block = a.first + tg.z;
  const uint G = a.heads / a.groups, head = kv * G + g;
  const uint p = a.pos + row;
  const uint lo = a.window > 0 && p + 1 > a.window ? p + 1 - a.window : 0;
  const uint from = max(lo, block * a.chunk), to = min(p + 1, (block + 1) * a.chunk);
  const size_t ld = (size_t)a.groups * DH;
  const device bfloat* qp = Q + ((size_t)row * a.heads + head) * DH + 8 * lane;
  const float4 q0 = a.scale * float4(*(const device bfloat4*)qp), q1 = a.scale * float4(*(const device bfloat4*)(qp + 4));
  float4 acc0 = 0.0f, acc1 = 0.0f;
  float top = -INFINITY, mass = 0.0f;
  for (uint i0 = from; i0 < to; i0 += 4) {
    float score[4];
    float4 v0[4], v1[4];
    TF_UNROLL_4
    for (uint u = 0; u < 4; ++u) {
      const bool live = i0 + u < to; // past the block's end: a masked key (read at the block's first, weight 0)
      const uint i = live ? i0 + u : from;
      const size_t at = (size_t)(a.ring ? i % a.ring : i) * ld + kv * DH + 8 * lane;
      v0[u] = float4(*(const device bfloat4*)(V + at));
      v1[u] = float4(*(const device bfloat4*)(V + at + 4));
      const float sc = simd_sum(dot(q0, float4(*(const device bfloat4*)(K + at))) + dot(q1, float4(*(const device bfloat4*)(K + at + 4))));
      score[u] = live ? sc : -INFINITY;
    }
    const float peak = max(max(top, max(score[0], score[1])), max(score[2], score[3]));
    const float rescale = fast::exp(top - peak);
    mass *= rescale;
    acc0 *= rescale;
    acc1 *= rescale;
    TF_UNROLL_4
    for (uint u = 0; u < 4; ++u) {
      const float w = fast::exp(score[u] - peak);
      mass += w;
      acc0 += w * v0[u];
      acc1 += w * v1[u];
    }
    top = peak;
  }
  const size_t slot = ((size_t)head * a.rows + row) * a.blocks + tg.z;
  if (lane == 0) { PM[slot] = top; PL[slot] = mass; }
  *(device float4*)(PO + slot * DH + 8 * lane) = acc0;
  *(device float4*)(PO + slot * DH + 8 * lane + 4) = acc1;
}

// MLX's sdpa_vector_2pass_2: 32 simdgroups a (head, row) fold the blocks, eight output dims a simdgroup.
[[kernel]] void sp_merge_vec(const device float* PM [[buffer(0)]], const device float* PL [[buffer(1)]],
    const device float* PO [[buffer(2)]], constant sp_vec_args& a [[buffer(3)]], device bfloat* O [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint DH = 256, PER = DH / 32;
  threadgroup float merged[32 * 32];
  const uint head = tg.x, row = tg.y;
  const size_t at = (size_t)head * a.rows + row;
  const device float* ms = PM + at * a.blocks;
  const device float* ls = PL + at * a.blocks;
  // lanes and simdgroups take blocks by absolute index (first + b): a row's sums never move with the window's first row
  const uint lane0 = (lane + 32 - a.first % 32) % 32, sg0 = (sg + 32 - a.first % 32) % 32;
  float top = -INFINITY;
  for (uint b = lane0; b < a.blocks; b += 32) top = max(top, ms[b]);
  top = simd_max(top);
  float total = 0.0f;
  for (uint b = lane0; b < a.blocks; b += 32) total += ls[b] > 0.0f ? ls[b] * fast::exp(ms[b] - top) : 0.0f;
  total = simd_sum(total);
  float acc[PER];
  for (uint j = 0; j < PER; ++j) acc[j] = 0.0f;
  for (uint b = sg0; b < a.blocks; b += 32) {
    if (ls[b] <= 0.0f) continue;
    const float f = fast::exp(ms[b] - top);
    for (uint j = 0; j < PER; ++j) acc[j] += f * PO[(at * a.blocks + b) * DH + PER * lane + j];
  }
  for (uint j = 0; j < PER; ++j) {
    merged[lane * 32 + sg] = acc[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc[j] = simd_sum(merged[sg * 32 + lane]) / total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0)
    for (uint j = 0; j < PER; ++j) O[(row * a.heads + head) * DH + PER * sg + j] = bfloat(acc[j]);
}

inline bool sp_better(float a, uint ia, float b, uint ib) {
  if (isnan(a)) return !isnan(b) || ia < ib;
  if (isnan(b)) return false;
  return a > b || (a == b && ia < ib);
}

// The greedy draw (nemotron_glue's tf_argmax_bf16): a bf16 logits row's argmax, ties to the lower id, into ids[0].
[[kernel]] void sp_argmax(const device bfloat* logits [[buffer(0)]], device uint* ids [[buffer(1)]],
    constant uint& vocab [[buffer(2)]], uint t [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float vals[32];
  threadgroup uint idxs[32];
  float best = -INFINITY;
  uint at = 0xffffffffu;
  for (uint i = t; i < vocab; i += 1024) {
    const float v = float(logits[i]);
    if (sp_better(v, i, best, at)) { best = v; at = i; }
  }
  for (ushort d = 16; d > 0; d >>= 1) {
    const float ov = simd_shuffle_xor(best, d);
    const uint oi = simd_shuffle_xor(at, d);
    if (sp_better(ov, oi, best, at)) { best = ov; at = oi; }
  }
  if (lane == 0) { vals[sg] = best; idxs[sg] = at; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  best = vals[lane];
  at = idxs[lane];
  for (ushort d = 16; d > 0; d >>= 1) {
    const float ov = simd_shuffle_xor(best, d);
    const uint oi = simd_shuffle_xor(at, d);
    if (sp_better(ov, oi, best, at)) { best = ov; at = oi; }
  }
  if (lane == 0) ids[0] = at;
}
