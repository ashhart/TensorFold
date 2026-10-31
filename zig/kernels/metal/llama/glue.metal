// Llama's glue at HF's rounding points (modeling_llama in bf16): bf16 residual and norm, rope with bf16 cos and sin.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint DH = LL_DH; // the head dim (128), defined by the loader

struct ll_args { uint rows, width, heads; float eps; };

// Row r of residual R = bf16 embedding row ids[r] (the release's table, [vocab, width]).
[[kernel]] void ll_embed(const device uint* ids [[buffer(0)]], const device bfloat* E [[buffer(1)]],
    constant ll_args& a [[buffer(2)]], device bfloat* R [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  R[(size_t)p.y * a.width + p.x] = E[(size_t)ids[p.y] * a.width + p.x];
}

// LlamaRMSNorm on a row: out = bf16(w * bf16(x * rsqrt(mean(x^2) + eps))), the mean in fp32 over the bf16 row.
inline void ll_norm_row(const device bfloat* x, const device bfloat* w, constant ll_args& a, device bfloat* out,
    float sum, uint t, uint n, threadgroup float* part) {
  sum = simd_sum(sum);
  if (t % 32 == 0) part[t / 32] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < n / 32; ++s) total += part[s];
  const float inv = precise::rsqrt(total / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += n) out[i] = bfloat(float(w[i]) * float(bfloat(float(x[i]) * inv)));
}

// The residual's rows normed by w into out.
[[kernel]] void ll_norm(const device bfloat* R [[buffer(0)]], const device bfloat* w [[buffer(1)]],
    constant ll_args& a [[buffer(2)]], device bfloat* out [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const device bfloat* x = R + (size_t)row * a.width;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += n) {
    const float v = float(x[i]);
    sum = fma(v, v, sum);
  }
  ll_norm_row(x, w, a, out + (size_t)row * a.width, sum, t, n, part);
}

// R = bf16(R + X): the reference's bf16 residual add.
[[kernel]] void ll_add(device bfloat* R [[buffer(0)]], const device bfloat* X [[buffer(1)]],
    constant ll_args& a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t i = (size_t)p.y * a.width + p.x;
  R[i] = bfloat(float(R[i]) + float(X[i]));
}

// ll_add then ll_norm on a row, one threadgroup of 256.
[[kernel]] void ll_add_norm(device bfloat* R [[buffer(0)]], const device bfloat* X [[buffer(1)]],
    const device bfloat* w [[buffer(2)]], constant ll_args& a [[buffer(3)]], device bfloat* out [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint n [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  device bfloat* x = R + (size_t)row * a.width;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += n) {
    const bfloat v = bfloat(float(x[i]) + float(X[(size_t)row * a.width + i]));
    x[i] = v;
    const float f = float(v);
    sum = fma(f, f, sum);
  }
  ll_norm_row(x, w, a, out + (size_t)row * a.width, sum, t, n, part);
}

struct ll_prep_args { uint rows, heads, kv_heads, pos, stride; };

// HF's apply_rotary_pos_emb, pairs (d, d + DH / 2): bf16(bf16(x * c) + bf16(o * s)), c and s the angle's cos and sin
[[kernel]] void ll_prep(const device bfloat* X [[buffer(0)]], const device float* inv [[buffer(1)]],
    constant ll_prep_args& a [[buffer(2)]], device bfloat* QN [[buffer(3)]], device bfloat* KC [[buffer(4)]],
    device bfloat* VC [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  // in bf16 as HF rounds them; q rotated to QN, k rotated into the cache at its position, v beside it
  constexpr uint PER = DH / 32, HALF = DH / 2;
  const uint unit = tg.x, row = tg.y;
  if (unit >= a.heads + a.kv_heads || row >= a.rows) return;
  const bool is_q = unit < a.heads;
  const uint head = is_q ? unit : unit - a.heads;
  const device bfloat* src = X + (size_t)row * a.stride + (is_q ? 0 : a.heads * DH) + (size_t)head * DH;
  const size_t slot = a.pos + row;
  const size_t kvw = (size_t)a.kv_heads * DH;
  device bfloat* y = is_q ? QN + ((size_t)row * a.heads + head) * DH : KC + slot * kvw + (size_t)head * DH;
  const float position = float(a.pos + row);
  for (uint j = 0; j < PER; ++j) {
    const uint d = lane + 32 * j;
    const uint i = d % HALF;
    const float angle = position * inv[i];
    const float c = float(bfloat(precise::cos(angle))), s = float(bfloat(precise::sin(angle)));
    const float x = float(src[d]);
    const float other = d < HALF ? -float(src[d + HALF]) : float(src[d - HALF]);
    y[d] = bfloat(float(bfloat(x * c)) + float(bfloat(other * s)));
  }
  if (is_q) return;
  const device bfloat* vs = X + (size_t)row * a.stride + (a.heads + a.kv_heads) * DH + (size_t)head * DH;
  device bfloat* vd = VC + slot * kvw + (size_t)head * DH;
  for (uint j = 0; j < PER; ++j) vd[lane + 32 * j] = vs[lane + 32 * j];
}

// torch's bf16 SiLU gate on a stacked gate|up row (gate at 0, up at a.width): A = bf16(bf16(silu(g)) * u).
[[kernel]] void ll_silu_gu(const device bfloat* GU [[buffer(0)]], constant ll_args& a [[buffer(1)]],
    device bfloat* A [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t at = (size_t)p.y * 2 * a.width + p.x;
  const float g = float(GU[at]), u = float(GU[at + a.width]);
  const float silu = float(bfloat(g / (1.0f + precise::exp(-g))));
  A[(size_t)p.y * a.width + p.x] = bfloat(silu * u);
}

#define TF_UNROLL_4 _Pragma("clang loop unroll(full)")

struct ll_vec_args { uint rows, pos, heads, groups, blocks, chunk, first; float scale; };

// Decode attention partials (sdpa_vector lanes, DH / 32 dims a lane): a (key head, row, key block) a threadgroup.
[[kernel]] void ll_decode_vec(const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]], constant ll_vec_args& a [[buffer(3)]], device float* PM [[buffer(4)]],
    device float* PL [[buffer(5)]], device float* PO [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint g [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint PER = DH / 32;
  static_assert(PER == 4, "a lane holds one bfloat4 of a head");
  const uint kv = tg.x, row = tg.y, block = a.first + tg.z;
  const uint G = a.heads / a.groups, head = kv * G + g;
  const uint p = a.pos + row;
  const uint from = block * a.chunk, to = min(p + 1, (block + 1) * a.chunk);
  const size_t ld = (size_t)a.groups * DH;
  const device bfloat* qp = Q + ((size_t)row * a.heads + head) * DH + PER * lane;
  const float4 q0 = a.scale * float4(*(const device bfloat4*)qp);
  float4 acc0 = 0.0f;
  float top = -INFINITY, mass = 0.0f;
  for (uint i0 = from; i0 < to; i0 += 4) {
    float score[4];
    float4 v0[4];
    TF_UNROLL_4
    for (uint u = 0; u < 4; ++u) {
      const bool live = i0 + u < to; // past the block's end: a masked key (read at the block's first, weight 0)
      const uint i = live ? i0 + u : from;
      const size_t at = (size_t)i * ld + kv * DH + PER * lane;
      v0[u] = float4(*(const device bfloat4*)(V + at));
      const float sc = simd_sum(dot(q0, float4(*(const device bfloat4*)(K + at))));
      score[u] = live ? sc : -INFINITY;
    }
    const float peak = max(max(top, max(score[0], score[1])), max(score[2], score[3]));
    const float rescale = fast::exp(top - peak);
    mass *= rescale;
    acc0 *= rescale;
    TF_UNROLL_4
    for (uint u = 0; u < 4; ++u) {
      const float w = fast::exp(score[u] - peak);
      mass += w;
      acc0 += w * v0[u];
    }
    top = peak;
  }
  const size_t slot = ((size_t)head * a.rows + row) * a.blocks + tg.z;
  if (lane == 0) { PM[slot] = top; PL[slot] = mass; }
  *(device float4*)(PO + slot * DH + PER * lane) = acc0;
}

// MLX's sdpa_vector_2pass_2: 32 simdgroups a (head, row) fold the blocks, DH / 32 output dims a simdgroup.
[[kernel]] void ll_merge_vec(const device float* PM [[buffer(0)]], const device float* PL [[buffer(1)]],
    const device float* PO [[buffer(2)]], constant ll_vec_args& a [[buffer(3)]], device bfloat* O [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint PER = DH / 32;
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

inline bool ll_better(float a, uint ia, float b, uint ib) {
  if (isnan(a)) return !isnan(b) || ia < ib;
  if (isnan(b)) return false;
  return a > b || (a == b && ia < ib);
}

// The greedy draw (nemotron_glue's tf_argmax_bf16): a bf16 logits row's argmax, ties to the lower id, into ids[0].
[[kernel]] void ll_argmax(const device bfloat* logits [[buffer(0)]], device uint* ids [[buffer(1)]],
    constant uint& vocab [[buffer(2)]], uint t [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float vals[32];
  threadgroup uint idxs[32];
  float best = -INFINITY;
  uint at = 0xffffffffu;
  for (uint i = t; i < vocab; i += 1024) {
    const float v = float(logits[i]);
    if (ll_better(v, i, best, at)) { best = v; at = i; }
  }
  for (ushort d = 16; d > 0; d >>= 1) {
    const float ov = simd_shuffle_xor(best, d);
    const uint oi = simd_shuffle_xor(at, d);
    if (ll_better(ov, oi, best, at)) { best = ov; at = oi; }
  }
  if (lane == 0) { vals[sg] = best; idxs[sg] = at; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  best = vals[lane];
  at = idxs[lane];
  for (ushort d = 16; d > 0; d >>= 1) {
    const float ov = simd_shuffle_xor(best, d);
    const uint oi = simd_shuffle_xor(at, d);
    if (ll_better(ov, oi, best, at)) { best = ov; at = oi; }
  }
  if (lane == 0) ids[0] = at;
}

struct ll_mv_args { uint rows, n, k, ldy; };

// Y[r, j] = bf16(sum_i X[r, i] * W[j, i]) over bf16 W [n, k] in fp32 (torch's bf16 linear): a simdgroup a row j.
template <uint ROWS>
inline void ll_matvec_t(const device bfloat* X, const device bfloat* W, constant ll_mv_args& a, device bfloat* Y,
    uint3 tg, uint sg, uint lane) {
  constexpr uint SG = 8;
  const uint j = tg.x * SG + sg;
  if (j >= a.n) return;
  const device bfloat* w = W + (size_t)j * a.k;
  float acc[ROWS];
  TF_UNROLL_4
  for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0f;
  for (uint i = 8 * lane; i < a.k; i += 256) {
    const float4 w0 = float4(*(const device bfloat4*)(w + i)), w1 = float4(*(const device bfloat4*)(w + i + 4));
    TF_UNROLL_4
    for (uint r = 0; r < ROWS; ++r) {
      const device bfloat* x = X + (size_t)r * a.k + i;
      acc[r] += dot(w0, float4(*(const device bfloat4*)x)) + dot(w1, float4(*(const device bfloat4*)(x + 4)));
    }
  }
  TF_UNROLL_4
  for (uint r = 0; r < ROWS; ++r) {
    const float s = simd_sum(acc[r]);
    if (lane == 0 && r < a.rows) Y[(size_t)r * a.ldy + j] = bfloat(s);
  }
}

#define LL_MATVEC(ROWS)                                                                                            \
  [[kernel]] void ll_matvec_##ROWS(const device bfloat* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],      \
      constant ll_mv_args& a [[buffer(2)]], device bfloat* Y [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]], \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                      \
    ll_matvec_t<ROWS>(X, W, a, Y, tg, sg, lane);                                                                   \
  }
LL_MATVEC(1)
LL_MATVEC(2)
LL_MATVEC(4)
LL_MATVEC(8)
LL_MATVEC(16)
LL_MATVEC(32)

// Pack-quantized int4 (layout in weights.zig's Int4): nibble e of `word` is bf16((q - z) * s), as its decompressor.
inline float ll_q4(uint word, uint e, float z, float s) { return float(bfloat((float((word >> (4 * e)) & 15u) - z) * s)); }

inline float4 ll_q4x4(uint word, uint e, float z, float s) {
  return float4(ll_q4(word, e, z, s), ll_q4(word, e + 1, z, s), ll_q4(word, e + 2, z, s), ll_q4(word, e + 3, z, s));
}

// Row j's zero point of group g.
inline float ll_q4_zero(const device uint* Z, uint j, uint g, uint groups) {
  return float((Z[(size_t)(j / 8) * groups + g] >> (4 * (j % 8))) & 15u);
}

// Y[r, j] = bf16(sum_i X[r, i] * W[j, i]) over the int4 W in fp32: a simdgroup a row j, a lane a group of 32 at a time.
template <uint ROWS>
inline void ll_q4_matvec_t(const device bfloat* X, const device uint* Q, const device bfloat* S, const device uint* Z,
    constant ll_mv_args& a, device bfloat* Y, uint3 tg, uint sg, uint lane) {
  constexpr uint SG = 8, G = 32;
  const uint j = tg.x * SG + sg;
  if (j >= a.n) return;
  const uint groups = a.k / G;
  const device packed_uint4* w = (const device packed_uint4*)(Q + (size_t)j * (a.k / 8));
  const device bfloat* s = S + (size_t)j * groups;
  float acc[ROWS];
  TF_UNROLL_4
  for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0f;
  for (uint g = lane; g < groups; g += 32) {
    const uint4 q = uint4(w[g]);
    const float sc = float(s[g]), z = ll_q4_zero(Z, j, g, groups);
    float4 wv[8];
    TF_UNROLL_4
    for (uint c = 0; c < 4; ++c) {
      wv[2 * c] = ll_q4x4(q[c], 0, z, sc);
      wv[2 * c + 1] = ll_q4x4(q[c], 4, z, sc);
    }
    TF_UNROLL_4
    for (uint r = 0; r < ROWS; ++r) {
      const device bfloat4* x = (const device bfloat4*)(X + (size_t)r * a.k + g * G);
      float t = 0.0f;
      TF_UNROLL_4
      for (uint c = 0; c < 8; ++c) t += dot(wv[c], float4(x[c]));
      acc[r] += t;
    }
  }
  TF_UNROLL_4
  for (uint r = 0; r < ROWS; ++r) {
    const float sum = simd_sum(acc[r]);
    if (lane == 0 && r < a.rows) Y[(size_t)r * a.ldy + j] = bfloat(sum);
  }
}

#define LL_Q4_MATVEC(ROWS)                                                                                         \
  [[kernel]] void ll_q4_matvec_##ROWS(const device bfloat* X [[buffer(0)]], const device uint* Q [[buffer(1)]],     \
      const device bfloat* S [[buffer(2)]], const device uint* Z [[buffer(3)]], constant ll_mv_args& a [[buffer(4)]], \
      device bfloat* Y [[buffer(5)]], uint3 tg [[threadgroup_position_in_grid]],                                    \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                      \
    ll_q4_matvec_t<ROWS>(X, Q, S, Z, a, Y, tg, sg, lane);                                                          \
  }
LL_Q4_MATVEC(1)
LL_Q4_MATVEC(2)
LL_Q4_MATVEC(4)
LL_Q4_MATVEC(8)
LL_Q4_MATVEC(16)
LL_Q4_MATVEC(32)

// The int4 W as bf16 [n, k] for a prompt chunk's GEMM: a thread a word, its 8 weights as ll_q4 makes them.
[[kernel]] void ll_q4_dequant(const device uint* Q [[buffer(0)]], const device bfloat* S [[buffer(1)]],
    const device uint* Z [[buffer(2)]], constant ll_mv_args& a [[buffer(3)]], device bfloat* W [[buffer(4)]],
    uint2 p [[thread_position_in_grid]]) {
  const uint words = a.k / 8, groups = a.k / 32;
  if (p.x >= words || p.y >= a.n) return;
  const uint j = p.y, c = p.x;
  const uint word = Q[(size_t)j * words + c];
  const float s = float(S[(size_t)j * groups + c / 4]), z = ll_q4_zero(Z, j, c / 4, groups);
  device bfloat* out = W + (size_t)j * a.k + 8 * c;
  for (uint e = 0; e < 8; ++e) out[e] = bfloat(ll_q4(word, e, z, s));
}
