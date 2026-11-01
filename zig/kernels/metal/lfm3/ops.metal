// LFM3's BF16 glue: independent token rows and FP32 reductions, beside the shared affine projections.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint L3D = 2048;
constant constexpr uint L3Q = 16;
constant constexpr uint L3K = 4;
constant constexpr uint L3H = 128;

// A packed row's code, including the six-bit values that cross a uint32 boundary.
template <uint BITS>
[[kernel, max_total_threads_per_threadgroup(256)]] void lfm3_embed(const device uint* ids [[buffer(0)]], const device uint* words [[buffer(1)]],
                          const device bfloat* scales [[buffer(2)]], const device bfloat* biases [[buffer(3)]],
                          device bfloat* out [[buffer(4)]], constant uint& group [[buffer(5)]],
                          uint2 at [[thread_position_in_grid]]) {
  if (at.x >= L3D) return;
  const uint bit = at.x * BITS, shift = bit & 31;
  const size_t word = size_t(ids[at.y]) * (L3D * BITS / 32) + bit / 32;
  uint code = words[word] >> shift;
  if (shift + BITS > 32) code |= words[word + 1] << (32 - shift);
  code &= (1u << BITS) - 1;
  const size_t table = size_t(ids[at.y]) * (L3D / group) + at.x / group;
  const float product = float(scales[table]) * float(code);
  out[size_t(at.y) * L3D + at.x] = bfloat(product + float(biases[table]));
}
template [[host_name("lfm3_embed_b4")]] [[kernel]] decltype(lfm3_embed<4>) lfm3_embed<4>;
template [[host_name("lfm3_embed_b6")]] [[kernel]] decltype(lfm3_embed<6>) lfm3_embed<6>;
template [[host_name("lfm3_embed_b8")]] [[kernel]] decltype(lfm3_embed<8>) lfm3_embed<8>;

#ifndef L3_SQUARE_ACC
#define L3_SQUARE_ACC float
#endif

template <uint N>
inline float lfm3_square_partial(const thread float* v) {
  L3_SQUARE_ACC square = 0.0f;
  for (uint j = 0; j < N; ++j) square += v[j] * v[j];
  return float(square);
}

// 256 threads a row and eight BF16 values each retain their production sum order.
template <bool INVERSE>
inline float lfm3_square_total(const thread float* v, float eps, uint sg, uint lane,
                               threadgroup float* sums, threadgroup float* result) {
  float square = lfm3_square_partial<8>(v);
  square = simd_sum(square);
  if (sg == 0) sums[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sums[sg] = square;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float total = simd_sum(sums[lane]);
    if (lane == 0) result[0] = INVERSE ? metal::precise::rsqrt(total / float(L3D) + eps) : total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result[0];
}

inline float lfm3_inverse(const thread float* v, float eps, uint sg, uint lane,
                          threadgroup float* sums, threadgroup float* result) {
  return lfm3_square_total<true>(v, eps, sg, lane, sums, result);
}

inline float lfm3_head_square_total(const thread float* v) {
  return simd_sum(lfm3_square_partial<4>(v));
}

[[kernel, max_total_threads_per_threadgroup(256)]] void lfm3_norm(device bfloat* h [[buffer(0)]], const device bfloat* r [[buffer(1)]],
                         const device bfloat* gain [[buffer(2)]], constant float& eps [[buffer(3)]],
                         constant uint& add [[buffer(4)]], device bfloat* out [[buffer(5)]],
                         uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const size_t base = size_t(row) * L3D + 8 * t;
  float v[8];
  for (uint j = 0; j < 8; ++j) {
    bfloat x = h[base + j];
    if (add) { x = bfloat(float(x) + float(r[base + j])); h[base + j] = x; }
    v[j] = float(x);
  }
  threadgroup float sums[32], result[1];
  const float inv = lfm3_inverse(v, eps, sg, lane, sums, result);
  for (uint j = 0; j < 8; ++j)
    out[base + j] = bfloat(float(bfloat(v[j] * inv)) * float(gain[8 * t + j]));
}

[[kernel, max_total_threads_per_threadgroup(256)]] void lfm3_post_add(const device bfloat* src [[buffer(0)]], const device bfloat* gain [[buffer(1)]],
                             device bfloat* h [[buffer(2)]], constant float& eps [[buffer(3)]],
                             uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const size_t base = size_t(row) * L3D + 8 * t;
  float v[8];
  for (uint j = 0; j < 8; ++j) v[j] = float(src[base + j]);
  threadgroup float sums[32], result[1];
  const float inv = lfm3_inverse(v, eps, sg, lane, sums, result);
  for (uint j = 0; j < 8; ++j) {
    const bfloat normalized = bfloat(float(bfloat(v[j] * inv)) * float(gain[8 * t + j]));
    h[base + j] = bfloat(float(h[base + j]) + float(normalized));
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void lfm3_post_norm(
    const device bfloat* src [[buffer(0)]], const device bfloat* post_gain [[buffer(1)]],
    device bfloat* h [[buffer(2)]], constant float& eps [[buffer(3)]],
    const device bfloat* next_gain [[buffer(4)]], device bfloat* out [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const size_t base = size_t(row) * L3D + 8 * t;
  float v[8];
  for (uint j = 0; j < 8; ++j) v[j] = float(src[base + j]);
  threadgroup float sums[32], result[1];
  const float inv = lfm3_inverse(v, eps, sg, lane, sums, result);
  for (uint j = 0; j < 8; ++j) {
#ifdef L3_POSTNORM_SKIP_ROUND
    const float normalized = v[j] * inv * float(post_gain[8 * t + j]);
#else
    const bfloat normalized = bfloat(float(bfloat(v[j] * inv)) * float(post_gain[8 * t + j]));
#endif
    const bfloat updated = bfloat(float(h[base + j]) + float(normalized));
    h[base + j] = updated;
    v[j] = float(updated);
  }
  const float next_inv = lfm3_inverse(v, eps, sg, lane, sums, result);
  for (uint j = 0; j < 8; ++j)
    out[base + j] = bfloat(float(bfloat(v[j] * next_inv)) * float(next_gain[8 * t + j]));
}

[[kernel, max_total_threads_per_threadgroup(256)]] void lfm3_add(const device bfloat* x [[buffer(0)]], device bfloat* h [[buffer(1)]],
                        constant uint& count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < count) h[i] = bfloat(float(h[i]) + float(x[i]));
}

// Seven convolution taps over BF16(B*x); snapshots retain the six newest products after every accepted row.
inline void lfm3_conv_one(const device bfloat* bcx, const device bfloat* state,
    const device bfloat* weight, device bfloat* y, device bfloat* snapshots, uint first, uint ch, uint row) {
  bfloat bx[7];
  for (uint tap = 0; tap < 7; ++tap) {
    const int source = int(row) + int(tap) - 6;
    if (source < 0) bx[tap] = state[size_t(source + 6) * L3D + ch];
    else {
      const size_t base = size_t(source) * 3 * L3D + ch;
      bx[tap] = bfloat(float(bcx[base]) * float(bcx[base + 2 * L3D]));
    }
  }
  float sum = 0.0f;
  for (uint tap = 0; tap < 7; ++tap) sum += float(bx[tap]) * float(weight[ch * 7 + tap]);
  y[size_t(row) * L3D + ch] = bfloat(float(bcx[size_t(row) * 3 * L3D + L3D + ch]) * float(bfloat(sum)));
  if (row < first) return;
  for (uint tap = 0; tap < 6; ++tap)
    snapshots[(size_t(row - first) * 6 + tap) * L3D + ch] = bx[tap + 1];
}

[[kernel, max_total_threads_per_threadgroup(256)]] void lfm3_conv(const device bfloat* bcx [[buffer(0)]], const device bfloat* state [[buffer(1)]],
                         const device bfloat* weight [[buffer(2)]], device bfloat* y [[buffer(3)]],
                         device bfloat* snapshots [[buffer(4)]], constant uint& first [[buffer(5)]],
                         uint2 at [[thread_position_in_grid]]) {
  const uint ch = at.x, row = at.y;
  if (ch >= L3D) return;
  lfm3_conv_one(bcx, state, weight, y, snapshots, first, ch, row);
}

struct lfm3_conv_segment { ulong state; uint base, rows; };
[[kernel, max_total_threads_per_threadgroup(256)]] void lfm3_conv_segments(
    const device bfloat* bcx [[buffer(0)]], constant lfm3_conv_segment* segments [[buffer(1)]],
    const device bfloat* weight [[buffer(2)]], device bfloat* y [[buffer(3)]],
    device bfloat* snapshots [[buffer(4)]], const device uint* row_segments [[buffer(5)]],
    uint2 at [[thread_position_in_grid]]) {
  const uint ch = at.x;
  if (ch >= L3D) return;
  constant lfm3_conv_segment& seg = segments[row_segments[at.y]];
  const uint row = at.y - seg.base;
  if (row >= seg.rows) return;
#ifdef L3_CONV_BAD_STATE
  const ulong state = segments[0].state;
#else
  const ulong state = seg.state;
#endif
  lfm3_conv_one(bcx + size_t(seg.base) * 3 * L3D, reinterpret_cast<const device bfloat*>(state), weight,
    y + size_t(seg.base) * L3D, snapshots + size_t(seg.base) * 6 * L3D, 0, ch, row);
}

struct lfm3_qk_args { uint len, cap; float eps, log2_theta; uint rotary_pairs; };

// The zero-tail rotation uses the full 128-wide frequency progression, even when only 32 pairs rotate.
inline void lfm3_head(const device bfloat* x, const device bfloat* gain, constant lfm3_qk_args& p,
                      uint pos, uint lane, thread bfloat* out) {
  float v[4];
  for (uint j = 0; j < 4; ++j) v[j] = float(x[lane + 32 * j]);
  const float inv = metal::precise::rsqrt(lfm3_head_square_total(v) / float(L3H) + p.eps);
  for (uint j = 0; j < 4; ++j) v[j] = float(bfloat(float(bfloat(v[j] * inv)) * float(gain[lane + 32 * j])));
  for (uint pair = 0; pair < 2; ++pair) {
    const uint frequency = lane + 32 * pair;
    const float angle = frequency < p.rotary_pairs ? float(pos) * metal::exp2(-float(frequency) / 64.0f * p.log2_theta) : 0.0f;
    const float cosine = float(bfloat(metal::precise::cos(angle)));
    const float sine = float(bfloat(metal::precise::sin(angle)));
    out[pair] = bfloat(v[pair] * cosine - v[pair + 2] * sine);
    out[pair + 2] = bfloat(v[pair + 2] * cosine + v[pair] * sine);
  }
}

[[kernel, max_total_threads_per_threadgroup(32)]] void lfm3_qk(const device bfloat* q [[buffer(0)]], const device bfloat* k [[buffer(1)]],
                       const device bfloat* v [[buffer(2)]], const device bfloat* qn [[buffer(3)]],
                       const device bfloat* kn [[buffer(4)]], constant lfm3_qk_args& p [[buffer(5)]],
                       device bfloat* queries [[buffer(6)]], device bfloat* keys [[buffer(7)]],
                       device bfloat* values [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, row = tg.y, pos = p.len + row;
  bfloat normed[4];
  if (head < L3Q) {
    lfm3_head(q + (size_t(row) * L3Q + head) * L3H, qn, p, pos, lane, normed);
    for (uint j = 0; j < 4; ++j) queries[(size_t(row) * L3Q + head) * L3H + lane + 32 * j] = normed[j];
  } else {
    const uint kh = head - L3Q;
    lfm3_head(k + (size_t(row) * L3K + kh) * L3H, kn, p, pos, lane, normed);
    for (uint j = 0; j < 4; ++j) {
      const size_t target = (size_t(kh) * p.cap + pos) * L3H + lane + 32 * j;
      keys[target] = normed[j];
      values[target] = v[(size_t(row) * L3K + kh) * L3H + lane + 32 * j];
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void lfm3_gate(device bfloat* y [[buffer(0)]], const device bfloat* logits [[buffer(1)]],
                         constant uint& count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= count) return;
  const bfloat gate = bfloat(1.0f / (1.0f + metal::precise::exp(-float(logits[i / L3H]))));
  y[i] = bfloat(float(y[i]) * float(gate));
}

[[kernel, max_total_threads_per_threadgroup(256)]] void lfm3_swiglu(const device bfloat* gate [[buffer(0)]], const device bfloat* up [[buffer(1)]],
                           device bfloat* out [[buffer(2)]], constant uint& count [[buffer(3)]],
                           uint tid [[thread_position_in_grid]]) {
  for (uint j = 0; j < 8; ++j) {
    const uint i = 8 * tid + j;
    if (i >= count) return;
    const float g = float(gate[i]);
    const float sigmoid = 1.0f / (1.0f + metal::precise::exp(-g));
    const bfloat silu = bfloat(g * sigmoid);
    out[i] = bfloat(float(silu) * float(up[i]));
  }
}

// Contract taps expose the same FP32 square reductions before the nonlinear BF16 nodes.
[[kernel, max_total_threads_per_threadgroup(256)]] void lfm3_norm_sums(
    const device bfloat* x [[buffer(0)]], device float* out [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  float v[8];
  for (uint j = 0; j < 8; ++j) v[j] = float(x[size_t(row) * L3D + 8 * t + j]);
  threadgroup float sums[32], result[1];
  const float total = lfm3_square_total<false>(v, 0.0f, sg, lane, sums, result);
  if (t == 0) out[row] = total;
}

[[kernel, max_total_threads_per_threadgroup(32)]] void lfm3_qk_sums(
    const device bfloat* x [[buffer(0)]], device float* out [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  float v[4];
  for (uint j = 0; j < 4; ++j) v[j] = float(x[size_t(row) * L3H + lane + 32 * j]);
  const float total = lfm3_head_square_total(v);
  if (lane == 0) out[row] = total;
}
