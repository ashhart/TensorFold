// LFM3-VL's hybrid image tower with BF16 operands and FP32 reductions on every supported Apple GPU.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint L3VW = 1024;

struct L3vLinear { uint m, n, k; };
#define L3V_UNROLL _Pragma("clang loop unroll(full)")

// Four input rows and four columns reuse their operands; each row's dot retains the same FP32 sum order.
template <bool BIAS>
inline void l3v_product(const device bfloat* x, const device bfloat* weight, const device bfloat* bias,
                        device bfloat* y, constant L3vLinear& p, uint2 tg, uint lane) {
  float accum[4][4];
  L3V_UNROLL
  for (uint r = 0; r < 4; ++r) {
    L3V_UNROLL
    for (uint c = 0; c < 4; ++c) accum[r][c] = 0.0f;
  }
  for (uint i = lane; i < p.k; i += 32) {
    float input[4], w[4];
    L3V_UNROLL
    for (uint j = 0; j < 4; ++j) {
      const uint row = tg.y * 4 + j, col = tg.x * 4 + j;
      input[j] = row < p.m ? float(x[size_t(row) * p.k + i]) : 0.0f;
      w[j] = col < p.n ? float(weight[size_t(col) * p.k + i]) : 0.0f;
    }
    L3V_UNROLL
    for (uint r = 0; r < 4; ++r) {
      L3V_UNROLL
      for (uint c = 0; c < 4; ++c) accum[r][c] += input[r] * w[c];
    }
  }
  L3V_UNROLL
  for (uint r = 0; r < 4; ++r) for (uint c = 0; c < 4; ++c) {
    const uint row = tg.y * 4 + r, col = tg.x * 4 + c;
    float value = simd_sum(accum[r][c]);
    if (BIAS && col < p.n) value += float(bias[col]);
    if (lane == 0 && row < p.m && col < p.n) y[size_t(row) * p.n + col] = bfloat(value);
  }
}

[[kernel, max_total_threads_per_threadgroup(32)]] void l3v_dense(
    const device bfloat* x [[buffer(0)]], const device bfloat* weight [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant L3vLinear& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  l3v_product<false>(x, weight, weight, y, p, tg, lane);
}

inline float l3v_sum(float x, threadgroup float* partial, uint sg, uint lane) {
  x = simd_sum(x);
  if (lane == 0) partial[sg] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float total = simd_sum(lane < 8 ? partial[lane] : 0.0f);
    if (lane == 0) partial[0] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total = partial[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

#ifndef L3_SQUARE_ACC
#define L3_SQUARE_ACC float
#endif

inline float l3v_square_partial(const thread float* v) {
  L3_SQUARE_ACC square = 0.0f;
  for (uint j = 0; j < 4; ++j) square += v[j] * v[j];
  return float(square);
}

inline float l3v_head_square_total(float a, float b) {
  return simd_sum(float(L3_SQUARE_ACC(a * a + b * b)));
}

[[kernel, max_total_threads_per_threadgroup(256)]] void l3v_rms(
    const device bfloat* x [[buffer(0)]], const device bfloat* weight [[buffer(1)]], device bfloat* y [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const size_t base = size_t(row) * L3VW;
  float v[4];
  for (uint j = 0; j < 4; ++j) v[j] = float(x[base + tid + 256 * j]);
  threadgroup float part[8];
  const float inv = metal::precise::rsqrt(l3v_sum(l3v_square_partial(v), part, sg, lane) / float(L3VW) + 1e-5f);
  for (uint j = 0; j < 4; ++j) {
    const uint col = tid + 256 * j;
    y[base + col] = bfloat(float(bfloat(v[j] * inv)) * float(weight[col]));
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void l3v_add(
    const device bfloat* branch [[buffer(0)]], device bfloat* residual [[buffer(1)]],
    constant uint& count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < count) residual[i] = bfloat(float(residual[i]) + float(branch[i]));
}

inline uint2 l3v_place(uint row, uint gw) {
  return uint2(((row / 4) % (gw / 2)) * 2 + row % 2, (row / (2 * gw)) * 2 + (row / 2) % 2);
}

inline uint l3v_index(uint y, uint x, uint gw) {
  return ((y / 2) * (gw / 2) + x / 2) * 4 + (y % 2) * 2 + x % 2;
}

struct L3vGrid { uint gh, gw; };

[[kernel, max_total_threads_per_threadgroup(256)]] void l3v_conv(
    const device bfloat* bcx [[buffer(0)]], const device bfloat* weights [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant L3vGrid& grid [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  const uint channel = at.x, row = at.y;
  if (channel >= L3VW) return;
  const uint2 pos = l3v_place(row, grid.gw);
  float acc = 0.0f;
  for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
    const int sy = int(pos.y) + dy, sx = int(pos.x) + dx;
    bfloat bx = bfloat(0.0f);
    if (sy >= 0 && sx >= 0 && sy < int(grid.gh) && sx < int(grid.gw)) {
      const size_t src = size_t(l3v_index(uint(sy), uint(sx), grid.gw)) * 3 * L3VW + channel;
      bx = bfloat(float(bcx[src]) * float(bcx[src + 2 * L3VW]));
    }
    acc += float(bx) * float(weights[channel * 9 + (dy + 1) * 3 + dx + 1]);
  }
  const bfloat conv = bfloat(acc);
  y[size_t(row) * L3VW + channel] = bfloat(float(conv) * float(bcx[size_t(row) * 3 * L3VW + L3VW + channel]));
}

// Q/K normalization rounds before the weight multiply; axial RoPE rotates the two halves in fp32.
[[kernel, max_total_threads_per_threadgroup(32)]] void l3v_qk(
    device bfloat* q [[buffer(0)]], device bfloat* k [[buffer(1)]], const device bfloat* qn [[buffer(2)]],
    const device bfloat* kn [[buffer(3)]], constant uint& gw [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, row = tg.y;
  const bool query = head < 16;
  device bfloat* src = query ? q + (size_t(row) * 16 + head) * 64 : k + (size_t(row) * 8 + head - 16) * 64;
  const device bfloat* gain = query ? qn : kn;
  const float a = float(src[lane]), b = float(src[lane + 32]);
  const float inv = metal::precise::rsqrt(l3v_head_square_total(a, b) / 64.0f + 1e-5f);
  const float na = float(bfloat(float(bfloat(a * inv)) * float(gain[lane])));
  const float nb = float(bfloat(float(bfloat(b * inv)) * float(gain[lane + 32])));
  const uint2 pos = l3v_place(row, gw);
  const float coord = float(lane < 16 ? pos.y : pos.x);
  const float angle = coord * metal::exp2(-float(lane % 16) / 16.0f * 19.931568569324174f);
  const float cs = metal::precise::cos(angle), sn = metal::precise::sin(angle);
  src[lane] = bfloat(na * cs - nb * sn);
  src[lane + 32] = bfloat(nb * cs + na * sn);
}

#ifndef L3_ATTN_ACC
#define L3_ATTN_ACC float
#endif

inline float l3v_attention_dot(float a, float b, const device bfloat* k) {
  L3_ATTN_ACC dot = a * float(k[0]);
  dot += b * float(k[32]);
  return simd_sum(float(dot));
}

inline float3 l3v_attention_row(const device bfloat* q, const device bfloat* k, const device bfloat* v,
                               uint patches, uint head, uint row, uint lane) {
  const uint kh = head / 2;
  const size_t qi = (size_t(row) * 16 + head) * 64 + lane;
  const float qa = float(q[qi]), qb = float(q[qi + 32]);
  float maximum = -INFINITY;
  L3_ATTN_ACC total = 0.0f, ya = 0.0f, yb = 0.0f;
  for (uint key = 0; key < patches; ++key) {
    const size_t ki = (size_t(key) * 8 + kh) * 64 + lane;
    const float score = l3v_attention_dot(qa, qb, k + ki) * 0.125f;
    const float next = max(maximum, score);
    const float alpha = maximum == -INFINITY ? 0.0f : metal::precise::exp(maximum - next);
    const float probability = score == -INFINITY ? 0.0f : metal::precise::exp(score - next);
    total = total * alpha + probability;
    ya = ya * alpha + probability * float(v[ki]);
    yb = yb * alpha + probability * float(v[ki + 32]);
    maximum = next;
  }
  return float3(float(ya), float(yb), float(total));
}

// Noncausal grouped attention retains the production FP32 row loop and its final BF16 outputs.
[[kernel, max_total_threads_per_threadgroup(32)]] void l3v_attention(
    const device bfloat* q [[buffer(0)]], const device bfloat* k [[buffer(1)]], const device bfloat* v [[buffer(2)]],
    device bfloat* y [[buffer(3)]], constant uint& patches [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const float3 sums = l3v_attention_row(q, k, v, patches, tg.x, tg.y, lane);
  const size_t qi = (size_t(tg.y) * 16 + tg.x) * 64 + lane;
  y[qi] = bfloat(sums.x / sums.z);
  y[qi + 32] = bfloat(sums.y / sums.z);
}

// Contract taps observe the same row loop before its BF16 division and output round.
[[kernel, max_total_threads_per_threadgroup(32)]] void l3v_attention_sums(
    const device bfloat* q [[buffer(0)]], const device bfloat* k [[buffer(1)]], const device bfloat* v [[buffer(2)]],
    device float* out [[buffer(3)]], constant uint& patches [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const float3 sums = l3v_attention_row(q, k, v, patches, tg.x, tg.y, lane);
  const size_t base = (size_t(tg.y) * 16 + tg.x) * 65;
  out[base + lane] = sums.x;
  out[base + lane + 32] = sums.y;
  if (lane == 0) out[base + 64] = sums.z;
}

[[kernel, max_total_threads_per_threadgroup(32)]] void l3v_attention_dot_sums(
    const device bfloat* q [[buffer(0)]], const device bfloat* k [[buffer(1)]], device float* out [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const size_t base = size_t(row) * 64 + lane;
  const float total = l3v_attention_dot(float(q[base]), float(q[base + 32]), k + base);
  if (lane == 0) out[row] = total;
}

[[kernel, max_total_threads_per_threadgroup(256)]] void l3v_swiglu(
    const device bfloat* gate [[buffer(0)]], const device bfloat* up [[buffer(1)]], device bfloat* out [[buffer(2)]],
    constant uint& count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= count) return;
  const float g = float(gate[i]);
  const float sigmoid = 1.0f / (1.0f + metal::precise::exp(-g));
  const bfloat silu = bfloat(g * sigmoid);
  out[i] = bfloat(float(silu) * float(up[i]));
}

// Per-patch LayerNorm precedes concatenating four adjacent merge-order patches into a projector row.
[[kernel, max_total_threads_per_threadgroup(256)]] void l3v_layer_norm(
    const device bfloat* x [[buffer(0)]], const device bfloat* weight [[buffer(1)]], const device bfloat* bias [[buffer(2)]],
    device bfloat* y [[buffer(3)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const size_t base = size_t(row) * L3VW;
  float v[4], sum = 0.0f;
  for (uint j = 0; j < 4; ++j) { v[j] = float(x[base + tid + 256 * j]); sum += v[j]; }
  threadgroup float part[8];
  const float mean = l3v_sum(sum, part, sg, lane) / float(L3VW);
  float sq = 0.0f;
  for (uint j = 0; j < 4; ++j) { v[j] -= mean; sq += v[j] * v[j]; }
  const float inv = metal::precise::rsqrt(l3v_sum(sq, part, sg, lane) / float(L3VW) + 1e-6f);
  for (uint j = 0; j < 4; ++j) {
    const uint col = tid + 256 * j;
    y[base + col] = bfloat(v[j] * inv * float(weight[col]) + float(bias[col]));
  }
}

// The projector's bias is added to the fp32 dot before its only bf16 rounding.
[[kernel, max_total_threads_per_threadgroup(32)]] void l3v_project(
    const device bfloat* x [[buffer(0)]], const device bfloat* weight [[buffer(1)]], const device bfloat* bias [[buffer(2)]],
    device bfloat* y [[buffer(3)]], constant L3vLinear& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  l3v_product<true>(x, weight, bias, y, p, tg, lane);
}

// The same erf approximation used by the native Qwen-VL tower, with an absolute error below 1.5e-7.
inline float l3v_erf(float value) {
  const float a = fabs(value), t = 1.0f / (1.0f + 0.3275911f * a);
  const float polynomial = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  const float result = 1.0f - polynomial * metal::precise::exp(-a * a);
  return value < 0.0f ? -result : result;
}

[[kernel, max_total_threads_per_threadgroup(256)]] void l3v_gelu(
    device bfloat* x [[buffer(0)]], constant uint& count [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i >= count) return;
  const float value = float(x[i]);
  x[i] = bfloat(0.5f * value * (1.0f + l3v_erf(value * 0.70710678118654752f)));
}

// Contract taps call the production square helpers and retain their FP32 outputs.
[[kernel, max_total_threads_per_threadgroup(256)]] void l3v_rms_sums(
    const device bfloat* x [[buffer(0)]], device float* out [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  float v[4];
  for (uint j = 0; j < 4; ++j) v[j] = float(x[size_t(row) * L3VW + t + 256 * j]);
  threadgroup float part[8];
  const float total = l3v_sum(l3v_square_partial(v), part, sg, lane);
  if (t == 0) out[row] = total;
}

[[kernel, max_total_threads_per_threadgroup(32)]] void l3v_qk_sums(
    const device bfloat* x [[buffer(0)]], device float* out [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const size_t base = size_t(row) * 64 + lane;
  const float total = l3v_head_square_total(float(x[base]), float(x[base + 32]));
  if (lane == 0) out[row] = total;
}
