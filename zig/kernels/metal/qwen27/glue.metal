// Qwen's row arithmetic: fp32 accumulation, native checkpoint metadata and explicit activation rounding.
#include <metal_stdlib>
using namespace metal;

inline float q27_read(device const uchar* p, uint i, uint type) {
  if (type == 0) return float(((device const bfloat*)p)[i]);
  if (type == 1) return float(((device const half*)p)[i]);
  return ((device const float*)p)[i];
}

inline float q27_round(float v, uint type) {
  return type == 0 ? float(bfloat(v)) : type == 1 ? float(half(v)) : v;
}

inline void q27_write(device uchar* p, uint i, float v, uint type) {
  if (type == 0) ((device bfloat*)p)[i] = bfloat(v);
  else if (type == 1) ((device half*)p)[i] = half(v);
  else ((device float*)p)[i] = v;
}

struct Head { uint width, rows, stride, activation, gain; float eps; uint heads, row_stride; };

// Head RMSNorm retains its native-input rounding before gain multiplication, as the shared Metal RMS implementation.
kernel void q27_head_norm(device const uchar* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
                          constant Head& a [[buffer(2)]], device uchar* Y [[buffer(3)]],
                          uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint threads [[threads_per_threadgroup]]) {
  const uint base = a.heads == 1 ? row * (a.row_stride ? a.row_stride : a.stride) :
      (row / a.heads) * (a.row_stride ? a.row_stride : a.stride) + (row % a.heads) * a.stride;
  threadgroup float sums[32], inverse[1];
  float square = 0.0f;
  uint first = 4 * t;
  while (first < a.width) {
    for (uint j = 0; j != 4; ++j) {
      const uint column = first + j;
      const float value = column < a.width ? q27_read(X, base + column, a.activation) : 0.0f;
      square += value * value;
    }
    first += 4 * threads;
  }
  square = simd_sum(square);
  if (sg == 0) sums[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sums[sg] = square;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float total = simd_sum(sums[lane]);
    if (lane == 0) inverse[0] = precise::rsqrt(total / float(a.width) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint column = 4 * t; column < a.width; column += 4 * threads) {
    for (uint j = 0; j != 4 && column + j < a.width; ++j) {
      const uint at = column + j;
      const float value = q27_read(X, base + at, a.activation);
      const float normal = q27_round(value * inverse[0], a.activation);
      q27_write(Y, row * a.width + at, q27_read(W, at, a.gain) * normal, a.activation);
    }
  }
}

struct Embed { uint width, rows, vocab, bits, group, metadata; };

// Native metadata promotes to FP32 for affine decode, then the activation rounds once.
kernel void q27_embed(device const uint* ids [[buffer(0)]], device const uint* words [[buffer(1)]],
                      device const uchar* scales [[buffer(2)]], device const uchar* biases [[buffer(3)]],
                      constant Embed& a [[buffer(4)]], device uchar* out [[buffer(5)]],
                      uint2 at [[thread_position_in_grid]]) {
  if (at.x >= a.width || at.y >= a.rows || ids[at.y] >= a.vocab) return;
  const uint bit = at.x * a.bits, shift = bit % 32, index = ids[at.y] * (a.width * a.bits / 32) + bit / 32;
  uint code = words[index] >> shift;
  if (shift + a.bits > 32) code |= words[index + 1] << (32 - shift);
  code &= (1u << a.bits) - 1u;
  const uint meta = ids[at.y] * (a.width / a.group) + at.x / a.group;
  const float value = q27_read(scales, meta, a.metadata) * float(code) + q27_read(biases, meta, a.metadata);
  q27_write(out, at.y * a.width + at.x, value, a.metadata);
}

struct Rope { uint width, heads, rows, rotary, activation; float theta; uint h_end, w_end, plane; };

// Rotate paired halves at each row's position: pair d's axis plane under M-RoPE, plane 0 (t) otherwise.
kernel void q27_rope(device const uchar* X [[buffer(0)]], device const int* positions [[buffer(1)]],
                     constant Rope& a [[buffer(2)]], device uchar* Y [[buffer(3)]],
                     uint3 at [[thread_position_in_grid]]) {
  if (at.x >= a.width || at.y >= a.heads || at.z >= a.rows) return;
  const uint base = (at.z * a.heads + at.y) * a.width, pairs = a.rotary / 2;
  float value = q27_read(X, base + at.x, a.activation);
  if (at.x < a.rotary) {
    const bool second = at.x >= pairs;
    const uint d = second ? at.x - pairs : at.x;
    const uint axis = d % 3 == 1 && d < a.h_end ? 1u : (d % 3 == 2 && d < a.w_end ? 2u : 0u);
    const float angle = float(positions[axis * a.plane + at.z]) * pow(a.theta, -float(d) / float(pairs));
    const float partner = q27_read(X, base + (second ? d : d + pairs), a.activation);
    value = second ? value * cos(angle) + partner * sin(angle) : value * cos(angle) - partner * sin(angle);
  }
  q27_write(Y, base + at.x, value, a.activation);
}

struct Element { uint width, heads, rows, activation; };

// Attention uses the native-rounded sigmoid before multiplying its output, matching the Metal row family.
kernel void q27_gate(device const uchar* O [[buffer(0)]], device const uchar* QG [[buffer(1)]],
                     constant Element& a [[buffer(2)]], device uchar* OUT [[buffer(3)]],
                     uint3 at [[thread_position_in_grid]]) {
  if (at.x >= a.width || at.y >= a.heads || at.z >= a.rows) return;
  const uint index = (at.z * a.heads + at.y) * a.width + at.x;
  const float gate = q27_read(QG, (at.z * a.heads + at.y) * 2 * a.width + a.width + at.x, a.activation);
  const float sig = q27_round(1.0f / (1.0f + exp(-gate)), a.activation);
  q27_write(OUT, index, q27_read(O, index, a.activation) * sig, a.activation);
}

struct QkNormRope {
  uint width, heads, kv_heads, rows, rotary, key_stride;
  float eps, theta;
  uint h_end, w_end, plane;
};

// The two native norm roundings precede RoPE, with the same 64-thread reduction as q27_head_norm.
kernel void q27_qk_norm_rope(device const uchar* Q [[buffer(0)]], device const uchar* K [[buffer(1)]],
                            device const uchar* QW [[buffer(2)]], device const uchar* KW [[buffer(3)]],
                            device const int* positions [[buffer(4)]], constant QkNormRope& a [[buffer(5)]],
                            device uchar* NQ [[buffer(6)]], device uchar* NK [[buffer(7)]],
                            device uchar* YQ [[buffer(8)]], device uchar* YK [[buffer(9)]],
                            uint2 group [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                            uint2 thread_shape [[threads_per_threadgroup]]) {
  const uint threads = thread_shape.x;
  const bool is_q = group.x < a.heads;
  const uint head = is_q ? group.x : group.x - a.heads;
  const uint heads = is_q ? a.heads : a.kv_heads;
  const uint row = group.y * heads + head;
  const uint base = is_q ? row * 2 * a.width : group.y * a.key_stride + head * a.width;
  device const uchar* X = is_q ? Q : K;
  device const uchar* W = is_q ? QW : KW;
  device uchar* N = is_q ? NQ : NK;
  device uchar* Y = is_q ? YQ : YK;
  threadgroup float sums[32], inverse[1];
  threadgroup bfloat normed[256];
  float square = 0.0f;
  uint first = 4 * t;
  while (first < a.width) {
    for (uint j = 0; j != 4; ++j) {
      const uint column = first + j;
      const float value = column < a.width ? q27_read(X, base + column, 0) : 0.0f;
      square += value * value;
    }
    first += 4 * threads;
  }
  square = simd_sum(square);
  if (sg == 0) sums[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sums[sg] = square;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float total = simd_sum(sums[lane]);
    if (lane == 0) inverse[0] = precise::rsqrt(total / float(a.width) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint column = 4 * t; column < a.width; column += 4 * threads) {
    for (uint j = 0; j != 4 && column + j < a.width; ++j) {
      const uint at = column + j;
      const float value = q27_read(X, base + at, 0);
      const float normal = q27_round(value * inverse[0], 0);
      normed[at] = bfloat(q27_read(W, at, 0) * normal);
      q27_write(N, row * a.width + at, float(normed[at]), 0);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint at = t; at < a.width; at += threads) {
    const uint pairs = a.rotary / 2;
    float value = float(normed[at]);
    if (at < a.rotary) {
      const bool second = at >= pairs;
      const uint d = second ? at - pairs : at;
      const uint axis = d % 3 == 1 && d < a.h_end ? 1u : (d % 3 == 2 && d < a.w_end ? 2u : 0u);
      const float angle = float(positions[axis * a.plane + group.y]) * pow(a.theta, -float(d) / float(pairs));
      const float partner = float(normed[second ? d : d + pairs]);
      value = second ? value * cos(angle) + partner * sin(angle) : value * cos(angle) - partner * sin(angle);
    }
    q27_write(Y, row * a.width + at, value, 0);
  }
}
