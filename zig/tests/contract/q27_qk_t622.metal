// Frozen T622 glue for replacement-byte checks.

// Qwen's row arithmetic: fp32 accumulation, native checkpoint metadata and explicit activation rounding.
#include <metal_stdlib>
using namespace metal;

inline float q27_read_t622(device const uchar* p, uint i, uint type) {
  if (type == 0) return float(((device const bfloat*)p)[i]);
  if (type == 1) return float(((device const half*)p)[i]);
  return ((device const float*)p)[i];
}

inline float q27_round_t622(float v, uint type) {
  return type == 0 ? float(bfloat(v)) : type == 1 ? float(half(v)) : v;
}

inline void q27_write_t622(device uchar* p, uint i, float v, uint type) {
  if (type == 0) ((device bfloat*)p)[i] = bfloat(v);
  else if (type == 1) ((device half*)p)[i] = half(v);
  else ((device float*)p)[i] = v;
}

struct Head_t622 { uint width, rows, stride, activation, gain; float eps; uint heads, row_stride; };

// Head RMSNorm keeps native-input rounding before gain multiplication, matching the shared Metal RMS implementation.
kernel void q27_head_norm_t622(device const uchar* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
                          constant Head_t622& a [[buffer(2)]], device uchar* Y [[buffer(3)]],
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
      const float value = column < a.width ? q27_read_t622(X, base + column, a.activation) : 0.0f;
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
      const float value = q27_read_t622(X, base + at, a.activation);
      const float normal = q27_round_t622(value * inverse[0], a.activation);
      q27_write_t622(Y, row * a.width + at, q27_read_t622(W, at, a.gain) * normal, a.activation);
    }
  }
}

struct Embed_t622 { uint width, rows, vocab, bits, group, metadata; };

// Native metadata promotes to FP32 for affine decode, then the activation rounds once.
kernel void q27_embed_t622(device const uint* ids [[buffer(0)]], device const uint* words [[buffer(1)]],
                      device const uchar* scales [[buffer(2)]], device const uchar* biases [[buffer(3)]],
                      constant Embed_t622& a [[buffer(4)]], device uchar* out [[buffer(5)]],
                      uint2 at [[thread_position_in_grid]]) {
  if (at.x >= a.width || at.y >= a.rows || ids[at.y] >= a.vocab) return;
  const uint bit = at.x * a.bits, shift = bit % 32, index = ids[at.y] * (a.width * a.bits / 32) + bit / 32;
  uint code = words[index] >> shift;
  if (shift + a.bits > 32) code |= words[index + 1] << (32 - shift);
  code &= (1u << a.bits) - 1u;
  const uint meta = ids[at.y] * (a.width / a.group) + at.x / a.group;
  const float value = q27_read_t622(scales, meta, a.metadata) * float(code) + q27_read_t622(biases, meta, a.metadata);
  q27_write_t622(out, at.y * a.width + at.x, value, a.metadata);
}

struct Rope_t622 { uint width, heads, rows, rotary, activation; float theta; uint h_end, w_end, plane; };

// Rotate paired halves at each row's position: pair d's axis plane under M-RoPE, plane 0 (t) otherwise.
kernel void q27_rope_t622(device const uchar* X [[buffer(0)]], device const int* positions [[buffer(1)]],
                     constant Rope_t622& a [[buffer(2)]], device uchar* Y [[buffer(3)]],
                     uint3 at [[thread_position_in_grid]]) {
  if (at.x >= a.width || at.y >= a.heads || at.z >= a.rows) return;
  const uint base = (at.z * a.heads + at.y) * a.width, pairs = a.rotary / 2;
  float value = q27_read_t622(X, base + at.x, a.activation);
  if (at.x < a.rotary) {
    const bool second = at.x >= pairs;
    const uint d = second ? at.x - pairs : at.x;
    const uint axis = d % 3 == 1 && d < a.h_end ? 1u : (d % 3 == 2 && d < a.w_end ? 2u : 0u);
    const float angle = float(positions[axis * a.plane + at.z]) * pow(a.theta, -float(d) / float(pairs));
    const float partner = q27_read_t622(X, base + (second ? d : d + pairs), a.activation);
    value = second ? value * cos(angle) + partner * sin(angle) : value * cos(angle) - partner * sin(angle);
  }
  q27_write_t622(Y, base + at.x, value, a.activation);
}

struct Element_t622 { uint width, heads, rows, activation; };

// Attention uses the native-rounded sigmoid before multiplying its output, matching the Metal row family.
kernel void q27_gate_t622(device const uchar* O [[buffer(0)]], device const uchar* QG [[buffer(1)]],
                     constant Element_t622& a [[buffer(2)]], device uchar* OUT [[buffer(3)]],
                     uint3 at [[thread_position_in_grid]]) {
  if (at.x >= a.width || at.y >= a.heads || at.z >= a.rows) return;
  const uint index = (at.z * a.heads + at.y) * a.width + at.x;
  const float gate = q27_read_t622(QG, (at.z * a.heads + at.y) * 2 * a.width + a.width + at.x, a.activation);
  const float sig = q27_round_t622(1.0f / (1.0f + exp(-gate)), a.activation);
  q27_write_t622(OUT, index, q27_read_t622(O, index, a.activation) * sig, a.activation);
}
