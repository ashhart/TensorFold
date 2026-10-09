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

struct Norm { uint width, rows, activation, gain, residual; float eps; };

// Each thread owns 16 contiguous values, then simd_sum and simdgroups in increasing order as row_glue.add_norm.
[[max_total_threads_per_threadgroup(1024)]]
kernel void q27_norm(device const uchar* X [[buffer(0)]], device const uchar* R [[buffer(1)]],
                     device const uchar* W [[buffer(2)]], constant Norm& a [[buffer(3)]],
                     device uchar* H [[buffer(4)]], device uchar* Y [[buffer(5)]],
                     uint2 group [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sums[32];
  const uint row = group.y, base = row * a.width + 16 * t;
  float values[16], square = 0.0f;
  for (uint j = 0; j < 16; ++j) {
    float h = q27_read(X, base + j, a.activation);
    if (a.residual) h = q27_round(h + q27_read(R, base + j, a.activation), a.activation);
    values[j] = h;
    q27_write(H, base + j, h, a.activation);
    square = fma(h, h, square);
  }
  square = simd_sum(square);
  if (lane == 0) sums[sg] = square;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < a.width / 512; ++s) total += sums[s];
  const float inv = rsqrt(total / float(a.width) + a.eps);
  for (uint j = 0; j < 16; ++j) q27_write(Y, base + j, q27_read(W, 16 * t + j, a.gain) * (values[j] * inv), a.activation);
}

struct Head { uint width, rows, stride, activation, gain; float eps; };

// Head RMSNorm retains its native-input rounding before gain multiplication, as the shared Metal RMS implementation.
kernel void q27_head_norm(device const uchar* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
                          constant Head& a [[buffer(2)]], device uchar* Y [[buffer(3)]],
                          uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint threads [[threads_per_threadgroup]]) {
  threadgroup float sums[32], inverse[1];
  float square = 0.0f;
  uint first = 4 * t;
  while (first < a.width) {
    for (uint j = 0; j != 4; ++j) {
      const uint column = first + j;
      const float value = column < a.width ? q27_read(X, row * a.stride + column, a.activation) : 0.0f;
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
      const float value = q27_read(X, row * a.stride + at, a.activation);
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

struct Rope { uint width, heads, rows, rotary, activation; float theta; };

// Rotate paired halves at each row's absolute position; untouched dimensions copy their native bits.
kernel void q27_rope(device const uchar* X [[buffer(0)]], device const int* positions [[buffer(1)]],
                     constant Rope& a [[buffer(2)]], device uchar* Y [[buffer(3)]],
                     uint3 at [[thread_position_in_grid]]) {
  if (at.x >= a.width || at.y >= a.heads || at.z >= a.rows) return;
  const uint base = (at.z * a.heads + at.y) * a.width, pairs = a.rotary / 2;
  float value = q27_read(X, base + at.x, a.activation);
  if (at.x < a.rotary) {
    const bool second = at.x >= pairs;
    const uint d = second ? at.x - pairs : at.x;
    const float angle = float(positions[at.z]) * pow(a.theta, -float(d) / float(pairs));
    const float partner = q27_read(X, base + (second ? d : d + pairs), a.activation);
    value = second ? value * cos(angle) + partner * sin(angle) : value * cos(angle) - partner * sin(angle);
  }
  q27_write(Y, base + at.x, value, a.activation);
}

struct Element { uint width, heads, rows, activation; };

// SiLU(gate) times up reads stacked [gate|up] rows with one final native activation rounding.
kernel void q27_mlp(device const uchar* GU [[buffer(0)]], constant Element& a [[buffer(1)]],
                    device uchar* OUT [[buffer(2)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= a.width || at.y >= a.rows) return;
  const float gate = q27_read(GU, at.y * 2 * a.width + at.x, a.activation);
  const float up = q27_read(GU, at.y * 2 * a.width + a.width + at.x, a.activation);
  q27_write(OUT, at.y * a.width + at.x, gate / (1.0f + exp(-gate)) * up, a.activation);
}

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

struct Post { uint width, heads, rows, stride, offset, activation, gain; float eps; };

// GDN post-norm retains the rounded normalized result before the fp32 SiLU product, as row_glue.gdn_post.
kernel void q27_gdn_post(device const uchar* REC [[buffer(0)]], device const uchar* Z [[buffer(1)]],
                         device const uchar* W [[buffer(2)]], constant Post& a [[buffer(3)]],
                         device uchar* OUT [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint head = group.y, row = group.z, per = a.width / 32;
  float values[16], square = 0.0f;
  for (uint j = 0; j < per; ++j) {
    values[j] = q27_read(REC, (row * a.heads + head) * a.width + lane * per + j, a.activation);
    square += values[j] * values[j];
  }
  const float inv = rsqrt(simd_sum(square) / float(a.width) + a.eps);
  for (uint j = 0; j < per; ++j) {
    const uint d = lane * per + j;
    const float norm = q27_round(q27_read(W, d, a.gain) * (values[j] * inv), a.activation);
    const float z = q27_read(Z, row * a.stride + a.offset + head * a.width + d, a.activation);
    q27_write(OUT, (row * a.heads + head) * a.width + d, z / (1.0f + exp(-z)) * norm, a.activation);
  }
}
