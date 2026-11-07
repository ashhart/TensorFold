// Gemma 4 E4B per-layer inputs.
#include <metal_stdlib>
using namespace metal;

// Each pipeline's constants: D (hidden), P (per-layer width), L (layers), T (threads a row), GELU_C0/C1 (kernels.zig).

// The per-layer table's rows [rows, L * P] times sqrt(P): 4-bit codes to code * scale + bias, then a bf16 multiply.
kernel void gemma_ple_embed(const device uint* ids [[buffer(0)]], const device uchar* weight [[buffer(1)]],
                            const device bfloat* scales [[buffer(2)]], const device bfloat* biases [[buffer(3)]],
                            device bfloat* out [[buffer(4)]], constant float& scale [[buffer(5)]],
                            uint2 pos [[thread_position_in_grid]]) {
  const uint b = pos.x, r = pos.y;
  const size_t id = ids[r];
  const uchar v = weight[id * (L * P / 2) + b];
  const size_t g = id * (L * P / 64) + (2 * b) / 64;
  const float s = float(scales[g]), z = float(biases[g]);
  out[size_t(r) * L * P + 2 * b] = bfloat(float(bfloat(s * float(v & 0xf) + z)) * scale);
  out[size_t(r) * L * P + 2 * b + 1] = bfloat(float(bfloat(s * float(v >> 4) + z)) * scale);
}

// A row's T threads' sum, in a fixed order whatever the window.
inline float ple_reduce(float v, threadgroup float* p, uint lane, uint sg) {
  const float s = simd_sum(v);
  if (lane == 0) p[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float out = 0.0f;
  for (int q = 0; q < T / 32; q++) out += p[q];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out;
}

kernel void gemma_ple_combine(const device bfloat* PRJ [[buffer(0)]], const device bfloat* TOK [[buffer(1)]],
                              const device bfloat* NW [[buffer(2)]], constant float& eps [[buffer(3)]],
                              constant float& psc [[buffer(4)]], constant float& isc [[buffer(5)]],
                              device bfloat* PLI [[buffer(6)]], uint2 tp [[thread_position_in_threadgroup]],
                              uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
  // One group per (layer, row), a thread per column: DeepMind _modules.py encode_per_layer_input.
  threadgroup float part[T / 32];
  const uint col = tp.x, layer = tg.x, row = tg.y;
  const size_t at = size_t(row) * L * P + size_t(layer) * P + col;
  // The model projection times 1/sqrt(D), as a bf16 scalar, rounded to bf16.
  const float x = float(bfloat(float(PRJ[at]) * float(bfloat(psc))));
  // RMSNorm over the P columns: x * rsqrt(mean(x^2) + eps) rounded to bf16, then times the weight, rounded to bf16.
  const float ms = ple_reduce(x * x, part, lane, sg) / float(P);
  const float xn = float(bfloat(x * precise::rsqrt(ms + eps)));
  const float nrm = float(bfloat(xn * float(NW[col])));
  // (norm + table row) times rsqrt(2) as a bf16 scalar, each op rounded to bf16.
  const float sum = float(bfloat(nrm + float(TOK[at])));
  PLI[at] = bfloat(sum * float(bfloat(isc)));
}
