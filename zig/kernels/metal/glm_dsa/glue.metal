// GLM-5.3's glue for lane rows in fp32: embeddings, residual adds with the next RMSNorm, SiLU times up.
#include <metal_stdlib>
using namespace metal;

struct GlueArgs {
  uint width, x_stride, y_stride, parts;
  uint part_stride, pad0, pad1, pad2;
  float eps, pad3, pad4, pad5;
};

inline float glm_glue_lanes(float v) {
  for (ushort d = 16; d > 0; d >>= 1) v += simd_shuffle_xor(v, d);
  return v;
}

// Row r of the hidden state from token ids[r]: E4M3 codes times the row's fp32 scale (the reference's decode).
kernel void glm_embed(const device uint* ids [[buffer(0)]], const device uchar* fp8 [[buffer(1)]],
                      const device float* scale [[buffer(2)]], constant GlueArgs& a [[buffer(3)]],
                      device float* h [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                      uint t [[thread_index_in_threadgroup]]) {
  const uint id = ids[r];
  const float s = scale[id];
  for (uint i = t; i < a.width; i += 256) {
    const uint code = fp8[ulong(id) * a.width + i];
    const float v = float(as_type<half>(ushort((code & 0x7Fu) << 7))) * 256.0f;
    h[ulong(r) * a.y_stride + i] = ((code & 0x80u) ? -v : v) * s;
  }
}

// h[r] += the parts' sum (in part order), then y[r] = w * (h[r] / rms(h[r])); parts 0 only normalizes.
kernel void glm_add_rms(device float* h [[buffer(0)]], const device float* add [[buffer(1)]],
                        const device bfloat* w [[buffer(2)]], constant GlueArgs& a [[buffer(3)]],
                        device float* y [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                        uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8];
  device float* x = h + ulong(r) * a.x_stride;
  float s = 0.0f;
  for (uint i = t; i < a.width; i += 256) {
    float v = x[i];
    if (a.parts > 0) {
      float sum = add[ulong(r) * a.x_stride + i];
      for (uint p = 1; p < a.parts; ++p) sum += add[ulong(p) * a.part_stride + ulong(r) * a.x_stride + i];
      v += sum;
      x[i] = v;
    }
    s = fma(v, v, s);
  }
  s = glm_glue_lanes(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (uint g = 0; g < 8; ++g) var += part[g];
  const float rs = precise::rsqrt(var / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += 256) y[ulong(r) * a.y_stride + i] = float(w[i]) * (x[i] * rs);
}

// act[p][c] = silu(g) * u for gate at column c and up at column width + c of row p (x_stride apart).
kernel void glm_silu_mul(const device float* gu [[buffer(0)]], constant GlueArgs& a [[buffer(1)]],
                         device float* act [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]]) {
  const uint p = tg.y, c = tg.x * 256 + t;
  if (c >= a.width) return;
  const float g = gu[ulong(p) * a.x_stride + c], u = gu[ulong(p) * a.x_stride + a.width + c];
  act[ulong(p) * a.y_stride + c] = precise::divide(g, 1.0f + precise::exp(-g)) * u;
}

// Row r of dst = row idx[r] of src, `a.x` 32-bit words a row (a.y and a.z the source and destination strides).
kernel void glm_gather(const device uint* src [[buffer(0)]], const device uint* idx [[buffer(1)]],
                       constant uint4& a [[buffer(2)]], device uint* dst [[buffer(3)]],
                       uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  const device uint* s = src + ulong(idx[r]) * a.y;
  for (uint i = t; i < a.x; i += 256) dst[ulong(r) * a.z + i] = s[i];
}
