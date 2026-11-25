// Clean-room fused SwiGLU activation + down_proj per-64-group x-sums.
//
// Input: gate_up[rows, 2 * inter] where each row holds [gate(0..inter), up(0..inter)].
// Output: act[rows, inter] = silu(gate) * up, plus sums[inter/64, sums_stride].
// Grid: threadgroups (inter / 64, rows) of 32 threads (1 simdgroup per group/row, zero TG barriers).
#include <metal_stdlib>
using namespace metal;

struct MlpActParams {
  uint rows;
  uint inter;
  uint sums_stride;
};

kernel void gu_activate_sums(const device bfloat* gate_up [[buffer(0)]],
                             device bfloat* act [[buffer(1)]],
                             device float* sums [[buffer(2)]],
                             constant MlpActParams& p [[buffer(3)]],
                             uint2 grp [[threadgroup_position_in_grid]],
                             ushort lane [[thread_index_in_simdgroup]]) {
  const uint g = grp.x;
  const uint row = grp.y;
  if (row >= p.rows) return;

  const device bfloat* g_ptr = gate_up + (size_t)row * (2 * p.inter);
  const device bfloat* u_ptr = g_ptr + p.inter;
  const uint i0 = g * 64 + lane;
  const uint i1 = g * 64 + 32 + lane;

  const float gv0 = float(g_ptr[i0]);
  const float gv1 = float(g_ptr[i1]);
  const float uv0 = float(u_ptr[i0]);
  const float uv1 = float(u_ptr[i1]);

  const bfloat y0 = bfloat((gv0 / (1.0f + exp(-gv0))) * uv0);
  const bfloat y1 = bfloat((gv1 / (1.0f + exp(-gv1))) * uv1);

  device bfloat* out = act + (size_t)row * p.inter;
  out[i0] = y0;
  out[i1] = y1;

  const float gs = simd_sum(float(y0) + float(y1));
  if (lane == 0) {
    sums[(size_t)g * p.sums_stride + row] = gs;
  }
}

// Fused GeGLU (tanh-approx GELU) activation + down_proj per-64-group x-sums (Gemma 2 / 3 / 4).
// Grid: threadgroups (inter / 64, rows) of 32 threads.
kernel void geglu_activate_sums(const device bfloat* gate_up [[buffer(0)]],
                                device bfloat* act [[buffer(1)]],
                                device float* sums [[buffer(2)]],
                                constant MlpActParams& p [[buffer(3)]],
                                uint2 grp [[threadgroup_position_in_grid]],
                                ushort lane [[thread_index_in_simdgroup]]) {
  const uint g = grp.x;
  const uint row = grp.y;
  if (row >= p.rows) return;

  const device bfloat* g_ptr = gate_up + (size_t)row * (2 * p.inter);
  const device bfloat* u_ptr = g_ptr + p.inter;
  const uint i0 = g * 64 + lane;
  const uint i1 = g * 64 + 32 + lane;

  const float gv0 = float(g_ptr[i0]);
  const float gv1 = float(g_ptr[i1]);
  const float uv0 = float(u_ptr[i0]);
  const float uv1 = float(u_ptr[i1]);

  constexpr float c0 = 0.7978845608028654f; // sqrt(2 / pi)
  constexpr float c1 = 0.044715f;
  const float ge0 = (gv0 >= 10.0f) ? gv0 : ((gv0 <= -10.0f) ? 0.0f : 0.5f * gv0 * (1.0f + tanh(clamp(c0 * (gv0 + c1 * gv0 * gv0 * gv0), -15.0f, 15.0f))));
  const float ge1 = (gv1 >= 10.0f) ? gv1 : ((gv1 <= -10.0f) ? 0.0f : 0.5f * gv1 * (1.0f + tanh(clamp(c0 * (gv1 + c1 * gv1 * gv1 * gv1), -15.0f, 15.0f))));

  const bfloat y0 = bfloat(ge0 * uv0);
  const bfloat y1 = bfloat(ge1 * uv1);

  device bfloat* out = act + (size_t)row * p.inter;
  out[i0] = y0;
  out[i1] = y1;

  if (p.sums_stride != 0) {
    const float gs = simd_sum(float(y0) + float(y1));
    if (lane == 0) sums[(size_t)g * p.sums_stride + row] = gs;
  }
}

struct SoftcapParams {
  uint count;
  float cap;
};

// In-place final logit softcapping: logits[i] = cap * tanh(logits[i] / cap) (Gemma 2/3/4).
kernel void logit_softcap(device bfloat* logits [[buffer(0)]],
                          constant SoftcapParams& p [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {
  if (tid >= p.count) return;
  const float v = float(logits[tid]);
  logits[tid] = bfloat(p.cap * tanh(clamp(v / p.cap, -15.0f, 15.0f)));
}
