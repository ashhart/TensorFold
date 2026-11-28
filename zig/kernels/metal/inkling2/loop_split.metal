// Main's loop gate bits in three launches over every row (m's rows a.m_stride floats apart), after loop.metal.

struct InkLoopSplitArgs { uint rows, m_stride; };

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_loop_norm(
    const device float* h [[buffer(0)]], const device float* m [[buffer(1)]], device bfloat* normalized [[buffer(2)]],
    constant InkLoopSplitArgs& a [[buffer(3)]], uint t [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows) return;
  const device float* h_row = h + size_t(row) * INKLING_LOOP_H;
  const device float* m_row = m + size_t(row) * a.m_stride;
  float hv[9], mv[9];
  float h_ss = 0.0f, m_ss = 0.0f;
  for (uint k = 0; k < 9; ++k) {
    const uint d = t + k * 256;
    const float hx = h_row[d];
    const float mx = m_row[d];
    hv[k] = hx;
    mv[k] = mx;
    h_ss += hx * hx;
    m_ss += mx * mx;
  }
  threadgroup float sums[2 * 8], inverse[2];
  const float h_red = simd_sum(h_ss);
  const float m_red = simd_sum(m_ss);
  if (lane == 0) {
    sums[t / 32] = h_red;
    sums[8 + t / 32] = m_red;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) for (uint s = 0; s < 2; ++s) {
    float ss = 0.0f;
    for (uint j = 0; j < 8; ++j) ss += sums[s * 8 + j];
    inverse[s] = precise::rsqrt(precise::divide(ss, 2304.0f) + INKLING_LOOP_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device bfloat* out = normalized + size_t(row) * (2 * INKLING_LOOP_H);
  for (uint k = 0; k < 9; ++k) {
    const uint d = t + k * 256;
    out[d] = bfloat(hv[k] * inverse[0]);
    out[INKLING_LOOP_H + d] = bfloat(mv[k] * inverse[1]);
  }
}

// One thread a (rank, row), vectorized bfloat4 serial ascending dot over the stored BF16 norms.
[[max_total_threads_per_threadgroup(32)]]
kernel void tf_inkling2_loop_down(
    const device bfloat* normalized [[buffer(0)]], const device bfloat* down [[buffer(1)]],
    device float* zeta [[buffer(2)]],
    constant InkLoopSplitArgs& a [[buffer(3)]], uint2 tid [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  const uint row = group.y, r = group.x * 32 + tid.x;
  if (row >= a.rows || r >= INKLING_LOOP_RANK) return;
  const device bfloat4* x4 = (const device bfloat4*)(normalized + size_t(row) * (2 * INKLING_LOOP_H));
  const device bfloat4* w4 = (const device bfloat4*)(down + size_t(r) * (2 * INKLING_LOOP_H));
  float dot = 0.0f;
  for (uint j = 0; j < (2 * INKLING_LOOP_H) / 4; ++j) {
    const float4 wv = float4(w4[j]);
    const float4 xv = float4(x4[j]);
    dot += wv.x * xv.x; dot += wv.y * xv.y; dot += wv.z * xv.z; dot += wv.w * xv.w;
  }
  zeta[size_t(row) * INKLING_LOOP_RANK + r] = float(bfloat(dot));
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_loop_apply(
    const device float* h [[buffer(0)]], const device float* m [[buffer(1)]], const device float* zeta [[buffer(2)]],
    const device bfloat* forget [[buffer(3)]], const device bfloat* gain [[buffer(4)]], device float* out [[buffer(5)]],
    constant InkLoopSplitArgs& a [[buffer(6)]], uint t [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  if (row >= a.rows) return;
  const device float4* z4 = (const device float4*)(zeta + size_t(row) * INKLING_LOOP_RANK);
  threadgroup float4 tg_z4[INKLING_LOOP_RANK / 4];
  if (t < INKLING_LOOP_RANK / 4) tg_z4[t] = z4[t];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = t; d < INKLING_LOOP_H; d += 256) {
    const device bfloat4* fw = (const device bfloat4*)(forget + size_t(d) * INKLING_LOOP_RANK);
    const device bfloat4* gw = (const device bfloat4*)(gain + size_t(d) * INKLING_LOOP_RANK);
    float fdot = 0.0f, gdot = 0.0f;
    for (uint j = 0; j < INKLING_LOOP_RANK / 4; ++j) {
      const float4 fv = float4(fw[j]);
      const float4 gv = float4(gw[j]);
      const float4 zv = tg_z4[j];
      fdot += fv.x * zv.x; fdot += fv.y * zv.y; fdot += fv.z * zv.z; fdot += fv.w * zv.w;
      gdot += gv.x * zv.x; gdot += gv.y * zv.y; gdot += gv.z * zv.z; gdot += gv.w * zv.w;
    }
    const float f = precise::tanh(fdot);
    const float g = precise::tanh(gdot);
    // Four separately rounded binary32 steps; do not fold these into one expression.
    const float keep = 1.0f - f;
    const float left = keep * h[size_t(row) * INKLING_LOOP_H + d];
    const float right = g * m[size_t(row) * a.m_stride + d];
    out[size_t(row) * INKLING_LOOP_H + d] = left + right;
  }
}
