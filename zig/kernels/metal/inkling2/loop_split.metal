// Main's loop gate bits in three launches over every row (m's rows a.m_stride floats apart), after loop.metal.

struct InkLoopSplitArgs { uint rows, m_stride; };

kernel void tf_inkling2_loop_norm(
    const device float* h [[buffer(0)]], const device float* m [[buffer(1)]], device bfloat* normalized [[buffer(2)]],
    constant InkLoopSplitArgs& a [[buffer(3)]], uint t [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows) return;
  const device float* inputs[2] = { h + size_t(row) * INKLING_LOOP_H,
                                    m + size_t(row) * a.m_stride };
  threadgroup float sums[2 * 8], inverse[2];
  for (uint s = 0; s < 2; ++s) {
    float ss = 0.0f;
    for (uint d = t; d < INKLING_LOOP_H; d += 256) {
      const float x = inputs[s][d];
      ss += x * x;
    }
    const float reduced = simd_sum(ss);
    if (lane == 0) sums[s * 8 + t / 32] = reduced;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) for (uint s = 0; s < 2; ++s) {
    float ss = 0.0f;
    for (uint j = 0; j < 8; ++j) ss += sums[s * 8 + j];
    inverse[s] = precise::rsqrt(precise::divide(ss, 2304.0f) + INKLING_LOOP_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device bfloat* out = normalized + size_t(row) * (2 * INKLING_LOOP_H);
  for (uint d = t; d < INKLING_LOOP_H; d += 256) {
    out[d] = bfloat(h[size_t(row) * INKLING_LOOP_H + d] * inverse[0]);
    out[INKLING_LOOP_H + d] = bfloat(m[size_t(row) * a.m_stride + d] * inverse[1]);
  }
}

// One thread a (rank, row), main's serial ascending dot over the stored BF16 norms.
kernel void tf_inkling2_loop_down(
    const device bfloat* normalized [[buffer(0)]], const device bfloat* down [[buffer(1)]],
    device float* zeta [[buffer(2)]],
    constant InkLoopSplitArgs& a [[buffer(3)]], uint t [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  const uint row = group.y, r = group.x * 32 + t;
  if (row >= a.rows || r >= INKLING_LOOP_RANK) return;
  const device bfloat* x = normalized + size_t(row) * (2 * INKLING_LOOP_H);
  const device bfloat* w = down + size_t(r) * (2 * INKLING_LOOP_H);
  float dot = 0.0f;
  for (uint d = 0; d < 2 * INKLING_LOOP_H; ++d) dot += float(w[d]) * float(x[d]);
  zeta[size_t(row) * INKLING_LOOP_RANK + r] = float(bfloat(dot));
}

kernel void tf_inkling2_loop_apply(
    const device float* h [[buffer(0)]], const device float* m [[buffer(1)]], const device float* zeta [[buffer(2)]],
    const device bfloat* forget [[buffer(3)]], const device bfloat* gain [[buffer(4)]], device float* out [[buffer(5)]],
    constant InkLoopSplitArgs& a [[buffer(6)]], uint t [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  if (row >= a.rows) return;
  const device float* z = zeta + size_t(row) * INKLING_LOOP_RANK;
  for (uint d = t; d < INKLING_LOOP_H; d += 256) {
    float fdot = 0.0f, gdot = 0.0f;
    for (uint r = 0; r < INKLING_LOOP_RANK; ++r) {
      fdot += float(forget[size_t(d) * INKLING_LOOP_RANK + r]) * z[r];
      gdot += float(gain[size_t(d) * INKLING_LOOP_RANK + r]) * z[r];
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
