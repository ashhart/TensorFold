struct InkLoopRolesArgs { uint rows, stride, roles; };
constant uint INK2_GATE_SCRATCH_STRIDE = 256u * (2u * INKLING_LOOP_H + INKLING_LOOP_RANK);

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_loop_roles_norm(
    const device float* h0 [[buffer(0)]], const device float* h1 [[buffer(1)]],
    const device float* h2 [[buffer(2)]], const device float* m0 [[buffer(3)]],
    const device float* m1 [[buffer(4)]], const device float* m2 [[buffer(5)]],
    device bfloat* normalized [[buffer(6)]], constant InkLoopRolesArgs& a [[buffer(7)]],
    uint t [[thread_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg.x, role = tg.z;
  if (row >= a.rows || role >= a.roles) return;
  const device float* h = (role == 0u) ? h0 : ((role == 1u) ? h1 : h2);
  const device float* m = (role == 0u) ? m0 : ((role == 1u) ? m1 : m2);
  const device float* inputs[2] = { h + size_t(row) * INKLING_LOOP_H,
                                    m + size_t(row) * a.stride };
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
  device bfloat* y = normalized + size_t(role) * INK2_GATE_SCRATCH_STRIDE + size_t(row) * (2 * INKLING_LOOP_H);
  for (uint d = t; d < INKLING_LOOP_H; d += 256) {
    y[d] = bfloat(inputs[0][d] * inverse[0]);
    y[INKLING_LOOP_H + d] = bfloat(inputs[1][d] * inverse[1]);
  }
}

// Each role keeps the serial column chain of the plain gate.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_loop_roles_zeta(
    const device bfloat* normalized [[buffer(0)]], const device bfloat* down0 [[buffer(1)]],
    const device bfloat* down1 [[buffer(2)]], const device bfloat* down2 [[buffer(3)]],
    device bfloat* zeta [[buffer(4)]], constant InkLoopRolesArgs& a [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg.x * 256 + sg * 32 + lane, row = tg.y, role = tg.z;
  if (r >= INKLING_LOOP_RANK || row >= a.rows || role >= a.roles) return;
  const device bfloat* down = (role == 0u) ? down0 : ((role == 1u) ? down1 : down2);
  const device uint4* w = (const device uint4*)(down + size_t(r) * (2 * INKLING_LOOP_H));
  const device uint4* x = (const device uint4*)(normalized + size_t(role) * INK2_GATE_SCRATCH_STRIDE + size_t(row) * (2 * INKLING_LOOP_H));
  float dot = 0.0f;
  for (uint j = 0; j < 2 * INKLING_LOOP_H / 8; j += 8) {
    uint4 wv[8];
    for (uint i = 0; i < 8; ++i) wv[i] = w[j + i];
    for (uint i = 0; i < 8; ++i) {
      const uint4 xv = x[j + i];
      dot = ink_loop_pair(dot, wv[i].x, xv.x);
      dot = ink_loop_pair(dot, wv[i].y, xv.y);
      dot = ink_loop_pair(dot, wv[i].z, xv.z);
      dot = ink_loop_pair(dot, wv[i].w, xv.w);
    }
  }
  zeta[size_t(role) * INK2_GATE_SCRATCH_STRIDE + row * INKLING_LOOP_RANK + r] = bfloat(dot);
}

// The third: each output's forget and gain dots over zeta in rank order per grid.z role, then the four-step blend.
[[max_total_threads_per_threadgroup(32)]]
kernel void tf_inkling2_loop_roles_blend(
    const device float* h0 [[buffer(0)]], const device float* h1 [[buffer(1)]],
    const device float* h2 [[buffer(2)]], const device float* m0 [[buffer(3)]],
    const device float* m1 [[buffer(4)]], const device float* m2 [[buffer(5)]],
    const device bfloat* zeta [[buffer(6)]], const device bfloat* forget0 [[buffer(7)]],
    const device bfloat* forget1 [[buffer(8)]], const device bfloat* forget2 [[buffer(9)]],
    const device bfloat* gain0 [[buffer(10)]], const device bfloat* gain1 [[buffer(11)]],
    const device bfloat* gain2 [[buffer(12)]], device float* out0 [[buffer(13)]],
    device float* out1 [[buffer(14)]], device float* out2 [[buffer(15)]],
    constant InkLoopRolesArgs& a [[buffer(16)]], uint3 at [[thread_position_in_grid]]) {
  const uint d = at.x, row = at.y, role = at.z;
  if (d >= INKLING_LOOP_H || row >= a.rows || role >= a.roles) return;
  const device float* h = (role == 0u) ? h0 : ((role == 1u) ? h1 : h2);
  const device float* m = (role == 0u) ? m0 : ((role == 1u) ? m1 : m2);
  const device bfloat* forget = (role == 0u) ? forget0 : ((role == 1u) ? forget1 : forget2);
  const device bfloat* gain = (role == 0u) ? gain0 : ((role == 1u) ? gain1 : gain2);
  device float* out = (role == 0u) ? out0 : ((role == 1u) ? out1 : out2);
  zeta += size_t(role) * INK2_GATE_SCRATCH_STRIDE;
  {
    float fdot = 0.0f, gdot = 0.0f;
    for (uint r = 0; r < INKLING_LOOP_RANK; ++r) {
      const float z = float(zeta[row * INKLING_LOOP_RANK + r]);
      fdot += float(forget[size_t(d) * INKLING_LOOP_RANK + r]) * z;
      gdot += float(gain[size_t(d) * INKLING_LOOP_RANK + r]) * z;
    }
    const float f = precise::tanh(fdot);
    const float g = precise::tanh(gdot);
    // Four separately rounded binary32 steps; do not fold these into one expression.
    const float keep = 1.0f - f;
    const float left = keep * h[size_t(row) * INKLING_LOOP_H + d];
    const float right = g * m[size_t(row) * a.stride + d];
    out[size_t(row) * INKLING_LOOP_H + d] = left + right;
  }
}
