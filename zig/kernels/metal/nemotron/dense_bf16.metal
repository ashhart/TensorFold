// y = x R^T, R bf16 [N, K] (no scales): nemotron_h_puzzle's latent projections fc1/fc2_latent_proj and the gate,
// which are stored in full bf16. Same shape as tf_residual (a simdgroup an output column, 8 rows a pass) but it
// WRITES y instead of accumulating, so a fresh projection needs no zeroing first. Hand-authored for this PR.
#include <metal_stdlib>
using namespace metal;

kernel void tf_nemotron_dense_bf16(const device bfloat* x [[buffer(0)]],
                                   const device bfloat* w [[buffer(1)]],
                                   device bfloat* y [[buffer(2)]],
                                   constant uint3& dims [[buffer(3)]],
                                   uint2 pos [[thread_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]]) {
  const uint rows = dims.x, n = dims.y, width = dims.z, j = pos.x / 32, r0 = pos.y * 8;
  if (j >= n) return;
  const device bfloat4* wj = (const device bfloat4*)(w + size_t(j) * width);
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (uint i = lane; i < width / 4; i += 32) {
    const float4 wi = float4(wj[i]);
    for (uint r = 0; r < 8 && r0 + r < rows; r++) acc[r] += dot(wi, float4(((const device bfloat4*)(x + size_t(r0 + r) * width))[i]));
  }
  for (uint r = 0; r < 8; r++) {
    const float s = simd_sum(acc[r]);
    if (lane == 0 && r0 + r < rows) y[size_t(r0 + r) * n + j] = bfloat(s);
  }
}
