// nemotron_h_puzzle (Bolt): the picks summed by weight in the latent as mlx_lm, a sequential fp32 fma (hand-authored)
#include <metal_stdlib>
using namespace metal;

kernel void tf_nemotron_combine_latent(const device bfloat* Y [[buffer(0)]],   // [rows * K, L] the picks' outputs
                                       const device float* WE [[buffer(1)]],   // [rows * K] their weights
                                       device bfloat* C [[buffer(2)]],         // [rows, L]
                                       constant uint3& dims [[buffer(3)]],     // rows, K, L
                                       uint2 pos [[thread_position_in_grid]]) {
  const uint c = pos.x, r = pos.y, K = dims.y, L = dims.z;
  if (c >= L || r >= dims.x) return;
  float routed = 0.0f;
  for (uint e = 0; e < K; e++) routed = fma(float(Y[(r * K + e) * L + c]), WE[r * K + e], routed);
  C[r * L + c] = bfloat(routed);
}
