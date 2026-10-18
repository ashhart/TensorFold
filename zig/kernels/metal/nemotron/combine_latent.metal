// nemotron_h_puzzle (Bolt): each row's routed picks combined by weight in the latent, as mlx_lm's
// (y * scores[..., None]).sum(axis=-2).astype(bf16) before fc2_latent_proj, with add_norm_moe_xs's routed sum
// (a sequential fma over the picks into fp32, one bf16 rounding). Hand-authored for this PR.
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
