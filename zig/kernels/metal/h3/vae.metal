// H3 video VAE on Metal: the ViT decoder block kernels in fp32.
// Reference: zig/src/families/h3/vae/reference/vae_video.py (MLX port target).
// These run beside the transformer kernels; names must not collide, hence vae_*.

// RMSNorm in fp32: x * rsqrt(mean(x^2) + eps) * weight. One threadgroup per row.
[[kernel]] void h3_vae_rmsnorm(
  const device float* X [[buffer(0)]],
  const device float* W [[buffer(1)]],
  device float* Y [[buffer(2)]],
  constant uint* P [[buffer(3)]],          // P[0] = D
  uint tid [[thread_index_in_threadgroup]],
  uint tg [[threadgroup_position_in_grid]]) {
  const uint D = P[0];
  float acc = 0.0f;
  for (uint i = tid; i < D; i += 256) acc += X[tg * D + i] * X[tg * D + i];
  // one-warp reduction through simd_sum
  acc = simd_sum(acc);
  const float inv = rsqrt(acc / float(D) + 1e-5f);
  for (uint i = tid; i < D; i += 256) Y[tg * D + i] = X[tg * D + i] * inv * W[i];
}

// SwiGLU FFN: out = w2b applied after silu(gate) * value, elementwise part only.
// [rows, 2F] fused (first F gate, next F value) -> [rows, F] gated output.
[[kernel]] void h3_vae_swiglu(
  const device float* FUSED [[buffer(0)]],
  device float* OUT [[buffer(1)]],
  constant uint* P [[buffer(2)]],          // P[0] = F
  uint tid [[thread_index_in_threadgroup]],
  uint tg [[threadgroup_position_in_grid]]) {
  const uint F = P[0];
  for (uint i = tid; i < F; i += 256) {
    const float gate = FUSED[tg * 2 * F + i];
    const float value = FUSED[tg * 2 * F + F + i];
    OUT[tg * F + i] = value * gate / (1.0f + exp(-gate));
  }
}

// Attention: softmax(q k^T) v in fp32 within one tile; q and k are already
// RMS-normalized and rotated by the caller. One threadgroup z per head.
[[kernel]] void h3_vae_attention(
  const device float* Q [[buffer(0)]],     // [heads, rows, head_dim]
  const device float* K [[buffer(1)]],
  const device float* V [[buffer(2)]],
  device float* Y [[buffer(3)]],           // [heads, rows, head_dim]
  constant uint* P [[buffer(4)]],          // P[0] = rows, P[1] = head_dim
  uint3 tg [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint simd_lane [[thread_index_in_simdgroup]],
  uint simd_group [[thread_index_in_simdgroup]]) {
  const uint rows = P[0], hd = P[1];
  const uint head = tg.z, r = tg.x;
  float scores[512];
  float max_s = -1e30f;
  for (uint k = 0; k < rows; k++) {
    float dot = 0.0f;
    for (uint i = 0; i < hd; i++) dot += Q[(head * rows + r) * hd + i] * K[(head * rows + k) * hd + i];
    scores[k] = dot * rsqrt(float(hd));
    if (scores[k] > max_s) max_s = scores[k];
  }
  float sum = 0.0f;
  for (uint k = 0; k < rows; k++) { scores[k] = exp(scores[k] - max_s); sum += scores[k]; }
  for (uint i = tid; i < hd; i += 256) {
    float out = 0.0f;
    for (uint k = 0; k < rows; k++) out += scores[k] * V[(head * rows + k) * hd + i];
    Y[(head * rows + r) * hd + i] = out / sum;
  }
}
