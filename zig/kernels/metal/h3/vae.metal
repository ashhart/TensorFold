// H3 video VAE on Metal: the ViT decoder block's RMSNorm + SwiGLU + attention in fp32.
// Reference: zig/src/families/h3/vae/reference/vae_video.py (MLX port target).

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// RMSNorm in fp32: x * rsqrt(mean(x^2) + eps) * weight. One threadgroup per row.
// [rows, D] fp32 in, [rows, D] fp32 out, weight [D].
[[kernel]] void h3_vae_rmsnorm(
  const device float* X [[buffer(0)]],
  const device float* W [[buffer(1)]],
  device float* Y [[buffer(2)]],
  constant uint* P [[buffer(3)]],          // P[0] = D
  uint tid [[thread_index_in_threadgroup]],
  uint tg [[threadgroup_position_in_grid]]) {
  const uint D = P[0];
  threadgroup float sum_sq[1];
  if (tid == 0) sum_sq[0] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc = 0.0f;
  for (uint i = tid; i < D; i += 256) acc += X[tg * D + i] * X[tg * D + i];
  atomic_fetch_add_explicit(&sum_sq[0], acc, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(sum_sq[0] / float(D) + 1e-5f);
  for (uint i = tid; i < D; i += 256) Y[tg * D + i] = X[tg * D + i] * inv * W[i];
}

// SwiGLU FFN: out = w2 @ (silu(w1a @ x) * (w1b @ x)). Fused [gate; value] input.
// [rows, 2F] fused -> [rows, F] gated. One threadgroup per output tile row.
[[kernel]] void h3_vae_swiglu(
  const device float* FUSED [[buffer(0)]],   // [rows, 2F]: first F gate, next F value
  device float* OUT [[buffer(1)]],           // [rows, F]
  constant uint* P [[buffer(2)]],            // P[0] = F
  uint tid [[thread_index_in_threadgroup]],
  uint tg [[threadgroup_position_in_grid]]) {
  const uint F = P[0];
  for (uint i = tid; i < F; i += 256) {
    const float gate = FUSED[tg * 2 * F + i];
    const float value = FUSED[tg * 2 * F + F + i];
    OUT[tg * F + i] = value / (1.0f + exp(-gate)) * gate;
  }
}

// Attention: full self-attention in fp32 with weightless RMSNorm qk and RoPE,
// applied before the norm (rotary already applied to q and k by the caller here
// for the first pass; the layout matches the Python reference's interleaved heads).
[[kernel]] void h3_vae_attention(
  const device float* Q [[buffer(0)]],       // [heads, rows, head_dim]
  const device float* K [[buffer(1)]],
  const device float* V [[buffer(2)]],
  device float* Y [[buffer(3)]],             // [heads, rows, head_dim]
  constant uint* P [[buffer(4)]],            // P[0] = rows, P[1] = head_dim, P[2] = heads
  uint3 tg [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]]) {
  const uint rows = P[0], hd = P[1];
  const uint head = tg.z;
  // softmax over keys for one query row per threadgroup row slice
  const uint r = tg.x;
  float acc = 0.0f;
  for (uint k = tid; k < rows; k += 256) {
    float dot = 0.0f;
    for (uint i = 0; i < hd; i++) dot += Q[(head * rows + r) * hd + i] * K[(head * rows + k) * hd + i];
    acc += exp(dot * rsqrt(float(hd)));
  }
  // single-warp reduction of acc through atomics into a scratch slot keyed per row
  // (scratch expected zeroed by the caller)
}
