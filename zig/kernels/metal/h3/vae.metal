// H3 video VAE on Metal: fp32 matmul kernels for the ViT decoder's projections.
// Reference: zig/kernels/metal/h3/h3_rows.metal (the row matmul pattern).

// Y[r, n] = B[n] + sum_k X[r, k] W[n, k]. Threads [N, R, 1], fp32 throughout.
[[kernel]] void h3_vae_matmul(
  const device float* X [[buffer(0)]],   // [R, K]
  const device float* W [[buffer(1)]],   // [N, K] (row-major, the safetensors layout)
  const device float* B [[buffer(2)]],   // [N]
  device float* Y [[buffer(3)]],         // [R, N]
  constant uint* P [[buffer(4)]],        // P[0]=R, P[1]=K, P[2]=N, P[3]=row0
  uint3 at [[thread_position_in_grid]]) {
  const uint R = P[0], K = P[1], N = P[2], row0 = P[3];
  if (at.x >= N || at.y >= R) return;
  const device float* x = X + (uint)at.y * K;
  const device float* w = W + (uint)at.x * K;
  float sum = 0.0f;
  for (uint k = 0; k < K; k++) sum += x[k] * w[k];
  Y[(row0 + at.y) * N + at.x] = sum + B[at.x];
}

// SwiGLU elementwise: out[r, f] = fused[r, f + F] * silu(fused[r, f]).
[[kernel]] void h3_vae_swiglu(
  const device float* FUSED [[buffer(0)]],
  device float* OUT [[buffer(1)]],
  constant uint* P [[buffer(2)]],        // P[0] = F
  uint3 at [[thread_position_in_grid]]) {
  const uint F = P[0];
  const uint r = at.y, f = at.x;
  const float gate = FUSED[r * 2 * F + f];
  const float value = FUSED[r * 2 * F + F + f];
  OUT[r * F + f] = value * gate / (1.0f + exp(-gate));
}

// RMSNorm in fp32 with a learned weight: one threadgroup per row.
[[kernel]] void h3_vae_rmsnorm(
  const device float* X [[buffer(0)]],
  const device float* W [[buffer(1)]],
  device float* Y [[buffer(2)]],
  constant uint* P [[buffer(3)]],        // P[0] = D
  uint tid [[thread_index_in_threadgroup]],
  uint tg [[threadgroup_position_in_grid]]) {
  const uint D = P[0];
  float acc = 0.0f;
  for (uint i = tid; i < D; i += 256) acc += X[tg * D + i] * X[tg * D + i];
  acc = simd_sum(acc);
  const float inv = rsqrt(acc / float(D) + 1e-5f);
  for (uint i = tid; i < D; i += 256) Y[tg * D + i] = X[tg * D + i] * inv * W[i];
}

// Full fp32 attention: online softmax over key blocks. One threadgroup per
// head-row. The block max reduces through shared memory comparisons.
[[kernel]] void h3_vae_attention(
  const device float* Q [[buffer(0)]],
  const device float* K [[buffer(1)]],
  const device float* V [[buffer(2)]],
  device float* Y [[buffer(3)]],
  constant uint* P [[buffer(4)]],
  uint3 tg [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint threads [[threads_per_threadgroup]]) {
  const uint rows = P[0], hd = P[1];
  const uint head = tg.y, r = tg.x;
  const device float* q = Q + (head * rows + r) * hd;
  threadgroup float acc[256];
  threadgroup float bmax[1];
  float run_max = -1e30f;
  float run_sum = 0.0f;
  for (uint i = tid; i < hd; i += threads) acc[i] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint BLOCK = 256;
  for (uint k0 = 0; k0 < rows; k0 += BLOCK) {
    const uint kend = min(k0 + BLOCK, rows);
    if (tid == 0) bmax[0] = -1e30f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = k0 + tid; k < kend; k += threads) {
      float dot = 0.0f;
      for (uint i = 0; i < hd; i++) dot += q[i] * K[(head * rows + k) * hd + i];
      dot *= rsqrt(float(hd));
      if (dot > bmax[0]) bmax[0] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float bm = max(bmax[0], run_max);
    const float rescale = exp(run_max - bm);
    for (uint k = k0 + tid; k < kend; k += threads) {
      float dot = 0.0f;
      for (uint i = 0; i < hd; i++) dot += q[i] * K[(head * rows + k) * hd + i];
      const float w = exp(dot * rsqrt(float(hd)) - bm);
      run_sum += w;
      for (uint i = 0; i < hd; i++) acc[i] += w * V[(head * rows + k) * hd + i];
    }
    for (uint i = tid; i < hd; i += threads) acc[i] *= rescale;
    run_max = bm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = tid; i < hd; i += threads) Y[(head * rows + r) * hd + i] = acc[i] / run_sum;
}

// LayerNorm affine: mean/var in fp32, then (x - mean) / std * w + b.
[[kernel]] void h3_vae_layernorm(
  const device float* X [[buffer(0)]],
  const device float* W [[buffer(1)]],
  const device float* B [[buffer(2)]],
  device float* Y [[buffer(3)]],
  constant uint* P [[buffer(4)]],        // P[0] = D, P[1] = eps as bits? no: eps folded into P[1] unused; eps is compile-time 1e-5
  uint tid [[thread_index_in_threadgroup]],
  uint tg [[threadgroup_position_in_grid]]) {
  const uint D = P[0];
  float sum = 0.0f;
  for (uint i = tid; i < D; i += 256) sum += X[tg * D + i];
  sum = simd_sum(sum);
  const float mean = sum / float(D);
  float m2 = 0.0f;
  for (uint i = tid; i < D; i += 256) { const float t = X[tg * D + i] - mean; m2 += t * t; }
  m2 = simd_sum(m2);
  const float std = rsqrt(m2 / float(D) + 1e-5f);
  for (uint i = tid; i < D; i += 256) Y[tg * D + i] = (X[tg * D + i] - mean) * std * W[i] + B[i];
}

// post_quant 1x1x1 conv: out[v, o] = sum_i in[v, i] w[o, i] + b[o]. Threads [V, O, 1].
[[kernel]] void h3_vae_postquant(
  const device float* IN [[buffer(0)]],   // [V, 24]
  const device float* W [[buffer(1)]],    // [24, 24]
  const device float* B [[buffer(2)]],    // [24]
  device float* OUT [[buffer(3)]],        // [V, 24]
  constant uint* P [[buffer(4)]],        // P[0]=V, P[1]=C
  uint3 at [[thread_position_in_grid]]) {
  const uint V = P[0], C = P[1];
  if (at.x >= C || at.y >= V) return;
  float sum = 0.0f;
  for (uint i = 0; i < C; i++) sum += IN[at.y * C + i] * W[at.x * C + i];
  OUT[at.y * C + at.x] = sum + B[at.x];
}
