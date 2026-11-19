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
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(acc / float(D) + 1e-5f);
  for (uint i = tid; i < D; i += 256) Y[tg * D + i] = X[tg * D + i] * inv * W[i];
}

// Full attention reading the interleaved qkv layout directly:
// QKV[r, h*3*hd + part*hd + i]. Softmax(q k^T / sqrt(hd)) v per head.
// Writes attended[h][r][i] (transposed to heads-major for the out projection).
[[kernel]] void h3_vae_attention_il(
  const device float* QKV [[buffer(0)]],
  device float* Y [[buffer(1)]],
  constant uint* P [[buffer(2)]],          // P[0]=rows
  uint tg [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint threads [[threads_per_threadgroup]]) {
  const uint rows = P[0];
  const uint hd = 64, heads = 32, dim = heads * hd;
  const uint r = tg % rows; const uint head = tg / rows;
  const device float* q = QKV + r * 3 * dim + head * 3 * hd;
  const device float* kbase = QKV;
  const device float* vbase = QKV;
  threadgroup float acc[64];
  float run_max = -1e30f;
  float run_sum = 0.0f;
  for (uint i = tid; i < hd; i += threads) acc[i] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint BLOCK = 256;
  for (uint k0 = 0; k0 < rows; k0 += BLOCK) {
    const uint kend = min(k0 + BLOCK, rows);
    threadgroup float bmax[1];
    if (tid == 0) bmax[0] = -1e30f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = k0 + tid; k < kend; k += threads) {
      float dot = 0.0f;
      const device float* kk = kbase + k * 3 * dim + head * 3 * hd + hd;
      for (uint i = 0; i < hd; i++) dot += q[i] * kk[i];
      dot *= rsqrt(float(hd));
      if (dot > bmax[0]) bmax[0] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float bm = max(bmax[0], run_max);
    const float rescale = exp(run_max - bm);
    for (uint k = k0 + tid; k < kend; k += threads) {
      float dot = 0.0f;
      const device float* kk = kbase + k * 3 * dim + head * 3 * hd + hd;
      for (uint i = 0; i < hd; i++) dot += q[i] * kk[i];
      const float wgt = exp(dot * rsqrt(float(hd)) - bm);
      run_sum += wgt;
      const device float* vv = vbase + k * 3 * dim + head * 3 * hd + 2 * hd;
      for (uint i = 0; i < hd; i++) acc[i] += wgt * vv[i];
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
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float rstd = rsqrt(m2 / float(D) + 1e-5f);
  for (uint i = tid; i < D; i += 256) Y[tg * D + i] = (X[tg * D + i] - mean) * rstd * W[i] + B[i];
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

// unpatchify + denormalize: token output [voxels, 3*4*16*16] to RGB bytes
// [frames, H*16, W*16, 3]. One thread per output byte.
[[kernel]] void h3_vae_unpatch(
  const device float* PROJ [[buffer(0)]],
  const device float* MEAN [[buffer(1)]],
  const device float* STD [[buffer(2)]],
  device uchar* RGB [[buffer(3)]],
  constant uint* P [[buffer(4)]],
  uint3 at [[thread_position_in_grid]]) {
  const uint depth = P[0], h = P[1], w = P[2], out_w = P[3], hb = P[4];
  const uint pt = 4, p = 16;
  const uint f = at.z;
  const uint t = f / pt;
  const uint dt = f % pt;
  const uint row = at.y;
  const uint hh = row / p;
  const uint dh = row % p;
  const uint col3 = at.x;
  const uint c = col3 % 3;
  const uint col = col3 / 3;
  const uint ww = col / p;
  const uint dw = col % p;
  if (t >= depth || hh >= h || ww >= w) return;
  const uint v = (t * h + hh) * w + ww;
  const float o = PROJ[v * hb + c * pt * p * p + dt * p * p + dh * p + dw];
  const float value = o * STD[c] + MEAN[c];
  RGB[(f * h * p + row) * out_w + col3] = (uchar)clamp((value * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f);
}


// V extraction in the interleaved qkv layout: OUT[h*rows*hd + r*hd + i] =
// QKV[r * 3*dim + h*3*hd + 2*hd + i]. One thread per output element.
[[kernel]] void h3_vae_vextract(
  const device float* QKV [[buffer(0)]],
  device float* OUT [[buffer(1)]],
  constant uint* P [[buffer(2)]],          // P[0] = rows
  uint3 at [[thread_position_in_grid]]) {
  const uint rows = P[0];
  const uint hd = 64, heads = 32, dim = heads * hd;
  const uint r = at.y;
  const uint i = at.x % hd;
  const uint h = at.x / hd;
  OUT[(h * rows + r) * hd + i] = QKV[r * 3 * dim + h * 3 * hd + 2 * hd + i];
}
