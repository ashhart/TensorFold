// drama3 row kernels: the bf16 GEMV, zero-centred RMS norms, backbone frame embeddings and Gated DeltaNet pieces.
#pragma once
#include "drama3_common.cuh"

// y[n] = sum_k W[n, k] x[k]: W bf16 [N, K] (K % 256 == 0), a warp a row, x staged in K * 2 bytes of dynamic shared.
extern "C" __global__ void __launch_bounds__(256) d3_gemv(const bf16* __restrict__ W, const bf16* __restrict__ x,
                                                          int K, int N, bf16* __restrict__ y, float* __restrict__ yf) {
  extern __shared__ uint4 xs[];
  const int chunks = K >> 3;
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.x * 8 + (threadIdx.x >> 5);
  const uint4* row = reinterpret_cast<const uint4*>(W + (size_t)min(n, N - 1) * K);
  // the row's first 8 chunks a lane are in flight before x is staged (they do not depend on it)
  uint4 w[8];
  int c = lane;
#pragma unroll
  for (int u = 0; u < 8; ++u) w[u] = c + u * 32 < chunks ? ld_stream(row + c + u * 32) : make_uint4(0, 0, 0, 0);
  // launched with programmatic dependent launch, the weights stream while the kernel before finishes; x is its output
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
  for (int i = threadIdx.x; i < chunks; i += blockDim.x) xs[i] = reinterpret_cast<const uint4*>(x)[i];
  __syncthreads();
  float acc = 0.f;
  while (true) {
#pragma unroll
    for (int u = 0; u < 8; ++u)
      if (c + u * 32 < chunks) acc += dot8(w[u], xs[c + u * 32]);
    c += 8 * 32;
    if (c >= chunks) break;
#pragma unroll
    for (int u = 0; u < 8; ++u) w[u] = c + u * 32 < chunks ? ld_stream(row + c + u * 32) : make_uint4(0, 0, 0, 0);
  }
  acc = warp_sum(acc);
  if (lane == 0 && n < N) {
    if (y) y[n] = f2b(acc);
    if (yf) yf[n] = rb(acc);  // logits: rounded to bf16 as the bf16 heads return them
  }
}

// y = bf16(x * rsqrt(mean(x^2) + eps) * g), g = 1 + w when zero_centred else w, fp32 inside; a block a row.
extern "C" __global__ void d3_rmsnorm(const bf16* __restrict__ x, int x_stride, const bf16* __restrict__ w,
                                      bf16* __restrict__ y, int y_stride, int D, float eps, int zero_centred) {
  const bf16* xr = x + (size_t)blockIdx.x * x_stride;
  bf16* yr = y + (size_t)blockIdx.x * y_stride;
  float ss = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = b2f(xr[i]);
    ss += v * v;
  }
  const float inv = rsqrtf(block_sum(ss) / D + eps);
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float g = zero_centred ? 1.f + b2f(w[i]) : b2f(w[i]);
    yr[i] = f2b(b2f(xr[i]) * inv * g);
  }
}

// r = bf16(r + a) (a's rows a_stride apart), then y = zero-centred rmsnorm(r); r is [rows, D] contiguous.
extern "C" __global__ void d3_add_rmsnorm(bf16* __restrict__ r, const bf16* __restrict__ a, int a_stride,
                                          const bf16* __restrict__ w, bf16* __restrict__ y, int D, float eps) {
  bf16* rr = r + (size_t)blockIdx.x * D;
  const bf16* ar = a + (size_t)blockIdx.x * a_stride;
  bf16* yr = y + (size_t)blockIdx.x * D;
  float ss = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = rb(b2f(rr[i]) + b2f(ar[i]));
    rr[i] = f2b(v);
    ss += v * v;
  }
  const float inv = rsqrtf(block_sum(ss) / D + eps);
  for (int i = threadIdx.x; i < D; i += blockDim.x) yr[i] = f2b(b2f(rr[i]) * inv * (1.f + b2f(w[i])));
}

// Frame rows [rows, 11] to embeddings: text rows embed[main], semantic rows bf16(bf16(embed + cb1[code1]) / sqrt 2).
extern "C" __global__ void d3_embed_backbone(const int* __restrict__ frames, const bf16* __restrict__ embed,
                                             const bf16* __restrict__ cb_emb, bf16* __restrict__ out, int D,
                                             int sem_begin, int sem_end, int cb_size) {
  const int* f = frames + (size_t)blockIdx.x * 11;
  const int tok = f[0];
  const bool sem = is_semantic(tok, sem_begin, sem_end);
  int code1 = min(max(f[2], 0), cb_size - 1);
  const bf16* e = embed + (size_t)tok * D;
  const bf16* c = cb_emb + (size_t)(cb_size + code1) * D;
  bf16* o = out + (size_t)blockIdx.x * D;
  const float scale = 0.70710678118654752f;  // 1 / sqrt(1 + 1)
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = b2f(e[i]);
    o[i] = sem ? f2b(rb(v + b2f(c[i])) * scale) : e[i];
  }
}

// GDN depthwise causal conv (kernel 4) + SiLU over the qkv channels of [qkv | z | b | a] rows; state keeps 3 inputs.
extern "C" __global__ void d3_conv_silu(const bf16* __restrict__ proj, int ps, int T, const bf16* __restrict__ w,
                                        bf16* __restrict__ state, bf16* __restrict__ out, int C) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const float w0 = b2f(w[c * 4 + 0]), w1 = b2f(w[c * 4 + 1]), w2 = b2f(w[c * 4 + 2]), w3 = b2f(w[c * 4 + 3]);
  float s0 = b2f(state[c * 3 + 0]), s1 = b2f(state[c * 3 + 1]), s2 = b2f(state[c * 3 + 2]);
  for (int t = 0; t < T; ++t) {
    const float x = b2f(proj[(size_t)t * ps + c]);
    float acc = s0 * w0;
    acc = fmaf(s1, w1, acc);
    acc = fmaf(s2, w2, acc);
    acc = fmaf(x, w3, acc);
    out[(size_t)t * C + c] = f2b(siluf_(acc));
    s0 = s1;
    s1 = s2;
    s2 = x;
  }
  state[c * 3 + 0] = f2b(s0);
  state[c * 3 + 1] = f2b(s1);
  state[c * 3 + 2] = f2b(s2);
}

// GDN recurrence: a warp per (value head, 32 value columns), fp32 state [128][128], l2-normed q / k, gated delta rule.
extern "C" __global__ void __launch_bounds__(32) d3_gdn(const bf16* __restrict__ conv, const bf16* __restrict__ proj,
                                                        int ps, int T, const bf16* __restrict__ A_log,
                                                        const bf16* __restrict__ dt_bias, float* __restrict__ S,
                                                        float* __restrict__ o) {
  constexpr int DK = 128, DV = 128, NV = 32;
  const int h = blockIdx.x, slice = blockIdx.y, lane = threadIdx.x;
  const int kh = h >> 1;
  const int v = slice * 32 + lane;
  __shared__ float qs[DK], ks[DK];
  float s[DK];
  float* Sh = S + (size_t)h * DK * DV;
#pragma unroll
  for (int k = 0; k < DK; ++k) s[k] = Sh[k * DV + v];
  const float A = -expf(b2f(A_log[h]));
  const float dtb = b2f(dt_bias[h]);
  const float qscale = rsqrtf((float)DK);
  for (int t = 0; t < T; ++t) {
    const bf16* row = conv + (size_t)t * 8192;
    float qv[4], kv[4];
    float qq = 0.f, kk = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      qv[j] = b2f(row[kh * DK + lane + 32 * j]);
      kv[j] = b2f(row[2048 + kh * DK + lane + 32 * j]);
      qq += qv[j] * qv[j];
      kk += kv[j] * kv[j];
    }
    qq = warp_sum(qq);
    kk = warp_sum(kk);
    const float qi = rsqrtf(qq + 1e-6f) * qscale, ki = rsqrtf(kk + 1e-6f);
    __syncwarp();
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      qs[lane + 32 * j] = qv[j] * qi;
      ks[lane + 32 * j] = kv[j] * ki;
    }
    __syncwarp();
    const bf16* pr = proj + (size_t)t * ps;
    const float bb = sigmoidf_(b2f(pr[12288 + h]));
    const float g = A * softplusf_(b2f(pr[12288 + NV + h]) + dtb);
    const float decay = expf(g);
    const float vv = b2f(row[4096 + h * DV + v]);
    float mem = 0.f;
#pragma unroll
    for (int k = 0; k < DK; ++k) {
      s[k] *= decay;
      mem = fmaf(s[k], ks[k], mem);
    }
    const float delta = (vv - mem) * bb;
    float out = 0.f;
#pragma unroll
    for (int k = 0; k < DK; ++k) {
      s[k] = fmaf(ks[k], delta, s[k]);
      out = fmaf(s[k], qs[k], out);
    }
    o[(size_t)t * (NV * DV) + h * DV + v] = out;
  }
#pragma unroll
  for (int k = 0; k < DK; ++k) Sh[k * DV + v] = s[k];
}

// Gated RMSNorm per 128-wide value head: y = bf16(o * rsqrt(mean(o^2) + eps) * w * silu(z)); grid (T, 4), 256 threads.
extern "C" __global__ void d3_gated_norm(const float* __restrict__ o, const bf16* __restrict__ proj, int ps,
                                         const bf16* __restrict__ w, bf16* __restrict__ y, float eps) {
  const int t = blockIdx.x, h = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  const float* orow = o + (size_t)t * 4096 + h * 128;
  const bf16* z = proj + (size_t)t * ps + 8192 + h * 128;
  float v[4], ss = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] = orow[lane + 32 * j];
    ss += v[j] * v[j];
  }
  const float inv = rsqrtf(warp_sum(ss) / 128.f + eps);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = lane + 32 * j;
    const float n = v[j] * inv * b2f(w[i]);
    y[(size_t)t * 4096 + h * 128 + i] = f2b(n * siluf_(b2f(z[i])));
  }
}
