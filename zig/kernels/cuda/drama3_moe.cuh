// drama3 MoE glue (router, SwiGLU, combine, FP8 activation rows) and the codebook tower's input mixing.
#pragma once
#include "drama3_common.cuh"

// Router: fp32 softmax over 256 logits of [shared gate | up | router | shared gate 1] rows, top 8, renormalized.
extern "C" __global__ void d3_router(const bf16* __restrict__ sp, int ss, int r_off, int* __restrict__ picks,
                                     float* __restrict__ weights) {
  const int t = blockIdx.x, e = threadIdx.x;  // 256 threads
  const float lg = b2f(sp[(size_t)t * ss + r_off + e]);
  const float m = block_max(lg);
  const float p = expf(lg - m);
  const float sum = block_sum(p);
  float prob = p / sum;
  __shared__ float best_v[8];
  __shared__ int best_i[8];
  __shared__ float wv[32];
  __shared__ int wi[32];
  float mine = prob;
  for (int k = 0; k < 8; ++k) {
    float v = mine;
    int idx = e;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(FULL, v, o);
      const int oi = __shfl_xor_sync(FULL, idx, o);
      if (ov > v || (ov == v && oi < idx)) {
        v = ov;
        idx = oi;
      }
    }
    if ((e & 31) == 0) {
      wv[e >> 5] = v;
      wi[e >> 5] = idx;
    }
    __syncthreads();
    if (e == 0) {
      float bv = wv[0];
      int bi = wi[0];
      for (int j = 1; j < 8; ++j)
        if (wv[j] > bv || (wv[j] == bv && wi[j] < bi)) {
          bv = wv[j];
          bi = wi[j];
        }
      best_v[k] = bv;
      best_i[k] = bi;
    }
    __syncthreads();
    if (e == best_i[k]) mine = -1.f;
    __syncthreads();
  }
  if (e < 8) {
    float s = 0.f;
    for (int j = 0; j < 8; ++j) s += best_v[j];
    picks[t * 8 + e] = best_i[e];
    weights[t * 8 + e] = best_v[e] / s;
  }
}

// y [T, N] = bf16(bf16(silu(g)) * u) with g, u at offsets of rows `stride` apart.
extern "C" __global__ void d3_swiglu(const bf16* __restrict__ x, int stride, int g_off, int u_off, int N,
                                     bf16* __restrict__ y) {
  const int t = blockIdx.y;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
    const float g = b2f(x[(size_t)t * stride + g_off + i]), u = b2f(x[(size_t)t * stride + u_off + i]);
    y[(size_t)t * N + i] = f2b(rb(siluf_(g)) * u);
  }
}

// r = bf16(r + bf16(bf16(sum_k w_k down_k) + bf16(bf16(sigmoid(sg)) * shared))); grid (T), 256 threads.
extern "C" __global__ void d3_moe_combine(bf16* __restrict__ r, const float* __restrict__ down,
                                          const float* __restrict__ w, const bf16* __restrict__ shared,
                                          const bf16* __restrict__ sp, int ss, int sg_off, int D) {
  const int t = blockIdx.x;
  const float sg = rb(sigmoidf_(b2f(sp[(size_t)t * ss + sg_off])));
  float wk[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) wk[k] = w[t * 8 + k];
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) acc = fmaf(wk[k], down[((size_t)t * 8 + k) * D + i], acc);
    const float sh = rb(sg * b2f(shared[(size_t)t * D + i]));
    const float moe = rb(rb(acc) + sh);
    r[(size_t)t * D + i] = f2b(b2f(r[(size_t)t * D + i]) + moe);
  }
}

// Per row and 128 inputs: scale = max(amax, 1e-10) / 448, q = e4m3(x / scale) stored as bf16, scales s [rows, K/128].
extern "C" __global__ void d3_act_quant(const bf16* __restrict__ x, int xs, bf16* __restrict__ q, int qs,
                                        float* __restrict__ s) {
  const int r = blockIdx.x, g = blockIdx.y, i = threadIdx.x;
  const float v = b2f(x[(size_t)r * xs + g * 128 + i]);
  const float amax = block_max(fabsf(v));
  const float sc = fmaxf(amax, 1e-10f) / 448.f;
  const float y = fminf(fmaxf(__fdiv_rn(v, sc), -448.f), 448.f);
  const __nv_fp8_e4m3 e(y);
  q[(size_t)r * qs + g * 128 + i] = f2b(float(e));
  if (i == 0) s[(size_t)r * gridDim.y + g] = sc;
}

// Tower input [enorm(e) | hnorm(h)]: e = table[label code], plus on layer 0 the input frame's code rows 2..9.
extern "C" __global__ void d3_tower_mix(const int* __restrict__ lf, int lfs, int code_col, const int* __restrict__ inf,
                                        int with_shared, const bf16* __restrict__ table,
                                        const bf16* __restrict__ cb_emb, const bf16* __restrict__ h,
                                        const bf16* __restrict__ enorm_w, const bf16* __restrict__ hnorm_w,
                                        bf16* __restrict__ out, int D, int sem_begin, int sem_end, int cb, float eps) {
  const int t = blockIdx.x;
  const int* l = lf + (size_t)t * lfs;
  const int* in = inf + (size_t)t * 11;
  const bool lsem = is_semantic(l[0], sem_begin, sem_end);
  const int code = min(max(l[code_col], 0), cb - 1);
  const bool shared = with_shared && is_semantic(in[0], sem_begin, sem_end);
  __shared__ float ev[2048];
  float se = 0.f, sh = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float e = lsem ? b2f(table[(size_t)code * D + i]) : 0.f;
    if (shared) {
      float s = 0.f;
      for (int c = 2; c < 10; ++c) s += b2f(cb_emb[(size_t)(c * cb + min(max(in[1 + c], 0), cb - 1)) * D + i]);
      e = rb(e + rb(s));
    }
    ev[i] = e;
    se += e * e;
    const float hv = b2f(h[(size_t)t * D + i]);
    sh += hv * hv;
  }
  const float ie = rsqrtf(block_sum(se) / D + eps);
  const float ih = rsqrtf(block_sum(sh) / D + eps);
  bf16* o = out + (size_t)t * 2 * D;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    o[i] = f2b(ev[i] * ie * (1.f + b2f(enorm_w[i])));
    o[D + i] = f2b(b2f(h[(size_t)t * D + i]) * ih * (1.f + b2f(hnorm_w[i])));
  }
}
