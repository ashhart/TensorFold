// drama3 gated full attention (backbone and tower): 16 query heads, 2 KV heads, head dim 256, rotary on 64 dims.
#pragma once
#include "drama3_common.cuh"

// q / k zero-centred RMSNorm then rotary (NeoX halves or interleaved pairs), q to qout, k / v into cache row pos.
extern "C" __global__ void d3_qk_rope(const bf16* __restrict__ proj, int ps, int q_off, int q_head_stride, int k_off,
                                      int v_off, const bf16* __restrict__ qn_w, const bf16* __restrict__ kn_w,
                                      const int* __restrict__ pos0, bf16* __restrict__ qout, bf16* __restrict__ kc,
                                      bf16* __restrict__ vc, int interleaved, float theta, float eps) {
  const int t = blockIdx.x, hh = blockIdx.y, i = threadIdx.x;
  const int pos = pos0[0] + t;
  const bf16* row = proj + (size_t)t * ps;
  const bool isq = hh < 16;
  const bf16* src = isq ? row + q_off + hh * q_head_stride : row + k_off + (hh - 16) * 256;
  const float x = b2f(src[i]);
  const float inv = rsqrtf(block_sum(x * x) / 256.f + eps);
  const float wv = b2f((isq ? qn_w : kn_w)[i]);
  __shared__ float xn[256];
  xn[i] = rb(x * inv * (1.f + wv));
  __syncthreads();
  float outv = xn[i];
  if (i < 64) {
    int j, partner;
    float sign;
    if (interleaved) {
      j = i >> 1;
      partner = i ^ 1;
      sign = (i & 1) ? 1.f : -1.f;
    } else {
      j = i & 31;
      partner = i < 32 ? i + 32 : i - 32;
      sign = i < 32 ? -1.f : 1.f;
    }
    const float inv_freq = 1.f / powf(theta, (float)(2 * j) / 64.f);
    const float ang = (float)pos * inv_freq;
    float c = cosf(ang), s = sinf(ang);
    outv = xn[i] * c + sign * xn[partner] * s;
  }
  if (isq) {
    qout[((size_t)t * 16 + hh) * 256 + i] = f2b(outv);
  } else {
    const int g = hh - 16;
    kc[((size_t)pos * 2 + g) * 256 + i] = f2b(outv);
    vc[((size_t)pos * 2 + g) * 256 + i] = row[v_off + g * 256 + i];
  }
}

// Decode attention of row t over cache rows 0..pos0 + t: 8 warps online softmax, merged; out = bf16(a) * sigmoid(g).
extern "C" __global__ void __launch_bounds__(256) d3_attn_decode(const bf16* __restrict__ q, const bf16* __restrict__ kc,
                                                                 const bf16* __restrict__ vc, const int* __restrict__ pos0,
                                                                 const bf16* __restrict__ proj, int ps, int gate_off,
                                                                 int gate_head_stride, bf16* __restrict__ out) {
  const int t = blockIdx.x, h = blockIdx.y, g = h >> 3;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int L = pos0[0] + t + 1;
  const uint4 qv = reinterpret_cast<const uint4*>(q + ((size_t)t * 16 + h) * 256)[lane];
  float m = -INFINITY, l = 0.f, acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = 0.f;
  const float scale = 0.0625f;  // 1/sqrt(256)
  for (int p = w; p < L; p += 8) {
    const uint4 kv = reinterpret_cast<const uint4*>(kc + ((size_t)p * 2 + g) * 256)[lane];
    const float sc = warp_sum(dot8(qv, kv)) * scale;
    const uint4 vv = reinterpret_cast<const uint4*>(vc + ((size_t)p * 2 + g) * 256)[lane];
    const float nm = fmaxf(m, sc);
    const float corr = expf(m - nm), e = expf(sc - nm);
    l = l * corr + e;
    acc[0] = acc[0] * corr + e * lo(vv.x);
    acc[1] = acc[1] * corr + e * hi(vv.x);
    acc[2] = acc[2] * corr + e * lo(vv.y);
    acc[3] = acc[3] * corr + e * hi(vv.y);
    acc[4] = acc[4] * corr + e * lo(vv.z);
    acc[5] = acc[5] * corr + e * hi(vv.z);
    acc[6] = acc[6] * corr + e * lo(vv.w);
    acc[7] = acc[7] * corr + e * hi(vv.w);
    m = nm;
  }
  __shared__ float sm[8], sl[8], sacc[8][256];
  if (lane == 0) {
    sm[w] = m;
    sl[w] = l;
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) sacc[w][lane * 8 + j] = acc[j];
  __syncthreads();
  const int i = threadIdx.x;
  float M = -INFINITY;
#pragma unroll
  for (int k = 0; k < 8; ++k) M = fmaxf(M, sm[k]);
  float Lsum = 0.f, A = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    if (sm[k] == -INFINITY) continue;
    const float c = expf(sm[k] - M);
    Lsum += sl[k] * c;
    A += sacc[k][i] * c;
  }
  const float a = rb(A / Lsum);
  const float gt = b2f(proj[(size_t)t * ps + gate_off + h * gate_head_stride + i]);
  out[((size_t)t * 16 + h) * 256 + i] = f2b(a * rb(sigmoidf_(gt)));
}

// bf16 to fp32 copy (the V cache rows for the prompt PV GEMM).
extern "C" __global__ void d3_bf16_f32(const bf16* __restrict__ x, float* __restrict__ y, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] = b2f(x[i]);
}

// Split-key decode attention: block (row, head, split) writes max, sum and unnormalized output for its key range.
extern "C" __global__ void __launch_bounds__(256) d3_attn_partial(const bf16* __restrict__ q, const bf16* __restrict__ kc,
                                                                  const bf16* __restrict__ vc, const int* __restrict__ pos0,
                                                                  float* __restrict__ pml, float* __restrict__ pacc) {
  const int t = blockIdx.x, h = blockIdx.y, sp = blockIdx.z, S = gridDim.z, g = h >> 3;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int L = pos0[0] + t + 1;
  const int chunk = (L + S - 1) / S;
  const int k0 = sp * chunk, k1 = min(L, k0 + chunk);
  const uint4 qv = reinterpret_cast<const uint4*>(q + ((size_t)t * 16 + h) * 256)[lane];
  float m = -INFINITY, l = 0.f, acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = 0.f;
  for (int p = k0 + w; p < k1; p += 8) {
    const uint4 kv = reinterpret_cast<const uint4*>(kc + ((size_t)p * 2 + g) * 256)[lane];
    const uint4 vv = reinterpret_cast<const uint4*>(vc + ((size_t)p * 2 + g) * 256)[lane];
    const float sc = warp_sum(dot8(qv, kv)) * 0.0625f;
    const float nm = fmaxf(m, sc);
    const float corr = expf(m - nm), e = expf(sc - nm);
    l = l * corr + e;
    acc[0] = acc[0] * corr + e * lo(vv.x);
    acc[1] = acc[1] * corr + e * hi(vv.x);
    acc[2] = acc[2] * corr + e * lo(vv.y);
    acc[3] = acc[3] * corr + e * hi(vv.y);
    acc[4] = acc[4] * corr + e * lo(vv.z);
    acc[5] = acc[5] * corr + e * hi(vv.z);
    acc[6] = acc[6] * corr + e * lo(vv.w);
    acc[7] = acc[7] * corr + e * hi(vv.w);
    m = nm;
  }
  __shared__ float sm[8], sl[8], sacc[8][256];
  if (lane == 0) {
    sm[w] = m;
    sl[w] = l;
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) sacc[w][lane * 8 + j] = acc[j];
  __syncthreads();
  const int i = threadIdx.x;
  float M = -INFINITY;
#pragma unroll
  for (int k = 0; k < 8; ++k) M = fmaxf(M, sm[k]);
  float Ls = 0.f, A = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    if (sm[k] == -INFINITY) continue;
    const float c = expf(sm[k] - M);
    Ls += sl[k] * c;
    A += sacc[k][i] * c;
  }
  const size_t slot = ((size_t)t * 16 + h) * S + sp;
  pacc[slot * 256 + i] = A;
  if (i == 0) {
    pml[slot * 2] = M;
    pml[slot * 2 + 1] = Ls;
  }
}

// Merges d3_attn_partial's splits per (row, head) and applies the sigmoid gate: out = bf16(a) * bf16(sigmoid(g)).
extern "C" __global__ void d3_attn_merge(const float* __restrict__ pml, const float* __restrict__ pacc, int S,
                                         const bf16* __restrict__ proj, int ps, int gate_off, int gate_head_stride,
                                         bf16* __restrict__ out) {
  const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
  const size_t base = ((size_t)t * 16 + h) * S;
  float M = -INFINITY;
  for (int k = 0; k < S; ++k) M = fmaxf(M, pml[(base + k) * 2]);
  float Ls = 0.f, A = 0.f;
  for (int k = 0; k < S; ++k) {
    const float mk = pml[(base + k) * 2];
    if (mk == -INFINITY) continue;
    const float c = expf(mk - M);
    Ls += pml[(base + k) * 2 + 1] * c;
    A += pacc[(base + k) * 256 + i] * c;
  }
  const float a = rb(A / Ls);
  const float gt = b2f(proj[(size_t)t * ps + gate_off + h * gate_head_stride + i]);
  out[((size_t)t * 16 + h) * 256 + i] = f2b(a * rb(sigmoidf_(gt)));
}

// Prompt attention through cuBLAS: qg [2][T][8][256] (grouped per KV head) from q [T, 16, 256].
extern "C" __global__ void d3_q_group(const bf16* __restrict__ q, bf16* __restrict__ qg, int T) {
  const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
  qg[(((size_t)(h >> 3) * T + t) * 8 + (h & 7)) * 256 + i] = q[((size_t)t * 16 + h) * 256 + i];
}

// Causal softmax in place over fp32 scores [2][T * 8][ld]: row (g, t, h) keeps keys 0..pos0 + t, zeroes the rest.
extern "C" __global__ void d3_causal_softmax(float* __restrict__ s, int T, int ld, const int* __restrict__ pos0) {
  const size_t r = blockIdx.x;
  const int t = (int)((r / 8) % T);
  const int L = pos0[0] + t + 1;
  float* sr = s + r * ld;
  float m = -INFINITY;
  for (int j = threadIdx.x; j < L; j += blockDim.x) m = fmaxf(m, sr[j]);
  m = block_max(m);
  float sum = 0.f;
  for (int j = threadIdx.x; j < L; j += blockDim.x) sum += expf(sr[j] - m);
  const float inv = 1.f / block_sum(sum);
  __syncthreads();
  for (int j = threadIdx.x; j < ld; j += blockDim.x) sr[j] = j < L ? expf(sr[j] - m) * inv : 0.f;
}

// The PV GEMM's fp32 O [2][T][8][256] to out [T, 16, 256] = bf16(O) * bf16(sigmoid(gate)).
extern "C" __global__ void d3_attn_gate(const float* __restrict__ og, const bf16* __restrict__ proj, int ps,
                                        int gate_off, int gate_head_stride, bf16* __restrict__ out, int T) {
  const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
  const float a = rb(og[(((size_t)(h >> 3) * T + t) * 8 + (h & 7)) * 256 + i]);
  const float gt = b2f(proj[(size_t)t * ps + gate_off + h * gate_head_stride + i]);
  out[((size_t)t * 16 + h) * 256 + i] = f2b(a * rb(sigmoidf_(gt)));
}
