// music3 AR kernels: rows, the decode GEMV (bf16 / int8), the body and depth decoder attention.
#pragma once
#include "music3_common.cuh"

// ------------------------------------------------------------------------------------------------ AR: rows

// out[r] = table[ids[r]] (bf16 rows of dim).
extern "C" __global__ void m3_embed(const int* __restrict__ ids, const bf16* __restrict__ table, bf16* __restrict__ out, int dim) {
  const size_t r = blockIdx.x;
  const uint4* src = reinterpret_cast<const uint4*>(table + (size_t)ids[r] * dim);
  uint4* dst = reinterpret_cast<uint4*>(out + r * dim);
  for (int i = threadIdx.x; i < dim / 8; i += blockDim.x) dst[i] = src[i];
}

// out[b] = table[*code + offset] for rows b < rows (the same code for both CFG rows).
extern "C" __global__ void m3_embed_code(const int* __restrict__ code, int offset, const bf16* __restrict__ table, bf16* __restrict__ out, int rows, int dim) {
  const uint4* src = reinterpret_cast<const uint4*>(table + (size_t)(*code + offset) * dim);
  for (int r = 0; r < rows; ++r) {
    uint4* dst = reinterpret_cast<uint4*>(out + (size_t)r * dim);
    for (int i = threadIdx.x; i < dim / 8; i += blockDim.x) dst[i] = src[i];
  }
}

// A frame fed back: (audio[c0] + sum extra[c_i + (i - 1) 1024]) / sqrt(8), rounded as torch rounds it, both rows.
extern "C" __global__ void m3_frame_embed(const int* __restrict__ codes, const bf16* __restrict__ audio, const bf16* __restrict__ extra, bf16* __restrict__ out, int rows, int dim, float scale) {
  for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < dim; d += gridDim.x * blockDim.x) {
    float s = 0.f;
#pragma unroll
    for (int i = 1; i < 8; ++i) s += b2f(extra[(size_t)(codes[i] + (i - 1) * 1024) * dim + d]);
    const float c0 = b2f(audio[(size_t)codes[0] * dim + d]);
    const float v = rb(rb(c0 + rb(s)) * scale);
    for (int r = 0; r < rows; ++r) out[(size_t)r * dim + d] = f2b(v);
  }
}

// RMS norm of rows (x rows xs apart, y rows ys apart): y = x rsqrt(mean(x^2) + eps) w, fp32 inside, one rounding.
extern "C" __global__ void m3_rmsnorm(const bf16* __restrict__ x, int xs, const bf16* __restrict__ w, bf16* __restrict__ y, int ys, int dim, float eps) {
  __shared__ float sh[32];
  const bf16* xr = x + (size_t)blockIdx.x * xs;
  float ss = 0.f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = b2f(xr[i]);
    ss = fmaf(v, v, ss);
  }
  ss = block_sum(ss, sh);
  const float rs = 1.0f / sqrtf(ss / (float)dim + eps);
  bf16* yr = y + (size_t)blockIdx.x * ys;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) yr[i] = f2b(b2f(xr[i]) * rs * b2f(w[i]));
}

// x = round(x + a): prompt rows' residual adds after cuBLAS GEMMs.
extern "C" __global__ void m3_add(bf16* __restrict__ x, const bf16* __restrict__ a, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) x[i] = f2b(b2f(x[i]) + b2f(a[i]));
}

// y[r, k] = round(round(silu(g)) u) of gate | up rows [rows, 2 width] (prompt rows).
extern "C" __global__ void m3_swiglu(const bf16* __restrict__ gu, bf16* __restrict__ y, int width, size_t rows) {
  const size_t n = rows * width;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const size_t r = i / width, k = i % width;
    const float g = b2f(gu[r * 2 * width + k]), u = b2f(gu[r * 2 * width + width + k]);
    y[i] = f2b(rb(g / (1.f + expf(-g))) * u);
  }
}

// One bf16 weight row a block to int8 with a symmetric scale absmax / 127 a group of 128 inputs (fp32, [n, k / 128]).
extern "C" __global__ void m3_quant_groups(const bf16* __restrict__ w, int k, int8_t* __restrict__ q, float* __restrict__ sc) {
  const bf16* wr = w + (size_t)blockIdx.x * k;
  int8_t* qr = q + (size_t)blockIdx.x * k;
  const int groups = k / 128, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  for (int gi = warp; gi < groups; gi += nw) {
    float v[4], m = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[j] = b2f(wr[gi * 128 + lane + 32 * j]);
      m = fmaxf(m, fabsf(v[j]));
    }
    m = warp_max(m);
    const float s = m > 0.f ? m / 127.f : 1.f;
    const float inv = 1.f / s;
#pragma unroll
    for (int j = 0; j < 4; ++j) qr[gi * 128 + lane + 32 * j] = (int8_t)max(-127, min(127, __float2int_rn(v[j] * inv)));
    if (lane == 0) sc[(size_t)blockIdx.x * groups + gi] = s;
  }
}

// int8 rows back to bf16 (round(q s)) for the prompt rows' cuBLAS GEMMs.
extern "C" __global__ void m3_dequant(const int8_t* __restrict__ q, const float* __restrict__ sc, bf16* __restrict__ w, int k, size_t n) {
  const size_t total = n * k;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) w[i] = f2b((float)q[i] * sc[i / k]);
}

// One bf16 weight row a block to int8 with a symmetric scale absmax / 127 (fp32).
extern "C" __global__ void m3_quant_rows(const bf16* __restrict__ w, int k, int8_t* __restrict__ q, float* __restrict__ sc) {
  __shared__ float sh[32];
  const bf16* wr = w + (size_t)blockIdx.x * k;
  float m = 0.f;
  for (int i = threadIdx.x; i < k; i += blockDim.x) m = fmaxf(m, fabsf(b2f(wr[i])));
  m = block_max(m, sh);
  const float s = m > 0.f ? m / 127.f : 1.f;
  const float inv = 1.f / s;
  int8_t* qr = q + (size_t)blockIdx.x * k;
  for (int i = threadIdx.x; i < k; i += blockDim.x) qr[i] = (int8_t)max(-127, min(127, __float2int_rn(b2f(wr[i]) * inv)));
  if (threadIdx.x == 0) sc[blockIdx.x] = s;
}

// ------------------------------------------------------------------------------------------------ AR: the GEMV

template <bool I8>
__device__ __forceinline__ void wfloats(const uint4 w4, float* wf) {
  const unsigned ws[4] = {w4.x, w4.y, w4.z, w4.w};
  if (I8) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned u = ws[j] ^ 0x80808080u;
      wf[4 * j + 0] = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7650)) - 8388736.f;
      wf[4 * j + 1] = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7651)) - 8388736.f;
      wf[4 * j + 2] = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7652)) - 8388736.f;
      wf[4 * j + 3] = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7653)) - 8388736.f;
    }
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      wf[2 * j] = bfu(ws[j] & 0xFFFF);
      wf[2 * j + 1] = bfu(ws[j] >> 16);
    }
  }
}

// A weight load that streams past L1 (each byte is read once a frame).
__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// y = W x for M rows (bf16 / int8 W), residual or fp32 logits out, PAIR: SwiGLU of rows n, half + n; PDL prefetch.
template <int M, bool I8, bool PAIR, bool GRP = false>
__device__ __forceinline__ void gemv(const void* __restrict__ wv, const float* __restrict__ sc, const bf16* __restrict__ x, int xs, int K, int N, bf16* y, int ys, float* __restrict__ yf, const bf16* res, int res_div) {
  const int n = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  constexpr int E = I8 ? 16 : 8;  // weights a 16-byte load
  constexpr int R = PAIR ? 2 : 1;
  constexpr int U = PAIR ? 4 : 8;
  const int rows = PAIR ? N / 2 : N;
  const int chunks = K / E;
  const int nn = min(n, rows - 1);
  const char* wb = reinterpret_cast<const char*>(wv);
  const uint4* wr[R];
#pragma unroll
  for (int r = 0; r < R; ++r) wr[r] = reinterpret_cast<const uint4*>(wb + (size_t)(nn + r * rows) * K * (I8 ? 1 : 2));
  uint4 w[R][U];
  int c = lane;
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int r = 0; r < R; ++r) w[r][u] = c + u * 32 < chunks ? ld_stream(wr[r] + c + u * 32) : make_uint4(0, 0, 0, 0);
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
  while (true) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int cc = c + u * 32;
      if (cc >= chunks) break;
      float wf[R][E];
      float gs[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        wfloats<I8>(w[r][u], wf[r]);
        if (GRP) gs[r] = __ldg(sc + (size_t)(nn + r * rows) * (K / 128) + cc / 8);
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const uint4* xp = reinterpret_cast<const uint4*>(x + (size_t)m * xs + cc * E);
        float xf[E];
#pragma unroll
        for (int hh = 0; hh < E / 8; ++hh) {
          const uint4 v = __ldg(xp + hh);
          const unsigned vs[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            xf[8 * hh + 2 * j] = bfu(vs[j] & 0xFFFF);
            xf[8 * hh + 2 * j + 1] = bfu(vs[j] >> 16);
          }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
          if (GRP) {  // int8 groups of 128 inputs, a scale each: the chunk's partial sum, then times its scale
            float part = 0.f;
#pragma unroll
            for (int e = 0; e < E; ++e) part = fmaf(wf[r][e], xf[e], part);
            acc[r][m] = fmaf(gs[r], part, acc[r][m]);
          } else {
#pragma unroll
            for (int e = 0; e < E; ++e) acc[r][m] = fmaf(wf[r][e], xf[e], acc[r][m]);
          }
        }
      }
    }
    c += U * 32;
    if (c >= chunks) break;
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int r = 0; r < R; ++r) w[r][u] = c + u * 32 < chunks ? ld_stream(wr[r] + c + u * 32) : make_uint4(0, 0, 0, 0);
  }
  if (n >= rows) return;
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float v[R];
#pragma unroll
    for (int r = 0; r < R; ++r) v[r] = warp_sum(acc[r][m]) * (I8 && !GRP ? sc[n + r * rows] : 1.f);
    if (lane == 0) {
      if (PAIR) {
        const float g = rb(v[0]), u = rb(v[1]);
        y[(size_t)m * ys + n] = f2b(rb(g / (1.f + __expf(-g))) * u);
      } else {
        if (yf) yf[(size_t)m * N + n] = rb(v[0]);
        if (y) {
          float rr = rb(v[0]);
          if (res) rr = rb(b2f(res[(size_t)(m / res_div) * N + n]) + rr);
          y[(size_t)m * ys + n] = f2b(rr);
        }
      }
    }
  }
}

#define GEMV(NAME, M, I8, PAIR, GRP)                                                                                  \
  extern "C" __global__ void __launch_bounds__(256) NAME(const void* __restrict__ w, const float* __restrict__ sc, const bf16* __restrict__ x, int xs, int K, int N, bf16* y, int ys, float* __restrict__ yf, const bf16* res, int res_div) { \
    gemv<M, I8, PAIR, GRP>(w, sc, x, xs, K, N, y, ys, yf, res, res_div);                                                     \
  }
GEMV(m3_gemv_bf16_2, 2, false, false, false)
GEMV(m3_gemv_bf16_4, 4, false, false, false)
GEMV(m3_gemv_i8_2, 2, true, false, false)
GEMV(m3_gemv_i8_4, 4, true, false, false)
GEMV(m3_glu_bf16_2, 2, false, true, false)
GEMV(m3_glu_bf16_4, 4, false, true, false)
GEMV(m3_glu_i8_2, 2, true, true, false)
GEMV(m3_glu_i8_4, 4, true, true, false)
GEMV(m3_gemv_g8_2, 2, true, false, true)
GEMV(m3_gemv_g8_4, 4, true, false, true)
GEMV(m3_glu_g8_2, 2, true, true, true)
GEMV(m3_glu_g8_4, 4, true, true, true)

// ------------------------------------------------------------------------------------------------ AR: body attention

// Body q / k: a head's RMS norm, then rotate-half rope; q to q_out, k and v to the caches. grid (rows, 48).
extern "C" __global__ void m3_qk_rope(const bf16* __restrict__ qkv, const bf16* __restrict__ qn, const bf16* __restrict__ kn, const float* __restrict__ inv_freq, const int* __restrict__ pos, int t, bf16* __restrict__ q_out, bf16* __restrict__ kc, bf16* __restrict__ vc, int ctx, float eps) {
  __shared__ float sh[32];
  __shared__ float v_sh[128];
  const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int b = r / t, p = *pos + r % t;
  const bf16* row = qkv + (size_t)r * 6144;
  if (h >= 40) {  // v
    const int g = h - 40;
    vc[(((size_t)b * 8 + g) * ctx + p) * 128 + d] = row[4096 + 1024 + g * 128 + d];
    return;
  }
  const bool isq = h < 32;
  const bf16* src = isq ? row + h * 128 : row + 4096 + (h - 32) * 128;
  const float x = b2f(src[d]);
  const float ss = block_sum(x * x, sh);
  const float rs = 1.0f / sqrtf(ss / 128.f + eps);
  const float nx = rb(x * rs * b2f((isq ? qn : kn)[d]));
  v_sh[d] = nx;
  __syncthreads();
  const int i = d & 63;
  const float ang = (float)p * inv_freq[i];
  const float c = cosf(ang), s = sinf(ang);
  const float other = d < 64 ? -v_sh[d + 64] : v_sh[d - 64];
  const bf16 o = f2b(nx * c + other * s);
  if (isq)
    q_out[((size_t)r * 32 + h) * 128 + d] = o;
  else
    kc[(((size_t)b * 8 + (h - 32)) * ctx + p) * 128 + d] = o;
}

// Split-key attention: a block a (kv head, row, split), warps the group's 4 q heads; partials ml / acc a split.
extern "C" __global__ void __launch_bounds__(128) m3_attn_part(const bf16* __restrict__ q, const bf16* __restrict__ kc, const bf16* __restrict__ vc, const int* __restrict__ pos, int t, int ctx, int S, float scale, float* __restrict__ ml, float* __restrict__ acc) {
  const int g = blockIdx.x, r = blockIdx.y, sp = blockIdx.z;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int h = g * 4 + w;
  const int b = r / t;
  const int L = *pos + r % t + 1;
  const int chunk = (L + S - 1) / S;
  const int k0 = sp * chunk, k1 = min(L, k0 + chunk);
  const uint2 qv = *reinterpret_cast<const uint2*>(q + ((size_t)r * 32 + h) * 128 + lane * 4);
  const float q0 = bfu(qv.x & 0xFFFF) * scale, q1 = bfu(qv.x >> 16) * scale, q2 = bfu(qv.y & 0xFFFF) * scale, q3 = bfu(qv.y >> 16) * scale;
  const bf16* kb = kc + ((size_t)b * 8 + g) * ctx * 128 + lane * 4;
  const bf16* vb = vc + ((size_t)b * 8 + g) * ctx * 128 + lane * 4;
  float m = -INFINITY, l = 0.f, a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  for (int j = k0; j < k1; ++j) {
    const uint2 kv = __ldg(reinterpret_cast<const uint2*>(kb + (size_t)j * 128));
    float s = q0 * bfu(kv.x & 0xFFFF) + q1 * bfu(kv.x >> 16) + q2 * bfu(kv.y & 0xFFFF) + q3 * bfu(kv.y >> 16);
    s = warp_sum(s);
    const uint2 vv = __ldg(reinterpret_cast<const uint2*>(vb + (size_t)j * 128));
    const float mn = fmaxf(m, s);
    const float cor = __expf(m - mn), e = __expf(s - mn);
    l = l * cor + e;
    a0 = a0 * cor + e * bfu(vv.x & 0xFFFF);
    a1 = a1 * cor + e * bfu(vv.x >> 16);
    a2 = a2 * cor + e * bfu(vv.y & 0xFFFF);
    a3 = a3 * cor + e * bfu(vv.y >> 16);
    m = mn;
  }
  const size_t o = ((size_t)r * 32 + h) * S + sp;
  if (lane == 0) {
    ml[o * 2] = m;
    ml[o * 2 + 1] = l;
  }
  float4* ap = reinterpret_cast<float4*>(acc + o * 128 + lane * 4);
  *ap = make_float4(a0, a1, a2, a3);
}

// The S partials of a (row, head) into out [rows, 32 * 128] bf16. grid (rows, 32), block 128.
extern "C" __global__ void m3_attn_merge(const float* __restrict__ ml, const float* __restrict__ acc, int S, bf16* __restrict__ out) {
  const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const size_t base = ((size_t)r * 32 + h) * S;
  float m = -INFINITY;
  for (int s = 0; s < S; ++s) m = fmaxf(m, ml[(base + s) * 2]);
  float l = 0.f, a = 0.f;
  for (int s = 0; s < S; ++s) {
    const float ms = ml[(base + s) * 2];
    if (ms == -INFINITY) continue;
    const float c = __expf(ms - m);
    l += ml[(base + s) * 2 + 1] * c;
    a += acc[(base + s) * 128 + d] * c;
  }
  out[((size_t)r * 32 + h) * 128 + d] = f2b(a / l);
}

// ------------------------------------------------------------------------------------------------ AR: depth decoder

// Depth rows' k and v into the caches [b][16][8][256]: pass row r is (position p0 + r / B, batch r % B).
extern "C" __global__ void m3_depth_kv(const bf16* __restrict__ qkv, int p0, int B, bf16* __restrict__ kc, bf16* __restrict__ vc) {
  const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int p = p0 + r / B, b = r % B;
  const bf16* row = qkv + (size_t)r * 12288;
  const size_t o = (((size_t)b * 16 + h) * 8 + p) * 256 + d;
  kc[o] = row[4096 + h * 256 + d];
  vc[o] = row[8192 + h * 256 + d];
}

// Causal attention of the depth rows over positions <= p (at most 8), out [rows, 4096]. grid (rows, 16), block 256.
extern "C" __global__ void m3_depth_attn(const bf16* __restrict__ qkv, int p0, int B, const bf16* __restrict__ kc, const bf16* __restrict__ vc, float scale, bf16* __restrict__ out) {
  __shared__ float sh[32];
  __shared__ float pr[8];
  const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int p = p0 + r / B, b = r % B;
  const float qd = b2f(qkv[(size_t)r * 12288 + h * 256 + d]);
  const bf16* kb = kc + ((size_t)b * 16 + h) * 8 * 256;
  const bf16* vb = vc + ((size_t)b * 16 + h) * 8 * 256;
  for (int j = 0; j <= p; ++j) {
    const float s = block_sum(qd * b2f(kb[j * 256 + d]), sh);
    if (d == 0) pr[j] = s * scale;
  }
  __syncthreads();
  float m = -INFINITY;
  for (int j = 0; j <= p; ++j) m = fmaxf(m, pr[j]);
  float l = 0.f, a = 0.f;
  for (int j = 0; j <= p; ++j) {
    const float e = __expf(pr[j] - m);
    l += e;
    a = fmaf(e, b2f(vb[j * 256 + d]), a);
  }
  out[(size_t)r * 4096 + h * 256 + d] = f2b(a / l);
}
