// MLA attention bodies shared by GLM (glm_glue) and the mla image: chunk tiles over the latent (+ roped columns).
#pragma once

#include <cuda_fp8.h>
#include "glm_router_mma.cuh"

namespace glm_attn {

constexpr int LW = 512, KT = 32, CH = 512, KPAD = LW + 8;

// p fp32 summed as Triton's tile does: a thread's two columns, the quad (xor 2, 1), then warps (w0 + w2) + (w1 + w3).
__device__ __forceinline__ float quad_sum(float v) {
  v = v + __shfl_xor_sync(0xffffffffu, v, 2);
  return v + __shfl_xor_sync(0xffffffffu, v, 1);
}
__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
}

__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
  tf_common::cp16z(dst, src, ok);
}
__device__ __forceinline__ void commit() { tf_common::cp_commit(); }
template <int N>
__device__ __forceinline__ void wait() { tf_common::cp_wait<N>(); }
template <bool TRANS>
__device__ __forceinline__ void ldm4(uint32_t* r, const void* p) {
  const unsigned a = tf_common::smem(p);
  if (TRANS) qmm_frag::ldmatrix4t(r, a);
  else qmm_frag::ldmatrix4(r, a);
}

// A KW-wide bf16 row's 16-byte unit u in shared memory, swizzled by the row so ldmatrix's eight rows hit eight banks.
template <int KW = LW>
__device__ __forceinline__ int wide(int row, int u) { return row * KW + ((u ^ (row & 7)) << 3); }
// p's [32][32] rows (64 bytes): unit u of row r.
__device__ __forceinline__ int narrow(int row, int u) { return row * KT + ((u ^ ((row >> 1) & 3)) << 3); }

// latent._dense_chunks/_sparse_chunks (sparse 0/1), block (row, HB heads, chunk), 8 warps: online softmax by 32 keys.
template <int HB, int RW = 0>
constexpr int wide_smem() {
  return (HB * (LW + RW) + 2 * KT * (LW + RW) + HB * KT) * 2 + (HB * 4 + 2 * HB) * 4;
}

// RW roped columns (Ling's 64) follow the latent in q and keys; SPLIT: block z is chunk z * sw + sr, keys page z.
template <int HB, bool FP8 = false, int RW = 0, bool SPLIT = false>
__device__ void chunks_wide(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* pos, const int* tok,
                            const int* cnt, float* po, float* pm, float* pl, int64_t rows, int64_t heads, int64_t w,
                            float scale, int sparse, const int* pt, int64_t pstep, const float* scales = nullptr,
                            int sw = 1, int sr = 0) {
  static_assert(!(FP8 && RW != 0), "the E4M3 cache holds the latent alone");
  static_assert(!(SPLIT && FP8), "a split cache holds BF16 rows");
  constexpr int MT = HB / 16, KW = LW + RW;
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* kvs = qs + HB * KW;
  __nv_bfloat16* ps = kvs + 2 * KT * KW;
  float* red = reinterpret_cast<float*>(ps + HB * KT);
  float* stat = red + HB * 4;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5, g = lane >> 2, q = lane & 3;
  const int kg = warp & 3, hh = warp >> 2;
  const int64_t r = blockIdx.x, h0 = (int64_t)blockIdx.y * HB, c = blockIdx.z;
  const int64_t n = sparse ? cnt[r] : 0, limit = sparse ? 0 : (int64_t)pos[0] + r;
  const int64_t start = (SPLIT ? c * sw + sr : c) * CH;
  const bool live = sparse ? start < n : start <= limit;
  // a dense chunk's 512 keys are one page, its rows from that page's; a sparse key's row comes as the token list has it
  const __nv_bfloat16* paged = !FP8 && pt && !sparse && live ? lc + ((int64_t)pt[c] * (pstep / 2) - start * KW) : lc;
  const __nv_bfloat16* keys_at = SPLIT ? lc + (c * CH - start) * KW : paged;
  auto load_kv = [&](int tile) {
    __nv_bfloat16* dst = kvs + (tile & 1) * KT * KW;
    const int64_t first = start + tile * KT;
    for (int idx = t; idx < KT * (KW / 8); idx += blockDim.x) {
      const int key = idx / (KW / 8), u = idx % (KW / 8);
      const int64_t ki = first + key;
      const bool ok = sparse ? ki < n : ki <= limit;
      const int64_t src = ok ? (sparse ? (int64_t)tok[r * w + ki] : ki) : 0;
      if constexpr (FP8) {
        const unsigned char* codes = reinterpret_cast<const unsigned char*>(lc);
        const float row_scale = ok ? scales[src] : 1.0f;
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
          const uint16_t bits = ok ? (uint16_t)codes[src * LW + u * 8 + j] |
                                        ((uint16_t)codes[src * LW + u * 8 + j + 1] << 8) : 0;
          const float2 v = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(bits, __NV_E4M3)));
          dst[wide(key, u) + j] = __float2bfloat16_rn(__fmul_rn(v.x, row_scale));
          dst[wide(key, u) + j + 1] = __float2bfloat16_rn(__fmul_rn(v.y, row_scale));
        }
      } else {
        cp16(dst + wide<KW>(key, u), keys_at + src * KW + u * 8, ok);
      }
    }
  };
  float m[MT][2], l[MT][2], o[MT][8][4];
  for (int i = 0; i < MT; ++i)
    for (int e = 0; e < 2; ++e) m[i][e] = -INFINITY, l[i][e] = 0.0f;
  for (int i = 0; i < MT; ++i)
    for (int j = 0; j < 8; ++j)
      for (int e = 0; e < 4; ++e) o[i][j][e] = 0.0f;
  constexpr int tiles = CH / KT;
  if (live) {
    for (int idx = t; idx < HB * (KW / 8); idx += blockDim.x) {
      const int hd = idx / (KW / 8), u = idx % (KW / 8);
      const bool ok = h0 + hd < heads;
      cp16(qs + wide<KW>(hd, u), qa + (r * heads + (ok ? h0 + hd : 0)) * KW + u * 8, ok);
    }
    load_kv(0);
    commit();
    load_kv(1);
    commit();
  }
  for (int tile = 0; live && tile < tiles; ++tile) {
    if constexpr (FP8) {
      wait<0>();
    } else {
      if (tile + 1 < tiles)
        wait<1>();
      else
        wait<0>();
    }
    __syncthreads();
    const __nv_bfloat16* kv = kvs + (tile & 1) * KT * KW;
    const int64_t first = start + tile * KT;
    float sc[4];
    if (hh < MT) {
      float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
      for (int kk = 0; kk < KW; kk += 32) {
        uint32_t a0[4], a1[4], b[4];
        ldm4<false>(a0, qs + wide<KW>(hh * 16 + (lane & 15), (kk >> 3) + (lane >> 4)));
        ldm4<false>(a1, qs + wide<KW>(hh * 16 + (lane & 15), ((kk + 16) >> 3) + (lane >> 4)));
        ldm4<false>(b, kv + wide<KW>(kg * 8 + (lane & 7), (kk >> 3) + (lane >> 3)));
        glm_router::mma16816(acc, a0, b);
        glm_router::mma16816(acc, a1, b + 2);
      }
      for (int e = 0; e < 2; ++e) {
        const int64_t ki = first + kg * 8 + q * 2 + e;
        const bool ok = sparse ? ki < n : ki <= limit;
        sc[e] = ok ? __fmul_rn(acc[e], scale) : -INFINITY;
        sc[e + 2] = ok ? __fmul_rn(acc[e + 2], scale) : -INFINITY;
      }
      for (int hf = 0; hf < 2; ++hf) {
        const float mx = quad_max(fmaxf(sc[hf * 2], sc[hf * 2 + 1]));
        if (q == 0) red[(hh * 16 + hf * 8 + g) * 4 + kg] = mx;
      }
    }
    __syncthreads();
    if (t < HB * 4) {
      float v = red[t];
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
      if ((t & 3) == 0) stat[t / 4] = v;
    }
    __syncthreads();
    float alpha[MT][2];
    for (int i = 0; i < MT; ++i)
      for (int hf = 0; hf < 2; ++hf) {
        const float tm = stat[i * 16 + hf * 8 + g];
        const bool active = tm != -INFINITY;
        const float nm = active ? fmaxf(m[i][hf], tm) : m[i][hf];
        alpha[i][hf] = active ? (m[i][hf] == -INFINITY ? 0.0f : glm_triton::exp_approx(m[i][hf] - nm)) : 1.0f;
        if (i == hh) {
          float p[2];
          for (int e = 0; e < 2; ++e) {
            const int64_t ki = first + kg * 8 + q * 2 + e;
            const bool ok = sparse ? ki < n : ki <= limit;
            p[e] = active && ok ? glm_triton::exp_approx(sc[hf * 2 + e] - nm) : 0.0f;
          }
          const int hr = i * 16 + hf * 8 + g, col = kg * 8 + q * 2;
          *reinterpret_cast<__nv_bfloat162*>(ps + narrow(hr, col >> 3) + (col & 7)) = __floats2bfloat162_rn(p[0], p[1]);
          const float s = quad_sum(p[0] + p[1]);
          if (q == 0) red[hr * 4 + kg] = s;
        }
        m[i][hf] = nm;
      }
    __syncthreads();
    if (t < HB * 4) {
      float v = red[t];
      v = v + __shfl_xor_sync(0xffffffffu, v, 2);
      v = v + __shfl_xor_sync(0xffffffffu, v, 1);
      if ((t & 3) == 0) stat[HB + t / 4] = v;
    }
    for (int i = 0; i < MT; ++i)
      for (int j = 0; j < 8; ++j)
        for (int e = 0; e < 4; ++e) o[i][j][e] = __fmul_rn(o[i][j][e], alpha[i][e / 2]);
#pragma unroll
    for (int k16 = 0; k16 < KT; k16 += 16) {
      uint32_t a[MT][4];
      for (int i = 0; i < MT; ++i) ldm4<false>(a[i], ps + narrow(i * 16 + (lane & 15), (k16 >> 3) + (lane >> 4)));
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        uint32_t b[4];
        ldm4<true>(b, kv + wide<KW>(k16 + (lane & 15), ((warp * 64 + j * 8) >> 3) + (lane >> 4)));
        for (int i = 0; i < MT; ++i) {
          glm_router::mma16816(o[i][j], a[i], b);
          glm_router::mma16816(o[i][j + 1], a[i], b + 2);
        }
      }
    }
    __syncthreads();
    for (int i = 0; i < MT; ++i)
      for (int hf = 0; hf < 2; ++hf) l[i][hf] = __fmaf_rn(l[i][hf], alpha[i][hf], stat[HB + i * 16 + hf * 8 + g]);
    if (tile + 2 < tiles) load_kv(tile + 2);
    commit();
  }
  for (int i = 0; i < MT; ++i)
    for (int e = 0; e < 4; ++e) {
      const int64_t hd = h0 + i * 16 + g + (e / 2) * 8;
      if (hd >= heads) continue;
      const int64_t base = (c * rows + r) * heads + hd;
      for (int j = 0; j < 8; ++j) po[base * LW + warp * 64 + j * 8 + q * 2 + e % 2] = o[i][j][e];
      if (warp == 0 && q == 0 && e % 2 == 0) {
        pm[base] = m[i][e / 2];
        pl[base] = l[i][e / 2];
      }
    }
}

}  // namespace glm_attn

// Ling-3.1's forms on the latent: per-rank merge, q absorb and v expand at D-wide heads, rope, head gate.
namespace tf_mla {

using namespace glm_triton;

// tf_glm_merge's loop (kept apart: GLM's SASS) over one rank's chunks of a split cache; o, m, l kept in fp32.
__device__ __forceinline__ void merge_part(const float* po, const float* pm, const float* pl, float* qo, float* qm,
                                           float* ql, int64_t rows, int64_t heads, int nch) {
  constexpr int LW = glm_attn::LW;
  const int64_t r = blockIdx.x, h = blockIdx.y;
  const int k0 = threadIdx.x * 4;
  float m = -INFINITY, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int c = 0; c < nch; ++c) {
    const int64_t base = (c * rows + r) * heads + h;
    const float cm = pm[base], cl = pl[base];
    const bool active = cl > 0.0f;
    const float nm = active ? fmaxf(m, cm) : m;
    const float b = active ? glm_triton::exp_approx(cm - nm) : 0.0f;
    if (c == 0) {
      for (int j = 0; j < 4; ++j) o[j] = __fmaf_rn(b, po[base * LW + k0 + j], 0.0f);
      l = __fmaf_rn(b, cl, 0.0f);
    } else {
      const float a = active ? (m == -INFINITY ? 0.0f : glm_triton::exp_approx(m - nm)) : 1.0f;
      for (int j = 0; j < 4; ++j) o[j] = __fmaf_rn(o[j], a, __fmul_rn(b, po[base * LW + k0 + j]));
      l = __fmaf_rn(l, a, __fmul_rn(b, cl));
    }
    m = nm;
  }
  for (int j = 0; j < 4; ++j) qo[(r * heads + h) * LW + k0 + j] = o[j];
  if (threadIdx.x == 0) {
    qm[r * heads + h] = m;
    ql[r * heads + h] = l;
  }
}

// tf_glm_absorb_q's arithmetic at D-wide q_nope: q rows qs apart a head, qa rows qaw apart (the roped columns after).
template <int D>
__device__ __forceinline__ void absorb_q(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa,
                                         int64_t rows, int64_t heads, int64_t rb, int64_t qs, int64_t qaw) {
  constexpr int LW = 512, R = D / 32;
  __shared__ float smem[128];
  const int t = threadIdx.x, k0 = t >> 2, nq = t & 3, warp = t >> 5;
  const int64_t h = blockIdx.x, n0 = (int64_t)blockIdx.y * 32 + nq * 8, r0 = blockIdx.z * rb;
  float w[R][8];
  for (int rep = 0; rep < R; ++rep)
    for (int j = 0; j < 8; ++j) w[rep][j] = bf(wk[(h * D + k0 + 32 * rep) * LW + n0 + j]);
  for (int64_t r = r0; r < r0 + rb && r < rows; ++r) {
    float qv[R];
    for (int rep = 0; rep < R; ++rep) qv[rep] = bf(q[(r * heads + h) * qs + k0 + 32 * rep]);
    for (int j = 0; j < 8; ++j) {
      float a = fmul(w[1][j], qv[1]);
      a = __fmaf_rn(w[0][j], qv[0], a);
      for (int rep = 2; rep < R; ++rep) a = __fmaf_rn(w[rep][j], qv[rep], a);
      for (int o = 16; o >= 4; o >>= 1) a = a + __shfl_xor_sync(0xffffffffu, a, o);
      if ((t & 28) == 0) smem[(nq * 8 + j) * 4 + warp] = a;
    }
    __syncthreads();
    float v = smem[t];
    v = v + __shfl_xor_sync(0xffffffffu, v, 2);
    v = v + __shfl_xor_sync(0xffffffffu, v, 1);
    if ((t & 3) == 0) qa[(r * heads + h) * qaw + (int64_t)blockIdx.y * 32 + t / 4] = __float2bfloat16_rn(v);
    __syncthreads();
  }
}

// tf_glm_expand_v's arithmetic at DV-wide v heads.
template <int DV>
__device__ __forceinline__ void expand_v(const __nv_bfloat16* ol, const __nv_bfloat16* wv, __nv_bfloat16* out,
                                         int64_t rows, int64_t heads, int64_t rb) {
  constexpr int LW = 512;
  __shared__ float smem[32];
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5, kb = (warp & 1) * 256 + lane * 8, nb = warp >> 1;
  const int64_t h = blockIdx.x, n0 = (int64_t)blockIdx.y * 16, r0 = blockIdx.z * rb;
  float w[8][8];
  for (int rep = 0; rep < 8; ++rep)
    for (int j = 0; j < 8; ++j) w[rep][j] = bf(wv[(h * DV + n0 + nb + 2 * rep) * LW + kb + j]);
  for (int64_t r = r0; r < r0 + rb && r < rows; ++r) {
    float o[8];
    for (int j = 0; j < 8; ++j) o[j] = bf(ol[(r * heads + h) * LW + kb + j]);
    for (int rep = 0; rep < 8; ++rep) {
      float a = fmul(w[rep][1], o[1]);
      a = __fmaf_rn(w[rep][0], o[0], a);
      for (int j = 2; j < 8; ++j) a = __fmaf_rn(w[rep][j], o[j], a);
      for (int s = 16; s > 0; s >>= 1) a = a + __shfl_xor_sync(0xffffffffu, a, s);
      if (lane == 0) smem[(nb + 2 * rep) * 2 + (warp & 1)] = a;
    }
    __syncthreads();
    if (t < 32) {
      float v = smem[t];
      v = v + __shfl_xor_sync(0xffffffffu, v, 1);
      if ((t & 1) == 0) out[(r * heads + h) * DV + n0 + t / 2] = __float2bfloat16_rn(v);
    }
    __syncthreads();
  }
}

// GPT-J pairs, block (row, head) of 32: x + r xs + h xh into out + r os + h oh, in place ok; cos and sin times scale.
__device__ __forceinline__ void rope(const __nv_bfloat16* x, int64_t xs, int64_t xh, __nv_bfloat16* out, int64_t os,
                                     int64_t oh, const int* pos, const float* inv_freq, float scale) {
  const int64_t r = blockIdx.x, h = blockIdx.y;
  const int i = threadIdx.x;  // pair i at angle fp32 inv_freq[i] (pos + r); explicit rn ops, no fma contraction
  const __nv_bfloat16* src = x + r * xs + h * xh + 2 * i;
  const float x0 = bf(src[0]), x1 = bf(src[1]);
  float s, c;
  sincosf(__fmul_rn(inv_freq[i], (float)((int64_t)pos[0] + r)), &s, &c);
  s = __fmul_rn(s, scale);  // YaRN's attention factor on cos and sin (exact at 1)
  c = __fmul_rn(c, scale);
  __nv_bfloat16* dst = out + r * os + h * oh + 2 * i;
  dst[0] = __float2bfloat16_rn(__fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s)));
  dst[1] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s)));
}

// The head-wise output gate: o [rows][heads][dv] times bf16(sigmoid(float(g [rows][gs], column h))), rounded to bf16.
__device__ __forceinline__ void gate(__nv_bfloat16* o, int64_t heads, const __nv_bfloat16* g, int64_t gs) {
  const int64_t r = blockIdx.x, h = blockIdx.y, dv = blockDim.x;
  const float z = bf(g[r * gs + h]);
  const float s = rbf(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(0.0f - z))));
  __nv_bfloat16* p = o + (r * heads + h) * dv + threadIdx.x;
  *p = __float2bfloat16_rn(__fmul_rn(bf(*p), s));
}

}  // namespace tf_mla
