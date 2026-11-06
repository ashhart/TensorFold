// Unlimited-OCR-2's block router and attention on CUDA as decoder.metal's; PLANT > 0: a planted control.
#pragma once
#include "uocr_dec_rows.cuh"

namespace tf_uocr {

// A sum kept as two floats (hi + lo exact), so a running total rounds to fp32 once at each read, as a double would.
struct UdTwo {
  float hi, lo;
};

__device__ __forceinline__ UdTwo ud_two_add(UdTwo a, float b) {
  const float s = a.hi + b;
  const float bb = s - a.hi;
  const float err = (a.hi - (s - bb)) + (b - bb);
  const float lo = a.lo + err;
  const float hi = s + lo;
  return {hi, lo - (hi - s)};
}

// Whether candidate (v, i) beats (bv, bi): the larger score, ties to the lower block (PLANT 1: to the higher).
template <int PLANT>
__device__ __forceinline__ bool ud_beats(float v, unsigned i, float bv, unsigned bi) {
  if (v > bv) return true;
  if (v != bv || i == UD_NONE || bi == UD_NONE) return false;
  return PLANT == 1 ? i > bi : i < bi;
}

// A remote block's router score: q . kb / sqrt(128) clamped at +-1e4, times 128 / sqrt(128).
__device__ __forceinline__ float ud_score(const float* q, const float* kb) {
  float z = 0.0f;
  for (int d = 0; d < UD_RD; d++) z += q[d] * kb[d];
  return ud_clamp(z * UD_INV_SQRT_HD) * UD_ROUTER_SCALE;
}

// A row a block of 256: the top 12 of 16 remote blocks by entmax-1.5 into SEL and BIAS; PLANT 1 ties high, 2 no mean.
template <int PLANT>
__device__ __forceinline__ void ud_route(const bf16* RQ, const float* KB, const unsigned* masks, const int* qmask,
                                         unsigned* SEL, float* BIAS, int pos0, int words) {
  __shared__ float part[33];
  __shared__ float q[UD_RD];
  __shared__ float score[4096];
  __shared__ float best_v[8];
  __shared__ unsigned best_i[8];
  __shared__ float top_v[UD_TOP];
  __shared__ unsigned top_i[UD_TOP];
  const int row = blockIdx.x, t = threadIdx.x, lane = t & 31, sg = t >> 5;
  const int pos = pos0 + row;
  const int nrem = pos / UD_BLK >= UD_LOCAL_BLOCKS ? pos / UD_BLK - UD_LOCAL_BLOCKS + 1 : 0;
  if (t < UD_RD) q[t] = ld(RQ + (size_t)row * UD_RD + t);
  __syncthreads();
  const int m = qmask[row];
  float own = 0.0f;
  for (int c = t; c < nrem; c += 256) {
    const bool cand = m < 0 || (c / 32 < words && ((masks[(size_t)m * words + c / 32] >> (c % 32)) & 1u) != 0);
    score[c] = cand ? ud_score(q, KB + (size_t)c * UD_RD) : -INFINITY;
    own += cand ? 1.0f : 0.0f;
  }
  // no candidate on the query's page among the remote blocks: every remote block is one
  if (block_sum(own, part) == 0.0f) {
    for (int c = t; c < nrem; c += 256) score[c] = ud_score(q, KB + (size_t)c * UD_RD);
  }
  __syncthreads();
  // the 16 best in order, ties to the lower block
  int n = 0;
  for (; n < UD_TOP; n++) {
    float bv = -INFINITY;
    unsigned bi = UD_NONE;
    for (int c = t; c < nrem; c += 256) {
      const float v = score[c];
      if (v != -INFINITY && ud_beats<PLANT>(v, c, bv, bi)) {
        bv = v;
        bi = c;
      }
    }
    for (int off = 16; off > 0; off /= 2) {
      const float ov = __shfl_down_sync(0xffffffffu, bv, off);
      const unsigned oi = __shfl_down_sync(0xffffffffu, bi, off);
      if (ud_beats<PLANT>(ov, oi, bv, bi)) {
        bv = ov;
        bi = oi;
      }
    }
    if (lane == 0) {
      best_v[sg] = bv;
      best_i[sg] = bi;
    }
    __syncthreads();
    if (t == 0) {
      float v = best_v[0];
      unsigned i = best_i[0];
      for (int s = 1; s < 8; s++)
        if (ud_beats<PLANT>(best_v[s], best_i[s], v, i)) {
          v = best_v[s];
          i = best_i[s];
        }
      top_v[n] = v;
      top_i[n] = i;
      if (i != UD_NONE) score[i] = -INFINITY;
    }
    __syncthreads();
    if (top_i[n] == UD_NONE) break;
  }
  if (t != 0) return;
  unsigned* sel = SEL + (size_t)row * UD_READ;
  float* bias = BIAS + (size_t)row * UD_READ;
  const int keep = min(n, UD_READ);
  // entmax-1.5 over the n sorted scores: X = (x - max) / 2, the threshold from running means, Y = max(X - tau, 0)^2
  float X[UD_TOP], tau[UD_TOP];
  UdTwo s1 = {0.0f, 0.0f}, s2 = {0.0f, 0.0f};
  int support = 0;
  for (int i = 0; i < n; i++) {
    X[i] = (top_v[i] - top_v[0]) / 2.0f;
    s1 = ud_two_add(s1, X[i]);
    s2 = ud_two_add(s2, X[i] * X[i]);
    const float rho = (float)(i + 1);
    const float mean = (s1.hi + s1.lo) / rho, msq = (s2.hi + s2.lo) / rho;
    const float ss = rho * (msq - mean * mean);
    const float delta = (1.0f - ss) / rho;
    tau[i] = mean - sqrtf(fmaxf(delta, 0.0f));
    support += tau[i] <= X[i] ? 1 : 0;
  }
  const float ts = n > 0 ? tau[max(support, 1) - 1] : 0.0f;
  float logp[UD_READ];
  UdTwo total = {0.0f, 0.0f};
  for (int i = 0; i < keep; i++) {
    const float y = fmaxf(X[i] - ts, 0.0f);
    logp[i] = logf(fmaxf(y * y, UD_FLOOR));
    total = ud_two_add(total, logp[i]);
  }
  const float mu = (total.hi + total.lo) / (float)max(keep, 1);
  for (int i = 0; i < UD_READ; i++) {
    sel[i] = i < keep ? top_i[i] : UD_NONE;
    bias[i] = i < keep ? (PLANT == 2 ? logp[i] : logp[i] - mu) : 0.0f;
  }
}

// 4 bf16 at p (8-byte aligned) widened.
__device__ __forceinline__ float4 ud_load4(const bf16* p) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
  const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
  return make_float4(a.x, a.y, b.x, b.y);
}

// Warps a (head, row) block of tf_uocr_attend: each takes every 8th key, as decoder.metal's simdgroups do.
constexpr int UD_ATT_WARPS = 8;

// A block's maximum of every thread's `v` (fmaxf, dropping NaN), returned to all of them.
__device__ __forceinline__ float ud_block_max(float v, float* part) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, n = blockDim.x >> 5;
  v = warp_max(v);
  if (lane == 0) part[w] = v;
  __syncthreads();
  if (w == 0) {
    float m = lane < n ? part[lane] : -INFINITY;
    m = warp_max(m);
    if (lane == 0) part[32] = m;
  }
  __syncthreads();
  const float out = part[32];
  __syncthreads();
  return out;
}

// A (head, row) a block of 256: fp32 attention, window and read blocks; PLANT 1 reads 129 local keys, 2 no biases.
template <int PLANT>
__device__ __forceinline__ void ud_attend(const bf16* Q, const bf16* K, const bf16* V, const unsigned* SEL,
                                          const float* BIAS, bf16* O, int pos0, int bf16_bias) {
  constexpr int MAXK = UD_LOCAL + 1 + UD_READ * UD_BLK;
  constexpr int window = PLANT == 1 ? UD_LOCAL + 1 : UD_LOCAL;
  constexpr int NT = UD_ATT_WARPS * 32;
  __shared__ unsigned keys[MAXK];
  __shared__ float kbias[MAXK];
  __shared__ float sc[MAXK];
  __shared__ float4 pv[UD_ATT_WARPS][32];
  __shared__ float part[33];
  __shared__ int start[UD_READ + 1];
  const int h = blockIdx.x, row = blockIdx.y, t = threadIdx.x, w = t >> 5, lane = t & 31;
  const unsigned pos = pos0 + row;
  const unsigned lo = pos >= window - 1 ? pos - (window - 1) : 0;
  const int nloc = pos - lo + 1;
  const unsigned* sel = SEL + (size_t)row * UD_READ;
  const float* bias = BIAS + (size_t)row * UD_READ;
  if (t == 0) {
    int n = nloc;
    for (int s = 0; s < UD_READ; s++) {
      start[s] = n;
      const unsigned c = sel[s];
      if (c != UD_NONE && c * UD_BLK < lo) n += min((unsigned)UD_BLK, lo - c * UD_BLK);
    }
    start[UD_READ] = n;
  }
  __syncthreads();
  const int total = start[UD_READ];
  for (int i = t; i < total; i += NT) {
    if (i < nloc) {
      keys[i] = lo + i;
      kbias[i] = 0.0f;
    } else {
      int s = 0;
      while (start[s + 1] <= i) s++;
      keys[i] = sel[s] * UD_BLK + (i - start[s]);
      kbias[i] = PLANT == 2 ? 0.0f : (bf16_bias != 0 ? bf(bias[s]) : bias[s]);
    }
  }
  __syncthreads();
  // a lane's 4 channels of q and of each key: their 4-term dot, then the warp's sum
  const int o = h * UD_HD + 4 * lane;
  const float4 q = ud_load4(Q + (size_t)row * UD_H + o);
  for (int i = w; i < total; i += UD_ATT_WARPS) {
    const float4 k = ud_load4(K + (size_t)keys[i] * UD_H + o);
    const float s = warp_sum(q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w);
    if (lane == 0) sc[i] = s * UD_INV_SQRT_HD + kbias[i];
  }
  __syncthreads();
  float top = -INFINITY;
  for (int i = t; i < total; i += NT) top = fmaxf(top, sc[i]);
  top = ud_block_max(top, part);
  float mass = 0.0f;
  for (int i = t; i < total; i += NT) {
    const float e = expf(sc[i] - top);
    sc[i] = e;
    mass += e;
  }
  mass = block_sum(mass, part);
  for (int i = t; i < total; i += NT) sc[i] = sc[i] / mass;
  __syncthreads();
  // P x V: warp w over every 8th key, then the warps' sums added in warp order
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int i = w; i < total; i += UD_ATT_WARPS) {
    const float4 v = ud_load4(V + (size_t)keys[i] * UD_H + o);
    const float p = sc[i];
    acc.x += p * v.x;
    acc.y += p * v.y;
    acc.z += p * v.z;
    acc.w += p * v.w;
  }
  pv[w][lane] = acc;
  __syncthreads();
  if (w != 0) return;
  for (int s = 1; s < UD_ATT_WARPS; s++) {
    const float4 b = pv[s][lane];
    acc.x += b.x;
    acc.y += b.y;
    acc.z += b.z;
    acc.w += b.w;
  }
  const __nv_bfloat162 x = __floats2bfloat162_rn(acc.x, acc.y);
  const __nv_bfloat162 y = __floats2bfloat162_rn(acc.z, acc.w);
  uint2 u;
  u.x = *reinterpret_cast<const unsigned*>(&x);
  u.y = *reinterpret_cast<const unsigned*>(&y);
  *reinterpret_cast<uint2*>(O + (size_t)row * UD_H + o) = u;
}

}  // namespace tf_uocr
