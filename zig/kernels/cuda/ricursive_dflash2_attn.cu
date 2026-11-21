// Ripple's DFlash2 draft attention and top-32 on CUDA: dflash2.metal's rc_df_vec, ll_merge_vec, rc_df_topk, controls.

#include <cuda_bf16.h>
#include <math.h>
#include <stddef.h>

#include "common/reduce.cuh"

// tf_rcda, not tf_rcd: the kernels' extern "C" symbols take that prefix.
namespace tf_rcda {

typedef __nv_bfloat16 bf16;

// Head dim 128, a lane's dims 4 lane..4 lane + 3; 16 block rows a stream; 128 keys a chunk (rc_df_vec's blocks).
constexpr int D = 128, PER = 4, BLOCK = 16, CHUNK = 128;
// The partial: 8 warps share the chunk; K then V of 128 keys in dynamic shared memory (64 KiB, opted into at load).
constexpr int NT = 256, WARPS = NT / 32;
// rc_df_topk: 2,048 values a chunk, 1,024 threads, the best 32.
constexpr int TOPN = 2048, TOPT = 1024, TOP = 32;

// A stream: its layer ring (K, then V a ring further), its block's first position, its lowest visible one, its window.
struct Segment {
  unsigned long long ring;
  int anchor, from, span, rows;
};

// Planted faults (controls only): the partial's key range, the merge's output rounding, the top-32 order and inputs.
enum : int { CAUSAL = 1, FLOOR = 2, NO_BLOCK = 4 };
enum : int { TRUNC = 1 };
enum : int { TIES = 1, DROP = 2 };

// Row r's lowest key: max(p + 1 - span, from) at p = anchor + r (FLOOR: one key lower).
__device__ __forceinline__ int window_lo(const Segment& g, int r, int f) {
  const int lo = max(g.anchor + r + 1 - g.span, g.from) - ((f & FLOOR) != 0 ? 1 : 0);
  return max(lo, 0);
}

// One past the block's last key (NO_BLOCK: the block rows left out; CAUSAL: row r's own position the last).
__device__ __forceinline__ int window_hi(const Segment& g, int r, int f) {
  if ((f & NO_BLOCK) != 0) return g.anchor;
  if ((f & CAUSAL) != 0) return g.anchor + r + 1;
  return g.anchor + BLOCK;
}

// Grid (kv_heads, streams, cap): a block's 128-key chunk read once for the group's 10 query heads and 16 block rows.
template <int F>
__device__ __forceinline__ void attn_part(const bf16* __restrict__ Q, const Segment* __restrict__ segs,
                                          float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO,
                                          int kv_heads, int group, int cap, float scale) {
  extern __shared__ __align__(16) unsigned char smem[];
  bf16* Ks = reinterpret_cast<bf16*>(smem);
  bf16* Vs = Ks + CHUNK * D;
  const int kv = blockIdx.x, s = blockIdx.y, z = blockIdx.z;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int heads = kv_heads * group;
  const Segment g = segs[s];
  const int lo0 = window_lo(g, 0, F);  // row 0's window starts lowest
  int hi0 = 0;                         // the union's end: the largest row end
  for (int r = 0; r < BLOCK; ++r) hi0 = max(hi0, window_hi(g, r, F));
  // chunk z of stream s is absolute 128-key chunk first + z; rows see max(p + 1 - span, from) to anchor + 16
  const int first = lo0 / CHUNK;
  const int base = (first + z) * CHUNK;
  const int k0 = max(lo0, base), k1 = min(hi0, base + CHUNK);
  if (k0 < k1) {
    // the chunk's keys of key head kv, ring row p % rows, 16 bytes a thread a step: read once for the 160 (head, row)s
    const bf16* K = reinterpret_cast<const bf16*>(g.ring);
    const bf16* V = K + (size_t)g.rows * kv_heads * D;
    for (int i = threadIdx.x; i < (k1 - k0) * (D / 8); i += NT) {
      const int key = k0 + i / (D / 8), part = (i % (D / 8)) * 8;
      const size_t at = ((size_t)(key % g.rows) * kv_heads + kv) * D + part;
      const int local = (key - base) * D + part;
      *reinterpret_cast<uint4*>(Ks + local) = *reinterpret_cast<const uint4*>(K + at);
      *reinterpret_cast<uint4*>(Vs + local) = *reinterpret_cast<const uint4*>(V + at);
    }
  }
  __syncthreads();
  for (int pair = w; pair < group * BLOCK; pair += WARPS) {
    const int head = kv * group + pair / BLOCK, r = pair % BLOCK;
    // partials: PM the running max, PL the mass, PO the sums; no keys in the chunk: PM -inf, PL 0
    const size_t slot = (((size_t)s * heads + head) * BLOCK + r) * cap + z;
    const int from = max(window_lo(g, r, F), base), to = min(window_hi(g, r, F), base + CHUNK);
    if (from >= to) {
      if (lane == 0) {
        PM[slot] = -INFINITY;
        PL[slot] = 0.0f;
      }
      continue;
    }
    // rc_df_vec: qv = scale * q in fp32 (the softmax scale's rounding point)
    const bf16* q = Q + (((size_t)s * BLOCK + r) * heads + head) * D + PER * lane;
    float qv[PER], acc[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      qv[j] = __fmul_rn(scale, __bfloat162float(q[j]));
      acc[j] = 0.0f;
    }
    float top = -INFINITY, mass = 0.0f;
    for (int i0 = from; i0 < to; i0 += 4) {
      float score[4], vr[4][PER];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const bool live = i0 + u < to;
        const int local = ((live ? i0 + u : from) - base) * D + PER * lane;  // a key past the end reads `from`, as Metal
        const uint2 kb = *reinterpret_cast<const uint2*>(Ks + local);
        const uint2 vb = *reinterpret_cast<const uint2*>(Vs + local);
        const __nv_bfloat162* kp = reinterpret_cast<const __nv_bfloat162*>(&kb);
        const __nv_bfloat162* vp = reinterpret_cast<const __nv_bfloat162*>(&vb);
        const float2 k01 = __bfloat1622float2(kp[0]), k23 = __bfloat1622float2(kp[1]);
        const float2 v01 = __bfloat1622float2(vp[0]), v23 = __bfloat1622float2(vp[1]);
        // the lane's 4-term dot (fma explicit), then the warp's butterfly: every lane holds the same score
        float d = __fmul_rn(qv[0], k01.x);
        d = __fmaf_rn(qv[1], k01.y, d);
        d = __fmaf_rn(qv[2], k23.x, d);
        d = __fmaf_rn(qv[3], k23.y, d);
        const float sc = tf_common::warp_sum(d);
        score[u] = live ? sc : -INFINITY;
        // a key past the end adds nothing (Metal adds 0 times `from`'s value, NaN where that value is infinite)
        vr[u][0] = live ? v01.x : 0.0f;
        vr[u][1] = live ? v01.y : 0.0f;
        vr[u][2] = live ? v23.x : 0.0f;
        vr[u][3] = live ? v23.y : 0.0f;
      }
      // the group's max (fmaxf: a NaN score is skipped), the old mass and sums rescaled, then each key's weight
      const float peak = fmaxf(fmaxf(top, fmaxf(score[0], score[1])), fmaxf(score[2], score[3]));
      const float rescale = expf(__fsub_rn(top, peak));
      mass = __fmul_rn(mass, rescale);
#pragma unroll
      for (int j = 0; j < PER; ++j) acc[j] = __fmul_rn(acc[j], rescale);
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const float wt = expf(__fsub_rn(score[u], peak));
        mass = __fadd_rn(mass, wt);
#pragma unroll
        for (int j = 0; j < PER; ++j) acc[j] = __fadd_rn(acc[j], __fmul_rn(wt, vr[u][j]));
      }
      top = peak;
    }
    if (lane == 0) {
      PM[slot] = top;
      PL[slot] = mass;
    }
    *reinterpret_cast<float4*>(PO + slot * D + PER * lane) = make_float4(acc[0], acc[1], acc[2], acc[3]);
  }
}

// A warp a (row, head), 8 a block: chunks merged as ll_merge_vec (a NaN mass makes the sums NaN), to bf16 (TRUNC: rz).
template <int F>
__device__ __forceinline__ void attn_merge(const float* __restrict__ PM, const float* __restrict__ PL,
                                           const float* __restrict__ PO, bf16* __restrict__ O, int rows, int heads,
                                           int cap) {
  const int unit = blockIdx.x * WARPS + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (unit >= rows * heads) return;
  const int row = unit / heads, head = unit % heads;
  const int s = row / BLOCK, r = row % BLOCK;
  const size_t at = (((size_t)s * heads + head) * BLOCK + r) * cap;
  const float* ms = PM + at;
  const float* ls = PL + at;
  float top = -INFINITY;
  for (int b = 0; b < cap; ++b) top = fmaxf(top, ms[b]);
  // the max, the masses times exp(max_b - max) where mass > 0, the sums by the same factors where not mass <= 0
  float total = 0.0f;
  for (int b = 0; b < cap; ++b)
    if (ls[b] > 0.0f) total = __fadd_rn(total, __fmul_rn(ls[b], expf(__fsub_rn(ms[b], top))));
  float acc[PER] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int b = 0; b < cap; ++b) {
    if (ls[b] <= 0.0f) continue;
    const float f = expf(__fsub_rn(ms[b], top));
    const float4 o = *reinterpret_cast<const float4*>(PO + (at + b) * D + PER * lane);
    acc[0] = __fadd_rn(acc[0], __fmul_rn(f, o.x));
    acc[1] = __fadd_rn(acc[1], __fmul_rn(f, o.y));
    acc[2] = __fadd_rn(acc[2], __fmul_rn(f, o.z));
    acc[3] = __fadd_rn(acc[3], __fmul_rn(f, o.w));
  }
  bf16* y = O + ((size_t)row * heads + head) * D + PER * lane;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const float v = __fdiv_rn(acc[j], total);
    y[j] = (F & TRUNC) != 0 ? __float2bfloat16_rz(v) : __float2bfloat16_rn(v);
  }
}

// rc_df_before: a first by value, then by the lower id (TIES: the higher id).
template <int F>
__device__ __forceinline__ bool before(float a, unsigned ia, float b, unsigned ib) {
  return a > b || (a == b && ((F & TIES) != 0 ? ia > ib : ia < ib));
}

// Grid (chunks, rows), 1,024 threads: row r's chunk c (L, or IDS: VIN, IIN) sorted; best 32 at (r chunks + c) 32.
template <int F, bool IDS>
__device__ __forceinline__ void top32(const bf16* __restrict__ L, const float* __restrict__ VIN,
                                      const unsigned* __restrict__ IIN, int n, int stride, int chunks,
                                      float* __restrict__ VOUT, unsigned* __restrict__ IOUT) {
  // padding is (-inf, 0xffffffff); DROP (the merge's control) reads the first 32 inputs, chunk 0's bests, as padding
  __shared__ float v[TOPN];
  __shared__ unsigned id[TOPN];
  const int row = blockIdx.y, chunk = blockIdx.x, t = threadIdx.x;
  for (int i = t; i < TOPN; i += TOPT) {
    const int at = chunk * TOPN + i;
    if (at < n && !((F & DROP) != 0 && at < TOP)) {
      const size_t src = (size_t)row * stride + at;
      v[i] = IDS ? VIN[src] : __bfloat162float(L[src]);
      id[i] = IDS ? IIN[src] : (unsigned)at;
    } else {
      v[i] = -INFINITY;
      id[i] = 0xffffffffu;
    }
  }
  __syncthreads();
  for (int k = 2; k <= TOPN; k <<= 1)
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = t; i < TOPN; i += TOPT) {
        const int l = i ^ j;
        if (l > i) {
          const bool down = (i & k) == 0;  // descending runs where the k-bit is clear
          const bool swap = down ? before<F>(v[l], id[l], v[i], id[i]) : before<F>(v[i], id[i], v[l], id[l]);
          if (swap) {
            const float tv = v[i];
            v[i] = v[l];
            v[l] = tv;
            const unsigned ti = id[i];
            id[i] = id[l];
            id[l] = ti;
          }
        }
      }
      __syncthreads();
    }
  if (t < TOP) {
    VOUT[((size_t)row * chunks + chunk) * TOP + t] = v[t];
    IOUT[((size_t)row * chunks + chunk) * TOP + t] = id[t];
  }
}

}  // namespace tf_rcda

#define TF_RCD_ATTN_PART(NAME, F)                                                                                    \
  extern "C" __global__ void __launch_bounds__(tf_rcda::NT)                                                         \
      NAME(const tf_rcda::bf16* __restrict__ Q, const tf_rcda::Segment* __restrict__ segs, float* __restrict__ PM, \
           float* __restrict__ PL, float* __restrict__ PO, int kv_heads, int group, int cap, float scale) {        \
    tf_rcda::attn_part<F>(Q, segs, PM, PL, PO, kv_heads, group, cap, scale);                                       \
  }

#define TF_RCD_ATTN_MERGE(NAME, F)                                                                                  \
  extern "C" __global__ void __launch_bounds__(tf_rcda::NT)                                                        \
      NAME(const float* __restrict__ PM, const float* __restrict__ PL, const float* __restrict__ PO,              \
           tf_rcda::bf16* __restrict__ O, int rows, int heads, int cap) {                                          \
    tf_rcda::attn_merge<F>(PM, PL, PO, O, rows, heads, cap);                                                       \
  }

#define TF_RCD_TOP32_PART(NAME, F)                                                                                  \
  extern "C" __global__ void __launch_bounds__(tf_rcda::TOPT)                                                      \
      NAME(const tf_rcda::bf16* __restrict__ L, int n, int stride, int chunks, float* __restrict__ VOUT,          \
           unsigned* __restrict__ IOUT) {                                                                          \
    tf_rcda::top32<F, false>(L, nullptr, nullptr, n, stride, chunks, VOUT, IOUT);                                  \
  }

#define TF_RCD_TOP32_MERGE(NAME, F)                                                                                 \
  extern "C" __global__ void __launch_bounds__(tf_rcda::TOPT)                                                      \
      NAME(const float* __restrict__ VIN, const unsigned* __restrict__ IIN, int n, int stride, int chunks,        \
           float* __restrict__ VOUT, unsigned* __restrict__ IOUT) {                                                \
    tf_rcda::top32<F, true>(nullptr, VIN, IIN, n, stride, chunks, VOUT, IOUT);                                     \
  }

TF_RCD_ATTN_PART(tf_rcd_attn_part, 0)
TF_RCD_ATTN_MERGE(tf_rcd_attn_merge, 0)
TF_RCD_TOP32_PART(tf_rcd_top32_part, 0)
TF_RCD_TOP32_MERGE(tf_rcd_top32_merge, 0)

// Planted controls: causal block, floor a key low, no block rows, truncating merge, higher-id ties, chunk 0 dropped.
TF_RCD_ATTN_PART(tf_rcd_ctl_attn_causal, tf_rcda::CAUSAL)
TF_RCD_ATTN_PART(tf_rcd_ctl_attn_floor, tf_rcda::FLOOR)
TF_RCD_ATTN_PART(tf_rcd_ctl_attn_no_block, tf_rcda::NO_BLOCK)
TF_RCD_ATTN_MERGE(tf_rcd_ctl_attn_merge_trunc, tf_rcda::TRUNC)
TF_RCD_TOP32_PART(tf_rcd_ctl_top32_part_ties, tf_rcda::TIES)
TF_RCD_TOP32_MERGE(tf_rcd_ctl_top32_merge_ties, tf_rcda::TIES)
TF_RCD_TOP32_MERGE(tf_rcd_ctl_top32_merge_drop, tf_rcda::DROP)
