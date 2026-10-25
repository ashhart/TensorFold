// MiniMax M3.1's MSA sparse-indexer on CUDA, the exllamav3-fork m3_sparse_attn contracts by hand:
// block-max-pool scores (fp32), the fixed-width top-16 with the tie contract, and the block-gather
// decode attention over the fp16 paged cache. The grading reference is this family's CPU mirror
// (zig/src/families/minimax_m3_vl/sparse.zig, 3/3 green) and the fork's INTERFACE-M3-KERNELS.md §1-2.
//
// Contract numbers (asserted in the family's config.zig): index_n_heads 4 == num_key_value_heads,
// index_head_dim 128, index_block_size 128, index_topk_blocks 16, index_local_blocks 1, init_block 0.
#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

namespace m3 {

constexpr int D = 128;        // index_head_dim == head_dim == num_key_value_heads' 4 x 128? D is the per-head width
constexpr int BLOCK = 128;    // index_block_size: keys per scored block; a 128-run never straddles PAGE 256
constexpr int TOPK = 16;      // index_topk_blocks: the fixed selection width
constexpr int PAGE = 256;     // the paged cache's in-tree PAGE_SIZE
constexpr float NEG_INF = -__builtin_huge_valf();
constexpr float POS_INF = __builtin_huge_valf();

__device__ __forceinline__ float h2f(__half v) { return __half2float(v); }
__device__ __forceinline__ bool pnan(float s) { return s != s; } // the NaN rule: propagate, amax keeps NaN

// f32 warp sum, a lane's 4-dim partial then all 32 lanes (glue convention: no contraction, the flags keep it).
__device__ __forceinline__ float warp_sum(float s) {
  for (int o = 16; o > 0; o >>= 1) s = s + __shfl_xor_sync(0xffffffffu, s, o);
  return s;
}

}  // namespace m3

// m3_indexer_scores (_ref and _m3_indexer_scores_kernel): one program per (query row, key block,
// batch); its warp w owns index head w alone, so the four heads run without any cross-warp state.
// Scoring is fp32; a key is masked iff k_pos < 0 or k_pos > q_pos (right-open causal on content
// positions); the block score is the masked amax over the block's 128 keys, then boosts OVERRIDE:
// the query's own block (local_blocks=1) plus the leading [0, init) blocks become +inf after the
// full reduction. A NaN dot makes the block score NaN (torch amax's class). Slot anchor: blocks
// are absolute key-slot runs, NB = ceil(Sk/128).
extern "C" __global__ void __launch_bounds__(128)
    tf_m3_scores(const __half* __restrict__ q,      // (B, H, Sq, D) index queries
                 const __half* __restrict__ k,      // (B, Sk, D) index keys (plane rows or fresh K)
                 const int64_t* __restrict__ qpos,  // (B, Sq) content positions of the queries
                 const int64_t* __restrict__ kpos,  // (B, Sk); -1 = padded/invalid
                 float* __restrict__ out,           // (B, H, Sq, NB)
                 int64_t Sq, int64_t NB, int64_t Sk, int64_t H, int local, int init) {
  const int64_t sq = blockIdx.x, nb = blockIdx.y, b = blockIdx.z;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int64_t qp = qpos[b * Sq + sq];
  // Every program re-reads its k rows four times (one per head); the transient stays (D,) per warp.
  const __half* kb = k + (b * Sk + nb * m3::BLOCK) * m3::D + lane * 4;
  float best = m3::NEG_INF;
  bool nan = false;
  for (int j = 0; j < m3::BLOCK; ++j) {
    const int64_t slot = nb * m3::BLOCK + j;
    if (slot >= Sk) break; // blocks anchored to absolute slots: this block's own short tail
    const int64_t kp = kpos[b * Sk + slot];
    if (kp < 0 || kp > qp) continue; // masked iff out of the right-open causal window
    const __half* qb = q + ((b * H + w) * Sq + sq) * m3::D + lane * 4;
    const __half* kr = kb + j * m3::D;
    float s = 0.0f;
    for (int m = 0; m < 4; ++m) s = s + m3::h2f(qb[m]) * m3::h2f(kr[m]); // fp16 products are exact in fp32
    s = m3::warp_sum(s);
    if (lane == 0) {
      if (m3::pnan(s)) nan = true;         // amax keeps a NaN: the score's class follows
      else if (s > best) best = s;         // ties at greater... amax keeps the value, ids do not matter
    }
  }
  if (lane == 0) {
    const int64_t qb = qp / m3::BLOCK; // the query's own block, slot-anchored
    const bool boost = (nb >= qb - (local - 1) && nb <= qb) || nb < init;
    if (nan) best = m3::POS_INF / m3::POS_INF; // a NaN row: NaN wins the max like torch's amax
    out[((b * H + w) * Sq + sq) * NB + nb] = boost ? m3::POS_INF : best;
  }
}

// m3_topk16 (_m3_topk_kernel): one score row per program, k picks by iterative argmax with per-pick
// invalidation marks; descending score, ties at the SMALLER block id, valid ids left-packed, the
// fixed topk width with -1 right-pad (-inf never yields an id). NaN never wins a comparison: it
// is the stated topk rule and the CPU mirror checks it.
extern "C" __global__ void __launch_bounds__(128)
    tf_m3_topk(const float* __restrict__ scores, int32_t* __restrict__ out,
               int64_t NB, int64_t rows, int k) {
  const int64_t r = blockIdx.x;
  const float* row = scores + r * NB;
  int32_t* dst = out + r * k;
  __shared__ unsigned int marks[32];      // NB <= 1024: one mark bit per block (the glue checks the bound)
  for (int i = threadIdx.x; i < 32; i += blockDim.x) marks[i] = 0u;
  __shared__ float s_best;
  __shared__ int i_best, sh_written;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  __shared__ float wbest[4];
  __shared__ int wid[4];
  if (t == 0) sh_written = 0;
  __syncthreads();
  bool done = false;
  for (int pick = 0; pick < k; ++pick) {
    if (done) break;
    float bs = m3::NEG_INF;
    int bi = -1;
    for (int64_t i = t; i < NB; i += blockDim.x) {
      if (marks[i >> 5] & (1u << (i & 31))) continue;
      const float s = row[i];
      if (s > bs && !m3::pnan(s)) { bs = s; bi = (int)i; } // plain >: ties keep the earlier (smaller) id
    }
    for (int o = 16; o > 0; o >>= 1) {
      const float os = __shfl_down_sync(0xffffffffu, bs, o);
      const int oi = __shfl_down_sync(0xffffffffu, bi, o);
      if (os > bs || (os == bs && oi < bi && oi >= 0)) { bs = os; bi = oi; }
    }
    if (lane == 0) { wbest[warp] = bs; wid[warp] = bi; }
    __syncthreads();
    if (t == 0) {
      float s = m3::NEG_INF;
      int i = -1;
      for (int wp = 0; wp < 4; ++wp)
        if (wbest[wp] > s || (wbest[wp] == s && wid[wp] < i && wid[wp] >= 0)) { s = wbest[wp]; i = wid[wp]; }
      s_best = s;
      i_best = i;
    }
    __syncthreads();
    if (s_best == m3::NEG_INF) { // exhausted: the fixed width's -1 right-pad, left-packed so far
      if (t == 0)
        for (; sh_written < k; ++sh_written) dst[sh_written] = -1;
      done = true;
      __syncthreads();
      break;
    }
    if (t == 0) {
      dst[sh_written++] = i_best;
      atomicOr(&marks[i_best >> 5], 1u << (i_best & 31)); // the per-pick invalidation mark
    }
    __syncthreads();
  }
}
// m3_sparse_decode_attn's split kernel (_m3_decode_kernel): one program per (batch, q head, split);
// 128 threads, a dim each. Its split of the fixed k-wide selection (-1 ids skipped) gathers the
// fp16 paged cache through the block table; GQA maps q head hq onto kv head hq/group. Online
// softmax in fp32 (acc, m, l): every finite key contributes — a sub-max key adds __expf(full -
// m_new) * v on top of the rescale-carried sum (never wipes it); a NaN score drops the whole key
// (m, l, acc unchanged). A key is masked iff its logical position >= cache_seqlens. The partials
// land row-major in the (B, Hq, nsplit, D + 2) fp32 scratch the combine kernel merges.
// Zero-valid-key splits exit with l = 0; the combine then returns an all-zero row, never a NaN.
extern "C" __global__ void __launch_bounds__(128)
    tf_m3_decode(const __half* __restrict__ q,        // (B, Hq, D)
                 const __half* __restrict__ kt,       // (P, PAGE, Hkv, D) fp16 paged K
                 const __half* __restrict__ vt,       // (P, PAGE, Hkv, D)
                 const int32_t* __restrict__ table,   // (B, npp) logical page -> physical page
                 const int32_t* __restrict__ sel,     // (B, Hkv, k) fixed-width, -1 right-pad
                 const int64_t* __restrict__ seqlens, // (B,) logical key lengths (before the append)
                 float* __restrict__ part,            // (B, Hq, nsplit, D + 2) acc | m | l
                 int64_t Hq, int64_t Hkv, int64_t npp, float scale, int k, int nsplit) {
  const int64_t b = blockIdx.x, hq = blockIdx.y;
  const int split = blockIdx.z, t = threadIdx.x;
  const int per = (k + nsplit - 1) / nsplit;
  const int kvh = (int)(hq / (Hq / Hkv)); // the GQA mapping: index heads == kv heads, per-head lists
  const __half* qb = q + (b * Hq + hq) * m3::D + t;
  const int32_t* row = sel + (b * Hkv + kvh) * k;
  __shared__ float wsum[4], sh_alpha, sh_p;
  float acc = 0.0f, m = m3::NEG_INF, l = 0.0f;
  const int lo = split * per, hi = lo + per < k ? lo + per : k;
  for (int si = lo; si < hi; ++si) {
    const int bid = row[si];
    if (bid < 0) continue; // left-packed: -1 only at the tail
    const __half* kr = kt + (((int64_t)bid * m3::BLOCK + 0) * Hkv + kvh) * m3::D + t;
    for (int j = 0; j < m3::BLOCK; ++j) {
      const int64_t p = (int64_t)bid * m3::BLOCK + j;
      if (p >= seqlens[b]) continue;
      const int64_t page = table[b * npp + p / m3::PAGE]; // 128 runs never straddle the 256 page
      const int64_t off = p % m3::PAGE;
      const __half* krow = kt + ((page * m3::PAGE + off) * Hkv + kvh) * m3::D + t;
      const __half* vrow = vt + ((page * m3::PAGE + off) * Hkv + kvh) * m3::D + t;
      float s = m3::h2f(qb[0]) * m3::h2f(krow[0]); // fp16 products exact in fp32; the flags keep add+mul apart
      s = m3::warp_sum(s);
      const int warp = t >> 5, lane = t & 31;
      if (lane == 0) wsum[warp] = s;
      __syncthreads();
      float m_new = m, alpha = 1.0f, pj = 0.0f;
      if (t == 0) {
        const float full = scale * ((wsum[0] + wsum[1]) + (wsum[2] + wsum[3]));
        if (full == full) { // a NaN score drops the whole key: m, l and acc stay as they were
          m_new = full > m ? full : m;
          alpha = m == m3::NEG_INF ? 0.0f : __expf(m - m_new); // rescale the carried sum (1 when full <= m)
          pj = __expf(full - m_new);
          l = l * alpha + pj;
          m = m_new;
        } else { alpha = 1.0f; pj = 0.0f; m_new = m; }
        sh_alpha = alpha;
        sh_p = pj;
      }
      __syncthreads();
      acc = acc * sh_alpha + sh_p * m3::h2f(vrow[0]);
      __syncthreads();
    }
  }
  float* dst = part + (((b * Hq + hq) * nsplit) + split) * (m3::D + 2);
  dst[t] = acc;
  if (t == 0) {
    dst[m3::D] = m;
    dst[m3::D + 1] = l;
  }
}

// tf_m3_swigluoai: the M3.1 MoE/dense activation rows, the fork's activation.cu:124 contract by
// hand: (clamp(u, -limit, limit) + 1) * min(g, limit) * sigmoid(1.702 * min(g, limit)) -> fp16.
// The mainline EXL3 experts' fused gateup epilogue computes SiLU x up (exl3_experts.cu's
// tf_exl3_gateup_epilogue), which cannot express the alpha = 1.702 sigmoid form, so this family
// applies its own activation between the grouped gate/up GEMM partials and the down GEMM. The
// grading reference is serve.zig's swigluoai (the same contract at f32 exp).
extern "C" __global__ void __launch_bounds__(128)
    tf_m3_swigluoai(const __half* __restrict__ g,   // (rows, N) fp16 gate rows
                    const __half* __restrict__ u,   // (rows, N) fp16 up rows
                    __half* __restrict__ o,         // (rows, N) fp16 out
                    int64_t n, float limit) {
  const int64_t p = blockIdx.x;
  const int64_t base = ((int64_t)blockIdx.y * blockDim.x + threadIdx.x) * 4;
  const __half* gr = g + p * n;
  const __half* ur = u + p * n;
  __half* dst = o + p * n;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int64_t i = base + j;
    if (i >= n) return;
    float gv = m3::h2f(gr[i]);
    float uv = m3::h2f(ur[i]);
    gv = gv > limit ? limit : gv;              // min(g, limit): clipped from above only
    uv = uv > limit ? limit : (uv < -limit ? -limit : uv); // torch.clamp both ways
    const float glu = gv / (1.0f + __expf(-1.702f * gv)); // sigmoid(1.702 g), the fork's fast exp
    dst[i] = __float2half_rn((uv + 1.0f) * glu);
  }
}

// _m3_decode_combine_kernel: the four splits' fp32 (acc, m, l) merge into the row's fp16 output
// in the q dtype; an all-l-zero row (no selected keys were valid) returns zeros, never a NaN.
extern "C" __global__ void __launch_bounds__(128)
    tf_m3_decode_combine(const float* __restrict__ part, __half* __restrict__ out,
                         int64_t Hq, int nsplit) {
  const int64_t b = blockIdx.x, hq = blockIdx.y;
  const int t = threadIdx.x;
  const float* base = part + (((b * Hq + hq) * nsplit) + 0) * (m3::D + 2);
  __shared__ float ms[8], ls[8], sh_l, sh_m;
  if (t == 0) {
    float mstar = m3::NEG_INF, ltot = 0.0f;
    for (int i = 0; i < nsplit; ++i) {
      const float m = base[i * (m3::D + 2) + m3::D], l = base[i * (m3::D + 2) + m3::D + 1];
      ms[i] = l > 0.0f ? m : m3::NEG_INF; // an empty split never moves the max
      ls[i] = l;
      if (mstar < ms[i]) mstar = ms[i];
    }
    // the merge's denominator is the max-rescaled total: a split's l only lives at its own m, so
    // l * exp(m - m*) restores the shared scale before summing (a raw sum understates every
    // split whose max sits below the global one — with nsplit=1 the weight is 1 and it vanishes)
    for (int i = 0; i < nsplit; ++i)
      if (ls[i] > 0.0f) ltot += __expf(ms[i] - mstar) * ls[i];
    sh_m = mstar;
    sh_l = ltot;
  }
  __syncthreads();
  __half* dst = out + (b * Hq + hq) * m3::D + t;
  if (sh_l == 0.0f) { *dst = __float2half_rn(0.0f); return; } // zero-valid-key row: zeros, not NaN
  float v = 0.0f;
  for (int i = 0; i < nsplit; ++i)
    if (ls[i] > 0.0f) v += __expf(ms[i] - sh_m) * base[i * (m3::D + 2) + t];
  *dst = __float2half_rn(v / sh_l);
}
