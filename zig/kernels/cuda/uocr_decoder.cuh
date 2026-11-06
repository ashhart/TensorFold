// Unlimited-OCR-2's decoder steps on CUDA (uocr/decoder.metal's).
#pragma once
#include "uocr_common.cuh"
#include "uocr_dec_attend.cuh"
#include "uocr_dec_moe.cuh"

// Token rows: x[i] = embed[ids[i]] (1280 wide), image tokens' rows left for their pictures. Grid (rows, 5).
extern "C" __global__ void __launch_bounds__(256) tf_uocr_embed(const unsigned* __restrict__ ids, const bf16* __restrict__ embed,
                                                                bf16* __restrict__ x, unsigned image_token) {
  const int c = blockIdx.y * 256 + threadIdx.x;
  const unsigned id = ids[blockIdx.x];
  if (c >= 1280 || id == image_token) return;
  x[(size_t)blockIdx.x * 1280 + c] = embed[(size_t)id * 1280 + c];
}

// Gathered rows: y[i] = x[rows[i]] (1280 wide). Grid (count, 5).
extern "C" __global__ void __launch_bounds__(256) tf_uocr_gather(const bf16* __restrict__ x, const unsigned* __restrict__ rows,
                                                                 bf16* __restrict__ y) {
  ud_gather<0>(x, rows, y);
}

// An expert's rows back to their slots: slots[dest[i]] = y[i]. Grid (count, 5).
extern "C" __global__ void __launch_bounds__(256) tf_uocr_scatter(const bf16* __restrict__ y, const unsigned* __restrict__ dest,
                                                                  bf16* __restrict__ slots) {
  const int c = blockIdx.y * 256 + threadIdx.x;
  if (c >= 1280) return;
  slots[(size_t)dest[blockIdx.x] * 1280 + c] = y[(size_t)blockIdx.x * 1280 + c];
}

// One token's projection y[n] = bf16(W[n, :] . x) with fp32 sums: 4 rows a warp, 8 warps a block; k a multiple of 8.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_matvec(const bf16* __restrict__ x, const bf16* __restrict__ W,
                                                                 bf16* __restrict__ y, int n, int k) {
  ud_matvec<0>(x, W, y, n, k);
}

// The greedy pick on the device: the first maximum of bf16 logits, EOS skipped when `bar_eos`.
extern "C" __global__ void __launch_bounds__(1024) tf_uocr_argmax(const bf16* __restrict__ logits, int n, int bar_eos,
                                                                  unsigned* __restrict__ out) {
  __shared__ float bv[32];
  __shared__ int bi[32];
  float best = -INFINITY;
  int at = 0x7fffffff;
  for (int i = threadIdx.x; i < n; i += 1024) {
    if (bar_eos && i == 1) continue;
    const float v = ld(logits + i);
    if (v > best) {
      best = v;
      at = i;
    }
  }
  // the warp's best, ties to the lower id
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, at, o);
    if (ov > best || (ov == best && oi < at)) {
      best = ov;
      at = oi;
    }
  }
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  if (lane == 0) {
    bv[w] = best;
    bi[w] = at;
  }
  __syncthreads();
  if (w != 0) return;
  best = bv[lane];
  at = bi[lane];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, at, o);
    if (ov > best || (ov == best && oi < at)) {
      best = ov;
      at = oi;
    }
  }
  if (lane == 0) out[0] = (unsigned)(at == 0x7fffffff ? 0 : at);
}

// The decoder's steps (uocr_dec_*.cuh); every pointer is a device address, every count an int.

// Grid (rows), 256 threads: the residual add when `add`, then the RMS norm.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_norm(bf16* R, const bf16* __restrict__ A, const bf16* __restrict__ w,
                                                               bf16* __restrict__ Y, int add, float eps) {
  ud_norm<0>(R, A, w, Y, add, eps);
}

// Grid (rows), 128 threads: rotary q, the caches' keys and values at pos0, the router's q and keys.
extern "C" __global__ void __launch_bounds__(128) tf_uocr_qk(const bf16* __restrict__ P, const bf16* __restrict__ RP,
                                                             const bf16* __restrict__ COS, const bf16* __restrict__ SIN,
                                                             const bf16* __restrict__ qn, const bf16* __restrict__ kn,
                                                             bf16* __restrict__ Q, bf16* __restrict__ K, bf16* __restrict__ V,
                                                             bf16* __restrict__ RQ, bf16* __restrict__ RK, int pos0, float eps) {
  ud_qk<0>(P, RP, COS, SIN, qn, kn, Q, K, V, RQ, RK, pos0, eps);
}

// Grid (blocks from c0), 32 threads: each filled block's summary key.
extern "C" __global__ void __launch_bounds__(32) tf_uocr_block_keys(const bf16* __restrict__ RK, const bf16* __restrict__ hc,
                                                                    float* __restrict__ KB, int c0) {
  ud_block_keys<0>(RK, hc, KB, c0);
}

// Grid (rows), 256 threads: each row's 12 remote blocks and their biases.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_route(const bf16* __restrict__ RQ, const float* __restrict__ KB,
                                                                const unsigned* __restrict__ masks, const int* __restrict__ qmask,
                                                                unsigned* __restrict__ SEL, float* __restrict__ BIAS, int pos0,
                                                                int words) {
  ud_route<0>(RQ, KB, masks, qmask, SEL, BIAS, pos0, words);
}

// Grid (10 heads, rows), 256 threads: each row's local and routed attention, a block a head.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_attend(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                                                                 const bf16* __restrict__ V, const unsigned* __restrict__ SEL,
                                                                 const float* __restrict__ BIAS, bf16* __restrict__ O, int pos0,
                                                                 int bf16_bias) {
  ud_attend<0>(Q, K, V, SEL, BIAS, O, pos0, bf16_bias);
}

// Grid (rows), 1,024 threads: each row's 6 experts and their weights.
extern "C" __global__ void __launch_bounds__(1024) tf_uocr_moe_gate(const bf16* __restrict__ X, const bf16* __restrict__ G,
                                                                   unsigned* __restrict__ IDS, float* __restrict__ WTS) {
  ud_moe_gate<0>(X, G, IDS, WTS);
}

// Grid (rows, width / 256 rounded up), 256 threads: SwiGLU rows.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_silu_mul(const bf16* __restrict__ G, const bf16* __restrict__ U,
                                                                   bf16* __restrict__ Y, int width, int gstride, int ustride) {
  ud_silu_mul<0>(G, U, Y, width, gstride, ustride);
}

// Grid (6 x 2 x width / 32), 256 threads: one token's picked experts' gate and up rows.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_expert_gu(const bf16* __restrict__ x, const unsigned long long* __restrict__ E,
                                                                    const unsigned* __restrict__ ids, bf16* __restrict__ GU, int width) {
  ud_expert_gu<0>(x, E, ids, GU, width);
}

// Grid (6 x 1280 / 32), 256 threads: one token's picked experts' down rows.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_expert_down(const bf16* __restrict__ A, const unsigned long long* __restrict__ E,
                                                                      const unsigned* __restrict__ ids, bf16* __restrict__ D, int width) {
  ud_expert_down<0>(A, E, ids, D, width);
}

// Grid (rows, 5), 256 threads: the experts' weighted sum, then the shared expert's rows.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_combine(const bf16* __restrict__ D, const unsigned* __restrict__ slot,
                                                                  const float* __restrict__ WTS, const bf16* __restrict__ S,
                                                                  bf16* __restrict__ Y) {
  ud_combine<0>(D, slot, WTS, S, Y);
}
