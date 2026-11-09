// Ripple's DFlash2 drafter glue (tp2_draft.zig): tap gathers, norms, 4-tap convolutions, head norms, attention, top-k.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include "common/reduce.cuh"

using tf_common::warp_sum;

namespace tf_ripd {

constexpr int HD = 128;  // the drafter's head width
constexpr int CHUNK = 256;  // keys an attention partial takes

// A planted control's fault (the contract's red runs, tf_rtp2d_ctl_*); NONE is the served kernel.
enum Drop { NONE = 0, DROP = 1 };

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// A drafting stream in a hold: its context ring, the first position the ring holds, the pending row's position.
struct Seg { unsigned long long ring; long long begin, position; };

}  // namespace tf_ripd

using namespace tf_ripd;

// Columns [col, col + width) of ring rows (start + i) % TR (rows of stride bf16) into OUT row i. Grid (rows), 256.
template <int D>
__device__ __forceinline__ void gather_rows(const __nv_bfloat16* __restrict__ RING, int TR, long long start, int stride, int col,
                                            int width, __nv_bfloat16* __restrict__ OUT) {
  const long long row = (start + blockIdx.x + (D == DROP ? 1 : 0)) % TR;
  const uint4* src = reinterpret_cast<const uint4*>(RING + row * stride + col);
  uint4* dst = reinterpret_cast<uint4*>(OUT + (long long)blockIdx.x * width);
  for (int i = threadIdx.x; i < width / 8; i += blockDim.x) dst[i] = src[i];
}

// OUT row j = rms(X row) * W, out bf16 (kind 0) or fp16 (kind 1); block > 0 drops each block's first row. Grid (rows).
extern "C" __global__ void __launch_bounds__(256) tf_rtp2d_rms(const float* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                              void* __restrict__ OUT, int H, float eps, int kind, int block) {
  __shared__ float part[8];
  const int j = blockIdx.x;
  const int src = block > 0 ? j / (block - 1) * block + 1 + j % (block - 1) : j;
  const float* x = X + (long long)src * H;
  float s = 0.0f;
  for (int i = threadIdx.x; i < H; i += 256) s += x[i] * x[i];
  s = warp_sum(s);
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int k = 0; k < 8; ++k) total += part[k];
  const float inv = rsqrtf(total / (float)H + eps);
  for (int i = threadIdx.x; i < H; i += 256) {
    const float v = rbf(x[i] * inv) * bf(W[i]);
    if (kind == 0) reinterpret_cast<__nv_bfloat16*>(OUT)[(long long)j * H + i] = __float2bfloat16_rn(v);
    else reinterpret_cast<__half*>(OUT)[(long long)j * H + i] = __float2half_rn(v);
  }
}

// Control: tf_rtp2d_rms's copy with x / rms left unrounded before the weight.
extern "C" __global__ void __launch_bounds__(256) tf_rtp2d_ctl_rms(const float* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                              void* __restrict__ OUT, int H, float eps, int kind, int block) {
  __shared__ float part[8];
  const int j = blockIdx.x;
  const int src = block > 0 ? j / (block - 1) * block + 1 + j % (block - 1) : j;
  const float* x = X + (long long)src * H;
  float s = 0.0f;
  for (int i = threadIdx.x; i < H; i += 256) s += x[i] * x[i];
  s = warp_sum(s);
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int k = 0; k < 8; ++k) total += part[k];
  const float inv = rsqrtf(total / (float)H + eps);
  for (int i = threadIdx.x; i < H; i += 256) {
    const float v = x[i] * inv * bf(W[i]);
    if (kind == 0) reinterpret_cast<__nv_bfloat16*>(OUT)[(long long)j * H + i] = __float2bfloat16_rn(v);
    else reinterpret_cast<__half*>(OUT)[(long long)j * H + i] = __float2half_rn(v);
  }
}

// y[t, c] = sum over lags l of (BASE[b, l, c] + DYN[t, b * K + l, c / group]) * x[t - l, c], within t's block.
template <typename TX, typename TO>
__device__ __forceinline__ void conv(const TX* X, const float* DYN, const __nv_bfloat16* BASE, const float* RES, TO* OUT, int D,
                                     int group, int K, int branch, int block_rows) {
  const int t = blockIdx.x;
  const int c = blockIdx.y * blockDim.x + threadIdx.x;
  if (c >= D) return;
  const int groups = D / group, g = c / group, local = t % block_rows;
  float total = 0.0f;
  for (int l = 0; l < K && l <= local; ++l) {
    const float x = (float)X[(size_t)(t - l) * D + c];
    const float wgt = bf(BASE[(size_t)(branch * K + l) * D + c]) + DYN[(size_t)t * (2 * K * groups) + (branch * K + l) * groups + g];
    total += wgt * x;
  }
  if (RES != nullptr) total += RES[(size_t)t * D + c];
  OUT[(size_t)t * D + c] = (TO)total;
}

// The convolution's first branch over a normed block (bf16 in and out). Grid (rows, D / 256), 256 threads.
extern "C" __global__ void tf_rtp2d_conv_in(const __nv_bfloat16* __restrict__ X, const float* __restrict__ DYN,
                                           const __nv_bfloat16* __restrict__ BASE, __nv_bfloat16* __restrict__ OUT, int D, int group,
                                           int K, int block_rows) {
  const int t = blockIdx.x;
  const int c = blockIdx.y * blockDim.x + threadIdx.x;
  if (c >= D) return;
  const int groups = D / group, g = c / group, local = t % block_rows;
  float total = 0.0f;
  for (int l = 0; l < K && l <= local; ++l) {
    const float wgt = bf(BASE[(size_t)l * D + c]) + DYN[(size_t)t * (2 * K * groups) + l * groups + g];
    total += wgt * bf(X[(size_t)(t - l) * D + c]);
  }
  OUT[(size_t)t * D + c] = __float2bfloat16_rn(total);
}

// The second branch over a sublayer's fp32 output, plus the residual (fp32 out). Grid (rows, D / 256), 256 threads.
extern "C" __global__ void tf_rtp2d_conv_out(const float* __restrict__ X, const float* __restrict__ DYN,
                                            const __nv_bfloat16* __restrict__ BASE, const float* __restrict__ RES,
                                            float* __restrict__ OUT, int D, int group, int K, int block_rows) {
  conv<float, float>(X, DYN, BASE, RES, OUT, D, group, K, 1, block_rows);
}

// One head of a row: RMSNorm over 128 (weight W), then rotate-half RoPE at `pos` unless rope == 0; lane holds 4 values.
template <int D>
__device__ __forceinline__ void head_norm(const float* x, const __nv_bfloat16* W, __nv_bfloat16* out, long long pos, float eps,
                                          float theta, int rope) {
  const int t = threadIdx.x;
  float v[4], acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    v[i] = x[4 * t + i];
    acc += v[i] * v[i];
  }
  acc = warp_sum(acc);
  const float inv = rsqrtf(acc / (float)HD + eps);
  float n[4];
#pragma unroll
  for (int i = 0; i < 4; i++) n[i] = rbf(rbf(v[i] * inv) * bf(W[4 * t + i]));
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int d = 4 * t + i;
    const float partner = __shfl_xor_sync(0xffffffffu, n[i], 16);  // element d +- 64 sits 16 lanes away
    if (!rope) {
      out[d] = __float2bfloat16_rn(n[i]);
      continue;
    }
    const int j = d < HD / 2 ? d : d - HD / 2;
    const float angle = (float)pos * powf(theta, -(float)j / (float)(HD / 2));
    float sn, cs;
    sincosf(angle, &sn, &cs);
    const bool low = D == DROP ? d >= HD / 2 : d < HD / 2;
    out[d] = __float2bfloat16_rn(low ? n[i] * cs - partner * sn : n[i] * cs + partner * sn);
  }
}

// Block rows' heads normed (roped at the stream's position + row) into OUT [rows, heads, 128]. Grid (rows * heads).
template <int D>
__device__ __forceinline__ void heads_rows(const float* __restrict__ X, int stride, const __nv_bfloat16* __restrict__ W,
                                           __nv_bfloat16* __restrict__ OUT, int heads, const Seg* __restrict__ SEGS, int block_rows,
                                           float eps, float theta, int rope) {
  const int head = blockIdx.x % heads;
  const int m = blockIdx.x / heads;
  const long long pos = SEGS[m / block_rows].position + m % block_rows;
  head_norm<D>(X + (size_t)m * stride + head * HD, W, OUT + ((size_t)m * heads + head) * HD, pos, eps, theta, rope);
}

// Context rows at pos0 + r: keys normed (and roped), values, into a layer's rings at pos % RR. Grid (rows * kv).
template <int D>
__device__ __forceinline__ void kv_rows(const float* __restrict__ K, const float* __restrict__ V, const __nv_bfloat16* __restrict__ W,
                                        __nv_bfloat16* __restrict__ RK, __nv_bfloat16* __restrict__ RV, int kv, long long pos0, int RR,
                                        float eps, float theta, int rope) {
  const int head = blockIdx.x % kv;
  const int r = blockIdx.x / kv;
  const long long pos = pos0 + r;
  const size_t slot = (size_t)(pos % RR) * kv * HD + head * HD;
  head_norm<NONE>(K + ((size_t)r * kv + head) * HD, W, RK + slot, pos, eps, theta, rope);
  const float* v = V + ((size_t)r * kv + head) * HD;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float x = v[4 * threadIdx.x + i];
    RV[slot + 4 * threadIdx.x + i] = D == DROP ? __float2bfloat16_rz(x) : __float2bfloat16_rn(x);
  }
}

// OUT bf16 [n] = X fp32 [n].
template <int D>
__device__ __forceinline__ void f2b_rows(const float* __restrict__ X, __nv_bfloat16* __restrict__ OUT, long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) OUT[i] = D == DROP ? __float2bfloat16_rz(X[i]) : __float2bfloat16_rn(X[i]);
}

// OUT bf16 [n] = silu(G) * U (fp32 in).
template <int D>
__device__ __forceinline__ void silu_rows(const float* __restrict__ G, const float* __restrict__ U, __nv_bfloat16* __restrict__ OUT,
                                          long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float g = G[i];
  if (D == DROP) OUT[i] = __float2bfloat16_rn(rbf(g / (1.0f + __expf(-g))) * U[i]);
  else OUT[i] = __float2bfloat16_rn(g / (1.0f + __expf(-g)) * U[i]);
}

// OUT fp32 [rows, H] = the bf16 embedding rows of IDS (unscaled: DFlash reads the target's table as it is stored).
template <int D>
__device__ __forceinline__ void embed_rows(const int* __restrict__ IDS, const __nv_bfloat16* __restrict__ T, float* __restrict__ OUT,
                                           int H) {
  const __nv_bfloat16* e = T + (long long)IDS[blockIdx.x] * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x)
    OUT[(long long)blockIdx.x * H + i] = D == DROP ? __half2float(reinterpret_cast<const __half*>(e)[i]) : bf(e[i]);
}

// 256 keys of block row m, head h: its stream's ring [first, position), then its block's rows (no mask within).
extern "C" __global__ void __launch_bounds__(128) tf_rtp2d_attn(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KB,
                                                               const __nv_bfloat16* __restrict__ VB, const Seg* __restrict__ SEGS,
                                                               int block_rows, int heads, int kv_heads, int RR, int window, int layer,
                                                               float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
  constexpr int DPL = HD / 32, WARPS = 4, SLICE = CHUNK / WARPS, PER = SLICE / 32;
  __shared__ float qs[HD];
  __shared__ float ps[CHUNK];
  __shared__ float wm[WARPS], wl[WARPS];
  __shared__ float wo[WARPS][HD];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int m = blockIdx.x, h = blockIdx.y, chunk = blockIdx.z, chunks = gridDim.z;
  const int stream = m / block_rows;
  const Seg sg = SEGS[stream];
  const int kvh = h / (heads / kv_heads), width = kv_heads * HD;
  const __nv_bfloat16* RK = reinterpret_cast<const __nv_bfloat16*>(sg.ring) + (size_t)layer * 2 * RR * width;
  const __nv_bfloat16* RV = RK + (size_t)RR * width;
  long long first = sg.position + (m % block_rows) - window + 1;
  if (first < sg.begin) first = sg.begin;
  if (first > sg.position) first = sg.position;
  const int ring_keys = (int)(sg.position - first), keys = ring_keys + block_rows;
  const size_t slot = ((size_t)m * heads + h) * chunks + chunk;
  if (chunk * CHUNK >= keys) {
    if (threadIdx.x == 0) { PM[slot] = -INFINITY; PL[slot] = 0.0f; }
    return;
  }
  const int start = chunk * CHUNK + warp * SLICE, n = max(min(keys - start, SLICE), 0);
  auto row_at = [&](int e) -> size_t {
    const int index = start + e;
    const size_t row = index < ring_keys ? (size_t)((first + index) % RR) : (size_t)(stream * block_rows + index - ring_keys);
    return (row * kv_heads + kvh) * HD;
  };
  for (int i = threadIdx.x; i < HD; i += 32 * WARPS) qs[i] = bf(Q[((size_t)m * heads + h) * HD + i]);
  __syncthreads();
  const float scale = rsqrtf((float)HD);
  float mx = -INFINITY, score[PER];
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int e = lane + 32 * j;
    score[j] = -INFINITY;
    if (e >= n) continue;
    const uint4* k = reinterpret_cast<const uint4*>((start + e < ring_keys ? RK : KB) + row_at(e));
    float dot = 0.0f;
#pragma unroll
    for (int c = 0; c < HD / 8; ++c) {
      const uint4 kw = k[c];
      const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&kw);
#pragma unroll
      for (int i = 0; i < 8; ++i) dot = fmaf(qs[8 * c + i], bf(kb[i]), dot);
    }
    score[j] = dot * scale;
    mx = fmaxf(mx, score[j]);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const float p = lane + 32 * j < n ? __expf(score[j] - mx) : 0.0f;
    ps[warp * SLICE + lane + 32 * j] = p;
    sum += p;
  }
  sum = warp_sum(sum);
  __syncwarp();
  float o[DPL];
#pragma unroll
  for (int i = 0; i < DPL; ++i) o[i] = 0.0f;
  for (int e = 0; e < n; ++e) {
    const uint2 vw = *reinterpret_cast<const uint2*>((start + e < ring_keys ? RV : VB) + row_at(e) + lane * DPL);
    const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&vw);
    const float p = ps[warp * SLICE + e];
#pragma unroll
    for (int i = 0; i < DPL; ++i) o[i] = fmaf(p, bf(vb[i]), o[i]);
  }
  if (lane == 0) { wm[warp] = n > 0 ? mx : -INFINITY; wl[warp] = n > 0 ? sum : 0.0f; }
#pragma unroll
  for (int i = 0; i < DPL; ++i) wo[warp][lane * DPL + i] = o[i];
  __syncthreads();
  if (warp != 0) return;
  float top = -INFINITY;
#pragma unroll
  for (int w = 0; w < WARPS; ++w) if (wl[w] > 0.0f) top = fmaxf(top, wm[w]);
  float total = 0.0f, out[DPL];
#pragma unroll
  for (int i = 0; i < DPL; ++i) out[i] = 0.0f;
#pragma unroll
  for (int w = 0; w < WARPS; ++w) {
    if (wl[w] == 0.0f) continue;
    const float sc = __expf(wm[w] - top);
    total = fmaf(wl[w], sc, total);
#pragma unroll
    for (int i = 0; i < DPL; ++i) out[i] = fmaf(wo[w][lane * DPL + i], sc, out[i]);
  }
  if (lane == 0) { PM[slot] = total > 0.0f ? top : -INFINITY; PL[slot] = total; }
#pragma unroll
  for (int i = 0; i < DPL; ++i) PO[slot * HD + lane * DPL + i] = out[i];
}

// Control: tf_rtp2d_attn's copy with each ring key read from the ring's next row.
extern "C" __global__ void __launch_bounds__(128) tf_rtp2d_ctl_attn(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KB,
                                                               const __nv_bfloat16* __restrict__ VB, const Seg* __restrict__ SEGS,
                                                               int block_rows, int heads, int kv_heads, int RR, int window, int layer,
                                                               float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
  constexpr int DPL = HD / 32, WARPS = 4, SLICE = CHUNK / WARPS, PER = SLICE / 32;
  __shared__ float qs[HD];
  __shared__ float ps[CHUNK];
  __shared__ float wm[WARPS], wl[WARPS];
  __shared__ float wo[WARPS][HD];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int m = blockIdx.x, h = blockIdx.y, chunk = blockIdx.z, chunks = gridDim.z;
  const int stream = m / block_rows;
  const Seg sg = SEGS[stream];
  const int kvh = h / (heads / kv_heads), width = kv_heads * HD;
  const __nv_bfloat16* RK = reinterpret_cast<const __nv_bfloat16*>(sg.ring) + (size_t)layer * 2 * RR * width;
  const __nv_bfloat16* RV = RK + (size_t)RR * width;
  long long first = sg.position + (m % block_rows) - window + 1;
  if (first < sg.begin) first = sg.begin;
  if (first > sg.position) first = sg.position;
  const int ring_keys = (int)(sg.position - first), keys = ring_keys + block_rows;
  const size_t slot = ((size_t)m * heads + h) * chunks + chunk;
  if (chunk * CHUNK >= keys) {
    if (threadIdx.x == 0) { PM[slot] = -INFINITY; PL[slot] = 0.0f; }
    return;
  }
  const int start = chunk * CHUNK + warp * SLICE, n = max(min(keys - start, SLICE), 0);
  auto row_at = [&](int e) -> size_t {
    const int index = start + e;
    const size_t row = index < ring_keys ? (size_t)((first + index + 1) % RR) : (size_t)(stream * block_rows + index - ring_keys);
    return (row * kv_heads + kvh) * HD;
  };
  for (int i = threadIdx.x; i < HD; i += 32 * WARPS) qs[i] = bf(Q[((size_t)m * heads + h) * HD + i]);
  __syncthreads();
  const float scale = rsqrtf((float)HD);
  float mx = -INFINITY, score[PER];
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int e = lane + 32 * j;
    score[j] = -INFINITY;
    if (e >= n) continue;
    const uint4* k = reinterpret_cast<const uint4*>((start + e < ring_keys ? RK : KB) + row_at(e));
    float dot = 0.0f;
#pragma unroll
    for (int c = 0; c < HD / 8; ++c) {
      const uint4 kw = k[c];
      const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&kw);
#pragma unroll
      for (int i = 0; i < 8; ++i) dot = fmaf(qs[8 * c + i], bf(kb[i]), dot);
    }
    score[j] = dot * scale;
    mx = fmaxf(mx, score[j]);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const float p = lane + 32 * j < n ? __expf(score[j] - mx) : 0.0f;
    ps[warp * SLICE + lane + 32 * j] = p;
    sum += p;
  }
  sum = warp_sum(sum);
  __syncwarp();
  float o[DPL];
#pragma unroll
  for (int i = 0; i < DPL; ++i) o[i] = 0.0f;
  for (int e = 0; e < n; ++e) {
    const uint2 vw = *reinterpret_cast<const uint2*>((start + e < ring_keys ? RV : VB) + row_at(e) + lane * DPL);
    const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&vw);
    const float p = ps[warp * SLICE + e];
#pragma unroll
    for (int i = 0; i < DPL; ++i) o[i] = fmaf(p, bf(vb[i]), o[i]);
  }
  if (lane == 0) { wm[warp] = n > 0 ? mx : -INFINITY; wl[warp] = n > 0 ? sum : 0.0f; }
#pragma unroll
  for (int i = 0; i < DPL; ++i) wo[warp][lane * DPL + i] = o[i];
  __syncthreads();
  if (warp != 0) return;
  float top = -INFINITY;
#pragma unroll
  for (int w = 0; w < WARPS; ++w) if (wl[w] > 0.0f) top = fmaxf(top, wm[w]);
  float total = 0.0f, out[DPL];
#pragma unroll
  for (int i = 0; i < DPL; ++i) out[i] = 0.0f;
#pragma unroll
  for (int w = 0; w < WARPS; ++w) {
    if (wl[w] == 0.0f) continue;
    const float sc = __expf(wm[w] - top);
    total = fmaf(wl[w], sc, total);
#pragma unroll
    for (int i = 0; i < DPL; ++i) out[i] = fmaf(wo[w][lane * DPL + i], sc, out[i]);
  }
  if (lane == 0) { PM[slot] = total > 0.0f ? top : -INFINITY; PL[slot] = total; }
#pragma unroll
  for (int i = 0; i < DPL; ++i) PO[slot * HD + lane * DPL + i] = out[i];
}

// The partials of each (row, head) merged in chunk order into OUT bf16 [rows, heads, 128]. Grid (rows, heads).
extern "C" __global__ void __launch_bounds__(32) tf_rtp2d_merge(const float* __restrict__ PM, const float* __restrict__ PL,
                                                               const float* __restrict__ PO, int heads, int chunks,
                                                               __nv_bfloat16* __restrict__ OUT) {
  constexpr int DPL = HD / 32;
  const int lane = threadIdx.x, m = blockIdx.x, h = blockIdx.y;
  const size_t base = ((size_t)m * heads + h) * chunks;
  float mx = -INFINITY, sum = 0.0f, out[DPL];
#pragma unroll
  for (int i = 0; i < DPL; ++i) out[i] = 0.0f;
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const size_t slot = base + chunk;
    if (PL[slot] == 0.0f) continue;
    const float next = fmaxf(mx, PM[slot]), a = __expf(mx - next), b = __expf(PM[slot] - next);
    sum = sum * a + PL[slot] * b;
#pragma unroll
    for (int i = 0; i < DPL; ++i) out[i] = fmaf(b, PO[slot * HD + lane * DPL + i], out[i] * a);
    mx = next;
  }
#pragma unroll
  for (int i = 0; i < DPL; ++i) OUT[((size_t)m * heads + h) * HD + lane * DPL + i] = __float2bfloat16_rn(out[i] / sum);
}

// Control: tf_rtp2d_merge's copy with the running sum never rescaled.
extern "C" __global__ void __launch_bounds__(32) tf_rtp2d_ctl_merge(const float* __restrict__ PM, const float* __restrict__ PL,
                                                               const float* __restrict__ PO, int heads, int chunks,
                                                               __nv_bfloat16* __restrict__ OUT) {
  constexpr int DPL = HD / 32;
  const int lane = threadIdx.x, m = blockIdx.x, h = blockIdx.y;
  const size_t base = ((size_t)m * heads + h) * chunks;
  float mx = -INFINITY, sum = 0.0f, out[DPL];
#pragma unroll
  for (int i = 0; i < DPL; ++i) out[i] = 0.0f;
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const size_t slot = base + chunk;
    if (PL[slot] == 0.0f) continue;
    const float next = fmaxf(mx, PM[slot]), a = __expf(mx - next), b = __expf(PM[slot] - next);
    sum = sum + PL[slot] * b;
#pragma unroll
    for (int i = 0; i < DPL; ++i) out[i] = fmaf(b, PO[slot * HD + lane * DPL + i], out[i] * a);
    mx = next;
  }
#pragma unroll
  for (int i = 0; i < DPL; ++i) OUT[((size_t)m * heads + h) * HD + lane * DPL + i] = __float2bfloat16_rn(out[i] / sum);
}

// A row's slice of logits: its K best (value, id; ties to the lower id) into PV / PI [rows, slices, K].
extern "C" __global__ void __launch_bounds__(256) tf_rtp2d_topk(const float* __restrict__ L, int V, int slice, int K,
                                                               float* __restrict__ PV, int* __restrict__ PI) {
  extern __shared__ float sv[];
  __shared__ float bv[8];
  __shared__ int bi[8];
  const int row = blockIdx.y, s0 = blockIdx.x * slice;
  const float* l = L + (long long)row * V;
  for (int i = threadIdx.x; i < slice; i += 256) {
    const int c = s0 + i;
    const float v = c < V ? l[c] : -INFINITY;
    sv[i] = v == v ? v : -INFINITY;
  }
  __syncthreads();
  float* pv = PV + ((long long)row * gridDim.x + blockIdx.x) * K;
  int* pi = PI + ((long long)row * gridDim.x + blockIdx.x) * K;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  for (int k = 0; k < K; ++k) {
    float best = -INFINITY;
    int at = slice;
    for (int i = threadIdx.x; i < slice; i += 256) {
      const float v = sv[i];
      if (v > best) best = v, at = i;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, at, o);
      if (ov > best || (ov == best && oi < at)) best = ov, at = oi;
    }
    if (lane == 0) bv[w] = best, bi[w] = at;
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int j = 1; j < 8; ++j) if (bv[j] > best || (bv[j] == best && bi[j] < at)) best = bv[j], at = bi[j];
      pv[k] = best;
      pi[k] = at < slice ? s0 + at : -1;
      if (at < slice) sv[at] = -INFINITY;
    }
    __syncthreads();
  }
}

// Control: tf_rtp2d_topk's copy with lane ties to the higher id.
extern "C" __global__ void __launch_bounds__(256) tf_rtp2d_ctl_topk(const float* __restrict__ L, int V, int slice, int K,
                                                               float* __restrict__ PV, int* __restrict__ PI) {
  extern __shared__ float sv[];
  __shared__ float bv[8];
  __shared__ int bi[8];
  const int row = blockIdx.y, s0 = blockIdx.x * slice;
  const float* l = L + (long long)row * V;
  for (int i = threadIdx.x; i < slice; i += 256) {
    const int c = s0 + i;
    const float v = c < V ? l[c] : -INFINITY;
    sv[i] = v == v ? v : -INFINITY;
  }
  __syncthreads();
  float* pv = PV + ((long long)row * gridDim.x + blockIdx.x) * K;
  int* pi = PI + ((long long)row * gridDim.x + blockIdx.x) * K;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  for (int k = 0; k < K; ++k) {
    float best = -INFINITY;
    int at = slice;
    for (int i = threadIdx.x; i < slice; i += 256) {
      const float v = sv[i];
      if (v > best) best = v, at = i;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, at, o);
      if (ov > best || (ov == best && oi > at)) best = ov, at = oi;
    }
    if (lane == 0) bv[w] = best, bi[w] = at;
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int j = 1; j < 8; ++j) if (bv[j] > best || (bv[j] == best && bi[j] < at)) best = bv[j], at = bi[j];
      pv[k] = best;
      pi[k] = at < slice ? s0 + at : -1;
      if (at < slice) sv[at] = -INFINITY;
    }
    __syncthreads();
  }
}

// The served kernels (NONE) and the contract's planted controls (DROP, tf_rtp2d_ctl_*): each name a literal token.
#define TF_RTP2D(BOUNDS, NAME, CTL, BODY, PARAMS, ARGS)                          \
  extern "C" __global__ void BOUNDS NAME PARAMS { BODY<NONE> ARGS; } \
  extern "C" __global__ void BOUNDS CTL PARAMS { BODY<DROP> ARGS; }

TF_RTP2D(, tf_rtp2d_gather, tf_rtp2d_ctl_gather, gather_rows,
         (const __nv_bfloat16* __restrict__ RING, int TR, long long start, int stride, int col, int width, __nv_bfloat16* __restrict__ OUT),
         (RING, TR, start, stride, col, width, OUT))
TF_RTP2D(__launch_bounds__(32), tf_rtp2d_heads, tf_rtp2d_ctl_heads, heads_rows,
         (const float* __restrict__ X, int stride, const __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ OUT, int heads,
          const Seg* __restrict__ SEGS, int block_rows, float eps, float theta, int rope),
         (X, stride, W, OUT, heads, SEGS, block_rows, eps, theta, rope))
TF_RTP2D(__launch_bounds__(32), tf_rtp2d_kv_put, tf_rtp2d_ctl_kv_put, kv_rows,
         (const float* __restrict__ K, const float* __restrict__ V, const __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ RK,
          __nv_bfloat16* __restrict__ RV, int kv, long long pos0, int RR, float eps, float theta, int rope),
         (K, V, W, RK, RV, kv, pos0, RR, eps, theta, rope))
TF_RTP2D(, tf_rtp2d_f2b, tf_rtp2d_ctl_f2b, f2b_rows, (const float* __restrict__ X, __nv_bfloat16* __restrict__ OUT, long long n), (X, OUT, n))
TF_RTP2D(, tf_rtp2d_silu, tf_rtp2d_ctl_silu, silu_rows,
         (const float* __restrict__ G, const float* __restrict__ U, __nv_bfloat16* __restrict__ OUT, long long n), (G, U, OUT, n))
TF_RTP2D(, tf_rtp2d_embed, tf_rtp2d_ctl_embed, embed_rows,
         (const int* __restrict__ IDS, const __nv_bfloat16* __restrict__ T, float* __restrict__ OUT, int H), (IDS, T, OUT, H))
#undef TF_RTP2D

// The convolutions' controls: conv_in on the second branch's weights, conv_out on the first's.
extern "C" __global__ void tf_rtp2d_ctl_conv_in(const __nv_bfloat16* __restrict__ X, const float* __restrict__ DYN,
                                               const __nv_bfloat16* __restrict__ BASE, __nv_bfloat16* __restrict__ OUT, int D, int group,
                                               int K, int block_rows) {
  const int t = blockIdx.x;
  const int c = blockIdx.y * blockDim.x + threadIdx.x;
  if (c >= D) return;
  const int groups = D / group, g = c / group, local = t % block_rows;
  float total = 0.0f;
  for (int l = 0; l < K && l <= local; ++l) {
    const float wgt = bf(BASE[(size_t)(K + l) * D + c]) + DYN[(size_t)t * (2 * K * groups) + (K + l) * groups + g];
    total += wgt * bf(X[(size_t)(t - l) * D + c]);
  }
  OUT[(size_t)t * D + c] = __float2bfloat16_rn(total);
}
extern "C" __global__ void tf_rtp2d_ctl_conv_out(const float* __restrict__ X, const float* __restrict__ DYN,
                                                const __nv_bfloat16* __restrict__ BASE, const float* __restrict__ RES,
                                                float* __restrict__ OUT, int D, int group, int K, int block_rows) {
  conv<float, float>(X, DYN, BASE, RES, OUT, D, group, K, 0, block_rows);
}
