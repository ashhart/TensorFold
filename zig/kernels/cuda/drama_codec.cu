// Drama 3 codec decode (modded DAC quantizer decode and decoder) in sglang-omni's FP16 arithmetic, FP32 reductions.

#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float h2f(__half v) { return __half2float(v); }
__device__ __forceinline__ __half f2h(float v) { return __float2half_rn(v); }
// The FP32 value an FP16 tensor would hold.
__device__ __forceinline__ float rh(float v) { return __half2float(__float2half_rn(v)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets the total.
__device__ float block_sum(float v, float* part) {
  v = warp_sum(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31, n = blockDim.x >> 5;
  __syncthreads();
  if (l == 0) part[w] = v;
  __syncthreads();
  return warp_sum(l < n ? part[l] : 0.f);
}

__device__ float block_max(float v, float* part) {
  v = warp_max(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31, n = blockDim.x >> 5;
  __syncthreads();
  if (l == 0) part[w] = v;
  __syncthreads();
  return warp_max(l < n ? part[l] : -__int_as_float(0x7f800000));
}

constexpr int kLatent = 1024;     // DownsampleResidualVectorQuantize input_dim
constexpr int kCodebookDim = 8;   // codebook_dim
constexpr int kBooks = 10;        // 1 semantic + 9 residual
constexpr int kSemanticSize = 4096;
constexpr int kResidualSize = 1024;
constexpr int kHeadDim = 64;

}  // namespace

// rvq.py decode before post_module: clamped codes, each book's row through out_proj, residual books summed from 0.
extern "C" __global__ void __launch_bounds__(256) drama_codec_rvq_latent(
    const uint32_t* __restrict__ codes, const __half* __restrict__ semantic_book,
    const __half* __restrict__ residual_books, const __half* __restrict__ out_w,
    const __half* __restrict__ out_b, __half* __restrict__ z) {
  const size_t t = blockIdx.x;
  __shared__ float e[kBooks][kCodebookDim];
  if (threadIdx.x < kBooks * kCodebookDim) {
    const int book = threadIdx.x / kCodebookDim, i = threadIdx.x % kCodebookDim;
    uint32_t code = codes[t * kBooks + book];
    const __half* row;
    if (book == 0) {
      code = min(code, (uint32_t)(kSemanticSize - 1));
      row = semantic_book + (size_t)code * kCodebookDim;
    } else {
      code = min(code, (uint32_t)(kResidualSize - 1));
      row = residual_books + ((size_t)(book - 1) * kResidualSize + code) * kCodebookDim;
    }
    e[book][i] = h2f(row[i]);
  }
  __syncthreads();
  for (int c = threadIdx.x; c < kLatent; c += blockDim.x) {
    float semantic = 0.f, residual = 0.f;
    for (int book = 0; book < kBooks; ++book) {
      const __half* w = out_w + ((size_t)book * kLatent + c) * kCodebookDim;
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < kCodebookDim; ++i) acc = fmaf(e[book][i], h2f(w[i]), acc);
      const float q = rh(rh(acc) + h2f(out_b[book * kLatent + c]));
      if (book == 0) semantic = q;
      else if (book == 1) residual = 0.f + q;
      else residual = rh(residual + q);
    }
    z[t * kLatent + c] = f2h(semantic + residual);
  }
}

// modded_dac.py RMSNorm: h(h(x * rsqrt(mean(x * x) + eps)) * w), statistics in FP32.
extern "C" __global__ void __launch_bounds__(256) drama_codec_rms_norm(
    const __half* __restrict__ x, const __half* __restrict__ w, __half* __restrict__ y, int width,
    float eps) {
  __shared__ float part[32];
  const __half* xr = x + (size_t)blockIdx.x * width;
  __half* yr = y + (size_t)blockIdx.x * width;
  float s = 0.f;
  for (int c = threadIdx.x; c < width; c += blockDim.x) {
    const float v = h2f(xr[c]);
    s += __fmul_rn(v, v);
  }
  s = block_sum(s, part);
  const float r = rsqrtf(__fadd_rn(s / (float)width, eps));
  for (int c = threadIdx.x; c < width; c += blockDim.x)
    yr[c] = f2h(__fmul_rn(rh(__fmul_rn(h2f(xr[c]), r)), h2f(w[c])));
}

// apply_rotary_emb on q and k of a fused qkv row, pairs (2i, 2i + 1) in FP32 as eager PyTorch evaluates them.
extern "C" __global__ void __launch_bounds__(256) drama_codec_rope(
    __half* __restrict__ qkv, const __half* __restrict__ table, int width, int ld) {
  __half* row = qkv + (size_t)blockIdx.x * ld;
  const __half* tab = table + (size_t)blockIdx.x * kHeadDim;
  const int pairs = width / 2;
  for (int p = threadIdx.x; p < 2 * pairs; p += blockDim.x) {
    const int which = p / pairs, pp = p % pairs, i = pp % (kHeadDim / 2);
    __half* e = row + which * width + pp * 2;
    const float x0 = h2f(e[0]), x1 = h2f(e[1]);
    const float c = h2f(tab[2 * i]), s = h2f(tab[2 * i + 1]);
    e[0] = f2h(__fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s)));
    e[1] = f2h(__fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s)));
  }
}

// Causal window attention as Torch's efficient path: 64-key tiles, exp2 softmax, FP16 P for P*V, FP32 sums.
extern "C" __global__ void __launch_bounds__(128) drama_codec_window_attention(
    const __half* __restrict__ qkv, __half* __restrict__ out, int heads, int window, float scale) {
  constexpr int tile = 64;
  __shared__ float q[kHeadDim], p[tile], part[32];
  const int t = blockIdx.x, h = blockIdx.y, j = threadIdx.x;
  const size_t ld = (size_t)3 * heads * kHeadDim;
  if (j < kHeadDim) q[j] = h2f(qkv[(size_t)t * ld + h * kHeadDim + j]);
  __syncthreads();
  const int first = max(0, t - window + 1);
  float maximum = -__int_as_float(0x7f800000), denominator = 0.f, numerator = 0.f;
  for (int start = first / tile * tile; start <= t; start += tile) {
    const int key = start + j;
    const bool valid = j < tile && key >= first && key <= t;
    float score = -__int_as_float(0x7f800000);
    if (valid) {
      const __half* k = qkv + (size_t)key * ld + (size_t)heads * kHeadDim + h * kHeadDim;
      float dot = 0.f;
#pragma unroll 16
      for (int d = 0; d < kHeadDim; ++d) dot = fmaf(q[d], h2f(k[d]), dot);
      score = (dot * scale) * 1.4426950408889634074f;
    }
    const float next_maximum = fmaxf(maximum, block_max(score, part));
    const float rescale = exp2f(maximum - next_maximum);
    const float probability = valid ? exp2f(score - next_maximum) : 0.f;
    if (j < tile) p[j] = rh(probability);
    const float mass = block_sum(probability, part);  // also publishes p[]
    denominator = denominator * rescale + mass;
    if (j < kHeadDim) {
      numerator *= rescale;
      for (int i = 0; i < tile && start + i <= t; ++i) {
        if (start + i < first) continue;
        const __half v = qkv[(size_t)(start + i) * ld + (size_t)2 * heads * kHeadDim + h * kHeadDim + j];
        numerator = fmaf(p[i], h2f(v), numerator);
      }
    }
    maximum = next_maximum;
    __syncthreads();
  }
  if (j < kHeadDim)
    out[(size_t)t * heads * kHeadDim + h * kHeadDim + j] = f2h(numerator / denominator);
}

// FeedForward's silu(w1(x)) * w3(x) over a fused (w1 | w3) GEMM output, each step rounded.
extern "C" __global__ void __launch_bounds__(256) drama_codec_silu_mul(
    const __half* __restrict__ gu, __half* __restrict__ out, long long rows, int width) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * width) return;
  const long long r = i / width, c = i % width;
  const float a = h2f(gu[r * 2 * width + c]);
  const float b = h2f(gu[r * 2 * width + width + c]);
  out[i] = f2h(rh(a / (1.0f + expf(-a))) * b);
}

// nn.LayerNorm over channels (eps 1e-6) as PyTorch's kernel: FP32 mean and biased variance.
extern "C" __global__ void __launch_bounds__(256) drama_codec_layer_norm(
    const __half* __restrict__ x, const __half* __restrict__ w, const __half* __restrict__ b,
    __half* __restrict__ y, int width, float eps) {
  __shared__ float part[32];
  const __half* xr = x + (size_t)blockIdx.x * width;
  __half* yr = y + (size_t)blockIdx.x * width;
  float s = 0.f;
  for (int c = threadIdx.x; c < width; c += blockDim.x) s += h2f(xr[c]);
  const float mean = block_sum(s, part) / (float)width;
  float v = 0.f;
  for (int c = threadIdx.x; c < width; c += blockDim.x) {
    const float d = h2f(xr[c]) - mean;
    v = fmaf(d, d, v);
  }
  const float rstd = rsqrtf(block_sum(v, part) / (float)width + eps);
  for (int c = threadIdx.x; c < width; c += blockDim.x)
    yr[c] = f2h(fmaf(h2f(w[c]), rstd * (h2f(xr[c]) - mean), h2f(b[c])));
}

// ConvNeXtBlock.dwconv: causal depthwise conv, kernel 7, cuDNN then bias add_.
extern "C" __global__ void __launch_bounds__(256) drama_codec_dwconv7(
    const __half* __restrict__ x, const __half* __restrict__ w, const __half* __restrict__ b,
    __half* __restrict__ y, long long rows, int width) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * width) return;
  const long long t = i / width;
  const int c = (int)(i % width);
  float acc = 0.f;
#pragma unroll
  for (int k = 0; k < 7; ++k) {
    const long long r = t - 6 + k;
    if (r >= 0) acc = fmaf(h2f(x[r * width + c]), h2f(w[c * 7 + k]), acc);
  }
  y[i] = f2h(rh(acc) + h2f(b[c]));
}

// dac Snake1d as the Triton kernel: y = x + inv * s * s with s = sin(alpha * x), the final multiply-add fused.
extern "C" __global__ void __launch_bounds__(256) drama_codec_snake(
    const __half* x, const float* __restrict__ alpha, const float* __restrict__ inv_alpha, __half* y,
    long long n, int width) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int c = (int)(i % width);
  const float v = h2f(x[i]);
  const float s = sinf(alpha[c] * v);
  y[i] = f2h(fmaf(inv_alpha[c] * s, s, v));
}

// The last decoder conv (96 -> 1, kernel 7, causal) and tanh, written as FP32 PCM.
extern "C" __global__ void __launch_bounds__(256) drama_codec_conv_out_tanh(
    const __half* __restrict__ x, const __half* __restrict__ w, const __half* __restrict__ b,
    float* __restrict__ y, long long rows, int width) {
  const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows) return;
  float acc = 0.f;
  for (int k = 0; k < 7; ++k) {
    const long long r = t - 6 + k;
    if (r < 0) continue;
    const __half* xr = x + r * width;
    const __half* wk = w + k * width;
    for (int c = 0; c < width; ++c) acc = fmaf(h2f(xr[c]), h2f(wk[c]), acc);
  }
  const float v = rh(rh(acc) + h2f(b[0]));
  y[t] = rh(tanhf(v));
}

// One implicit GEMM for every conv, transposed conv and linear; the epilogue rounds to FP16 in the reference's order.
struct GemmParams {
  const __half* x;
  const __half* w;
  const __half* bias;      // [period] or null
  const __half* scale;     // [period] or null
  const __half* residual;  // [m][ldr] or null (may alias y)
  __half* y;
  long long m;
  int n, cin, taps, dilation, ldx, ldy, ldr, period, bias_mode, act;
};

namespace {
constexpr int BM = 128, BN = 128, BK = 32, LDS = BK + 8;

__device__ __forceinline__ float gemm_epilogue(const GemmParams& p, long long m, int n, float acc) {
  const int bn = n % p.period;
  float v;
  if (p.bias_mode == 1) {
    v = rh(acc + h2f(p.bias[bn]));
  } else {
    v = rh(acc);
    if (p.bias_mode == 2) v = rh(v + h2f(p.bias[bn]));
  }
  if (p.act == 1) v = rh(v * 0.5f * (1.0f + erff(v * 0.70710678118654752440f)));
  if (p.scale != nullptr) v = rh(v * h2f(p.scale[bn]));
  if (p.residual != nullptr) v = rh(h2f(p.residual[m * p.ldr + n]) + v);
  return v;
}
}  // namespace

// 128 x 128 tiles, 8 warps of 64 x 32; cin, ldx and ldy multiples of 8, x and w 16-byte aligned.
extern "C" __global__ void __launch_bounds__(256) drama_codec_conv_gemm(const GemmParams p) {
  using namespace nvcuda;
  __shared__ __align__(128) __half as[BM * LDS];
  __shared__ __align__(128) __half bs[BN * LDS];
  __shared__ __align__(128) float cs[8][16 * 16];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp >> 2, wn = warp & 3;
  const long long m0 = (long long)blockIdx.x * BM;
  const int n0 = blockIdx.y * BN;
  const int ktot = p.taps * p.cin;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  uint4 ra[2], rb[2];
  auto fetch = [&](int k0) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * 256;
      const int r = v >> 2, c = (v & 3) * 8;
      const int kk = k0 + c;
      uint4 a = make_uint4(0u, 0u, 0u, 0u), b = make_uint4(0u, 0u, 0u, 0u);
      const long long m = m0 + r;
      if (m < p.m && kk < ktot) {
        const int tap = kk / p.cin, ci = kk - tap * p.cin;
        const long long rin = m - (long long)(p.taps - 1 - tap) * p.dilation;
        if (rin >= 0) a = *reinterpret_cast<const uint4*>(p.x + rin * p.ldx + ci);
      }
      const int n = n0 + r;
      if (n < p.n && kk < ktot) b = *reinterpret_cast<const uint4*>(p.w + (size_t)n * ktot + kk);
      ra[i] = a;
      rb[i] = b;
    }
  };

  fetch(0);
  for (int k0 = 0; k0 < ktot; k0 += BK) {
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * 256;
      const int r = v >> 2, c = (v & 3) * 8;
      *reinterpret_cast<uint4*>(&as[r * LDS + c]) = ra[i];
      *reinterpret_cast<uint4*>(&bs[r * LDS + c]) = rb[i];
    }
    __syncthreads();
    if (k0 + BK < ktot) fetch(k0 + BK);
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af[4];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf[2];
#pragma unroll
      for (int i = 0; i < 4; ++i) wmma::load_matrix_sync(af[i], as + (wm * 64 + i * 16) * LDS + kk, LDS);
#pragma unroll
      for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(bf[j], bs + (wn * 32 + j * 16) * LDS + kk, LDS);
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
  }

  float* c = cs[warp];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      wmma::store_matrix_sync(c, acc[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      for (int e = lane; e < 256; e += 32) {
        const long long m = m0 + wm * 64 + i * 16 + (e >> 4);
        const int n = n0 + wn * 32 + j * 16 + (e & 15);
        if (m < p.m && n < p.n) p.y[m * p.ldy + n] = f2h(gemm_epilogue(p, m, n, c[e]));
      }
      __syncwarp();
    }
  }
}
