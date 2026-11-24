// MOSS-Image's image decoder (FLUX.2's VAE decoder) in fp32: the metal file's steps per picture tensor.
#include <math.h>
#include <stdint.h>

#include "common/reduce.cuh"

using tf_common::warp_max;
using tf_common::warp_sum;

// The latents x [h * w, 128] with the batch norm undone (x * s + m), each 2 x 2 patch unpacked: z [(2h)(2w), 32].
extern "C" __global__ void __launch_bounds__(256) tf_md_unpack(const float* __restrict__ X, const float* __restrict__ S,
                                                               const float* __restrict__ M, float* __restrict__ Z,
                                                               int rows, int cols) {
  const int w2 = cols * 2;
  const int at = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int c = threadIdx.x & 31;
  if (at >= rows * 2 * w2) return;
  const int py = at / w2, px = at % w2;
  const int k = c * 4 + (py % 2) * 2 + px % 2;
  const float v = X[(size_t)((py / 2) * cols + px / 2) * 128 + k] * S[k];
  Z[(size_t)at * 32 + c] = v + M[k];
}

// y [pixels, cin] . w [cin, cout] + b (+ r): a 64 x 64 tile a block, fp32 sums. B is [k, n] (bt 0) or [n, k] (bt 1).
#define TF_MD_BM 64
#define TF_MD_BN 64
#define TF_MD_BK 16

extern "C" __global__ void __launch_bounds__(256) tf_md_gemm(const float* __restrict__ A, const float* __restrict__ B,
                                                            const float* __restrict__ bias,
                                                            const float* __restrict__ resid, float* __restrict__ C,
                                                            int m, int n, int k, int lda, int ldb, int bt, int add) {
  __shared__ float As[TF_MD_BM][TF_MD_BK + 1];
  __shared__ float Bs[TF_MD_BK][TF_MD_BN + 1];
  const int tid = threadIdx.x;
  const int tx = tid & 15, ty = tid >> 4;
  const int m0 = blockIdx.y * TF_MD_BM, n0 = blockIdx.x * TF_MD_BN;
  // a thread's 4 pixels and 4 channels of the tile
  const int arow = tid >> 2, acol = (tid & 3) * 4;
  const int brow = tid >> 4, bcol = (tid & 15) * 4;
  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
  for (int k0 = 0; k0 < k; k0 += TF_MD_BK) {
    const int gm = m0 + arow, gk = k0 + acol;
#pragma unroll
    for (int i = 0; i < 4; ++i) As[arow][acol + i] = (gm < m && gk + i < k) ? A[(size_t)gm * lda + gk + i] : 0.0f;
    if (bt) {
      const int gn = n0 + arow;
#pragma unroll
      for (int i = 0; i < 4; ++i) Bs[acol + i][arow] = (gn < n && gk + i < k) ? B[(size_t)gn * ldb + gk + i] : 0.0f;
    } else {
      const int gk2 = k0 + brow;
#pragma unroll
      for (int i = 0; i < 4; ++i) Bs[brow][bcol + i] = (gk2 < k && n0 + bcol + i < n) ? B[(size_t)gk2 * ldb + n0 + bcol + i] : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < TF_MD_BK; ++kk) {
      const float a0 = As[tx * 4 + 0][kk], a1 = As[tx * 4 + 1][kk], a2 = As[tx * 4 + 2][kk], a3 = As[tx * 4 + 3][kk];
      const float b0 = Bs[kk][ty * 4 + 0], b1 = Bs[kk][ty * 4 + 1], b2 = Bs[kk][ty * 4 + 2], b3 = Bs[kk][ty * 4 + 3];
      acc[0][0] += a0 * b0; acc[0][1] += a0 * b1; acc[0][2] += a0 * b2; acc[0][3] += a0 * b3;
      acc[1][0] += a1 * b0; acc[1][1] += a1 * b1; acc[1][2] += a1 * b2; acc[1][3] += a1 * b3;
      acc[2][0] += a2 * b0; acc[2][1] += a2 * b1; acc[2][2] += a2 * b2; acc[2][3] += a2 * b3;
      acc[3][0] += a3 * b0; acc[3][1] += a3 * b1; acc[3][2] += a3 * b2; acc[3][3] += a3 * b3;
    }
    __syncthreads();
  }
  const int cm = m0 + tx * 4, cn = n0 + ty * 4;
  float bj[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) bj[j] = (bias != nullptr && cn + j < n) ? bias[cn + j] : 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (cm + i >= m) break;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (cn + j >= n) break;
      const size_t at = (size_t)(cm + i) * n + cn + j;
      float v = acc[i][j] + bj[j];
      if (add) v += resid[at];
      C[at] = v;
    }
  }
}

// y = conv3x3(x) + b (+ r) at rows x cols, u [9 taps][cin][cout]; 8 x 16 output pixels by 32 channels a block.
#define TF_MD_TH 8
#define TF_MD_TW 16
#define TF_MD_TN 32
#define TF_MD_TC 16
#define TF_MD_PR (TF_MD_TH + 2)
#define TF_MD_PC (TF_MD_TW + 2)

extern "C" __global__ void __launch_bounds__(256) tf_md_conv3x3(const float* __restrict__ X, const float* __restrict__ U,
                                                               const float* __restrict__ bias,
                                                               const float* __restrict__ resid, float* __restrict__ Y,
                                                               int rows, int cols, int cin, int cout, int up, int add) {
  __shared__ __align__(16) float patch[TF_MD_PR * TF_MD_PC * TF_MD_TC];
  __shared__ float W[9 * TF_MD_TC * TF_MD_TN];
  const int tid = threadIdx.x;
  const int pg = tid & 31, cg = tid >> 5;
  const int pr = pg >> 2, pc0 = (pg & 3) * 4;
  const int y0 = blockIdx.y * TF_MD_TH, x0 = blockIdx.z * TF_MD_TW, n0 = blockIdx.x * TF_MD_TN;
  const int in_cols = up ? cols / 2 : cols;
  const int oy0 = y0 + pr, ox0 = x0 + pc0;
  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
  for (int c0 = 0; c0 < cin; c0 += TF_MD_TC) {
    for (int e = tid; e < TF_MD_PR * TF_MD_PC * (TF_MD_TC / 4); e += 256) {
      const int px = e / (TF_MD_TC / 4), q = e % (TF_MD_TC / 4);
      const int iy = y0 - 1 + px / TF_MD_PC, ix = x0 - 1 + px % TF_MD_PC;
      float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (iy >= 0 && ix >= 0 && iy < rows && ix < cols) {
        const int sy = up ? iy / 2 : iy, sx = up ? ix / 2 : ix;
        const float* src = X + ((size_t)sy * in_cols + sx) * cin + c0 + q * 4;
        const int left = cin - (c0 + q * 4);
        if (left >= 4) {
          v = *(const float4*)src;
        } else if (left > 0) {
          v.x = src[0];
          if (left > 1) v.y = src[1];
          if (left > 2) v.z = src[2];
        }
      }
      *(float4*)(patch + px * TF_MD_TC + q * 4) = v;
    }
    for (int e = tid; e < 9 * TF_MD_TC * TF_MD_TN; e += 256) {
      const int tap = e / (TF_MD_TC * TF_MD_TN), r = e % (TF_MD_TC * TF_MD_TN);
      const int kk = r / TF_MD_TN, nn = r % TF_MD_TN;
      const int c = c0 + kk, gn = n0 + nn;
      W[e] = (c < cin && gn < cout) ? U[((size_t)tap * cin + c) * cout + gn] : 0.0f;
    }
    __syncthreads();
    const float* base = patch + (pr * TF_MD_PC + pc0) * TF_MD_TC;
#pragma unroll 1
    for (int kk = 0; kk < TF_MD_TC; ++kk) {
#pragma unroll
      for (int tap = 0; tap < 9; ++tap) {
        const float* q = base + ((tap / 3) * TF_MD_PC + tap % 3) * TF_MD_TC + kk;
        const float a0 = q[0], a1 = q[TF_MD_TC], a2 = q[2 * TF_MD_TC], a3 = q[3 * TF_MD_TC];
        const float* w = &W[(tap * TF_MD_TC + kk) * TF_MD_TN + cg * 4];
        const float b0 = w[0], b1 = w[1], b2 = w[2], b3 = w[3];
        acc[0][0] += a0 * b0; acc[0][1] += a0 * b1; acc[0][2] += a0 * b2; acc[0][3] += a0 * b3;
        acc[1][0] += a1 * b0; acc[1][1] += a1 * b1; acc[1][2] += a1 * b2; acc[1][3] += a1 * b3;
        acc[2][0] += a2 * b0; acc[2][1] += a2 * b1; acc[2][2] += a2 * b2; acc[2][3] += a2 * b3;
        acc[3][0] += a3 * b0; acc[3][1] += a3 * b1; acc[3][2] += a3 * b2; acc[3][3] += a3 * b3;
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int oy = oy0, ox = ox0 + i;
    if (oy >= rows || ox >= cols) continue;
    const size_t row = (size_t)oy * cols + ox;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = n0 + cg * 4 + j;
      if (col >= cout) break;
      const size_t at = row * cout + col;
      float v = acc[i][j] + bias[col];
      if (add) v += resid[at];
      Y[at] = v;
    }
  }
}

// GroupNorm's parts over `pixels` pixels of x [pixels, channels]: a chunk's sums of a group, or of (x - mean)^2.
extern "C" __global__ void tf_md_group_part(const float* __restrict__ X, const float* __restrict__ stats,
                                            float* __restrict__ part, int pixels, int channels, int chunk,
                                            int chunks, int centered) {
  __shared__ float sums[384];
  const int t = threadIdx.x, tg = blockIdx.x;
  const int size = channels / 32;
  const int first = tg * chunk, last = min(first + chunk, pixels);
  const float mean = centered ? stats[t / size] : 0.0f;
  float s = 0.0f;
  for (int px = first; px < last; ++px) {
    const float v = X[(size_t)px * channels + t];
    const float d = v - mean;
    s += centered ? d * d : v;
  }
  sums[t] = s;
  __syncthreads();
  if (t < 32) {
    float all = 0.0f;
    for (int c = 0; c < size; ++c) all += sums[t * size + c];
    part[t * chunks + tg] = all;
  }
}

// stats from part's chunk sums, a warp a group: mean[32] when not centered, else rstd[32] = 1 / sqrt(var + eps).
extern "C" __global__ void tf_md_group_finish(const float* __restrict__ part, float* __restrict__ stats, int pixels,
                                              int channels, int chunks, int centered, float eps) {
  const int lane = threadIdx.x & 31, sg = threadIdx.x >> 5;
  float s = 0.0f;
  for (int i = lane; i < chunks; i += 32) s += part[sg * chunks + i];
  s = warp_sum(s);
  if (lane != 0) return;
  const float n = (float)pixels * (float)(channels / 32);
  if (centered)
    stats[32 + sg] = 1.0f / sqrtf(s / n + eps);
  else
    stats[sg] = s / n;
}

// y = GroupNorm(x) by stats (mean, rstd a group) with g, beta, then SiLU when silu; x, y [pixels, channels].
extern "C" __global__ void __launch_bounds__(1024) tf_md_norm(const float* __restrict__ X, const float* __restrict__ stats,
                                                             const float* __restrict__ G, const float* __restrict__ B,
                                                             float* __restrict__ Y, int channels, int silu) {
  const int c = threadIdx.x, px = blockIdx.x;
  if (c >= channels) return;
  const int grp = c / (channels / 32);
  const size_t i = (size_t)px * channels + c;
  float v = (X[i] - stats[grp]) * stats[32 + grp];
  v = v * G[c] + B[c];
  Y[i] = silu ? v / (1.0f + expf(-v)) : v;
}

// s [rows, rows] = softmax(s * scale) a row: the row's max, then exp, then the row's sum.
extern "C" __global__ void __launch_bounds__(256) tf_md_softmax(float* __restrict__ S, int rows, float scale) {
  float* row = S + (size_t)blockIdx.x * rows;
  __shared__ float scratch[8];
  float most = -INFINITY;
  for (int i = threadIdx.x; i < rows; i += 256) most = fmaxf(most, row[i] * scale);
  most = tf_common::block_max_tree(most, scratch, 8);
  float add = 0.0f;
  for (int i = threadIdx.x; i < rows; i += 256) {
    const float e = expf(row[i] * scale - most);
    row[i] = e;
    add += e;
  }
  const float total = tf_common::block_sum_tree(add, scratch, 8);
  for (int i = threadIdx.x; i < rows; i += 256) row[i] = row[i] / total;
}
