// MOSS-Image's image decoder controls: each breaks one gate on purpose, and that gate must go red.

// Gate 3: tf_md_gemm without its bias: the sums where float64 adds the output's own bias to them.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_md_gemm_bias(const float* __restrict__ A, const float* __restrict__ B,
                                                                    const float* __restrict__ bias,
                                                                    const float* __restrict__ resid, float* __restrict__ C,
                                                                    int m, int n, int k, int lda, int ldb, int bt, int add) {
  (void)bias;
  const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
  for (int e = threadIdx.x; e < 64 * 64; e += 256) {
    const int i = m0 + e / 64, j = n0 + e % 64;
    if (i >= m || j >= n) continue;
    float s = 0.0f;
    for (int kk = 0; kk < k; ++kk)
      s += bt ? A[(size_t)i * lda + kk] * B[(size_t)j * ldb + kk] : A[(size_t)i * lda + kk] * B[(size_t)kk * ldb + j];
    if (add) s += resid[(size_t)i * n + j];
    C[(size_t)i * n + j] = s;
  }
}

// Gate 3: tf_md_conv3x3 without its bias: the taps' sums where float64 adds the output channel's own bias.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_md_conv3x3_bias(const float* __restrict__ X,
                                                                       const float* __restrict__ U,
                                                                       const float* __restrict__ bias,
                                                                       const float* __restrict__ resid,
                                                                       float* __restrict__ Y, int rows, int cols, int cin,
                                                                       int cout, int up, int add) {
  (void)bias;
  const int y0 = blockIdx.y * 8, x0 = blockIdx.z * 16, n0 = blockIdx.x * 32;
  const int in_cols = up ? cols / 2 : cols;
  for (int e = threadIdx.x; e < 8 * 16 * 32; e += 256) {
    const int oy = y0 + e / (16 * 32), ox = x0 + (e % (16 * 32)) / 32, col = n0 + e % 32;
    if (oy >= rows || ox >= cols || col >= cout) continue;
    float s = 0.0f;
    for (int tap = 0; tap < 9; ++tap) {
      const int iy = oy + tap / 3 - 1, ix = ox + tap % 3 - 1;
      if (iy < 0 || ix < 0 || iy >= rows || ix >= cols) continue;
      const int sy = up ? iy / 2 : iy, sx = up ? ix / 2 : ix;
      for (int c = 0; c < cin; ++c) s += X[(size_t)(sy * in_cols + sx) * cin + c] * U[((size_t)tap * cin + c) * cout + col];
    }
    if (add) s += resid[(size_t)(oy * cols + ox) * cout + col];
    Y[(size_t)(oy * cols + ox) * cout + col] = s;
  }
}

// Gate 3: tf_md_group_part's centered pass summing the values where float64 sums their squares about the mean.
extern "C" __global__ void tf_ctl_md_group_part_centered(const float* __restrict__ X, const float* __restrict__ stats,
                                                         float* __restrict__ part, int pixels, int channels, int chunk,
                                                         int chunks, int centered) {
  (void)stats;
  (void)centered;
  __shared__ float sums[384];
  const int t = threadIdx.x, tg = blockIdx.x;
  const int size = channels / 32;
  const int first = tg * chunk, last = min(first + chunk, pixels);
  float s = 0.0f;
  for (int px = first; px < last; ++px) s += X[(size_t)px * channels + t];
  sums[t] = s;
  __syncthreads();
  if (t < 32) {
    float all = 0.0f;
    for (int c = 0; c < size; ++c) all += sums[t * size + c];
    part[t * chunks + tg] = all;
  }
}

// Gate 3: tf_md_group_finish's divisor without the group's channels: pixels where float64 takes pixels x channels / 32.
extern "C" __global__ void __launch_bounds__(1024) tf_ctl_md_group_finish_pixels(const float* __restrict__ part,
                                                                               float* __restrict__ stats, int pixels,
                                                                               int channels, int chunks, int centered,
                                                                               float eps) {
  const int lane = threadIdx.x & 31, sg = threadIdx.x >> 5;
  float s = 0.0f;
  for (int i = lane; i < chunks; i += 32) s += part[sg * chunks + i];
  s = tf_common::warp_sum(s);
  if (lane != 0) return;
  const float n = (float)pixels;
  if (centered) stats[32 + sg] = 1.0f / sqrtf(s / n + eps);
  else stats[sg] = s / n;
}

// Gate 3: tf_md_softmax without its score scale: each row's values where float64 scales them before the maximum.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_md_softmax_scale(float* __restrict__ S, int rows, float scale) {
  (void)scale;
  float* row = S + (size_t)blockIdx.x * rows;
  __shared__ float scratch[8];
  float most = -INFINITY;
  for (int i = threadIdx.x; i < rows; i += 256) most = fmaxf(most, row[i]);
  most = tf_common::block_max_tree(most, scratch, 8);
  float add = 0.0f;
  for (int i = threadIdx.x; i < rows; i += 256) {
    const float e = expf(row[i] - most);
    row[i] = e;
    add += e;
  }
  const float total = tf_common::block_sum_tree(add, scratch, 8);
  for (int i = threadIdx.x; i < rows; i += 256) row[i] = row[i] / total;
}

// Gate 3: tf_md_norm without beta: (x - mean) rstd gamma where float64 gives that plus beta, so its ranges go red.
extern "C" __global__ void __launch_bounds__(1024) tf_ctl_md_norm_beta(const float* __restrict__ X, const float* __restrict__ stats,
                                                                     const float* __restrict__ G, const float* __restrict__ B,
                                                                     float* __restrict__ Y, int channels, int silu) {
  (void)B;
  const int c = threadIdx.x, px = blockIdx.x;
  if (c >= channels) return;
  const int grp = c / (channels / 32);
  const size_t i = (size_t)px * channels + c;
  float v = (X[i] - stats[grp]) * stats[32 + grp];
  v = v * G[c];
  Y[i] = silu ? v / (1.0f + expf(-v)) : v;
}

// Gate 3: tf_md_unpack without its batch norm mean: (x s) where float64 gives (x s) + m, so its ranges go red.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_md_unpack_mean(const float* __restrict__ X,
                                                                        const float* __restrict__ S,
                                                                        const float* __restrict__ M,
                                                                        float* __restrict__ Z, int rows, int cols) {
  (void)M;
  const int w2 = cols * 2;
  const int at = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int c = threadIdx.x & 31;
  if (at >= rows * 2 * w2) return;
  const int py = at / w2, px = at % w2;
  const int k = c * 4 + (py % 2) * 2 + px % 2;
  Z[(size_t)at * 32 + c] = X[(size_t)((py / 2) * cols + px / 2) * 128 + k] * S[k];
}
