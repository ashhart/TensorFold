// MOSS-Image's image decoder controls: each breaks one gate on purpose, and that gate must go red.

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
