// A hyper-connection stream's RMS: 1 / sqrt(mean of its squares + eps), from nt partial sums a row and stream.
#include <metal_stdlib>
inline float stream_rinv(const device float* ssp, int r, int s, int nt, int streams, int dims, float eps) {
  float total = 0.0f;
  for (int j = 0; j < nt; j++) total += ssp[(r * nt + j) * streams + s];
  return metal::rsqrt(total / float(dims) + eps);
}
