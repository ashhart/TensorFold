// MOSS-Image's glue on CUDA: moss_image.metal's kernels op for op, rounded where the publisher's torch code rounds.
#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_moss {

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

}  // namespace tf_moss

using namespace tf_moss;

// y = bf16(bf16(x * rsqrt(mean(x^2) + eps)) * w), then if modulated bf16(bf16(y * s) + b); a block of 256 a row.
extern "C" __global__ void __launch_bounds__(256) tf_moss_norm(const bf16* __restrict__ X, const bf16* __restrict__ W,
                                                               const bf16* __restrict__ S, const bf16* __restrict__ B,
                                                               bf16* __restrict__ Y, int width, float eps,
                                                               int modulated) {
  __shared__ float part[8];
  const bf16* x = X + (size_t)blockIdx.x * width;
  bf16* y = Y + (size_t)blockIdx.x * width;
  float a = 0.0f;
  for (int i = threadIdx.x; i < width; i += 256) a += ld(x + i) * ld(x + i);
  a = warp_sum(a);
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = a;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < 8; ++w) total += part[w];
  const float inv = 1.0f / sqrtf(total / (float)width + eps);
  for (int i = threadIdx.x; i < width; i += 256) {
    float v = bf(bf(ld(x + i) * inv) * ld(W + i));
    if (modulated) v = bf(v * ld(S + i)) + ld(B + i);
    y[i] = __float2bfloat16_rn(v);
  }
}

// q, k heads RMSNormed, halves (c, c + 64) turned by cos, sin; q | k | v packed for full attention. A warp a head.
extern "C" __global__ void __launch_bounds__(256) tf_moss_qk(const bf16* __restrict__ QR, const bf16* __restrict__ KR,
                                                             const bf16* __restrict__ VR, const bf16* __restrict__ NQ,
                                                             const bf16* __restrict__ NK, const float* __restrict__ CS,
                                                             const float* __restrict__ SN, bf16* __restrict__ QKV,
                                                             int rows, int heads, int stride, float eps) {
  const int r = blockIdx.x, h = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (h >= heads || r >= rows) return;
  const size_t from = (size_t)r * stride + h * 128 + lane * 4;
  float q[4], k[4], qq = 0.0f, kk = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    q[i] = ld(QR + from + i);
    k[i] = ld(KR + from + i);
    qq += q[i] * q[i];
    kk += k[i] * k[i];
  }
  const float qn = 1.0f / sqrtf(warp_sum(qq) / 128.0f + eps);
  const float kn = 1.0f / sqrtf(warp_sum(kk) / 128.0f + eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    q[i] = bf(bf(q[i] * qn) * ld(NQ + lane * 4 + i));
    k[i] = bf(bf(k[i] * kn) * ld(NK + lane * 4 + i));
  }
  // lane l holds channels 4l..4l+3; its partner half sits 16 lanes away. The first half turns by -partner.
  const float sign = lane < 16 ? -1.0f : 1.0f;
  const size_t row = (size_t)r * 3 * heads * 128;
  bf16* qo = QKV + row + h * 128 + lane * 4;
  bf16* ko = qo + heads * 128;
  bf16* vo = ko + heads * 128;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float qp = sign * __shfl_xor_sync(0xffffffffu, q[i], 16);
    const float kp = sign * __shfl_xor_sync(0xffffffffu, k[i], 16);
    const int j = (lane & 15) * 4 + i;
    const float c = CS[(size_t)r * 64 + j], s = SN[(size_t)r * 64 + j];
    qo[i] = __float2bfloat16_rn(bf(q[i] * c) + bf(qp * s));
    ko[i] = __float2bfloat16_rn(bf(k[i] * c) + bf(kp * s));
    vo[i] = VR[from + i];
  }
}

// o = situ(h, g) = (4 tanh(g / 4) sigmoid(g)) (25 tanh(h / 25)), each op rounded to bf16; h and g rows `stride` apart.
extern "C" __global__ void __launch_bounds__(256) tf_moss_situ(const bf16* __restrict__ H, const bf16* __restrict__ G,
                                                               bf16* __restrict__ O, int width, int stride, int rows) {
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  const size_t i = (size_t)r * stride + c;
  const float gv = ld(G + i), hv = ld(H + i);
  const float sig = bf(1.0f / (1.0f + expf(-gv)));
  const float left = bf(bf(4.0f * bf(tanhf(bf(gv * 0.25f)))) * sig);
  // torch divides by a host scalar as a product with its fp32 reciprocal
  const float right = bf(25.0f * bf(tanhf(bf(hv * (1.0f / 25.0f)))));
  O[(size_t)r * width + c] = __float2bfloat16_rn(left * right);
}

// x = bf16(x + bf16(g * y)) by column; g is a modulation gate, or ones for an ungated residual.
extern "C" __global__ void __launch_bounds__(256) tf_moss_gate_add(bf16* __restrict__ X, const bf16* __restrict__ Y,
                                                                   const bf16* __restrict__ G, int width, int rows) {
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  const size_t i = (size_t)r * width + c;
  X[i] = __float2bfloat16_rn(ld(X + i) + bf(ld(G + c) * ld(Y + i)));
}
