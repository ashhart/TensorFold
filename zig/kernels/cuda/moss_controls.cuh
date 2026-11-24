// MOSS-Image's kernel controls: each breaks one gate on purpose, and that gate must go red.

// Gate 3: moss_image.cu's steps, each with one of the publisher's bf16 points dropped; their range checks must go red.
namespace tf_ctl_moss {
typedef __nv_bfloat16 bf16;
__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }
using tf_common::warp_sum;
// tf_moss_norm; DROP 0: bf16(x inv), 1: bf16(y s).
template <int DROP>
__device__ void norm(const bf16* X, const bf16* W, const bf16* S, const bf16* B, bf16* Y, int width, float eps,
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
  for (int w = 0; w < 8; ++w) total += part[w];
  const float inv = 1.0f / sqrtf(total / (float)width + eps);
  for (int i = threadIdx.x; i < width; i += 256) {
    const float xn = ld(x + i) * inv;
    float v = bf((DROP == 0 ? xn : bf(xn)) * ld(W + i));
    if (modulated) v = (DROP == 1 ? v * ld(S + i) : bf(v * ld(S + i))) + ld(B + i);
    y[i] = __float2bfloat16_rn(v);
  }
}
}  // namespace tf_ctl_moss

extern "C" __global__ void __launch_bounds__(256) tf_ctl_moss_norm_inv(const __nv_bfloat16* X, const __nv_bfloat16* W,
                                                                       const __nv_bfloat16* S, const __nv_bfloat16* B,
                                                                       __nv_bfloat16* Y, int width, float eps,
                                                                       int modulated) {
  tf_ctl_moss::norm<0>(X, W, S, B, Y, width, eps, modulated);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_moss_norm_scale(const __nv_bfloat16* X, const __nv_bfloat16* W,
                                                                         const __nv_bfloat16* S, const __nv_bfloat16* B,
                                                                         __nv_bfloat16* Y, int width, float eps,
                                                                         int modulated) {
  tf_ctl_moss::norm<1>(X, W, S, B, Y, width, eps, modulated);
}

// tf_moss_qk without bf16(q cos).
extern "C" __global__ void __launch_bounds__(256) tf_ctl_moss_qk_cos(const __nv_bfloat16* QR, const __nv_bfloat16* KR,
                                                                     const __nv_bfloat16* VR, const __nv_bfloat16* NQ,
                                                                     const __nv_bfloat16* NK, const float* CS,
                                                                     const float* SN, __nv_bfloat16* QKV, int rows,
                                                                     int heads, int stride, float eps) {
  using namespace tf_ctl_moss;
  const int r = blockIdx.x, h = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (h >= heads || r >= rows) return;
  const size_t from = (size_t)r * stride + h * 128 + lane * 4;
  float q[4], k[4], qq = 0.0f, kk = 0.0f;
  for (int i = 0; i < 4; ++i) {
    q[i] = ld(QR + from + i);
    k[i] = ld(KR + from + i);
    qq += q[i] * q[i];
    kk += k[i] * k[i];
  }
  const float qn = 1.0f / sqrtf(warp_sum(qq) / 128.0f + eps);
  const float kn = 1.0f / sqrtf(warp_sum(kk) / 128.0f + eps);
  for (int i = 0; i < 4; ++i) {
    q[i] = bf(bf(q[i] * qn) * ld(NQ + lane * 4 + i));
    k[i] = bf(bf(k[i] * kn) * ld(NK + lane * 4 + i));
  }
  const float sign = lane < 16 ? -1.0f : 1.0f;
  bf16* qo = QKV + (size_t)r * 3 * heads * 128 + h * 128 + lane * 4;
  bf16* ko = qo + heads * 128;
  bf16* vo = ko + heads * 128;
  for (int i = 0; i < 4; ++i) {
    const float qp = sign * __shfl_xor_sync(0xffffffffu, q[i], 16);
    const float kp = sign * __shfl_xor_sync(0xffffffffu, k[i], 16);
    const int j = (lane & 15) * 4 + i;
    const float c = CS[(size_t)r * 64 + j], s = SN[(size_t)r * 64 + j];
    qo[i] = __float2bfloat16_rn(q[i] * c + bf(qp * s));
    ko[i] = __float2bfloat16_rn(bf(k[i] * c) + bf(kp * s));
    vo[i] = VR[from + i];
  }
}

// tf_moss_situ without bf16(sigmoid(g)).
extern "C" __global__ void __launch_bounds__(256) tf_ctl_moss_situ_sigmoid(const __nv_bfloat16* H, const __nv_bfloat16* G,
                                                                           __nv_bfloat16* O, int width, int stride,
                                                                           int rows) {
  using namespace tf_ctl_moss;
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  const size_t i = (size_t)r * stride + c;
  const float gv = ld(G + i), hv = ld(H + i);
  const float sig = 1.0f / (1.0f + expf(-gv));
  const float left = bf(bf(4.0f * bf(tanhf(bf(gv * 0.25f)))) * sig);
  const float right = bf(25.0f * bf(tanhf(bf(hv * (1.0f / 25.0f)))));
  O[(size_t)r * width + c] = __float2bfloat16_rn(left * right);
}

// tf_moss_gate_add without bf16(g y).
extern "C" __global__ void __launch_bounds__(256) tf_ctl_moss_gate_add_product(__nv_bfloat16* X, const __nv_bfloat16* Y,
                                                                               const __nv_bfloat16* G, int width,
                                                                               int rows) {
  using namespace tf_ctl_moss;
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  const size_t i = (size_t)r * width + c;
  X[i] = __float2bfloat16_rn(ld(X + i) + ld(G + c) * ld(Y + i));
}
// Gate 3: tf_md_* decoder kernels and tf_me_* encoder kernels, one arithmetic step dropped each.
#include "moss_decoder_controls.cuh"
#include "moss_encoder_controls.cuh"
