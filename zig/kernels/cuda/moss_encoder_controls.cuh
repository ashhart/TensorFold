// MOSS-Image's prompt encoder controls: each breaks one gate on purpose, and that gate must go red.

// Gate 3: tf_me_norm without its weight's (1 + w): x's normed value where float64 gives that times (1 + w).
extern "C" __global__ void __launch_bounds__(256) tf_ctl_me_norm_weight(const __nv_bfloat16* __restrict__ X,
                                                                       const __nv_bfloat16* __restrict__ W,
                                                                       __nv_bfloat16* __restrict__ Y, int width,
                                                                       int in_stride, int out_stride, float eps) {
  (void)W;
  __shared__ float part[8];
  const __nv_bfloat16* x = X + (size_t)blockIdx.x * in_stride;
  __nv_bfloat16* y = Y + (size_t)blockIdx.x * out_stride;
  float a = 0.0f;
  for (int i = threadIdx.x; i < width; i += 256) a += __bfloat162float(x[i]) * __bfloat162float(x[i]);
  a = tf_common::block_sum_tree(a, part, 8);
  const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(a, (float)width), eps)));
  for (int i = threadIdx.x; i < width; i += 256) y[i] = __float2bfloat16_rn(__bfloat162float(x[i]) * inv);
}

// Gate 3: tf_me_head without its norm weight's (1 + w): the normed, turned head where float64 keeps (1 + w).
extern "C" __global__ void __launch_bounds__(256) tf_ctl_me_head_weight(const __nv_bfloat16* __restrict__ X,
                                                                       const __nv_bfloat16* __restrict__ W,
                                                                       const float* __restrict__ CS,
                                                                       const float* __restrict__ SN,
                                                                       __nv_bfloat16* __restrict__ Y, int rows,
                                                                       int heads, int in_stride, int head_stride,
                                                                       float eps, int normed) {
  (void)W;
  const int r = blockIdx.x, h = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (h >= heads) return;
  const size_t from = (size_t)r * in_stride + (size_t)h * head_stride + lane * 8;
  const size_t to = ((size_t)h * rows + r) * 256 + lane * 8;
  if (normed == 0) {
#pragma unroll
    for (int i = 0; i < 8; ++i) Y[to + i] = X[from + i];
    return;
  }
  float v[8];
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[i] = __bfloat162float(X[from + i]);
    ss += v[i] * v[i];
  }
  const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(tf_common::warp_sum(ss), 256.0f), eps)));
  float t[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) t[i] = __bfloat162float(__float2bfloat16_rn(v[i] * inv));
  const float sign = (lane & 4) != 0 ? 1.0f : -1.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float partner = sign * __shfl_xor_sync(0xffffffffu, t[i], 4);
    float o = t[i];
    if (lane < 8) {
      const int j = (lane & 3) * 8 + i;
      o = __bfloat162float(__float2bfloat16_rn(t[i] * CS[(size_t)r * 32 + j])) +
          __bfloat162float(__float2bfloat16_rn(partner * SN[(size_t)r * 32 + j]));
    }
    Y[to + i] = __float2bfloat16_rn(o);
  }
}

// Gate 3: tf_me_gate with its sigmoid left in fp32: a bf16(sigmoid g) where float64 rounds the sigmoid first.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_me_gate_sigmoid(const __nv_bfloat16* __restrict__ A,
                                                                        const __nv_bfloat16* __restrict__ QG,
                                                                        __nv_bfloat16* __restrict__ O, int width) {
  const int row = blockIdx.y, col = blockIdx.x * 256 + threadIdx.x;
  if (col >= width) return;
  const size_t i = (size_t)row * width + col;
  const float g = __bfloat162float(QG[(size_t)row * 2 * width + (col / 256) * 512 + 256 + col % 256]);
  O[i] = __float2bfloat16_rn(__bfloat162float(A[i]) * __fdiv_rn(1.0f, 1.0f + expf(-g)));
}
