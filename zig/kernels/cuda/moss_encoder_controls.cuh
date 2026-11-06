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

// Gate 3: tf_me_silu_mul without its up projection: bf16(silu g) where float64 multiplies it by u.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_me_silu_mul_up(const __nv_bfloat16* __restrict__ G,
                                                                       const __nv_bfloat16* __restrict__ U,
                                                                       __nv_bfloat16* __restrict__ O, int width) {
  (void)U;
  const int row = blockIdx.y, col = blockIdx.x * 256 + threadIdx.x;
  if (col >= width) return;
  const size_t i = (size_t)row * width + col;
  const float g = __bfloat162float(G[i]);
  O[i] = __float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(__fdiv_rn(g, 1.0f + expf(-g)))));
}

// Gate 3: tf_me_conv without its silu: bf16(the taps' bf16 sum) where float64 takes the silu of it.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_me_conv_silu(const __nv_bfloat16* __restrict__ X,
                                                                     const __nv_bfloat16* __restrict__ W,
                                                                     __nv_bfloat16* __restrict__ Y, int width) {
  const int row = blockIdx.y, col = blockIdx.x * 256 + threadIdx.x;
  if (col >= width) return;
  float s = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    if (row + k >= 3) s += __bfloat162float(W[col * 4 + k]) * __bfloat162float(X[(size_t)(row + k - 3) * width + col]);
  }
  Y[(size_t)row * width + col] = __float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(s)));
}

// Gate 3: tf_me_gated_norm without its learned weight: bf16(bf16(x inv) silu z) where float64 takes w's product.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_me_gated_norm_weight(const __nv_bfloat16* __restrict__ X,
                                                                             const __nv_bfloat16* __restrict__ Z,
                                                                             const __nv_bfloat16* __restrict__ W,
                                                                             __nv_bfloat16* __restrict__ Y,
                                                                             int width, float eps) {
  (void)W;
  const int r = blockIdx.x, h = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (h * 128 >= width) return;
  const size_t at = (size_t)r * width + h * 128 + lane * 4;
  float v[4];
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[i] = __bfloat162float(X[at + i]);
    ss += v[i] * v[i];
  }
  const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(tf_common::warp_sum(ss), 128.0f), eps)));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float n = __bfloat162float(__float2bfloat16_rn(v[i] * inv));
    const float z = __bfloat162float(Z[at + i]);
    Y[at + i] = __float2bfloat16_rn(n * __fdiv_rn(z, 1.0f + expf(-z)));
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

// Gate 3: tf_me_attention without its score scale: 16x sharper weights where float64 multiplies the dot by 1/16.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_me_attention_scale(const __nv_bfloat16* __restrict__ Q,
                                                                           const __nv_bfloat16* __restrict__ K,
                                                                           const __nv_bfloat16* __restrict__ V,
                                                                           __nv_bfloat16* __restrict__ O, int rows,
                                                                           int heads, int kv_heads, float scale) {
  (void)scale;
  __shared__ float part[8];
  const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int group = heads / kv_heads, kvh = h / group;
  const float q = __bfloat162float(Q[((size_t)h * rows + t) * 256 + d]);
  float m = -INFINITY, l = 0.0f, acc = 0.0f;
  for (int j = 0; j <= t; ++j) {
    const size_t at = ((size_t)kvh * rows + j) * 256 + d;
    const float s = tf_common::block_sum_tree(q * __bfloat162float(K[at]), part, 8);
    const float top = fmaxf(m, s);
    const float old = expf(m - top), e = expf(s - top);
    l = l * old + e;
    acc = acc * old + e * __bfloat162float(V[at]);
    m = top;
  }
  O[((size_t)t * heads + h) * 256 + d] = __float2bfloat16_rn(__fdiv_rn(acc, l));
}

// Gate 3: tf_me_gdn_prep leaving beta's sigmoid in fp32: bf16(sigmoid b) where float64 rounds the sigmoid first.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_me_gdn_prep_beta(const __nv_bfloat16* __restrict__ MIX,
                                                                         const __nv_bfloat16* __restrict__ A,
                                                                         const __nv_bfloat16* __restrict__ B,
                                                                         const __nv_bfloat16* __restrict__ A_LOG,
                                                                         const __nv_bfloat16* __restrict__ DT_BIAS,
                                                                         float* __restrict__ QN, float* __restrict__ KN,
                                                                         float* __restrict__ G, float* __restrict__ BETA,
                                                                         int rows, int kheads, int vheads, int mix_stride,
                                                                         float scale) {
  const int r = blockIdx.x, h = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (h >= kheads) return;
  const size_t row = (size_t)r * mix_stride;
  const int kdim = kheads * 128;
  float q[4], k[4];
  float qq = 0.0f, kk = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    q[i] = __bfloat162float(MIX[row + h * 128 + lane * 4 + i]);
    k[i] = __bfloat162float(MIX[row + kdim + h * 128 + lane * 4 + i]);
    qq += q[i] * q[i];
    kk += k[i] * k[i];
  }
  const float qi = __fdiv_rn(1.0f, __fsqrt_rn(tf_common::warp_sum(qq) + 1e-6f));
  const float ki = __fdiv_rn(1.0f, __fsqrt_rn(tf_common::warp_sum(kk) + 1e-6f));
  const size_t to = ((size_t)r * kheads + h) * 128 + lane * 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    QN[to + i] = q[i] * qi * scale;
    KN[to + i] = k[i] * ki;
  }
  const int per = vheads / kheads;
  if (lane < per) {
    const int v = h * per + lane;
    const size_t at = (size_t)r * vheads + v;
    BETA[at] = __fdiv_rn(1.0f, 1.0f + expf(-__bfloat162float(B[at])));
    const float x = __bfloat162float(A[at]) + __bfloat162float(DT_BIAS[v]);
    const float soft = x > 20.0f ? x : logf(1.0f + expf(x));
    G[at] = -expf(__bfloat162float(A_LOG[v])) * soft;
  }
}

// Gate 3: tf_me_gdn without its decay: the state never forgets where float64 multiplies it by exp g a row.
extern "C" __global__ void __launch_bounds__(128) tf_ctl_me_gdn_decay(const float* __restrict__ QN,
                                                                      const float* __restrict__ KN,
                                                                      const __nv_bfloat16* __restrict__ MIX,
                                                                      const float* __restrict__ G,
                                                                      const float* __restrict__ BETA,
                                                                      __nv_bfloat16* __restrict__ OUT, int rows,
                                                                      int kheads, int vheads, int mix_stride) {
  __shared__ float kq[256];
  const int h = blockIdx.x, j = threadIdx.x;
  const int per = vheads / kheads, kh = h / per;
  const size_t voff = (size_t)2 * kheads * 128 + (size_t)h * 128 + j;
  float s[128];
#pragma unroll
  for (int i = 0; i < 128; ++i) s[i] = 0.0f;
  for (int t = 0; t < rows; ++t) {
    const size_t at = ((size_t)t * kheads + kh) * 128 + j;
    kq[j] = KN[at];
    kq[128 + j] = QN[at];
    __syncthreads();
    const float v = __bfloat162float(MIX[(size_t)t * mix_stride + voff]);
    float kv = 0.0f;
#pragma unroll
    for (int i = 0; i < 128; ++i) kv += s[i] * kq[i];
    const float delta = (v - kv) * BETA[(size_t)t * vheads + h];
    float o = 0.0f;
#pragma unroll
    for (int i = 0; i < 128; ++i) {
      s[i] += kq[i] * delta;
      o += s[i] * kq[128 + i];
    }
    OUT[((size_t)t * vheads + h) * 128 + j] = __float2bfloat16_rn(o);
    __syncthreads();
  }
}
