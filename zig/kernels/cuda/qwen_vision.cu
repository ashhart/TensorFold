// Qwen3.5 vision tower kernels (0.6.6's --vision in bf16); epilogues add the bias to cuBLASLt's fp32, round once.
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float rnd(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// Patch embedding plus the resampled position grid: x = bf16(acc + b) + bf16(sum_t pos[idx_t] * w_t).
extern "C" __global__ void __launch_bounds__(256) tf_qv_embed(const float* __restrict__ acc,
                                                              const __nv_bfloat16* __restrict__ bias,
                                                              const __nv_bfloat16* __restrict__ pos,
                                                              const int* __restrict__ idx,
                                                              const float* __restrict__ wts,
                                                              __nv_bfloat16* __restrict__ out, int D) {
    const long long row = blockIdx.x;
    const int* ri = idx + row * 4;
    const float* rw = wts + row * 4;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float h = rnd(acc[row * D + i] + bf(bias[i]));
        float p = 0.0f;
        for (int t = 0; t < 4; ++t) p += bf(pos[(long long)ri[t] * D + i]) * rw[t];
        out[row * D + i] = __float2bfloat16_rn(h + rnd(p));
    }
}

// LayerNorm of bf16 rows in fp32 (mean, then the biased variance about it), bf16 out; a block a row.
extern "C" __global__ void __launch_bounds__(256) tf_qv_layernorm(const __nv_bfloat16* __restrict__ x,
                                                                  const __nv_bfloat16* __restrict__ w,
                                                                  const __nv_bfloat16* __restrict__ b,
                                                                  __nv_bfloat16* __restrict__ y, int D, float eps) {
    __shared__ float part[32];
    __shared__ float stat;
    const long long row = blockIdx.x;
    const __nv_bfloat16* xr = x + row * D;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    float s = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += bf(xr[i]);
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) part[warp] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < warps; ++i) t += part[i];
        stat = t / D;
    }
    __syncthreads();
    const float mean = stat;
    float v = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float d = bf(xr[i]) - mean;
        v += d * d;
    }
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < warps; ++i) t += part[i];
        stat = rsqrtf(t / D + eps);
    }
    __syncthreads();
    const float r = stat;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        y[row * D + i] = __float2bfloat16_rn((bf(xr[i]) - mean) * r * bf(w[i]) + bf(b[i]));
}

// qkv = bf16(acc + b) split into q, k, v [heads][rows][hd]; q and k get the axial 2-D rotary in fp32 (h, then w).
extern "C" __global__ void __launch_bounds__(128) tf_qv_qkv_rope(const float* __restrict__ acc,
                                                                 const __nv_bfloat16* __restrict__ bias,
                                                                 const int* __restrict__ hw,
                                                                 const float* __restrict__ inv,
                                                                 __nv_bfloat16* __restrict__ q,
                                                                 __nv_bfloat16* __restrict__ k,
                                                                 __nv_bfloat16* __restrict__ v, int rows, int heads,
                                                                 int hd) {
    const long long row = blockIdx.x;
    const int head = blockIdx.y;
    const int D = heads * hd;
    const int half = hd / 2, quarter = hd / 4;
    const float ph = static_cast<float>(hw[row * 2]), pw = static_cast<float>(hw[row * 2 + 1]);
    const float* a = acc + row * 3 * D;
    const long long o = (static_cast<long long>(head) * rows + row) * hd;
    for (int j = threadIdx.x; j < hd; j += blockDim.x) {
        const int c = head * hd + j;
        // rotate_half pairs j with j +- hd/2; angle h * inv[jj] for the first quarter's frequencies, w * inv after
        const int pj = j < half ? j + half : j - half;
        const int pc = head * hd + pj;
        const int jj = j % half;
        const float ang = jj < quarter ? ph * inv[jj] : pw * inv[jj - quarter];
        const float cs = cosf(ang), sn = sinf(ang);
        const float sign = j < half ? -1.0f : 1.0f;
        const float qx = rnd(a[c] + bf(bias[c])), qp = rnd(a[pc] + bf(bias[pc]));
        const float kx = rnd(a[D + c] + bf(bias[D + c])), kp = rnd(a[D + pc] + bf(bias[D + pc]));
        q[o + j] = __float2bfloat16_rn(qx * cs + sign * qp * sn);
        k[o + j] = __float2bfloat16_rn(kx * cs + sign * kp * sn);
        v[o + j] = __float2bfloat16_rn(a[2 * D + c] + bf(bias[2 * D + c]));
    }
}

// Attention within one image's rows [s0, s0 + n); a thread a query row, online softmax in fp32, 64-row key tiles.
template <int HD>
__device__ void qv_attention(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                             const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ out, int rows,
                             int heads, int s0, int n, float scale) {
    constexpr int T = 64;
    __shared__ float ks[T][HD];
    __shared__ float vs[T][HD];
    const int head = blockIdx.y;
    const int qi = blockIdx.x * T + threadIdx.x;
    const bool live = qi < n;
    const long long base = static_cast<long long>(head) * rows + s0;
    float qr[HD], acc[HD];
    float m = -INFINITY, l = 0.0f;
    if (live) {
        const __nv_bfloat16* qp = q + (base + qi) * HD;
        for (int d = 0; d < HD; ++d) {
            qr[d] = bf(qp[d]) * scale;
            acc[d] = 0.0f;
        }
    }
    for (int t0 = 0; t0 < n; t0 += T) {
        const int count = min(T, n - t0);
        __syncthreads();
        for (int i = threadIdx.x; i < count * HD; i += blockDim.x) {
            const int r = i / HD, d = i % HD;
            ks[r][d] = bf(k[(base + t0 + r) * HD + d]);
            vs[r][d] = bf(v[(base + t0 + r) * HD + d]);
        }
        __syncthreads();
        if (!live) continue;
        for (int r = 0; r < count; ++r) {
            float s = 0.0f;
#pragma unroll
            for (int d = 0; d < HD; ++d) s += qr[d] * ks[r][d];
            const float mn = fmaxf(m, s);
            const float corr = __expf(m - mn);
            const float p = __expf(s - mn);
            l = l * corr + p;
#pragma unroll
            for (int d = 0; d < HD; ++d) acc[d] = acc[d] * corr + p * vs[r][d];
            m = mn;
        }
    }
    if (!live) return;
    const float il = 1.0f / l;
    __nv_bfloat16* op = out + static_cast<long long>(s0 + qi) * heads * HD + head * HD;
    for (int d = 0; d < HD; ++d) op[d] = __float2bfloat16_rn(acc[d] * il);
}

extern "C" __global__ void __launch_bounds__(64) tf_qv_attention72(const __nv_bfloat16* __restrict__ q,
                                                                   const __nv_bfloat16* __restrict__ k,
                                                                   const __nv_bfloat16* __restrict__ v,
                                                                   __nv_bfloat16* __restrict__ out, int rows,
                                                                   int heads, int s0, int n, float scale) {
    qv_attention<72>(q, k, v, out, rows, heads, s0, n, scale);
}

extern "C" __global__ void __launch_bounds__(64) tf_qv_attention64(const __nv_bfloat16* __restrict__ q,
                                                                   const __nv_bfloat16* __restrict__ k,
                                                                   const __nv_bfloat16* __restrict__ v,
                                                                   __nv_bfloat16* __restrict__ out, int rows,
                                                                   int heads, int s0, int n, float scale) {
    qv_attention<64>(q, k, v, out, rows, heads, s0, n, scale);
}

// y = gelu(bf16(acc + b)), the tanh form (gelu_pytorch_tanh) or the erf form (nn.GELU), computed in fp32, bf16 out.
extern "C" __global__ void __launch_bounds__(256) tf_qv_bias_gelu(const float* __restrict__ acc,
                                                                  const __nv_bfloat16* __restrict__ bias,
                                                                  __nv_bfloat16* __restrict__ y, int D, int tanh_form) {
    const long long row = blockIdx.x;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float x = rnd(acc[row * D + i] + bf(bias[i]));
        float g;
        if (tanh_form) {
            const float kBeta = 0.7978845608028654f;  // sqrt(2 / pi)
            g = 0.5f * x * (1.0f + tanhf(kBeta * (x + 0.044715f * x * x * x)));
        } else {
            g = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
        }
        y[row * D + i] = __float2bfloat16_rn(g);
    }
}

// x += bf16(acc + b), the residual add in bf16 (both operands bf16, the sum rounded).
extern "C" __global__ void __launch_bounds__(256) tf_qv_bias_residual(const float* __restrict__ acc,
                                                                      const __nv_bfloat16* __restrict__ bias,
                                                                      __nv_bfloat16* __restrict__ x, int D) {
    const long long row = blockIdx.x;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float y = rnd(acc[row * D + i] + bf(bias[i]));
        x[row * D + i] = __float2bfloat16_rn(bf(x[row * D + i]) + y);
    }
}

// y = bf16(acc + b): the merger's last Linear.
extern "C" __global__ void __launch_bounds__(256) tf_qv_bias_cast(const float* __restrict__ acc,
                                                                  const __nv_bfloat16* __restrict__ bias,
                                                                  __nv_bfloat16* __restrict__ y, int D) {
    const long long row = blockIdx.x;
    for (int i = threadIdx.x; i < D; i += blockDim.x) y[row * D + i] = __float2bfloat16_rn(acc[row * D + i] + bf(bias[i]));
}
