// GLM-5.3-Flash's KDA (Kimi delta attention) on CUDA: one block of 1024 threads per head, a chain of R rows.
//
// Per row, following the Hugging Face definition: the depthwise conv over [conv state; q|k|v rows] (4 taps,
// fp32) with SiLU and one bf16 rounding; fp32 L2 norms of q and k (eps inside the sum, q times DK^-0.5); the
// per-channel decay g_i = exp(lower * sigmoid(exp(A_log) * (a_i + dt_bias_i))) in fp32; beta = bf16(sigmoid(b));
// the delta rule in fp32 (decay along the key channel, read with k, correct toward v, read out with q);
// read-out rounded to bf16; then the gated RMSNorm bf16(w * (y * rsqrt(mean(y^2) + eps)) * sigmoid(gate)).
// Warp w owns value rows 4w .. 4w + 3, lane l the key columns 4l .. 4l + 3. The state update is one routine
// (``update``) shared with ``replay``, compiled without FMA contraction, so a replayed prefix of a window gives
// the bits of the serial steps.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int DK = 128, DV = 128, TAPS = 4;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float warp_sum(float x) {
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

// One delta-rule step on this thread's 4 x 4 block of the state: per-channel decay, read (k), correct toward v.
__device__ __forceinline__ void update(float (&s)[4][4], const float (&kk)[4], const float (&gg)[4],
                                       const float* vrow, int warp, float beta) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[j][i] = s[j][i] * gg[i];
            kv = kv + s[j][i] * kk[i];
        }
        kv = warp_sum(kv);
        const float delta = (vrow[warp * 4 + j] - kv) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
    }
}

__global__ void __launch_bounds__(1024) chain_kernel(
        int H, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off,
        const __nv_bfloat16* __restrict__ A, int a_stride, const __nv_bfloat16* __restrict__ G, int g_stride,
        const __nv_bfloat16* __restrict__ cs, const __nv_bfloat16* __restrict__ cw,
        const float* __restrict__ state_in, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        const __nv_bfloat16* __restrict__ norm_w, float eps, float lower, int rows,
        __nv_bfloat16* __restrict__ out, float* __restrict__ state_out,
        float* __restrict__ k_save, __nv_bfloat16* __restrict__ v_save, float* __restrict__ g_save,
        float* __restrict__ b_save) {
    const int C = 3 * H * DK;                         // conv channels: q | k | v
    const int h = blockIdx.x;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float qs[DK], ks[DK], vs[DV], ys[DV], gs[DK];
    __shared__ float beta_s, rinv;
    int c = -1;
    if (t < 3 * DK) c = (t / DK) * H * DK + h * DK + (t % DK);
    float s[4][4];
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    const float decay_rate = expf(a_log[h]);
    for (int r = 0; r < rows; ++r) {
        if (c >= 0) {
            float acc = 0.0f;
#pragma unroll
            for (int tap = 0; tap < TAPS; ++tap) {
                const int at = r + tap;
                const float x = at < TAPS - 1 ? __bfloat162float(cs[(size_t)at * C + c])
                                              : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * p_stride + c]);
                acc = acc + __bfloat162float(cw[(size_t)c * TAPS + tap]) * x;
            }
            const float act = bf(acc / (1.0f + expf(-acc)));
            if (t < DK) qs[t] = act;
            else if (t < 2 * DK) ks[t - DK] = act;
            else vs[t - 2 * DK] = act;
        } else if (t >= 512 && t < 512 + DK) {
            const int i = t - 512;
            const float a = __bfloat162float(A[(size_t)r * a_stride + h * DK + i]) + dt_bias[h * DK + i];
            gs[i] = expf(lower * sigmoidf_(decay_rate * a));
        } else if (t == 1023) {
            beta_s = bf(sigmoidf_(__bfloat162float(P[(size_t)r * p_stride + b_off + h])));
        }
        __syncthreads();
        if (warp < 2) {
            float* x = warp == 0 ? qs : ks;
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { v4[i] = x[lane * 4 + i]; ss = ss + v4[i] * v4[i]; }
            ss = warp_sum(ss);
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            __syncwarp();
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float y = v4[i] * inv;
                if (warp == 0) y = y * (1.0f / sqrtf((float)DK));
                x[lane * 4 + i] = y;
            }
        }
        __syncthreads();
        const float beta = beta_s;
        float kk[4], qq[4], gg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { kk[i] = ks[lane * 4 + i]; qq[i] = qs[lane * 4 + i]; gg[i] = gs[lane * 4 + i]; }
        update(s, kk, gg, vs, warp, beta);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
            o = warp_sum(o);
            if (lane == 0) ys[warp * 4 + j] = bf(o);
        }
        if (k_save != nullptr) {
            const size_t base = ((size_t)r * H + h) * DK;
            if (t < DK) { k_save[base + t] = ks[t]; g_save[base + t] = gs[t]; }
            if (t >= DK && t < DK + DV) v_save[base + t - DK] = __float2bfloat16_rn(vs[t - DK]);
            if (t == 0) b_save[r * H + h] = beta;
        }
        __syncthreads();
        if (warp == 0) {
            float ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { const float y = ys[lane * 4 + i]; ss = ss + y * y; }
            ss = warp_sum(ss);
            if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
        }
        __syncthreads();
        if (t < DV) {
            const float yn = ys[t] * rinv;
            const float yw = __bfloat162float(norm_w[t]) * yn;
            const float gate = __bfloat162float(G[(size_t)r * g_stride + h * DV + t]);
            out[(size_t)r * H * DV + h * DV + t] = __float2bfloat16_rn(yw * sigmoidf_(gate));
        }
        __syncthreads();
    }
    if (state_out != nullptr) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
    }
}

__global__ void __launch_bounds__(1024) replay_kernel(
        int H, const float* __restrict__ state_in, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save,
        const float* __restrict__ b_save, int rows, float* __restrict__ state_out) {
    const int h = blockIdx.x;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float vs[DV];
    float s[4][4];
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    for (int r = 0; r < rows; ++r) {
        const size_t base = ((size_t)r * H + h) * DK;
        if (t < DV) vs[t] = __bfloat162float(v_save[base + t]);
        __syncthreads();
        float kk[4], gg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { kk[i] = k_save[base + lane * 4 + i]; gg[i] = g_save[base + lane * 4 + i]; }
        update(s, kk, gg, vs, warp, b_save[r * H + h]);
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
}

// All layers at once: block (layer, head); per-layer strides of the state buffers and saved rows.
__global__ void __launch_bounds__(1024) replay_layers_kernel(
        int H, const float* __restrict__ state_in, size_t state_stride, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save, const float* __restrict__ b_save,
        size_t kv_stride, size_t b_stride, int rows, float* __restrict__ state_out) {
    const int layer = blockIdx.x / H, h = blockIdx.x % H;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float vs[DV];
    float s[4][4];
    const float* sin = state_in + layer * state_stride;
    float* sout = state_out + layer * state_stride;
    const float* ks = k_save + layer * kv_stride;
    const __nv_bfloat16* vsv = v_save + layer * kv_stride;
    const float* gsv = g_save + layer * kv_stride;
    const float* bsv = b_save + layer * b_stride;
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = sin[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    for (int r = 0; r < rows; ++r) {
        const size_t base = ((size_t)r * H + h) * DK;
        if (t < DV) vs[t] = __bfloat162float(vsv[base + t]);
        __syncthreads();
        float kk[4], gg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { kk[i] = ks[base + lane * 4 + i]; gg[i] = gsv[base + lane * 4 + i]; }
        update(s, kk, gg, vs, warp, bsv[r * H + h]);
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) sout[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
}

}  // namespace

template <typename T>
static T* ptr(const at::Tensor& x) { return x.defined() && x.numel() ? (T*)x.data_ptr() : nullptr; }

void kda_chain_cuda(const at::Tensor& P, int64_t p_stride, int64_t b_off, const at::Tensor& A, int64_t a_stride,
                    const at::Tensor& G, int64_t g_stride, const at::Tensor& cs, const at::Tensor& cw,
                    const at::Tensor& state_in, const at::Tensor& a_log, const at::Tensor& dt_bias,
                    const at::Tensor& norm_w, double eps, double lower, int64_t rows, at::Tensor& out,
                    at::Tensor& state_out, at::Tensor& k_save, at::Tensor& v_save, at::Tensor& g_save,
                    at::Tensor& b_save) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int H = (int)a_log.numel();
    chain_kernel<<<H, 1024, 0, stream>>>(
        H, ptr<__nv_bfloat16>(P), (int)p_stride, (int)b_off, ptr<__nv_bfloat16>(A), (int)a_stride,
        ptr<__nv_bfloat16>(G), (int)g_stride, ptr<__nv_bfloat16>(cs), ptr<__nv_bfloat16>(cw), ptr<float>(state_in),
        ptr<float>(a_log), ptr<float>(dt_bias), ptr<__nv_bfloat16>(norm_w), (float)eps, (float)lower, (int)rows,
        ptr<__nv_bfloat16>(out), ptr<float>(state_out), ptr<float>(k_save), ptr<__nv_bfloat16>(v_save),
        ptr<float>(g_save), ptr<float>(b_save));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void kda_replay_cuda(const at::Tensor& state_in, const at::Tensor& k_save, const at::Tensor& v_save,
                     const at::Tensor& g_save, const at::Tensor& b_save, int64_t rows, at::Tensor& state_out) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int H = (int)b_save.size(1);
    replay_kernel<<<H, 1024, 0, stream>>>(H, ptr<float>(state_in), ptr<float>(k_save), ptr<__nv_bfloat16>(v_save),
                                          ptr<float>(g_save), ptr<float>(b_save), (int)rows, ptr<float>(state_out));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void kda_replay_layers_cuda(const at::Tensor& state_in, int64_t state_stride, const at::Tensor& k_save,
                            const at::Tensor& v_save, const at::Tensor& g_save, const at::Tensor& b_save,
                            int64_t kv_stride, int64_t b_stride, int64_t layers, int64_t heads, int64_t rows,
                            at::Tensor& state_out) {
    auto stream = at::cuda::getCurrentCUDAStream();
    replay_layers_kernel<<<(int)(layers * heads), 1024, 0, stream>>>(
        (int)heads, ptr<float>(state_in), (size_t)state_stride, ptr<float>(k_save), ptr<__nv_bfloat16>(v_save),
        ptr<float>(g_save), ptr<float>(b_save), (size_t)kv_stride, (size_t)b_stride, (int)rows, ptr<float>(state_out));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
