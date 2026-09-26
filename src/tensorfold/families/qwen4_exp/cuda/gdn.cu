// Flash Next's Gated DeltaNet on CUDA: one block of 1024 threads per value head, a chain of R rows.
//
// Per row: the depthwise conv over [conv state; projection rows] (4 taps, fp32) with SiLU (one bf16
// rounding), fp32 L2 norms of q and k (eps inside the sum, q times DK^-0.5), g = exp(-exp(A_log) *
// softplus(a + dt_bias)) in fp32, beta = bf16(sigmoid(b)), the delta-rule update and read-out in fp32,
// and the sigmoid-gated RMSNorm of the read-out. Warp w owns state rows 4w .. 4w + 3, lane l the columns
// 4l .. 4l + 3. The state update is one routine (``update``) shared with ``replay``, compiled without FMA
// contraction, so a replayed prefix of a window gives the bits of the serial steps.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int DK = 128, DV = 128, TAPS = 4;
// NK key heads and NV value heads: the whole layer (16, 48) or one tensor-parallel rank's share (8, 24)

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float warp_sum(float x) {
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float softplusf_(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

// One delta-rule step on this thread's 4 x 4 block of the state: decay, read (k), correct toward v.
__device__ __forceinline__ void update(float (&s)[4][4], const float (&kk)[4], const float* vrow, int warp,
                                       float g, float beta) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[j][i] = s[j][i] * g;
            kv = kv + s[j][i] * kk[i];
        }
        kv = warp_sum(kv);
        const float delta = (vrow[warp * 4 + j] - kv) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
    }
}

template <int NK, int NV>
__global__ void __launch_bounds__(1024) chain_kernel(
        const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs,
        const __nv_bfloat16* __restrict__ cw, const float* __restrict__ state_in,
        const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        const __nv_bfloat16* __restrict__ norm_w, float eps, int rows,
        __nv_bfloat16* __restrict__ out, float* __restrict__ xs, float* __restrict__ state_out,
        float* __restrict__ k_save, __nv_bfloat16* __restrict__ v_save, float* __restrict__ g_save,
        float* __restrict__ b_save) {
    constexpr int C = 2 * NK * DK + NV * DV;          // conv channels: q | k | v
    constexpr int PW = C + NV * DV + 2 * NV;          // projection row: qkv | z | b | a
    const int hv = blockIdx.x, hk = hv / (NV / NK);
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float qs[DK], ks[DK], vs[DV], ys[DV];
    __shared__ float gates[2], rinv;
    int c = -1;
    if (t < DK) c = hk * DK + t;
    else if (t < 2 * DK) c = NK * DK + hk * DK + (t - DK);
    else if (t < 2 * DK + DV) c = 2 * NK * DK + hv * DV + (t - 2 * DK);
    float s[4][4];
    const size_t sbase = (size_t)hv * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    for (int r = 0; r < rows; ++r) {
        if (c >= 0) {
            float acc = 0.0f;
#pragma unroll
            for (int tap = 0; tap < TAPS; ++tap) {
                const int at = r + tap;
                const float x = at < TAPS - 1 ? __bfloat162float(cs[at * C + c])
                                              : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * PW + c]);
                acc = acc + __bfloat162float(cw[c * TAPS + tap]) * x;
            }
            const float act = bf(acc / (1.0f + expf(-acc)));
            if (t < DK) qs[t] = act;
            else if (t < 2 * DK) ks[t - DK] = act;
            else vs[t - 2 * DK] = act;
        }
        __syncthreads();
        if (warp < 2) {
            float* x = warp == 0 ? qs : ks;
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { v4[i] = x[lane * 4 + i]; ss = ss + v4[i] * v4[i]; }
            ss = warp_sum(ss);
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            if (warp == 0) inv = inv * (1.0f / sqrtf((float)DK));
            __syncwarp();
#pragma unroll
            for (int i = 0; i < 4; ++i) x[lane * 4 + i] = v4[i] * inv;
        } else if (warp == 2 && lane == 0) {
            const float b = __bfloat162float(P[(size_t)r * PW + C + NV * DV + hv]);
            const float a = __bfloat162float(P[(size_t)r * PW + C + NV * DV + NV + hv]);
            gates[0] = expf(-expf(a_log[hv]) * softplusf_(a + dt_bias[hv]));
            gates[1] = bf(sigmoidf_(b));
        }
        __syncthreads();
        const float g = gates[0], beta = gates[1];
        float kk[4], qq[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { kk[i] = ks[lane * 4 + i]; qq[i] = qs[lane * 4 + i]; }
        update(s, kk, vs, warp, g, beta);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
            o = warp_sum(o);
            if (lane == 0) ys[warp * 4 + j] = bf(o);
        }
        // what a replay of this row needs
        if (k_save != nullptr) {
            if (t < DK && hv % (NV / NK) == 0) k_save[((size_t)r * NK + hk) * DK + t] = ks[t];
            if (t < DV) v_save[((size_t)r * NV + hv) * DV + t] = __float2bfloat16_rn(vs[t]);
            if (t == 0) { g_save[r * NV + hv] = g; b_save[r * NV + hv] = beta; }
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
            const float yn = bf(bf(ys[t] * rinv) * __bfloat162float(norm_w[t]));
            const float z = __bfloat162float(P[(size_t)r * PW + C + hv * DV + t]);
            const float o = bf(yn * sigmoidf_(z));
            out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(o);
            const float gs = warp_sum(o);
            if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + warp] = gs;
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

template <int NK, int NV>
__global__ void __launch_bounds__(1024) replay_kernel(
        const float* __restrict__ state_in, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save,
        const float* __restrict__ b_save, int rows, float* __restrict__ state_out) {
    const int hv = blockIdx.x, hk = hv / (NV / NK);
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float vs[DV];
    float s[4][4];
    const size_t sbase = (size_t)hv * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    for (int r = 0; r < rows; ++r) {
        if (t < DV) vs[t] = __bfloat162float(v_save[((size_t)r * NV + hv) * DV + t]);
        __syncthreads();
        float kk[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) kk[i] = k_save[((size_t)r * NK + hk) * DK + lane * 4 + i];
        update(s, kk, vs, warp, g_save[r * NV + hv], b_save[r * NV + hv]);
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
}

}  // namespace

template <typename T>
static T* ptr(const at::Tensor& x) { return x.defined() && x.numel() ? (T*)x.data_ptr() : nullptr; }

void gdn_chain_cuda(const at::Tensor& P, const at::Tensor& cs, const at::Tensor& cw, const at::Tensor& state_in,
                    const at::Tensor& a_log, const at::Tensor& dt_bias, const at::Tensor& norm_w, double eps,
                    int64_t rows, at::Tensor& out, at::Tensor& xs, at::Tensor& state_out, at::Tensor& k_save,
                    at::Tensor& v_save, at::Tensor& g_save, at::Tensor& b_save) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int nv = (int)a_log.numel();
    auto launch = [&](auto kernel, int blocks) {
        kernel<<<blocks, 1024, 0, stream>>>(
        ptr<__nv_bfloat16>(P), ptr<__nv_bfloat16>(cs), ptr<__nv_bfloat16>(cw), ptr<float>(state_in),
        ptr<float>(a_log), ptr<float>(dt_bias), ptr<__nv_bfloat16>(norm_w), (float)eps, (int)rows,
        ptr<__nv_bfloat16>(out), ptr<float>(xs), ptr<float>(state_out), ptr<float>(k_save),
        ptr<__nv_bfloat16>(v_save), ptr<float>(g_save), ptr<float>(b_save));
    };
    if (nv == 48) launch(chain_kernel<16, 48>, 48);
    else if (nv == 24) launch(chain_kernel<8, 24>, 24);
    else TORCH_CHECK(false, "gdn chain: 48 or 24 value heads");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn_replay_cuda(const at::Tensor& state_in, const at::Tensor& k_save, const at::Tensor& v_save,
                     const at::Tensor& g_save, const at::Tensor& b_save, int64_t rows, at::Tensor& state_out) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int nv = (int)g_save.size(1);
    if (nv == 48)
        replay_kernel<16, 48><<<48, 1024, 0, stream>>>(
            ptr<float>(state_in), ptr<float>(k_save), ptr<__nv_bfloat16>(v_save), ptr<float>(g_save),
            ptr<float>(b_save), (int)rows, ptr<float>(state_out));
    else if (nv == 24)
        replay_kernel<8, 24><<<24, 1024, 0, stream>>>(
            ptr<float>(state_in), ptr<float>(k_save), ptr<__nv_bfloat16>(v_save), ptr<float>(g_save),
            ptr<float>(b_save), (int)rows, ptr<float>(state_out));
    else TORCH_CHECK(false, "gdn replay: 48 or 24 value heads");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
