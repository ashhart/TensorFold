// Prefill chain: steps never depend on where a chunk starts, so any chunking gives the same bits (not verify's).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>

// ROCm builds (hipified): 64-bit lane masks, HIP's bf16 conversion (round to nearest even) and a kernel
// pointer cast for the attribute call; on CUDA each macro is exactly the original spelling.
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
#define TF_FULL_MASK 0xffffffffffffffffull
#define TF_FLOAT2BF16(x) __float2bfloat16(x)
#define TF_KERNEL_PTR(k) reinterpret_cast<const void*>(k)
#else
#define TF_FULL_MASK 0xffffffffu
#define TF_FLOAT2BF16(x) __float2bfloat16_rn(x)
#define TF_KERNEL_PTR(k) (k)
#endif

namespace {

constexpr int DK = 128;
constexpr int DV = 128;

__device__ __forceinline__ float4 widen(uint2 w) {
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&w.x);
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&w.y);
    return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
}

__device__ __forceinline__ float4 widen(float4 w) { return w; }

// A stage's keys, queries, gates and value rows, loaded into registers a stage ahead (in flight while the current
// stage computes) and then stored as chain_kernel stages them: keys and queries as fp32 float4s, values bf16.
template <typename QK, int THREADS, int ROWS, int STEPS>
struct Prefetch {
    using Raw = typename std::conditional<std::is_same<QK, float>::value, float4, uint2>::type;
    static constexpr int KQ = STEPS * (DK / 4) / THREADS, VV = STEPS * ROWS / 8 / THREADS;
    static_assert(KQ * THREADS == STEPS * (DK / 4) && VV * THREADS * 8 == STEPS * ROWS, "whole slots a thread");
    Raw kr[KQ], qr[KQ];
    uint4 vr[VV];
    float gr, br;

    __device__ __forceinline__ void fetch(const QK* q, const QK* k, const __nv_bfloat16* v, const float* g,
                                          const float* beta, int t0, int W, int hk, int hv, int head, int key_head,
                                          int col0) {
#pragma unroll
        for (int it = 0; it < KQ; ++it) {
            const int i = threadIdx.x + it * THREADS, st = i / (DK / 4), c = i % (DK / 4);
            const size_t at = (static_cast<size_t>(min(t0 + st, W - 1)) * hk + key_head) * DK + 4 * c;
            kr[it] = *reinterpret_cast<const Raw*>(k + at);
            qr[it] = *reinterpret_cast<const Raw*>(q + at);
        }
#pragma unroll
        for (int it = 0; it < VV; ++it) {
            const int i = threadIdx.x + it * THREADS, st = i / (ROWS / 8), c = i % (ROWS / 8);
            vr[it] = *reinterpret_cast<const uint4*>(v + (static_cast<size_t>(min(t0 + st, W - 1)) * hv + head) * DV
                                                     + col0 + 8 * c);
        }
        if (threadIdx.x < STEPS) {
            const size_t at = static_cast<size_t>(min(t0 + static_cast<int>(threadIdx.x), W - 1)) * hv + head;
            gr = g[at];
            br = beta[at];
        }
    }

    __device__ __forceinline__ void store(float4 (*ks)[DK / 4], float4 (*qs)[DK / 4], float* gs, float* bs,
                                          __nv_bfloat16 (*vs)[ROWS]) const {
#pragma unroll
        for (int it = 0; it < KQ; ++it) {
            const int i = threadIdx.x + it * THREADS;
            ks[i / (DK / 4)][i % (DK / 4)] = widen(kr[it]);
            qs[i / (DK / 4)][i % (DK / 4)] = widen(qr[it]);
        }
#pragma unroll
        for (int it = 0; it < VV; ++it) {
            const int i = threadIdx.x + it * THREADS;
            *reinterpret_cast<uint4*>(&vs[i / (ROWS / 8)][8 * (i % (ROWS / 8))]) = vr[it];
        }
        if (threadIdx.x < STEPS) {
            gs[threadIdx.x] = gr;
            bs[threadIdx.x] = br;
        }
    }
};

template <typename QK, int ROWS, int STEPS>
__global__ void __launch_bounds__(2 * ROWS) chain_kernel(
        const QK* __restrict__ q, const QK* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const float* __restrict__ g, const float* __restrict__ beta, const float* __restrict__ state,
        float* __restrict__ last, __nv_bfloat16* __restrict__ y, int W, int hk, int hv) {
    __shared__ float4 ks[STEPS][DK / 4], qs[STEPS][DK / 4];
    __shared__ float gs[STEPS], bs[STEPS];
    __shared__ __align__(16) __nv_bfloat16 vs[STEPS][ROWS];
    const int head = blockIdx.x, row = blockIdx.y * ROWS + (threadIdx.x >> 1), half = threadIdx.x & 1;
    const int key_head = head / (hv / hk);
    float s[64];
    const float* s0 = state + (static_cast<size_t>(head) * DV + row) * DK;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float4 t = *reinterpret_cast<const float4*>(s0 + 8 * j + 4 * half);
        s[4 * j] = t.x; s[4 * j + 1] = t.y; s[4 * j + 2] = t.z; s[4 * j + 3] = t.w;
    }
    Prefetch<QK, 2 * ROWS, ROWS, STEPS> pf;                // each stage's loads in flight during the one before
    pf.fetch(q, k, v, g, beta, 0, W, hk, hv, head, key_head, blockIdx.y * ROWS);
    for (int t0 = 0; t0 < W; t0 += STEPS) {
        const int n = min(STEPS, W - t0);
        __syncthreads();
        pf.store(ks, qs, gs, bs, vs);
        __syncthreads();
        if (t0 + STEPS < W) pf.fetch(q, k, v, g, beta, t0 + STEPS, W, hk, hv, head, key_head, blockIdx.y * ROWS);
        for (int tt = 0; tt < n; ++tt) {
            const float gt = gs[tt], bt = bs[tt];
            const size_t vat = (static_cast<size_t>(t0 + tt) * hv + head) * DV + row;
            const float vt = __bfloat162float(vs[tt][threadIdx.x >> 1]);
            float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float4 kk = ks[tt][2 * j + half];
                s[4 * j] = s[4 * j] * gt;
                s[4 * j + 1] = s[4 * j + 1] * gt;
                s[4 * j + 2] = s[4 * j + 2] * gt;
                s[4 * j + 3] = s[4 * j + 3] * gt;
                m[0] = __fmaf_rn(s[4 * j], kk.x, m[0]);
                m[1] = __fmaf_rn(s[4 * j + 1], kk.y, m[1]);
                m[2] = __fmaf_rn(s[4 * j + 2], kk.z, m[2]);
                m[3] = __fmaf_rn(s[4 * j + 3], kk.w, m[3]);
            }
            float mem = (m[0] + m[1]) + (m[2] + m[3]);
            mem = mem + __shfl_xor_sync(TF_FULL_MASK, mem, 1);
            const float delta = (vt - mem) * bt;
            float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float4 kk = ks[tt][2 * j + half], qq = qs[tt][2 * j + half];
                s[4 * j] = __fmaf_rn(kk.x, delta, s[4 * j]);
                s[4 * j + 1] = __fmaf_rn(kk.y, delta, s[4 * j + 1]);
                s[4 * j + 2] = __fmaf_rn(kk.z, delta, s[4 * j + 2]);
                s[4 * j + 3] = __fmaf_rn(kk.w, delta, s[4 * j + 3]);
                o[0] = __fmaf_rn(s[4 * j], qq.x, o[0]);
                o[1] = __fmaf_rn(s[4 * j + 1], qq.y, o[1]);
                o[2] = __fmaf_rn(s[4 * j + 2], qq.z, o[2]);
                o[3] = __fmaf_rn(s[4 * j + 3], qq.w, o[3]);
            }
            float out = (o[0] + o[1]) + (o[2] + o[3]);
            out = out + __shfl_xor_sync(TF_FULL_MASK, out, 1);
            if (half == 0) y[vat] = TF_FLOAT2BF16(out);
        }
    }
    float* s1 = last + (static_cast<size_t>(head) * DV + row) * DK;
#pragma unroll
    for (int j = 0; j < 16; ++j)
        *reinterpret_cast<float4*>(s1 + 8 * j + 4 * half) = make_float4(s[4 * j], s[4 * j + 1], s[4 * j + 2],
                                                                        s[4 * j + 3]);
}

template <typename QK, int ROWS>
void launch(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g, const at::Tensor& beta,
            const at::Tensor& state, at::Tensor& last, at::Tensor& y) {
    constexpr int STEPS = 32;
    const int W = q.size(0), hk = q.size(1), hv = v.size(1);
    const dim3 grid(hv, DV / ROWS);
    chain_kernel<QK, ROWS, STEPS><<<grid, 2 * ROWS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const QK*>(q.data_ptr()), reinterpret_cast<const QK*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()), g.data_ptr<float>(), beta.data_ptr<float>(),
        state.data_ptr<float>(), last.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), W, hk, hv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

// Fewer heads than SMs: a block takes 64 value rows, not 128, so the chunk still fills the GPU.
void gdn_prefill_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                      const at::Tensor& beta, const at::Tensor& state, at::Tensor& last, at::Tensor& y, int sms) {
    const bool wide = v.size(1) >= sms;
    if (q.scalar_type() == at::kFloat) {
        if (wide) launch<float, 128>(q, k, v, g, beta, state, last, y);
        else launch<float, 64>(q, k, v, g, beta, state, last, y);
    } else {
        if (wide) launch<__nv_bfloat16, 128>(q, k, v, g, beta, state, last, y);
        else launch<__nv_bfloat16, 64>(q, k, v, g, beta, state, last, y);
    }
}
