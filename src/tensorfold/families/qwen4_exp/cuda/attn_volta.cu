// Flash Next's chunked attention partials (o, m, l) on sm_70: one block a (row, key head, chunk), fixed fp32 order.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {

constexpr int THREADS = 256, WARPS = THREADS / 32, MAX_G = 16;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// q [R, H, D] bf16; kc, vc [cap, HK, D] bf16; partials po [R, NCH, H, D], pm / pl [R, NCH, H] fp32.
template <int D>
__global__ void __launch_bounds__(THREADS) chunks_kernel(
        const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KC, const __nv_bfloat16* __restrict__ VC,
        const int* __restrict__ POS0, float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
        const int* __restrict__ IDS, const int* __restrict__ NKR, const int* __restrict__ SPR,
        int H, int HK, int G, int CH, int NCH, float scale, int IDW, int qsa) {
    extern __shared__ float smem[];
    const int r = blockIdx.x, hk = blockIdx.y, c = blockIdx.z;
    int n = POS0[0] + r + 1;
    bool sparse = false;
    if (qsa) {
        sparse = SPR[r] != 0;
        if (sparse) n = NKR[r];
    }
    const int start = c * CH;
    if (start >= n) return;                              // the merge never reads chunks past a row's keys
    const int cnt = min(n - start, CH);
    float* qs = smem;                                    // [G][D]
    float* sc = qs + G * D;                              // [G][CH]: scores, then p
    int* kid = reinterpret_cast<int*>(sc + G * CH);      // [CH]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int i = tid; i < G * D; i += THREADS)
        qs[i] = __bfloat162float(Q[((long)r * H + hk * G + i / D) * D + i % D]);
    for (int j = tid; j < cnt; j += THREADS) kid[j] = sparse ? IDS[(long)r * IDW + start + j] : start + j;
    __syncthreads();
    // scores: a warp a key, a lane D/32 inputs, the lanes' sums added in one fixed tree
    constexpr int PER = D / 32;
    for (int j = warp; j < cnt; j += WARPS) {
        const __nv_bfloat16* krow = KC + ((long)kid[j] * HK + hk) * D + lane * PER;
        float k[PER];
#pragma unroll
        for (int t = 0; t < PER; ++t) k[t] = __bfloat162float(krow[t]);
        for (int g = 0; g < G; ++g) {
            float s = 0.f;
#pragma unroll
            for (int t = 0; t < PER; ++t) s = fmaf(qs[g * D + lane * PER + t], k[t], s);
            s = warp_sum(s);
            if (lane == 0) sc[g * CH + j] = s * scale;
        }
    }
    __syncthreads();
    // each head's max and sum over the chunk's keys (a warp a head), p = exp(s - m) in place
    __shared__ float hm[MAX_G], hl[MAX_G];
    for (int g = warp; g < G; g += WARPS) {
        float m = -INFINITY;
        for (int j = lane; j < cnt; j += 32) m = fmaxf(m, sc[g * CH + j]);
        m = warp_max(m);
        float l = 0.f;
        for (int j = lane; j < cnt; j += 32) {
            const float p = expf(sc[g * CH + j] - m);
            sc[g * CH + j] = p;
            l += p;
        }
        l = warp_sum(l);
        if (lane == 0) { hm[g] = m; hl[g] = l; }
    }
    __syncthreads();
    // o[g][d] = sum over the chunk's keys in order of p[g][j] * v[j][d]: a thread an input dimension
    for (int d = tid; d < D; d += THREADS) {
        float acc[MAX_G];
#pragma unroll
        for (int g = 0; g < MAX_G; ++g) acc[g] = 0.f;
        for (int j = 0; j < cnt; ++j) {
            const float v = __bfloat162float(VC[((long)kid[j] * HK + hk) * D + d]);
#pragma unroll
            for (int g = 0; g < MAX_G; ++g)
                if (g < G) acc[g] = fmaf(sc[g * CH + j], v, acc[g]);
        }
        for (int g = 0; g < G; ++g) PO[(((long)r * NCH + c) * H + hk * G + g) * D + d] = acc[g];
    }
    if (tid < G) {
        const long base = ((long)r * NCH + c) * H + hk * G + tid;
        PM[base] = hm[tid];
        PL[base] = hl[tid];
    }
}

}  // namespace

void chunks(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor pos0, torch::Tensor po,
            torch::Tensor pm, torch::Tensor pl, torch::Tensor ids, torch::Tensor nk, torch::Tensor sparse,
            int64_t rows, int64_t chunks, int64_t ch, int64_t nch, double scale, int64_t idw, bool qsa) {
    TORCH_CHECK(q.dim() == 3 && kc.dim() == 3 && q.scalar_type() == at::kBFloat16 &&
                kc.scalar_type() == at::kBFloat16 && vc.scalar_type() == at::kBFloat16, "bf16 q and caches");
    const int h = (int)q.size(1), d = (int)q.size(2), hk = (int)kc.size(1), g = h / hk;
    TORCH_CHECK(d == 256 || d == 128 || d == 64, "head_dim 64, 128 or 256");
    TORCH_CHECK(g >= 1 && g <= MAX_G && h % hk == 0 && ch <= 1024, "at most 16 query heads a key head, chunks of 1024");
    TORCH_CHECK(q.stride(2) == 1 && q.stride(1) == d && kc.is_contiguous() && vc.is_contiguous(), "contiguous rows");
    c10::cuda::CUDAGuard guard(q.device());
    const size_t smem = (size_t)g * d * 4 + (size_t)g * ch * 4 + (size_t)ch * 4;
    dim3 grid((unsigned)rows, (unsigned)hk, (unsigned)chunks);
    auto st = at::cuda::getCurrentCUDAStream();
#define LAUNCH(DD)                                                                                                      \
    {                                                                                                                   \
        auto k = chunks_kernel<DD>;                                                                                     \
        if (smem > 48 * 1024) C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,      \
                                                                  (int)smem));                                         \
        k<<<grid, THREADS, smem, st>>>(                                                                                 \
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(kc.data_ptr()), \
            reinterpret_cast<const __nv_bfloat16*>(vc.data_ptr()), pos0.data_ptr<int>(), po.data_ptr<float>(),        \
            pm.data_ptr<float>(), pl.data_ptr<float>(), ids.data_ptr<int>(), nk.data_ptr<int>(),                       \
            sparse.data_ptr<int>(), h, hk, g, (int)ch, (int)nch, (float)scale, (int)idw, qsa ? 1 : 0);                 \
    }
    if (d == 256) LAUNCH(256)
    else if (d == 128) LAUNCH(128)
    else LAUNCH(64)
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chunks", &chunks, "chunked attention partials (o, m, l) a (row, key head, chunk), sm_70, bf16 caches");
}
