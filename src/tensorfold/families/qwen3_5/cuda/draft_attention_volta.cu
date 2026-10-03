// Volta (sm_70) DFlash2 block attention: draft_attention.py's _block_attention without tl.dot.
//
// Drafts only propose (the target verifies every token), so the context is split across blocks for parallelism:
// block (stream j, kv head g, split c) takes KC context keys (the last split takes the block's own keys) for the
// G x L query rows of that kv head and writes a partial (max, sum, unnormalized output); a second kernel folds
// the splits in order and normalizes. Masks follow the Triton kernel: a context key is seen while
// s + r - key < window + 1, a block key always (causal: key <= r).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

namespace {

constexpr int KC = 32, THREADS = 128;           // keys a split (a block's own keys fit one): 74 KB of smem at 64 rows x D 128
constexpr size_t SMEM_LIMIT = 96 * 1024;        // Volta's opt-in dynamic shared memory

__device__ __forceinline__ float bf(unsigned short v) { return __uint_as_float(static_cast<unsigned>(v) << 16); }

template <int D>
__global__ void __launch_bounds__(THREADS) split_kernel(
        const unsigned short* __restrict__ Q, const unsigned short* __restrict__ KB, const unsigned short* __restrict__ VB,
        const long* __restrict__ TABLE, const long* __restrict__ LENS, float* __restrict__ PM, float* __restrict__ PL,
        float* __restrict__ PO, int R, int L, int G, int window, int nsplit, bool causal, float scale) {
    const int j = blockIdx.x, g = blockIdx.y, c = blockIdx.z;
    const int rows = G * L;                                  // query rows of this kv head: (head, block row)
    const int s = (int)LENS[j];
    const bool block_keys = c == nsplit - 1;
    const int k0 = c * KC;
    const int nk = block_keys ? L : max(0, min(KC, s - k0));

    extern __shared__ float smem[];
    float* sq = smem;                                        // rows x D
    float* sk = sq + rows * D;                               // KC x (D + 1)
    float* sv = sk + KC * (D + 1);                           // KC x D
    float* sp = sv + KC * D;                                 // rows x KC
    const long pbase = (((long)j * gridDim.y + g) * nsplit + c) * rows;

    if (nk == 0) {
        for (int i = threadIdx.x; i < rows; i += THREADS) { PM[pbase + i] = -INFINITY; PL[pbase + i] = 0.f; }
        for (int i = threadIdx.x; i < rows * D; i += THREADS) PO[pbase * D + i] = 0.f;
        return;
    }
    for (int i = threadIdx.x; i < rows * D; i += THREADS) {
        const int row = i / D, d = i % D, head = g * G + row / L, r = row % L;
        sq[i] = bf(Q[((long)head * R + j * L + r) * D + d]);
    }
    const unsigned short* kc = reinterpret_cast<const unsigned short*>(TABLE[2 * j]) + (long)g * s * D;
    const unsigned short* vc = reinterpret_cast<const unsigned short*>(TABLE[2 * j + 1]) + (long)g * s * D;
    for (int i = threadIdx.x; i < nk * D; i += THREADS) {
        const int t = i / D, d = i % D;
        const long src = block_keys ? ((long)g * R + j * L + t) * D + d : (long)(k0 + t) * D + d;
        sk[t * (D + 1) + d] = bf(block_keys ? KB[src] : kc[src]);
        sv[t * D + d] = bf(block_keys ? VB[src] : vc[src]);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < rows * KC; i += THREADS) {
        const int row = i / KC, t = i % KC, r = row % L;
        float a = -INFINITY;
        bool seen = t < nk;
        if (seen) seen = block_keys ? (!causal || t <= r) : (s + r - (k0 + t) < window + 1);
        if (seen) {
            a = 0.f;
            const float* qr = sq + row * D;
            const float* kr = sk + t * (D + 1);
#pragma unroll 8
            for (int d = 0; d < D; ++d) a = fmaf(qr[d], kr[d], a);
            a *= scale;
        }
        sp[i] = a;
    }
    __syncthreads();
    // per row: max and sum over this split's keys (one thread a row)
    for (int row = threadIdx.x; row < rows; row += THREADS) {
        float m = -INFINITY;
        for (int t = 0; t < KC; ++t) m = fmaxf(m, sp[row * KC + t]);
        float l = 0.f;
        for (int t = 0; t < KC; ++t) {
            const float e = m == -INFINITY ? 0.f : expf(sp[row * KC + t] - m);
            sp[row * KC + t] = e;
            l += e;
        }
        PM[pbase + row] = m;
        PL[pbase + row] = l;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < rows * D; i += THREADS) {
        const int row = i / D, d = i % D;
        float o = 0.f;
        for (int t = 0; t < nk; ++t) o = fmaf(sp[row * KC + t], sv[t * D + d], o);
        PO[pbase * D + i] = o;
    }
}

__global__ void fold_kernel(const float* __restrict__ PM, const float* __restrict__ PL, const float* __restrict__ PO,
                            unsigned short* __restrict__ OUT, int L, int G, int HKV, int D, int nsplit) {
    const int j = blockIdx.x, g = blockIdx.y, row = blockIdx.z;   // row: (head in group, block row)
    const int rows = G * L, head = g * G + row / L, r = row % L;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float M = -INFINITY, Ls = 0.f, O = 0.f;
        for (int c = 0; c < nsplit; ++c) {
            const long b = (((long)j * HKV + g) * nsplit + c) * rows + row;
            const float cm = PM[b], cl = PL[b];
            if (!(cl > 0.f)) continue;
            const float nm = fmaxf(M, cm);
            const float a = M == -INFINITY ? 0.f : expf(M - nm), e = expf(cm - nm);
            O = O * a + PO[b * D + d] * e;
            Ls = Ls * a + cl * e;
            M = nm;
        }
        const float v = O / Ls;
        const __nv_bfloat16 h = __float2bfloat16_rn(v);
        OUT[((long)j * L + r) * (G * HKV * D) + (long)head * D + d] = *reinterpret_cast<const unsigned short*>(&h);
    }
}

}  // namespace

torch::Tensor block_attention(torch::Tensor q, torch::Tensor kb, torch::Tensor vb, torch::Tensor table,
                              torch::Tensor lens, int64_t length, int64_t window, double scale, bool causal,
                              int64_t max_context) {
    const int H = q.size(0), R = q.size(1), D = q.size(2), HKV = kb.size(0), G = H / HKV, L = length;
    const int S = R / L;
    TORCH_CHECK(D == 128 || D == 64, "draft attention (Volta): head dim 64 or 128, not ", D);
    TORCH_CHECK(L <= KC, "draft attention (Volta): a block of at most ", KC, " rows, not ", L);
    const int nsplit = (int)((max_context + KC - 1) / KC) + 1;
    auto opts = q.options().dtype(at::kFloat);
    const int rows = G * L;
    auto pm = torch::empty({S, HKV, nsplit, rows}, opts), pl = torch::empty_like(pm);
    auto po = torch::empty({S, HKV, nsplit, rows, D}, opts);
    auto out = torch::empty({R, H * D}, q.options());
    auto st = at::cuda::getCurrentCUDAStream();
    const size_t smem = (size_t)(rows * D + KC * (D + 1) + KC * D + rows * KC) * sizeof(float);
    TORCH_CHECK(smem <= SMEM_LIMIT, "draft attention (Volta): ", G, " query heads a key head by ", L, " rows at head dim ",
                D, " need ", smem, " bytes of shared memory, over the ", SMEM_LIMIT, " a block can take");
    auto u = [](const torch::Tensor& t) { return reinterpret_cast<const unsigned short*>(t.data_ptr<at::BFloat16>()); };
#define LAUNCH(DV)                                                                                                  \
    do {                                                                                                            \
        cudaFuncSetAttribute(split_kernel<DV>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);             \
        split_kernel<DV><<<dim3(S, HKV, nsplit), THREADS, smem, st>>>(u(q), u(kb), u(vb), table.data_ptr<long>(),   \
            lens.data_ptr<long>(), pm.data_ptr<float>(), pl.data_ptr<float>(), po.data_ptr<float>(), R, L, G,      \
            (int)window, nsplit, causal, (float)scale);                                                             \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                             \
    } while (0)
    if (D == 128) LAUNCH(128); else LAUNCH(64);
#undef LAUNCH
    fold_kernel<<<dim3(S, HKV, rows), 128, 0, st>>>(pm.data_ptr<float>(), pl.data_ptr<float>(), po.data_ptr<float>(),
        reinterpret_cast<unsigned short*>(out.data_ptr<at::BFloat16>()), L, G, HKV, D, nsplit);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_attention", &block_attention, "Volta DFlash2 block attention, context split across blocks");
}
