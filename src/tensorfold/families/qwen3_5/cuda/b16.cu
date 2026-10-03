// y = x @ W + bias for an unquantized fp16/bf16 W: a warp an output column (several rows sharing its weight loads),
// lanes in fixed k order, a fixed xor butterfly, so a row's bits are its own.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// ROCm builds (hipified): 64-bit lane masks, HIP's bf16 conversion (round to nearest even), a plain load for one
// element (HIP's __ldg has no bf16 overload); on CUDA each macro is exactly the original spelling.
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
#define TF_ROCM 1
#define TF_FULL_MASK 0xffffffffffffffffull
#define TF_FLOAT2BF16(x) __float2bfloat16(x)
#define TF_LDG1(p) (*(p))
#else
#define TF_FULL_MASK 0xffffffffu
#define TF_FLOAT2BF16(x) __float2bfloat16_rn(x)
#define TF_LDG1(p) __ldg(p)
#endif

namespace {

template <typename T>
__device__ __forceinline__ float f2f(T v);
template <>
__device__ __forceinline__ float f2f<half>(half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float f2f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T f_from(float v);
template <>
__device__ __forceinline__ half f_from<half>(float v) { return __float2half_rn(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 f_from<__nv_bfloat16>(float v) { return TF_FLOAT2BF16(v); }

// Eight elements (16 bytes) as fp32.
template <typename T>
__device__ __forceinline__ void ld8(const T* __restrict__ p, float (&v)[8]) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if constexpr (std::is_same_v<T, half>) {
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w[i]));
            v[2 * i] = f.x; v[2 * i + 1] = f.y;
        } else {
            const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[i]));
            v[2 * i] = f.x; v[2 * i + 1] = f.y;
        }
    }
}

// Grid (ceil(N / WARPS), ceil(M / R)): a warp one output column for R rows, each weight load serving the R rows (the
// block's warps share the rows' loads in L1); a row's sum keeps the one-row order (lanes in fixed k order, the same xor
// butterfly). K % 8 == 0, 16-byte rows.
template <typename T, int R, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) b16_kernel(const T* __restrict__ x, const T* __restrict__ w0,
                                                         const T* __restrict__ bias, T* __restrict__ y0, int M, int K,
                                                         int N0, const T* __restrict__ w1, T* __restrict__ y1, int N1) {
    const T* w = blockIdx.z ? w1 : w0;                     // grid z 2: a second weight of the same rows, its own out
    T* y = blockIdx.z ? y1 : y0;
    const int N = blockIdx.z ? N1 : N0;
    const int lane = threadIdx.x & 31;
    const int row0 = blockIdx.y * R;
    const int col = blockIdx.x * WARPS + (threadIdx.x >> 5);
    if (col >= N) return;
    const T* wr = w + (size_t)col * K;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    for (int k = 8 * lane; k + 8 <= K; k += 256) {
        float b[8];
        ld8(wr + k, b);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r >= M) break;
            float a[8];
            ld8(x + (size_t)(row0 + r) * K + k, a);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[r] = fmaf(a[i], b[i], acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r >= M) break;
        float v = acc[r];
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(TF_FULL_MASK, v, m);
        if (lane == 0) {
            if (bias != nullptr) v += f2f(TF_LDG1(bias + col));
            y[(size_t)(row0 + r) * N + col] = f_from<T>(v);
        }
    }
}

template <typename T, int R, int WARPS>
void launch(const at::Tensor& x, const at::Tensor& w, const void* bp, at::Tensor& y, int M, int K, int N,
            const at::Tensor* w1, at::Tensor* y1) {
    const int N1 = w1 ? (int)w1->size(0) : 0, wide = N1 > N ? N1 : N;
    const dim3 block(WARPS * 32), grid((unsigned)((wide + WARPS - 1) / WARPS), (unsigned)((M + R - 1) / R),
                                       w1 ? 2u : 1u);
    b16_kernel<T, R, WARPS><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const T*>(x.data_ptr()), reinterpret_cast<const T*>(w.data_ptr()),
        reinterpret_cast<const T*>(bp), reinterpret_cast<T*>(y.data_ptr()), M, K, N,
        w1 ? reinterpret_cast<const T*>(w1->data_ptr()) : nullptr, y1 ? reinterpret_cast<T*>(y1->data_ptr()) : nullptr,
        N1);
}

template <typename T>
void by_rows(const at::Tensor& x, const at::Tensor& w, const void* bp, at::Tensor& y, int M, int K, int N,
             const at::Tensor* w1 = nullptr, at::Tensor* y1 = nullptr) {
    if (M >= 512) launch<T, 16, 16>(x, w, bp, y, M, K, N, w1, y1);    // rows a warp, warps a block: never a row's bits
    else if (M >= 64) launch<T, 4, 8>(x, w, bp, y, M, K, N, w1, y1);
    else launch<T, 1, 4>(x, w, bp, y, M, K, N, w1, y1);
}


#if !defined(TF_ROCM)   // ldmatrix / cp.async / mma.sync PTX: NVIDIA only; ROCm prompt rows take b16_linear (b16.py)
// Prompt rows on the bf16 mma: BM x 64 tiles of 2 x 2 warps, 64 inputs a stage, one fp32 chain over K a row, so a
// row's bits never depend on its chunk or the tile height (they differ from the one-row kernel's, as prompts' do).
constexpr int PBN = 64, PST = 3, PROW = 128;

template <int BM>
constexpr int pstage() { return (BM + PBN) * PROW; }

__device__ __forceinline__ uint32_t sm(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sm(p)));
}

__device__ __forceinline__ void cpz(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sm(dst)), "l"(src), "r"(ok ? 16 : 0));
}

__device__ __forceinline__ int pswz(int r, int c) { return r * PROW + ((c ^ (r & 7)) << 4); }

template <int BM>
__global__ void __launch_bounds__(128) b16_prompt_kernel(const __nv_bfloat16* __restrict__ x,
                                                         const __nv_bfloat16* __restrict__ w0,
                                                         __nv_bfloat16* __restrict__ y0, int M, int K, int N0,
                                                         const __nv_bfloat16* __restrict__ w1,
                                                         __nv_bfloat16* __restrict__ y1, int N1) {
    const __nv_bfloat16* w = blockIdx.z ? w1 : w0;         // grid z 2: a second weight of the same rows
    __nv_bfloat16* y = blockIdx.z ? y1 : y0;
    const int N = blockIdx.z ? N1 : N0;
    constexpr int MI = BM / 32, STAGE = pstage<BM>();                     // m16 tiles a warp
    extern __shared__ __align__(128) unsigned char pbuf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 1, wn = warp & 1;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * PBN, KT = K / 64;
    auto load = [&](int s, int kt) {
        unsigned char* p = pbuf + s * STAGE;
        for (int c = tid; c < (BM + PBN) * 8; c += 128) {
            const int r = c >> 3, ch = c & 7;
            const bool wrow = r >= BM;
            const int src_row = wrow ? n0 + r - BM : m0 + r, lim = wrow ? N : M;
            const __nv_bfloat16* base = wrow ? w : x;
            cpz(p + pswz(r, ch), base + (size_t)(src_row < lim ? src_row : 0) * K + kt * 64 + ch * 8, src_row < lim);
        }
    };
    float acc[MI][4][4] = {};
    for (int s = 0; s < PST - 1; ++s) {
        if (s < KT) load(s, s);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int kt = 0; kt < KT; ++kt) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(PST - 2));
        __syncthreads();
        if (kt + PST - 1 < KT) load((kt + PST - 1) % PST, kt + PST - 1);
        asm volatile("cp.async.commit_group;\n" ::);
        const unsigned char* p = pbuf + (kt % PST) * STAGE;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            uint32_t a[MI][4], b[2][4];
#pragma unroll
            for (int i = 0; i < MI; ++i)
                ldsm4(a[i], p + pswz(wm * (BM / 2) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8,
                                     ks * 2 + (lane >> 4)));
#pragma unroll
            for (int j = 0; j < 2; ++j)
                ldsm4(b[j], p + pswz(BM + wn * 32 + j * 16 + (lane & 7) + ((lane >> 4) << 3),
                                     ks * 2 + ((lane >> 3) & 1)));
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, "
                                 "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                                 : "r"(a[i][0]), "r"(a[i][1]), "r"(a[i][2]), "r"(a[i][3]), "r"(b[j >> 1][(j & 1) * 2]),
                                   "r"(b[j >> 1][(j & 1) * 2 + 1]));
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / 2) + i * 16 + (lane >> 2) + h * 8;
                const int col = n0 + wn * 32 + j * 8 + (lane & 3) * 2;
                if (row >= M) continue;
                if (col < N) y[(size_t)row * N + col] = __float2bfloat16_rn(acc[i][j][2 * h]);
                if (col + 1 < N) y[(size_t)row * N + col + 1] = __float2bfloat16_rn(acc[i][j][2 * h + 1]);
            }
}

template <int BM>
void prompt_launch(const at::Tensor& x, const at::Tensor& w, at::Tensor& y, int M, int K, int N,
                   const at::Tensor* w1, at::Tensor* y1) {
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(b16_prompt_kernel<BM>, cudaFuncAttributeMaxDynamicSharedMemorySize, PST * pstage<BM>());
        configured = true;
    }
    const int N1 = w1 ? (int)w1->size(0) : 0, wide = N1 > N ? N1 : N;
    const dim3 grid((unsigned)((wide + PBN - 1) / PBN), (unsigned)((M + BM - 1) / BM), w1 ? 2u : 1u);
    b16_prompt_kernel<BM><<<grid, 128, PST * pstage<BM>(), at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), M, K, N,
        w1 ? reinterpret_cast<const __nv_bfloat16*>(w1->data_ptr()) : nullptr,
        y1 ? reinterpret_cast<__nv_bfloat16*>(y1->data_ptr()) : nullptr, N1);
}

#endif  // !TF_ROCM

}  // namespace

static void pair_in(const at::Tensor& x, const at::Tensor& w) {
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.dim() == 2 && w.scalar_type() == x.scalar_type() &&
                w.size(1) == x.size(1) && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
                "w: contiguous 16-byte aligned (N, K) of x's dtype");
}

at::Tensor b16_linear(const at::Tensor& x, const at::Tensor& w, const at::Tensor& bias) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.dim() == 2, "w must be a contiguous 2-d CUDA tensor");
    TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x and w must share a dtype (cast x first)");
    const bool is_half = x.scalar_type() == at::kHalf;
    TORCH_CHECK(is_half || x.scalar_type() == at::kBFloat16, "only fp16 and bf16 weights are supported");
    const int M = (int)x.size(0), K = (int)x.size(1), N = (int)w.size(0);
    TORCH_CHECK((int)w.size(1) == K, "w's K must be x's K");
    TORCH_CHECK(K % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
                "K must be a multiple of 8 and x, w 16-byte aligned");
    at::cuda::CUDAGuard guard(x.device());
    auto y = at::empty({M, N}, x.options());
    const void* bp = nullptr;
    if (bias.defined() && bias.numel()) {
        TORCH_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.numel() == N, "bias must be [N]");
        bp = bias.data_ptr();
    }
    if (is_half) by_rows<half>(x, w, bp, y, M, K, N);
    else by_rows<__nv_bfloat16>(x, w, bp, y, M, K, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// Two weights of the same rows in one launch (the GDN gates b and a): each output ``b16_linear``'s bits.
std::vector<at::Tensor> b16_linear_pair(const at::Tensor& x, const at::Tensor& w0, const at::Tensor& w1) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.size(1) % 8 == 0 &&
                reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf), "x: contiguous fp16/bf16 rows");
    pair_in(x, w0);
    pair_in(x, w1);
    const int M = (int)x.size(0), K = (int)x.size(1);
    at::cuda::CUDAGuard guard(x.device());
    auto y0 = at::empty({M, w0.size(0)}, x.options()), y1 = at::empty({M, w1.size(0)}, x.options());
    if (x.scalar_type() == at::kHalf) by_rows<half>(x, w0, nullptr, y0, M, K, (int)w0.size(0), &w1, &y1);
    else by_rows<__nv_bfloat16>(x, w0, nullptr, y0, M, K, (int)w0.size(0), &w1, &y1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y0, y1};
}

#if !defined(TF_ROCM)
static int prompt_bm(int64_t bm, int M, int N) {
    if (bm) return (int)bm;
    const long long want = 2LL * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const long long cols = (N + PBN - 1) / PBN;
    int b = 128;
    while (b > 32 && cols * ((M + b - 1) / b) < want) b /= 2;
    return b;
}

// Prompt rows (bf16 x, bf16 w): the mma kernel above; K a multiple of 64. ``bm`` 0 picks the tile height by
// blocks a GPU (narrow weights, like the GDN gates, take short tiles); a row's bits never depend on it.
at::Tensor b16_prompt(const at::Tensor& x, const at::Tensor& w, int64_t bm) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 2,
                "x: contiguous (M, K) bf16");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kBFloat16 && w.dim() == 2 &&
                w.size(1) == x.size(1) && x.size(1) % 64 == 0, "w: contiguous (N, K) bf16, K a multiple of 64");
    const int M = (int)x.size(0), K = (int)x.size(1), N = (int)w.size(0);
    at::cuda::CUDAGuard guard(x.device());
    auto y = at::empty({M, N}, x.options());
    const int b = prompt_bm(bm, M, N);
    if (b == 32) prompt_launch<32>(x, w, y, M, K, N, nullptr, nullptr);
    else if (b == 64) prompt_launch<64>(x, w, y, M, K, N, nullptr, nullptr);
    else prompt_launch<128>(x, w, y, M, K, N, nullptr, nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// Two weights of the same prompt rows in one launch: each output ``b16_prompt``'s bits.
std::vector<at::Tensor> b16_prompt_pair(const at::Tensor& x, const at::Tensor& w0, const at::Tensor& w1, int64_t bm) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 &&
                x.size(1) % 64 == 0, "x: contiguous (M, K) bf16, K a multiple of 64");
    pair_in(x, w0);
    pair_in(x, w1);
    const int M = (int)x.size(0), K = (int)x.size(1), N = (int)w0.size(0);
    at::cuda::CUDAGuard guard(x.device());
    auto y0 = at::empty({M, N}, x.options()), y1 = at::empty({M, w1.size(0)}, x.options());
    const int b = prompt_bm(bm, M, 2 * (N > w1.size(0) ? N : (int)w1.size(0)));
    if (b == 32) prompt_launch<32>(x, w0, y0, M, K, N, &w1, &y1);
    else if (b == 64) prompt_launch<64>(x, w0, y0, M, K, N, &w1, &y1);
    else prompt_launch<128>(x, w0, y0, M, K, N, &w1, &y1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y0, y1};
}
#else
at::Tensor b16_prompt(const at::Tensor&, const at::Tensor&, int64_t) {
    TORCH_CHECK(false, "b16_prompt is NVIDIA-only (PTX mma); on ROCm prompt rows go through b16_linear");
}

std::vector<at::Tensor> b16_prompt_pair(const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t) {
    TORCH_CHECK(false, "b16_prompt_pair is NVIDIA-only (PTX mma); on ROCm prompt rows go through b16_linear_pair");
}
#endif  // !TF_ROCM
