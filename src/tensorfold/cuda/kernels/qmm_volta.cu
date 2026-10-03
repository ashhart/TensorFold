// Volta (sm_70) matmuls for the 4-bit affine checkpoint: fp16 tensor cores (mma.m8n8k4, the only shape Volta has)
// with fp32 sums, for decode's few rows; one cuBLASLt GEMM over a dense fp16 copy of each weight for prompt chunks.
//
// Rows. A bf16 row is first scaled by a power of two so its largest value lands in [2^14, 2^15): every bf16 value
// then converts to fp16 exactly (8 significant bits fit in 11, and nothing in the row underflows), and the row's
// scale comes back in the epilogue. ``prep`` keeps rows row-major; ``prep884`` writes them in the decode kernel's
// fragment order [M/8][K/8][8 rows][8 inputs], so one k-step of an 8-row tile is one 128-byte line.
//
// Weights. A column's 64-input group expands to fp16(q * s + b): the nibble is placed in an fp16 mantissa, 1024 is
// subtracted, and one fma applies the group's bf16 scale and bias (exact in fp16 for their normal range). The decode
// kernel reads the ``Tiled`` layout [N/32][K/64][32 columns][8 words], a warp's 32 columns of one group as 1 KB
// contiguous; the stored MLX layout (column runs of K/8 words) works too, less coalesced.
//
// Bits. The decode kernel holds one column per thread and walks the row's K in group order with fp32 sums, the K
// split a function of the weight's shape (``split_k884``) and its slices added in slice order, so a row's result
// never depends on how many rows share its launch. Prompt rows take one cuBLASLt algorithm per weight shape, pinned
// with split-K off, so their chain over K is the same at every row count too; it is a different chain from decode's.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <map>
#include <mutex>
#include <tuple>

namespace {

constexpr int THREADS = 128;    // a decode block: 4 warps
constexpr int COLS = 128;       // weight columns a decode block covers: 4 warps x 4 quadpairs x 8

__device__ __forceinline__ float bf2f(unsigned short v) { return __uint_as_float(static_cast<unsigned>(v) << 16); }

// ---- rows -------------------------------------------------------------------------------------------------------

// The block's largest value, to every thread (``red`` holds one float per warp).
__device__ __forceinline__ float block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    v = red[0];
    for (int w = 1; w < (int)(blockDim.x >> 5); ++w) v = fmaxf(v, red[w]);
    return v;
}

// The exponent the row is scaled by: 2^-e puts its largest value in [2^14, 2^15); an empty or non-finite row keeps 1.
__device__ __forceinline__ int row_exponent(float amax) { return amax > 0.f && isfinite(amax) ? ilogbf(amax) - 14 : 0; }

// Where input k of a row goes: row-major, or the fragment order [M/8][K/8][8 rows][8 inputs].
template <bool FRAG>
__device__ __forceinline__ long row_base(int row, int K) {
    return FRAG ? (long)(row >> 3) * (K / 8) * 64 + (row & 7) * 8 : (long)row * K;
}
template <bool FRAG>
__device__ __forceinline__ long piece_at(long base, int c) { return base + (long)c * (FRAG ? 64 : 8); }   // 8 inputs

// One row a block: 512 threads, five 16-byte pieces each held in registers, so the row is read once (K <= 20480).
constexpr int PREP_THREADS = 512, PREP_PIECES = 5;
template <bool FRAG>
__global__ void __launch_bounds__(PREP_THREADS) prep_kernel(const unsigned short* __restrict__ x, long ldx, int M,
                                                            int K, __half* __restrict__ out, float* __restrict__ rs) {
    const int row = blockIdx.x, C = K / 8;
    const long base = row_base<FRAG>(row, K);
    if (row >= M) {                                          // FRAG's padded rows read as zeros
        for (int c = threadIdx.x; c < C; c += PREP_THREADS) *reinterpret_cast<uint4*>(out + piece_at<FRAG>(base, c)) = make_uint4(0, 0, 0, 0);
        return;
    }
    uint4 v[PREP_PIECES];
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < PREP_PIECES; ++j) {
        const int c = threadIdx.x + j * PREP_THREADS;
        v[j] = c < C ? *reinterpret_cast<const uint4*>(x + row * ldx + (long)c * 8) : make_uint4(0, 0, 0, 0);
        const unsigned w4[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
#pragma unroll
        for (int h = 0; h < 4; ++h)
            amax = fmaxf(amax, fmaxf(fabsf(bf2f(w4[h] & 0xffff)), fabsf(bf2f(w4[h] >> 16))));
    }
    __shared__ float red[PREP_THREADS / 32];
    const int e = row_exponent(block_max(amax, red));
#pragma unroll
    for (int j = 0; j < PREP_PIECES; ++j) {
        const int c = threadIdx.x + j * PREP_THREADS;
        if (c >= C) continue;
        const unsigned w4[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
        unsigned o4[4];
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const __half lo = __float2half_rn(ldexpf(bf2f(w4[h] & 0xffff), -e));
            const __half hi = __float2half_rn(ldexpf(bf2f(w4[h] >> 16), -e));
            o4[h] = (unsigned)__half_as_ushort(lo) | ((unsigned)__half_as_ushort(hi) << 16);
        }
        *reinterpret_cast<uint4*>(out + piece_at<FRAG>(base, c)) = make_uint4(o4[0], o4[1], o4[2], o4[3]);
    }
    if (threadIdx.x == 0) rs[row] = ldexpf(1.f, e);
}

// The same values one element at a time: rows wider than the fast kernel holds, or views it cannot load 16 bytes from.
template <bool FRAG>
__global__ void prep_any_kernel(const unsigned short* __restrict__ x, long ldx, int M, int K, __half* __restrict__ out,
                                float* __restrict__ rs) {
    const int row = blockIdx.x;
    const long base = row_base<FRAG>(row, K);
    auto at = [&](int k) { return FRAG ? base + (long)(k >> 3) * 64 + (k & 7) : base + k; };
    if (row >= M) {
        for (int k = threadIdx.x; k < K; k += blockDim.x) out[at(k)] = __float2half_rn(0.f);
        return;
    }
    const unsigned short* xr = x + row * ldx;
    float amax = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) amax = fmaxf(amax, fabsf(bf2f(xr[k])));
    __shared__ float red[32];
    const int e = row_exponent(block_max(amax, red));
    for (int k = threadIdx.x; k < K; k += blockDim.x) out[at(k)] = __float2half_rn(ldexpf(bf2f(xr[k]), -e));
    if (threadIdx.x == 0) rs[row] = ldexpf(1.f, e);
}

bool fast_prep_fits(const torch::Tensor& x) {
    return x.size(1) / 8 <= PREP_THREADS * PREP_PIECES && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0
        && x.stride(0) % 8 == 0;
}

// ---- weights ----------------------------------------------------------------------------------------------------

// Eight nibbles of one word -> eight fp16 values q * s + b in input order, two per instruction.
__device__ __forceinline__ int4 dequant8(unsigned w, __half2 s2, __half2 b2) {
    const unsigned magic = 0x64006400u;                    // fp16 1024.0 in each half
    const __half2 k1024 = __halves2half2(__ushort_as_half(0x6400), __ushort_as_half(0x6400));
    __half2 v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {                           // v[j] = (q_j, q_{j+4})
        unsigned u = ((w >> (4 * j)) & 0x000F000Fu) | magic;
        v[j] = __hsub2(*reinterpret_cast<__half2*>(&u), k1024);
    }
    __half2 o[4] = {__lows2half2(v[0], v[1]), __lows2half2(v[2], v[3]),
                    __highs2half2(v[0], v[1]), __highs2half2(v[2], v[3])};
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __hfma2(o[j], s2, b2);
    return *reinterpret_cast<int4*>(o);
}

// A group's bf16 scale and bias as fp16 pairs (bf16 -> fp32 -> fp16: exact for their normal range).
__device__ __forceinline__ void group_factors(unsigned short s, unsigned short b, __half2& s2, __half2& b2) {
    const __half sh = __float2half_rn(bf2f(s)), bh = __float2half_rn(bf2f(b));
    s2 = __halves2half2(sh, sh);
    b2 = __halves2half2(bh, bh);
}

// ---- decode: mma.m8n8k4 from registers, one weight column per thread ---------------------------------------------
//
// Fragment layout, measured on a V100 (closed form, 0 mismatches): quadpair q = (lane >> 2) & 3,
// idx = lane % 4 + 4 * (lane >= 16). A .row: the thread holds A[idx][0..3]. B .col: B[0..3][idx].
// D f32: d[i] sits at row (lane & 1) + 2 * ((i >> 1) & 1) + 4 * (lane >= 16), col (i & 1) + 2 * ((lane >> 1) & 1) + 4 * (i >> 2).
// The four quadpairs share A (the same 8 rows) and each owns 8 weight columns, so a warp covers 32 columns.

__device__ __forceinline__ void mma884(float (&d)[8], unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// MT 8-row tiles a block (1, 2 or 4); TILED: the ``Tiled`` layout, else the stored one; FRAG: rows from ``prep884``.
template <int MT, bool TILED, bool FRAG>
__global__ void __launch_bounds__(THREADS) qmm884_kernel(
        const __half* __restrict__ X, const float* __restrict__ RS, const int* __restrict__ W,
        const unsigned short* __restrict__ S, const unsigned short* __restrict__ B, void* __restrict__ OUT,
        float* __restrict__ PART, int M, int N, int K, int gper, int sk, bool f32) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int q = (lane >> 2) & 3, idx = (lane & 3) + 4 * (lane >= 16);
    const int s = blockIdx.y, m0 = blockIdx.z * 8 * MT;
    const int g0 = s * gper, g1 = g0 + gper;
    const int KG = K / 64, K8 = K / 8;
    const long wstride = TILED ? 32 * 8 : 8;                    // words between consecutive groups of one column
    const long sstride = TILED ? 32 : 1;
    const int cbase = blockIdx.x * COLS + warp * 32 + q * 8;
    const int col = cbase + idx;
    const bool nok = col < N;
    const int c32 = nok ? col : 0;
    const int4* wp = reinterpret_cast<const int4*>(TILED ? W + ((long)(c32 >> 5) * KG * 32 + (c32 & 31)) * 8 : W + (long)c32 * K8);
    const unsigned short* sp = TILED ? S + (long)(c32 >> 5) * KG * 32 + (c32 & 31) : S + (long)c32 * KG;
    const unsigned short* bp = TILED ? B + (long)(c32 >> 5) * KG * 32 + (c32 & 31) : B + (long)c32 * KG;

    // FRAG: a warp's eight row loads of one k-step are one 128-byte line (rows past M are zeros); row-major: one
    // row's eight k-steps share a line (rows past M are skipped).
    const __half* xr[MT];
    bool rok[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (FRAG) {
            rok[t] = m0 + t * 8 < M;
            xr[t] = X + ((long)(rok[t] ? (m0 >> 3) + t : 0) * K8) * 64 + idx * 8;
        } else {
            const int r = m0 + t * 8 + idx;
            rok[t] = r < M;
            xr[t] = X + (long)(rok[t] ? r : 0) * K;
        }
    }
    float acc[MT][8];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[t][i] = 0.f;

    // the column's current group in registers, the next one's loads in flight
    int4 c0 = make_int4(0, 0, 0, 0), c1 = c0;
    unsigned short cs = 0, cb = 0;
    if (nok) {
        c0 = __ldcs(wp + g0 * wstride / 4); c1 = __ldcs(wp + g0 * wstride / 4 + 1);
        cs = __ldg(sp + g0 * sstride); cb = __ldg(bp + g0 * sstride);
    }
    for (int g = g0; g < g1; ++g) {
        int4 n0 = make_int4(0, 0, 0, 0), n1 = n0;
        unsigned short ns = 0, nb = 0;
        if (nok && g + 1 < g1) {
            n0 = __ldcs(wp + (g + 1) * wstride / 4); n1 = __ldcs(wp + (g + 1) * wstride / 4 + 1);
            ns = __ldg(sp + (g + 1) * sstride); nb = __ldg(bp + (g + 1) * sstride);
        }
        __half2 s2, b2;
        group_factors(cs, cb, s2, b2);
        const unsigned words[8] = {(unsigned)c0.x, (unsigned)c0.y, (unsigned)c0.z, (unsigned)c0.w,
                                   (unsigned)c1.x, (unsigned)c1.y, (unsigned)c1.z, (unsigned)c1.w};
#pragma unroll
        for (int wi = 0; wi < 8; ++wi) {
            const int4 o = dequant8(words[wi], s2, b2);         // (k0,k1) (k2,k3) (k4,k5) (k6,k7)
            const int k = g * 64 + wi * 8;
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                const uint4 a = rok[t] ? __ldg(reinterpret_cast<const uint4*>(xr[t] + (FRAG ? (long)(k >> 3) * 64 : (long)k)))
                                       : make_uint4(0, 0, 0, 0);
                mma884(acc[t], a.x, a.y, (unsigned)o.x, (unsigned)o.y);
                mma884(acc[t], a.z, a.w, (unsigned)o.z, (unsigned)o.w);
            }
        }
        c0 = n0; c1 = n1; cs = ns; cb = nb;
    }

#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int m = m0 + t * 8 + (lane & 1) + 2 * ((i >> 1) & 1) + 4 * (lane >= 16);
            const int n = cbase + (i & 1) + 2 * ((lane >> 1) & 1) + 4 * (i >> 2);
            if (m >= M || n >= N) continue;
            const float v = acc[t][i] * RS[m];
            if (sk > 1) PART[((long)s * M + m) * N + n] = v;
            else if (f32) static_cast<float*>(OUT)[(long)m * N + n] = v;
            else static_cast<__nv_bfloat16*>(OUT)[(long)m * N + n] = __float2bfloat16_rn(v);
        }
}

// Split-K slices summed in slice order (the same order whatever the row count).
__global__ void reduce_kernel(const float* __restrict__ part, void* __restrict__ out, long total, int sk, bool f32) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= total) return;
    float a = part[i];
    for (int s = 1; s < sk; ++s) a += part[s * total + i];
    if (f32) static_cast<float*>(out)[i] = a;
    else static_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(a);
}

// ---- prompts: a dense fp16 copy of the weight, then cuBLASLt --------------------------------------------------------

// A Tiled weight -> dense fp16 (N, K), each value the fp16(q * s + b) the decode kernel expands. One thread a word.
__global__ void dequant_kernel(const unsigned* __restrict__ w, const unsigned short* __restrict__ s,
                               const unsigned short* __restrict__ b, __half* __restrict__ out, int N, int K) {
    const int kw = K / 8;                                                // words a row
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;        // n * kw + word
    if (idx >= (long)N * kw) return;
    const long n = idx / kw;
    const int word = idx % kw, g = word >> 3, j = word & 7, kg = K / 64;
    const long cell = ((n >> 5) * kg + g) * 32 + (n & 31);
    __half2 s2, b2;
    group_factors(s[cell], b[cell], s2, b2);
    reinterpret_cast<int4*>(out)[idx] = dequant8(w[cell * 8 + j], s2, b2);
}

// y (M, N) fp32 sums of the scaled rows -> y * rs[row], as bf16 or as fp32 in place.
__global__ void unscale_kernel(const float* __restrict__ y, const float* __restrict__ rs, void* __restrict__ out,
                               long N, long total4, bool f32) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total4) return;
    const float r = rs[(i * 4) / N];
    float4 v = reinterpret_cast<const float4*>(y)[i];
    v.x *= r; v.y *= r; v.z *= r; v.w *= r;
    if (f32) { reinterpret_cast<float4*>(out)[i] = v; return; }
    __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
    uint2 u = {*reinterpret_cast<unsigned*>(&lo), *reinterpret_cast<unsigned*>(&hi)};
    reinterpret_cast<uint2*>(out)[i] = u;
}

#define LT_CHECK(call) \
    do { cublasStatus_t st_ = (call); TORCH_CHECK(st_ == CUBLAS_STATUS_SUCCESS, #call " failed: ", (int)st_); } while (0)

struct LtLayout {                                              // a matrix layout freed on every exit path
    cublasLtMatrixLayout_t h = nullptr;
    LtLayout(cudaDataType type, long rows, long cols, long ld) { LT_CHECK(cublasLtMatrixLayoutCreate(&h, type, rows, cols, ld)); }
    ~LtLayout() { if (h) cublasLtMatrixLayoutDestroy(h); }
    LtLayout(const LtLayout&) = delete;
    LtLayout& operator=(const LtLayout&) = delete;
};

struct LtPreference {
    cublasLtMatmulPreference_t h = nullptr;
    LtPreference() { LT_CHECK(cublasLtMatmulPreferenceCreate(&h)); }
    ~LtPreference() { if (h) cublasLtMatmulPreferenceDestroy(h); }
    LtPreference(const LtPreference&) = delete;
    LtPreference& operator=(const LtPreference&) = delete;
};

// One algorithm a weight shape: cuBLASLt's first choice for 4096 rows that uses no split K, pinned with split K off
// and no workspace, so the kernel (and a row's chain over K) is the same at every row count.
struct LtPlan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t weight = nullptr;                   // W (N, K) row-major, read transposed
    cublasLtMatmulAlgo_t algo{};
};
constexpr int PLAN_ROWS = 4096;                                // the row count the algorithm is chosen for

std::map<std::tuple<int, long, long>, LtPlan> lt_plans;        // by (device, N, K), kept for the process
std::mutex lt_lock;

const LtPlan& lt_plan(cublasLtHandle_t handle, int device, long N, long K) {
    std::lock_guard<std::mutex> guard(lt_lock);
    const auto key = std::make_tuple(device, N, K);
    auto it = lt_plans.find(key);
    if (it != lt_plans.end()) return it->second;
    LtPlan p;
    LT_CHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t trans = CUBLAS_OP_T;
    LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    LT_CHECK(cublasLtMatrixLayoutCreate(&p.weight, CUDA_R_16F, K, N, K));
    const LtLayout rows(CUDA_R_16F, K, PLAN_ROWS, K), sums(CUDA_R_32F, N, PLAN_ROWS, N);
    const LtPreference pref;
    const size_t workspace = 0;
    LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref.h, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
                                                  sizeof(workspace)));
    cublasLtMatmulHeuristicResult_t found[8];
    int count = 0;
    LT_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, p.op, p.weight, rows.h, sums.h, sums.h, pref.h, 8, found, &count));
    TORCH_CHECK(count > 0, "qmm_volta: cuBLASLt offers no algorithm for the prompt GEMM of a ", N, "x", K, " weight");
    int pick = 0;
    for (int i = 0; i < count; ++i) {                          // the best one that already runs without split K
        uint32_t splits = 0;
        size_t written = 0;
        cublasLtMatmulAlgoConfigGetAttribute(&found[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splits, sizeof(splits), &written);
        if (splits <= 1) { pick = i; break; }
    }
    p.algo = found[pick].algo;
    const uint32_t one = 1, none = CUBLASLT_REDUCTION_SCHEME_NONE;
    LT_CHECK(cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &one, sizeof(one)));
    LT_CHECK(cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &none, sizeof(none)));
    return lt_plans.emplace(key, p).first->second;
}

}  // namespace

// ---- host ---------------------------------------------------------------------------------------------------------

std::vector<torch::Tensor> prep(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1, "prep: x must be (M, K) bf16 rows");
    const int M = x.size(0), K = x.size(1);
    auto x16 = torch::empty({M, K}, x.options().dtype(at::kHalf));
    auto rs = torch::empty({M}, x.options().dtype(at::kFloat));
    if (M == 0) return {x16, rs};
    const auto xp = reinterpret_cast<const unsigned short*>(x.data_ptr<at::BFloat16>());
    const auto op = reinterpret_cast<__half*>(x16.data_ptr<at::Half>());
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (fast_prep_fits(x)) prep_kernel<false><<<M, PREP_THREADS, 0, stream>>>(xp, x.stride(0), M, K, op, rs.data_ptr<float>());
    else prep_any_kernel<false><<<M, 256, 0, stream>>>(xp, x.stride(0), M, K, op, rs.data_ptr<float>());
    return {x16, rs};
}

std::vector<torch::Tensor> prep884(torch::Tensor x, int64_t frag_min) {
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1, "prep884: x must be (M, K) bf16 rows");
    const int M = x.size(0), K = x.size(1);
    TORCH_CHECK(K % 64 == 0, "prep884: K must be a multiple of 64, not ", K);
    if (M < frag_min) return prep(x);                        // a few rows: row-major reads share cache lines
    const int M8 = (M + 7) / 8;
    auto xf = torch::empty({M8, K / 8, 8, 8}, x.options().dtype(at::kHalf));
    auto rs = torch::empty({M}, x.options().dtype(at::kFloat));
    if (M == 0) return {xf, rs};
    const auto xp = reinterpret_cast<const unsigned short*>(x.data_ptr<at::BFloat16>());
    const auto op = reinterpret_cast<__half*>(xf.data_ptr<at::Half>());
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (fast_prep_fits(x)) prep_kernel<true><<<M8 * 8, PREP_THREADS, 0, stream>>>(xp, x.stride(0), M, K, op, rs.data_ptr<float>());
    else prep_any_kernel<true><<<M8 * 8, 256, 0, stream>>>(xp, x.stride(0), M, K, op, rs.data_ptr<float>());
    return {xf, rs};
}

// ``x16``, ``rs`` from ``prep`` (M, K) or ``prep884`` (M/8, K/8, 8, 8); ``w`` the Tiled words (tiled) or the stored
// (N, K/8) words; ``n_cols`` N; ``mt`` 8-row tiles a block; ``sk`` K slices, with ``part`` (sk, M, N) fp32 when > 1.
void qmm884(torch::Tensor x16, torch::Tensor rs, torch::Tensor w, torch::Tensor s, torch::Tensor b, torch::Tensor out,
            c10::optional<torch::Tensor> part, int64_t sk, int64_t mt, bool f32, int64_t n_cols, bool tiled) {
    const bool frag = x16.dim() == 4;
    TORCH_CHECK(frag ? x16.size(2) == 8 && x16.size(3) == 8 : x16.dim() == 2, "qmm884: rows must come from prep or prep884");
    const int M = rs.size(0), K = frag ? x16.size(1) * 8 : x16.size(1), N = n_cols;
    TORCH_CHECK(frag ? x16.size(0) * 8 >= M : x16.size(0) == M, "qmm884: the row tiles do not cover the rows");
    TORCH_CHECK(x16.is_contiguous() && w.is_contiguous() && s.is_contiguous() && b.is_contiguous() && out.is_contiguous(),
                "qmm884: contiguous tensors");
    TORCH_CHECK(K % 64 == 0 && w.numel() >= (long)N * K / 8 && (K / 64) % sk == 0, "qmm884: bad K or K split");
    if (M == 0) return;
    const auto stream = at::cuda::getCurrentCUDAStream();
    float* p = nullptr;
    if (sk > 1) {
        TORCH_CHECK(part.has_value() && part->numel() >= sk * (int64_t)M * N, "qmm884: a K split needs its partial buffer");
        p = part->data_ptr<float>();
    }
    const int gper = K / 64 / sk;
    const auto X = reinterpret_cast<const __half*>(x16.data_ptr<at::Half>());
    const auto S = reinterpret_cast<const unsigned short*>(s.data_ptr<at::BFloat16>());
    const auto Bp = reinterpret_cast<const unsigned short*>(b.data_ptr<at::BFloat16>());
    const dim3 grid((N + COLS - 1) / COLS, sk, (M + 8 * mt - 1) / (8 * mt));
#define QMM884(MTV, TL, FR) qmm884_kernel<MTV, TL, FR><<<grid, THREADS, 0, stream>>>( \
        X, rs.data_ptr<float>(), w.data_ptr<int>(), S, Bp, out.data_ptr(), p, M, N, K, gper, sk, f32)
#define QMM884_TILE(MTV) \
    if (tiled) { if (frag) QMM884(MTV, true, true); else QMM884(MTV, true, false); } \
    else { if (frag) QMM884(MTV, false, true); else QMM884(MTV, false, false); }
    if (mt == 1) { QMM884_TILE(1) } else if (mt == 2) { QMM884_TILE(2) } else if (mt == 4) { QMM884_TILE(4) }
    else TORCH_CHECK(false, "qmm884: mt is 1, 2 or 4, not ", mt);
#undef QMM884_TILE
#undef QMM884
    if (sk > 1) {
        const long total = (long)M * N;
        reduce_kernel<<<(total + 255) / 256, 256, 0, stream>>>(p, out.data_ptr(), total, sk, f32);
    }
}

torch::Tensor dequant(torch::Tensor w, torch::Tensor s, torch::Tensor b, int64_t n) {
    TORCH_CHECK(w.is_contiguous() && s.is_contiguous() && b.is_contiguous() && w.dim() == 4 && w.size(2) == 32,
                "dequant: a Tiled weight");
    const long K = w.size(1) * 64, words = n * (K / 8);
    auto out = torch::empty({n, K}, w.options().dtype(at::kHalf));
    if (words == 0) return out;
    dequant_kernel<<<(words + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const unsigned*>(w.data_ptr<int>()), reinterpret_cast<const unsigned short*>(s.data_ptr<at::BFloat16>()),
        reinterpret_cast<const unsigned short*>(b.data_ptr<at::BFloat16>()), reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        (int)n, (int)K);
    return out;
}

torch::Tensor unscale(torch::Tensor y, torch::Tensor rs, bool f32) {
    TORCH_CHECK(y.is_contiguous() && y.scalar_type() == at::kFloat && y.dim() == 2 && y.size(1) % 4 == 0,
                "unscale: (M, N) fp32 with N a multiple of 4");
    auto out = f32 ? y : torch::empty_like(y, y.options().dtype(at::kBFloat16));
    const long total4 = y.numel() / 4;
    if (total4 == 0) return out;
    unscale_kernel<<<(total4 + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        y.data_ptr<float>(), rs.data_ptr<float>(), out.data_ptr(), y.size(1), total4, f32);
    return out;
}

torch::Tensor gemm(torch::Tensor x16, torch::Tensor w16) {
    TORCH_CHECK(x16.is_contiguous() && w16.is_contiguous() && x16.scalar_type() == at::kHalf && w16.scalar_type() == at::kHalf,
                "gemm: contiguous fp16 rows and weight");
    const long M = x16.size(0), K = x16.size(1), N = w16.size(0);
    TORCH_CHECK(w16.size(1) == K, "gemm: rows of ", K, " inputs against a weight of ", w16.size(1));
    auto out = torch::empty({M, N}, x16.options().dtype(at::kFloat));
    if (M == 0) return out;
    const auto handle = at::cuda::getCurrentCUDABlasLtHandle();
    const LtPlan& p = lt_plan(handle, x16.get_device(), N, K);
    const LtLayout rows(CUDA_R_16F, K, M, K), sums(CUDA_R_32F, N, M, N);
    const float alpha = 1.f, beta = 0.f;
    LT_CHECK(cublasLtMatmul(handle, p.op, &alpha, w16.data_ptr<at::Half>(), p.weight, x16.data_ptr<at::Half>(), rows.h,
                            &beta, out.data_ptr<float>(), sums.h, out.data_ptr<float>(), sums.h, &p.algo, nullptr, 0,
                            at::cuda::getCurrentCUDAStream()));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prep", &prep, "bf16 rows -> exact fp16 copies scaled by a power of two, and the scales");
    m.def("prep884", &prep884, "the same in the decode kernel's fragment order [M/8][K/8][8][8]");
    m.def("qmm884", &qmm884, "decode matmul: mma.m8n8k4 from registers, one weight column a thread");
    m.def("dequant", &dequant, "a Tiled 4-bit weight -> dense fp16 (N, K), the decode kernel's values");
    m.def("gemm", &gemm, "fp16 rows x fp16 (N, K) weight -> fp32 sums, one pinned cuBLASLt algorithm a shape");
    m.def("unscale", &unscale, "fp32 sums of scaled rows -> the rows' sums, bf16 or fp32 in place");
}
