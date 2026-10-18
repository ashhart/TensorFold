// Copy of src/tensorfold/families/deepseek_v41/cuda/x3seg.cu (blob 97123dbc731d): sass_check.sh
#define DSV41_X3SEG_NO_TORCH


// Segment-table EXL3 linears: several 1-128-row projections in ONE launch, each program upstream's linear_kernel's.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#ifndef DSV41_X3SEG_NO_TORCH     // the kernels alone (nvcc compile test, no torch)
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include <algorithm>
#include <vector>

#include "../exl3_decode.cuh"

using namespace tf_exl3;

namespace dsv41_x3seg {  // program p = (nb, split) of the last segment with p0 <= p: the unfused launch's bits

constexpr int MAXSEG = 8;
enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

// -- upstream linear.cu's helpers, verbatim -------------------------------------------------------------------------
__device__ __forceinline__ void load4(const void* p, int dtype, size_t i, float (&v)[4]) {
    if (dtype == F32) {
        const float4 u = *reinterpret_cast<const float4*>(static_cast<const float*>(p) + i);
        v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
    } else if (dtype == BF16) {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p) + i);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    } else {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const half*>(p) + i);
        const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    }
}

__device__ __forceinline__ void store4(void* p, int dtype, size_t i, const float (&v)[4]) {
    if (dtype == F32) {
        *reinterpret_cast<float4*>(static_cast<float*>(p) + i) = make_float4(v[0], v[1], v[2], v[3]);
    } else if (dtype == BF16) {
        __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p) + i) = u;
    } else {
        half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<half*>(p) + i) = u;
    }
}

__device__ __forceinline__ void finish(float (&v)[4], int lane, const half* svh, const half* bias, int col) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = v[j] * HAD_SCALE * __half2float(__ldg(svh + col + j));
        if (bias) v[j] += __half2float(__ldg(bias + col + j));
    }
}

template <int K2>
__host__ __device__ constexpr bool step_shuffled() {
    return K2 == 2 || K2 == 4;
}

template <int K2>
__host__ __device__ constexpr int step_regs() {
    return step_shuffled<K2>() ? tile_words<K2>() / 4 : 8 * lane_words<K2>();
}

template <int K2>
__device__ __forceinline__ void load_step(const uint32_t* step, int lane, uint32_t (&raw)[step_regs<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
#pragma unroll
        for (int c = 0; c < step_regs<K2>(); ++c) raw[c] = __ldg(step + c * 32 + lane);
    } else {
        int word, offset;
        lane_start<K2>(lane, word, offset);
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < LW; ++q) raw[j * LW + q] = __ldg(step + j * TW + (word + q) % TW);
    }
}

template <int K2>
__device__ __forceinline__ void step_lane_words(const uint32_t (&raw)[step_regs<K2>()], int j, int lane, bool prev,
                                                uint32_t (&w)[lane_words<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
        static_assert(LW == 2, "a lane's windows span two words at 1 and 2 bits");
        constexpr int LPW = 8 / K2;
        const uint32_t r = raw[j * TW / 32];
        const int base = (j * TW) % 32;
        const uint32_t own = __shfl_sync(0xffffffffu, r, base + lane / LPW);
        const uint32_t before = __shfl_sync(0xffffffffu, r, base + (lane / LPW + TW - 1) % TW);
        w[0] = prev ? before : own;
        w[1] = prev ? own : 0u;
    } else {
#pragma unroll
        for (int q = 0; q < LW; ++q) w[q] = raw[j * LW + q];
    }
}

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// -- the segment tables (kernel parameters: nothing to copy, graph-capture safe) -------------------------------------
struct RotSeg {
    const void* x;          // [M, K] rows at x + row * ld (fp16, bf16 or fp32)
    const half* suh;        // [K]
    half* xh;               // [M, K] fp16, contiguous
    long long ld;
    int K, x_dtype;
    const void* nw;         // TF_DSV41_RMS_FOLD: x is bf16 un-normed; this segment's input is rmsnorm's row
    int nw_dtype;           //   bf16(fp32((x * rn) * w)), rn as Triton's rmsnorm row computes it (null: off)
    float inv_k, eps;
};
struct RotTable {
    RotSeg s[MAXSEG];
    int n;
};

struct LinSeg {
    const half* xh;         // [M, K] fp16 (the segment's rot_in output)
    const uint32_t* T;      // trellis words, upstream's "strips" layout
    long long stride_k, stride_nb;
    const half* svh;        // [N]
    void* y;                // output element (row, col) at y + row * ld + col
    float* Z;               // [SK, M, N] fp32 when SK > 1
    int* counters;          // [8 * N / 128], left zero
    int K, N, SK, wk, p0, ld;
};
struct LinTable {
    LinSeg s[MAXSEG];
    int n;
};

// Triton's rmsnorm rn for row `xr` (bf16, K) by 128 threads, in its sm_120 / sm_121 PTX's float64 order
__device__ __forceinline__ double row_rn(const __nv_bfloat16* xr, int K, float inv_k, float eps) {
    __shared__ double ws[4];
    const int t = threadIdx.x;
    double A[4] = {0.0, 0.0, 0.0, 0.0};
    for (int c = 0; c < K; c += 512) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int e = c + t + 128 * i;
            const double v = e < K ? static_cast<double>(__bfloat162float(xr[e])) : 0.0;
            A[i] = __dadd_rn(A[i], __dmul_rn(v, v));
        }
    }
    double s = __dadd_rn(__dadd_rn(__dadd_rn(A[0], A[1]), A[2]), A[3]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s = __dadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
    if ((t & 31) == 0) ws[t >> 5] = s;
    __syncthreads();
    const double w0 = __dadd_rn(ws[0], ws[2]), w1 = __dadd_rn(ws[1], ws[3]);
    const double ss = __dadd_rn(w0, w1);
    const double a = __dadd_rn(__dmul_rn(ss, static_cast<double>(inv_k)), static_cast<double>(eps));
    double rn;
    asm("rsqrt.approx.f64 %0, %1;" : "=d"(rn) : "d"(a));
    return rn;
}

__global__ void __launch_bounds__(128) seg_rot_in_kernel(const RotTable tab) {  // upstream rot_in, z = segment
    pdl_wait();                                               // x belongs to the kernels before (every CTA waits)
    pdl_trigger();
    RotSeg s = tab.s[0];
#pragma unroll
    for (int i = 1; i < MAXSEG; ++i)
        if (i == (int)blockIdx.z) s = tab.s[i];
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    double rn = 0.0;
    if (s.nw != nullptr)                                      // the whole CTA (uniform: its segment and row)
        rn = row_rn(static_cast<const __nv_bfloat16*>(s.x) + (size_t)row * s.ld, s.K, s.inv_k, s.eps);
    if (blk * 128 >= s.K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], sc[4];
    load4(s.x, s.x_dtype, (size_t)row * s.ld + k, v);
    if (s.nw != nullptr) {                                    // _rms_row's y = (x * rn) * w -> fp32 -> bf16
        float w[4];
        load4(s.nw, s.nw_dtype, k, w);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const double y = __dmul_rn(__dmul_rn(rn, static_cast<double>(v[j])), static_cast<double>(w[j]));
            v[j] = __bfloat162float(__float2bfloat16_rn(__double2float_rn(y)));
        }
    }
    load4(s.suh, F16, k, sc);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= sc[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4(s.xh, F16, (size_t)row * s.K + k, v);
}

// upstream's linear_kernel<K2, CB, WK> at its segment's (nb, split, NB); MINB bounds registers only, never values
template <int K2, int CB, int WK, int MINB>
__global__ void __launch_bounds__(WK * 32, MINB) seg_linear_kernel(const LinTable tab, int y_dtype, int M) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats
    __shared__ int last;
    const int RH = min(M, 8);

    LinSeg s = tab.s[0];
#pragma unroll
    for (int i = 1; i < MAXSEG; ++i)
        if (i < tab.n && (int)blockIdx.x >= tab.s[i].p0) s = tab.s[i];
    const half* __restrict__ xh = s.xh;
    const half* __restrict__ svh = s.svh;
    void* __restrict__ y = s.y;
    float* __restrict__ Z = s.Z;
    int* __restrict__ counters = s.counters;
    const int K = s.K, N = s.N, SK = s.SK, wk = s.wk, ld = s.ld;

    const int NB = N >> 7, q = (int)blockIdx.x - s.p0, nb = q % NB, split = q / NB;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const bool on = warp < wk;                                // WK 4 segment, 8-warp launch: warps 4-7 idle, same sums
    const int per_warp = (K >> 4) / SK / wk;
    const int kt0 = split * (per_warp * wk) + warp * per_warp;
    const int steps = on ? per_warp : 0;
    const uint32_t* tiles = s.T + nb * s.stride_nb;
    const int col0 = nb * 128;
    bool prev = false;
    if constexpr (step_shuffled<K2>()) {
        int word, offset;
        lane_start<K2>(lane, word, offset);
        prev = word != lane / (8 / K2);
    }

    for (int m0 = 0, pass = 0; m0 < M; m0 += 16, ++pass) {
        const int R = min(16, M - m0);

        float acc[8][2][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

        const int r0 = m0 + (g < R ? g : R - 1), r1 = m0 + (g + 8 < R ? g + 8 : R - 1);
        const half* x0 = xh + (size_t)r0 * K;
        const half* x1 = xh + (size_t)r1 * K;
        const uint32_t* tile = tiles + (size_t)kt0 * s.stride_k;
        constexpr bool PF = K2 <= 12;
        constexpr int SR = step_regs<K2>();
        uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
        if constexpr (PF) {
            if (steps > 0) load_step<K2>(tile, lane, cur);
        }
        if (pass == 0) pdl_wait();                            // weights above, xh / Z / counters below
#pragma unroll 1
        for (int i = 0; i < steps; ++i) {
            const int kt = kt0 + i;
            uint32_t a[4];
            a[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
            a[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t));
            a[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
            a[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t + 8));
            if constexpr (PF) {
                if (i + 1 < steps) load_step<K2>(tile + (size_t)(i + 1) * s.stride_k, lane, nxt);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[LW];
                if constexpr (PF) step_lane_words<K2>(cur, j, lane, prev, w);
                else ldg_lane_words<K2>(tile + (size_t)i * s.stride_k + j * TW, lane, w);
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
                mma16816(acc[j][0], a, b0);
                mma16816(acc[j][1], a, b1);
            }
            if constexpr (PF) {
#pragma unroll
                for (int qq = 0; qq < SR; ++qq) cur[qq] = nxt[qq];
            }
        }

        if (m0 + 16 >= M) pdl_trigger();                     // the last pass's weights are read

        // the warps' sums, added in warp order, rows 0-7 of the pass and then rows 8-15
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = min(R - rlo, 8);
            __syncthreads();
            if (on && g < RH) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * 128 + col) =
                            rlo ? make_float2(acc[i][h][2], acc[i][h][3]) : make_float2(acc[i][h][0], acc[i][h][1]);
                    }
            }
            __syncthreads();

            if (SK == 1) {
                for (int r = warp; r < rn; r += WK) {
                    float v[4];
                    const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
                    for (int w = 1; w < wk; ++w) {
                        const float4 qv = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                        v[0] += qv.x; v[1] += qv.y; v[2] += qv.z; v[3] += qv.w;
                    }
                    finish(v, lane, svh, nullptr, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + rlo + r) * ld + col0 + 4 * lane, v);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 sm = *reinterpret_cast<const float4*>(red + r * 128 + c);
                    for (int w = 1; w < wk; ++w) {
                        const float4 qv = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                        sm.x += qv.x; sm.y += qv.y; sm.z += qv.z; sm.w += qv.w;
                    }
                    *reinterpret_cast<float4*>(Z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = sm;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(counters + pass * NB + nb, 1) == SK - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < R; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float4 sm = __ldcg(reinterpret_cast<const float4*>(Z + at));
                    for (int qq = 1; qq < SK; ++qq) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)qq * M * N + at));
                        sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w;
                    }
                    float v[4] = {sm.x, sm.y, sm.z, sm.w};
                    finish(v, lane, svh, nullptr, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + r) * ld + col0 + 4 * lane, v);
                }
                if (threadIdx.x == 0) counters[pass * NB + nb] = 0;
            }
        }
        __syncthreads();
    }
}

#ifndef DSV41_X3SEG_NO_TORCH
int dtype_of(const at::Tensor& t) {
    return t.scalar_type() == at::kFloat ? F32 : t.scalar_type() == at::kBFloat16 ? BF16 : F16;
}
#endif

}  // namespace dsv41_x3seg

#ifndef DSV41_X3SEG_NO_TORCH

using namespace dsv41_x3seg;

namespace {
// a launch with the PDL attribute when pdl (cudaLaunchKernelEx: graph-capturable, the attribute becomes the edge)
template <typename K, typename... A>
void launch_ex(K kernel, dim3 grid, dim3 block, int smem, bool pdl, A... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = (size_t)smem;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...));
}
}  // namespace

void dsv41_x3seg_rot_in_cuda(const std::vector<at::Tensor>& xs, const std::vector<at::Tensor>& suh, at::Tensor& xh,
                             const std::vector<int64_t>& xh_off, bool pdl, const std::vector<at::Tensor>& nw,
                             double eps) {
    RotTable tab{};
    tab.n = (int)xs.size();
    const int M = (int)xs[0].size(0);
    int kmax = 0;
    for (int i = 0; i < tab.n; ++i) {
        const int K = (int)xs[i].size(1);
        tab.s[i] = RotSeg{xs[i].data_ptr(), reinterpret_cast<const half*>(suh[i].data_ptr()),
                          reinterpret_cast<half*>(xh.data_ptr()) + xh_off[i], (long long)xs[i].stride(0), K,
                          dtype_of(xs[i]), nullptr, 0, 0.f, 0.f};
        if (i < (int)nw.size() && nw[i].numel()) {             // fp32 1 / K and eps: Triton's float arguments
            TORCH_CHECK(xs[i].scalar_type() == at::kBFloat16 && nw[i].numel() == K && nw[i].is_contiguous(),
                        "x3seg rot_in: the folded norm takes bf16 rows and a [K] weight");
            tab.s[i].nw = nw[i].data_ptr();
            tab.s[i].nw_dtype = dtype_of(nw[i]);
            tab.s[i].inv_k = static_cast<float>(1.0 / K);
            tab.s[i].eps = static_cast<float>(eps);
        }
        kmax = std::max(kmax, K);
    }
    dim3 grid((unsigned)((kmax / 128 + 3) / 4), (unsigned)M, (unsigned)tab.n);
    launch_ex(seg_rot_in_kernel, grid, dim3(128), 0, pdl, tab);
}

// meta: 11 ints a segment (K, N, SK, wk, stride_k, stride_nb, xh_off, z_off, c_off, col, p0); ys[i] its output
void dsv41_x3seg_linear_cuda(const at::Tensor& xh, const std::vector<at::Tensor>& T,
                             const std::vector<at::Tensor>& svh, const std::vector<at::Tensor>& ys,
                             const c10::optional<at::Tensor>& Z, at::Tensor& counters,
                             const std::vector<int64_t>& meta, int64_t programs, int64_t K2, int64_t WK,
                             int64_t minb, bool pdl) {
    LinTable tab{};
    tab.n = (int)T.size();
    const int M = (int)ys[0].size(0);
    const int y_dtype = dtype_of(ys[0]);
    float* zptr = Z ? Z->data_ptr<float>() : nullptr;
    for (int i = 0; i < tab.n; ++i) {
        const int64_t* m = meta.data() + 11 * i;
        const size_t esz = ys[i].element_size();
        tab.s[i] = LinSeg{reinterpret_cast<const half*>(xh.data_ptr()) + m[6],
                          reinterpret_cast<const uint32_t*>(T[i].data_ptr()), (long long)m[4], (long long)m[5],
                          reinterpret_cast<const half*>(svh[i].data_ptr()),
                          static_cast<char*>(ys[i].data_ptr()) + (size_t)m[9] * esz,
                          m[2] > 1 ? zptr + m[7] : nullptr, counters.data_ptr<int>() + m[8],
                          (int)m[0], (int)m[1], (int)m[2], (int)m[3], (int)m[10], (int)ys[i].stride(0)};
    }
    const int smem = (int)(WK * std::min(M, 8) * 128 * sizeof(float));
#define TF_SEG_LAUNCH(K2_, WK_, MINB_)                                                                                \
    if (K2 == K2_ && WK == WK_ && (WK_ == 8 || minb == MINB_)) {                                                     \
        auto kernel = seg_linear_kernel<K2_, 2, WK_, MINB_>;                                                     \
        if (smem > 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);    \
        launch_ex(kernel, dim3((unsigned)programs), dim3((unsigned)(WK_ * 32)), smem, pdl, tab, y_dtype, M);       \
        return;                                                                                                    \
    }
#define TF_SEG_WIDTH(K2_) TF_SEG_LAUNCH(K2_, 4, 3) TF_SEG_LAUNCH(K2_, 4, 2) TF_SEG_LAUNCH(K2_, 8, 1)
    TF_SEG_WIDTH(4) TF_SEG_WIDTH(6) TF_SEG_WIDTH(8) TF_SEG_WIDTH(10) TF_SEG_WIDTH(12) TF_SEG_WIDTH(16)
#undef TF_SEG_WIDTH
#undef TF_SEG_LAUNCH
    TORCH_CHECK(false, "x3seg: unsupported EXL3 width / warps / blocks: K2=", K2, " WK=", WK, " MINB=", minb);
}
#endif  // DSV41_X3SEG_NO_TORCH

// The instantiations dsv41_x3seg_linear_cuda launches (TF_SEG_WIDTH), mul1; not when dense3.cu includes this file
#ifndef DSV41_DENSE3_NO_TORCH
template __global__ void dsv41_x3seg::seg_linear_kernel<4, 2, 4, 3>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_x3seg::seg_linear_kernel<4, 2, 4, 2>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_x3seg::seg_linear_kernel<4, 2, 8, 1>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_x3seg::seg_linear_kernel<6, 2, 4, 3>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_x3seg::seg_linear_kernel<6, 2, 4, 2>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_x3seg::seg_linear_kernel<6, 2, 8, 1>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_x3seg::seg_linear_kernel<8, 2, 4, 3>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_x3seg::seg_linear_kernel<8, 2, 4, 2>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_x3seg::seg_linear_kernel<8, 2, 8, 1>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_x3seg::seg_linear_kernel<10, 2, 4, 3>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_x3seg::seg_linear_kernel<10, 2, 4, 2>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_x3seg::seg_linear_kernel<10, 2, 8, 1>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_x3seg::seg_linear_kernel<12, 2, 4, 3>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_x3seg::seg_linear_kernel<12, 2, 4, 2>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_x3seg::seg_linear_kernel<12, 2, 8, 1>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_x3seg::seg_linear_kernel<16, 2, 4, 3>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_x3seg::seg_linear_kernel<16, 2, 4, 2>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_x3seg::seg_linear_kernel<16, 2, 8, 1>(const dsv41_x3seg::LinTable, int, int);
#endif
