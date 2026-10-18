// Copy of src/tensorfold/families/deepseek_v41/cuda/dense3.cu (blob 21c3bed265fe): sass_check.sh
#define DSV41_DENSE3_NO_TORCH



// Dense EXL3 linears v3 (TF_DSV41_DENSE_V3): x3seg's programs and bits, trellis read from a repacked "lanes" layout.

// Lanes load geometry after antirez/ds4 (cuda/mmq/ds4_mmq.cu q8_0_aligned_dense_vec), which measured it on GB10.

#define DSV41_X3SEG_NO_TORCH          // x3seg.cu's device code only (its tables, helpers and statements)
#include "x3seg.cu"
#ifndef DSV41_DENSE3_NO_TORCH         // the kernels alone (nvcc without torch)
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

namespace dsv41_dense3 {

using namespace tf_exl3;
using dsv41_x3seg::LinSeg;
using dsv41_x3seg::LinTable;
using dsv41_x3seg::MAXSEG;
using dsv41_x3seg::pdl_trigger;
using dsv41_x3seg::pdl_wait;

// k steps a load group and a lane's words of a group (whole 16-byte chunks); kernels built for K2 4-12, layout to 16
__host__ __device__ constexpr int group_steps(int K2) { return K2 == 16 ? 1 : 2; }
template <int K2>
__host__ __device__ constexpr int gwords() { return group_steps(K2) * K2; }

// Bits [bo, bo + n) (MSB first) of words R[wi], R[wi + 1] as the low n bits of a 64-bit value (bo + n <= 64)
template <int GW>
__device__ __forceinline__ uint64_t take_bits(const uint32_t (&R)[GW], int p, int n) {
    const int wi = p >> 5, bo = p & 31;
    const uint64_t x = ((uint64_t)R[wi] << 32) | (wi + 1 < GW ? (uint64_t)R[wi + 1] : 0ull);
    return n == 64 ? x : (x << bo) >> (64 - n);
}

// B fragments of tiles j, j + 1 of step g from group words R: own bits + 16 prefix bits of lane - 1 (one shuffle)
template <int K2, int CB>
__device__ __forceinline__ void pair_frags(const uint32_t (&R)[gwords<K2>()], int g, int j, int lane,
                                           uint32_t (&b)[2][2][2]) {
    constexpr int n = 4 * K2;
    const uint64_t o0 = take_bits(R, (g * 8 + j) * n, n), o1 = take_bits(R, (g * 8 + j + 1) * n, n);
    const uint32_t tails = (uint32_t)(o0 & 0xffffu) | ((uint32_t)(o1 & 0xffffu) << 16);
    const uint32_t pre = __shfl_sync(0xffffffffu, tails, (lane + 31) & 31);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint64_t o = h ? o1 : o0;
        const uint64_t p = h ? (pre >> 16) : (pre & 0xffffu);
        uint32_t s[8];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int sh = (7 - v) * K2 / 2;                 // the window of value 8 lane + v: bits [sh, sh + 16)
            uint64_t w = o >> sh;
            if (n - sh < 16) w |= p << (n - sh);
            s[v] = (uint32_t)(w & 0xffffu);
        }
        b[h][0][0] = decode2<CB>(s[0], s[1]);
        b[h][0][1] = decode2<CB>(s[2], s[3]);
        b[h][1][0] = decode2<CB>(s[4], s[5]);
        b[h][1][1] = decode2<CB>(s[6], s[7]);
    }
}

// a lane's chunks of one group: GW / 4 coalesced 16-byte loads (warp: 512 contiguous bytes each)
template <int GW>
__device__ __forceinline__ void load_group(const uint32_t* lane_base, uint32_t (&R)[GW]) {
#pragma unroll
    for (int c = 0; c < GW / 4; ++c) {
        const uint4 u = __ldg(reinterpret_cast<const uint4*>(lane_base + c * 128));
        R[4 * c] = u.x; R[4 * c + 1] = u.y; R[4 * c + 2] = u.z; R[4 * c + 3] = u.w;
    }
}

__device__ __forceinline__ void load_a(const half* x0, const half* x1, int kt, int t, uint32_t (&a)[4]) {
    a[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
    a[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t));
    a[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
    a[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t + 8));
}

// x3seg's seg_linear_kernel with the lanes layout's loads, the rest verbatim; needs per_warp % group_steps(K2) == 0
template <int K2, int CB, int WK, int MINB>
__global__ void __launch_bounds__(WK * 32, MINB) lanes_linear_kernel(const LinTable tab, int y_dtype, int M) {
    constexpr int G = group_steps(K2), GW = gwords<K2>();
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
    const bool on = warp < wk;                                // a WK = 4 segment in an 8-warp launch: warps 4-7 idle
    const int per_warp = (K >> 4) / SK / wk;
    const int kt0 = split * (per_warp * wk) + warp * per_warp;
    const int steps = on ? per_warp : 0, groups = steps / G;
    const uint32_t* base = s.T + nb * s.stride_nb + (size_t)kt0 * s.stride_k + 4 * lane;
    const size_t gstride = (size_t)G * s.stride_k;
    const int col0 = nb * 128;

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
        uint32_t cur[GW], nxt[GW];
        if (groups > 0) load_group(base, cur);
        if (pass == 0) pdl_wait();                            // weights above, xh / Z / counters below
        uint32_t a[4], an[4];
        if (steps > 0) load_a(x0, x1, kt0, t, a);
#pragma unroll 1
        for (int gi = 0; gi < groups; ++gi) {
            if (gi + 1 < groups) load_group(base + (size_t)(gi + 1) * gstride, nxt);
#pragma unroll
            for (int st = 0; st < G; ++st) {
                const int i = gi * G + st;
                if (i + 1 < steps) load_a(x0, x1, kt0 + i + 1, t, an);   // the next step's input, one step early
#pragma unroll
                for (int j = 0; j < 8; j += 2) {
                    uint32_t b[2][2][2];
                    pair_frags<K2, CB>(cur, st, j, lane, b);
                    mma16816(acc[j][0], a, b[0][0]);
                    mma16816(acc[j][1], a, b[0][1]);
                    mma16816(acc[j + 1][0], a, b[1][0]);
                    mma16816(acc[j + 1][1], a, b[1][1]);
                }
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = an[c];
            }
#pragma unroll
            for (int c = 0; c < GW; ++c) cur[c] = nxt[c];
        }

        if (m0 + 16 >= M) pdl_trigger();                     // the last pass's weights are read

        // -- x3seg's statements from here on (the warps' sums in warp order, Z, the last arriver, the epilogue) --
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
                    dsv41_x3seg::finish(v, lane, svh, nullptr, col0 + 4 * lane);
                    dsv41_x3seg::store4(y, y_dtype, (size_t)(m0 + rlo + r) * ld + col0 + 4 * lane, v);
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
                    dsv41_x3seg::finish(v, lane, svh, nullptr, col0 + 4 * lane);
                    dsv41_x3seg::store4(y, y_dtype, (size_t)(m0 + r) * ld + col0 + 4 * lane, v);
                }
                if (threadIdx.x == 0) counters[pass * NB + nb] = 0;
            }
        }
        __syncthreads();
    }
}

// Load-time relayout, a thread a dst word: group gi holds lane L's word w at gi G 32 K2 + ((w / 4) 32 + L) 4 + w % 4
__device__ __forceinline__ uint32_t bit_of(const uint32_t* tile, int sb) { return (tile[sb >> 5] >> (31 - (sb & 31))) & 1u; }

__global__ void to_lanes_kernel(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst, long long total, int K2,
                                int KT) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int G = group_steps(K2), n = 4 * K2, TW = 4 * K2;
    const long long strip_words = (long long)KT * 32 * K2, gw = (long long)G * 32 * K2;
    const long long strip = i / strip_words, r = i % strip_words, gi = r / gw;
    const int qd = (int)(r % gw), c = qd / 128, L = (qd % 128) / 4, w = 4 * c + qd % 4;
    const uint32_t* s0 = src + strip * strip_words;
    uint32_t out = 0;
    for (int b = 0; b < 32; ++b) {
        const int lb = 32 * w + b, tt = lb / n, ob = lb % n;
        const long long kt = gi * G + tt / 8;
        out |= bit_of(s0 + kt * 32 * K2 + (tt % 8) * TW, n * L + ob) << (31 - b);
    }
    dst[i] = out;
}

__global__ void to_strips_kernel(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst, long long total, int K2,
                                 int KT) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int G = group_steps(K2), n = 4 * K2, TW = 4 * K2;
    const long long strip_words = (long long)KT * 32 * K2;
    const long long strip = i / strip_words, r = i % strip_words, kt = r / (32 * K2);
    const int tw = (int)(r % (32 * K2)), j = tw / TW, wd = tw % TW;
    const long long gi = kt / G;
    const int tt = (int)(kt % G) * 8 + j;
    const uint32_t* s0 = src + strip * strip_words + gi * (long long)G * 32 * K2;
    uint32_t out = 0;
    for (int b = 0; b < 32; ++b) {
        const int sb = 32 * wd + b, L = sb / n, lb = tt * n + sb % n, w = lb >> 5;
        out |= ((s0[((w >> 2) * 32 + L) * 4 + (w & 3)] >> (31 - (lb & 31))) & 1u) << (31 - b);
    }
    dst[i] = out;
}

// W_q [K, N] fp16 from the lanes layout (upstream unpack_kernel's values): one warp a (16-column tile, k step)
template <int K2, int CB>
__global__ void __launch_bounds__(32) lanes_unpack_kernel(const uint32_t* __restrict__ T, half* __restrict__ W, int N,
                                                          int64_t stride_k, int64_t stride_nb) {
    constexpr int G = group_steps(K2), GW = gwords<K2>();
    const int kt = blockIdx.y, nt = blockIdx.x, lane = threadIdx.x, j = nt & 6;
    uint32_t R[GW];
    load_group(T + (nt >> 3) * stride_nb + (size_t)(kt / G) * G * stride_k + 4 * lane, R);
    uint32_t b[2][2][2];
    pair_frags<K2, CB>(R, kt % G, j, lane, b);
    const int h = nt & 1;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const uint32_t x = b[h][v >> 2][(v >> 1) & 1];
        const unsigned short u = (v & 1) ? (unsigned short)(x >> 16) : (unsigned short)(x & 0xffffu);
        W[(size_t)(kt * 16 + value_row(lane, v)) * N + nt * 16 + value_col(lane, v)] = __ushort_as_half(u);
    }
}

}  // namespace dsv41_dense3

#ifndef DSV41_DENSE3_NO_TORCH

using namespace dsv41_dense3;

namespace {
template <typename Kn, typename... A>
void launch_ex(Kn kernel, dim3 grid, dim3 block, int smem, bool pdl, A... args) {
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

// x3seg's linear launcher (meta: 11 ints a segment) for words in the lanes layout
void dsv41_dense3_linear_cuda(const at::Tensor& xh, const std::vector<at::Tensor>& T,
                              const std::vector<at::Tensor>& svh, const std::vector<at::Tensor>& ys,
                              const c10::optional<at::Tensor>& Z, at::Tensor& counters,
                              const std::vector<int64_t>& meta, int64_t programs, int64_t K2, int64_t WK,
                              int64_t minb, bool pdl) {
    LinTable tab{};
    tab.n = (int)T.size();
    const int M = (int)ys[0].size(0);
    const auto st = ys[0].scalar_type();
    const int y_dtype = st == at::kFloat ? dsv41_x3seg::F32 : st == at::kBFloat16 ? dsv41_x3seg::BF16 : dsv41_x3seg::F16;
    float* zptr = Z ? Z->data_ptr<float>() : nullptr;
    for (int i = 0; i < tab.n; ++i) {
        const int64_t* m = meta.data() + 11 * i;
        const int64_t per_warp = m[0] / 16 / m[2] / m[3];
        TORCH_CHECK(per_warp % group_steps((int)K2) == 0, "dense3: ", per_warp, " k steps a warp do not split into ",
                    group_steps((int)K2), "-step load groups (dense3.why_not)");
        const size_t esz = ys[i].element_size();
        tab.s[i] = LinSeg{reinterpret_cast<const half*>(xh.data_ptr()) + m[6],
                          reinterpret_cast<const uint32_t*>(T[i].data_ptr()), (long long)m[4], (long long)m[5],
                          reinterpret_cast<const half*>(svh[i].data_ptr()),
                          static_cast<char*>(ys[i].data_ptr()) + (size_t)m[9] * esz,
                          m[2] > 1 ? zptr + m[7] : nullptr, counters.data_ptr<int>() + m[8],
                          (int)m[0], (int)m[1], (int)m[2], (int)m[3], (int)m[10], (int)ys[i].stride(0)};
    }
    const int smem = (int)(WK * std::min(M, 8) * 128 * sizeof(float));
#define TF_D3_LAUNCH(K2_, WK_, MINB_)                                                                               \
    if (K2 == K2_ && WK == WK_ && minb == MINB_) {                                                                  \
        auto kernel = lanes_linear_kernel<K2_, 2, WK_, MINB_>;                                                    \
        if (smem > 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);    \
        launch_ex(kernel, dim3((unsigned)programs), dim3((unsigned)(WK_ * 32)), smem, pdl, tab, y_dtype, M);       \
        return;                                                                                                    \
    }
#define TF_D3_WIDTH(K2_) TF_D3_LAUNCH(K2_, 4, 3) TF_D3_LAUNCH(K2_, 4, 4) TF_D3_LAUNCH(K2_, 8, 1) TF_D3_LAUNCH(K2_, 8, 2)
    TF_D3_WIDTH(4) TF_D3_WIDTH(6) TF_D3_WIDTH(8) TF_D3_WIDTH(10) TF_D3_WIDTH(12)   // mul1; 8 bits: no attention matrix
#undef TF_D3_WIDTH
#undef TF_D3_LAUNCH
    TORCH_CHECK(false, "dense3: unsupported EXL3 width / warps / blocks: K2=", K2, " WK=", WK, " MINB=", minb);
}

// src -> dst (int32 words of one matrix, [N/128, K/16, 8, 4 K2]); to_lanes or back
void dsv41_dense3_relayout_cuda(const at::Tensor& src, at::Tensor& dst, int64_t K2, int64_t K, bool lanes) {
    const long long total = src.numel();
    const unsigned blocks = (unsigned)((total + 255) / 256);
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* s = reinterpret_cast<const uint32_t*>(src.data_ptr());
    auto* d = reinterpret_cast<uint32_t*>(dst.data_ptr());
    if (lanes)
        to_lanes_kernel<<<blocks, 256, 0, stream>>>(s, d, total, (int)K2, (int)(K / 16));
    else
        to_strips_kernel<<<blocks, 256, 0, stream>>>(s, d, total, (int)K2, (int)(K / 16));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsv41_dense3_unpack_cuda(const at::Tensor& T, at::Tensor& W, int64_t stride_k, int64_t stride_nb, int64_t K2) {
    const int K = (int)W.size(0), N = (int)W.size(1);
    dim3 grid((unsigned)(N / 16), (unsigned)(K / 16));
    auto stream = at::cuda::getCurrentCUDAStream();
#define TF_D3_UNPACK(K2_)                                                                                           \
    if (K2 == K2_) {                                                                                                \
        lanes_unpack_kernel<K2_, 2><<<grid, 32, 0, stream>>>(reinterpret_cast<const uint32_t*>(T.data_ptr()),      \
                                                              reinterpret_cast<half*>(W.data_ptr()), N, stride_k, \
                                                              stride_nb);                                          \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                            \
        return;                                                                                                    \
    }
    TF_D3_UNPACK(4) TF_D3_UNPACK(6) TF_D3_UNPACK(8) TF_D3_UNPACK(10) TF_D3_UNPACK(12)
#undef TF_D3_UNPACK
    TORCH_CHECK(false, "dense3: unsupported EXL3 width K2=", K2);
}
#endif  // DSV41_DENSE3_NO_TORCH

// The instantiations dsv41_dense3_linear_cuda / _unpack_cuda launch (TF_D3_WIDTH, TF_D3_UNPACK): mul1.
template __global__ void dsv41_dense3::lanes_linear_kernel<4, 2, 4, 3>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_linear_kernel<4, 2, 4, 4>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_linear_kernel<4, 2, 8, 1>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_linear_kernel<4, 2, 8, 2>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_unpack_kernel<4, 2>(const uint32_t*, half*, int, int64_t, int64_t);
template __global__ void dsv41_dense3::lanes_linear_kernel<6, 2, 4, 3>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_linear_kernel<6, 2, 4, 4>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_linear_kernel<6, 2, 8, 1>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_linear_kernel<6, 2, 8, 2>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_unpack_kernel<6, 2>(const uint32_t*, half*, int, int64_t, int64_t);
template __global__ void dsv41_dense3::lanes_linear_kernel<8, 2, 4, 3>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_linear_kernel<8, 2, 4, 4>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_linear_kernel<8, 2, 8, 1>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_linear_kernel<8, 2, 8, 2>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_unpack_kernel<8, 2>(const uint32_t*, half*, int, int64_t, int64_t);
template __global__ void dsv41_dense3::lanes_linear_kernel<10, 2, 4, 3>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_linear_kernel<10, 2, 4, 4>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_linear_kernel<10, 2, 8, 1>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_linear_kernel<10, 2, 8, 2>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_unpack_kernel<10, 2>(const uint32_t*, half*, int, int64_t, int64_t);
template __global__ void dsv41_dense3::lanes_linear_kernel<12, 2, 4, 3>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_linear_kernel<12, 2, 4, 4>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_linear_kernel<12, 2, 8, 1>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_linear_kernel<12, 2, 8, 2>(const dsv41_x3seg::LinTable, int, int);
template __global__ void dsv41_dense3::lanes_unpack_kernel<12, 2>(const uint32_t*, half*, int, int64_t, int64_t);
