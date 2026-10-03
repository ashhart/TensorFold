// y = x @ W + bias for an unquantized fp16/bf16 W: a warp an output column (several rows sharing its weight loads),
// lanes in fixed k order, a fixed xor butterfly, so a row's bits are its own.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

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
__device__ __forceinline__ __nv_bfloat16 f_from<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

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
// C > 1: a warp C adjacent columns, each row load serving C weight loads (a column's sum order is unchanged).
template <typename T, int R, int WARPS, int C = 1>
__global__ void __launch_bounds__(WARPS * 32) b16_kernel(const T* __restrict__ x, const T* __restrict__ w0,
                                                         const T* __restrict__ bias, T* __restrict__ y0, int M, int K,
                                                         int N0, const T* __restrict__ w1, T* __restrict__ y1, int N1) {
    const T* w = blockIdx.z ? w1 : w0;                     // grid z 2: a second weight of the same rows, its own out
    T* y = blockIdx.z ? y1 : y0;
    const int N = blockIdx.z ? N1 : N0;
    const int lane = threadIdx.x & 31;
    const int row0 = blockIdx.y * R;
    const int col0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * C;
    if (col0 >= N) return;
    const T* wr[C];
#pragma unroll
    for (int c = 0; c < C; ++c) wr[c] = w + (size_t)(col0 + c < N ? col0 + c : col0) * K;
    float acc[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) acc[r][c] = 0.f;
    for (int k = 8 * lane; k + 8 <= K; k += 256) {
        float b[C][8];
#pragma unroll
        for (int c = 0; c < C; ++c) ld8(wr[c] + k, b[c]);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r >= M) break;
            float a[8];
            ld8(x + (size_t)(row0 + r) * K + k, a);
#pragma unroll
            for (int c = 0; c < C; ++c)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[r][c] = fmaf(a[i], b[c][i], acc[r][c]);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r >= M) break;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            const int col = col0 + c;
            float v = acc[r][c];
#pragma unroll
            for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
            if (lane == 0 && col < N) {
                if (bias != nullptr) v += f2f(__ldg(bias + col));
                y[(size_t)(row0 + r) * N + col] = f_from<T>(v);
            }
        }
    }
}

template <typename T, int R, int WARPS, int C = 1>
void launch(const at::Tensor& x, const at::Tensor& w, const void* bp, at::Tensor& y, int M, int K, int N,
            const at::Tensor* w1, at::Tensor* y1) {
    const int N1 = w1 ? (int)w1->size(0) : 0, wide = N1 > N ? N1 : N;
    const dim3 block(WARPS * 32), grid((unsigned)((wide + WARPS * C - 1) / (WARPS * C)), (unsigned)((M + R - 1) / R),
                                       w1 ? 2u : 1u);
    b16_kernel<T, R, WARPS, C><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const T*>(x.data_ptr()), reinterpret_cast<const T*>(w.data_ptr()),
        reinterpret_cast<const T*>(bp), reinterpret_cast<T*>(y.data_ptr()), M, K, N,
        w1 ? reinterpret_cast<const T*>(w1->data_ptr()) : nullptr, y1 ? reinterpret_cast<T*>(y1->data_ptr()) : nullptr,
        N1);
}

// A verify window (2-63 rows) staged 256 inputs at a time in shared memory for WARPS x C columns; b16_kernel's bits.
template <typename T, int R, int WARPS, int C>
__global__ void __launch_bounds__(WARPS * 32) b16_window_kernel(const T* __restrict__ x, const T* __restrict__ w0,
                                                                const T* __restrict__ bias, T* __restrict__ y0, int M,
                                                                int K, int N0, const T* __restrict__ w1,
                                                                T* __restrict__ y1, int N1) {
    const T* w = blockIdx.z ? w1 : w0;
    T* y = blockIdx.z ? y1 : y0;
    const int N = blockIdx.z ? N1 : N0;
    __shared__ __align__(16) T xs[R][256];
    const int lane = threadIdx.x & 31;
    const int row0 = blockIdx.y * R;
    const int col0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * C;
    const bool live = col0 < N;                            // no early return: the block syncs on every chunk
    const T* wr[C];
#pragma unroll
    for (int c = 0; c < C; ++c) wr[c] = w + (size_t)(col0 + c < N ? col0 + c : 0) * K;
    float acc[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) acc[r][c] = 0.f;
    for (int kb = 0; kb < K; kb += 256) {
        for (int i = threadIdx.x; i < R * 32; i += WARPS * 32) {
            const int r = i >> 5, k = kb + 8 * (i & 31);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (row0 + r < M && k + 8 <= K) v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)(row0 + r) * K + k));
            *reinterpret_cast<uint4*>(&xs[r][8 * (i & 31)]) = v;
        }
        __syncthreads();
        const int k = kb + 8 * lane;
        if (live && k + 8 <= K) {
            float b[C][8];
#pragma unroll
            for (int c = 0; c < C; ++c) ld8(wr[c] + k, b[c]);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (row0 + r >= M) break;
                float a[8];
                const uint4 u = *reinterpret_cast<const uint4*>(&xs[r][8 * lane]);
                const uint32_t q[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if constexpr (std::is_same_v<T, half>) {
                        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&q[i]));
                        a[2 * i] = f.x; a[2 * i + 1] = f.y;
                    } else {
                        const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&q[i]));
                        a[2 * i] = f.x; a[2 * i + 1] = f.y;
                    }
                }
#pragma unroll
                for (int c = 0; c < C; ++c)
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[r][c] = fmaf(a[i], b[c][i], acc[r][c]);
            }
        }
        __syncthreads();
    }
    if (!live) return;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r >= M) break;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            const int col = col0 + c;
            float v = acc[r][c];
#pragma unroll
            for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
            if (lane == 0 && col < N) {
                if (bias != nullptr) v += f2f(__ldg(bias + col));
                y[(size_t)(row0 + r) * N + col] = f_from<T>(v);
            }
        }
    }
}

template <typename T, int R, int WARPS, int C>
void launch_window(const at::Tensor& x, const at::Tensor& w, const void* bp, at::Tensor& y, int M, int K, int N,
                   const at::Tensor* w1, at::Tensor* y1) {
    const int N1 = w1 ? (int)w1->size(0) : 0, wide = N1 > N ? N1 : N;
    const dim3 block(WARPS * 32), grid((unsigned)((wide + WARPS * C - 1) / (WARPS * C)), (unsigned)((M + R - 1) / R),
                                       w1 ? 2u : 1u);
    b16_window_kernel<T, R, WARPS, C><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const T*>(x.data_ptr()), reinterpret_cast<const T*>(w.data_ptr()),
        reinterpret_cast<const T*>(bp), reinterpret_cast<T*>(y.data_ptr()), M, K, N,
        w1 ? reinterpret_cast<const T*>(w1->data_ptr()) : nullptr, y1 ? reinterpret_cast<T*>(y1->data_ptr()) : nullptr,
        N1);
}

// Decode rows on the bf16/fp16 m16n8k16 mma: one fp32 chain a row over K in fixed order, the same at any row count.
template <typename T>
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    if constexpr (std::is_same_v<T, half>)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                     "{%8, %9}, {%0, %1, %2, %3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    else
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                     "{%8, %9}, {%0, %1, %2, %3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <typename T, int MT, int NT, int V, int PF, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) b16_mma_kernel(const T* __restrict__ x, const T* __restrict__ w0,
                                                             const T* __restrict__ bias, T* __restrict__ y0, int M,
                                                             int K, int N0, const T* __restrict__ w1,
                                                             T* __restrict__ y1, int N1) {
    // lane q reads V contiguous 16-byte runs of its rows and column a step; run v feeds mma pair v in both operands
    constexpr int KS = 32 * V;
    const T* w = blockIdx.z ? w1 : w0;
    T* y = blockIdx.z ? y1 : y0;
    const int N = blockIdx.z ? N1 : N0;
    const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
    const int n0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * (8 * NT);
    if (n0 >= N) return;
    const int m0 = blockIdx.y * (16 * MT);
    const T* wp[NT];
    bool wok[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int n = n0 + 8 * j + g;
        wok[j] = n < N;
        wp[j] = w + (size_t)(wok[j] ? n : 0) * K + 8 * V * q;
    }
    const T* xp[MT][2];
    bool xok[MT][2];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = m0 + 16 * i + g + 8 * h;
            xok[i][h] = r < M;
            xp[i][h] = x + (size_t)(xok[i][h] ? r : 0) * K + 8 * V * q;
        }
    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;
    const uint4 z = make_uint4(0, 0, 0, 0);
    uint4 b[PF][NT][V], a[PF][MT][2][V];
    auto load = [&](int s, int k) {
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int v = 0; v < V; ++v)
                b[s][j][v] = wok[j] && k < K ? __ldg(reinterpret_cast<const uint4*>(wp[j] + k) + v) : z;
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int v = 0; v < V; ++v)
                    a[s][i][h][v] = xok[i][h] && k < K ? __ldg(reinterpret_cast<const uint4*>(xp[i][h] + k) + v) : z;
    };
#pragma unroll
    for (int s = 0; s < PF; ++s) load(s, KS * s);
    for (int k = 0; k < K; k += KS * PF) {
#pragma unroll
        for (int s = 0; s < PF; ++s) {
            if (k + KS * s >= K) break;
#pragma unroll
            for (int v = 0; v < V; ++v)
#pragma unroll
                for (int i = 0; i < MT; ++i)
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        const uint4 &a0 = a[s][i][0][v], &a1 = a[s][i][1][v], &bb = b[s][j][v];
                        mma16816<T>(acc[i][j], a0.x, a1.x, a0.y, a1.y, bb.x, bb.y);
                        mma16816<T>(acc[i][j], a0.z, a1.z, a0.w, a1.w, bb.z, bb.w);
                    }
            load(s, k + KS * (s + PF));
        }
    }
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + 16 * i + g + 8 * h;
                if (row >= M) continue;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int col = n0 + 8 * j + 2 * q + e;
                    if (col >= N) continue;
                    float v = acc[i][j][2 * h + e];
                    if (bias != nullptr) v += f2f(__ldg(bias + col));
                    y[(size_t)row * N + col] = f_from<T>(v);
                }
            }
}

template <typename T, int MT, int NT, int V, int PF, int WARPS>
void launch_mma(const at::Tensor& x, const at::Tensor& w, const void* bp, at::Tensor& y, int M, int K, int N,
                const at::Tensor* w1, at::Tensor* y1) {
    const int N1 = w1 ? (int)w1->size(0) : 0, wide = N1 > N ? N1 : N, cols = 8 * NT * WARPS;
    const dim3 block(WARPS * 32), grid((unsigned)((wide + cols - 1) / cols), (unsigned)((M + 16 * MT - 1) / (16 * MT)),
                                       w1 ? 2u : 1u);
    b16_mma_kernel<T, MT, NT, V, PF, WARPS><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const T*>(x.data_ptr()), reinterpret_cast<const T*>(w.data_ptr()),
        reinterpret_cast<const T*>(bp), reinterpret_cast<T*>(y.data_ptr()), M, K, N,
        w1 ? reinterpret_cast<const T*>(w1->data_ptr()) : nullptr, y1 ? reinterpret_cast<T*>(y1->data_ptr()) : nullptr,
        N1);
}

template <typename T>
void by_rows(const at::Tensor& x, const at::Tensor& w, const void* bp, at::Tensor& y, int M, int K, int N,
             const at::Tensor* w1 = nullptr, at::Tensor* y1 = nullptr) {
#ifndef TF_B16_SIMT
    if (K % 64 == 0) {                              // the mma path, at every row count of this shape
        if (M <= 16) launch_mma<T, 1, 1, 2, 2, 4>(x, w, bp, y, M, K, N, w1, y1);
        else if (M <= 32) launch_mma<T, 2, 2, 2, 1, 4>(x, w, bp, y, M, K, N, w1, y1);
        else if (M <= 64) launch_mma<T, 4, 2, 2, 1, 4>(x, w, bp, y, M, K, N, w1, y1);
        else launch_mma<T, 8, 1, 2, 1, 4>(x, w, bp, y, M, K, N, w1, y1);
        return;
    }
#endif
    if (M >= 512) launch<T, 16, 16>(x, w, bp, y, M, K, N, w1, y1);    // rows a warp, warps a block: never a row's bits
    // a window (2-511 rows) shares each weight load across its rows; a row's sum keeps the one-row order
    else if (M >= 17) launch_window<T, 32, 4, 2>(x, w, bp, y, M, K, N, w1, y1);
    else if (M >= 9) launch_window<T, 16, 4, 4>(x, w, bp, y, M, K, N, w1, y1);
    else if (M >= 2) launch_window<T, 8, 4, 4>(x, w, bp, y, M, K, N, w1, y1);
    else launch<T, 1, 4>(x, w, bp, y, M, K, N, w1, y1);
}


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
