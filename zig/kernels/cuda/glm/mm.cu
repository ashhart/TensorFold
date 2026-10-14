// GLM-5.3-Flash on CUDA: the dense linears, ours. One tensor-core kernel for every row count (a decode window's few
// rows, sixteen streams' windows at once, a prompt chunk's 2048): each group of 64 inputs is summed from zero by the
// bf16 mma, then acc = fma(P, scale, acc) in group order, and the K slices (fixed by the shape) are added in slice
// order, so a row's bits never depend on the rows beside it.
//
// Weights are tiled at load (glm_mm_tile*): a column tile of 8 outputs and a group of 64 inputs is 32 lanes' chunks of
// 16 values in mma B-fragment order (lane l: column l / 4; value 4 kt + j at input 16 kt + {2t, 2t+1, 2t+8, 2t+9}[j],
// t = l % 4), each chunk BITS / 2 words (ops.cu's 16-value packing; BITS 16: the bf16 values themselves). Scales are
// fp16 [N / 8][K / 64][8].

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "../flashnext/qmm_frag.cuh"

namespace {

using namespace qmm_frag;

// 16 codes from a packed chunk (ops.cu's layout), stored + 2^(BITS-1)
template <int BITS>
__device__ __forceinline__ void codes16(const uint32_t* __restrict__ p, int (&q)[16]) {
    if constexpr (BITS == 4) {
#pragma unroll
        for (int i = 0; i < 16; ++i) q[i] = (int)((p[i / 8] >> (4 * (i % 8))) & 15u) - 8;
    } else if constexpr (BITS == 8) {
#pragma unroll
        for (int i = 0; i < 16; ++i) q[i] = (int)((p[i / 4] >> (8 * (i % 4))) & 255u) - 128;
    } else {
#pragma unroll
        for (int i = 0; i < 15; ++i) q[i] = (int)((p[i / 5] >> (6 * (i % 5))) & 63u) - 32;
        q[15] = (int)((p[0] >> 30) | ((p[1] >> 30) << 2) | ((p[2] >> 30) << 4)) - 32;
    }
}

template <int BITS>
__device__ __forceinline__ void pack16(const int (&q)[16], uint32_t* __restrict__ out) {
    constexpr int half = 1 << (BITS - 1);
    uint32_t w[BITS / 2];
#pragma unroll
    for (int i = 0; i < BITS / 2; ++i) w[i] = 0u;
    if constexpr (BITS == 4) {
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i / 8] |= (uint32_t)(q[i] + half) << (4 * (i % 8));
    } else if constexpr (BITS == 8) {
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i / 4] |= (uint32_t)(q[i] + half) << (8 * (i % 4));
    } else {
#pragma unroll
        for (int i = 0; i < 15; ++i) w[i / 5] |= (uint32_t)(q[i] + half) << (6 * (i % 5));
        const uint32_t last = (uint32_t)(q[15] + half);
#pragma unroll
        for (int j = 0; j < 3; ++j) w[j] |= ((last >> (2 * j)) & 3u) << 30;
    }
#pragma unroll
    for (int i = 0; i < BITS / 2; ++i) out[i] = w[i];
}

// input of value v (0..15) of lane ``lane``'s chunk within its group of 64
__device__ __forceinline__ int frag_k(int lane, int v) {
    const int t = lane & 3, kt = v >> 2, j = v & 3;
    return kt * 16 + 2 * t + (j & 1) + (j >> 1) * 8;
}

// the B-fragment bf16 pairs of a chunk: pairs 2 kt and 2 kt + 1 are step kt's b0 and b1
template <int BITS>
__device__ __forceinline__ void bpairs(const uint32_t* __restrict__ p, uint32_t (&b)[8]) {
    if constexpr (BITS == 16) {
#pragma unroll
        for (int i = 0; i < 8; ++i) b[i] = p[i];
    } else {
        int q[16];
        codes16<BITS>(p, q);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const __nv_bfloat162 h = __floats2bfloat162_rn((float)q[2 * i], (float)q[2 * i + 1]);   // exact integers
            b[i] = *reinterpret_cast<const uint32_t*>(&h);
        }
    }
}

template <int BITS, int BM, int WM, int STAGES>
struct Tile {
    static constexpr int WN = 4, BN = 64, NT = 2, MT = BM / WM / 16, THREADS = WM * WN * 32;
    static constexpr int WPC = BITS / 2;
    static constexpr int ROW = 128, CHUNKS = 8;               // a row's 64 inputs: 128 bytes, 8 chunks of 16
    static constexpr int X = BM * ROW;
    static constexpr int W = 8 * 32 * WPC * 4;                // 8 column tiles' chunks
    static constexpr int S = 8 * 8 * 2;                       // their scales (fp16)
    static constexpr int STAGE = X + W + S;
    static constexpr int SMEM = STAGES * STAGE;
};

template <int BITS, int BM, int WM, int STAGES>
__device__ __forceinline__ void mm_kernel(const __nv_bfloat16* __restrict__ x, long long ldx,
        const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part,
        int f32, int M, int N, int K, int SK) {
    using T = Tile<BITS, BM, WM, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / T::WN, wn = warp % T::WN;
    const int KG = K >> 6, per = KG / SK, N8 = N >> 3;
    const int n0 = blockIdx.x * T::BN, m0 = blockIdx.y * BM, slice = blockIdx.z, g0 = slice * per;
    const int t0 = n0 >> 3;                                   // the block's first column tile
    auto stage = [&](int st) { return buf + st * T::STAGE; };
    auto loadX = [&](int st, int g) {
        unsigned char* p = stage(st);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + (long long)row * ldx + g * 64 + ch * 8, m0 + r < M);
        }
    };
    auto loadW = [&](int st, int g) {
        unsigned char* pw = stage(st) + T::X;
        constexpr int PER_TILE = 32 * T::WPC * 4 / 16;        // 16-byte pieces of one column tile's chunks
        for (int c = tid; c < 8 * PER_TILE; c += T::THREADS) {
            const int tt = c / PER_TILE, off = c % PER_TILE;
            if (t0 + tt >= N8) continue;
            const long long src = ((long long)(t0 + tt) * KG + g) * 32 * T::WPC * 4 + off * 16;
            cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + src);
        }
        if constexpr (BITS != 16) {
            unsigned char* ps = pw + T::W;
            for (int tt = tid; tt < 8; tt += T::THREADS) {
                if (t0 + tt >= N8) continue;
                cp16(ps + tt * 16, s + ((long long)(t0 + tt) * KG + g) * 8);
            }
        }
    };
    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    // dependent launch: the first stages' weights stream in while the kernel before still runs; its outputs (this
    // one's input, and any buffer this one writes) only after griddepcontrol.wait
    grid_launch();
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st)
        if (st < per) loadW(st, g0 + st);
    grid_wait();
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if (st < per) loadX(st, g0 + st);
        commit();
    }
    for (int it = 0; it < per; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < per) {
            loadX(next % STAGES, g0 + next);
            loadW(next % STAGES, g0 + next);
        }
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const __half* ps = reinterpret_cast<const __half*>(p + T::X + T::W);
        uint32_t b[T::NT][8];
#pragma unroll
        for (int j = 0; j < T::NT; ++j) bpairs<BITS>(pw + ((wn * T::NT + j) * 32 + lane) * T::WPC, b[j]);
        float d[T::MT][T::NT][4];
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint32_t a[T::MT][4];
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    if (kt == 0) mma0(d[i][j], a[i], b[j][0], b[j][1]);
                    else mma(d[i][j], a[i], b[j][2 * kt], b[j][2 * kt + 1]);
                }
        }
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            float s0 = 1.0f, s1 = 1.0f;
            if constexpr (BITS != 16) {
                const __half2 h = *reinterpret_cast<const __half2*>(ps + (wn * T::NT + j) * 8 + (lane & 3) * 2);
                s0 = __low2float(h);
                s1 = __high2float(h);
            }
#pragma unroll
            for (int i = 0; i < T::MT; ++i)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][j][e] = __fmaf_rn(d[i][j][e], (e & 1) ? s1 : s0, acc[i][j][e]);
        }
    }
    wait<0>();
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = n0 + (wn * T::NT + j) * 8 + (lane & 3) * 2;
            if (col >= N) continue;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                if (SK > 1) {
                    float* dst = part + ((long long)slice * M + row) * N + col;
                    dst[0] = v0;
                    dst[1] = v1;
                } else if (f32) {
                    float* dst = reinterpret_cast<float*>(out) + (long long)row * N + col;
                    dst[0] = v0;
                    dst[1] = v1;
                } else {
                    *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + (long long)row * N + col) =
                        __floats2bfloat162_rn(v0, v1);
                }
            }
        }
}

}  // namespace

// The K slices' partials added in slice order, then rounded once.
extern "C" __global__ void glm_mm_reduce(const float* __restrict__ part, void* __restrict__ out, long long total, int SK, int f32) {
    qmm_frag::grid_launch();
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float acc = part[i];
    for (int sl = 1; sl < SK; ++sl) acc = acc + part[sl * total + i];
    if (f32) reinterpret_cast<float*>(out)[i] = acc;
    else reinterpret_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(acc);
}

// one instance a (bits, row bucket): (16, 1 warp row), (32, 1), (64, 2), (128, 2); three stages
#define GLM_MM_ALL(B)                                                                                                  \
    extern "C" __global__ void __launch_bounds__(128) glm_mm##B##_16(const __nv_bfloat16* __restrict__ x, long long ldx, \
        const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part, \
        int f32, int M, int N, int K, int SK) { mm_kernel<B, 16, 1, 3>(x, ldx, w, s, out, part, f32, M, N, K, SK); }     \
    extern "C" __global__ void __launch_bounds__(128) glm_mm##B##_32(const __nv_bfloat16* __restrict__ x, long long ldx, \
        const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part, \
        int f32, int M, int N, int K, int SK) { mm_kernel<B, 32, 1, 3>(x, ldx, w, s, out, part, f32, M, N, K, SK); }     \
    extern "C" __global__ void __launch_bounds__(256) glm_mm##B##_64(const __nv_bfloat16* __restrict__ x, long long ldx, \
        const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part, \
        int f32, int M, int N, int K, int SK) { mm_kernel<B, 64, 2, 3>(x, ldx, w, s, out, part, f32, M, N, K, SK); }     \
    extern "C" __global__ void __launch_bounds__(256) glm_mm##B##_128(const __nv_bfloat16* __restrict__ x, long long ldx, \
        const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part, \
        int f32, int M, int N, int K, int SK) { mm_kernel<B, 128, 2, 3>(x, ldx, w, s, out, part, f32, M, N, K, SK); }
GLM_MM_ALL(4)
GLM_MM_ALL(6)
GLM_MM_ALL(8)
GLM_MM_ALL(16)

// -- tiling at load ---------------------------------------------------------------------------------------------------
// packed rows (ops.cu's [N][K / 16][BITS / 2] words, fp16 scales [N][K / 64]) -> the tiled chunks and scales
template <int BITS>
__device__ __forceinline__ void tile_packed(const uint32_t* __restrict__ src, const __half* __restrict__ ss,
                                            uint32_t* __restrict__ dst, __half* __restrict__ ds, int N, int K) {
    constexpr int WPC = BITS / 2;
    const int KG = K >> 6, N8 = N >> 3;
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;   // a destination chunk: (tile, group, lane)
    if (c >= (long long)N8 * KG * 32) return;
    const int lane = (int)(c & 31);
    const long long tg = c >> 5;
    const int g = (int)(tg % KG), tt = (int)(tg / KG);
    const int n = tt * 8 + (lane >> 2);
    const uint32_t* row = src + (long long)n * (K >> 4) * WPC;
    int q[16];
#pragma unroll
    for (int v = 0; v < 16; ++v) {
        const int k = g * 64 + frag_k(lane, v);
        int sq[16];
        codes16<BITS>(row + (long long)(k >> 4) * WPC, sq);
        q[v] = sq[k & 15];
    }
    pack16<BITS>(q, dst + c * WPC);
    if (lane < 8) ds[(tg * 8) + lane] = ss[(long long)(tt * 8 + lane) * KG + g];
}

extern "C" __global__ void glm_mm_tile4(const uint32_t* src, const __half* ss, uint32_t* dst, __half* ds, int N, int K) { tile_packed<4>(src, ss, dst, ds, N, K); }
extern "C" __global__ void glm_mm_tile6(const uint32_t* src, const __half* ss, uint32_t* dst, __half* ds, int N, int K) { tile_packed<6>(src, ss, dst, ds, N, K); }
extern "C" __global__ void glm_mm_tile8(const uint32_t* src, const __half* ss, uint32_t* dst, __half* ds, int N, int K) { tile_packed<8>(src, ss, dst, ds, N, K); }

// bf16 rows [N][K] -> the tiled chunks (16 bf16 a chunk, in pair order)
extern "C" __global__ void glm_mm_tile16(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst, int N, int K) {
    const int KG = K >> 6, N8 = N >> 3;
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= (long long)N8 * KG * 32) return;
    const int lane = (int)(c & 31);
    const long long tg = c >> 5;
    const int g = (int)(tg % KG), tt = (int)(tg / KG);
    const int n = tt * 8 + (lane >> 2);
#pragma unroll
    for (int v = 0; v < 16; ++v) dst[c * 16 + v] = src[(long long)n * K + g * 64 + frag_k(lane, v)];
}
